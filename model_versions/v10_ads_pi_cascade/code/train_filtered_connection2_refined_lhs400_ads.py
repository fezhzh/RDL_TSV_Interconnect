# -*- coding: utf-8 -*-
"""Retrain the refined-LHS400 ADS Connection2 NN after filtering hard samples.

Run this file directly in VS Code. No command-line arguments are required.

The script reuses the existing ADS single-device cache and optimized pi targets
from the refined-LHS400 ADS run. It excludes the highest-error training samples
identified by the latest S-parameter continuation, then retrains the neural
network with the same pi-parameter pretrain -> S-parameter training flow.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10.py"
WRAPPER_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py"

SOURCE_RUN_LABEL = "ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads"
HARD_METRICS_RUN_LABEL = "ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_sparam_continue"
FILTERED_RUN_LABEL = "ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_filtered_train20"

FILTER_TRAIN_COUNT = 20
PARAM_EPOCHS = 80
PARAM_PATIENCE = 24
SPARAM_EPOCHS = 180
SPARAM_PATIENCE = 45
SPARAM_LR = 8e-6


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.6g}")
    rows = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(rows)


def main() -> None:
    mod = load_module(TRAIN_SCRIPT, "v10_train_ads_pi_cascade_filtered_connection2")
    wrapper = load_module(WRAPPER_SCRIPT, "v10_train_ads_pi_cascade_connection2_wrapper_filtered")

    source_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / SOURCE_RUN_LABEL
    hard_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / HARD_METRICS_RUN_LABEL
    output_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / FILTERED_RUN_LABEL
    targets_path = source_dir / "pi_optimized_targets.csv"
    source_report_path = source_dir / "training_report.json"
    hard_metrics_path = hard_dir / "pi_sparam_metrics.csv"

    for path in [targets_path, source_report_path, hard_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    mod.OUTPUT_DIR = output_dir
    mod.ADS_CACHE_DIR = source_dir / "ads_single_device_cache"
    mod.PARAM_EPOCHS = PARAM_EPOCHS
    mod.PARAM_PATIENCE = PARAM_PATIENCE
    mod.SPARAM_EPOCHS = SPARAM_EPOCHS
    mod.SPARAM_PATIENCE = SPARAM_PATIENCE
    mod.SPARAM_LR = SPARAM_LR
    mod.PARAM_ANCHOR_WEIGHT = 0.0
    mod.ADS_DEVICE_LENGTH_SCALE = 1.0
    mod.collect_samples = lambda: wrapper.collect_connection2_samples(mod)

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    ads_settings = source_report["simulation"]["ads_settings"]
    hard_metrics = pd.read_csv(hard_metrics_path)
    excluded = (
        hard_metrics[hard_metrics["split"].eq("train")]
        .sort_values("pi_nn_nmse_s11_s21_ri", ascending=False)
        .head(FILTER_TRAIN_COUNT)
        [["sample_id", "dut_index", "pi_nn_nmse_s11_s21_ri", "pi_nn_s11_db_mae", "pi_nn_s21_db_mae"]]
        .reset_index(drop=True)
    )
    excluded_ids = set(excluded["sample_id"])

    mod.set_seed(mod.RANDOM_SEED)
    dut_df = mod.collect_samples()
    dut_df.loc[dut_df["sample_id"].isin(excluded_ids), "split"] = "excluded_train"
    sim = mod.load_single_device_simulation(dut_df, ads_settings)
    masks = mod.split_masks(dut_df)

    x_raw = dut_df[mod.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = mod.normalize_by_train(x_raw, masks["train"])

    pi_targets = pd.read_csv(targets_path)
    if list(pi_targets["sample_id"]) != list(dut_df["sample_id"]):
        pi_targets = dut_df[["sample_id"]].merge(pi_targets, on="sample_id", how="left")
    y_raw = pi_targets[mod.pi_target_columns()].to_numpy(dtype=np.float64)
    y_norm, y_mean, y_std = mod.normalize_by_train(y_raw, masks["train"])
    arrays = (x_norm, y_norm, masks, y_mean, y_std, sim)

    device = torch.device("cuda" if mod.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = mod.PiConnectionNet(input_dim=x_norm.shape[1]).to(dtype=mod.REAL_DTYPE, device=device)

    param_history = mod.train_param_model(model, x_norm, y_norm, masks, device)
    pretrain_metrics, pretrain_params = mod.evaluate_model(model, dut_df, arrays, device)
    pretrain_summary = mod.summarize_metrics(pretrain_metrics)
    sparam_history = mod.train_sparam_model(model, arrays, device)
    final_metrics, final_params = mod.evaluate_model(model, dut_df, arrays, device)
    final_summary = mod.summarize_metrics(final_metrics)
    plot_dir, plot_files = mod.save_comparison_plots(model, dut_df, arrays, final_metrics, device)

    history = pd.concat([param_history, sparam_history], ignore_index=True)
    excluded.to_csv(output_dir / "excluded_train_samples.csv", index=False, encoding="utf-8-sig")
    pi_targets.to_csv(output_dir / "pi_optimized_targets.csv", index=False, encoding="utf-8-sig")
    pretrain_metrics.to_csv(output_dir / "pi_sparam_metrics_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_summary.to_csv(output_dir / "pi_sparam_summary_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_params.to_csv(output_dir / "pi_param_predictions_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    final_metrics.to_csv(output_dir / "pi_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(output_dir / "pi_sparam_summary.csv", index=False, encoding="utf-8-sig")
    final_params.to_csv(output_dir / "pi_param_predictions.csv", index=False, encoding="utf-8-sig")
    history.to_csv(output_dir / "pi_training_history.csv", index=False, encoding="utf-8-sig")

    checkpoint_path = output_dir / "pi_connection_net_filtered.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "stage": "filtered_train_param_pretrain_sparam",
                "source_run_label": SOURCE_RUN_LABEL,
                "hard_metrics_run_label": HARD_METRICS_RUN_LABEL,
                "filter_train_count": FILTER_TRAIN_COUNT,
                "excluded_sample_ids": sorted(excluded_ids),
                "ads_settings": ads_settings,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "param_epochs_completed": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
                "sparam_epochs_completed": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
            },
        },
        checkpoint_path,
    )

    report = {
        "source_run_label": SOURCE_RUN_LABEL,
        "hard_metrics_run_label": HARD_METRICS_RUN_LABEL,
        "filtered_run_label": FILTERED_RUN_LABEL,
        "output_dir": str(output_dir),
        "filter_rule": f"exclude top {FILTER_TRAIN_COUNT} train samples by continued pi_nn_nmse_s11_s21_ri",
        "train_count_after_filter": int(masks["train"].sum()),
        "excluded_train_count": len(excluded_ids),
        "test_count": int(masks["test"].sum()),
        "checkpoint": str(checkpoint_path),
        "summary_after_param_pretrain": pretrain_summary.to_dict(orient="records"),
        "summary_final": final_summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_files,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v10 Filtered Connection2 Refined-LHS400 Training Validation",
                "",
                f"- Source run: `{SOURCE_RUN_LABEL}`",
                f"- Hard-sample metrics run: `{HARD_METRICS_RUN_LABEL}`",
                f"- Output: `{output_dir}`",
                f"- Filter rule: exclude top `{FILTER_TRAIN_COUNT}` train samples by continued `pi_nn_nmse_s11_s21_ri`.",
                f"- Train count after filter: `{int(masks['train'].sum())}`",
                f"- Excluded train count: `{len(excluded_ids)}`",
                f"- Test count: `{int(masks['test'].sum())}`",
                f"- Checkpoint: `{checkpoint_path}`",
                f"- Comparison plots: `{plot_dir}`",
                "",
                "## Excluded Train Samples",
                "",
                dataframe_to_markdown(excluded),
                "",
                "## After Param Pretrain",
                "",
                mod.dataframe_to_markdown(pretrain_summary),
                "",
                "## Final Summary",
                "",
                mod.dataframe_to_markdown(final_summary),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Excluded train samples:", flush=True)
    print(excluded.to_string(index=False), flush=True)
    print("After param pretrain:", flush=True)
    print(pretrain_summary.to_string(index=False), flush=True)
    print("Final summary:", flush=True)
    print(final_summary.to_string(index=False), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
