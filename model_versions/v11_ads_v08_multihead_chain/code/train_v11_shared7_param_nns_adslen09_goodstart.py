# -*- coding: utf-8 -*-
"""Train shared 7-parameter NNs from ADS-length-0.9 good-start targets.

Run this file directly in VS Code. No command-line arguments are required.

The network architecture is seven independent `input -> 30 -> 30 -> 20 -> 1`
models, one for each shared connection-circuit parameter. The predicted
parameters are cascaded back through the 13-device ADS-length-0.9 network and
validated against the HFSS full-chain S-parameters.
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
NN_SOURCE_SCRIPT = THIS_DIR / "train_v11_shared7_param_nns_from_all_goodstart.py"
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
OPT_RESULT_LABEL = "v11_sharedopt_c30_adslen09_goodstart_bad"
OPT_TARGET_FILE = "v08_shared_adslen09_goodstart_bad_targets.csv"
SOURCE_ADS_LABEL = "v11_sharedopt_c30_adslen09"
RUN_LABEL = "v11_shared7_param_nns_adslen09_goodstart"
ADS_DEVICE_LENGTH_SCALE = 0.9
SIGN_EPS = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parameter_sign_stats(table: pd.DataFrame, value_prefix: str, param_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in param_names:
        col = f"{value_prefix}_{name}" if value_prefix else name
        values = table[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "value_set": value_prefix or "target",
                "parameter": name,
                "count": int(len(values)),
                "negative_count": int(np.sum(values < -SIGN_EPS)),
                "zero_count": int(np.sum(np.abs(values) <= SIGN_EPS)),
                "positive_count": int(np.sum(values > SIGN_EPS)),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    nnsrc = load_module(NN_SOURCE_SCRIPT, "v11_shared7_nn_source_for_adslen09")
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_adslen09_nn")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_wrapper_for_adslen09_nn")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_adslen09_nn")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    opt_dir = version_root / "results" / OPT_RESULT_LABEL
    source_ads_dir = version_root / "results" / SOURCE_ADS_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_ads_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = ADS_DEVICE_LENGTH_SCALE
    base.set_seed(base.RANDOM_SEED)

    opt_target_path = opt_dir / OPT_TARGET_FILE
    opt_targets = pd.read_csv(opt_target_path, encoding="utf-8-sig")
    dut_df = wrapper.collect_v11_samples(base)
    if list(dut_df["sample_id"]) != list(opt_targets["sample_id"]):
        opt_targets = dut_df[["sample_id"]].merge(opt_targets, on="sample_id", how="left")
    if opt_targets[wrapper.V08_PARAM_NAMES].isna().any().any():
        raise ValueError("Optimized target table is missing parameter values after sample alignment.")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied to l_tmrdl, l_bsmrdl, and h_tsv by the v11 base ADS runner."
    sim = base.load_single_device_simulation(dut_df, settings)

    masks = nnsrc.split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = opt_targets[wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = nnsrc.normalize_by_train(x_raw, masks["train"])
    y_norm, y_mean, y_std = nnsrc.normalize_by_train(y_raw, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = wrapper.SharedV08ParamNet(input_dim=x_norm.shape[1]).to(dtype=base.REAL_DTYPE, device=device)
    history = nnsrc.train_param_model(base, wrapper, model, x_norm, y_norm, masks, device)
    pred_params = nnsrc.predict_params(base, model, x_norm, y_mean, y_std, device)
    metrics, pred_table = nnsrc.evaluate_predictions(base, wrapper, dut_df, sim, opt_targets, pred_params)
    summary = nnsrc.summarize(metrics)
    target_sign_stats = parameter_sign_stats(opt_targets, "", wrapper.V08_PARAM_NAMES)
    pred_sign_stats = parameter_sign_stats(pred_table, "pred", wrapper.V08_PARAM_NAMES)
    sign_stats = pd.concat([target_sign_stats, pred_sign_stats], ignore_index=True)

    nnsrc.save_summary_plots(base, output_dir, history, metrics, pred_table, wrapper)
    plot_dir, plot_paths = nnsrc.save_comparison_plots(base, wrapper, output_dir, dut_df, sim, metrics, opt_targets, pred_params)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": "v11_shared7_param_nns_adslen09_goodstart",
                "architecture": "seven independent input->30->30->20->1 networks",
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": wrapper.V08_PARAM_NAMES,
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "source_optimized_targets": str(opt_target_path),
                "source_ads_cache": str(base.ADS_CACHE_DIR),
                "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
                "connection_count": wrapper.CONNECTION_COUNT,
                "v08_scale_factors": wrapper.V08_SCALE_FACTORS.tolist(),
                "freq_hz": sim.freq_hz.tolist(),
            },
        },
        output_dir / "shared7_param_nns.pt",
    )

    history.to_csv(output_dir / "shared7_param_training_history.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_dir / "shared7_param_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "optimized_vs_shared7_nn_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "optimized_vs_shared7_nn_summary.csv", index=False, encoding="utf-8-sig")
    sign_stats.to_csv(output_dir / "target_and_predicted_parameter_sign_stats.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "source_optimized_targets": str(opt_target_path),
        "source_ads_cache": str(base.ADS_CACHE_DIR),
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "architecture": "seven independent input->30->30->20->1 networks",
        "samples": int(len(dut_df)),
        "epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "best_val_loss": float(history["val_loss"].min()) if len(history) else None,
        "summary": summary.to_dict(orient="records"),
        "parameter_sign_stats": sign_stats.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
    }
    (output_dir / "shared7_param_nn_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "shared7_param_nn_report.md").write_text(
        "\n".join(
            [
                "# V11 ADS Length 0.9 Shared 7-Parameter NN Report",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source optimized targets: `{opt_target_path}`",
                f"- Source ADS cache: `{base.ADS_CACHE_DIR}`",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}`",
                "- Network: seven independent `input -> 30 -> 30 -> 20 -> 1` models, one per circuit parameter.",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Best validation loss: `{report['best_val_loss']}`",
                "",
                "## S-Parameter Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
                "",
                "## Outputs",
                "",
                f"- Checkpoint: `{output_dir / 'shared7_param_nns.pt'}`",
                f"- Training history: `{output_dir / 'shared7_param_training_history.csv'}`",
                f"- Parameter predictions: `{output_dir / 'shared7_param_predictions.csv'}`",
                f"- Metrics: `{output_dir / 'optimized_vs_shared7_nn_metrics.csv'}`",
                f"- Summary: `{output_dir / 'optimized_vs_shared7_nn_summary.csv'}`",
                f"- Parameter sign stats: `{output_dir / 'target_and_predicted_parameter_sign_stats.csv'}`",
                f"- Training/NMSE plot: `{output_dir / 'training_and_nmse_summary.png'}`",
                f"- Parameter scatter: `{output_dir / 'parameter_prediction_scatter.png'}`",
                f"- Comparison plots: `{plot_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "validation_archive.md").write_text(
        "\n".join(
            [
                "# Validation Archive",
                "",
                f"- Entry: `{Path(__file__).name}`",
                "- Status: completed",
                f"- Samples: `{len(dut_df)}`",
                f"- Train/val/test: `{int(masks['train'].sum())}` / `{int(masks['val'].sum())}` / `{int(masks['test'].sum())}`",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Source ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                "",
                "## S-Parameter Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)
    print(dataframe_to_markdown(sign_stats), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
