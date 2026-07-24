# -*- coding: utf-8 -*-
"""Continue v10 signed-pi training with pure S-parameter loss and no param clamp.

Run this file directly in VS Code. No command-line arguments are required.
It loads the latest 150/50 signed-pi ADS-length-0.9 checkpoint, removes the
connection-parameter output range clamp through the imported training module,
and continues training only against the complex S-parameter target.
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
SOURCE_RUN_LABEL = "ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09"
CONTINUE_RUN_LABEL = "ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_unbounded_sparam_continue"

CONTINUE_EPOCHS = 120
CONTINUE_PATIENCE = 30
CONTINUE_LR = 1e-5


def load_train_module():
    spec = importlib.util.spec_from_file_location("v10_train_ads_pi_cascade", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training script: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    mod = load_train_module()
    source_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / SOURCE_RUN_LABEL
    output_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / CONTINUE_RUN_LABEL
    checkpoint_path = source_dir / "pi_connection_net.pt"
    targets_path = source_dir / "pi_optimized_targets.csv"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing source optimized targets: {targets_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mod.OUTPUT_DIR = output_dir
    mod.ADS_CACHE_DIR = source_dir / "ads_single_device_cache"
    mod.SPARAM_EPOCHS = CONTINUE_EPOCHS
    mod.SPARAM_PATIENCE = CONTINUE_PATIENCE
    mod.SPARAM_LR = CONTINUE_LR
    mod.PARAM_ANCHOR_WEIGHT = 0.0

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]

    mod.set_seed(mod.RANDOM_SEED)
    dut_df = mod.collect_samples()
    sim = mod.load_single_device_simulation(dut_df, metadata["ads_settings"])
    masks = mod.split_masks(dut_df)

    x_raw = dut_df[mod.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    x_mean = np.asarray(metadata["x_mean"], dtype=np.float64)
    x_std = np.asarray(metadata["x_std"], dtype=np.float64)
    x_norm = (x_raw - x_mean) / x_std

    pi_targets = pd.read_csv(targets_path)
    y_raw = pi_targets[mod.pi_target_columns()].to_numpy(dtype=np.float64)
    y_mean = np.asarray(metadata["y_mean"], dtype=np.float64)
    y_std = np.asarray(metadata["y_std"], dtype=np.float64)
    y_norm = (y_raw - y_mean) / y_std
    arrays = (x_norm, y_norm, masks, y_mean, y_std, sim)

    device = torch.device("cuda" if mod.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = mod.PiConnectionNet(input_dim=x_norm.shape[1]).to(dtype=mod.REAL_DTYPE, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    before_metrics, before_pred_params = mod.evaluate_model(model, dut_df, arrays, device)
    before_summary = mod.summarize_metrics(before_metrics)
    history = mod.train_sparam_model(model, arrays, device)
    after_metrics, after_pred_params = mod.evaluate_model(model, dut_df, arrays, device)
    after_summary = mod.summarize_metrics(after_metrics)
    plot_dir, plot_files = mod.save_comparison_plots(model, dut_df, arrays, after_metrics, device)

    before_metrics.to_csv(output_dir / "pi_sparam_metrics_before_continue.csv", index=False, encoding="utf-8-sig")
    before_summary.to_csv(output_dir / "pi_sparam_summary_before_continue.csv", index=False, encoding="utf-8-sig")
    before_pred_params.to_csv(output_dir / "pi_param_predictions_before_continue.csv", index=False, encoding="utf-8-sig")
    history.to_csv(output_dir / "pi_sparam_continue_history.csv", index=False, encoding="utf-8-sig")
    after_metrics.to_csv(output_dir / "pi_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    after_summary.to_csv(output_dir / "pi_sparam_summary.csv", index=False, encoding="utf-8-sig")
    after_pred_params.to_csv(output_dir / "pi_param_predictions.csv", index=False, encoding="utf-8-sig")
    pi_targets.to_csv(output_dir / "pi_optimized_targets.csv", index=False, encoding="utf-8-sig")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                **metadata,
                "stage": "unbounded_sparam_continue",
                "source_run_label": SOURCE_RUN_LABEL,
                "continue_run_label": CONTINUE_RUN_LABEL,
                "param_output_constraint": "none",
                "continue_epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
                "continue_lr": CONTINUE_LR,
                "continue_patience": CONTINUE_PATIENCE,
            },
        },
        output_dir / "pi_connection_net_unbounded_sparam_continue.pt",
    )

    report = {
        "source_run_label": SOURCE_RUN_LABEL,
        "continue_run_label": CONTINUE_RUN_LABEL,
        "source_checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "param_output_constraint": "none",
        "loss": "pure complex S-parameter loss",
        "continue_epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "continue_lr": CONTINUE_LR,
        "continue_patience": CONTINUE_PATIENCE,
        "ads_device_length_scale": metadata.get("ads_device_length_scale", mod.ADS_DEVICE_LENGTH_SCALE),
        "summary_before_continue": before_summary.to_dict(orient="records"),
        "summary_after_continue": after_summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_files,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v10 Unbounded S-Parameter Continuation Validation",
                "",
                f"- Source run: `{SOURCE_RUN_LABEL}`",
                f"- Output: `{output_dir}`",
                "- Parameter output constraint: none",
                "- Loss: pure complex S-parameter loss",
                f"- Continue epochs completed: {report['continue_epochs_completed']}",
                f"- Continue learning rate: {CONTINUE_LR}",
                f"- Comparison plots: `{plot_dir}`",
                "",
                "## Before Continue",
                "",
                mod.dataframe_to_markdown(before_summary),
                "",
                "## After Continue",
                "",
                mod.dataframe_to_markdown(after_summary),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Before continue:", flush=True)
    print(before_summary.to_string(index=False), flush=True)
    print("After continue:", flush=True)
    print(after_summary.to_string(index=False), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
