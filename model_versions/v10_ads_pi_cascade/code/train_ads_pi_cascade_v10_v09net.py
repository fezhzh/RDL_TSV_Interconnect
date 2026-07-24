# -*- coding: utf-8 -*-
"""Train v10 ADS pi cascade with the v09-style multi-head network.

Run this file directly in VS Code. No command-line arguments are required.
It reuses the current 150/50 ADS-length-0.9 single-device cache and optimized
pi target dataset, then retrains only the neural network with the larger v09
multi-head architecture.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10.py"
SOURCE_RUN_LABEL = "ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09"
RUN_LABEL = "ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_v09net"

PARAM_EPOCHS = 950
PARAM_PATIENCE = 120
SPARAM_EPOCHS = 600
SPARAM_PATIENCE = 120
PARAM_LR = 8e-4
SPARAM_LR = 1e-5
BATCH_SIZE = 12


def load_train_module():
    spec = importlib.util.spec_from_file_location("v10_train_ads_pi_cascade", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training script: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_v09_style_net(mod):
    class V09StylePiConnectionNet(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.SiLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.LayerNorm(128),
            )
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(128, 64),
                        nn.SiLU(),
                        nn.LayerNorm(64),
                        nn.Linear(64, len(mod.PI_PARAM_NAMES)),
                    )
                    for _ in range(mod.CONNECTION_COUNT)
                ]
            )

        def forward(self, x):
            z = self.trunk(x)
            return torch.cat([head(z) for head in self.heads], dim=1)

    return V09StylePiConnectionNet


def main():
    mod = load_train_module()
    source_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / SOURCE_RUN_LABEL
    output_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / RUN_LABEL
    targets_path = source_dir / "pi_optimized_targets.csv"
    source_checkpoint = source_dir / "pi_connection_net.pt"
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing optimized pi target dataset: {targets_path}")
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Missing source checkpoint metadata: {source_checkpoint}")

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    ads_settings = checkpoint["metadata"]["ads_settings"]

    output_dir.mkdir(parents=True, exist_ok=True)
    mod.OUTPUT_DIR = output_dir
    mod.ADS_CACHE_DIR = source_dir / "ads_single_device_cache"
    mod.PARAM_EPOCHS = PARAM_EPOCHS
    mod.PARAM_PATIENCE = PARAM_PATIENCE
    mod.SPARAM_EPOCHS = SPARAM_EPOCHS
    mod.SPARAM_PATIENCE = SPARAM_PATIENCE
    mod.PARAM_LR = PARAM_LR
    mod.SPARAM_LR = SPARAM_LR
    mod.BATCH_SIZE = BATCH_SIZE
    mod.PARAM_ANCHOR_WEIGHT = 0.0
    mod.PiConnectionNet = make_v09_style_net(mod)

    mod.set_seed(mod.RANDOM_SEED)
    dut_df = mod.collect_samples()
    sim = mod.load_single_device_simulation(dut_df, ads_settings)
    pi_targets = pd.read_csv(targets_path)
    masks = mod.split_masks(pi_targets)

    x_raw = pi_targets[mod.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = pi_targets[mod.pi_target_columns()].to_numpy(dtype=np.float64)
    x_norm, x_mean, x_std = mod.normalize_by_train(x_raw, masks["train"])
    y_norm, y_mean, y_std = mod.normalize_by_train(y_raw, masks["train"])
    arrays = (x_norm, y_norm, masks, y_mean, y_std, sim)

    device = torch.device("cuda" if mod.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    model = mod.PiConnectionNet(input_dim=x_norm.shape[1]).to(dtype=mod.REAL_DTYPE, device=device)

    param_history = mod.train_param_model(model, x_norm, y_norm, masks, device)
    pretrain_metrics, pretrain_pred_params = mod.evaluate_model(model, dut_df, arrays, device)
    pretrain_summary = mod.summarize_metrics(pretrain_metrics)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "run_label": RUN_LABEL,
                "source_run_label": SOURCE_RUN_LABEL,
                "model_architecture": "v09_style_multihead_for_v10_pi",
                "feature_columns": mod.STRUCTURE_COLUMNS,
                "target_columns": mod.pi_target_columns(),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "pi_param_names": mod.PI_PARAM_NAMES,
                "connection_count": mod.CONNECTION_COUNT,
                "freq_hz": sim.freq_hz.tolist(),
                "ads_settings": ads_settings,
                "ads_device_length_scale": checkpoint["metadata"].get(
                    "ads_device_length_scale", mod.ADS_DEVICE_LENGTH_SCALE
                ),
                "param_output_constraint": "none",
            },
        },
        output_dir / "pi_connection_net_param_pretrain.pt",
    )

    sparam_history = mod.train_sparam_model(model, arrays, device)
    history = pd.concat([param_history, sparam_history], ignore_index=True)
    metrics, pred_params = mod.evaluate_model(model, dut_df, arrays, device)
    summary = mod.summarize_metrics(metrics)
    plot_dir, plot_files = mod.save_comparison_plots(model, dut_df, arrays, metrics, device)

    pi_targets.to_csv(output_dir / "pi_optimized_targets.csv", index=False, encoding="utf-8-sig")
    pretrain_metrics.to_csv(output_dir / "pi_sparam_metrics_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_summary.to_csv(output_dir / "pi_sparam_summary_after_param_pretrain.csv", index=False, encoding="utf-8-sig")
    pretrain_pred_params.to_csv(
        output_dir / "pi_param_predictions_after_param_pretrain.csv", index=False, encoding="utf-8-sig"
    )
    history.to_csv(output_dir / "pi_training_history.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "pi_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "pi_sparam_summary.csv", index=False, encoding="utf-8-sig")
    pred_params.to_csv(output_dir / "pi_param_predictions.csv", index=False, encoding="utf-8-sig")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "run_label": RUN_LABEL,
                "source_run_label": SOURCE_RUN_LABEL,
                "model_architecture": "v09_style_multihead_for_v10_pi",
                "feature_columns": mod.STRUCTURE_COLUMNS,
                "target_columns": mod.pi_target_columns(),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "pi_param_names": mod.PI_PARAM_NAMES,
                "connection_count": mod.CONNECTION_COUNT,
                "freq_hz": sim.freq_hz.tolist(),
                "ads_settings": ads_settings,
                "ads_device_length_scale": checkpoint["metadata"].get(
                    "ads_device_length_scale", mod.ADS_DEVICE_LENGTH_SCALE
                ),
                "param_output_constraint": "none",
            },
        },
        output_dir / "pi_connection_net.pt",
    )

    report = {
        "run_label": RUN_LABEL,
        "source_run_label": SOURCE_RUN_LABEL,
        "output_dir": str(output_dir),
        "model_architecture": {
            "trunk": "Linear(input,256)-SiLU-LayerNorm-Linear(256,256)-SiLU-LayerNorm-Linear(256,128)-SiLU-LayerNorm",
            "heads": "8 x [Linear(128,64)-SiLU-LayerNorm-Linear(64,4)]",
        },
        "param_output_constraint": "none",
        "param_epochs_completed": int(param_history["epoch"].iloc[-1]) if len(param_history) else 0,
        "sparam_epochs_completed": int(sparam_history["epoch"].iloc[-1]) if len(sparam_history) else 0,
        "pretrain_summary": pretrain_summary.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_files,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v10 V09-Style Network Validation",
                "",
                f"- Source run: `{SOURCE_RUN_LABEL}`",
                f"- Output: `{output_dir}`",
                "- Architecture: v09-style shared trunk and eight heads, adapted to four pi parameters per head",
                "- Parameter output constraint: none",
                "- Loss after pretrain: pure complex S-parameter loss",
                f"- Param epochs completed: {report['param_epochs_completed']}",
                f"- S-parameter epochs completed: {report['sparam_epochs_completed']}",
                f"- Comparison plots: `{plot_dir}`",
                "",
                "## Param Pretrain",
                "",
                mod.dataframe_to_markdown(pretrain_summary),
                "",
                "## Final S-Parameter Model",
                "",
                mod.dataframe_to_markdown(summary),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Param pretrain summary:", flush=True)
    print(pretrain_summary.to_string(index=False), flush=True)
    print("Final summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
