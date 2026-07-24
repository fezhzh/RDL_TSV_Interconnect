# -*- coding: utf-8 -*-
"""Regenerate v10 comparison plots from the saved checkpoint.

Run this file directly in VS Code. No command-line arguments are required.
It reuses the saved ADS cache, metrics CSV, and model checkpoint from the
default v10 output directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10.py"


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
    checkpoint_path = mod.OUTPUT_DIR / "pi_connection_net.pt"
    metrics_path = mod.OUTPUT_DIR / "pi_sparam_metrics.csv"
    targets_path = mod.OUTPUT_DIR / "pi_optimized_targets.csv"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing optimized target CSV: {targets_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]

    dut_df = mod.collect_samples()
    sim = mod.load_single_device_simulation(dut_df, metadata["ads_settings"])
    metrics = pd.read_csv(metrics_path)

    x_raw = dut_df[mod.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    x_mean = np.asarray(metadata["x_mean"], dtype=np.float64)
    x_std = np.asarray(metadata["x_std"], dtype=np.float64)
    x_norm = (x_raw - x_mean) / x_std

    y_raw = pd.read_csv(targets_path)[mod.pi_target_columns()].to_numpy(dtype=np.float64)
    y_mean = np.asarray(metadata["y_mean"], dtype=np.float64)
    y_std = np.asarray(metadata["y_std"], dtype=np.float64)
    y_norm = (y_raw - y_mean) / y_std
    masks = mod.split_masks(dut_df)
    arrays = (x_norm, y_norm, masks, y_mean, y_std, sim)

    device = torch.device("cpu")
    model = mod.PiConnectionNet(input_dim=x_norm.shape[1]).to(dtype=mod.REAL_DTYPE, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    plot_dir, plot_files = mod.save_comparison_plots(model, dut_df, arrays, metrics, device)
    print(f"Regenerated {len(plot_files)} comparison plots in: {plot_dir}")
    for path in plot_files:
        print(path)


if __name__ == "__main__":
    main()
