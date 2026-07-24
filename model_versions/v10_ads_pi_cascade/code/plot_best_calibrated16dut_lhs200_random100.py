# -*- coding: utf-8 -*-
"""Plot best test samples for the calibrated 16-DUT LHS200 100/100 v10 run.

Run this file directly in VS Code. No command-line arguments are required.
It reloads the saved checkpoint and cached ADS data, then saves the best test
samples by final Pi-NN NMSE.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10_calibrated16dut_lhs200_random100.py"
BEST_COUNT = 8


def load_train_module():
    spec = importlib.util.spec_from_file_location("v10_calibrated16dut_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_base():
    entry = load_train_module()
    base = entry.load_base_module()
    base.RUN_LABEL = "ads_pi_cascade_lhs200_random100train_100test_calibrated16dut"
    base.OUTPUT_DIR = base.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / base.RUN_LABEL
    base.ADS_CACHE_DIR = base.OUTPUT_DIR / "ads_single_device_cache"
    base.SIMULATION_BACKEND = "ads"
    base.LHS200_MODEL_COUNT = 100
    base.LHS200_TEST_COUNT = 100
    base.LHS200_RANDOM_SPLIT_SEED = 20260707
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0
    base.BASE_ADS_SETTINGS = {
        "calibration_source": "ads_single_device_calibration_16dut",
        "rdl_settings": {
            "er_si": 10.2,
            "cond": 5.8e7,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.85,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 0.8,
        },
        "tsv_settings": {
            "er_si": 11.9,
            "cond": 5.8e7,
            "tand": 0.005,
            "c1_scale": 1.0,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.1,
            "d_scale": 1.0,
        },
    }
    return base


def main():
    base = configure_base()
    output_dir = base.OUTPUT_DIR
    checkpoint_path = output_dir / "pi_connection_net.pt"
    metrics = pd.read_csv(output_dir / "pi_sparam_metrics.csv", encoding="utf-8-sig")
    pi_targets = pd.read_csv(output_dir / "pi_optimized_targets.csv", encoding="utf-8-sig")

    sort_col = "pi_nn_nmse_s11_s21_ri"
    best_metrics = metrics[metrics["split"].eq("test")].sort_values(sort_col, ascending=True).head(BEST_COUNT)
    best_ids = set(best_metrics["sample_id"])

    dut_df = base.collect_samples()
    sim = base.load_single_device_simulation(dut_df, base.BASE_ADS_SETTINGS)

    pi_targets = pi_targets.set_index("sample_id").loc[dut_df["sample_id"]].reset_index()
    x_raw = pi_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = pi_targets[base.pi_target_columns()].to_numpy(dtype=np.float64)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint["metadata"]
    x_mean = np.asarray(meta["x_mean"], dtype=np.float64)
    x_std = np.asarray(meta["x_std"], dtype=np.float64)
    y_mean = np.asarray(meta["y_mean"], dtype=np.float64)
    y_std = np.asarray(meta["y_std"], dtype=np.float64)
    x_norm = (x_raw - x_mean) / x_std
    y_norm = (y_raw - y_mean) / y_std

    device = torch.device("cpu")
    model = base.PiConnectionNet(input_dim=x_norm.shape[1]).to(dtype=base.REAL_DTYPE, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    freq_ghz = sim.freq_hz / 1e9
    group_dir = output_dir / "comparison_plots" / "best_test"
    group_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for _, metric in best_metrics.iterrows():
        idx = int(dut_df.index[dut_df["sample_id"].eq(metric["sample_id"])][0])
        x_b = torch.tensor(x_norm[idx : idx + 1], dtype=base.REAL_DTYPE, device=device)
        base_b = torch.tensor(sim.base_abcds[idx : idx + 1], dtype=base.COMPLEX_DTYPE, device=device)
        with torch.no_grad():
            p_flat = base.denormalize_params(model(x_b), y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, base.CONNECTION_COUNT, len(base.PI_PARAM_NAMES))
            pred_s = base.abcd2s_torch(base.cascade_with_pi_torch(base_b, p_all, omega_t)).cpu().numpy()[0]

        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[idx])))
        opt_p_flat = y_norm[idx] * y_std + y_mean
        optimized_s = base.abcd2s(base.cascade_with_pi_np(sim.base_abcds[idx], 2.0 * np.pi * sim.freq_hz, opt_p_flat))
        target_s = sim.target_s[idx]
        optimized_nmse = base.nmse_s11_s21_real_imag(target_s, optimized_s)
        direct_nmse = float(metric["direct_nmse_s11_s21_ri"])
        model_nmse = float(metric["pi_nn_nmse_s11_s21_ri"])

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
        title = (
            f"best_test | {metric['sample_id']} | NMSE direct={direct_nmse:.3e} | "
            f"optimized={optimized_nmse:.3e} | pi-NN={model_nmse:.3e}"
        )
        fig.suptitle(title, x=0.02, y=0.985, ha="left")
        for ax, (m, n, label, component_fn) in zip(
            axes.ravel(),
            [
                (0, 0, "S11 real", np.real),
                (0, 0, "S11 imag", np.imag),
                (1, 0, "S21 real", np.real),
                (1, 0, "S21 imag", np.imag),
            ],
        ):
            ax.plot(freq_ghz, component_fn(target_s[:, m, n]), label="HFSS simulation", color="black", linewidth=1.8)
            ax.plot(freq_ghz, component_fn(direct_s[:, m, n]), label="ADS direct cascade", color="#64748b", linestyle=":")
            ax.plot(freq_ghz, component_fn(optimized_s[:, m, n]), label="Optimized pi", color="#16a34a", linestyle="--")
            ax.plot(freq_ghz, component_fn(pred_s[:, m, n]), label="Pi-NN model", color="#dc2626", linestyle="-.")
            ax.set_title(label)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = group_dir / f"{metric['sample_id']}_comparison.png"
        fig.savefig(out_path)
        plt.close(fig)
        saved.append(
            {
                "sample_id": metric["sample_id"],
                "dut_index": int(metric["dut_index"]),
                "pi_nn_nmse_s11_s21_ri": model_nmse,
                "direct_nmse_s11_s21_ri": direct_nmse,
                "optimized_nmse_s11_s21_ri": optimized_nmse,
                "plot_path": str(out_path),
            }
        )

    summary = pd.DataFrame(saved)
    summary_path = output_dir / "best_test_comparison_plots.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    archive_path = output_dir / "best_test_comparison_plots_validation.md"
    archive_path.write_text(
        "\n".join(
            [
                "# Best Test Comparison Plot Validation",
                "",
                f"- Source result: `{output_dir}`",
                f"- Checkpoint: `{checkpoint_path}`",
                f"- Selection: lowest `{sort_col}` among test samples.",
                f"- Plot directory: `{group_dir}`",
                f"- Plot count: `{len(saved)}`",
                f"- Summary CSV: `{summary_path}`",
                "",
                "## Selected Samples",
                "",
                base.dataframe_to_markdown(summary),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(saved)} best-test plots: {group_dir}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
