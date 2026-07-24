# -*- coding: utf-8 -*-
"""Plot the best K-fold multi-head S-parameter comparison cases.

Run this file directly in VS Code. It reads existing K-fold metrics and
prediction CSV files, selects the test cases with the lowest
``multihead_mse_vs_hfss``, and saves HFSS / Direct / Optimized / Multi-head
comparison plots.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skrf as rf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V08_CODE_DIR = PROJECT_ROOT / "model_versions" / "v08_connection_multihead" / "code"
V07_CODE_DIR = PROJECT_ROOT / "model_versions" / "v07_connection_param_and_sparam_finetune" / "code"
V03_CODE_DIR = PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "code"
SPARAM_CODE_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "code"
for path in [V08_CODE_DIR, V07_CODE_DIR, V03_CODE_DIR, SPARAM_CODE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Calc_SP_and_Opt2 as opt2
import train_connection_network_multihead_sparam as base
import train_connection_network_params as param_train


KFOLD_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v08_connection_multihead"
    / "results"
    / "connection_network_multihead_sparam_with_cn3_kfold"
)
PLOT_DIR = KFOLD_DIR / "best_multihead_sparam_plots"
N_BEST_PLOTS = 6


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def main():
    metrics_path = KFOLD_DIR / "kfold_all_test_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing K-fold metrics: {metrics_path}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_path)
    selected = metrics.sort_values("multihead_mse_vs_hfss", ascending=True).head(N_BEST_PLOTS).copy()
    selected.to_csv(PLOT_DIR / "best_plot_selection.csv", index=False, encoding="utf-8-sig")

    arrays = base.build_training_arrays()
    _, _, _, _, _, _, _, _, base_abcds, _, dut_indices, _, _ = arrays

    made = []
    for _, row in selected.iterrows():
        dut_idx = int(row["dut_index"])
        fold = int(row["fold"])
        pred_path = KFOLD_DIR / f"fold_{fold:02d}" / "multihead_param_predictions.csv"
        pred_df = pd.read_csv(pred_path)
        pred_rows = pred_df[pred_df["dut_index"].astype(int) == dut_idx]
        if pred_rows.empty:
            raise RuntimeError(f"Missing prediction row for fold {fold}, dut{dut_idx}: {pred_path}")
        pred_row = pred_rows.iloc[0]

        array_indices = np.where(dut_indices.astype(int) == dut_idx)[0]
        if len(array_indices) == 0:
            raise RuntimeError(f"Missing base arrays for dut{dut_idx}")
        array_idx = int(array_indices[0])

        hfss_nw = rf.Network(str(opt2.S2P_DIR / f"dut{dut_idx}.s2p"))
        optimized_nw = rf.Network(str(opt2.OUTPUT_DIR / param_train.TARGET_VARIANT / f"dut{dut_idx}.s2p"))
        direct_s = opt2.abcd2s(opt2.cascade_direct(list(base_abcds[array_idx])))

        predicted_params = []
        for conn_idx in range(1, param_train.CONNECTION_COUNT + 1):
            for name in param_train.SCALE_COLUMNS:
                predicted_params.append(pred_row[f"pred_conn{conn_idx}_{name}"])

        pred_s = opt2.abcd2s(
            opt2.cascade_with_corrections(
                list(base_abcds[array_idx]),
                2.0 * np.pi * hfss_nw.f,
                np.asarray(predicted_params, dtype=np.float64),
                include_cn3=True,
            )
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        fig.suptitle(
            f"fold {fold:02d} / dut{dut_idx}.s2p best K-fold multi-head comparison",
            x=0.02,
            y=0.985,
            ha="left",
        )
        fig.text(
            0.02,
            0.955,
            (
                f"Direct={row['direct_mse_vs_hfss']:.3e} | "
                f"Optimized={row['optimized_mse_vs_hfss']:.3e} | "
                f"Multi-head={row['multihead_mse_vs_hfss']:.3e}"
            ),
            ha="left",
            va="top",
            fontsize=9,
            color="#475569",
        )

        freq_ghz = hfss_nw.f / 1e9
        for ax, (m, n, label) in zip(
            axes.ravel(),
            [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")],
        ):
            ax.plot(freq_ghz, db20(hfss_nw.s[:, m, n]), label="HFSS", color="black", linewidth=2.0)
            ax.plot(freq_ghz, db20(direct_s[:, m, n]), label="Direct", color="#64748b", linestyle=":", linewidth=1.6)
            ax.plot(freq_ghz, db20(optimized_nw.s[:, m, n]), label="Optimized", color="#dc2626", linestyle="--", linewidth=1.6)
            ax.plot(freq_ghz, db20(pred_s[:, m, n]), label="Multi-head NN", color="#16a34a", linestyle="-.", linewidth=1.6)
            ax.set_title(f"{label} magnitude")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.9, wspace=0.2, hspace=0.3)
        out_path = PLOT_DIR / f"fold{fold:02d}_dut{dut_idx}_best_multihead_sparam.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        made.append(out_path)

    print(f"Saved {len(made)} best-case plots to {PLOT_DIR}")
    print(selected[["fold", "dut_index", "multihead_mse_vs_hfss", "direct_mse_vs_hfss", "optimized_mse_vs_hfss"]].to_string(index=False))


if __name__ == "__main__":
    main()
