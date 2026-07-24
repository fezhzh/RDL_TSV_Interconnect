# -*- coding: utf-8 -*-
"""Plot RDL model accuracy comparison across five LHS training datasets.

Run directly in VS Code. No command-line arguments are required.
"""

import sys
import os
from pathlib import Path

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "results"
SUMMARY_CSV = RESULT_ROOT / "sparam_finetuned_models" / "summary_metrics.csv"
OUTPUT_DIR = RESULT_ROOT / "dataset_model_comparison_plots"

SAVE_PLOTS = True
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1").strip().lower() not in {"0", "false", "no"}
DPI = 180

DATASET_ORDER = ["lhs100", "lhs200", "lhs400", "lhs800", "lhs100_lhs200_lhs400_lhs800"]
DATASET_LABELS = {
    "lhs100": "LHS100",
    "lhs200": "LHS200",
    "lhs400": "LHS400",
    "lhs800": "LHS800",
    "lhs100_lhs200_lhs400_lhs800": "All",
}
DEVICE_ORDER = ["TMRDL", "BSMRDL"]
MODEL_LABELS = {
    "matlab_param_nn": "Param NN",
    "sparam_finetuned": "S-param finetuned",
}
COLORS = {
    "TMRDL": "#2563eb",
    "BSMRDL": "#dc2626",
    "matlab_param_nn": "#64748b",
    "sparam_finetuned": "#16a34a",
}


def load_test_metrics():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    df = df[df["split"].eq("test")].copy()
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["device"] = pd.Categorical(df["device"], DEVICE_ORDER, ordered=True)
    df = df.sort_values(["dataset", "device", "model"]).reset_index(drop=True)
    return df


def metric_values(df, model_name, device_name, metric_name):
    rows = df[(df["model"].eq(model_name)) & (df["device"].eq(device_name))].set_index("dataset")
    return np.array([rows.loc[name, metric_name] for name in DATASET_ORDER], dtype=float)


def save_or_show(fig, filename):
    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / filename}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_sparam_mse(df):
    x = np.arange(len(DATASET_ORDER))
    labels = [DATASET_LABELS[name] for name in DATASET_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    fig.suptitle("Test S-parameter MSE vs Training Dataset", x=0.02, ha="left", fontsize=13)

    for ax, device_name in zip(axes, DEVICE_ORDER):
        for model_name, linestyle, marker in [
            ("matlab_param_nn", "--", "o"),
            ("sparam_finetuned", "-", "s"),
        ]:
            values = metric_values(df, model_name, device_name, "sparam_mse")
            ax.plot(
                x,
                values,
                label=MODEL_LABELS[model_name],
                color=COLORS[model_name],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
            )
        ax.set_title(device_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Complex S MSE")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_or_show(fig, "test_sparam_mse_vs_dataset.png")


def plot_s11_s21_mae(df):
    x = np.arange(len(DATASET_ORDER))
    labels = [DATASET_LABELS[name] for name in DATASET_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("Test S11/S21 dB MAE vs Training Dataset After S-parameter Fine-tuning", x=0.02, ha="left", fontsize=13)

    for col, device_name in enumerate(DEVICE_ORDER):
        for row, (metric_name, title) in enumerate([("s11_db_mae", "S11 MAE"), ("s21_db_mae", "S21 MAE")]):
            ax = axes[row, col]
            values = metric_values(df, "sparam_finetuned", device_name, metric_name)
            ax.plot(x, values, color=COLORS[device_name], marker="o", linewidth=1.8)
            ax.set_title(f"{device_name} {title}")
            ax.set_ylabel("MAE (dB)")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_or_show(fig, "test_s11_s21_mae_after_finetune.png")


def plot_before_after_bars(df):
    labels = [DATASET_LABELS[name] for name in DATASET_ORDER]
    x = np.arange(len(DATASET_ORDER))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Before/After S-parameter Fine-tuning on Test Set", x=0.02, ha="left", fontsize=13)

    for col, device_name in enumerate(DEVICE_ORDER):
        for row, (metric_name, title, use_log) in enumerate(
            [("sparam_mse", "Complex S MSE", True), ("s21_db_mae", "S21 MAE (dB)", False)]
        ):
            ax = axes[row, col]
            before = metric_values(df, "matlab_param_nn", device_name, metric_name)
            after = metric_values(df, "sparam_finetuned", device_name, metric_name)
            ax.bar(x - width / 2, before, width, label="Param NN", color=COLORS["matlab_param_nn"])
            ax.bar(x + width / 2, after, width, label="S-param finetuned", color=COLORS["sparam_finetuned"])
            if use_log:
                ax.set_yscale("log")
            ax.set_title(f"{device_name} {title}")
            ax.grid(True, axis="y", which="both", alpha=0.28)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_or_show(fig, "test_before_after_finetune_bars.png")


def write_plot_table(df):
    out = df[df["model"].eq("sparam_finetuned")][
        ["dataset", "device", "sparam_mse", "s11_db_mae", "s21_db_mae", "s12_db_mae", "s22_db_mae"]
    ].copy()
    out["dataset"] = out["dataset"].astype(str)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "test_finetuned_metrics_for_plot.csv"
    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_file}")


def main():
    df = load_test_metrics()
    write_plot_table(df)
    plot_sparam_mse(df)
    plot_s11_s21_mae(df)
    plot_before_after_bars(df)
    print(f"Done. Plot directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
