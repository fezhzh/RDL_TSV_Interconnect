# -*- coding: utf-8 -*-
"""View saved RDL_TSV mat4 optimization results.

Run this file directly in VS Code after ``Calc_SP_and_Opt2.py`` has generated
Touchstone files under ``model_versions/v02_mat4_cascade_and_sparameter_optimization/results/RDL_TSV_mat4_opt2``.
"""

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skrf as rf

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HFSS_DIR = PROJECT_ROOT / "snp_data" / "RDL_TSV_Snp"
RESULT_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "results" / "RDL_TSV_mat4_opt2"

# Configure these values before running directly from VS Code.
PLOT_DUT_INDICES = None  # Example: [1, 2, 3]. Use None to scan saved results.
PLOT_LIMIT = None  # Use None to show every saved result.
SHOW_SUMMARY = True


def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).stem)
    return [int(part) if part.isdigit() else part for part in parts]


def dut_index(path):
    match = re.search(r"dut(\d+)\.s2p$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse DUT index from {path}")
    return int(match.group(1))


def db20(value):
    return 20.0 * np.log10(np.maximum(np.abs(value), 1e-30))


def mse(ref_s, pred_s):
    return float(np.mean(np.abs(ref_s - pred_s) ** 2))


def result_files():
    if PLOT_DUT_INDICES is None:
        files = sorted((RESULT_DIR / "optimized_with_cn3").glob("dut*.s2p"), key=natural_key)
    else:
        files = [RESULT_DIR / "optimized_with_cn3" / f"dut{idx}.s2p" for idx in PLOT_DUT_INDICES]
        files = [path for path in files if path.exists()]

    if PLOT_LIMIT is not None:
        files = files[:PLOT_LIMIT]
    if not files:
        raise FileNotFoundError(f"No optimized_with_cn3 dut*.s2p files found under {RESULT_DIR}")
    return files


def load_case(idx):
    paths = {
        "HFSS": HFSS_DIR / f"dut{idx}.s2p",
        "Direct": RESULT_DIR / "direct" / f"dut{idx}.s2p",
        "Opt with Cn3": RESULT_DIR / "optimized_with_cn3" / f"dut{idx}.s2p",
        "Opt w/o Cn3": RESULT_DIR / "optimized_without_cn3" / f"dut{idx}.s2p",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    return {name: rf.Network(str(path)) for name, path in paths.items()}


def plot_case(idx):
    networks = load_case(idx)
    hfss = networks["HFSS"]
    freq_ghz = hfss.f / 1e9
    ports = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]
    styles = {
        "HFSS": {"color": "black", "linestyle": "-", "linewidth": 2.2},
        "Direct": {"color": "#64748b", "linestyle": ":", "linewidth": 1.9},
        "Opt with Cn3": {"color": "#dc2626", "linestyle": "--", "linewidth": 1.9},
        "Opt w/o Cn3": {"color": "#ea580c", "linestyle": "-.", "linewidth": 1.9},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    fig.suptitle(f"dut{idx}.s2p optimization comparison", x=0.02, y=0.985, ha="left", fontsize=16)

    metric_text = [
        f"Direct MSE: {mse(hfss.s, networks['Direct'].s):.3e}",
        f"With Cn3 MSE: {mse(hfss.s, networks['Opt with Cn3'].s):.3e}",
        f"W/o Cn3 MSE: {mse(hfss.s, networks['Opt w/o Cn3'].s):.3e}",
    ]
    fig.text(0.02, 0.95, " | ".join(metric_text), ha="left", va="top", fontsize=10, color="#475569")

    for ax, (m, n, name) in zip(axes.ravel(), ports):
        for label, nw in networks.items():
            ax.plot(freq_ghz, db20(nw.s[:, m, n]), label=label, **styles[label])
        ax.set_title(f"{name} magnitude")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.9, wspace=0.2, hspace=0.3)
    plt.show()
    plt.close(fig)


def print_summary():
    summary_csv = RESULT_DIR / "optimization_summary.csv"
    if not summary_csv.exists():
        print(f"未找到汇总 CSV: {summary_csv}")
        return

    df = pd.read_csv(summary_csv)
    valid = df[df.get("error").isna()] if "error" in df else df
    if valid.empty:
        print("汇总 CSV 中没有有效样本。")
        return

    print(f"汇总文件: {summary_csv}")
    print(f"有效样本: {len(valid)} / {len(df)}")
    print(f"平均直接级联 MSE: {valid['direct_mse'].mean():.4e}")
    print(f"平均优化 MSE（含 Cn3）: {valid['optimized_with_cn3_mse'].mean():.4e}")
    print(f"平均优化 MSE（无 Cn3）: {valid['optimized_without_cn3_mse'].mean():.4e}")


def main():
    if SHOW_SUMMARY:
        print_summary()

    files = result_files()
    print(f"将显示 {len(files)} 个优化对比图")
    for path in files:
        idx = dut_index(path)
        print(f">>> 显示 dut{idx}.s2p")
        plot_case(idx)


if __name__ == "__main__":
    main()
