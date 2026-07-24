# -*- coding: utf-8 -*-
"""Plot the v12 TSV single-device model against LHS400_Connection2 HFSS data.

Run this file directly in VS Code. No command-line arguments are required.
The TSV model input uses the CSV `r_tsv` value directly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skrf as rf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
PLOT_SCRIPT = THIS_DIR / "plot_v12_single_device_model_vs_hfss.py"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "tsv_connection2_model_vs_hfss"
)
CSV_PATH = PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train" / "TSV_variations_record.csv"
SNP_DIR = PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train" / "TSV"
RANDOM_SEED = 20260712
PLOT_RANDOM_COUNT = 8
PLOT_WORST_COUNT = 8


def load_plot_module():
    spec = importlib.util.spec_from_file_location("v12_single_device_plot_helpers", PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plot helper: {PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_plot_module()


def nmse_s11_s21_ri(target, pred):
    return helpers.nmse_s11_s21_ri(target, pred)


def evaluate_tsv():
    model = helpers.TsvParamModel().to(dtype=helpers.REAL_DTYPE).eval()
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
    rows = []
    cache = {}
    for i, row in df.iterrows():
        dut = int(row["dut_index"])
        path = SNP_DIR / f"dut{dut}.s2p"
        nw = rf.Network(str(path))
        x = np.array([[float(row["r_tsv"]), float(row["h_tsv"]), float(row["pitch"])]], dtype=np.float64)
        with torch.no_grad():
            params = model(torch.tensor(x, dtype=helpers.REAL_DTYPE)).cpu().numpy()[0]
        pred = helpers.v09.circuit_params_to_s_np(params, float(row["h_tsv"]), nw.f)
        rows.append(
            {
                "dut_index": dut,
                "r_tsv": float(row["r_tsv"]),
                "h_tsv": float(row["h_tsv"]),
                "pitch": float(row["pitch"]),
                "s_mse": float(np.mean(np.abs(pred - nw.s) ** 2)),
                "nmse_s11_s21_ri": nmse_s11_s21_ri(nw.s, pred),
                "s11_db_mae": float(np.mean(np.abs(helpers.db20(pred[:, 0, 0]) - helpers.db20(nw.s[:, 0, 0])))),
                "s21_db_mae": float(np.mean(np.abs(helpers.db20(pred[:, 1, 0]) - helpers.db20(nw.s[:, 1, 0])))),
                "s11_phase_mae_deg": float(np.mean(np.abs(helpers.wrapped_phase_delta_deg(pred[:, 0, 0], nw.s[:, 0, 0])))),
                "s21_phase_mae_deg": float(np.mean(np.abs(helpers.wrapped_phase_delta_deg(pred[:, 1, 0], nw.s[:, 1, 0])))),
            }
        )
        cache[dut] = (row.copy(), nw.s, pred, nw.f)
        if (i + 1) % 100 == 0:
            print(f"TSV Connection2: evaluated {i + 1}/{len(df)}", flush=True)
    return pd.DataFrame(rows), cache


def plot_one(dut: int, row: pd.Series, hfss_s: np.ndarray, pred_s: np.ndarray, freq_hz: np.ndarray, out_path: Path):
    freq_ghz = freq_hz / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle(
        f"TSV Connection2 dut{dut} | r={float(row['r_tsv']):.2f} um, h={float(row['h_tsv']):.2f} um, pitch={float(row['pitch']):.2f} um",
        x=0.02,
        y=0.98,
        ha="left",
    )
    specs = [
        ("S11 magnitude (dB)", lambda s: helpers.db20(s[:, 0, 0])),
        ("S21 magnitude (dB)", lambda s: helpers.db20(s[:, 1, 0])),
        ("S11 phase (deg)", lambda s: np.rad2deg(np.unwrap(np.angle(s[:, 0, 0])))),
        ("S21 phase (deg)", lambda s: np.rad2deg(np.unwrap(np.angle(s[:, 1, 0])))),
    ]
    for ax, (title, fn) in zip(axes.ravel(), specs):
        ax.plot(freq_ghz, fn(hfss_s), label="HFSS", color="black", linewidth=1.8)
        ax.plot(freq_ghz, fn(pred_s), label="Model", color="#dc2626", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, cache = evaluate_tsv()
    summary = pd.DataFrame(
        [
            {
                "device": "TSV",
                "dataset": "LHS400_Connection2/train",
                "count": int(len(metrics)),
                "r_tsv_min": float(metrics["r_tsv"].min()),
                "r_tsv_max": float(metrics["r_tsv"].max()),
                "s_mse_mean": float(metrics["s_mse"].mean()),
                "s_mse_median": float(metrics["s_mse"].median()),
                "nmse_s11_s21_ri_mean": float(metrics["nmse_s11_s21_ri"].mean()),
                "nmse_s11_s21_ri_median": float(metrics["nmse_s11_s21_ri"].median()),
                "s11_db_mae_mean": float(metrics["s11_db_mae"].mean()),
                "s21_db_mae_mean": float(metrics["s21_db_mae"].mean()),
                "s11_phase_mae_deg_mean": float(metrics["s11_phase_mae_deg"].mean()),
                "s21_phase_mae_deg_mean": float(metrics["s21_phase_mae_deg"].mean()),
            }
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "tsv_connection2_model_vs_hfss_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "tsv_connection2_model_vs_hfss_summary.csv", index=False, encoding="utf-8-sig")

    random_ids = metrics["dut_index"].sample(n=min(PLOT_RANDOM_COUNT, len(metrics)), random_state=RANDOM_SEED).tolist()
    worst_ids = metrics.sort_values("nmse_s11_s21_ri", ascending=False).head(PLOT_WORST_COUNT)["dut_index"].tolist()
    for group, dut_ids in [("random", random_ids), ("worst_nmse", worst_ids)]:
        for dut in dut_ids:
            row, hfss_s, pred_s, freq_hz = cache[int(dut)]
            plot_one(int(dut), row, hfss_s, pred_s, freq_hz, OUTPUT_DIR / "plots" / group / f"TSV_connection2_dut{dut}_{group}.png")

    report = {
        "entry": Path(__file__).name,
        "output_dir": str(OUTPUT_DIR),
        "dataset": str(CSV_PATH),
        "snp_dir": str(SNP_DIR),
        "feature_mapping": "model input = [r_tsv, h_tsv, pitch]",
        "summary": summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "tsv_connection2_model_vs_hfss_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 TSV Connection2 Model vs HFSS Validation",
                "",
                f"- Entry: `{Path(__file__).name}`",
                "- Dataset: `HFSS_sim/LHS400_Connection2/train/TSV`.",
                "- Feature mapping: model input uses `[r_tsv, h_tsv, pitch]` directly.",
                f"- Output: `{OUTPUT_DIR}`",
                f"- Metrics CSV: `{OUTPUT_DIR / 'tsv_connection2_model_vs_hfss_metrics.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'tsv_connection2_model_vs_hfss_summary.csv'}`",
                f"- Plots: `{OUTPUT_DIR / 'plots'}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
