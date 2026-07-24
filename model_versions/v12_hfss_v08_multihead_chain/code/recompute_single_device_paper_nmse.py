# -*- coding: utf-8 -*-
"""Recompute paper-style NMSE for the v12 RDL and TSV single-device models.

Run this file directly in VS Code. No command-line arguments are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_rdl_connection2_sparam_model as rdl
import train_tsv_connection2_sparam_model as tsv


OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "single_device_paper_nmse_recalculation"
)
RDL_CHECKPOINT = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "rdl_connection2_sparam_model"
    / "rdl_connection2_sparam_net.pt"
)
TSV_CHECKPOINT = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "tsv_connection2_sparam_continue"
    / "tsv_connection2_sparam_continue_net.pt"
)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            cells.append(f"{value:.8g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paper_nmse(pred_s: np.ndarray, true_s: np.ndarray) -> float:
    y_pred = np.concatenate(
        [
            pred_s[:, 0, 0].real,
            pred_s[:, 0, 0].imag,
            pred_s[:, 1, 0].real,
            pred_s[:, 1, 0].imag,
        ]
    ).astype(float)
    y_true = np.concatenate(
        [
            true_s[:, 0, 0].real,
            true_s[:, 0, 0].imag,
            true_s[:, 1, 0].real,
            true_s[:, 1, 0].imag,
        ]
    ).astype(float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum((y_true - y_pred) ** 2) / denom)


def summarize(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("split", as_index=False)
        .agg(
            count=("dut_index", "count"),
            nmse_mean=("paper_nmse_s11_s21_ri", "mean"),
            nmse_median=("paper_nmse_s11_s21_ri", "median"),
            nmse_percent_mean=("paper_nmse_percent", "mean"),
            nmse_percent_median=("paper_nmse_percent", "median"),
        )
        .sort_values("split")
    )
    return detail, summary


def compute_rdl(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(rdl.PARAM_CSV, encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
    arrays = rdl.prepare_arrays(df)
    checkpoint = torch.load(RDL_CHECKPOINT, map_location=device, weights_only=False)
    model = rdl.RdlParamNet().to(dtype=rdl.REAL_DTYPE, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, pred_df = rdl.evaluate(model, df, arrays, device)

    rows = []
    for i, row in df.iterrows():
        pred_params = pred_df.iloc[i][[f"pred_{name}" for name in rdl.TARGET_PARAMS]].to_numpy(dtype=np.float64)
        pred_s = rdl.circuit_params_to_s_np(pred_params, row[rdl.LENGTH_COLUMN], arrays[6])
        target_s = arrays[5][i]
        split = "train" if arrays[3][i] else "val"
        nmse = paper_nmse(pred_s, target_s)
        rows.append(
            {
                "device": "RDL",
                "dut_index": int(row["dut_index"]),
                "split": split,
                "paper_nmse_s11_s21_ri": nmse,
                "paper_nmse_percent": nmse * 100.0,
            }
        )
    return summarize(rows)


def compute_tsv(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(tsv.PARAM_CSV, encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
    arrays = tsv.prepare_arrays(df)
    checkpoint = torch.load(TSV_CHECKPOINT, map_location=device, weights_only=False)
    model = tsv.TsvParamNet().to(dtype=tsv.REAL_DTYPE, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, pred_df = tsv.evaluate(model, df, arrays, device)

    rows = []
    for i, row in df.iterrows():
        pred_params = pred_df.iloc[i][[f"pred_{name}" for name in tsv.TARGET_PARAMS]].to_numpy(dtype=np.float64)
        pred_s = tsv.circuit_params_to_s_np(pred_params, row["h_tsv"], arrays[6])
        target_s = arrays[5][i]
        split = "train" if arrays[3][i] else "val"
        nmse = paper_nmse(pred_s, target_s)
        rows.append(
            {
                "device": "TSV",
                "dut_index": int(row["dut_index"]),
                "split": split,
                "paper_nmse_s11_s21_ri": nmse,
                "paper_nmse_percent": nmse * 100.0,
            }
        )
    return summarize(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    rdl_detail, rdl_summary = compute_rdl(device)
    tsv_detail, tsv_summary = compute_tsv(device)

    combined = pd.concat([rdl_summary.assign(device="RDL"), tsv_summary.assign(device="TSV")], ignore_index=True)
    combined = combined[
        ["device", "split", "count", "nmse_mean", "nmse_median", "nmse_percent_mean", "nmse_percent_median"]
    ]

    rdl_detail.to_csv(OUTPUT_DIR / "rdl_connection2_paper_nmse_detail.csv", index=False, encoding="utf-8-sig")
    rdl_summary.to_csv(OUTPUT_DIR / "rdl_connection2_paper_nmse_summary.csv", index=False, encoding="utf-8-sig")
    tsv_detail.to_csv(OUTPUT_DIR / "tsv_connection2_paper_nmse_detail.csv", index=False, encoding="utf-8-sig")
    tsv_summary.to_csv(OUTPUT_DIR / "tsv_connection2_paper_nmse_summary.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT_DIR / "single_device_paper_nmse_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 Single-Device Paper-Style NMSE Recalculation",
                "",
                "- Entry: `recompute_single_device_paper_nmse.py`",
                "- Metric: per-sample NMSE on linear `Re/Im(S11,S21)` curves.",
                "- Formula: `sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)`.",
                f"- RDL checkpoint: `{RDL_CHECKPOINT}`",
                f"- TSV checkpoint: `{TSV_CHECKPOINT}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(combined),
            ]
        ),
        encoding="utf-8",
    )
    print(combined.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
