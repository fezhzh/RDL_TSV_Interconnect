# -*- coding: utf-8 -*-
"""Recompute v12 errors using the Ye 2026 paper NMSE convention.

Run this file directly in VS Code. No command-line arguments are required.

The paper metric is:
NMSE = sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
where y is the flattened linear Re/Im curves of S11 and S21.
The v12 training pipeline already stores this per-sample value as
`v08_nn_nmse_s11_s21_ri`; this script archives it in paper-style decimal and
percent form for direct comparison with the paper tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
RUN_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "hfss_v08_symmetric_multihead_lhs150_50_connection2_new_rdl_tsv"
)
METRICS_CSV = RUN_DIR / "v08_sparam_metrics.csv"
OUTPUT_DIR = RUN_DIR / "paper_nmse_recalculation"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing v12 metrics CSV: {METRICS_CSV}")

    metrics = pd.read_csv(METRICS_CSV, encoding="utf-8-sig")
    required = ["sample_id", "split", "direct_nmse_s11_s21_ri", "v08_nn_nmse_s11_s21_ri"]
    missing = [name for name in required if name not in metrics.columns]
    if missing:
        raise ValueError(f"Missing required columns in {METRICS_CSV}: {missing}")

    paper_metrics = metrics[required].copy()
    paper_metrics = paper_metrics.rename(
        columns={
            "direct_nmse_s11_s21_ri": "direct_paper_nmse_decimal",
            "v08_nn_nmse_s11_s21_ri": "v08_nn_paper_nmse_decimal",
        }
    )
    paper_metrics["direct_paper_nmse_percent"] = paper_metrics["direct_paper_nmse_decimal"] * 100.0
    paper_metrics["v08_nn_paper_nmse_percent"] = paper_metrics["v08_nn_paper_nmse_decimal"] * 100.0
    paper_metrics["improvement_percent_points"] = (
        paper_metrics["direct_paper_nmse_percent"] - paper_metrics["v08_nn_paper_nmse_percent"]
    )
    paper_metrics["improvement_ratio_percent"] = (
        (paper_metrics["direct_paper_nmse_decimal"] - paper_metrics["v08_nn_paper_nmse_decimal"])
        / paper_metrics["direct_paper_nmse_decimal"].clip(lower=1e-30)
        * 100.0
    )

    summary = (
        paper_metrics.groupby("split", as_index=False)
        .agg(
            count=("sample_id", "count"),
            direct_nmse_mean_decimal=("direct_paper_nmse_decimal", "mean"),
            direct_nmse_median_decimal=("direct_paper_nmse_decimal", "median"),
            v08_nn_nmse_mean_decimal=("v08_nn_paper_nmse_decimal", "mean"),
            v08_nn_nmse_median_decimal=("v08_nn_paper_nmse_decimal", "median"),
            direct_nmse_mean_percent=("direct_paper_nmse_percent", "mean"),
            direct_nmse_median_percent=("direct_paper_nmse_percent", "median"),
            v08_nn_nmse_mean_percent=("v08_nn_paper_nmse_percent", "mean"),
            v08_nn_nmse_median_percent=("v08_nn_paper_nmse_percent", "median"),
            improvement_ratio_percent_mean=("improvement_ratio_percent", "mean"),
        )
        .sort_values("split")
    )

    paper_metrics.to_csv(OUTPUT_DIR / "paper_nmse_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "paper_nmse_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "entry": Path(__file__).name,
        "source_metrics_csv": str(METRICS_CSV),
        "output_dir": str(OUTPUT_DIR),
        "formula": "NMSE = sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)",
        "y_definition": "flattened linear [Re(S11), Im(S11), Re(S21), Im(S21)] curves",
        "note": "Percent values multiply the decimal NMSE by 100, matching the paper table style.",
        "summary": summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "paper_nmse_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 Paper-Style NMSE Recalculation",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Source metrics: `{METRICS_CSV}`",
                f"- Output: `{OUTPUT_DIR}`",
                "- Formula: `NMSE = sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)`",
                "- y definition: flattened linear `[Re(S11), Im(S11), Re(S21), Im(S21)]` curves.",
                "- Percent style: decimal NMSE multiplied by `100`, matching the paper tables.",
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
