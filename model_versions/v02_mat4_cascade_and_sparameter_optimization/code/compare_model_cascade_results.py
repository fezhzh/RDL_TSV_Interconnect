# -*- coding: utf-8 -*-
"""Compare full RDL_TSV HFSS S-parameters with cascaded mat4 device models.

The script reads each full-structure ``RDL_TSV_Snp/dut*.s2p`` file, parses the
geometry variables in its header, predicts every device block with the
corresponding ``RDL_TSV_mat4`` MATLAB model, cascades the blocks, and compares
the cascaded S-parameters against the HFSS full-structure S-parameters.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rdl_tsv_transition.dataset import prepare_structure_sample
from rdl_tsv_transition.plotting import configure_comparison_matplotlib, db20, style_frequency_axis


# Configure these values before running directly from VS Code.
S2P_DIR = PROJECT_ROOT / "snp_data" / "RDL_TSV_Snp"
MAT_DIR = PROJECT_ROOT / "model_versions" / "v01_matlab_mat_models" / "models" / "RDL_TSV_mat4"
OUT_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "results" / "RDL_TSV_mat4_cascade_compare"

# Use None for all DUT files. Set to an integer, such as 10, for a quick subset.
LIMIT = None

WRITE_CSV = True
WRITE_REPORT = True
WRITE_PLOTS = True

# Show one comparison figure for every processed DUT. This can open many windows
# when LIMIT is None.
SHOW_ALL_PLOTS = True
PROGRESS_EVERY = 25

OUTPUT_PREFIX = "rdl_tsv_mat4_cascade"
MODEL_NAME = "mat4_direct_cascade"


def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).stem)
    return [int(part) if part.isdigit() else part for part in parts]


def dut_index(path):
    match = re.search(r"dut(\d+)\.s2p$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse DUT index from {path}")
    return int(match.group(1))


def s2p_files():
    files = sorted(S2P_DIR.glob("dut*.s2p"), key=natural_key)
    if LIMIT is not None:
        files = files[:LIMIT]
    if not files:
        raise FileNotFoundError(f"No dut*.s2p files found under {S2P_DIR}")
    return files


def safe_rel_error(pred, ref):
    return np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-12)


def calc_metrics(pred_s, ref_s, label_prefix):
    diff = pred_s - ref_s
    metrics = {
        f"{label_prefix}_complex_mse": float(np.mean(np.abs(diff) ** 2)),
        f"{label_prefix}_complex_rmse": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        f"{label_prefix}_complex_mae": float(np.mean(np.abs(diff))),
        f"{label_prefix}_rel_mae": float(np.mean(safe_rel_error(pred_s, ref_s))),
    }
    for m, n, name in [(0, 0, "s11"), (1, 0, "s21"), (0, 1, "s12"), (1, 1, "s22")]:
        pred = pred_s[:, m, n]
        ref = ref_s[:, m, n]
        metrics[f"{label_prefix}_{name}_db_mae"] = float(np.mean(np.abs(db20(pred) - db20(ref))))
        metrics[f"{label_prefix}_{name}_db_max"] = float(np.max(np.abs(db20(pred) - db20(ref))))
        metrics[f"{label_prefix}_{name}_phase_mae_deg"] = float(
            np.mean(np.abs(np.unwrap(np.angle(pred)) - np.unwrap(np.angle(ref)))) * 180.0 / np.pi
        )
    return metrics


def summarize_metrics(summary_df):
    metric_cols = [
        col
        for col in summary_df.columns
        if col.endswith(("_mse", "_rmse", "_mae", "_db_mae", "_db_max", "_phase_mae_deg"))
    ]
    rows = []
    for col in metric_cols:
        values = summary_df[col].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": col,
                "mean": values.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "worst_file": summary_df.loc[values.idxmax(), "file"],
            }
        )
    return pd.DataFrame(rows)


def compact_metrics(summary_df):
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "mean_complex_mse": summary_df[f"{MODEL_NAME}_vs_hfss_complex_mse"].mean(),
                "median_complex_mse": summary_df[f"{MODEL_NAME}_vs_hfss_complex_mse"].median(),
                "max_complex_mse": summary_df[f"{MODEL_NAME}_vs_hfss_complex_mse"].max(),
                "mean_s11_db_mae": summary_df[f"{MODEL_NAME}_vs_hfss_s11_db_mae"].mean(),
                "mean_s21_db_mae": summary_df[f"{MODEL_NAME}_vs_hfss_s21_db_mae"].mean(),
                "mean_s11_phase_mae_deg": summary_df[f"{MODEL_NAME}_vs_hfss_s11_phase_mae_deg"].mean(),
                "mean_s21_phase_mae_deg": summary_df[f"{MODEL_NAME}_vs_hfss_s21_phase_mae_deg"].mean(),
            }
        ]
    )


def show_model_case_plot(nw_hfss, pred_s, title):
    configure_comparison_matplotlib()
    freq_ghz = nw_hfss.f / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=16, fontweight="semibold")

    mse = np.mean(np.abs(pred_s - nw_hfss.s) ** 2)
    s21_mae = np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(nw_hfss.s[:, 1, 0])))
    fig.text(
        0.02,
        0.953,
        f"{MODEL_NAME}: MSE {mse:.3e}, S21 MAE {s21_mae:.3f} dB",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )

    ports = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]
    for ax, (m, n, name) in zip(axes.ravel(), ports):
        ax.plot(freq_ghz, db20(nw_hfss.s[:, m, n]), label="HFSS", color="#1f77b4", linewidth=1.8)
        ax.plot(freq_ghz, db20(pred_s[:, m, n]), label=MODEL_NAME, color="#7e22ce", linestyle="--", linewidth=1.8)
        style_frequency_axis(ax, f"{name} magnitude", "Magnitude (dB)")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.91, wspace=0.18, hspace=0.28)
    plt.show()
    plt.close(fig)


def compare_one(path):
    idx = dut_index(path)
    sample = prepare_structure_sample(
        idx=idx,
        s2p_dir_abs=str(S2P_DIR),
        mat_dir_abs=str(MAT_DIR),
        max_points=None,
        verbose=False,
    )
    if sample is None:
        raise FileNotFoundError(path)

    pred_s = sample.direct_full_nw.s
    hfss_s = sample.hfss_nw.s
    row = {
        "file": Path(sample.s2p_file).name,
        "dut_index": idx,
        **{name: value for name, value in sample.header_params.items()},
    }
    row.update(calc_metrics(pred_s, hfss_s, f"{MODEL_NAME}_vs_hfss"))
    return sample, row


def main():
    files = s2p_files()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for n_done, path in enumerate(files, start=1):
        try:
            sample, row = compare_one(path)
            rows.append(row)

            if WRITE_PLOTS and SHOW_ALL_PLOTS:
                show_model_case_plot(
                    sample.hfss_nw,
                    sample.direct_full_nw.s,
                    f"{Path(sample.s2p_file).name} RDL_TSV mat4 direct cascade",
                )

            if PROGRESS_EVERY and n_done % PROGRESS_EVERY == 0:
                print(f"Processed {n_done}/{len(files)}: {path.name}")

        except Exception as exc:
            rows.append({"file": path.name, "dut_index": dut_index(path), "error": str(exc)})
            print(f"[skip] {path.name}: {exc}")

    summary_df = pd.DataFrame(rows)
    valid_df = summary_df[summary_df.get("error").isna()] if "error" in summary_df else summary_df

    summary_csv = OUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
    aggregate_csv = OUT_DIR / f"{OUTPUT_PREFIX}_aggregate.csv"
    compact_csv = OUT_DIR / f"{OUTPUT_PREFIX}_compact.csv"

    if WRITE_CSV:
        summary_df.to_csv(summary_csv, index=False)
        if len(valid_df):
            summarize_metrics(valid_df).to_csv(aggregate_csv, index=False)
            compact_metrics(valid_df).to_csv(compact_csv, index=False)

    report = {
        "hfss_dir": str(S2P_DIR),
        "mat_dir": str(MAT_DIR),
        "out_dir": str(OUT_DIR),
        "n_total": int(len(summary_df)),
        "n_valid": int(len(valid_df)),
        "model": MODEL_NAME,
        "summary_csv": str(summary_csv) if WRITE_CSV else None,
        "aggregate_csv": str(aggregate_csv) if WRITE_CSV else None,
        "compact_csv": str(compact_csv) if WRITE_CSV else None,
        "plots": "shown_interactively" if WRITE_PLOTS and SHOW_ALL_PLOTS else None,
    }
    if WRITE_REPORT:
        with open(OUT_DIR / "compare_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nCompare complete")
    print(f"  HFSS dir: {S2P_DIR}")
    print(f"  mat dir: {MAT_DIR}")
    print(f"  valid cases: {len(valid_df)} / {len(summary_df)}")
    if WRITE_CSV:
        print(f"  summary CSV: {summary_csv}")
        if len(valid_df):
            print(compact_metrics(valid_df).to_string(index=False))
    if WRITE_PLOTS:
        print("  plots: shown interactively for every processed DUT")


if __name__ == "__main__":
    main()
