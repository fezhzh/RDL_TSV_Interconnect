# -*- coding: utf-8 -*-
"""Plot ADS-vs-HFSS single-device comparisons for LHS400_Connection2.

Run this file directly in VS Code. No command-line arguments are required.
It selects 50 RDL samples and 50 TSV samples from
HFSS_sim/LHS400_Connection2/train, runs or reuses ADS single-device outputs,
and archives S-parameter comparison plots.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v11_base.py"
PROJECT_ROOT = THIS_DIR.parents[2]
DATASET_ROOT = PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v11_ads_v08_multihead_chain"
    / "results"
    / "ads_vs_hfss_lhs400_connection2_single_device_50"
)
ADS_CACHE_DIR = OUTPUT_DIR / "ads_single_device_cache"
SAMPLE_COUNT_PER_DEVICE = 50

RDL_SETTINGS = {
    "er_si": 9.8,
    "cond": 5.8e7,
    "tand": 0.005,
    "l_scale": 1.0,
    "w_scale": 0.65,
    "pitch_scale": 1.25,
    "h_tsv_scale": 1.0,
    "h_rdl_scale": 1.0,
    "freq_start_ghz": 0.1,
    "freq_stop_ghz": 100.0,
    "freq_step_ghz": 0.1,
}
TSV_SETTINGS = {
    "er_si": 11.9,
    "cond": 5.8e7,
    "tand": 0.005,
    "c1_scale": 1.0,
    "pitch_scale": 1.0,
    "h_tsv_scale": 1.2,
    "d_scale": 1.0,
    "freq_start_ghz": 0.1,
    "freq_stop_ghz": 100.0,
    "freq_step_ghz": 0.1,
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def s11_s21_real_imag_y(s_params: np.ndarray) -> np.ndarray:
    s11 = s_params[:, 0, 0]
    s21 = s_params[:, 1, 0]
    return np.column_stack([s11.real, s11.imag, s21.real, s21.imag]).ravel()


def nmse_s11_s21_real_imag(target: np.ndarray, pred: np.ndarray) -> float:
    y_true = s11_s21_real_imag_y(target)
    y_pred = s11_s21_real_imag_y(pred)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(numerator / max(denominator, 1e-30))


def mag_phase_mse(target: np.ndarray, pred: np.ndarray) -> float:
    target_pair = np.stack([target[:, 0, 0], target[:, 1, 0]], axis=-1)
    pred_pair = np.stack([pred[:, 0, 0], pred[:, 1, 0]], axis=-1)
    mag_loss = np.mean((np.abs(pred_pair) - np.abs(target_pair)) ** 2)
    phase_delta = np.angle(pred_pair) - np.angle(target_pair)
    phase_delta = np.arctan2(np.sin(phase_delta), np.cos(phase_delta))
    return float(mag_loss + np.mean(phase_delta**2))


def metrics_for(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mse_all_s": float(np.mean(np.abs(pred - target) ** 2)),
        "nmse_s11_s21_ri": nmse_s11_s21_real_imag(target, pred),
        "mag_phase_mse_s11_s21": mag_phase_mse(target, pred),
        "s11_db_mae": float(np.mean(np.abs(db20(pred[:, 0, 0]) - db20(target[:, 0, 0])))),
        "s21_db_mae": float(np.mean(np.abs(db20(pred[:, 1, 0]) - db20(target[:, 1, 0])))),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_samples(kind: str) -> pd.DataFrame:
    csv_path = DATASET_ROOT / f"{kind}_variations_record.csv"
    snp_dir = DATASET_ROOT / kind
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not snp_dir.exists():
        raise FileNotFoundError(f"Missing S-parameter directory: {snp_dir}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig").sort_values("dut_index").head(SAMPLE_COUNT_PER_DEVICE).copy()
    df["kind"] = kind
    df["file"] = df["dut_index"].map(lambda idx: f"dut{int(idx)}.s2p")
    df["hfss_snp_path"] = df["file"].map(lambda name: snp_dir / name)
    missing = [str(path) for path in df["hfss_snp_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing HFSS {kind} S-parameter files: {missing[:5]}")
    return df.reset_index(drop=True)


def normalize_row(kind: str, row: pd.Series) -> dict[str, float]:
    rec = row.to_dict()
    if kind == "RDL":
        rec["h_tmrdl"] = float(rec.get("h_tmrdl", rec.get("t_tmrdl")))
        rec["h_tsv"] = float(rec.get("h_tsv", 100.0))
    return rec


def simulate_ads(base, kind: str, row: pd.Series) -> Path:
    sample_id = f"LHS400_Connection2_{kind}_dut{int(row['dut_index'])}"
    rec = normalize_row(kind, row)
    if kind == "RDL":
        output_base = ADS_CACHE_DIR / "RDL" / sample_id
        return base.RDL_ADS.simulate_single_device(
            device_name="TMRDL",
            sample_id=sample_id,
            structure=rec,
            ads_settings=RDL_SETTINGS,
            output_base=output_base,
            reuse_existing=True,
        )
    output_base = ADS_CACHE_DIR / "TSV" / sample_id
    return base.TSV_ADS.simulate_single_device(
        device_name="TSV",
        sample_id=sample_id,
        structure=rec,
        ads_settings=TSV_SETTINGS,
        output_base=output_base,
        reuse_existing=True,
    )


def plot_one(plt, freq_ghz: np.ndarray, hfss_s: np.ndarray, ads_s: np.ndarray, row: dict[str, object], out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{row['kind']} dut{row['dut_index']} | NMSE={row['nmse_s11_s21_ri']:.3e} | "
        f"S11 MAE={row['s11_db_mae']:.3f} dB | S21 MAE={row['s21_db_mae']:.3f} dB",
        x=0.02,
        y=0.985,
        ha="left",
    )
    specs = [
        (0, 0, "S11 real", np.real),
        (0, 0, "S11 imag", np.imag),
        (1, 0, "S21 real", np.real),
        (1, 0, "S21 imag", np.imag),
    ]
    for ax, (m, n, title, component) in zip(axes.ravel(), specs):
        ax.plot(freq_ghz, component(hfss_s[:, m, n]), label="HFSS", color="black", linewidth=1.8)
        ax.plot(freq_ghz, component(ads_s[:, m, n]), label="ADS", color="#dc2626", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)


def save_summary_plot(plt, metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
    metrics.boxplot(column="nmse_s11_s21_ri", by="kind", ax=axes[0])
    axes[0].set_title("NMSE")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("NMSE")
    metrics.boxplot(column="s11_db_mae", by="kind", ax=axes[1])
    axes[1].set_title("S11 dB MAE")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("dB")
    metrics.boxplot(column="s21_db_mae", by="kind", ax=axes[2])
    axes[2].set_title("S21 dB MAE")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("dB")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ads_vs_hfss_single_device_metric_summary.png")
    plt.close(fig)


def main():
    base = load_module(BASE_SCRIPT, "v11_base_for_lhs400_single_device_ads_hfss_plot")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sample_frames = [load_samples("RDL"), load_samples("TSV")]
    all_rows = []
    for df in sample_frames:
        kind = str(df["kind"].iloc[0])
        kind_plot_dir = plot_dir / kind
        kind_plot_dir.mkdir(parents=True, exist_ok=True)
        for i, row in df.iterrows():
            hfss_nw = base.rf.Network(str(row["hfss_snp_path"]))
            ads_path = simulate_ads(base, kind, row)
            ads_nw = base.rf.Network(str(ads_path))
            if len(hfss_nw.f) != len(ads_nw.f) or not np.allclose(hfss_nw.f, ads_nw.f):
                raise ValueError(f"Frequency grid mismatch: {kind} dut{int(row['dut_index'])}")
            row_metrics = metrics_for(hfss_nw.s, ads_nw.s)
            out_row = {
                "kind": kind,
                "dut_index": int(row["dut_index"]),
                "sample_id": f"LHS400_Connection2_{kind}_dut{int(row['dut_index'])}",
                "hfss_snp_path": str(row["hfss_snp_path"]),
                "ads_snp_path": str(ads_path),
                **row_metrics,
            }
            all_rows.append(out_row)
            plot_one(
                base.plt,
                hfss_nw.f / 1e9,
                hfss_nw.s,
                ads_nw.s,
                out_row,
                kind_plot_dir / f"{out_row['sample_id']}_ads_vs_hfss.png",
            )
            print(f"[plot] {kind} {i + 1}/{len(df)} dut{int(row['dut_index'])}", flush=True)

    metrics = pd.DataFrame(all_rows)
    summary = (
        metrics.groupby("kind", as_index=False)
        .agg(
            count=("sample_id", "count"),
            nmse_mean=("nmse_s11_s21_ri", "mean"),
            nmse_median=("nmse_s11_s21_ri", "median"),
            s11_db_mae_mean=("s11_db_mae", "mean"),
            s21_db_mae_mean=("s21_db_mae", "mean"),
            mag_phase_mse_mean=("mag_phase_mse_s11_s21", "mean"),
        )
        .sort_values("kind")
    )
    save_summary_plot(base.plt, metrics)
    metrics.to_csv(OUTPUT_DIR / "ads_vs_hfss_single_device_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_vs_hfss_single_device_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "dataset_root": str(DATASET_ROOT),
        "sample_count_per_device": SAMPLE_COUNT_PER_DEVICE,
        "sample_selection": "first 50 rows after sorting by dut_index",
        "rdl_device_name_for_ads": "TMRDL",
        "rdl_settings": RDL_SETTINGS,
        "tsv_settings": TSV_SETTINGS,
        "output_dir": str(OUTPUT_DIR),
        "summary": summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "ads_vs_hfss_single_device_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "ads_vs_hfss_single_device_report.md").write_text(
        "\n".join(
            [
                "# ADS vs HFSS Single-Device Comparison",
                "",
                f"- Dataset: `{DATASET_ROOT}`",
                f"- Samples: first `{SAMPLE_COUNT_PER_DEVICE}` RDL and first `{SAMPLE_COUNT_PER_DEVICE}` TSV samples sorted by `dut_index`.",
                "- RDL ADS device: `TMRDL`; `t_tmrdl` is mapped to `h_tmrdl`.",
                "- TSV ADS device: `TSV`.",
                f"- Output: `{OUTPUT_DIR}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Per-sample metrics: `{OUTPUT_DIR / 'ads_vs_hfss_single_device_metrics.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'ads_vs_hfss_single_device_summary.csv'}`",
                f"- Summary plot: `{OUTPUT_DIR / 'ads_vs_hfss_single_device_metric_summary.png'}`",
                f"- RDL plots: `{plot_dir / 'RDL'}`",
                f"- TSV plots: `{plot_dir / 'TSV'}`",
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)


if __name__ == "__main__":
    main()
