# -*- coding: utf-8 -*-
"""Calibrate v11 ADS RDL/TSV single-device simulations on random LHS400 samples.

Run this file directly in VS Code. No command-line arguments are required.
It randomly selects 30 RDL and 30 TSV samples from LHS400_Connection2/train,
evaluates one-variable-at-a-time ADS setting candidates, and archives the best
settings and ADS/HFSS comparison plots.
"""

from __future__ import annotations

import importlib.util
import hashlib
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
    / "ads_single_device_calibration_lhs400_connection2_random30"
)
ADS_CACHE_DIR = OUTPUT_DIR / "ads_cache"
SAMPLE_COUNT_PER_DEVICE = 30
RANDOM_SEED = 20260709

BASE_RDL_SETTINGS = {
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
BASE_TSV_SETTINGS = {
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

RDL_OAT_SWEEP = {
    "er_si": [9.0, 9.8, 10.5],
    "w_scale": [0.55, 0.65, 0.75],
    "pitch_scale": [1.10, 1.25, 1.40],
    "h_rdl_scale": [0.85, 1.00, 1.15],
}
TSV_OAT_SWEEP = {
    "er_si": [10.8, 11.9, 12.5],
    "c1_scale": [0.75, 1.00, 1.25],
    "h_tsv_scale": [1.00, 1.20, 1.35],
    "d_scale": [0.90, 1.00, 1.10],
    "pitch_scale": [0.90, 1.00, 1.10],
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def settings_slug(settings: dict[str, float]) -> str:
    payload = json.dumps(settings, sort_keys=True, ensure_ascii=True)
    return "s" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def add_oat_candidates(base: dict[str, float], sweep: dict[str, list[float]]) -> list[tuple[str, dict[str, float]]]:
    rows = []
    seen = set()

    def add(label: str, settings: dict[str, float]) -> None:
        key = tuple(sorted((name, round(float(value), 12)) for name, value in settings.items()))
        if key not in seen:
            seen.add(key)
            rows.append((label, dict(settings)))

    add("baseline", base)
    for name, values in sweep.items():
        for value in values:
            candidate = dict(base)
            candidate[name] = float(value)
            add(f"{name}_{value:g}", candidate)
    return rows


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def sparam_y(s_params: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            s_params[:, 0, 0].real,
            s_params[:, 0, 0].imag,
            s_params[:, 1, 0].real,
            s_params[:, 1, 0].imag,
        ]
    )


def nmse_s11_s21_ri(true_s: np.ndarray, pred_s: np.ndarray) -> float:
    y_true = sparam_y(true_s)
    y_pred = sparam_y(pred_s)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(np.sum((y_true - y_pred) ** 2) / max(denom, 1e-30))


def mag_phase_mse(true_s: np.ndarray, pred_s: np.ndarray) -> float:
    true_pair = np.stack([true_s[:, 0, 0], true_s[:, 1, 0]], axis=-1)
    pred_pair = np.stack([pred_s[:, 0, 0], pred_s[:, 1, 0]], axis=-1)
    mag_loss = np.mean((np.abs(pred_pair) - np.abs(true_pair)) ** 2)
    phase_delta = np.angle(pred_pair) - np.angle(true_pair)
    phase_delta = np.arctan2(np.sin(phase_delta), np.cos(phase_delta))
    return float(mag_loss + np.mean(phase_delta**2))


def metrics_for(true_s: np.ndarray, pred_s: np.ndarray) -> dict[str, float]:
    return {
        "mse_all_s": float(np.mean(np.abs(pred_s - true_s) ** 2)),
        "nmse_s11_s21_ri": nmse_s11_s21_ri(true_s, pred_s),
        "mag_phase_mse_s11_s21": mag_phase_mse(true_s, pred_s),
        "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(true_s[:, 0, 0])))),
        "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(true_s[:, 1, 0])))),
    }


def load_random_samples(kind: str) -> pd.DataFrame:
    csv_path = DATASET_ROOT / f"{kind}_variations_record.csv"
    snp_dir = DATASET_ROOT / kind
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if len(df) < SAMPLE_COUNT_PER_DEVICE:
        raise ValueError(f"{kind} has {len(df)} rows, need {SAMPLE_COUNT_PER_DEVICE}")
    df = df.sample(n=SAMPLE_COUNT_PER_DEVICE, random_state=RANDOM_SEED).sort_values("dut_index").reset_index(drop=True)
    df["kind"] = kind
    df["file"] = df["dut_index"].map(lambda value: f"dut{int(value)}.s2p")
    df["hfss_snp_path"] = df["file"].map(lambda name: snp_dir / name)
    missing = [str(path) for path in df["hfss_snp_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing HFSS {kind} files: {missing[:5]}")
    return df


def normalize_row(kind: str, row: pd.Series) -> dict[str, float]:
    rec = row.to_dict()
    if kind == "RDL":
        rec["h_tmrdl"] = float(rec.get("h_tmrdl", rec.get("t_tmrdl")))
        rec["h_tsv"] = float(rec.get("h_tsv", 100.0))
    return rec


def simulate_ads(base, kind: str, row: pd.Series, settings: dict[str, float]) -> Path:
    sample_id = f"LHS400_Connection2_{kind}_dut{int(row['dut_index'])}"
    rec = normalize_row(kind, row)
    cache_base = ADS_CACHE_DIR / kind / settings_slug(settings) / sample_id
    cached_s2p = cache_base.with_suffix(".s2p")
    if cached_s2p.exists() and cached_s2p.stat().st_size > 0:
        return cached_s2p
    if kind == "RDL":
        return base.RDL_ADS.simulate_single_device(
            device_name="TMRDL",
            sample_id=sample_id,
            structure=rec,
            ads_settings=settings,
            output_base=cache_base,
            reuse_existing=True,
        )
    return base.TSV_ADS.simulate_single_device(
        device_name="TSV",
        sample_id=sample_id,
        structure=rec,
        ads_settings=settings,
        output_base=cache_base,
        reuse_existing=True,
    )


def evaluate_candidate(base, kind: str, label: str, settings: dict[str, float], samples: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for _, sample in samples.iterrows():
        hfss = base.rf.Network(str(sample["hfss_snp_path"]))
        ads_path = simulate_ads(base, kind, sample, settings)
        ads = base.rf.Network(str(ads_path))
        if len(hfss.f) != len(ads.f) or not np.allclose(hfss.f, ads.f):
            raise ValueError(f"Frequency mismatch: {kind} dut{int(sample['dut_index'])}")
        metric = metrics_for(hfss.s, ads.s)
        rows.append(
            {
                "kind": kind,
                "candidate": label,
                "dut_index": int(sample["dut_index"]),
                "sample_id": f"LHS400_Connection2_{kind}_dut{int(sample['dut_index'])}",
                "hfss_snp_path": str(sample["hfss_snp_path"]),
                "ads_snp_path": str(ads_path),
                **settings,
                **metric,
            }
        )
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    setting_cols = [
        col
        for col in [
            "er_si",
            "cond",
            "tand",
            "l_scale",
            "w_scale",
            "pitch_scale",
            "h_tsv_scale",
            "h_rdl_scale",
            "c1_scale",
            "d_scale",
        ]
        if col in detail.columns
    ]
    return (
        detail.groupby(["kind", "candidate"] + setting_cols, as_index=False, dropna=False)
        .agg(
            count=("sample_id", "count"),
            nmse_mean=("nmse_s11_s21_ri", "mean"),
            nmse_median=("nmse_s11_s21_ri", "median"),
            nmse_max=("nmse_s11_s21_ri", "max"),
            mse_mean=("mse_all_s", "mean"),
            mag_phase_mse_mean=("mag_phase_mse_s11_s21", "mean"),
            s11_db_mae_mean=("s11_db_mae", "mean"),
            s21_db_mae_mean=("s21_db_mae", "mean"),
        )
        .sort_values(["kind", "nmse_mean", "mse_mean"], ignore_index=True)
    )


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


def plot_comparison(base, kind: str, sample: pd.Series, settings: dict[str, float], metric_row: dict[str, object], out_path: Path):
    hfss = base.rf.Network(str(sample["hfss_snp_path"]))
    ads = base.rf.Network(str(simulate_ads(base, kind, sample, settings)))
    freq_ghz = hfss.f / 1e9
    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{kind} dut{int(sample['dut_index'])} | best NMSE={metric_row['nmse_s11_s21_ri']:.3e}",
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
        ax.plot(freq_ghz, component(hfss.s[:, m, n]), label="HFSS", color="black", linewidth=1.8)
        ax.plot(freq_ghz, component(ads.s[:, m, n]), label="ADS calibrated", color="#2563eb", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    base.plt.close(fig)


def save_metric_plot(base, detail_best: pd.DataFrame) -> None:
    fig, axes = base.plt.subplots(1, 3, figsize=(14, 4), dpi=150)
    detail_best.boxplot(column="nmse_s11_s21_ri", by="kind", ax=axes[0])
    axes[0].set_title("Best NMSE")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("NMSE")
    detail_best.boxplot(column="s11_db_mae", by="kind", ax=axes[1])
    axes[1].set_title("Best S11 dB MAE")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("dB")
    detail_best.boxplot(column="s21_db_mae", by="kind", ax=axes[2])
    axes[2].set_title("Best S21 dB MAE")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("dB")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ads_calibration_random30_metric_summary.png")
    base.plt.close(fig)


def main():
    base = load_module(BASE_SCRIPT, "v11_base_for_lhs400_random30_calibration")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    samples = {"RDL": load_random_samples("RDL"), "TSV": load_random_samples("TSV")}
    candidates = {
        "RDL": add_oat_candidates(BASE_RDL_SETTINGS, RDL_OAT_SWEEP),
        "TSV": add_oat_candidates(BASE_TSV_SETTINGS, TSV_OAT_SWEEP),
    }

    all_rows = []
    for kind in ["RDL", "TSV"]:
        for idx, (label, settings) in enumerate(candidates[kind], start=1):
            print(f"[calibrate] {kind} {idx}/{len(candidates[kind])} {label}", flush=True)
            all_rows.extend(evaluate_candidate(base, kind, label, settings, samples[kind]))

    detail = pd.DataFrame(all_rows)
    summary = summarize(detail)
    best_summary = summary.groupby("kind", as_index=False).first()
    detail.to_csv(OUTPUT_DIR / "ads_calibration_random30_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_calibration_random30_summary.csv", index=False, encoding="utf-8-sig")

    best_settings = {}
    best_detail_frames = []
    plot_paths = []
    plot_dir = OUTPUT_DIR / "best_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for _, best in best_summary.iterrows():
        kind = str(best["kind"])
        label = str(best["candidate"])
        base_settings = BASE_RDL_SETTINGS if kind == "RDL" else BASE_TSV_SETTINGS
        best_settings[kind] = dict(base_settings)
        for key in base_settings:
            if key in best and not pd.isna(best[key]):
                best_settings[kind][key] = float(best[key])
        best_detail = detail[detail["kind"].eq(kind) & detail["candidate"].eq(label)].copy()
        best_detail_frames.append(best_detail)
        kind_plot_dir = plot_dir / kind
        kind_plot_dir.mkdir(parents=True, exist_ok=True)
        metric_by_dut = best_detail.set_index("dut_index")
        for _, sample in samples[kind].iterrows():
            dut = int(sample["dut_index"])
            out_path = kind_plot_dir / f"LHS400_Connection2_{kind}_dut{dut}_best_ads_vs_hfss.png"
            plot_comparison(base, kind, sample, best_settings[kind], metric_by_dut.loc[dut].to_dict(), out_path)
            plot_paths.append(str(out_path))

    best_detail_all = pd.concat(best_detail_frames, ignore_index=True)
    best_detail_all.to_csv(OUTPUT_DIR / "ads_calibration_random30_best_per_sample.csv", index=False, encoding="utf-8-sig")
    save_metric_plot(base, best_detail_all)

    selected_duts = {kind: samples[kind]["dut_index"].astype(int).tolist() for kind in ["RDL", "TSV"]}
    report = {
        "dataset_root": str(DATASET_ROOT),
        "sample_count_per_device": SAMPLE_COUNT_PER_DEVICE,
        "random_seed": RANDOM_SEED,
        "selected_duts": selected_duts,
        "calibration_method": "one-variable-at-a-time candidate scan around current v11 ac_l400_ref2 settings",
        "best_settings": best_settings,
        "best_summary": best_summary.to_dict(orient="records"),
        "plot_count": len(plot_paths),
    }
    (OUTPUT_DIR / "ads_calibration_random30_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "ads_calibration_random30_report.md").write_text(
        "\n".join(
            [
                "# ADS Single-Device Calibration on Random LHS400_Connection2 Samples",
                "",
                f"- Dataset: `{DATASET_ROOT}`",
                f"- Random seed: `{RANDOM_SEED}`",
                f"- Samples: `{SAMPLE_COUNT_PER_DEVICE}` RDL and `{SAMPLE_COUNT_PER_DEVICE}` TSV samples.",
                "- Method: one-variable-at-a-time candidate scan around current v11 `ac_l400_ref2` ADS settings.",
                f"- Output: `{OUTPUT_DIR}`",
                "",
                "## Best Summary",
                "",
                dataframe_to_markdown(best_summary),
                "",
                "## Best Settings",
                "",
                "```json",
                json.dumps(best_settings, indent=2, ensure_ascii=False),
                "```",
                "",
                "## Selected DUTs",
                "",
                "```json",
                json.dumps(selected_duts, indent=2, ensure_ascii=False),
                "```",
                "",
                "## Outputs",
                "",
                f"- Detail CSV: `{OUTPUT_DIR / 'ads_calibration_random30_detail.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'ads_calibration_random30_summary.csv'}`",
                f"- Best per-sample CSV: `{OUTPUT_DIR / 'ads_calibration_random30_best_per_sample.csv'}`",
                f"- Metric plot: `{OUTPUT_DIR / 'ads_calibration_random30_metric_summary.png'}`",
                f"- Best ADS/HFSS plots: `{plot_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(best_summary), flush=True)


if __name__ == "__main__":
    main()
