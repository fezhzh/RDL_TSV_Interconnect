# -*- coding: utf-8 -*-
"""Calibrate v10 ADS single-device RDL/TSV simulations on a small HFSS subset.

Run this file directly in VS Code. No command-line arguments are required.

Primary variables:
- RDL/TSV: er_si, cond, tand
- TSV only: c1_scale

Secondary variables:
- RDL: l_scale, w_scale, pitch_scale, h_tsv_scale, h_rdl_scale
- TSV: pitch_scale, h_tsv_scale, d_scale
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V10_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade"
RDL_HELPER = V10_DIR / "rdl_ads_sim" / "ADS_Sim.py"
TSV_HELPER = V10_DIR / "tsv_ads_sim" / "ADS_Sim.py"
OUTPUT_DIR = V10_DIR / "results" / "ads_single_device_calibration_small"

SOURCE_DATASET = "LHS200"
SOURCE_SPLIT = "train"
SAMPLE_COUNT = 4
SAMPLE_DUTS = [100, 101, 102, 103]
REUSE_EXISTING = True

BASE_SETTINGS = {
    "er_si": 11.9,
    "cond": 5.8e7,
    "tand": 0.005,
    "c1_scale": 1.0,
    "l_scale": 1.0,
    "w_scale": 1.0,
    "pitch_scale": 1.0,
    "h_tsv_scale": 1.0,
    "h_rdl_scale": 1.0,
    "d_scale": 1.0,
}

RDL_PRIMARY_SWEEP = {
    "er_si": [10.8, 11.9, 12.5],
    "cond": [4.1e7, 5.8e7],
    "tand": [0.0, 0.005, 0.01],
}
TSV_PRIMARY_SWEEP = {
    "er_si": [10.8, 11.9, 12.5],
    "cond": [4.1e7, 5.8e7],
    "tand": [0.0, 0.005, 0.01],
    "c1_scale": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
}
RDL_SECONDARY_SWEEP = {
    "l_scale": [0.9, 1.0, 1.1],
    "w_scale": [0.9, 1.0, 1.1],
    "pitch_scale": [0.9, 1.0, 1.1],
    "h_tsv_scale": [0.9, 1.0, 1.1],
    "h_rdl_scale": [0.9, 1.0, 1.1],
}
TSV_SECONDARY_SWEEP = {
    "pitch_scale": [0.9, 1.0, 1.1],
    "h_tsv_scale": [0.9, 1.0, 1.1],
    "d_scale": [0.9, 1.0, 1.1],
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RDL_ADS = load_module(RDL_HELPER, "v10_rdl_ads_calibration")
TSV_ADS = load_module(TSV_HELPER, "v10_tsv_ads_calibration")


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def settings_for_scope(settings: dict[str, float], scope: str) -> dict[str, float]:
    if scope == "rdl":
        keys = ["er_si", "cond", "tand", "l_scale", "w_scale", "pitch_scale", "h_tsv_scale", "h_rdl_scale"]
    elif scope == "tsv":
        keys = ["er_si", "cond", "tand", "c1_scale", "pitch_scale", "h_tsv_scale", "d_scale"]
    else:
        raise ValueError(scope)
    return {key: float(settings[key]) for key in keys}


def settings_slug(settings: dict[str, float], scope: str) -> str:
    parts = []
    for key, value in settings_for_scope(settings, scope).items():
        text = f"{value:.5g}".replace("+", "").replace("-", "m").replace(".", "p")
        parts.append(f"{key}-{text}")
    return "__".join(parts)


def add_oat_candidates(base: dict[str, float], sweep: dict[str, list[float]], stage: str):
    rows = []
    seen = set()

    def add(settings, label):
        key = tuple(sorted((k, round(float(v), 12)) for k, v in settings.items()))
        if key not in seen:
            seen.add(key)
            rows.append((label, dict(settings)))

    add(base, f"{stage}_baseline")
    for name, values in sweep.items():
        for value in values:
            candidate = dict(base)
            candidate[name] = float(value)
            add(candidate, f"{stage}_{name}_{value:g}")
    return rows


def load_variation_table(design: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "HFSS_sim" / SOURCE_DATASET / SOURCE_SPLIT / f"{design}_variations_record.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.set_index("dut_index", drop=False)


def build_samples() -> dict[str, list[dict[str, object]]]:
    split_dir = PROJECT_ROOT / "HFSS_sim" / SOURCE_DATASET / SOURCE_SPLIT
    tsv_var = load_variation_table("TSV")
    tmrdl_var = load_variation_table("TMRDL")
    bsmrdl_var = load_variation_table("BSMRDL")
    samples: dict[str, list[dict[str, object]]] = {"TMRDL": [], "BSMRDL": [], "TSV": []}
    dut_ids = SAMPLE_DUTS[:SAMPLE_COUNT]

    for dut in dut_ids:
        tsv_row = tsv_var.loc[dut]
        for device_name, var_df in [("TMRDL", tmrdl_var), ("BSMRDL", bsmrdl_var)]:
            rec = var_df.loc[dut].to_dict()
            rec["h_tsv"] = float(tsv_row["h_tsv"])
            rec["sample_id"] = f"{SOURCE_DATASET}_{SOURCE_SPLIT}_dut{dut}_{device_name}"
            rec["dut_index"] = int(dut)
            rec["hfss_path"] = split_dir / device_name / f"dut{dut}.s2p"
            samples[device_name].append(rec)
        rec = tsv_row.to_dict()
        rec["sample_id"] = f"{SOURCE_DATASET}_{SOURCE_SPLIT}_dut{dut}_TSV"
        rec["dut_index"] = int(dut)
        rec["hfss_path"] = split_dir / "TSV" / f"dut{dut}.s2p"
        samples["TSV"].append(rec)
    return samples


def sparam_y(s: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            s[:, 0, 0].real,
            s[:, 0, 0].imag,
            s[:, 1, 0].real,
            s[:, 1, 0].imag,
        ]
    )


def nmse_s11_s21_ri(true_s: np.ndarray, pred_s: np.ndarray) -> float:
    y_true = sparam_y(true_s)
    y_pred = sparam_y(pred_s)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum((y_true - y_pred) ** 2) / denom)


def mse_complex(true_s: np.ndarray, pred_s: np.ndarray) -> float:
    return float(np.mean(np.abs(true_s - pred_s) ** 2))


def simulate_one(device_name: str, sample: dict[str, object], settings: dict[str, float], scope: str) -> Path:
    cache_base = OUTPUT_DIR / "ads_cache" / scope / settings_slug(settings, scope) / str(sample["sample_id"])
    if device_name == "TSV":
        return TSV_ADS.simulate_single_device(
            device_name,
            str(sample["sample_id"]),
            sample,
            settings_for_scope(settings, "tsv"),
            output_base=cache_base,
            reuse_existing=REUSE_EXISTING,
        )
    return RDL_ADS.simulate_single_device(
        device_name,
        str(sample["sample_id"]),
        sample,
        settings_for_scope(settings, "rdl"),
        output_base=cache_base,
        reuse_existing=REUSE_EXISTING,
    )


def evaluate_candidate(scope: str, label: str, settings: dict[str, float], samples: dict[str, list[dict[str, object]]]):
    devices = ["TMRDL", "BSMRDL"] if scope == "rdl" else ["TSV"]
    rows = []
    for device_name in devices:
        for sample in samples[device_name]:
            ads_path = simulate_one(device_name, sample, settings, scope)
            ads_nw = rf.Network(str(ads_path))
            hfss_nw = rf.Network(str(sample["hfss_path"]))
            if len(ads_nw.f) != len(hfss_nw.f) or not np.allclose(ads_nw.f, hfss_nw.f):
                raise ValueError(f"Frequency mismatch: {device_name} {sample['sample_id']}")
            rows.append(
                {
                    "scope": scope,
                    "candidate": label,
                    "device": device_name,
                    "sample_id": sample["sample_id"],
                    "dut_index": int(sample["dut_index"]),
                    "mse_complex": mse_complex(hfss_nw.s, ads_nw.s),
                    "nmse_s11_s21_ri": nmse_s11_s21_ri(hfss_nw.s, ads_nw.s),
                    "ads_path": str(ads_path),
                    **settings_for_scope(settings, scope),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    setting_cols = [
        col
        for col in ["er_si", "cond", "tand", "c1_scale", "l_scale", "w_scale", "pitch_scale", "h_tsv_scale", "h_rdl_scale", "d_scale"]
        if col in df.columns
    ]
    return (
        df.groupby(["scope", "candidate"] + setting_cols, as_index=False)
        .agg(
            count=("sample_id", "count"),
            mse_mean=("mse_complex", "mean"),
            mse_median=("mse_complex", "median"),
            nmse_mean=("nmse_s11_s21_ri", "mean"),
            nmse_median=("nmse_s11_s21_ri", "median"),
            nmse_max=("nmse_s11_s21_ri", "max"),
        )
        .sort_values(["scope", "nmse_mean", "mse_mean"], ignore_index=True)
    )


def run_scope(scope: str, samples: dict[str, list[dict[str, object]]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    primary_sweep = RDL_PRIMARY_SWEEP if scope == "rdl" else TSV_PRIMARY_SWEEP
    secondary_sweep = RDL_SECONDARY_SWEEP if scope == "rdl" else TSV_SECONDARY_SWEEP
    rows = []

    primary_candidates = add_oat_candidates(BASE_SETTINGS, primary_sweep, "primary")
    for i, (label, settings) in enumerate(primary_candidates, start=1):
        print(f"[{scope}] primary {i}/{len(primary_candidates)} {label}", flush=True)
        rows.extend(evaluate_candidate(scope, label, settings, samples))
    primary_summary = summarize(rows)
    best_primary = primary_summary.iloc[0]
    best_settings = dict(BASE_SETTINGS)
    for key in settings_for_scope(BASE_SETTINGS, scope):
        best_settings[key] = float(best_primary[key])

    secondary_candidates = add_oat_candidates(best_settings, secondary_sweep, "secondary")
    for i, (label, settings) in enumerate(secondary_candidates, start=1):
        print(f"[{scope}] secondary {i}/{len(secondary_candidates)} {label}", flush=True)
        rows.extend(evaluate_candidate(scope, label, settings, samples))

    detail = pd.DataFrame(rows)
    summary = summarize(rows)
    best = summary.iloc[0]
    best_settings = dict(BASE_SETTINGS)
    for key in settings_for_scope(BASE_SETTINGS, scope):
        best_settings[key] = float(best[key])
    return detail, summary, best_settings


def plot_best(scope: str, best_settings: dict[str, float], samples: dict[str, list[dict[str, object]]]) -> list[str]:
    plot_dir = OUTPUT_DIR / "plots" / scope
    plot_dir.mkdir(parents=True, exist_ok=True)
    devices = ["TMRDL", "BSMRDL"] if scope == "rdl" else ["TSV"]
    saved = []
    for device_name in devices:
        sample = samples[device_name][0]
        ads_path = simulate_one(device_name, sample, best_settings, scope)
        ads_nw = rf.Network(str(ads_path))
        hfss_nw = rf.Network(str(sample["hfss_path"]))
        freq_ghz = hfss_nw.f / 1e9
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
        fig.suptitle(f"{scope} best | {device_name} | {sample['sample_id']}", x=0.02, y=0.985, ha="left")
        for ax, (m, n, label, part) in zip(
            axes.ravel(),
            [
                (0, 0, "S11", "real"),
                (0, 0, "S11", "imag"),
                (1, 0, "S21", "real"),
                (1, 0, "S21", "imag"),
            ],
        ):
            hfss = hfss_nw.s[:, m, n].real if part == "real" else hfss_nw.s[:, m, n].imag
            ads = ads_nw.s[:, m, n].real if part == "real" else ads_nw.s[:, m, n].imag
            ax.plot(freq_ghz, hfss, label="HFSS", color="black", linewidth=2.0)
            ax.plot(freq_ghz, ads, label="ADS", color="#2563eb", linestyle="--")
            ax.set_title(f"{label} {part}")
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.9, wspace=0.22, hspace=0.35)
        out_path = plot_dir / f"{device_name}_{sample['dut_index']}_best_compare.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(str(out_path))
    return saved


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: f"{value:.6g}")
    columns = [str(col) for col in display_df.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display_df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in display_df.columns) + " |")
    return "\n".join(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    rdl_detail, rdl_summary, best_rdl = run_scope("rdl", samples)
    tsv_detail, tsv_summary, best_tsv = run_scope("tsv", samples)

    detail = pd.concat([rdl_detail, tsv_detail], ignore_index=True)
    summary = pd.concat([rdl_summary, tsv_summary], ignore_index=True)
    detail.to_csv(OUTPUT_DIR / "ads_single_device_calibration_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_single_device_calibration_summary.csv", index=False, encoding="utf-8-sig")

    best_plots = plot_best("rdl", best_rdl, samples) + plot_best("tsv", best_tsv, samples)
    best = {
        "source_dataset": SOURCE_DATASET,
        "source_split": SOURCE_SPLIT,
        "sample_duts": SAMPLE_DUTS[:SAMPLE_COUNT],
        "metric": "NMSE over flattened S11/S21 real and imaginary components",
        "best_rdl_settings": settings_for_scope(best_rdl, "rdl"),
        "best_tsv_settings": settings_for_scope(best_tsv, "tsv"),
        "plots": best_plots,
    }
    (OUTPUT_DIR / "best_ads_calibration_settings.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rdl_top = rdl_summary.head(8)
    tsv_top = tsv_summary.head(8)
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# ADS Single-Device Calibration Validation",
                "",
                f"- Source: `{SOURCE_DATASET}/{SOURCE_SPLIT}`",
                f"- DUTs: `{SAMPLE_DUTS[:SAMPLE_COUNT]}`",
                "- Metric: NMSE over flattened `S11.real`, `S11.imag`, `S21.real`, and `S21.imag`.",
                "- Primary variables: `er_si`, `cond`, `tand`, and TSV `c1_scale`.",
                "- Secondary variables: RDL `l_scale/w_scale/pitch_scale/h_tsv_scale/h_rdl_scale`; TSV `pitch_scale/h_tsv_scale/d_scale`.",
                f"- Detail CSV: `{OUTPUT_DIR / 'ads_single_device_calibration_detail.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'ads_single_device_calibration_summary.csv'}`",
                f"- Best settings JSON: `{OUTPUT_DIR / 'best_ads_calibration_settings.json'}`",
                "",
                "## Best RDL Settings",
                "",
                "```json",
                json.dumps(settings_for_scope(best_rdl, "rdl"), indent=2, ensure_ascii=False),
                "```",
                "",
                "## Best TSV Settings",
                "",
                "```json",
                json.dumps(settings_for_scope(best_tsv, "tsv"), indent=2, ensure_ascii=False),
                "```",
                "",
                "## Top RDL Candidates",
                "",
                dataframe_to_markdown(rdl_top),
                "",
                "## Top TSV Candidates",
                "",
                dataframe_to_markdown(tsv_top),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Best RDL settings:", json.dumps(settings_for_scope(best_rdl, "rdl"), ensure_ascii=False), flush=True)
    print("Best TSV settings:", json.dumps(settings_for_scope(best_tsv, "tsv"), ensure_ascii=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
