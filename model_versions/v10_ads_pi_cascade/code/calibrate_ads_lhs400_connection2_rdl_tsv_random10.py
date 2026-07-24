# -*- coding: utf-8 -*-
"""Calibrate ADS RDL and TSV settings on random LHS400_Connection2 samples.

Run this file directly in VS Code. No command-line arguments are required.

The script uses:
- 10 random RDL samples from ``HFSS_sim/LHS400_Connection2/train/RDL``
- 10 random TSV samples from ``HFSS_sim/LHS400_Connection2/train/TSV``

RDL uses the current ADS MLIN template. TSV uses the current ADS d_tsv template.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
V10_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade"
RDL_HELPER = V10_DIR / "rdl_ads_sim" / "ADS_Sim.py"
TSV_HELPER = V10_DIR / "tsv_ads_sim" / "ADS_Sim.py"
DATA_ROOT = PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train"
OUTPUT_DIR = V10_DIR / "results" / "ac_l400_rdl_tsv10"

RDL_RANDOM_SEED = 20260708
TSV_RANDOM_SEED = 20260709
SAMPLE_COUNT = 10
FIXED_RDL_H_TSV_UM = 100.0
REUSE_EXISTING = True
FREQ_SETTINGS = {"freq_start_ghz": 0.1, "freq_stop_ghz": 100.0, "freq_step_ghz": 0.1}

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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RDL_ADS = load_module(RDL_HELPER, "v10_lhs400_rdl_ads")
TSV_ADS = load_module(TSV_HELPER, "v10_lhs400_tsv_ads")


def settings_for_scope(settings: dict[str, float], scope: str) -> dict[str, float]:
    if scope == "rdl":
        keys = ["er_si", "cond", "tand", "l_scale", "w_scale", "pitch_scale", "h_tsv_scale", "h_rdl_scale"]
    elif scope == "tsv":
        keys = ["er_si", "cond", "tand", "c1_scale", "pitch_scale", "h_tsv_scale", "d_scale"]
    else:
        raise ValueError(scope)
    scoped = {key: float(settings[key]) for key in keys}
    scoped.update(FREQ_SETTINGS)
    return scoped


def settings_slug(settings: dict[str, float], scope: str) -> str:
    parts = []
    for key, value in settings_for_scope(settings, scope).items():
        if key.startswith("freq_"):
            continue
        text = f"{value:.5g}".replace("+", "").replace("-", "m").replace(".", "p")
        parts.append(f"{key}-{text}")
    return "__".join(parts)


def dedupe(candidates: list[tuple[str, dict[str, float]]]) -> list[tuple[str, dict[str, float]]]:
    seen = set()
    out = []
    for label, settings in candidates:
        key = tuple(sorted((name, round(float(value), 12)) for name, value in settings.items()))
        if key not in seen:
            seen.add(key)
            out.append((label, dict(settings)))
    return out


def rdl_candidates() -> list[tuple[str, dict[str, float]]]:
    best = dict(BASE_SETTINGS)
    best.update({"er_si": 9.8, "w_scale": 0.8, "pitch_scale": 1.1})
    candidates = [("baseline", dict(BASE_SETTINGS)), ("previous_mlin_best", best)]

    for er in [9.0, 9.4, 9.8, 10.2, 10.6]:
        for w in [0.75, 0.8, 0.85, 0.9]:
            for pitch in [1.05, 1.1, 1.15]:
                item = dict(best)
                item.update({"er_si": er, "w_scale": w, "pitch_scale": pitch})
                candidates.append((f"er{er:g}_w{w:g}_p{pitch:g}", item))

    for l_scale in [0.9, 0.95, 1.0, 1.05]:
        item = dict(best)
        item.update({"l_scale": l_scale})
        candidates.append((f"l{l_scale:g}", item))

    for h_rdl in [0.8, 0.9, 1.0, 1.1]:
        item = dict(best)
        item.update({"h_rdl_scale": h_rdl})
        candidates.append((f"hrdl{h_rdl:g}", item))

    for cond in [4.1e7, 5.0e7, 5.8e7]:
        for tand in [0.0, 0.005, 0.01]:
            item = dict(best)
            item.update({"cond": cond, "tand": tand})
            candidates.append((f"cond{cond:g}_tand{tand:g}", item))

    return dedupe(candidates)


def tsv_candidates() -> list[tuple[str, dict[str, float]]]:
    base = dict(BASE_SETTINGS)
    candidates = [("baseline", base)]

    for er in [9.8, 10.8, 11.9, 12.5, 13.5]:
        for c1 in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            for d_scale in [0.9, 1.0, 1.1]:
                item = dict(base)
                item.update({"er_si": er, "c1_scale": c1, "d_scale": d_scale})
                candidates.append((f"er{er:g}_c1{c1:g}_d{d_scale:g}", item))

    for pitch in [0.9, 1.0, 1.1]:
        for h_tsv in [0.9, 1.0, 1.1]:
            item = dict(base)
            item.update({"pitch_scale": pitch, "h_tsv_scale": h_tsv})
            candidates.append((f"pitch{pitch:g}_htsv{h_tsv:g}", item))

    for cond in [4.1e7, 5.0e7, 5.8e7]:
        for tand in [0.0, 0.005, 0.01]:
            item = dict(base)
            item.update({"cond": cond, "tand": tand})
            candidates.append((f"cond{cond:g}_tand{tand:g}", item))

    return dedupe(candidates)


def build_samples() -> dict[str, list[dict[str, object]]]:
    rdl_df = pd.read_csv(DATA_ROOT / "RDL_variations_record.csv", encoding="utf-8-sig")
    tsv_df = pd.read_csv(DATA_ROOT / "TSV_variations_record.csv", encoding="utf-8-sig")

    rdl_chosen = rdl_df.sample(n=SAMPLE_COUNT, random_state=RDL_RANDOM_SEED).sort_values("dut_index")
    tsv_chosen = tsv_df.sample(n=SAMPLE_COUNT, random_state=TSV_RANDOM_SEED).sort_values("dut_index")

    samples = {"RDL": [], "TSV": []}
    for _, row in rdl_chosen.iterrows():
        dut = int(row["dut_index"])
        rec = row.to_dict()
        rec["dut_index"] = dut
        rec["h_tmrdl"] = float(rec.pop("t_tmrdl"))
        rec["h_tsv"] = FIXED_RDL_H_TSV_UM
        rec["sample_id"] = f"LHS400C2_RDL_dut{dut}"
        rec["hfss_path"] = DATA_ROOT / "RDL" / f"dut{dut}.s2p"
        samples["RDL"].append(rec)

    for _, row in tsv_chosen.iterrows():
        dut = int(row["dut_index"])
        rec = row.to_dict()
        rec["dut_index"] = dut
        rec["sample_id"] = f"LHS400C2_TSV_dut{dut}"
        rec["hfss_path"] = DATA_ROOT / "TSV" / f"dut{dut}.s2p"
        samples["TSV"].append(rec)

    return samples


def sparam_y(s: np.ndarray) -> np.ndarray:
    return np.concatenate([s[:, 0, 0].real, s[:, 0, 0].imag, s[:, 1, 0].real, s[:, 1, 0].imag])


def nmse_s11_s21_ri(true_s: np.ndarray, pred_s: np.ndarray) -> float:
    y_true = sparam_y(true_s)
    y_pred = sparam_y(pred_s)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum((y_true - y_pred) ** 2) / denom)


def mse_complex(true_s: np.ndarray, pred_s: np.ndarray) -> float:
    return float(np.mean(np.abs(true_s - pred_s) ** 2))


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.6g}")
    rows = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(rows)


def simulate_one(scope: str, sample: dict[str, object], settings: dict[str, float]) -> Path:
    cache_base = OUTPUT_DIR / "cache" / scope / settings_slug(settings, scope) / str(sample["sample_id"])
    if scope == "rdl":
        return RDL_ADS.simulate_single_device(
            "TMRDL",
            str(sample["sample_id"]),
            sample,
            settings_for_scope(settings, "rdl"),
            output_base=cache_base,
            reuse_existing=REUSE_EXISTING,
        )
    return TSV_ADS.simulate_single_device(
        "TSV",
        str(sample["sample_id"]),
        sample,
        settings_for_scope(settings, "tsv"),
        output_base=cache_base,
        reuse_existing=REUSE_EXISTING,
    )


def evaluate_scope(scope: str, candidates: list[tuple[str, dict[str, float]]], samples: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows = []
    for idx, (label, settings) in enumerate(candidates, start=1):
        print(f"[{scope}] {idx}/{len(candidates)} {label}", flush=True)
        for sample in samples:
            ads_path = simulate_one(scope, sample, settings)
            ads_nw = rf.Network(str(ads_path))
            hfss_nw = rf.Network(str(sample["hfss_path"]))
            if len(ads_nw.f) != len(hfss_nw.f) or not np.allclose(ads_nw.f, hfss_nw.f):
                raise ValueError(f"Frequency mismatch: {sample['sample_id']}")
            row = {
                "scope": scope,
                "candidate": label,
                "sample_id": sample["sample_id"],
                "dut_index": int(sample["dut_index"]),
                "mse_complex": mse_complex(hfss_nw.s, ads_nw.s),
                "nmse_s11_s21_ri": nmse_s11_s21_ri(hfss_nw.s, ads_nw.s),
                "ads_path": str(ads_path),
            }
            row.update({key: value for key, value in settings_for_scope(settings, scope).items() if not key.startswith("freq_")})
            rows.append(row)

    detail = pd.DataFrame(rows)
    setting_cols = [
        col
        for col in ["er_si", "cond", "tand", "c1_scale", "l_scale", "w_scale", "pitch_scale", "h_tsv_scale", "h_rdl_scale", "d_scale"]
        if col in detail.columns
    ]
    summary = (
        detail.groupby(["scope", "candidate"] + setting_cols, as_index=False)
        .agg(
            count=("sample_id", "count"),
            mse_mean=("mse_complex", "mean"),
            mse_median=("mse_complex", "median"),
            nmse_mean=("nmse_s11_s21_ri", "mean"),
            nmse_median=("nmse_s11_s21_ri", "median"),
            nmse_max=("nmse_s11_s21_ri", "max"),
        )
        .sort_values(["nmse_mean", "mse_mean"], ignore_index=True)
    )
    best_row = summary.iloc[0]
    best_settings = dict(BASE_SETTINGS)
    for key in setting_cols:
        best_settings[key] = float(best_row[key])
    return detail, summary, best_settings


def plot_best(scope: str, best_settings: dict[str, float], samples: list[dict[str, object]]) -> list[str]:
    plot_dir = OUTPUT_DIR / "plots" / scope
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for sample in samples[:5]:
        ads_path = simulate_one(scope, sample, best_settings)
        ads_nw = rf.Network(str(ads_path))
        hfss_nw = rf.Network(str(sample["hfss_path"]))
        nmse = nmse_s11_s21_ri(hfss_nw.s, ads_nw.s)
        freq_ghz = hfss_nw.f / 1e9

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
        fig.suptitle(f"{scope.upper()} best | DUT {sample['dut_index']} | NMSE={nmse:.4g}", x=0.02, y=0.985, ha="left")
        for ax, (m, n, label, part) in zip(
            axes.ravel(),
            [(0, 0, "S11", "real"), (0, 0, "S11", "imag"), (1, 0, "S21", "real"), (1, 0, "S21", "imag")],
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
        out_path = plot_dir / f"{scope.upper()}_{sample['dut_index']}_best_compare.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(str(out_path))
    return saved


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()

    rdl_detail, rdl_summary, best_rdl = evaluate_scope("rdl", rdl_candidates(), samples["RDL"])
    tsv_detail, tsv_summary, best_tsv = evaluate_scope("tsv", tsv_candidates(), samples["TSV"])

    detail = pd.concat([rdl_detail, tsv_detail], ignore_index=True)
    summary = pd.concat([rdl_summary, tsv_summary], ignore_index=True)
    detail.to_csv(OUTPUT_DIR / "ads_calibration_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_calibration_summary.csv", index=False, encoding="utf-8-sig")

    plots = plot_best("rdl", best_rdl, samples["RDL"]) + plot_best("tsv", best_tsv, samples["TSV"])
    best = {
        "source_dataset": "LHS400_Connection2",
        "source_split": "train",
        "rdl_random_seed": RDL_RANDOM_SEED,
        "tsv_random_seed": TSV_RANDOM_SEED,
        "sample_count_each": SAMPLE_COUNT,
        "rdl_sample_duts": [int(sample["dut_index"]) for sample in samples["RDL"]],
        "tsv_sample_duts": [int(sample["dut_index"]) for sample in samples["TSV"]],
        "fixed_rdl_h_tsv_um": FIXED_RDL_H_TSV_UM,
        "ads_frequency_settings": FREQ_SETTINGS,
        "metric": "NMSE over flattened S11/S21 real and imaginary components",
        "best_rdl_settings": {key: value for key, value in settings_for_scope(best_rdl, "rdl").items() if not key.startswith("freq_")},
        "best_tsv_settings": {key: value for key, value in settings_for_scope(best_tsv, "tsv").items() if not key.startswith("freq_")},
        "plots": plots,
    }
    (OUTPUT_DIR / "best_ads_settings.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    archive = [
        "# LHS400_Connection2 Random-10 RDL and TSV ADS Calibration",
        "",
        "- ADS RDL template: current MLIN2 netlist.",
        "- ADS TSV template: current d_tsv netlist.",
        f"- Source: `{DATA_ROOT}`",
        f"- RDL random seed: `{RDL_RANDOM_SEED}`",
        f"- TSV random seed: `{TSV_RANDOM_SEED}`",
        f"- RDL DUTs: `{best['rdl_sample_duts']}`",
        f"- TSV DUTs: `{best['tsv_sample_duts']}`",
        f"- Fixed standalone RDL `h_tsv`: `{FIXED_RDL_H_TSV_UM} um`",
        "- Metric: NMSE over flattened `S11.real`, `S11.imag`, `S21.real`, and `S21.imag`.",
        f"- Detail CSV: `{OUTPUT_DIR / 'ads_calibration_detail.csv'}`",
        f"- Summary CSV: `{OUTPUT_DIR / 'ads_calibration_summary.csv'}`",
        f"- Best settings JSON: `{OUTPUT_DIR / 'best_ads_settings.json'}`",
        "",
        "## Best RDL Settings",
        "",
        "```json",
        json.dumps(best["best_rdl_settings"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Best TSV Settings",
        "",
        "```json",
        json.dumps(best["best_tsv_settings"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top RDL Candidates",
        "",
        dataframe_to_markdown(rdl_summary.head(10)),
        "",
        "## Top TSV Candidates",
        "",
        dataframe_to_markdown(tsv_summary.head(10)),
        "",
    ]
    (OUTPUT_DIR / "validation_archive.md").write_text("\n".join(archive), encoding="utf-8")

    print("Best RDL settings:", json.dumps(best["best_rdl_settings"], ensure_ascii=False), flush=True)
    print("Best TSV settings:", json.dumps(best["best_tsv_settings"], ensure_ascii=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
