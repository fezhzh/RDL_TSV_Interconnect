# -*- coding: utf-8 -*-
"""Refine ADS RDL/TSV calibration on random LHS400_Connection2 samples.

Run this file directly in VS Code. No command-line arguments are required.
It reuses the same random-10 RDL/TSV samples as
``calibrate_ads_lhs400_connection2_rdl_tsv_random10.py`` and writes a separate
short result directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
COARSE_SCRIPT = THIS_DIR / "calibrate_ads_lhs400_connection2_rdl_tsv_random10.py"
PROJECT_ROOT = THIS_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / "ac_l400_ref2"


def load_coarse_module():
    spec = importlib.util.spec_from_file_location("v10_lhs400_rdl_tsv_cal", COARSE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load calibration script: {COARSE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUTPUT_DIR = OUTPUT_DIR
    return module


def dedupe(candidates: list[tuple[str, dict[str, float]]]) -> list[tuple[str, dict[str, float]]]:
    seen = set()
    out = []
    for label, settings in candidates:
        key = tuple(sorted((name, round(float(value), 12)) for name, value in settings.items()))
        if key not in seen:
            seen.add(key)
            out.append((label, dict(settings)))
    return out


def rdl_refined_candidates(coarse) -> list[tuple[str, dict[str, float]]]:
    best = dict(coarse.BASE_SETTINGS)
    best.update(
        {
            "er_si": 9.4,
            "cond": 5.8e7,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.75,
            "pitch_scale": 1.15,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 1.0,
        }
    )
    candidates = [("prev_best", best)]

    for er in [8.6, 9.0, 9.2, 9.4, 9.6, 9.8]:
        for w in [0.65, 0.7, 0.75, 0.8]:
            for pitch in [1.1, 1.15, 1.2, 1.25]:
                item = dict(best)
                item.update({"er_si": er, "w_scale": w, "pitch_scale": pitch})
                candidates.append((f"er{er:g}_w{w:g}_p{pitch:g}", item))

    for l_scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
        for h_rdl in [0.8, 0.9, 1.0, 1.1]:
            item = dict(best)
            item.update({"l_scale": l_scale, "h_rdl_scale": h_rdl})
            candidates.append((f"l{l_scale:g}_hrdl{h_rdl:g}", item))

    for cond in [4.1e7, 5.0e7, 5.8e7, 6.5e7]:
        for tand in [0.0, 0.002, 0.005, 0.008, 0.01]:
            item = dict(best)
            item.update({"cond": cond, "tand": tand})
            candidates.append((f"cond{cond:g}_tand{tand:g}", item))

    return dedupe(candidates)


def tsv_refined_candidates(coarse) -> list[tuple[str, dict[str, float]]]:
    best = dict(coarse.BASE_SETTINGS)
    best.update(
        {
            "er_si": 11.9,
            "cond": 5.8e7,
            "tand": 0.005,
            "c1_scale": 1.0,
            "pitch_scale": 0.9,
            "h_tsv_scale": 1.1,
            "d_scale": 1.0,
        }
    )
    candidates = [("prev_best", best)]

    for pitch in [0.8, 0.85, 0.9, 0.95, 1.0]:
        for h_tsv in [1.0, 1.05, 1.1, 1.15, 1.2]:
            item = dict(best)
            item.update({"pitch_scale": pitch, "h_tsv_scale": h_tsv})
            candidates.append((f"pitch{pitch:g}_htsv{h_tsv:g}", item))

    for er in [10.8, 11.4, 11.9, 12.5]:
        for c1 in [0.85, 0.95, 1.0, 1.05, 1.15]:
            for d_scale in [0.9, 1.0, 1.1]:
                item = dict(best)
                item.update({"er_si": er, "c1_scale": c1, "d_scale": d_scale})
                candidates.append((f"er{er:g}_c1{c1:g}_d{d_scale:g}", item))

    for cond in [4.1e7, 5.0e7, 5.8e7, 6.5e7]:
        for tand in [0.0, 0.002, 0.005, 0.008, 0.01]:
            item = dict(best)
            item.update({"cond": cond, "tand": tand})
            candidates.append((f"cond{cond:g}_tand{tand:g}", item))

    return dedupe(candidates)


def main() -> None:
    coarse = load_coarse_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = coarse.build_samples()

    rdl_detail, rdl_summary, best_rdl = coarse.evaluate_scope("rdl", rdl_refined_candidates(coarse), samples["RDL"])
    tsv_detail, tsv_summary, best_tsv = coarse.evaluate_scope("tsv", tsv_refined_candidates(coarse), samples["TSV"])

    detail = pd.concat([rdl_detail, tsv_detail], ignore_index=True)
    summary = pd.concat([rdl_summary, tsv_summary], ignore_index=True)
    detail.to_csv(OUTPUT_DIR / "ads_calibration_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_calibration_summary.csv", index=False, encoding="utf-8-sig")

    plots = coarse.plot_best("rdl", best_rdl, samples["RDL"]) + coarse.plot_best("tsv", best_tsv, samples["TSV"])
    best = {
        "source_dataset": "LHS400_Connection2",
        "source_split": "train",
        "rdl_random_seed": coarse.RDL_RANDOM_SEED,
        "tsv_random_seed": coarse.TSV_RANDOM_SEED,
        "sample_count_each": coarse.SAMPLE_COUNT,
        "rdl_sample_duts": [int(sample["dut_index"]) for sample in samples["RDL"]],
        "tsv_sample_duts": [int(sample["dut_index"]) for sample in samples["TSV"]],
        "fixed_rdl_h_tsv_um": coarse.FIXED_RDL_H_TSV_UM,
        "ads_frequency_settings": coarse.FREQ_SETTINGS,
        "metric": "NMSE over flattened S11/S21 real and imaginary components",
        "best_rdl_settings": {key: value for key, value in coarse.settings_for_scope(best_rdl, "rdl").items() if not key.startswith("freq_")},
        "best_tsv_settings": {key: value for key, value in coarse.settings_for_scope(best_tsv, "tsv").items() if not key.startswith("freq_")},
        "plots": plots,
    }
    (OUTPUT_DIR / "best_ads_settings.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    archive = [
        "# LHS400_Connection2 Random-10 RDL and TSV ADS Refined Calibration",
        "",
        "- ADS RDL template: current MLIN2 netlist.",
        "- ADS TSV template: current d_tsv netlist.",
        f"- Source: `{coarse.DATA_ROOT}`",
        f"- RDL random seed: `{coarse.RDL_RANDOM_SEED}`",
        f"- TSV random seed: `{coarse.TSV_RANDOM_SEED}`",
        f"- RDL DUTs: `{best['rdl_sample_duts']}`",
        f"- TSV DUTs: `{best['tsv_sample_duts']}`",
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
        coarse.dataframe_to_markdown(rdl_summary.head(10)),
        "",
        "## Top TSV Candidates",
        "",
        coarse.dataframe_to_markdown(tsv_summary.head(10)),
        "",
    ]
    (OUTPUT_DIR / "validation_archive.md").write_text("\n".join(archive), encoding="utf-8")

    print("Best refined RDL settings:", json.dumps(best["best_rdl_settings"], ensure_ascii=False), flush=True)
    print("Best refined TSV settings:", json.dumps(best["best_tsv_settings"], ensure_ascii=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
