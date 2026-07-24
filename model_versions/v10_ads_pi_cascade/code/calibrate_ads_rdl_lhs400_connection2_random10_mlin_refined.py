# -*- coding: utf-8 -*-
"""Refine ADS MLIN RDL calibration on the same random LHS400_Connection2 samples.

Run this file directly in VS Code. No command-line arguments are required.
It reuses the random-10 sample selection from
``calibrate_ads_rdl_lhs400_connection2_random10.py`` and writes a separate
result directory so the coarse MLIN run remains available for comparison.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
COARSE_SCRIPT = THIS_DIR / "calibrate_ads_rdl_lhs400_connection2_random10.py"
PROJECT_ROOT = THIS_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / "ac_rdl_l400_mlin_ref"


def load_coarse_module():
    spec = importlib.util.spec_from_file_location("v10_ads_rdl_lhs400_rand10_coarse", COARSE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load coarse calibration script: {COARSE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUTPUT_DIR = OUTPUT_DIR
    return module


def dedupe_candidates(candidates):
    seen = set()
    out = []
    for label, settings in candidates:
        key = tuple(sorted((name, round(float(value), 12)) for name, value in settings.items()))
        if key not in seen:
            seen.add(key)
            out.append((label, dict(settings)))
    return out


def refined_rdl_candidates(cal):
    best = dict(cal.BASE_SETTINGS)
    best.update(
        {
            "er_si": 9.8,
            "cond": 5.8e7,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.8,
            "pitch_scale": 1.1,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 1.0,
        }
    )

    candidates = [("coarse_best", best)]

    for er in [8.8, 9.2, 9.6, 9.8, 10.0, 10.2, 10.6]:
        for w in [0.7, 0.75, 0.8, 0.85, 0.9]:
            for pitch in [1.05, 1.1, 1.15, 1.2]:
                item = dict(best)
                item.update({"er_si": er, "w_scale": w, "pitch_scale": pitch})
                candidates.append((f"er{er:g}_w{w:g}_pitch{pitch:g}", item))

    for l_scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
        for h_rdl in [0.7, 0.8, 0.9, 1.0, 1.1]:
            item = dict(best)
            item.update({"l_scale": l_scale, "h_rdl_scale": h_rdl})
            candidates.append((f"l{l_scale:g}_hrdl{h_rdl:g}", item))

    for cond in [3.2e7, 4.1e7, 5.0e7, 5.8e7, 6.5e7]:
        for tand in [0.0, 0.002, 0.005, 0.008, 0.01]:
            item = dict(best)
            item.update({"cond": cond, "tand": tand})
            candidates.append((f"cond{cond:g}_tand{tand:g}", item))

    for h_tsv in [0.8, 0.9, 1.0, 1.1, 1.2]:
        item = dict(best)
        item.update({"h_tsv_scale": h_tsv})
        candidates.append((f"htsv{h_tsv:g}", item))

    return dedupe_candidates(candidates)


def main():
    coarse = load_coarse_module()
    cal = coarse.load_base_module()
    cal.OUTPUT_DIR = OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = coarse.build_samples()
    rows = []
    candidates = refined_rdl_candidates(cal)
    for idx, (label, settings) in enumerate(candidates, start=1):
        print(f"[rdl refined] {idx}/{len(candidates)} {label}", flush=True)
        rows.extend(coarse.evaluate_candidate(cal, label, settings, samples))

    detail = pd.DataFrame(rows)
    summary = cal.summarize(rows)
    best_row = summary.iloc[0]
    best_settings = dict(cal.BASE_SETTINGS)
    for key in cal.settings_for_scope(cal.BASE_SETTINGS, "rdl"):
        best_settings[key] = float(best_row[key])

    detail.to_csv(OUTPUT_DIR / "ads_rdl_calibration_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_rdl_calibration_summary.csv", index=False, encoding="utf-8-sig")
    plot_files = coarse.plot_best(cal, best_settings, samples)

    best = {
        "source_dataset": "LHS400_Connection2",
        "source_split": "train",
        "random_seed": coarse.RANDOM_SEED,
        "sample_count": coarse.SAMPLE_COUNT,
        "sample_duts": [int(sample["dut_index"]) for sample in samples],
        "fixed_h_tsv_um": coarse.FIXED_H_TSV_UM,
        "ads_frequency_settings": coarse.FREQ_SETTINGS,
        "metric": "NMSE over flattened S11/S21 real and imaginary components",
        "best_rdl_settings": cal.settings_for_scope(best_settings, "rdl"),
        "plots": plot_files,
    }
    (OUTPUT_DIR / "best_ads_rdl_settings.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    archive = [
        "# LHS400_Connection2 Random-10 RDL ADS MLIN Refined Calibration",
        "",
        "- ADS RDL template: updated MLIN netlist.",
        f"- Source: `{coarse.DATA_ROOT / 'RDL'}`",
        f"- Variation table: `{coarse.DATA_ROOT / 'RDL_variations_record.csv'}`",
        f"- Random seed: `{coarse.RANDOM_SEED}`",
        f"- Sample DUTs: `{best['sample_duts']}`",
        f"- Fixed RDL substrate height input `h_tsv`: `{coarse.FIXED_H_TSV_UM} um`",
        "- Metric: NMSE over flattened `S11.real`, `S11.imag`, `S21.real`, and `S21.imag`.",
        f"- Detail CSV: `{OUTPUT_DIR / 'ads_rdl_calibration_detail.csv'}`",
        f"- Summary CSV: `{OUTPUT_DIR / 'ads_rdl_calibration_summary.csv'}`",
        f"- Best settings JSON: `{OUTPUT_DIR / 'best_ads_rdl_settings.json'}`",
        "",
        "## Best RDL Settings",
        "",
        "```json",
        json.dumps(cal.settings_for_scope(best_settings, "rdl"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top RDL Candidates",
        "",
        cal.dataframe_to_markdown(summary.head(10)),
        "",
    ]
    (OUTPUT_DIR / "validation_archive.md").write_text("\n".join(archive), encoding="utf-8")
    print("Best refined RDL settings:", json.dumps(cal.settings_for_scope(best_settings, "rdl"), ensure_ascii=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
