# -*- coding: utf-8 -*-
"""Refined small-sample ADS single-device calibration for v10.

Run this file directly in VS Code. No command-line arguments are required.
This second pass uses more DUTs and combined candidates around the first
calibration result instead of one-variable-at-a-time scans only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "calibrate_ads_single_devices_v10.py"
PROJECT_ROOT = THIS_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / "ads_single_device_calibration_refined"
SAMPLE_DUTS = [100, 101, 102, 103, 104, 105]


def load_base_module():
    spec = importlib.util.spec_from_file_location("v10_ads_single_device_calibration", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load calibration script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dedupe_candidates(candidates):
    seen = set()
    out = []
    for label, settings in candidates:
        key = tuple(sorted((name, round(float(value), 12)) for name, value in settings.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, dict(settings)))
    return out


def rdl_candidates(cal):
    base = dict(cal.BASE_SETTINGS)
    first_best = {
        **base,
        "er_si": 10.8,
        "cond": 5.8e7,
        "tand": 0.005,
        "l_scale": 1.0,
        "w_scale": 0.9,
        "pitch_scale": 1.0,
        "h_tsv_scale": 1.0,
        "h_rdl_scale": 1.0,
    }
    candidates = [("baseline", base), ("first_best", first_best)]

    for er in [9.8, 10.2, 10.8, 11.2, 11.9]:
        for w in [0.8, 0.85, 0.9, 0.95, 1.0]:
            item = dict(first_best)
            item.update({"er_si": er, "w_scale": w})
            candidates.append((f"er{er:g}_w{w:g}", item))

    for er in [10.2, 10.8]:
        for w in [0.85, 0.9]:
            for h_rdl in [0.8, 0.9, 1.0]:
                item = dict(first_best)
                item.update({"er_si": er, "w_scale": w, "h_rdl_scale": h_rdl})
                candidates.append((f"er{er:g}_w{w:g}_hrdl{h_rdl:g}", item))

    for er in [10.2, 10.8]:
        for w in [0.85, 0.9]:
            for pitch in [0.95, 1.0, 1.05, 1.1]:
                item = dict(first_best)
                item.update({"er_si": er, "w_scale": w, "pitch_scale": pitch})
                candidates.append((f"er{er:g}_w{w:g}_pitch{pitch:g}", item))

    for er in [10.2, 10.8]:
        for w in [0.85, 0.9]:
            for h_tsv in [0.9, 1.0, 1.1, 1.2]:
                item = dict(first_best)
                item.update({"er_si": er, "w_scale": w, "h_tsv_scale": h_tsv})
                candidates.append((f"er{er:g}_w{w:g}_htsv{h_tsv:g}", item))

    for l_scale in [0.85, 0.9, 0.95, 1.05]:
        item = dict(first_best)
        item.update({"l_scale": l_scale})
        candidates.append((f"l{l_scale:g}_first_best", item))

    for cond in [3.2e7, 4.1e7, 5.8e7]:
        for tand in [0.0, 0.005, 0.01]:
            item = dict(first_best)
            item.update({"cond": cond, "tand": tand})
            candidates.append((f"cond{cond:g}_tand{tand:g}", item))

    return dedupe_candidates(candidates)


def tsv_candidates(cal):
    base = dict(cal.BASE_SETTINGS)
    candidates = [("baseline", base)]

    for c1 in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        item = dict(base)
        item.update({"c1_scale": c1})
        candidates.append((f"c1{c1:g}", item))

    for c1 in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for d_scale in [0.85, 0.9, 1.0, 1.1, 1.15]:
            item = dict(base)
            item.update({"c1_scale": c1, "d_scale": d_scale})
            candidates.append((f"c1{c1:g}_d{d_scale:g}", item))

    for c1 in [0.75, 1.0, 1.25]:
        for pitch in [0.9, 1.0, 1.1]:
            for h_tsv in [0.9, 1.0, 1.1]:
                item = dict(base)
                item.update({"c1_scale": c1, "pitch_scale": pitch, "h_tsv_scale": h_tsv})
                candidates.append((f"c1{c1:g}_p{pitch:g}_h{h_tsv:g}", item))

    for er in [10.8, 11.9, 12.5]:
        for cond in [4.1e7, 5.8e7]:
            item = dict(base)
            item.update({"er_si": er, "cond": cond})
            candidates.append((f"er{er:g}_cond{cond:g}", item))

    return dedupe_candidates(candidates)


def run_candidates(cal, scope, candidates, samples):
    rows = []
    for idx, (label, settings) in enumerate(candidates, start=1):
        print(f"[{scope}] refined {idx}/{len(candidates)} {label}", flush=True)
        rows.extend(cal.evaluate_candidate(scope, label, settings, samples))
    detail = pd.DataFrame(rows)
    summary = cal.summarize(rows)
    best_row = summary.iloc[0]
    best_settings = dict(cal.BASE_SETTINGS)
    for key in cal.settings_for_scope(cal.BASE_SETTINGS, scope):
        best_settings[key] = float(best_row[key])
    return detail, summary, best_settings


def main():
    cal = load_base_module()
    cal.OUTPUT_DIR = OUTPUT_DIR
    cal.SAMPLE_DUTS = SAMPLE_DUTS
    cal.SAMPLE_COUNT = len(SAMPLE_DUTS)
    cal.REUSE_EXISTING = True
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = cal.build_samples()
    rdl_detail, rdl_summary, best_rdl = run_candidates(cal, "rdl", rdl_candidates(cal), samples)
    tsv_detail, tsv_summary, best_tsv = run_candidates(cal, "tsv", tsv_candidates(cal), samples)

    detail = pd.concat([rdl_detail, tsv_detail], ignore_index=True)
    summary = pd.concat([rdl_summary, tsv_summary], ignore_index=True)
    detail.to_csv(OUTPUT_DIR / "ads_single_device_calibration_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_single_device_calibration_summary.csv", index=False, encoding="utf-8-sig")

    plot_files = cal.plot_best("rdl", best_rdl, samples) + cal.plot_best("tsv", best_tsv, samples)
    best = {
        "source_dataset": cal.SOURCE_DATASET,
        "source_split": cal.SOURCE_SPLIT,
        "sample_duts": SAMPLE_DUTS,
        "metric": "NMSE over flattened S11/S21 real and imaginary components",
        "best_rdl_settings": cal.settings_for_scope(best_rdl, "rdl"),
        "best_tsv_settings": cal.settings_for_scope(best_tsv, "tsv"),
        "plots": plot_files,
    }
    (OUTPUT_DIR / "best_ads_calibration_settings.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rdl_top = rdl_summary.head(10)
    tsv_top = tsv_summary.head(10)
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# ADS Single-Device Refined Calibration Validation",
                "",
                f"- Source: `{cal.SOURCE_DATASET}/{cal.SOURCE_SPLIT}`",
                f"- DUTs: `{SAMPLE_DUTS}`",
                "- Search: combined refined candidates around the first calibration result.",
                "- Metric: NMSE over flattened `S11.real`, `S11.imag`, `S21.real`, and `S21.imag`.",
                f"- Detail CSV: `{OUTPUT_DIR / 'ads_single_device_calibration_detail.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'ads_single_device_calibration_summary.csv'}`",
                f"- Best settings JSON: `{OUTPUT_DIR / 'best_ads_calibration_settings.json'}`",
                "",
                "## Best RDL Settings",
                "",
                "```json",
                json.dumps(cal.settings_for_scope(best_rdl, "rdl"), indent=2, ensure_ascii=False),
                "```",
                "",
                "## Best TSV Settings",
                "",
                "```json",
                json.dumps(cal.settings_for_scope(best_tsv, "tsv"), indent=2, ensure_ascii=False),
                "```",
                "",
                "## Top RDL Candidates",
                "",
                cal.dataframe_to_markdown(rdl_top),
                "",
                "## Top TSV Candidates",
                "",
                cal.dataframe_to_markdown(tsv_top),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Best RDL settings:", json.dumps(cal.settings_for_scope(best_rdl, "rdl"), ensure_ascii=False), flush=True)
    print("Best TSV settings:", json.dumps(cal.settings_for_scope(best_tsv, "tsv"), ensure_ascii=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
