# -*- coding: utf-8 -*-
"""Recalibrate RDL ADS settings after the RDL netlist update.

Run this file directly in VS Code. No command-line arguments are required.
This pass reuses the 16-DUT calibration subset and refined RDL candidate search,
but writes to a fresh result directory so old ADS cache is not reused.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REFINED_SCRIPT = THIS_DIR / "calibrate_ads_single_devices_v10_refined.py"
PROJECT_ROOT = THIS_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / "ads_cal_rdl_update16"
SAMPLE_DUTS = [100, 113, 126, 140, 153, 166, 180, 193, 206, 219, 233, 246, 259, 273, 286, 299]
PREVIOUS_TSV_SETTINGS = {
    "er_si": 11.9,
    "cond": 5.8e7,
    "tand": 0.005,
    "c1_scale": 1.0,
    "pitch_scale": 1.0,
    "h_tsv_scale": 1.1,
    "d_scale": 1.0,
}


def load_refined_module():
    spec = importlib.util.spec_from_file_location("v10_ads_single_device_calibration_refined", REFINED_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load refined calibration script: {REFINED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    refined = load_refined_module()
    refined.OUTPUT_DIR = OUTPUT_DIR
    refined.SAMPLE_DUTS = SAMPLE_DUTS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cal = refined.load_base_module()
    cal.OUTPUT_DIR = OUTPUT_DIR
    cal.SAMPLE_DUTS = SAMPLE_DUTS
    cal.SAMPLE_COUNT = len(SAMPLE_DUTS)
    cal.REUSE_EXISTING = True

    samples = cal.build_samples()
    rdl_detail, rdl_summary, best_rdl = refined.run_candidates(cal, "rdl", refined.rdl_candidates(cal), samples)

    rdl_detail.to_csv(OUTPUT_DIR / "ads_single_device_calibration_detail.csv", index=False, encoding="utf-8-sig")
    rdl_summary.to_csv(OUTPUT_DIR / "ads_single_device_calibration_summary.csv", index=False, encoding="utf-8-sig")

    plot_files = cal.plot_best("rdl", best_rdl, samples)
    best = {
        "source_dataset": cal.SOURCE_DATASET,
        "source_split": cal.SOURCE_SPLIT,
        "sample_duts": SAMPLE_DUTS,
        "metric": "NMSE over flattened S11/S21 real and imaginary components",
        "note": "RDL-only recalibration after the RDL ADS netlist update. TSV settings are carried over from ads_single_device_calibration_16dut.",
        "best_rdl_settings": cal.settings_for_scope(best_rdl, "rdl"),
        "best_tsv_settings": PREVIOUS_TSV_SETTINGS,
        "plots": plot_files,
    }
    (OUTPUT_DIR / "best_ads_calibration_settings.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rdl_top = rdl_summary.head(10)
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# ADS RDL Net-Update Calibration Validation",
                "",
                f"- Source: `{cal.SOURCE_DATASET}/{cal.SOURCE_SPLIT}`",
                f"- DUTs: `{SAMPLE_DUTS}`",
                "- Scope: RDL-only recalibration after the RDL ADS netlist update.",
                "- TSV settings: carried over from `ads_single_device_calibration_16dut`.",
                "- Search: refined RDL candidates around previous calibration settings.",
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
                "## Carried-Over TSV Settings",
                "",
                "```json",
                json.dumps(PREVIOUS_TSV_SETTINGS, indent=2, ensure_ascii=False),
                "```",
                "",
                "## Top RDL Candidates",
                "",
                cal.dataframe_to_markdown(rdl_top),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Best RDL settings:", json.dumps(cal.settings_for_scope(best_rdl, "rdl"), ensure_ascii=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
