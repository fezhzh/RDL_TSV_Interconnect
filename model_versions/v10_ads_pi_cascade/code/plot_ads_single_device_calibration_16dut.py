# -*- coding: utf-8 -*-
"""Generate additional 16-DUT ADS single-device calibration comparison plots.

Run this file directly in VS Code. No command-line arguments are required.
It reuses the saved best 16-DUT calibration settings and ADS cache when present.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import skrf as rf


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "calibrate_ads_single_devices_v10.py"
PROJECT_ROOT = THIS_DIR.parents[2]
RESULT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / "ads_single_device_calibration_16dut"
BEST_JSON = RESULT_DIR / "best_ads_calibration_settings.json"
PLOT_DIR = RESULT_DIR / "plots_all_best"


def load_base_module():
    spec = importlib.util.spec_from_file_location("v10_ads_single_device_calibration", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load calibration script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plot_one(cal, device_name: str, sample: dict[str, object], settings: dict[str, float], scope: str) -> dict[str, object]:
    ads_path = cal.simulate_one(device_name, sample, settings, scope)
    ads_nw = rf.Network(str(ads_path))
    hfss_nw = rf.Network(str(sample["hfss_path"]))
    freq_ghz = hfss_nw.f / 1e9
    nmse = cal.nmse_s11_s21_ri(hfss_nw.s, ads_nw.s)
    mse = cal.mse_complex(hfss_nw.s, ads_nw.s)

    out_dir = PLOT_DIR / scope / device_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{device_name}_{sample['dut_index']}_best_compare.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    fig.suptitle(
        f"{scope} best | {device_name} | DUT {sample['dut_index']} | NMSE={nmse:.4g}",
        x=0.02,
        y=0.985,
        ha="left",
    )
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "scope": scope,
        "device": device_name,
        "dut_index": int(sample["dut_index"]),
        "sample_id": sample["sample_id"],
        "nmse_s11_s21_ri": nmse,
        "mse_complex": mse,
        "plot_path": str(out_path),
        "ads_path": str(ads_path),
    }


def main():
    cal = load_base_module()
    best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    cal.OUTPUT_DIR = RESULT_DIR
    cal.SAMPLE_DUTS = [int(value) for value in best["sample_duts"]]
    cal.SAMPLE_COUNT = len(cal.SAMPLE_DUTS)
    cal.REUSE_EXISTING = True

    samples = cal.build_samples()
    rows = []
    for device_name in ["TMRDL", "BSMRDL"]:
        for sample in samples[device_name]:
            rows.append(plot_one(cal, device_name, sample, best["best_rdl_settings"], "rdl"))
    for sample in samples["TSV"]:
        rows.append(plot_one(cal, "TSV", sample, best["best_tsv_settings"], "tsv"))

    summary = pd.DataFrame(rows).sort_values(["scope", "device", "dut_index"], ignore_index=True)
    summary_path = RESULT_DIR / "plots_all_best_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    archive_path = RESULT_DIR / "plots_all_best_validation.md"
    archive_path.write_text(
        "\n".join(
            [
                "# 16-DUT ADS Single-Device Additional Plot Validation",
                "",
                f"- Source result: `{RESULT_DIR}`",
                f"- Best settings JSON: `{BEST_JSON}`",
                f"- DUTs: `{cal.SAMPLE_DUTS}`",
                f"- Plot directory: `{PLOT_DIR}`",
                f"- Summary CSV: `{summary_path}`",
                f"- Plot count: `{len(summary)}`",
                "- Each plot compares HFSS and ADS for `S11.real`, `S11.imag`, `S21.real`, and `S21.imag`.",
                "",
                "## Mean NMSE By Device",
                "",
                cal.dataframe_to_markdown(
                    summary.groupby(["scope", "device"], as_index=False)
                    .agg(
                        count=("dut_index", "count"),
                        nmse_mean=("nmse_s11_s21_ri", "mean"),
                        nmse_median=("nmse_s11_s21_ri", "median"),
                        nmse_max=("nmse_s11_s21_ri", "max"),
                    )
                    .sort_values(["scope", "device"], ignore_index=True)
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(summary)} plots: {PLOT_DIR}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
