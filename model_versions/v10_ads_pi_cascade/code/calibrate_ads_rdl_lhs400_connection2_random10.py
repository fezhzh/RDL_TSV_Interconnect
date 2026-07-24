# -*- coding: utf-8 -*-
"""Calibrate ADS MLIN RDL settings on 10 random LHS400_Connection2 RDL samples.

Run this file directly in VS Code. No command-line arguments are required.
The RDL ADS netlist is expected to use the updated MLIN template.
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
BASE_CAL_SCRIPT = THIS_DIR / "calibrate_ads_single_devices_v10.py"
PROJECT_ROOT = THIS_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / "ads_cal_rdl_lhs400c2_rand10_mlin"
DATA_ROOT = PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train"
RANDOM_SEED = 20260708
SAMPLE_COUNT = 10
FIXED_H_TSV_UM = 100.0
FREQ_SETTINGS = {"freq_start_ghz": 0.1, "freq_stop_ghz": 100.0, "freq_step_ghz": 0.1}


def load_base_module():
    spec = importlib.util.spec_from_file_location("v10_ads_single_device_calibration", BASE_CAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load calibration script: {BASE_CAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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


def rdl_candidates(cal):
    base = dict(cal.BASE_SETTINGS)
    candidates = [("baseline", base)]

    for er in [9.8, 10.2, 10.8, 11.2, 11.9, 12.5]:
        for w in [0.8, 0.85, 0.9, 0.95, 1.0]:
            item = dict(base)
            item.update({"er_si": er, "w_scale": w})
            candidates.append((f"er{er:g}_w{w:g}", item))

    for er in [9.8, 10.2, 10.8, 11.9]:
        for w in [0.8, 0.85, 0.9]:
            for pitch in [0.9, 1.0, 1.05, 1.1]:
                item = dict(base)
                item.update({"er_si": er, "w_scale": w, "pitch_scale": pitch})
                candidates.append((f"er{er:g}_w{w:g}_pitch{pitch:g}", item))

    for er in [9.8, 10.2, 10.8]:
        for w in [0.8, 0.85, 0.9]:
            for h_rdl in [0.8, 0.9, 1.0, 1.1]:
                item = dict(base)
                item.update({"er_si": er, "w_scale": w, "h_rdl_scale": h_rdl})
                candidates.append((f"er{er:g}_w{w:g}_hrdl{h_rdl:g}", item))

    for l_scale in [0.85, 0.9, 0.95, 1.0, 1.05]:
        item = dict(base)
        item.update({"l_scale": l_scale})
        candidates.append((f"l{l_scale:g}", item))

    for h_tsv in [0.8, 0.9, 1.0, 1.1, 1.2]:
        item = dict(base)
        item.update({"h_tsv_scale": h_tsv})
        candidates.append((f"htsv{h_tsv:g}", item))

    for cond in [3.2e7, 4.1e7, 5.8e7]:
        for tand in [0.0, 0.005, 0.01]:
            item = dict(base)
            item.update({"cond": cond, "tand": tand})
            candidates.append((f"cond{cond:g}_tand{tand:g}", item))

    return dedupe_candidates(candidates)


def build_samples() -> list[dict[str, object]]:
    var_path = DATA_ROOT / "RDL_variations_record.csv"
    df = pd.read_csv(var_path, encoding="utf-8-sig")
    chosen = df.sample(n=SAMPLE_COUNT, random_state=RANDOM_SEED).sort_values("dut_index")
    samples = []
    for _, row in chosen.iterrows():
        dut = int(row["dut_index"])
        rec = row.to_dict()
        rec["dut_index"] = dut
        rec["h_tmrdl"] = float(rec.pop("t_tmrdl"))
        rec["h_tsv"] = FIXED_H_TSV_UM
        rec["sample_id"] = f"LHS400_Connection2_train_RDL_dut{dut}"
        rec["hfss_path"] = DATA_ROOT / "RDL" / f"dut{dut}.s2p"
        samples.append(rec)
    return samples


def evaluate_candidate(cal, label: str, settings: dict[str, float], samples: list[dict[str, object]]):
    rows = []
    ads_settings = {**cal.settings_for_scope(settings, "rdl"), **FREQ_SETTINGS}
    for sample in samples:
        ads_path = cal.RDL_ADS.simulate_single_device(
            "TMRDL",
            str(sample["sample_id"]),
            sample,
            ads_settings,
            output_base=OUTPUT_DIR / "ads_cache" / cal.settings_slug(settings, "rdl") / str(sample["sample_id"]),
            reuse_existing=True,
        )
        ads_nw = rf.Network(str(ads_path))
        hfss_nw = rf.Network(str(sample["hfss_path"]))
        if len(ads_nw.f) != len(hfss_nw.f) or not np.allclose(ads_nw.f, hfss_nw.f):
            raise ValueError(f"Frequency mismatch: {sample['sample_id']}")
        rows.append(
            {
                "scope": "rdl",
                "candidate": label,
                "device": "RDL",
                "sample_id": sample["sample_id"],
                "dut_index": int(sample["dut_index"]),
                "mse_complex": cal.mse_complex(hfss_nw.s, ads_nw.s),
                "nmse_s11_s21_ri": cal.nmse_s11_s21_ri(hfss_nw.s, ads_nw.s),
                "ads_path": str(ads_path),
                **cal.settings_for_scope(settings, "rdl"),
            }
        )
    return rows


def plot_best(cal, best_settings: dict[str, float], samples: list[dict[str, object]]) -> list[str]:
    plot_dir = OUTPUT_DIR / "plots" / "rdl"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    ads_settings = {**cal.settings_for_scope(best_settings, "rdl"), **FREQ_SETTINGS}
    for sample in samples[: min(5, len(samples))]:
        ads_path = cal.RDL_ADS.simulate_single_device(
            "TMRDL",
            str(sample["sample_id"]),
            sample,
            ads_settings,
            output_base=OUTPUT_DIR / "ads_cache" / cal.settings_slug(best_settings, "rdl") / str(sample["sample_id"]),
            reuse_existing=True,
        )
        ads_nw = rf.Network(str(ads_path))
        hfss_nw = rf.Network(str(sample["hfss_path"]))
        freq_ghz = hfss_nw.f / 1e9
        nmse = cal.nmse_s11_s21_ri(hfss_nw.s, ads_nw.s)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
        fig.suptitle(f"RDL best | DUT {sample['dut_index']} | NMSE={nmse:.4g}", x=0.02, y=0.985, ha="left")
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
        out_path = plot_dir / f"RDL_{sample['dut_index']}_best_compare.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(str(out_path))
    return saved


def main():
    cal = load_base_module()
    cal.OUTPUT_DIR = OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = build_samples()
    rows = []
    candidates = rdl_candidates(cal)
    for idx, (label, settings) in enumerate(candidates, start=1):
        print(f"[rdl] {idx}/{len(candidates)} {label}", flush=True)
        rows.extend(evaluate_candidate(cal, label, settings, samples))

    detail = pd.DataFrame(rows)
    summary = cal.summarize(rows)
    best_row = summary.iloc[0]
    best_settings = dict(cal.BASE_SETTINGS)
    for key in cal.settings_for_scope(cal.BASE_SETTINGS, "rdl"):
        best_settings[key] = float(best_row[key])

    detail.to_csv(OUTPUT_DIR / "ads_rdl_calibration_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "ads_rdl_calibration_summary.csv", index=False, encoding="utf-8-sig")
    plot_files = plot_best(cal, best_settings, samples)

    best = {
        "source_dataset": "LHS400_Connection2",
        "source_split": "train",
        "random_seed": RANDOM_SEED,
        "sample_count": SAMPLE_COUNT,
        "sample_duts": [int(sample["dut_index"]) for sample in samples],
        "fixed_h_tsv_um": FIXED_H_TSV_UM,
        "ads_frequency_settings": FREQ_SETTINGS,
        "metric": "NMSE over flattened S11/S21 real and imaginary components",
        "best_rdl_settings": cal.settings_for_scope(best_settings, "rdl"),
        "plots": plot_files,
    }
    (OUTPUT_DIR / "best_ads_rdl_settings.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    archive = [
        "# LHS400_Connection2 Random-10 RDL ADS Calibration",
        "",
        "- ADS RDL template: updated MLIN netlist.",
        f"- Source: `{DATA_ROOT / 'RDL'}`",
        f"- Variation table: `{DATA_ROOT / 'RDL_variations_record.csv'}`",
        f"- Random seed: `{RANDOM_SEED}`",
        f"- Sample DUTs: `{best['sample_duts']}`",
        f"- Fixed RDL substrate height input `h_tsv`: `{FIXED_H_TSV_UM} um`",
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
    print("Best RDL settings:", json.dumps(cal.settings_for_scope(best_settings, "rdl"), ensure_ascii=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
