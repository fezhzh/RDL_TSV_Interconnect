# -*- coding: utf-8 -*-
"""Optimize v11 shared connection circuits on LHS400 with positive parameters.

Run this file directly in VS Code. No command-line arguments are required.

This entry uses `HFSS_sim/LHS400_Connection2/train/TSV_RDL` targets that exist
on disk, cascades the 13 ADS device blocks, and optimizes one shared
7-parameter connection circuit repeated at all 12 connection positions. All
connection-circuit scale parameters are constrained to be positive.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
RUN_LABEL = "v11_positive_sharedopt_lhs400_connection2_adslen09"
DATASET_NAME = "LHS400_Connection2"
TARGET_DESIGN_NAME = "TSV_RDL"
ADS_DEVICE_LENGTH_SCALE = 0.9
POSITIVE_LOWER = 1e-9
POSITIVE_UPPER = 1e5
FILTER_THRESHOLD = 0.1
SELECTED_PLOT_COUNT = 12
ADS_SIM_RETRIES = 5


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_lhs400_rdl_tsv_samples(base) -> pd.DataFrame:
    dataset_root = base.PROJECT_ROOT / "HFSS_sim" / DATASET_NAME / "train"
    csv_path = dataset_root / f"{TARGET_DESIGN_NAME}_variations_record.csv"
    snp_dir = dataset_root / TARGET_DESIGN_NAME
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not snp_dir.exists():
        raise FileNotFoundError(snp_dir)

    df = pd.read_csv(csv_path, encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        dut_index = int(row["dut_index"])
        snp_path = snp_dir / f"dut{dut_index}.s2p"
        if not snp_path.exists():
            continue
        rec = row.to_dict()
        rec["dut_index"] = dut_index
        if "t_tmrdl" in rec and "h_tmrdl" not in rec:
            rec["h_tmrdl"] = float(rec.pop("t_tmrdl"))
        if "t_bsmrdl" in rec and "h_bsmrdl" not in rec:
            rec["h_bsmrdl"] = float(rec.pop("t_bsmrdl"))
        rec["split"] = "train"
        rec["source_root"] = DATASET_NAME
        rec["sample_id"] = f"{DATASET_NAME}_train_dut{dut_index}"
        rec["file"] = f"dut{dut_index}.s2p"
        rec["snp_path"] = snp_path
        rows.append(rec)
    out = pd.DataFrame(rows)
    missing = [col for col in base.STRUCTURE_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"{TARGET_DESIGN_NAME} sample table is missing columns: {missing}")
    if not out.empty:
        val_count = max(1, len(out) // 10)
        out.loc[out.index[-val_count:], "split"] = "val"
    return out.reset_index(drop=True)


def parameter_sign_summary(targets: pd.DataFrame, wrapper) -> pd.DataFrame:
    rows = []
    for name in wrapper.V08_PARAM_NAMES:
        values = targets[name].to_numpy(dtype=np.float64)
        rows.append(
            {
                "parameter": name,
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "nonpositive_count": int(np.sum(values <= 0.0)),
                "total_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def load_single_device_simulation_with_retries(base, dut_df: pd.DataFrame, settings: dict):
    last_error = None
    for attempt in range(1, ADS_SIM_RETRIES + 1):
        try:
            return base.load_single_device_simulation(dut_df, settings)
        except RuntimeError as exc:
            last_error = exc
            print(f"[ads-retry] attempt {attempt}/{ADS_SIM_RETRIES} failed: {exc}", flush=True)
            if attempt < ADS_SIM_RETRIES:
                time.sleep(3.0)
    raise RuntimeError(f"ADS single-device simulation failed after {ADS_SIM_RETRIES} attempts") from last_error


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_positive_lhs400_source")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_positive_lhs400_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_positive_lhs400_base")

    output_dir = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / RUN_LABEL
    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = output_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = ADS_DEVICE_LENGTH_SCALE
    base.OPT_MAX_NFEV = wrapper.OPT_MAX_NFEV
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper.V08_LOWER = np.full(len(wrapper.V08_PARAM_NAMES), POSITIVE_LOWER, dtype=np.float64)
    wrapper.V08_UPPER = np.full(len(wrapper.V08_PARAM_NAMES), POSITIVE_UPPER, dtype=np.float64)
    wrapper.V08_P0 = np.ones(len(wrapper.V08_PARAM_NAMES), dtype=np.float64)

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied to l_tmrdl, l_bsmrdl, and h_tsv before ADS simulation."

    dut_df = collect_lhs400_rdl_tsv_samples(base)
    print(f"Samples with existing RDL_TSV S2P: {len(dut_df)}", flush=True)
    print(f"Positive bounds: [{POSITIVE_LOWER}, {POSITIVE_UPPER}]", flush=True)
    sim = load_single_device_simulation_with_retries(base, dut_df, settings)
    opt_summary, shared_targets = wrapper.optimize_v08_shared_targets(base, dut_df, sim)

    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    target_by_id = shared_targets.set_index("sample_id")
    rows = []
    cache = {}
    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        params = target_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        optimized_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, params))
        direct = source.metric_row(base, target_s, direct_s)
        optimized = source.metric_row(base, target_s, optimized_s)
        row = {
            "sample_id": sample_id,
            "split": sample["split"],
            "file": sample["file"],
            "dut_index": int(sample["dut_index"]),
            "direct_mse_all_s": direct["mse_all_s"],
            "optimized_mse_all_s": optimized["mse_all_s"],
            "direct_nmse_s11_s21_ri": direct["nmse_s11_s21_ri"],
            "optimized_nmse_s11_s21_ri": optimized["nmse_s11_s21_ri"],
            "direct_mag_phase_mse_s11_s21": direct["mag_phase_mse_s11_s21"],
            "optimized_mag_phase_mse_s11_s21": optimized["mag_phase_mse_s11_s21"],
            "direct_s11_db_mae": direct["s11_db_mae"],
            "optimized_s11_db_mae": optimized["s11_db_mae"],
            "direct_s21_db_mae": direct["s21_db_mae"],
            "optimized_s21_db_mae": optimized["s21_db_mae"],
        }
        rows.append(row)
        cache[sample_id] = (target_s, direct_s, optimized_s, row)
        print(f"[compare] {i + 1}/{len(dut_df)} {sample_id}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = source.summarize(metrics)
    sign_summary = parameter_sign_summary(shared_targets, wrapper)
    excluded = metrics[metrics["optimized_nmse_s11_s21_ri"].gt(FILTER_THRESHOLD)].sort_values(
        "optimized_nmse_s11_s21_ri",
        ascending=False,
    )
    joined = opt_summary.merge(
        metrics[["sample_id", "optimized_mse_all_s", "optimized_nmse_s11_s21_ri"]],
        on="sample_id",
        how="left",
        suffixes=("", "_comparison"),
    )

    source.save_summary_plot(base, output_dir, metrics)
    selected_dir = output_dir / "comparison_plots" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.concat(
        [
            metrics.sort_values("optimized_nmse_s11_s21_ri").head(SELECTED_PLOT_COUNT // 2),
            metrics.sort_values("optimized_nmse_s11_s21_ri", ascending=False).head(SELECTED_PLOT_COUNT // 2),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    selected_paths = []
    for _, metric in selected.iterrows():
        target_s, direct_s, optimized_s, row = cache[str(metric["sample_id"])]
        out_path = selected_dir / f"{metric['sample_id']}.png"
        source.plot_sparams(base, freq_ghz, target_s, direct_s, optimized_s, row, out_path)
        selected_paths.append(str(out_path))

    opt_summary.to_csv(output_dir / "v08_positive_shared_optimization_summary.csv", index=False, encoding="utf-8-sig")
    shared_targets.to_csv(output_dir / "v08_positive_shared_optimized_targets.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "direct_vs_positive_optimized_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "direct_vs_positive_optimized_summary.csv", index=False, encoding="utf-8-sig")
    sign_summary.to_csv(output_dir / "positive_parameter_sign_summary.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(output_dir / "positive_optimized_nmse_gt_0p1.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(output_dir / "positive_optimization_metrics_joined.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "dataset": str(base.PROJECT_ROOT / "HFSS_sim" / DATASET_NAME / "train" / TARGET_DESIGN_NAME),
        "csv_rows": int(len(pd.read_csv(base.PROJECT_ROOT / "HFSS_sim" / DATASET_NAME / "train" / f"{TARGET_DESIGN_NAME}_variations_record.csv"))),
        "samples_with_existing_s2p": int(len(dut_df)),
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "positive_bounds": {"lower": POSITIVE_LOWER, "upper": POSITIVE_UPPER},
        "connection_mode": f"one positive shared 7-parameter circuit repeated at all {wrapper.CONNECTION_COUNT} positions",
        "device_sequence": base.DEVICE_SEQUENCE,
        "summary": summary.to_dict(orient="records"),
        "parameter_sign_summary": sign_summary.to_dict(orient="records"),
        "selected_plots": selected_paths,
    }
    (output_dir / "positive_optimization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "positive_optimization_report.md").write_text(
        "\n".join(
            [
                "# V11 Positive Shared Connection Optimization on LHS400 Connection2",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Dataset: `HFSS_sim/{DATASET_NAME}/train/{TARGET_DESIGN_NAME}`",
                f"- CSV rows: `{report['csv_rows']}`",
                f"- Existing `.s2p` samples optimized: `{len(dut_df)}`",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}`",
                f"- Positive parameter bounds: `[{POSITIVE_LOWER}, {POSITIVE_UPPER}]`",
                f"- Connection mode: `{report['connection_mode']}`",
                "",
                "## Summary",
                "",
                source.dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                source.dataframe_to_markdown(sign_summary),
                "",
                "## Outputs",
                "",
                f"- Optimized parameter targets: `{output_dir / 'v08_positive_shared_optimized_targets.csv'}`",
                f"- Per-sample metrics: `{output_dir / 'direct_vs_positive_optimized_metrics.csv'}`",
                f"- Summary CSV: `{output_dir / 'direct_vs_positive_optimized_summary.csv'}`",
                f"- Parameter sign summary: `{output_dir / 'positive_parameter_sign_summary.csv'}`",
                f"- Summary plot: `{output_dir / 'direct_vs_optimized_nmse_summary.png'}`",
                f"- Selected plots: `{selected_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "validation_archive.md").write_text(
        "\n".join(
            [
                "# Validation Archive",
                "",
                f"- Entry: `{Path(__file__).name}`",
                "- Status: completed",
                f"- Dataset: `HFSS_sim/{DATASET_NAME}/train/{TARGET_DESIGN_NAME}`",
                f"- CSV rows: `{report['csv_rows']}`",
                f"- Existing `.s2p` samples optimized: `{len(dut_df)}`",
                f"- ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Optimization JSON files: `{len(list((output_dir / 'v08_shared_sample_optimization').glob('*.json')))}`",
                f"- Selected plots: `{len(selected_paths)}`",
                f"- Positive sign check nonpositive total: `{int(sign_summary['nonpositive_count'].sum())}`",
                "",
                "## Summary",
                "",
                source.dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                source.dataframe_to_markdown(sign_summary),
            ]
        ),
        encoding="utf-8",
    )
    print(source.dataframe_to_markdown(summary), flush=True)
    print(source.dataframe_to_markdown(sign_summary), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
