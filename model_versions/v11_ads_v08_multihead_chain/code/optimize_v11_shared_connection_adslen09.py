# -*- coding: utf-8 -*-
"""Optimize v11 shared connection circuits with ADS RDL/TSV length scale 0.9.

Run this file directly in VS Code. No command-line arguments are required.

This entry repeats the calibrated v11 shared-connection optimization while
forcing the ADS single-device geometry scale to 0.9. In the v11 ADS base runner
that scale is applied to `l_tmrdl`, `l_bsmrdl`, and `h_tsv` before ADS
simulation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
RUN_LABEL = "v11_sharedopt_c30_adslen09"
ADS_DEVICE_LENGTH_SCALE = 0.9


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_adslen09")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_shared7_wrapper_for_adslen09")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_adslen09")

    output_dir = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / RUN_LABEL
    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = output_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = ADS_DEVICE_LENGTH_SCALE
    base.OPT_MAX_NFEV = wrapper.OPT_MAX_NFEV
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied by train_ads_pi_cascade_v11_base.scale_structure_for_ads to l_tmrdl, l_bsmrdl, and h_tsv."

    dut_df = wrapper.collect_v11_samples(base)
    print(f"Samples: {len(dut_df)}", flush=True)
    print(f"ADS_DEVICE_LENGTH_SCALE: {base.ADS_DEVICE_LENGTH_SCALE}", flush=True)
    sim = base.load_single_device_simulation(dut_df, settings)
    opt_summary, shared_targets = wrapper.optimize_v08_shared_targets(base, dut_df, sim)

    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    target_by_id = shared_targets.set_index("sample_id")
    rows = []
    cache = {}
    all_plot_root = output_dir / "comparison_plots" / "all_samples"
    all_plot_root.mkdir(parents=True, exist_ok=True)

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
        split_dir = all_plot_root / str(sample["split"])
        split_dir.mkdir(parents=True, exist_ok=True)
        source.plot_sparams(base, freq_ghz, target_s, direct_s, optimized_s, row, split_dir / f"{sample_id}.png")
        print(f"[compare] {i + 1}/{len(dut_df)} {sample_id}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = source.summarize(metrics)
    merged_opt = opt_summary.merge(
        metrics[["sample_id", "optimized_mse_all_s", "optimized_nmse_s11_s21_ri"]],
        on="sample_id",
        how="left",
        suffixes=("", "_comparison"),
    )
    excluded = metrics[metrics["optimized_nmse_s11_s21_ri"].gt(source.FILTER_THRESHOLD)].sort_values(
        "optimized_nmse_s11_s21_ri",
        ascending=False,
    )

    selected_dir = output_dir / "comparison_plots" / "selected_test"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.concat(
        [
            metrics[metrics["split"].eq("test")].sort_values("optimized_nmse_s11_s21_ri").head(6),
            metrics[metrics["split"].eq("test")].sort_values("optimized_nmse_s11_s21_ri", ascending=False).head(6),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    selected_paths = []
    for _, metric in selected.iterrows():
        target_s, direct_s, optimized_s, row = cache[str(metric["sample_id"])]
        out_path = selected_dir / f"{metric['sample_id']}.png"
        source.plot_sparams(base, freq_ghz, target_s, direct_s, optimized_s, row, out_path)
        selected_paths.append(str(out_path))

    source.save_summary_plot(base, output_dir, metrics)
    opt_summary.to_csv(output_dir / "v08_shared_optimization_summary.csv", index=False, encoding="utf-8-sig")
    shared_targets.to_csv(output_dir / "v08_shared_optimized_targets.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "direct_vs_optimized_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "direct_vs_optimized_summary.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(output_dir / "excluded_by_optimized_nmse_gt_0p1.csv", index=False, encoding="utf-8-sig")
    merged_opt.to_csv(output_dir / "optimization_metrics_joined.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "workflow_source": "建模流程.md",
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "ads_geometry_scale_note": settings["ads_geometry_scale_note"],
        "ads_settings": settings,
        "samples": int(len(dut_df)),
        "device_sequence": base.DEVICE_SEQUENCE,
        "connection_count": wrapper.CONNECTION_COUNT,
        "connection_mode": "one shared 7-parameter circuit per sample repeated at all 12 positions",
        "filter_threshold_nmse": source.FILTER_THRESHOLD,
        "excluded_by_threshold": excluded.groupby("split").size().to_dict() if len(excluded) else {},
        "summary": summary.to_dict(orient="records"),
        "selected_plots": selected_paths,
        "metric_definition": {
            "nmse_s11_s21_ri": "sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)",
            "y": "flattened [real(S11), imag(S11), real(S21), imag(S21)] over all frequency points",
        },
    }
    (output_dir / "optimization_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "optimization_report.md").write_text(
        "\n".join(
            [
                "# V11 Shared Connection Optimization ADS Length 0.9",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                "- Dataset: `HFSS_sim/LHS150_50_Connection2/train|test/TSV_RDL`",
                "- ADS calibration: random-30 LHS400_Connection2 best settings.",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}` applied to `l_tmrdl`, `l_bsmrdl`, and `h_tsv`.",
                f"- Structure: `{'-'.join(base.DEVICE_SEQUENCE)}`",
                f"- Connection mode: one optimized 7-parameter network per sample, repeated at all `{wrapper.CONNECTION_COUNT}` positions.",
                f"- Exclusion diagnostic: `optimized_nmse_s11_s21_ri > {source.FILTER_THRESHOLD}`.",
                "",
                "## Summary",
                "",
                source.dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Optimized parameter targets: `{output_dir / 'v08_shared_optimized_targets.csv'}`",
                f"- Per-sample comparison metrics: `{output_dir / 'direct_vs_optimized_metrics.csv'}`",
                f"- Summary CSV: `{output_dir / 'direct_vs_optimized_summary.csv'}`",
                f"- Excluded diagnostic CSV: `{output_dir / 'excluded_by_optimized_nmse_gt_0p1.csv'}`",
                f"- Summary plot: `{output_dir / 'direct_vs_optimized_nmse_summary.png'}`",
                f"- All comparison plots: `{all_plot_root}`",
                f"- Selected test plots: `{selected_dir}`",
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
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}`",
                f"- ADS cache: `{base.ADS_CACHE_DIR}`",
                f"- ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Optimization JSON files: `{len(list((output_dir / 'v08_shared_sample_optimization').glob('*.json')))}`",
                f"- Samples compared: `{len(metrics)}`",
                f"- All-sample plots: `{len(list(all_plot_root.rglob('*.png')))}`",
                f"- Selected plots: `{len(selected_paths)}`",
                f"- Summary plot: `{output_dir / 'direct_vs_optimized_nmse_summary.png'}`",
                "",
                "## Summary",
                "",
                source.dataframe_to_markdown(summary),
            ]
        ),
        encoding="utf-8",
    )
    print(source.dataframe_to_markdown(summary), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
