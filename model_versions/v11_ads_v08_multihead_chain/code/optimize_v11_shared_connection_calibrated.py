# -*- coding: utf-8 -*-
"""Optimize the v11 shared connection network with calibrated ADS devices.

Run this file directly in VS Code. No command-line arguments are required.

This script implements the current step in `建模流程.md`:
1. Read the LHS150_50_Connection2 HFSS full-chain target.
2. Build the 13-device ADS direct cascade with the random-30 calibrated
   single-device settings.
3. Optimize one 7-parameter connection circuit per sample and repeat that same
   circuit at all 12 connection positions.
4. Compare HFSS target, direct cascade, and optimized cascade.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
RUN_LABEL = "v11_sharedopt_c30"
FILTER_THRESHOLD = 0.1


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def db20(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def metric_row(base, target_s: np.ndarray, pred_s: np.ndarray) -> dict[str, float]:
    return {
        "mse_all_s": float(np.mean(np.abs(pred_s - target_s) ** 2)),
        "nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(target_s, pred_s),
        "mag_phase_mse_s11_s21": base.s11_s21_mag_phase_mse_np(target_s, pred_s),
        "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(target_s[:, 0, 0])))),
        "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(target_s[:, 1, 0])))),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split_name, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def summary_row(split_name: str, group: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "split": split_name,
        "count": int(len(group)),
        "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
        "optimized_nmse_mean": float(group["optimized_nmse_s11_s21_ri"].mean()),
        "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
        "optimized_nmse_median": float(group["optimized_nmse_s11_s21_ri"].median()),
        "optimized_better_count": int(
            (group["optimized_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
        ),
        "excluded_by_0p1_count": int(group["optimized_nmse_s11_s21_ri"].gt(FILTER_THRESHOLD).sum()),
        "direct_mse_mean": float(group["direct_mse_all_s"].mean()),
        "optimized_mse_mean": float(group["optimized_mse_all_s"].mean()),
        "direct_s11_db_mae_mean": float(group["direct_s11_db_mae"].mean()),
        "optimized_s11_db_mae_mean": float(group["optimized_s11_db_mae"].mean()),
        "direct_s21_db_mae_mean": float(group["direct_s21_db_mae"].mean()),
        "optimized_s21_db_mae_mean": float(group["optimized_s21_db_mae"].mean()),
    }


def plot_sparams(base, freq_ghz, target_s, direct_s, optimized_s, row: dict, out_path: Path) -> None:
    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{row['sample_id']} | direct NMSE={row['direct_nmse_s11_s21_ri']:.3e} | "
        f"optimized NMSE={row['optimized_nmse_s11_s21_ri']:.3e}",
        x=0.02,
        y=0.985,
        ha="left",
    )
    specs = [
        (0, 0, "S11 real", np.real),
        (0, 0, "S11 imag", np.imag),
        (1, 0, "S21 real", np.real),
        (1, 0, "S21 imag", np.imag),
    ]
    for ax, (m, n, title, component) in zip(axes.ravel(), specs):
        ax.plot(freq_ghz, component(target_s[:, m, n]), label="HFSS", color="black", linewidth=1.8)
        ax.plot(freq_ghz, component(direct_s[:, m, n]), label="ADS direct cascade", color="#64748b", linestyle=":")
        ax.plot(
            freq_ghz,
            component(optimized_s[:, m, n]),
            label="Optimized shared connection",
            color="#16a34a",
            linestyle="--",
        )
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    base.plt.close(fig)


def save_summary_plot(base, output_dir: Path, metrics: pd.DataFrame) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].scatter(
        metrics["direct_nmse_s11_s21_ri"],
        metrics["optimized_nmse_s11_s21_ri"],
        s=18,
        alpha=0.75,
    )
    max_nmse = float(max(metrics["direct_nmse_s11_s21_ri"].max(), metrics["optimized_nmse_s11_s21_ri"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linewidth=1.0, linestyle="--")
    axes[0].axhline(FILTER_THRESHOLD, color="#dc2626", linewidth=1.0, linestyle=":", label="0.1 filter")
    axes[0].set_xlabel("Direct cascade NMSE")
    axes[0].set_ylabel("Optimized shared-connection NMSE")
    axes[0].set_title("Direct vs optimized")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    metrics[["direct_nmse_s11_s21_ri", "optimized_nmse_s11_s21_ri"]].plot(kind="box", ax=axes[1])
    axes[1].set_title("NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "direct_vs_optimized_nmse_summary.png")
    base.plt.close(fig)


def calibrated_ads_settings() -> dict:
    sweep = {
        "freq_start_ghz": 0.1,
        "freq_stop_ghz": 100.0,
        "freq_step_ghz": 0.1,
    }
    return {
        "calibration_source": "ads_single_device_calibration_lhs400_connection2_random30",
        "dataset": "LHS150_50_Connection2",
        "target_design": "TSV_RDL",
        "single_device_calibration_dataset": "LHS400_Connection2/train",
        "connection_circuit": "v11_shared7_same_params_all_12",
        "rdl_settings": {
            "er_si": 9.8,
            "cond": 58_000_000.0,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.55,
            "pitch_scale": 1.25,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 1.0,
            **sweep,
        },
        "tsv_settings": {
            "er_si": 11.9,
            "cond": 58_000_000.0,
            "tand": 0.005,
            "c1_scale": 1.0,
            "pitch_scale": 1.1,
            "h_tsv_scale": 1.2,
            "d_scale": 1.0,
            **sweep,
        },
    }


def main() -> None:
    wrapper = load_module(WRAPPER_SCRIPT, "v11_shared7_wrapper_for_c30_opt")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_c30_opt")

    output_dir = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / RUN_LABEL
    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = output_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0
    base.OPT_MAX_NFEV = wrapper.OPT_MAX_NFEV
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = calibrated_ads_settings()
    dut_df = wrapper.collect_v11_samples(base)
    print(f"Samples: {len(dut_df)}", flush=True)
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
        direct = metric_row(base, target_s, direct_s)
        optimized = metric_row(base, target_s, optimized_s)
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
        plot_sparams(base, freq_ghz, target_s, direct_s, optimized_s, row, split_dir / f"{sample_id}.png")
        print(f"[compare] {i + 1}/{len(dut_df)} {sample_id}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    merged_opt = opt_summary.merge(
        metrics[["sample_id", "optimized_mse_all_s", "optimized_nmse_s11_s21_ri"]],
        on="sample_id",
        how="left",
        suffixes=("", "_comparison"),
    )
    excluded = metrics[metrics["optimized_nmse_s11_s21_ri"].gt(FILTER_THRESHOLD)].sort_values(
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
        plot_sparams(base, freq_ghz, target_s, direct_s, optimized_s, row, out_path)
        selected_paths.append(str(out_path))

    save_summary_plot(base, output_dir, metrics)
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
        "ads_settings": settings,
        "samples": int(len(dut_df)),
        "device_sequence": base.DEVICE_SEQUENCE,
        "connection_count": wrapper.CONNECTION_COUNT,
        "connection_mode": "one shared 7-parameter circuit per sample repeated at all 12 positions",
        "filter_threshold_nmse": FILTER_THRESHOLD,
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
                "# V11 Shared Connection Optimization",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                "- Dataset: `HFSS_sim/LHS150_50_Connection2/train|test/TSV_RDL`",
                "- ADS calibration: random-30 LHS400_Connection2 best settings.",
                f"- Structure: `{'-'.join(base.DEVICE_SEQUENCE)}`",
                f"- Connection mode: one optimized 7-parameter network per sample, repeated at all `{wrapper.CONNECTION_COUNT}` positions.",
                f"- Exclusion diagnostic: `optimized_nmse_s11_s21_ri > {FILTER_THRESHOLD}`.",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
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
                f"- Status: completed",
                f"- ADS cache: `{base.ADS_CACHE_DIR}`",
                f"- ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Samples compared: `{len(metrics)}`",
                f"- All-sample plots: `{len(list(all_plot_root.rglob('*.png')))}`",
                f"- Selected plots: `{len(selected_paths)}`",
                f"- Summary plot: `{output_dir / 'direct_vs_optimized_nmse_summary.png'}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
