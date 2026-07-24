# -*- coding: utf-8 -*-
"""Re-optimize v11 samples that became worse after the first shared fit.

Run this file directly in VS Code. No command-line arguments are required.

The first optimization in `optimize_v11_shared_connection_calibrated.py`
minimized all complex S-parameters. This continuation targets only the metric
used for judging the result: real/imag parts of S11 and S21.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


THIS_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
SOURCE_LABEL = "v11_sharedopt_c30"
RUN_LABEL = "v11_sharedopt_c30_reopt_worse"
MAX_NFEV_PER_START = 220
RANDOM_SEED = 20260710


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def s11_s21_ri_vector(s_params: np.ndarray) -> np.ndarray:
    s11 = s_params[:, 0, 0]
    s21 = s_params[:, 1, 0]
    return np.column_stack([s11.real, s11.imag, s21.real, s21.imag]).ravel()


def residual_s11_s21_ri(p, base, wrapper, base_abcds, target_s, omega, denom_scale):
    pred_s = base.abcd2s(wrapper.cascade_with_v08_np(base, base_abcds, omega, p))
    return (s11_s21_ri_vector(pred_s) - s11_s21_ri_vector(target_s)) / denom_scale


def candidate_starts(old_p: np.ndarray, rng: np.random.Generator, wrapper) -> list[np.ndarray]:
    starts = [
        old_p,
        wrapper.V08_P0,
        0.5 * old_p,
        1.5 * old_p,
        -old_p,
    ]
    scale = np.maximum(np.abs(old_p), 1.0)
    for sigma in [0.25, 0.75, 1.5]:
        for _ in range(2):
            starts.append(old_p + rng.normal(0.0, sigma, size=old_p.shape) * scale)
    clean = []
    seen = set()
    for p in starts:
        clipped = np.clip(np.asarray(p, dtype=np.float64), wrapper.V08_LOWER, wrapper.V08_UPPER)
        key = tuple(np.round(clipped, 10))
        if key not in seen:
            clean.append(clipped)
            seen.add(key)
    return clean


def plot_three(source, base, freq_ghz, target_s, direct_s, first_s, reopt_s, row: dict, out_path: Path) -> None:
    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{row['sample_id']} | direct={row['direct_nmse_s11_s21_ri']:.3e} | "
        f"first={row['first_nmse_s11_s21_ri']:.3e} | reopt={row['reopt_nmse_s11_s21_ri']:.3e}",
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
        ax.plot(freq_ghz, component(direct_s[:, m, n]), label="ADS direct", color="#64748b", linestyle=":")
        ax.plot(freq_ghz, component(first_s[:, m, n]), label="First optimized", color="#f97316", linestyle="--")
        ax.plot(freq_ghz, component(reopt_s[:, m, n]), label="Re-optimized", color="#16a34a", linestyle="-.")
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
    worse = metrics[metrics["was_reoptimized"]].copy()
    axes[0].scatter(worse["first_nmse_s11_s21_ri"], worse["reopt_nmse_s11_s21_ri"], s=24, alpha=0.8)
    max_nmse = float(max(worse["first_nmse_s11_s21_ri"].max(), worse["reopt_nmse_s11_s21_ri"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("First optimized NMSE")
    axes[0].set_ylabel("Re-optimized NMSE")
    axes[0].set_title("Re-optimized worsened samples")
    axes[0].grid(True, alpha=0.3)
    metrics[["direct_nmse_s11_s21_ri", "first_nmse_s11_s21_ri", "reopt_nmse_s11_s21_ri"]].plot(kind="box", ax=axes[1])
    axes[1].set_title("All-sample NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "reopt_worse_nmse_summary.png")
    base.plt.close(fig)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def summary_row(split: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split,
        "count": int(len(group)),
        "reoptimized_count": int(group["was_reoptimized"].sum()),
        "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
        "first_nmse_mean": float(group["first_nmse_s11_s21_ri"].mean()),
        "reopt_nmse_mean": float(group["reopt_nmse_s11_s21_ri"].mean()),
        "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
        "first_nmse_median": float(group["first_nmse_s11_s21_ri"].median()),
        "reopt_nmse_median": float(group["reopt_nmse_s11_s21_ri"].median()),
        "first_better_than_direct_count": int((group["first_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()),
        "reopt_better_than_direct_count": int((group["reopt_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()),
        "still_worse_than_direct_count": int((group["reopt_nmse_s11_s21_ri"] > group["direct_nmse_s11_s21_ri"]).sum()),
    }


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_reopt")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_shared7_wrapper_for_reopt")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_reopt")

    source_dir = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / SOURCE_LABEL
    output_dir = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)
    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0

    old_metrics = pd.read_csv(source_dir / "direct_vs_optimized_metrics.csv", encoding="utf-8-sig")
    old_targets = pd.read_csv(source_dir / "v08_shared_optimized_targets.csv", encoding="utf-8-sig")
    old_target_by_id = old_targets.set_index("sample_id")
    worse_ids = old_metrics.loc[
        old_metrics["optimized_nmse_s11_s21_ri"] > old_metrics["direct_nmse_s11_s21_ri"],
        "sample_id",
    ].tolist()
    print(f"Worsened samples to re-optimize: {len(worse_ids)}", flush=True)

    dut_df = wrapper.collect_v11_samples(base)
    sim = base.load_single_device_simulation(dut_df, source.calibrated_ads_settings())
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    rng = np.random.default_rng(RANDOM_SEED)

    reopt_param_rows = []
    attempt_rows = []
    final_metrics = []
    plot_dir = output_dir / "comparison_plots" / "reoptimized_worse"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        old_p = old_target_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        first_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, old_p))
        first_metric = source.metric_row(base, target_s, first_s)
        direct_metric = source.metric_row(base, target_s, direct_s)

        best_p = old_p.copy()
        best_s = first_s
        best_metric = first_metric
        best_start = -1
        starts = []
        if sample_id in worse_ids:
            y_true = s11_s21_ri_vector(target_s)
            denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2) / max(len(y_true), 1))
            denom = max(float(denom), 1e-30)
            starts = candidate_starts(old_p, rng, wrapper)
            for start_idx, p0 in enumerate(starts):
                res = least_squares(
                    residual_s11_s21_ri,
                    p0,
                    args=(base, wrapper, sim.base_abcds[i], target_s, omega, denom),
                    bounds=(wrapper.V08_LOWER, wrapper.V08_UPPER),
                    max_nfev=MAX_NFEV_PER_START,
                )
                cand_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, res.x))
                cand_metric = source.metric_row(base, target_s, cand_s)
                attempt_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": sample["split"],
                        "start_idx": start_idx,
                        "nfev": int(res.nfev),
                        "success": bool(res.success),
                        "nmse_s11_s21_ri": cand_metric["nmse_s11_s21_ri"],
                        "mse_all_s": cand_metric["mse_all_s"],
                    }
                )
                if cand_metric["nmse_s11_s21_ri"] < best_metric["nmse_s11_s21_ri"]:
                    best_p = res.x
                    best_s = cand_s
                    best_metric = cand_metric
                    best_start = start_idx
            row_for_plot = {
                "sample_id": sample_id,
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "first_nmse_s11_s21_ri": first_metric["nmse_s11_s21_ri"],
                "reopt_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
            }
            plot_three(source, base, freq_ghz, target_s, direct_s, first_s, best_s, row_for_plot, plot_dir / f"{sample_id}.png")
            print(
                f"[reopt] {sample_id}: direct={direct_metric['nmse_s11_s21_ri']:.3e}, "
                f"first={first_metric['nmse_s11_s21_ri']:.3e}, reopt={best_metric['nmse_s11_s21_ri']:.3e}",
                flush=True,
            )

        param_row = {"sample_id": sample_id, "split": sample["split"]}
        for name in base.STRUCTURE_COLUMNS:
            param_row[name] = float(sample[name])
        for param_idx, name in enumerate(wrapper.V08_PARAM_NAMES):
            param_row[name] = float(best_p[param_idx])
        reopt_param_rows.append(param_row)

        final_metrics.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "file": sample["file"],
                "dut_index": int(sample["dut_index"]),
                "was_reoptimized": sample_id in worse_ids,
                "best_start_idx": best_start,
                "start_count": len(starts),
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "first_nmse_s11_s21_ri": first_metric["nmse_s11_s21_ri"],
                "reopt_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
                "direct_mse_all_s": direct_metric["mse_all_s"],
                "first_mse_all_s": first_metric["mse_all_s"],
                "reopt_mse_all_s": best_metric["mse_all_s"],
                "direct_s11_db_mae": direct_metric["s11_db_mae"],
                "first_s11_db_mae": first_metric["s11_db_mae"],
                "reopt_s11_db_mae": best_metric["s11_db_mae"],
                "direct_s21_db_mae": direct_metric["s21_db_mae"],
                "first_s21_db_mae": first_metric["s21_db_mae"],
                "reopt_s21_db_mae": best_metric["s21_db_mae"],
            }
        )

    metrics = pd.DataFrame(final_metrics)
    summary = summarize(metrics)
    reopt_targets = pd.DataFrame(reopt_param_rows)
    attempts = pd.DataFrame(attempt_rows)
    save_summary_plot(base, output_dir, metrics)
    metrics.to_csv(output_dir / "reopt_direct_first_reopt_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "reopt_direct_first_reopt_summary.csv", index=False, encoding="utf-8-sig")
    reopt_targets.to_csv(output_dir / "v08_shared_reoptimized_targets.csv", index=False, encoding="utf-8-sig")
    attempts.to_csv(output_dir / "reoptimization_attempts.csv", index=False, encoding="utf-8-sig")
    remaining_worse = metrics[metrics["reopt_nmse_s11_s21_ri"] > metrics["direct_nmse_s11_s21_ri"]]
    remaining_worse.to_csv(output_dir / "still_worse_than_direct_after_reopt.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "source_result_dir": str(source_dir),
        "output_dir": str(output_dir),
        "reoptimized_sample_count": len(worse_ids),
        "max_nfev_per_start": MAX_NFEV_PER_START,
        "random_seed": RANDOM_SEED,
        "objective": "least_squares on normalized real/imag S11 and S21 residuals",
        "summary": summary.to_dict(orient="records"),
        "remaining_worse_count": int(len(remaining_worse)),
    }
    (output_dir / "reoptimization_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "reoptimization_report.md").write_text(
        "\n".join(
            [
                "# V11 Re-Optimization For Worsened Samples",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Source result: `{source_dir}`",
                f"- Output: `{output_dir}`",
                f"- Re-optimized samples: `{len(worse_ids)}`",
                f"- Objective: normalized real/imag `S11` and `S21` residuals.",
                f"- Max evaluations per start: `{MAX_NFEV_PER_START}`",
                "",
                "## Summary",
                "",
                source.dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Metrics: `{output_dir / 'reopt_direct_first_reopt_metrics.csv'}`",
                f"- Summary: `{output_dir / 'reopt_direct_first_reopt_summary.csv'}`",
                f"- Re-optimized targets: `{output_dir / 'v08_shared_reoptimized_targets.csv'}`",
                f"- Attempts: `{output_dir / 'reoptimization_attempts.csv'}`",
                f"- Remaining worse samples: `{output_dir / 'still_worse_than_direct_after_reopt.csv'}`",
                f"- Plots: `{plot_dir}`",
                f"- Summary plot: `{output_dir / 'reopt_worse_nmse_summary.png'}`",
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
                f"- Source result: `{source_dir}`",
                f"- ADS cache reused: `{base.ADS_CACHE_DIR}`",
                f"- Re-optimized samples: `{len(worse_ids)}`",
                f"- Re-optimization plots: `{len(list(plot_dir.glob('*.png')))}`",
                f"- Attempt rows: `{len(attempts)}`",
                f"- Remaining worse than direct: `{len(remaining_worse)}`",
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
