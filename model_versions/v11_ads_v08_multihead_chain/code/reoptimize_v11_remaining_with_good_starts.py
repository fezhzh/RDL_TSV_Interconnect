# -*- coding: utf-8 -*-
"""Use well-optimized samples as initial values for remaining bad v11 samples.

Run this file directly in VS Code. No command-line arguments are required.

This continuation tests whether the remaining poor fits are caused by bad
initial values. It only targets samples that are still worse than direct cascade
after `reoptimize_v11_worse_shared_connection.py`.
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
REOPT_SCRIPT = THIS_DIR / "reoptimize_v11_worse_shared_connection.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
FIRST_LABEL = "v11_sharedopt_c30"
REOPT_LABEL = "v11_sharedopt_c30_reopt_worse"
RUN_LABEL = "v11_sharedopt_c30_goodstart_remaining"
MAX_NFEV_PER_START = 260
GLOBAL_GOOD_STARTS = 24
NEAREST_GOOD_STARTS = 24


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


def geometry_distances(sample_row: pd.Series, good_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    values = good_df[feature_cols].to_numpy(dtype=np.float64)
    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-12)
    target = sample_row[feature_cols].to_numpy(dtype=np.float64)
    return pd.Series(np.linalg.norm(((values - center) / scale) - ((target - center) / scale), axis=1), index=good_df.index)


def build_good_start_list(
    sample_id: str,
    sample_row: pd.Series,
    good_df: pd.DataFrame,
    target_by_id: pd.DataFrame,
    wrapper,
    current_p: np.ndarray,
    first_p: np.ndarray,
    feature_cols: list[str],
) -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = [
        ("current_reopt", current_p),
        ("first_pass", first_p),
        ("unit", wrapper.V08_P0),
    ]
    global_ids = (
        good_df.sort_values("reopt_nmse_s11_s21_ri")
        .head(GLOBAL_GOOD_STARTS)["sample_id"]
        .tolist()
    )
    distances = geometry_distances(sample_row, good_df, feature_cols)
    nearest_ids = good_df.loc[distances.sort_values().head(NEAREST_GOOD_STARTS).index, "sample_id"].tolist()
    for source_id in global_ids:
        if source_id != sample_id:
            starts.append((f"global_good:{source_id}", target_by_id.loc[source_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)))
    for source_id in nearest_ids:
        if source_id != sample_id:
            starts.append((f"nearest_good:{source_id}", target_by_id.loc[source_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)))

    clean: list[tuple[str, np.ndarray]] = []
    seen = set()
    for label, p in starts:
        clipped = np.clip(np.asarray(p, dtype=np.float64), wrapper.V08_LOWER, wrapper.V08_UPPER)
        key = tuple(np.round(clipped, 10))
        if key not in seen:
            clean.append((label, clipped))
            seen.add(key)
    return clean


def plot_transfer(base, freq_ghz, target_s, direct_s, first_s, reopt_s, transfer_s, row: dict, out_path: Path) -> None:
    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{row['sample_id']} | direct={row['direct_nmse_s11_s21_ri']:.3e} | "
        f"reopt={row['previous_reopt_nmse_s11_s21_ri']:.3e} | good-start={row['goodstart_nmse_s11_s21_ri']:.3e}",
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
        ax.plot(freq_ghz, component(reopt_s[:, m, n]), label="Previous reopt", color="#7c3aed", linestyle="-.")
        ax.plot(freq_ghz, component(transfer_s[:, m, n]), label="Good-start reopt", color="#16a34a", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    base.plt.close(fig)


def summary_row(split: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split,
        "count": int(len(group)),
        "goodstart_target_count": int(group["was_goodstart_target"].sum()),
        "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
        "previous_reopt_nmse_mean": float(group["previous_reopt_nmse_s11_s21_ri"].mean()),
        "goodstart_nmse_mean": float(group["goodstart_nmse_s11_s21_ri"].mean()),
        "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
        "previous_reopt_nmse_median": float(group["previous_reopt_nmse_s11_s21_ri"].median()),
        "goodstart_nmse_median": float(group["goodstart_nmse_s11_s21_ri"].median()),
        "previous_reopt_better_than_direct_count": int(
            (group["previous_reopt_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
        ),
        "goodstart_better_than_direct_count": int(
            (group["goodstart_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
        ),
        "still_worse_than_direct_count": int(
            (group["goodstart_nmse_s11_s21_ri"] > group["direct_nmse_s11_s21_ri"]).sum()
        ),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def save_summary_plot(base, output_dir: Path, metrics: pd.DataFrame) -> None:
    target = metrics[metrics["was_goodstart_target"]]
    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].scatter(
        target["previous_reopt_nmse_s11_s21_ri"],
        target["goodstart_nmse_s11_s21_ri"],
        s=35,
        alpha=0.85,
    )
    max_nmse = float(max(target["previous_reopt_nmse_s11_s21_ri"].max(), target["goodstart_nmse_s11_s21_ri"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Previous reopt NMSE")
    axes[0].set_ylabel("Good-start NMSE")
    axes[0].set_title("Remaining bad samples")
    axes[0].grid(True, alpha=0.3)
    target[["direct_nmse_s11_s21_ri", "previous_reopt_nmse_s11_s21_ri", "goodstart_nmse_s11_s21_ri"]].plot(
        kind="box",
        ax=axes[1],
    )
    axes[1].set_title("Target-sample NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "goodstart_remaining_nmse_summary.png")
    base.plt.close(fig)


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_goodstarts")
    reopt = load_module(REOPT_SCRIPT, "v11_reopt_for_goodstarts")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_shared7_wrapper_for_goodstarts")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_goodstarts")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    first_dir = version_root / "results" / FIRST_LABEL
    reopt_dir = version_root / "results" / REOPT_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = first_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0

    first_targets = pd.read_csv(first_dir / "v08_shared_optimized_targets.csv", encoding="utf-8-sig")
    reopt_targets = pd.read_csv(reopt_dir / "v08_shared_reoptimized_targets.csv", encoding="utf-8-sig")
    prev_metrics = pd.read_csv(reopt_dir / "reopt_direct_first_reopt_metrics.csv", encoding="utf-8-sig")
    remaining = pd.read_csv(reopt_dir / "still_worse_than_direct_after_reopt.csv", encoding="utf-8-sig")
    target_ids = set(remaining["sample_id"])
    print(f"Remaining samples to good-start optimize: {len(target_ids)}", flush=True)

    feature_cols = list(base.STRUCTURE_COLUMNS)
    target_by_id = reopt_targets.set_index("sample_id")
    first_by_id = first_targets.set_index("sample_id")
    good_df = (
        prev_metrics[prev_metrics["reopt_nmse_s11_s21_ri"].lt(prev_metrics["direct_nmse_s11_s21_ri"])]
        .merge(reopt_targets[["sample_id", *feature_cols]], on="sample_id", how="left")
        .reset_index(drop=True)
    )

    dut_df = wrapper.collect_v11_samples(base)
    sim = base.load_single_device_simulation(dut_df, source.calibrated_ads_settings())
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9

    rows = []
    attempts = []
    transfer_targets = []
    plot_dir = output_dir / "comparison_plots" / "goodstart_remaining"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        first_p = first_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        current_p = target_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        first_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, first_p))
        prev_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, current_p))
        direct_metric = source.metric_row(base, target_s, direct_s)
        first_metric = source.metric_row(base, target_s, first_s)
        prev_metric = source.metric_row(base, target_s, prev_s)
        best_p = current_p
        best_s = prev_s
        best_metric = prev_metric
        best_label = "previous_reopt"
        start_count = 0

        if sample_id in target_ids:
            y_true = s11_s21_ri_vector(target_s)
            denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2) / max(len(y_true), 1))
            denom = max(float(denom), 1e-30)
            start_list = build_good_start_list(
                sample_id,
                sample,
                good_df,
                target_by_id,
                wrapper,
                current_p,
                first_p,
                feature_cols,
            )
            start_count = len(start_list)
            for start_idx, (label, p0) in enumerate(start_list):
                direct_start_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, p0))
                direct_start_metric = source.metric_row(base, target_s, direct_start_s)
                res = least_squares(
                    residual_s11_s21_ri,
                    p0,
                    args=(base, wrapper, sim.base_abcds[i], target_s, omega, denom),
                    bounds=(wrapper.V08_LOWER, wrapper.V08_UPPER),
                    max_nfev=MAX_NFEV_PER_START,
                )
                cand_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, res.x))
                cand_metric = source.metric_row(base, target_s, cand_s)
                attempts.append(
                    {
                        "sample_id": sample_id,
                        "split": sample["split"],
                        "start_idx": start_idx,
                        "start_label": label,
                        "start_nmse_s11_s21_ri": direct_start_metric["nmse_s11_s21_ri"],
                        "final_nmse_s11_s21_ri": cand_metric["nmse_s11_s21_ri"],
                        "final_mse_all_s": cand_metric["mse_all_s"],
                        "nfev": int(res.nfev),
                        "success": bool(res.success),
                    }
                )
                if cand_metric["nmse_s11_s21_ri"] < best_metric["nmse_s11_s21_ri"]:
                    best_p = res.x
                    best_s = cand_s
                    best_metric = cand_metric
                    best_label = label
            plot_row = {
                "sample_id": sample_id,
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "previous_reopt_nmse_s11_s21_ri": prev_metric["nmse_s11_s21_ri"],
                "goodstart_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
            }
            plot_transfer(base, freq_ghz, target_s, direct_s, first_s, prev_s, best_s, plot_row, plot_dir / f"{sample_id}.png")
            print(
                f"[good-start] {sample_id}: direct={direct_metric['nmse_s11_s21_ri']:.3e}, "
                f"prev={prev_metric['nmse_s11_s21_ri']:.3e}, best={best_metric['nmse_s11_s21_ri']:.3e}, "
                f"start={best_label}",
                flush=True,
            )

        param_row = {"sample_id": sample_id, "split": sample["split"]}
        for col in feature_cols:
            param_row[col] = float(sample[col])
        for p_idx, name in enumerate(wrapper.V08_PARAM_NAMES):
            param_row[name] = float(best_p[p_idx])
        transfer_targets.append(param_row)
        rows.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "file": sample["file"],
                "dut_index": int(sample["dut_index"]),
                "was_goodstart_target": sample_id in target_ids,
                "start_count": start_count,
                "best_start_label": best_label,
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "first_nmse_s11_s21_ri": first_metric["nmse_s11_s21_ri"],
                "previous_reopt_nmse_s11_s21_ri": prev_metric["nmse_s11_s21_ri"],
                "goodstart_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
                "direct_mse_all_s": direct_metric["mse_all_s"],
                "previous_reopt_mse_all_s": prev_metric["mse_all_s"],
                "goodstart_mse_all_s": best_metric["mse_all_s"],
                "direct_s11_db_mae": direct_metric["s11_db_mae"],
                "previous_reopt_s11_db_mae": prev_metric["s11_db_mae"],
                "goodstart_s11_db_mae": best_metric["s11_db_mae"],
                "direct_s21_db_mae": direct_metric["s21_db_mae"],
                "previous_reopt_s21_db_mae": prev_metric["s21_db_mae"],
                "goodstart_s21_db_mae": best_metric["s21_db_mae"],
            }
        )

    metrics = pd.DataFrame(rows)
    attempts_df = pd.DataFrame(attempts)
    targets_df = pd.DataFrame(transfer_targets)
    summary = summarize(metrics)
    remaining_after = metrics[metrics["goodstart_nmse_s11_s21_ri"].gt(metrics["direct_nmse_s11_s21_ri"])]
    save_summary_plot(base, output_dir, metrics)

    metrics.to_csv(output_dir / "goodstart_direct_first_prev_final_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "goodstart_remaining_summary.csv", index=False, encoding="utf-8-sig")
    attempts_df.to_csv(output_dir / "goodstart_attempts.csv", index=False, encoding="utf-8-sig")
    targets_df.to_csv(output_dir / "v08_shared_goodstart_targets.csv", index=False, encoding="utf-8-sig")
    remaining_after.to_csv(output_dir / "still_worse_than_direct_after_goodstart.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "source_first_result_dir": str(first_dir),
        "source_reopt_result_dir": str(reopt_dir),
        "output_dir": str(output_dir),
        "target_sample_count": len(target_ids),
        "good_start_pool_count": int(len(good_df)),
        "global_good_starts": GLOBAL_GOOD_STARTS,
        "nearest_good_starts": NEAREST_GOOD_STARTS,
        "max_nfev_per_start": MAX_NFEV_PER_START,
        "summary": summary.to_dict(orient="records"),
        "remaining_worse_count": int(len(remaining_after)),
    }
    (output_dir / "goodstart_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "goodstart_report.md").write_text(
        "\n".join(
            [
                "# V11 Good-Sample Initial-Value Re-Optimization",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- First-pass result: `{first_dir}`",
                f"- Previous re-optimization result: `{reopt_dir}`",
                f"- Output: `{output_dir}`",
                f"- Target samples: `{len(target_ids)}`",
                f"- Good-start pool: `{len(good_df)}` samples that are already better than direct cascade.",
                f"- Starts per target: previous reopt, first pass, unit vector, top `{GLOBAL_GOOD_STARTS}` global good samples, and `{NEAREST_GOOD_STARTS}` nearest-geometry good samples.",
                f"- Objective: normalized real/imag `S11` and `S21` residuals.",
                "",
                "## Summary",
                "",
                source.dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Metrics: `{output_dir / 'goodstart_direct_first_prev_final_metrics.csv'}`",
                f"- Summary: `{output_dir / 'goodstart_remaining_summary.csv'}`",
                f"- Attempts: `{output_dir / 'goodstart_attempts.csv'}`",
                f"- Final targets: `{output_dir / 'v08_shared_goodstart_targets.csv'}`",
                f"- Remaining worse samples: `{output_dir / 'still_worse_than_direct_after_goodstart.csv'}`",
                f"- Plots: `{plot_dir}`",
                f"- Summary plot: `{output_dir / 'goodstart_remaining_nmse_summary.png'}`",
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
                f"- Target samples: `{len(target_ids)}`",
                f"- Good-start pool: `{len(good_df)}`",
                f"- Attempt rows: `{len(attempts_df)}`",
                f"- Plots: `{len(list(plot_dir.glob('*.png')))}`",
                f"- Remaining worse than direct: `{len(remaining_after)}`",
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
