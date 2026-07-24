# -*- coding: utf-8 -*-
"""Re-optimize all v11 samples with good-sample initial values.

Run this file directly in VS Code. No command-line arguments are required.
For each sample, this script tries several circuit-parameter initial values
from samples that already have better final NMSE, then keeps the best result.
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
GOODSTART_SCRIPT = THIS_DIR / "reoptimize_v11_remaining_with_good_starts.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
FIRST_LABEL = "v11_sharedopt_c30"
SOURCE_LABEL = "v11_sharedopt_c30_goodstart_worst10pct"
RUN_LABEL = "v11_sharedopt_c30_goodstart_all"
MAX_NFEV_PER_START = 220
GLOBAL_GOOD_STARTS = 8
NEAREST_GOOD_STARTS = 8


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


def geometry_distances(sample_row: pd.Series, pool_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    values = pool_df[feature_cols].to_numpy(dtype=np.float64)
    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-12)
    target = sample_row[feature_cols].to_numpy(dtype=np.float64)
    return pd.Series(np.linalg.norm(((values - center) / scale) - ((target - center) / scale), axis=1), index=pool_df.index)


def build_start_list(
    sample_id: str,
    sample_row: pd.Series,
    metrics_df: pd.DataFrame,
    target_by_id: pd.DataFrame,
    first_by_id: pd.DataFrame,
    wrapper,
    current_p: np.ndarray,
    feature_cols: list[str],
) -> list[tuple[str, np.ndarray]]:
    current_nmse = float(metrics_df.loc[metrics_df["sample_id"].eq(sample_id), "source_nmse_s11_s21_ri"].iloc[0])
    better_pool = metrics_df[
        metrics_df["source_nmse_s11_s21_ri"].lt(current_nmse) & ~metrics_df["sample_id"].eq(sample_id)
    ].copy()
    if len(better_pool) < max(GLOBAL_GOOD_STARTS, NEAREST_GOOD_STARTS):
        better_pool = metrics_df[~metrics_df["sample_id"].eq(sample_id)].copy()

    starts: list[tuple[str, np.ndarray]] = [
        ("source_final", current_p),
        ("first_pass", first_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)),
        ("unit", wrapper.V08_P0),
    ]
    global_ids = better_pool.sort_values("source_nmse_s11_s21_ri").head(GLOBAL_GOOD_STARTS)["sample_id"].tolist()
    distances = geometry_distances(sample_row, better_pool, feature_cols)
    nearest_ids = better_pool.loc[distances.sort_values().head(NEAREST_GOOD_STARTS).index, "sample_id"].tolist()
    for source_id in global_ids:
        starts.append((f"global_good:{source_id}", target_by_id.loc[source_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)))
    for source_id in nearest_ids:
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


def summary_row(split: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split,
        "count": int(len(group)),
        "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
        "source_nmse_mean": float(group["source_nmse_s11_s21_ri"].mean()),
        "all_goodstart_nmse_mean": float(group["all_goodstart_nmse_s11_s21_ri"].mean()),
        "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
        "source_nmse_median": float(group["source_nmse_s11_s21_ri"].median()),
        "all_goodstart_nmse_median": float(group["all_goodstart_nmse_s11_s21_ri"].median()),
        "source_better_than_direct_count": int((group["source_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()),
        "all_better_than_direct_count": int(
            (group["all_goodstart_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
        ),
        "improved_count": int((group["all_goodstart_nmse_s11_s21_ri"] < group["source_nmse_s11_s21_ri"]).sum()),
        "unchanged_count": int(np.isclose(group["all_goodstart_nmse_s11_s21_ri"], group["source_nmse_s11_s21_ri"]).sum()),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def save_summary_plot(base, output_dir: Path, metrics: pd.DataFrame) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].scatter(metrics["source_nmse_s11_s21_ri"], metrics["all_goodstart_nmse_s11_s21_ri"], s=18, alpha=0.75)
    max_nmse = float(max(metrics["source_nmse_s11_s21_ri"].max(), metrics["all_goodstart_nmse_s11_s21_ri"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Before all-sample reopt NMSE")
    axes[0].set_ylabel("After all-sample reopt NMSE")
    axes[0].set_title("All samples")
    axes[0].grid(True, alpha=0.3)
    metrics[["direct_nmse_s11_s21_ri", "source_nmse_s11_s21_ri", "all_goodstart_nmse_s11_s21_ri"]].plot(
        kind="box",
        ax=axes[1],
    )
    axes[1].set_title("NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "all_goodstart_nmse_summary.png")
    base.plt.close(fig)


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_all_goodstarts")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_wrapper_for_all_goodstarts")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_all_goodstarts")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    first_dir = version_root / "results" / FIRST_LABEL
    source_dir = version_root / "results" / SOURCE_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = first_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0

    source_metrics_raw = pd.read_csv(source_dir / "worst10pct_direct_first_source_final_metrics.csv", encoding="utf-8-sig")
    source_targets = pd.read_csv(source_dir / "v08_shared_worst10pct_goodstart_targets.csv", encoding="utf-8-sig")
    first_targets = pd.read_csv(first_dir / "v08_shared_optimized_targets.csv", encoding="utf-8-sig")
    feature_cols = list(base.STRUCTURE_COLUMNS)

    source_metrics = source_metrics_raw.rename(columns={"worst10pct_nmse_s11_s21_ri": "source_nmse_s11_s21_ri"})
    source_metrics = source_metrics.merge(source_targets[["sample_id", *feature_cols]], on="sample_id", how="left")
    target_by_id = source_targets.set_index("sample_id")
    first_by_id = first_targets.set_index("sample_id")

    dut_df = wrapper.collect_v11_samples(base)
    sim = base.load_single_device_simulation(dut_df, source.calibrated_ads_settings())
    omega = 2.0 * np.pi * sim.freq_hz

    rows = []
    attempts = []
    targets = []
    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        current_p = target_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        current_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, current_p))
        direct_metric = source.metric_row(base, target_s, direct_s)
        current_metric = source.metric_row(base, target_s, current_s)
        best_p = current_p
        best_metric = current_metric
        best_label = "source_final"

        y_true = s11_s21_ri_vector(target_s)
        denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2) / max(len(y_true), 1))
        denom = max(float(denom), 1e-30)
        start_list = build_start_list(
            sample_id,
            sample,
            source_metrics,
            target_by_id,
            first_by_id,
            wrapper,
            current_p,
            feature_cols,
        )
        for start_idx, (label, p0) in enumerate(start_list):
            start_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, p0))
            start_metric = source.metric_row(base, target_s, start_s)
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
                    "start_nmse_s11_s21_ri": start_metric["nmse_s11_s21_ri"],
                    "final_nmse_s11_s21_ri": cand_metric["nmse_s11_s21_ri"],
                    "final_mse_all_s": cand_metric["mse_all_s"],
                    "nfev": int(res.nfev),
                    "success": bool(res.success),
                }
            )
            if cand_metric["nmse_s11_s21_ri"] < best_metric["nmse_s11_s21_ri"]:
                best_p = res.x
                best_metric = cand_metric
                best_label = label

        param_row = {"sample_id": sample_id, "split": sample["split"]}
        for col in feature_cols:
            param_row[col] = float(sample[col])
        for p_idx, name in enumerate(wrapper.V08_PARAM_NAMES):
            param_row[name] = float(best_p[p_idx])
        targets.append(param_row)
        rows.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "file": sample["file"],
                "dut_index": int(sample["dut_index"]),
                "start_count": len(start_list),
                "best_start_label": best_label,
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "source_nmse_s11_s21_ri": current_metric["nmse_s11_s21_ri"],
                "all_goodstart_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
                "direct_mse_all_s": direct_metric["mse_all_s"],
                "source_mse_all_s": current_metric["mse_all_s"],
                "all_goodstart_mse_all_s": best_metric["mse_all_s"],
                "direct_s11_db_mae": direct_metric["s11_db_mae"],
                "source_s11_db_mae": current_metric["s11_db_mae"],
                "all_goodstart_s11_db_mae": best_metric["s11_db_mae"],
                "direct_s21_db_mae": direct_metric["s21_db_mae"],
                "source_s21_db_mae": current_metric["s21_db_mae"],
                "all_goodstart_s21_db_mae": best_metric["s21_db_mae"],
            }
        )
        print(
            f"[all-goodstart] {i + 1}/{len(dut_df)} {sample_id}: "
            f"before={current_metric['nmse_s11_s21_ri']:.3e}, "
            f"after={best_metric['nmse_s11_s21_ri']:.3e}, start={best_label}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    attempts_df = pd.DataFrame(attempts)
    targets_df = pd.DataFrame(targets)
    summary = summarize(metrics)
    save_summary_plot(base, output_dir, metrics)
    remaining = metrics[metrics["all_goodstart_nmse_s11_s21_ri"].gt(metrics["direct_nmse_s11_s21_ri"])]

    metrics.to_csv(output_dir / "all_goodstart_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "all_goodstart_summary.csv", index=False, encoding="utf-8-sig")
    attempts_df.to_csv(output_dir / "all_goodstart_attempts.csv", index=False, encoding="utf-8-sig")
    targets_df.to_csv(output_dir / "v08_shared_all_goodstart_targets.csv", index=False, encoding="utf-8-sig")
    remaining.to_csv(output_dir / "still_worse_than_direct_after_all_goodstart.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "source_result_dir": str(source_dir),
        "output_dir": str(output_dir),
        "sample_count": int(len(metrics)),
        "global_good_starts": GLOBAL_GOOD_STARTS,
        "nearest_good_starts": NEAREST_GOOD_STARTS,
        "max_nfev_per_start": MAX_NFEV_PER_START,
        "attempt_rows": int(len(attempts_df)),
        "summary": summary.to_dict(orient="records"),
        "remaining_worse_count": int(len(remaining)),
    }
    (output_dir / "all_goodstart_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "all_goodstart_report.md").write_text(
        "\n".join(
            [
                "# V11 All-Sample Good-Start Re-Optimization",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Source result: `{source_dir}`",
                f"- Output: `{output_dir}`",
                f"- Samples: `{len(metrics)}`",
                f"- Starts per sample: source final, first pass, unit vector, up to `{GLOBAL_GOOD_STARTS}` global better starts, and up to `{NEAREST_GOOD_STARTS}` nearest better starts.",
                "- Objective: normalized real/imag `S11` and `S21` residuals.",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Metrics: `{output_dir / 'all_goodstart_metrics.csv'}`",
                f"- Summary: `{output_dir / 'all_goodstart_summary.csv'}`",
                f"- Attempts: `{output_dir / 'all_goodstart_attempts.csv'}`",
                f"- Final targets: `{output_dir / 'v08_shared_all_goodstart_targets.csv'}`",
                f"- Remaining worse samples: `{output_dir / 'still_worse_than_direct_after_all_goodstart.csv'}`",
                f"- Summary plot: `{output_dir / 'all_goodstart_nmse_summary.png'}`",
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
                f"- Samples: `{len(metrics)}`",
                f"- Attempt rows: `{len(attempts_df)}`",
                f"- Remaining worse than direct: `{len(remaining)}`",
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
