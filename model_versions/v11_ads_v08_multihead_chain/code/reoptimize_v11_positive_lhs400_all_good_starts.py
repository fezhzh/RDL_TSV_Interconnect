# -*- coding: utf-8 -*-
"""Re-optimize positive v11 LHS400 samples with good-sample initial values.

Run this file directly in VS Code. No command-line arguments are required.

The script reads the previous positive-parameter optimization on
``LHS400_Connection2/train/TSV_RDL`` and tries several good optimized parameter
sets as initial values for every sample. All seven connection-circuit scale
parameters remain constrained to positive values.
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
POSITIVE_SCRIPT = THIS_DIR / "optimize_v11_positive_shared_connection_lhs400_adslen09.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"

SOURCE_LABEL = "v11_positive_sharedopt_lhs400_connection2_adslen09"
RUN_LABEL = "v11_positive_sharedopt_lhs400_connection2_goodstart_all_adslen09"

ADS_DEVICE_LENGTH_SCALE = 0.9
POSITIVE_LOWER = 1e-9
POSITIVE_UPPER = 1e5
MAX_NFEV_PER_START = 120
GLOBAL_GOOD_STARTS = 6
NEAREST_GOOD_STARTS = 4
SELECTED_PLOT_COUNT = 12


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
    source_metrics: pd.DataFrame,
    target_by_id: pd.DataFrame,
    wrapper,
    current_p: np.ndarray,
    feature_cols: list[str],
) -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = [("source_positive", current_p), ("unit", wrapper.V08_P0)]
    pool = source_metrics[~source_metrics["sample_id"].eq(sample_id)].copy()
    global_ids = pool.sort_values("source_nmse_s11_s21_ri").head(GLOBAL_GOOD_STARTS)["sample_id"].tolist()

    distances = geometry_distances(sample_row, pool, feature_cols)
    nearest_ids = pool.loc[distances.sort_values().head(NEAREST_GOOD_STARTS).index, "sample_id"].tolist()

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
        "source_positive_nmse_mean": float(group["source_positive_nmse_s11_s21_ri"].mean()),
        "goodstart_nmse_mean": float(group["goodstart_nmse_s11_s21_ri"].mean()),
        "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
        "source_positive_nmse_median": float(group["source_positive_nmse_s11_s21_ri"].median()),
        "goodstart_nmse_median": float(group["goodstart_nmse_s11_s21_ri"].median()),
        "source_better_than_direct_count": int((group["source_positive_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()),
        "goodstart_better_than_direct_count": int((group["goodstart_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()),
        "improved_count": int((group["goodstart_nmse_s11_s21_ri"] < group["source_positive_nmse_s11_s21_ri"]).sum()),
        "unchanged_count": int(np.isclose(group["goodstart_nmse_s11_s21_ri"], group["source_positive_nmse_s11_s21_ri"]).sum()),
        "source_gt_0p1_count": int((group["source_positive_nmse_s11_s21_ri"] > 0.1).sum()),
        "goodstart_gt_0p1_count": int((group["goodstart_nmse_s11_s21_ri"] > 0.1).sum()),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def save_summary_plot(base, output_dir: Path, metrics: pd.DataFrame) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].scatter(
        metrics["source_positive_nmse_s11_s21_ri"],
        metrics["goodstart_nmse_s11_s21_ri"],
        s=18,
        alpha=0.75,
    )
    max_nmse = float(
        max(
            metrics["source_positive_nmse_s11_s21_ri"].max(),
            metrics["goodstart_nmse_s11_s21_ri"].max(),
        )
    )
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Before good-start NMSE")
    axes[0].set_ylabel("After good-start NMSE")
    axes[0].set_title("Positive all-sample re-optimization")
    axes[0].grid(True, alpha=0.3)
    metrics[
        [
            "direct_nmse_s11_s21_ri",
            "source_positive_nmse_s11_s21_ri",
            "goodstart_nmse_s11_s21_ri",
        ]
    ].plot(kind="box", ax=axes[1])
    axes[1].set_title("NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "positive_goodstart_nmse_summary.png")
    base.plt.close(fig)


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_positive_goodstart_source")
    positive = load_module(POSITIVE_SCRIPT, "v11_positive_goodstart_positive")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_positive_goodstart_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_positive_goodstart_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    source_dir = version_root / "results" / SOURCE_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = ADS_DEVICE_LENGTH_SCALE

    wrapper.V08_LOWER = np.full(len(wrapper.V08_PARAM_NAMES), POSITIVE_LOWER, dtype=np.float64)
    wrapper.V08_UPPER = np.full(len(wrapper.V08_PARAM_NAMES), POSITIVE_UPPER, dtype=np.float64)
    wrapper.V08_P0 = np.ones(len(wrapper.V08_PARAM_NAMES), dtype=np.float64)

    source_targets = pd.read_csv(source_dir / "v08_positive_shared_optimized_targets.csv", encoding="utf-8-sig")
    feature_cols = list(base.STRUCTURE_COLUMNS)
    source_metrics_raw = pd.read_csv(source_dir / "direct_vs_positive_optimized_metrics.csv", encoding="utf-8-sig")
    source_metrics = source_metrics_raw.rename(columns={"optimized_nmse_s11_s21_ri": "source_nmse_s11_s21_ri"})
    source_metrics = source_metrics.merge(source_targets[["sample_id", *feature_cols]], on="sample_id", how="left")
    target_by_id = source_targets.set_index("sample_id")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied to l_tmrdl, l_bsmrdl, and h_tsv before ADS simulation."

    dut_df = positive.collect_lhs400_rdl_tsv_samples(base)
    source_sample_ids = set(source_targets["sample_id"].astype(str))
    missing_source_ids = sorted(set(dut_df["sample_id"].astype(str)) - source_sample_ids)
    print(f"Samples with existing RDL_TSV S2P: {len(dut_df)}", flush=True)
    print(f"Samples with previous positive optimized targets: {len(source_sample_ids)}", flush=True)
    print(f"Samples missing previous targets and optimized from good starts: {len(missing_source_ids)}", flush=True)
    print(f"Reusing ADS cache: {base.ADS_CACHE_DIR}", flush=True)
    print(f"Positive bounds: [{POSITIVE_LOWER}, {POSITIVE_UPPER}]", flush=True)
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)

    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    rows = []
    attempts = []
    targets = []
    plot_cache = {}

    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        has_source_target = sample_id in target_by_id.index
        if has_source_target:
            current_p = target_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        else:
            current_p = wrapper.V08_P0.copy()
        current_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, current_p))
        direct_metric = source.metric_row(base, target_s, direct_s)
        current_metric = source.metric_row(base, target_s, current_s)
        best_p = current_p
        best_s = current_s
        best_metric = current_metric
        best_label = "source_positive"

        y_true = s11_s21_ri_vector(target_s)
        denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2) / max(len(y_true), 1))
        denom = max(float(denom), 1e-30)

        start_list = build_start_list(
            sample_id,
            sample,
            source_metrics,
            target_by_id,
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
                best_s = cand_s
                best_metric = cand_metric
                best_label = label

        target_row = {"sample_id": sample_id, "split": sample["split"]}
        for col in feature_cols:
            target_row[col] = float(sample[col])
        for p_idx, name in enumerate(wrapper.V08_PARAM_NAMES):
            target_row[name] = float(best_p[p_idx])
        targets.append(target_row)

        metric_row = {
            "sample_id": sample_id,
            "split": sample["split"],
            "file": sample["file"],
            "dut_index": int(sample["dut_index"]),
            "start_count": len(start_list),
            "best_start_label": best_label,
            "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
            "source_positive_nmse_s11_s21_ri": current_metric["nmse_s11_s21_ri"],
            "goodstart_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
            "direct_mse_all_s": direct_metric["mse_all_s"],
            "source_positive_mse_all_s": current_metric["mse_all_s"],
            "goodstart_mse_all_s": best_metric["mse_all_s"],
            "direct_s11_db_mae": direct_metric["s11_db_mae"],
            "source_positive_s11_db_mae": current_metric["s11_db_mae"],
            "goodstart_s11_db_mae": best_metric["s11_db_mae"],
            "direct_s21_db_mae": direct_metric["s21_db_mae"],
            "source_positive_s21_db_mae": current_metric["s21_db_mae"],
            "goodstart_s21_db_mae": best_metric["s21_db_mae"],
        }
        rows.append(metric_row)
        plot_cache[sample_id] = (target_s, direct_s, best_s, metric_row)
        print(
            f"[positive-goodstart] {i + 1}/{len(dut_df)} {sample_id}: "
            f"before={current_metric['nmse_s11_s21_ri']:.3e}, "
            f"after={best_metric['nmse_s11_s21_ri']:.3e}, start={best_label}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    attempts_df = pd.DataFrame(attempts)
    targets_df = pd.DataFrame(targets)
    summary = summarize(metrics)
    sign_summary = positive.parameter_sign_summary(targets_df, wrapper)
    remaining = metrics[metrics["goodstart_nmse_s11_s21_ri"].gt(metrics["direct_nmse_s11_s21_ri"])]

    save_summary_plot(base, output_dir, metrics)
    selected_dir = output_dir / "comparison_plots" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.concat(
        [
            metrics.sort_values("goodstart_nmse_s11_s21_ri").head(SELECTED_PLOT_COUNT // 2),
            metrics.sort_values("goodstart_nmse_s11_s21_ri", ascending=False).head(SELECTED_PLOT_COUNT // 2),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    selected_paths = []
    for _, metric in selected.iterrows():
        target_s, direct_s, best_s, row = plot_cache[str(metric["sample_id"])]
        plot_row = dict(row)
        plot_row["optimized_nmse_s11_s21_ri"] = plot_row["goodstart_nmse_s11_s21_ri"]
        plot_row["optimized_s11_db_mae"] = plot_row["goodstart_s11_db_mae"]
        plot_row["optimized_s21_db_mae"] = plot_row["goodstart_s21_db_mae"]
        out_path = selected_dir / f"{metric['sample_id']}.png"
        source.plot_sparams(base, freq_ghz, target_s, direct_s, best_s, plot_row, out_path)
        selected_paths.append(str(out_path))

    metrics.to_csv(output_dir / "positive_goodstart_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "positive_goodstart_summary.csv", index=False, encoding="utf-8-sig")
    attempts_df.to_csv(output_dir / "positive_goodstart_attempts.csv", index=False, encoding="utf-8-sig")
    targets_df.to_csv(output_dir / "v08_positive_goodstart_targets.csv", index=False, encoding="utf-8-sig")
    sign_summary.to_csv(output_dir / "positive_goodstart_parameter_sign_summary.csv", index=False, encoding="utf-8-sig")
    remaining.to_csv(output_dir / "still_worse_than_direct_after_positive_goodstart.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "source_result_dir": str(source_dir),
        "output_dir": str(output_dir),
        "sample_count": int(len(metrics)),
        "ads_cache_dir": str(base.ADS_CACHE_DIR),
        "positive_bounds": {"lower": POSITIVE_LOWER, "upper": POSITIVE_UPPER},
        "global_good_starts": GLOBAL_GOOD_STARTS,
        "nearest_good_starts": NEAREST_GOOD_STARTS,
        "max_nfev_per_start": MAX_NFEV_PER_START,
        "attempt_rows": int(len(attempts_df)),
        "summary": summary.to_dict(orient="records"),
        "parameter_sign_summary": sign_summary.to_dict(orient="records"),
        "remaining_worse_count": int(len(remaining)),
        "selected_plots": selected_paths,
    }
    (output_dir / "positive_goodstart_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "positive_goodstart_report.md").write_text(
        "\n".join(
            [
                "# V11 Positive All-Sample Good-Start Re-Optimization",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Source result: `{source_dir}`",
                f"- Output: `{output_dir}`",
                f"- Samples: `{len(metrics)}`",
                f"- Positive bounds: `[{POSITIVE_LOWER}, {POSITIVE_UPPER}]`",
                f"- Starts per sample: source positive result, unit vector, up to `{GLOBAL_GOOD_STARTS}` global best samples, and up to `{NEAREST_GOOD_STARTS}` geometry-nearest good samples.",
                "- Objective: normalized real/imag `S11` and `S21` residuals.",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_summary),
                "",
                "## Outputs",
                "",
                f"- Metrics: `{output_dir / 'positive_goodstart_metrics.csv'}`",
                f"- Summary: `{output_dir / 'positive_goodstart_summary.csv'}`",
                f"- Attempts: `{output_dir / 'positive_goodstart_attempts.csv'}`",
                f"- Final targets: `{output_dir / 'v08_positive_goodstart_targets.csv'}`",
                f"- Parameter sign summary: `{output_dir / 'positive_goodstart_parameter_sign_summary.csv'}`",
                f"- Remaining worse samples: `{output_dir / 'still_worse_than_direct_after_positive_goodstart.csv'}`",
                f"- Summary plot: `{output_dir / 'positive_goodstart_nmse_summary.png'}`",
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
                f"- Samples: `{len(metrics)}`",
                f"- Attempt rows: `{len(attempts_df)}`",
                f"- ADS cache files reused: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Remaining worse than direct: `{len(remaining)}`",
                f"- Positive sign check nonpositive total: `{int(sign_summary['nonpositive_count'].sum())}`",
                f"- Selected plots: `{len(selected_paths)}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_summary),
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)
    print(dataframe_to_markdown(sign_summary), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
