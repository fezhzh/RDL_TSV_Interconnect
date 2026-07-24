# -*- coding: utf-8 -*-
"""Re-optimize bad v11 ADS-length-0.9 samples with good-sample starts.

Run this file directly in VS Code. No command-line arguments are required.

The source run is `v11_sharedopt_c30_adslen09`. This continuation targets the
samples with first-pass optimized NMSE > 0.1, tries parameters from already-good
samples as initial values, and records sign statistics for the seven circuit
parameters.
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
FIRST_LABEL = "v11_sharedopt_c30_adslen09"
RUN_LABEL = "v11_sharedopt_c30_adslen09_goodstart_bad"
ADS_DEVICE_LENGTH_SCALE = 0.9
TARGET_NMSE_THRESHOLD = 0.1
MAX_NFEV_PER_START = 260
GLOBAL_GOOD_STARTS = 24
NEAREST_GOOD_STARTS = 24
SIGN_EPS = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def summary_row(split: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split,
        "count": int(len(group)),
        "target_count": int(group["was_reopt_target"].sum()),
        "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
        "first_nmse_mean": float(group["first_nmse_s11_s21_ri"].mean()),
        "goodstart_nmse_mean": float(group["goodstart_nmse_s11_s21_ri"].mean()),
        "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
        "first_nmse_median": float(group["first_nmse_s11_s21_ri"].median()),
        "goodstart_nmse_median": float(group["goodstart_nmse_s11_s21_ri"].median()),
        "first_better_than_direct_count": int((group["first_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()),
        "goodstart_better_than_direct_count": int(
            (group["goodstart_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
        ),
        "first_gt_0p1_count": int(group["first_nmse_s11_s21_ri"].gt(TARGET_NMSE_THRESHOLD).sum()),
        "goodstart_gt_0p1_count": int(group["goodstart_nmse_s11_s21_ri"].gt(TARGET_NMSE_THRESHOLD).sum()),
        "improved_count": int((group["goodstart_nmse_s11_s21_ri"] < group["first_nmse_s11_s21_ri"]).sum()),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def parameter_sign_stats(targets: pd.DataFrame, param_names: list[str], subset_col: str | None = None) -> pd.DataFrame:
    groups = [("all", targets)]
    if subset_col:
        groups.append(("reoptimized_targets", targets[targets[subset_col]]))
        groups.append(("not_reoptimized", targets[~targets[subset_col]]))
    rows = []
    for group_name, group in groups:
        for name in param_names:
            values = group[name].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "group": group_name,
                    "parameter": name,
                    "count": int(len(values)),
                    "negative_count": int(np.sum(values < -SIGN_EPS)),
                    "zero_count": int(np.sum(np.abs(values) <= SIGN_EPS)),
                    "positive_count": int(np.sum(values > SIGN_EPS)),
                    "min": float(np.min(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "max": float(np.max(values)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_summary_plot(base, output_dir: Path, metrics: pd.DataFrame, sign_stats: pd.DataFrame) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(13, 4), dpi=150)
    axes[0].scatter(metrics["first_nmse_s11_s21_ri"], metrics["goodstart_nmse_s11_s21_ri"], s=22, alpha=0.8)
    max_nmse = float(max(metrics["first_nmse_s11_s21_ri"].max(), metrics["goodstart_nmse_s11_s21_ri"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[0].axhline(TARGET_NMSE_THRESHOLD, color="#dc2626", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("First optimized NMSE")
    axes[0].set_ylabel("Good-start NMSE")
    axes[0].set_title("Good-start re-optimization")
    axes[0].grid(True, alpha=0.3)

    all_stats = sign_stats[sign_stats["group"].eq("all")]
    x = np.arange(len(all_stats))
    axes[1].bar(x, all_stats["negative_count"], label="negative", color="#ef4444")
    axes[1].bar(x, all_stats["positive_count"], bottom=all_stats["negative_count"], label="positive", color="#22c55e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_stats["parameter"], rotation=35, ha="right")
    axes[1].set_ylabel("Sample count")
    axes[1].set_title("Final parameter signs")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "goodstart_bad_nmse_and_parameter_signs.png")
    base.plt.close(fig)


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_adslen09_goodstarts")
    goodstart = load_module(GOODSTART_SCRIPT, "v11_goodstart_helpers_for_adslen09")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_shared7_wrapper_for_adslen09_goodstarts")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_adslen09_goodstarts")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    first_dir = version_root / "results" / FIRST_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = first_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = ADS_DEVICE_LENGTH_SCALE

    first_targets = pd.read_csv(first_dir / "v08_shared_optimized_targets.csv", encoding="utf-8-sig")
    first_metrics = pd.read_csv(first_dir / "direct_vs_optimized_metrics.csv", encoding="utf-8-sig")
    target_ids = set(first_metrics[first_metrics["optimized_nmse_s11_s21_ri"].gt(TARGET_NMSE_THRESHOLD)]["sample_id"])
    print(f"Target samples with first optimized NMSE > {TARGET_NMSE_THRESHOLD}: {len(target_ids)}", flush=True)

    feature_cols = list(base.STRUCTURE_COLUMNS)
    first_by_id = first_targets.set_index("sample_id")
    good_df = (
        first_metrics[
            first_metrics["optimized_nmse_s11_s21_ri"].le(TARGET_NMSE_THRESHOLD)
            & first_metrics["optimized_nmse_s11_s21_ri"].lt(first_metrics["direct_nmse_s11_s21_ri"])
        ]
        .rename(columns={"optimized_nmse_s11_s21_ri": "reopt_nmse_s11_s21_ri"})
        .merge(first_targets[["sample_id", *feature_cols]], on="sample_id", how="left")
        .reset_index(drop=True)
    )

    dut_df = wrapper.collect_v11_samples(base)
    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = ADS_DEVICE_LENGTH_SCALE
    sim = base.load_single_device_simulation(dut_df, settings)
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9

    rows = []
    attempts = []
    final_targets = []
    plot_dir = output_dir / "comparison_plots" / "goodstart_bad"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        first_p = first_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        first_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, first_p))
        direct_metric = source.metric_row(base, target_s, direct_s)
        first_metric = source.metric_row(base, target_s, first_s)
        best_p = first_p
        best_s = first_s
        best_metric = first_metric
        best_label = "first_pass"
        start_count = 0

        if sample_id in target_ids:
            y_true = goodstart.s11_s21_ri_vector(target_s)
            denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2) / max(len(y_true), 1))
            denom = max(float(denom), 1e-30)
            starts = goodstart.build_good_start_list(
                sample_id=sample_id,
                sample_row=sample,
                good_df=good_df,
                target_by_id=first_by_id,
                wrapper=wrapper,
                current_p=first_p,
                first_p=first_p,
                feature_cols=feature_cols,
            )
            for start_idx, (label, p0) in enumerate(starts):
                start_count += 1
                start_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, p0))
                start_metric = source.metric_row(base, target_s, start_s)
                res = least_squares(
                    goodstart.residual_s11_s21_ri,
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
                "previous_reopt_nmse_s11_s21_ri": first_metric["nmse_s11_s21_ri"],
                "goodstart_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
            }
            goodstart.plot_transfer(
                base,
                freq_ghz,
                target_s,
                direct_s,
                first_s,
                first_s,
                best_s,
                plot_row,
                plot_dir / f"{sample_id}.png",
            )

        param_row = {"sample_id": sample_id, "split": sample["split"], "was_reopt_target": sample_id in target_ids}
        for col in feature_cols:
            param_row[col] = sample[col]
        for name, value in zip(wrapper.V08_PARAM_NAMES, best_p):
            param_row[name] = float(value)
        final_targets.append(param_row)

        rows.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "was_reopt_target": sample_id in target_ids,
                "best_start_label": best_label,
                "start_count": start_count,
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "first_nmse_s11_s21_ri": first_metric["nmse_s11_s21_ri"],
                "goodstart_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
                "direct_mse_all_s": direct_metric["mse_all_s"],
                "first_mse_all_s": first_metric["mse_all_s"],
                "goodstart_mse_all_s": best_metric["mse_all_s"],
            }
        )
        print(
            f"[goodstart-adslen09] {i + 1}/{len(dut_df)} {sample_id}: "
            f"first={first_metric['nmse_s11_s21_ri']:.3e}, best={best_metric['nmse_s11_s21_ri']:.3e}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    targets = pd.DataFrame(final_targets)
    attempts_df = pd.DataFrame(attempts)
    summary = summarize(metrics)
    sign_stats = parameter_sign_stats(targets, wrapper.V08_PARAM_NAMES, subset_col="was_reopt_target")
    negative_samples = targets[
        targets[wrapper.V08_PARAM_NAMES].lt(-SIGN_EPS).any(axis=1)
    ][["sample_id", "split", "was_reopt_target", *wrapper.V08_PARAM_NAMES]].copy()
    negative_samples["negative_param_count"] = negative_samples[wrapper.V08_PARAM_NAMES].lt(-SIGN_EPS).sum(axis=1)

    metrics.to_csv(output_dir / "goodstart_bad_metrics.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(output_dir / "v08_shared_adslen09_goodstart_bad_targets.csv", index=False, encoding="utf-8-sig")
    attempts_df.to_csv(output_dir / "goodstart_bad_attempts.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "goodstart_bad_summary.csv", index=False, encoding="utf-8-sig")
    sign_stats.to_csv(output_dir / "parameter_sign_stats.csv", index=False, encoding="utf-8-sig")
    negative_samples.to_csv(output_dir / "samples_with_negative_parameters.csv", index=False, encoding="utf-8-sig")
    save_summary_plot(base, output_dir, metrics, sign_stats)

    report = {
        "run_label": RUN_LABEL,
        "source_label": FIRST_LABEL,
        "ads_device_length_scale": ADS_DEVICE_LENGTH_SCALE,
        "target_nmse_threshold": TARGET_NMSE_THRESHOLD,
        "target_count": int(len(target_ids)),
        "attempt_count": int(len(attempts_df)),
        "summary": summary.to_dict(orient="records"),
        "parameter_sign_stats": sign_stats.to_dict(orient="records"),
    }
    (output_dir / "goodstart_bad_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "goodstart_bad_report.md").write_text(
        "\n".join(
            [
                "# V11 ADS Length 0.9 Good-Start Re-Optimization",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Source: `{first_dir}`",
                f"- Output: `{output_dir}`",
                f"- ADS device length scale: `{ADS_DEVICE_LENGTH_SCALE}`",
                f"- Re-optimization target: first optimized NMSE > `{TARGET_NMSE_THRESHOLD}`",
                f"- Target samples: `{len(target_ids)}`",
                f"- Attempts: `{len(attempts_df)}`",
                "",
                "## NMSE Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
                "",
                "## Outputs",
                "",
                f"- Final targets: `{output_dir / 'v08_shared_adslen09_goodstart_bad_targets.csv'}`",
                f"- Metrics: `{output_dir / 'goodstart_bad_metrics.csv'}`",
                f"- Attempts: `{output_dir / 'goodstart_bad_attempts.csv'}`",
                f"- Parameter sign stats: `{output_dir / 'parameter_sign_stats.csv'}`",
                f"- Samples with negative parameters: `{output_dir / 'samples_with_negative_parameters.csv'}`",
                f"- Summary plot: `{output_dir / 'goodstart_bad_nmse_and_parameter_signs.png'}`",
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
                f"- Source ADS cache: `{base.ADS_CACHE_DIR}`",
                f"- Source ADS cache files: `{len(list(base.ADS_CACHE_DIR.glob('*.s2p')))}`",
                f"- Source samples: `{len(metrics)}`",
                f"- Re-optimized target samples: `{len(target_ids)}`",
                f"- Least-squares attempts: `{len(attempts_df)}`",
                f"- Comparison plots: `{len(list(plot_dir.glob('*.png')))}`",
                "",
                "## NMSE Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)
    print(dataframe_to_markdown(sign_stats), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
