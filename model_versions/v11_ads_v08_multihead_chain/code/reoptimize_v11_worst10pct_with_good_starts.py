# -*- coding: utf-8 -*-
"""Re-optimize the worst 10 percent final v11 samples with good-sample starts.

Run this file directly in VS Code. No command-line arguments are required.
The target set is the 20 samples with the largest final
`goodstart_nmse_s11_s21_ri` from the latest good-start run.
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
SOURCE_LABEL = "v11_sharedopt_c30_goodstart_remaining"
RUN_LABEL = "v11_sharedopt_c30_goodstart_worst10pct"
WORST_FRACTION = 0.10
MAX_NFEV_PER_START = 300
GLOBAL_GOOD_STARTS = 32
NEAREST_GOOD_STARTS = 32


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


def summary_row(split: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split,
        "count": int(len(group)),
        "worst10pct_target_count": int(group["was_worst10pct_target"].sum()),
        "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
        "source_final_nmse_mean": float(group["source_final_nmse_s11_s21_ri"].mean()),
        "worst10pct_nmse_mean": float(group["worst10pct_nmse_s11_s21_ri"].mean()),
        "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
        "source_final_nmse_median": float(group["source_final_nmse_s11_s21_ri"].median()),
        "worst10pct_nmse_median": float(group["worst10pct_nmse_s11_s21_ri"].median()),
        "source_better_than_direct_count": int(
            (group["source_final_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
        ),
        "worst10pct_better_than_direct_count": int(
            (group["worst10pct_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
        ),
        "target_improved_count": int(
            (
                group["was_worst10pct_target"]
                & (group["worst10pct_nmse_s11_s21_ri"] < group["source_final_nmse_s11_s21_ri"])
            ).sum()
        ),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in metrics.groupby("split", sort=True):
        rows.append(summary_row(split, group))
    rows.append(summary_row("all", metrics))
    return pd.DataFrame(rows)


def save_summary_plot(base, output_dir: Path, metrics: pd.DataFrame) -> None:
    target = metrics[metrics["was_worst10pct_target"]]
    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].scatter(
        target["source_final_nmse_s11_s21_ri"],
        target["worst10pct_nmse_s11_s21_ri"],
        s=35,
        alpha=0.85,
    )
    max_nmse = float(max(target["source_final_nmse_s11_s21_ri"].max(), target["worst10pct_nmse_s11_s21_ri"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Before worst-10% reopt NMSE")
    axes[0].set_ylabel("After worst-10% reopt NMSE")
    axes[0].set_title("Worst 10% target samples")
    axes[0].grid(True, alpha=0.3)
    target[["direct_nmse_s11_s21_ri", "source_final_nmse_s11_s21_ri", "worst10pct_nmse_s11_s21_ri"]].plot(
        kind="box",
        ax=axes[1],
    )
    axes[1].set_title("Target-sample NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "worst10pct_goodstart_nmse_summary.png")
    base.plt.close(fig)


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_c30_source_for_worst10pct")
    goodstart = load_module(GOODSTART_SCRIPT, "v11_goodstart_funcs_for_worst10pct")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_wrapper_for_worst10pct")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_worst10pct")

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

    source_metrics = pd.read_csv(source_dir / "goodstart_direct_first_prev_final_metrics.csv", encoding="utf-8-sig")
    source_targets = pd.read_csv(source_dir / "v08_shared_goodstart_targets.csv", encoding="utf-8-sig")
    first_targets = pd.read_csv(first_dir / "v08_shared_optimized_targets.csv", encoding="utf-8-sig")
    target_count = max(1, int(np.ceil(len(source_metrics) * WORST_FRACTION)))
    worst_ids = (
        source_metrics.sort_values("goodstart_nmse_s11_s21_ri", ascending=False)
        .head(target_count)["sample_id"]
        .tolist()
    )
    target_id_set = set(worst_ids)
    print(f"Worst 10 percent target samples: {len(worst_ids)}", flush=True)

    feature_cols = list(base.STRUCTURE_COLUMNS)
    current_by_id = source_targets.set_index("sample_id")
    first_by_id = first_targets.set_index("sample_id")
    good_df = (
        source_metrics[~source_metrics["sample_id"].isin(target_id_set)]
        .merge(source_targets[["sample_id", *feature_cols]], on="sample_id", how="left")
        .copy()
    )
    good_df["reopt_nmse_s11_s21_ri"] = good_df["goodstart_nmse_s11_s21_ri"]

    dut_df = wrapper.collect_v11_samples(base)
    sim = base.load_single_device_simulation(dut_df, source.calibrated_ads_settings())
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9

    rows = []
    attempts = []
    targets = []
    plot_dir = output_dir / "comparison_plots" / "worst10pct_goodstart"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        first_p = first_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        current_p = current_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        first_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, first_p))
        current_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, current_p))
        direct_metric = source.metric_row(base, target_s, direct_s)
        first_metric = source.metric_row(base, target_s, first_s)
        current_metric = source.metric_row(base, target_s, current_s)
        best_p = current_p
        best_s = current_s
        best_metric = current_metric
        best_label = "source_final"
        start_count = 0

        if sample_id in target_id_set:
            y_true = s11_s21_ri_vector(target_s)
            denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2) / max(len(y_true), 1))
            denom = max(float(denom), 1e-30)
            start_list = goodstart.build_good_start_list(
                sample_id,
                sample,
                good_df,
                current_by_id,
                wrapper,
                current_p,
                first_p,
                feature_cols,
            )
            start_count = len(start_list)
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
            plot_row = {
                "sample_id": sample_id,
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "previous_reopt_nmse_s11_s21_ri": current_metric["nmse_s11_s21_ri"],
                "goodstart_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
            }
            goodstart.plot_transfer(
                base,
                freq_ghz,
                target_s,
                direct_s,
                first_s,
                current_s,
                best_s,
                plot_row,
                plot_dir / f"{sample_id}.png",
            )
            print(
                f"[worst10pct] {sample_id}: before={current_metric['nmse_s11_s21_ri']:.3e}, "
                f"after={best_metric['nmse_s11_s21_ri']:.3e}, direct={direct_metric['nmse_s11_s21_ri']:.3e}, "
                f"start={best_label}",
                flush=True,
            )

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
                "was_worst10pct_target": sample_id in target_id_set,
                "start_count": start_count,
                "best_start_label": best_label,
                "direct_nmse_s11_s21_ri": direct_metric["nmse_s11_s21_ri"],
                "first_nmse_s11_s21_ri": first_metric["nmse_s11_s21_ri"],
                "source_final_nmse_s11_s21_ri": current_metric["nmse_s11_s21_ri"],
                "worst10pct_nmse_s11_s21_ri": best_metric["nmse_s11_s21_ri"],
                "direct_mse_all_s": direct_metric["mse_all_s"],
                "source_final_mse_all_s": current_metric["mse_all_s"],
                "worst10pct_mse_all_s": best_metric["mse_all_s"],
                "direct_s11_db_mae": direct_metric["s11_db_mae"],
                "source_final_s11_db_mae": current_metric["s11_db_mae"],
                "worst10pct_s11_db_mae": best_metric["s11_db_mae"],
                "direct_s21_db_mae": direct_metric["s21_db_mae"],
                "source_final_s21_db_mae": current_metric["s21_db_mae"],
                "worst10pct_s21_db_mae": best_metric["s21_db_mae"],
            }
        )

    metrics = pd.DataFrame(rows)
    attempts_df = pd.DataFrame(attempts)
    targets_df = pd.DataFrame(targets)
    summary = summarize(metrics)
    save_summary_plot(base, output_dir, metrics)
    remaining = metrics[metrics["worst10pct_nmse_s11_s21_ri"].gt(metrics["direct_nmse_s11_s21_ri"])]

    metrics.to_csv(output_dir / "worst10pct_direct_first_source_final_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "worst10pct_goodstart_summary.csv", index=False, encoding="utf-8-sig")
    attempts_df.to_csv(output_dir / "worst10pct_goodstart_attempts.csv", index=False, encoding="utf-8-sig")
    targets_df.to_csv(output_dir / "v08_shared_worst10pct_goodstart_targets.csv", index=False, encoding="utf-8-sig")
    remaining.to_csv(output_dir / "still_worse_than_direct_after_worst10pct.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "source_result_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_count": int(len(worst_ids)),
        "worst_fraction": WORST_FRACTION,
        "good_start_pool_count": int(len(good_df)),
        "global_good_starts": GLOBAL_GOOD_STARTS,
        "nearest_good_starts": NEAREST_GOOD_STARTS,
        "max_nfev_per_start": MAX_NFEV_PER_START,
        "summary": summary.to_dict(orient="records"),
        "remaining_worse_count": int(len(remaining)),
    }
    (output_dir / "worst10pct_goodstart_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "worst10pct_goodstart_report.md").write_text(
        "\n".join(
            [
                "# V11 Worst 10 Percent Good-Start Re-Optimization",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Source result: `{source_dir}`",
                f"- Output: `{output_dir}`",
                f"- Target samples: `{len(worst_ids)}`",
                f"- Good-start pool: `{len(good_df)}`",
                f"- Objective: normalized real/imag `S11` and `S21` residuals.",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Metrics: `{output_dir / 'worst10pct_direct_first_source_final_metrics.csv'}`",
                f"- Summary: `{output_dir / 'worst10pct_goodstart_summary.csv'}`",
                f"- Attempts: `{output_dir / 'worst10pct_goodstart_attempts.csv'}`",
                f"- Final targets: `{output_dir / 'v08_shared_worst10pct_goodstart_targets.csv'}`",
                f"- Remaining worse samples: `{output_dir / 'still_worse_than_direct_after_worst10pct.csv'}`",
                f"- Plots: `{plot_dir}`",
                f"- Summary plot: `{output_dir / 'worst10pct_goodstart_nmse_summary.png'}`",
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
                f"- Target samples: `{len(worst_ids)}`",
                f"- Good-start pool: `{len(good_df)}`",
                f"- Attempt rows: `{len(attempts_df)}`",
                f"- Plots: `{len(list(plot_dir.glob('*.png')))}`",
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
