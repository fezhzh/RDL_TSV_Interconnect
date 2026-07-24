# -*- coding: utf-8 -*-
"""Plot v11 shared-parameter optimized S-parameter comparisons.

Run this file directly in VS Code. No command-line arguments are required.
It uses one optimized 7-parameter circuit per sample and inserts the same
circuit at all 12 v11 connection positions.
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
TRAINING_RESULT_DIR = (
    THIS_DIR.parents[0]
    / "results"
    / "ads_v08circuit_shared_to_multihead12_lhs150_50_connection2"
)
OUTPUT_DIR = THIS_DIR.parents[0] / "results" / "shared_v08_optimization_sparam_plots_lhs150_50_connection2"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def metrics_for(base, target_s: np.ndarray, pred_s: np.ndarray) -> dict[str, float]:
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
    for split, group in metrics.groupby("split", sort=True):
        rows.append(
            {
                "split": split,
                "count": int(len(group)),
                "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
                "optimized_nmse_mean": float(group["optimized_nmse_s11_s21_ri"].mean()),
                "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
                "optimized_nmse_median": float(group["optimized_nmse_s11_s21_ri"].median()),
                "optimized_better_count": int(
                    (group["optimized_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
                ),
                "direct_s11_db_mae_mean": float(group["direct_s11_db_mae"].mean()),
                "optimized_s11_db_mae_mean": float(group["optimized_s11_db_mae"].mean()),
                "direct_s21_db_mae_mean": float(group["direct_s21_db_mae"].mean()),
                "optimized_s21_db_mae_mean": float(group["optimized_s21_db_mae"].mean()),
            }
        )
    group = metrics
    rows.append(
        {
            "split": "all",
            "count": int(len(group)),
            "direct_nmse_mean": float(group["direct_nmse_s11_s21_ri"].mean()),
            "optimized_nmse_mean": float(group["optimized_nmse_s11_s21_ri"].mean()),
            "direct_nmse_median": float(group["direct_nmse_s11_s21_ri"].median()),
            "optimized_nmse_median": float(group["optimized_nmse_s11_s21_ri"].median()),
            "optimized_better_count": int(
                (group["optimized_nmse_s11_s21_ri"] < group["direct_nmse_s11_s21_ri"]).sum()
            ),
            "direct_s11_db_mae_mean": float(group["direct_s11_db_mae"].mean()),
            "optimized_s11_db_mae_mean": float(group["optimized_s11_db_mae"].mean()),
            "direct_s21_db_mae_mean": float(group["direct_s21_db_mae"].mean()),
            "optimized_s21_db_mae_mean": float(group["optimized_s21_db_mae"].mean()),
        }
    )
    return pd.DataFrame(rows)


def plot_one(base, freq_ghz, target_s, direct_s, optimized_s, metric_row, out_path: Path) -> None:
    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{metric_row['sample_id']} | direct NMSE={metric_row['direct_nmse_s11_s21_ri']:.3e} | "
        f"shared-optimized NMSE={metric_row['optimized_nmse_s11_s21_ri']:.3e}",
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
        ax.plot(freq_ghz, component(target_s[:, m, n]), label="HFSS target", color="black", linewidth=1.8)
        ax.plot(freq_ghz, component(direct_s[:, m, n]), label="13-block direct", color="#64748b", linestyle=":")
        ax.plot(
            freq_ghz,
            component(optimized_s[:, m, n]),
            label="shared 7-param optimized",
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


def save_summary_plot(base, metrics: pd.DataFrame) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].scatter(
        metrics["direct_nmse_s11_s21_ri"],
        metrics["optimized_nmse_s11_s21_ri"],
        s=18,
        alpha=0.75,
    )
    max_nmse = float(max(metrics["direct_nmse_s11_s21_ri"].max(), metrics["optimized_nmse_s11_s21_ri"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("13-block direct NMSE")
    axes[0].set_ylabel("Shared-optimized NMSE")
    axes[0].set_title("Per-sample optimization effect")
    axes[0].grid(True, alpha=0.3)
    metrics[["direct_nmse_s11_s21_ri", "optimized_nmse_s11_s21_ri"]].plot(kind="box", ax=axes[1])
    axes[1].set_title("NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shared_optimization_nmse_summary.png")
    base.plt.close(fig)


def main():
    wrapper = load_module(WRAPPER_SCRIPT, "v11_shared7_wrapper_for_opt_plot")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_opt_plot")
    base.OUTPUT_DIR = OUTPUT_DIR
    base.ADS_CACHE_DIR = wrapper.SOURCE_ADS_CACHE_DIR
    base.SIMULATION_BACKEND = "ads"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_path = TRAINING_RESULT_DIR / "v08_shared_optimized_targets.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"Missing optimized target file: {target_path}")

    dut_df = wrapper.collect_v11_samples(base)
    targets = pd.read_csv(target_path, encoding="utf-8-sig")
    target_by_id = targets.set_index("sample_id")
    missing = sorted(set(dut_df["sample_id"]) - set(target_by_id.index))
    if missing:
        raise ValueError(f"Optimized target file is missing samples: {missing[:5]}")

    sim = base.load_single_device_simulation(dut_df, base.BASE_ADS_SETTINGS)
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9

    plot_root = OUTPUT_DIR / "plots"
    all_plot_dir = plot_root / "all_samples"
    all_plot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cache = {}
    for i, row in dut_df.iterrows():
        sample_id = row["sample_id"]
        params = target_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        target_s = sim.target_s[i]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        optimized_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, params))
        direct_metrics = metrics_for(base, target_s, direct_s)
        optimized_metrics = metrics_for(base, target_s, optimized_s)
        metric_row = {
            "sample_id": sample_id,
            "split": row["split"],
            "file": row["file"],
            "dut_index": int(row["dut_index"]),
            "direct_mse_all_s": direct_metrics["mse_all_s"],
            "optimized_mse_all_s": optimized_metrics["mse_all_s"],
            "direct_nmse_s11_s21_ri": direct_metrics["nmse_s11_s21_ri"],
            "optimized_nmse_s11_s21_ri": optimized_metrics["nmse_s11_s21_ri"],
            "direct_mag_phase_mse_s11_s21": direct_metrics["mag_phase_mse_s11_s21"],
            "optimized_mag_phase_mse_s11_s21": optimized_metrics["mag_phase_mse_s11_s21"],
            "direct_s11_db_mae": direct_metrics["s11_db_mae"],
            "optimized_s11_db_mae": optimized_metrics["s11_db_mae"],
            "direct_s21_db_mae": direct_metrics["s21_db_mae"],
            "optimized_s21_db_mae": optimized_metrics["s21_db_mae"],
        }
        rows.append(metric_row)
        cache[sample_id] = (i, target_s, direct_s, optimized_s, metric_row)
        split_dir = all_plot_dir / str(row["split"])
        split_dir.mkdir(parents=True, exist_ok=True)
        plot_one(base, freq_ghz, target_s, direct_s, optimized_s, metric_row, split_dir / f"{sample_id}.png")
        print(f"[plot] {i + 1}/{len(dut_df)} {sample_id}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    save_summary_plot(base, metrics)

    selected_dir = plot_root / "selected"
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
        _, target_s, direct_s, optimized_s, metric_row = cache[metric["sample_id"]]
        out_path = selected_dir / f"{metric['sample_id']}.png"
        plot_one(base, freq_ghz, target_s, direct_s, optimized_s, metric_row, out_path)
        selected_paths.append(str(out_path))

    metrics.to_csv(OUTPUT_DIR / "shared_optimization_sparam_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "shared_optimization_sparam_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "source_training_result_dir": str(TRAINING_RESULT_DIR),
        "optimized_targets": str(target_path),
        "output_dir": str(OUTPUT_DIR),
        "sample_count": int(len(dut_df)),
        "connection_count": wrapper.CONNECTION_COUNT,
        "connection_parameter_mode": "one shared 7-parameter circuit per sample, repeated at all 12 connection positions",
        "v08_param_names": wrapper.V08_PARAM_NAMES,
        "summary": summary.to_dict(orient="records"),
        "selected_plots": selected_paths,
    }
    (OUTPUT_DIR / "shared_optimization_sparam_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "shared_optimization_sparam_report.md").write_text(
        "\n".join(
            [
                "# V11 Shared 7-Parameter Optimization S-Parameter Comparison",
                "",
                f"- Source optimized targets: `{target_path}`",
                f"- Output: `{OUTPUT_DIR}`",
                f"- Samples: `{len(dut_df)}`",
                f"- Connection mode: one shared 7-parameter circuit per sample, repeated at all `{wrapper.CONNECTION_COUNT}` connection positions.",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Outputs",
                "",
                f"- Per-sample metrics: `{OUTPUT_DIR / 'shared_optimization_sparam_metrics.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'shared_optimization_sparam_summary.csv'}`",
                f"- Summary plot: `{OUTPUT_DIR / 'shared_optimization_nmse_summary.png'}`",
                f"- All sample plots: `{all_plot_dir}`",
                f"- Selected test plots: `{selected_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)


if __name__ == "__main__":
    main()
