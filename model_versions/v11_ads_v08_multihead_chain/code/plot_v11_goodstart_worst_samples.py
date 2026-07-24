# -*- coding: utf-8 -*-
"""Plot worst final v11 optimized samples.

Run this file directly in VS Code. No command-line arguments are required.
It uses the latest good-start final targets and plots the samples with the
largest final optimized NMSE.
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
FIRST_LABEL = "v11_sharedopt_c30"
GOODSTART_LABEL = "v11_sharedopt_c30_goodstart_remaining"
OUTPUT_SUBDIR = "worst_final_samples"
WORST_COUNT = 12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plot_one(base, freq_ghz, target_s, direct_s, first_s, final_s, metric_row, out_path: Path) -> None:
    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(
        f"{metric_row['sample_id']} | direct={metric_row['direct_nmse_s11_s21_ri']:.3e} | "
        f"first={metric_row['first_nmse_s11_s21_ri']:.3e} | final={metric_row['goodstart_nmse_s11_s21_ri']:.3e}",
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
        ax.plot(freq_ghz, component(final_s[:, m, n]), label="Final optimized", color="#16a34a", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    base.plt.close(fig)


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


def main() -> None:
    source = load_module(SOURCE_SCRIPT, "v11_source_for_worst_final_plot")
    wrapper = load_module(WRAPPER_SCRIPT, "v11_wrapper_for_worst_final_plot")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_worst_final_plot")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    first_dir = version_root / "results" / FIRST_LABEL
    goodstart_dir = version_root / "results" / GOODSTART_LABEL
    output_dir = goodstart_dir / "comparison_plots" / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base.OUTPUT_DIR = goodstart_dir
    base.ADS_CACHE_DIR = first_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0

    metrics = pd.read_csv(goodstart_dir / "goodstart_direct_first_prev_final_metrics.csv", encoding="utf-8-sig")
    first_targets = pd.read_csv(first_dir / "v08_shared_optimized_targets.csv", encoding="utf-8-sig").set_index("sample_id")
    final_targets = pd.read_csv(goodstart_dir / "v08_shared_goodstart_targets.csv", encoding="utf-8-sig").set_index("sample_id")
    worst = metrics.sort_values("goodstart_nmse_s11_s21_ri", ascending=False).head(WORST_COUNT).reset_index(drop=True)

    dut_df = wrapper.collect_v11_samples(base)
    sim = base.load_single_device_simulation(dut_df, source.calibrated_ads_settings())
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9

    plot_rows = []
    for rank, metric in worst.iterrows():
        sample_id = str(metric["sample_id"])
        idx = int(dut_df.index[dut_df["sample_id"].eq(sample_id)][0])
        first_p = first_targets.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        final_p = final_targets.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        target_s = sim.target_s[idx]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[idx])))
        first_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[idx], omega, first_p))
        final_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[idx], omega, final_p))
        out_path = output_dir / f"rank{rank + 1:02d}_{sample_id}.png"
        plot_one(base, freq_ghz, target_s, direct_s, first_s, final_s, metric, out_path)
        rec = metric.to_dict()
        rec["rank_by_final_nmse"] = rank + 1
        rec["plot_path"] = str(out_path)
        plot_rows.append(rec)
        print(f"[worst-final-plot] rank={rank + 1} {sample_id}", flush=True)

    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_csv(goodstart_dir / "worst_final_sample_plots.csv", index=False, encoding="utf-8-sig")
    report = {
        "entry": str(Path(__file__).name),
        "source_result_dir": str(goodstart_dir),
        "output_dir": str(output_dir),
        "worst_count": WORST_COUNT,
        "rank_metric": "goodstart_nmse_s11_s21_ri descending",
        "plots": plot_df[["rank_by_final_nmse", "sample_id", "goodstart_nmse_s11_s21_ri", "plot_path"]].to_dict(
            orient="records"
        ),
    }
    (goodstart_dir / "worst_final_sample_plots_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (goodstart_dir / "worst_final_sample_plots_report.md").write_text(
        "\n".join(
            [
                "# Worst Final Optimized Sample Plots",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Result source: `{goodstart_dir}`",
                f"- Output plots: `{output_dir}`",
                f"- Ranking metric: `goodstart_nmse_s11_s21_ri` descending.",
                "",
                "## Worst Samples",
                "",
                dataframe_to_markdown(
                    plot_df[
                        [
                            "rank_by_final_nmse",
                            "sample_id",
                            "split",
                            "direct_nmse_s11_s21_ri",
                            "first_nmse_s11_s21_ri",
                            "goodstart_nmse_s11_s21_ri",
                            "plot_path",
                        ]
                    ]
                ),
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(plot_df[["rank_by_final_nmse", "sample_id", "split", "goodstart_nmse_s11_s21_ri"]]))
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
