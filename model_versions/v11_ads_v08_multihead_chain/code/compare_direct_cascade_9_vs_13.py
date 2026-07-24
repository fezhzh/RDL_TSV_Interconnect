# -*- coding: utf-8 -*-
"""Compare 9-block and 13-block ADS direct cascades on the same HFSS targets.

Run this file directly in VS Code. No command-line arguments are required.
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
OUTPUT_DIR = (
    THIS_DIR.parents[0]
    / "results"
    / "direct_cascade_9_vs_13_lhs150_50_connection2"
)

SEQUENCE_9_BLOCKS = ["TMRDL", "TSV", "BSMRDL", "TSV", "TMRDL", "TSV", "BSMRDL", "TSV", "TMRDL"]
SEQUENCE_13_BLOCKS = [
    "TMRDL",
    "TSV",
    "BSMRDL",
    "TSV",
    "TMRDL",
    "TSV",
    "BSMRDL",
    "TSV",
    "TMRDL",
    "TSV",
    "BSMRDL",
    "TSV",
    "TMRDL",
]


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


def sparam_metrics(base, target_s: np.ndarray, pred_s: np.ndarray) -> dict[str, float]:
    return {
        "mse_all_s": float(np.mean(np.abs(pred_s - target_s) ** 2)),
        "nmse_s11_s21_ri": base.nmse_s11_s21_real_imag(target_s, pred_s),
        "mag_phase_mse_s11_s21": base.s11_s21_mag_phase_mse_np(target_s, pred_s),
        "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(target_s[:, 0, 0])))),
        "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(target_s[:, 1, 0])))),
    }


def load_simulation_for_sequence(base, wrapper, dut_df: pd.DataFrame, sequence: list[str]):
    base.DEVICE_SEQUENCE = list(sequence)
    base.CONNECTION_COUNT = len(sequence) - 1
    base.ADS_CACHE_DIR = wrapper.SOURCE_ADS_CACHE_DIR
    base.SIMULATION_BACKEND = "ads"
    return base.load_single_device_simulation(dut_df, base.BASE_ADS_SETTINGS)


def evaluate_sequence(base, dut_df: pd.DataFrame, sim, label: str) -> tuple[pd.DataFrame, list[np.ndarray]]:
    rows = []
    pred_s_list = []
    for i, row in dut_df.iterrows():
        pred_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
        pred_s_list.append(pred_s)
        metrics = sparam_metrics(base, sim.target_s[i], pred_s)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "file": row["file"],
                "dut_index": int(row["dut_index"]),
                "cascade": label,
                **metrics,
            }
        )
    return pd.DataFrame(rows), pred_s_list


def summarize(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in wide_df.groupby("split", sort=True):
        rows.append(
            {
                "split": split,
                "count": int(len(group)),
                "nmse_9_mean": float(group["nmse_s11_s21_ri_9"].mean()),
                "nmse_13_mean": float(group["nmse_s11_s21_ri_13"].mean()),
                "nmse_9_median": float(group["nmse_s11_s21_ri_9"].median()),
                "nmse_13_median": float(group["nmse_s11_s21_ri_13"].median()),
                "s11_db_mae_9_mean": float(group["s11_db_mae_9"].mean()),
                "s11_db_mae_13_mean": float(group["s11_db_mae_13"].mean()),
                "s21_db_mae_9_mean": float(group["s21_db_mae_9"].mean()),
                "s21_db_mae_13_mean": float(group["s21_db_mae_13"].mean()),
                "samples_13_better_nmse": int((group["nmse_s11_s21_ri_13"] < group["nmse_s11_s21_ri_9"]).sum()),
                "samples_9_better_nmse": int((group["nmse_s11_s21_ri_9"] < group["nmse_s11_s21_ri_13"]).sum()),
            }
        )
    group = wide_df
    rows.append(
        {
            "split": "all",
            "count": int(len(group)),
            "nmse_9_mean": float(group["nmse_s11_s21_ri_9"].mean()),
            "nmse_13_mean": float(group["nmse_s11_s21_ri_13"].mean()),
            "nmse_9_median": float(group["nmse_s11_s21_ri_9"].median()),
            "nmse_13_median": float(group["nmse_s11_s21_ri_13"].median()),
            "s11_db_mae_9_mean": float(group["s11_db_mae_9"].mean()),
            "s11_db_mae_13_mean": float(group["s11_db_mae_13"].mean()),
            "s21_db_mae_9_mean": float(group["s21_db_mae_9"].mean()),
            "s21_db_mae_13_mean": float(group["s21_db_mae_13"].mean()),
            "samples_13_better_nmse": int((group["nmse_s11_s21_ri_13"] < group["nmse_s11_s21_ri_9"]).sum()),
            "samples_9_better_nmse": int((group["nmse_s11_s21_ri_9"] < group["nmse_s11_s21_ri_13"]).sum()),
        }
    )
    return pd.DataFrame(rows)


def save_plots(base, dut_df: pd.DataFrame, sim13, pred9: list[np.ndarray], pred13: list[np.ndarray], wide_df: pd.DataFrame):
    plot_dir = OUTPUT_DIR / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    freq_ghz = sim13.freq_hz / 1e9

    fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].scatter(wide_df["nmse_s11_s21_ri_9"], wide_df["nmse_s11_s21_ri_13"], s=18, alpha=0.75)
    max_nmse = float(max(wide_df["nmse_s11_s21_ri_9"].max(), wide_df["nmse_s11_s21_ri_13"].max()))
    axes[0].plot([0, max_nmse], [0, max_nmse], color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("9-block direct NMSE")
    axes[0].set_ylabel("13-block direct NMSE")
    axes[0].set_title("Per-sample NMSE")
    axes[0].grid(True, alpha=0.3)
    wide_df[["nmse_s11_s21_ri_9", "nmse_s11_s21_ri_13"]].plot(kind="box", ax=axes[1])
    axes[1].set_title("NMSE distribution")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "direct_nmse_9_vs_13.png")
    base.plt.close(fig)

    candidates = wide_df[wide_df["split"].eq("test")].copy()
    candidates["improvement_13_over_9"] = candidates["nmse_s11_s21_ri_9"] - candidates["nmse_s11_s21_ri_13"]
    selected = pd.concat(
        [
            candidates.sort_values("improvement_13_over_9", ascending=False).head(3),
            candidates.sort_values("improvement_13_over_9", ascending=True).head(3),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id")

    saved = []
    for _, metric in selected.iterrows():
        idx = int(dut_df.index[dut_df["sample_id"].eq(metric["sample_id"])][0])
        target_s = sim13.target_s[idx]
        fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
        fig.suptitle(
            f"{metric['sample_id']} | 9-block NMSE={metric['nmse_s11_s21_ri_9']:.3e} | 13-block NMSE={metric['nmse_s11_s21_ri_13']:.3e}",
            x=0.02,
            y=0.985,
            ha="left",
        )
        plot_specs = [
            (0, 0, "S11 real", np.real),
            (0, 0, "S11 imag", np.imag),
            (1, 0, "S21 real", np.real),
            (1, 0, "S21 imag", np.imag),
        ]
        for ax, (m, n, title, fn) in zip(axes.ravel(), plot_specs):
            ax.plot(freq_ghz, fn(target_s[:, m, n]), label="HFSS target", color="black", linewidth=1.8)
            ax.plot(freq_ghz, fn(pred9[idx][:, m, n]), label="9-block direct", color="#2563eb", linestyle="--")
            ax.plot(freq_ghz, fn(pred13[idx][:, m, n]), label="13-block direct", color="#dc2626", linestyle=":")
            ax.set_title(title)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = plot_dir / f"{metric['sample_id']}_direct_9_vs_13.png"
        fig.savefig(out_path)
        base.plt.close(fig)
        saved.append(str(out_path))
    return saved


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


def main():
    wrapper = load_module(WRAPPER_SCRIPT, "v11_shared7_wrapper_for_direct_compare")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_base_for_direct_compare")
    base.OUTPUT_DIR = OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dut_df = wrapper.collect_v11_samples(base)
    sim9 = load_simulation_for_sequence(base, wrapper, dut_df, SEQUENCE_9_BLOCKS)
    metrics9, pred9 = evaluate_sequence(base, dut_df, sim9, "9_block")
    sim13 = load_simulation_for_sequence(base, wrapper, dut_df, SEQUENCE_13_BLOCKS)
    metrics13, pred13 = evaluate_sequence(base, dut_df, sim13, "13_block")

    wide = metrics9.merge(
        metrics13,
        on=["sample_id", "split", "file", "dut_index"],
        suffixes=("_9", "_13"),
    )
    summary = summarize(wide)
    plots = save_plots(base, dut_df, sim13, pred9, pred13, wide)

    metrics9.to_csv(OUTPUT_DIR / "direct_9_block_metrics.csv", index=False, encoding="utf-8-sig")
    metrics13.to_csv(OUTPUT_DIR / "direct_13_block_metrics.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUTPUT_DIR / "direct_9_vs_13_per_sample.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "direct_9_vs_13_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "dataset": wrapper.DATASET_NAME,
        "target_design": wrapper.TARGET_DESIGN_NAME,
        "sample_count": int(len(dut_df)),
        "sequence_9_blocks": SEQUENCE_9_BLOCKS,
        "sequence_13_blocks": SEQUENCE_13_BLOCKS,
        "ads_cache_dir": str(wrapper.SOURCE_ADS_CACHE_DIR),
        "metric_definition": {
            "nmse_s11_s21_ri": "flattened real/imag S11 and S21 over all frequency points",
            "comparison_scope": "same HFSS target samples, ADS direct cascade only, no connection network or NN",
        },
        "summary": summary.to_dict(orient="records"),
        "plots": plots,
    }
    (OUTPUT_DIR / "direct_9_vs_13_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "direct_9_vs_13_report.md").write_text(
        "\n".join(
            [
                "# Direct Cascade 9-block vs 13-block Comparison",
                "",
                f"- Dataset: `{wrapper.DATASET_NAME}`",
                f"- Target design: `{wrapper.TARGET_DESIGN_NAME}`",
                f"- Samples: `{len(dut_df)}`",
                f"- ADS cache: `{wrapper.SOURCE_ADS_CACHE_DIR}`",
                f"- 9-block sequence: `{'-'.join(SEQUENCE_9_BLOCKS)}`",
                f"- 13-block sequence: `{'-'.join(SEQUENCE_13_BLOCKS)}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## Conclusion",
                "",
                "Lower `nmse_s11_s21_ri` is better. This comparison uses the same HFSS target S-parameters for both cascades and does not include pi correction, optimization, or neural-network prediction.",
                "",
                "## Outputs",
                "",
                f"- Per-sample metrics: `{OUTPUT_DIR / 'direct_9_vs_13_per_sample.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'direct_9_vs_13_summary.csv'}`",
                f"- NMSE plot: `{OUTPUT_DIR / 'direct_nmse_9_vs_13.png'}`",
                f"- Selected curve plots: `{OUTPUT_DIR / 'comparison_plots'}`",
            ]
        ),
        encoding="utf-8",
    )
    print(dataframe_to_markdown(summary), flush=True)


if __name__ == "__main__":
    main()
