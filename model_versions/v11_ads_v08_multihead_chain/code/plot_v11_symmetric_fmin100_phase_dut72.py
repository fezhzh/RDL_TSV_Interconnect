# -*- coding: utf-8 -*-
"""Plot fmin100 phase-loss comparison for LHS400_Connection2_train_dut72.

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
TRAIN_SCRIPT = THIS_DIR / "train_v11_positive_multihead_sparam_from_shared.py"
RUN_LABEL = "v11_positive_symmetric_multihead_lc_fmin100_phase_log_adslen09"
SAMPLE_ID = "LHS400_Connection2_train_dut72"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def phase_deg(s_values: np.ndarray) -> np.ndarray:
    return np.angle(s_values, deg=True)


def unwrap_phase_deg(s_values: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(s_values)) * 180.0 / np.pi


def main() -> None:
    train = load_module(TRAIN_SCRIPT, "v11_phase_dut72_train")
    source = train.load_module(train.SOURCE_SCRIPT, "v11_phase_dut72_source")
    positive = train.load_module(train.POSITIVE_SCRIPT, "v11_phase_dut72_positive")
    wrapper = train.load_module(train.WRAPPER_SCRIPT, "v11_phase_dut72_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_phase_dut72_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    output_dir = version_root / "results" / RUN_LABEL
    plot_dir = output_dir / "phase_comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = version_root / "results" / train.OPT_RESULT_LABEL
    source_ads_dir = version_root / "results" / train.SOURCE_ADS_LABEL

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_ads_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = train.ADS_DEVICE_LENGTH_SCALE

    opt_targets_all = pd.read_csv(opt_dir / train.OPT_TARGET_FILE, encoding="utf-8-sig")
    dut_all = positive.collect_lhs400_rdl_tsv_samples(base)
    dut_df = dut_all[dut_all["sample_id"].astype(str).eq(SAMPLE_ID)].reset_index(drop=True)
    if dut_df.empty:
        raise FileNotFoundError(f"Sample not found: {SAMPLE_ID}")

    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left")
    if opt_targets[wrapper.V08_PARAM_NAMES].isna().any(axis=None):
        raise ValueError(f"Optimized parameters missing for {SAMPLE_ID}")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = train.ADS_DEVICE_LENGTH_SCALE
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9

    pred = pd.read_csv(output_dir / "positive_symmetric_multihead_lc_fmin100_phase_predictions.csv", encoding="utf-8-sig")
    pred_row = pred[pred["sample_id"].astype(str).eq(SAMPLE_ID)]
    if pred_row.empty:
        raise FileNotFoundError(f"Prediction row not found: {SAMPLE_ID}")
    pred_row = pred_row.iloc[0]
    pred_cols = [f"pred_{col}" for col in train.multihead_target_columns(wrapper)]
    pred_params = pred_row[pred_cols].to_numpy(dtype=np.float64)

    target_s = sim.target_s[0]
    direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[0])))
    opt_params = opt_targets.loc[0, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    optimized_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[0], omega, opt_params))
    model_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[0], omega, pred_params))

    curves = [
        ("HFSS simulation", target_s, "black", "-"),
        ("ADS direct", direct_s, "#64748b", ":"),
        ("Optimized shared", optimized_s, "#16a34a", "--"),
        ("Symmetric model", model_s, "#dc2626", "-"),
    ]

    fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=160)
    fig.suptitle(
        f"{SAMPLE_ID} phase comparison | symmetric-LC fmin100 phase-loss model",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
    )
    panels = [
        (0, 0, "S11 phase (wrapped)", phase_deg, "deg"),
        (1, 0, "S21 phase (wrapped)", phase_deg, "deg"),
        (0, 0, "S11 phase (unwrapped)", unwrap_phase_deg, "deg"),
        (1, 0, "S21 phase (unwrapped)", unwrap_phase_deg, "deg"),
    ]
    for ax, (m, n, title, fn, ylabel) in zip(axes.ravel(), panels):
        for label, s_matrix, color, style in curves:
            ax.plot(freq_ghz, fn(s_matrix[:, m, n]), label=label, color=color, linestyle=style, linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    phase_plot = plot_dir / f"{SAMPLE_ID}_phase_wrapped_unwrapped_fmin100_phase.png"
    fig.savefig(phase_plot)
    base.plt.close(fig)

    metric_row = {
        "sample_id": SAMPLE_ID,
        "direct_nmse_s11_s21_ri": float(base.nmse_s11_s21_real_imag(target_s, direct_s)),
        "optimized_nmse_s11_s21_ri": float(base.nmse_s11_s21_real_imag(target_s, optimized_s)),
        "model_nmse_s11_s21_ri": float(base.nmse_s11_s21_real_imag(target_s, model_s)),
        "plot": str(phase_plot),
    }
    pd.DataFrame([metric_row]).to_csv(plot_dir / f"{SAMPLE_ID}_phase_metrics_fmin100_phase.csv", index=False, encoding="utf-8-sig")
    (plot_dir / f"{SAMPLE_ID}_phase_report_fmin100_phase.json").write_text(
        json.dumps(metric_row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metric_row, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
