# -*- coding: utf-8 -*-
"""Plot selected hardcap-model S-parameter comparisons.

Run this file directly in VS Code. No command-line arguments are required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_v11_positive_multihead_sparam_from_shared.py"
RUN_LABEL = "v11_positive_symmetric_multihead_lc_hardcap_continue_log_adslen09"
SAMPLE_IDS = [
    "LHS400_Connection2_train_dut72",
    "LHS400_Connection2_train_dut123",
    "LHS400_Connection2_train_dut253",
    "LHS400_Connection2_train_dut258",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    train = load_module(TRAIN_SCRIPT, "v11_hardcap_plot_train")
    source = train.load_module(train.SOURCE_SCRIPT, "v11_hardcap_plot_source")
    positive = train.load_module(train.POSITIVE_SCRIPT, "v11_hardcap_plot_positive")
    wrapper = train.load_module(train.WRAPPER_SCRIPT, "v11_hardcap_plot_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_hardcap_plot_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    output_dir = version_root / "results" / RUN_LABEL
    plot_dir = output_dir / "comparison_plots"
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
    dut_df = dut_all[dut_all["sample_id"].astype(str).isin(SAMPLE_IDS)].reset_index(drop=True)
    if dut_df.empty:
        raise FileNotFoundError("No selected samples found.")
    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left")
    pred = pd.read_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_predictions.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_metrics.csv", encoding="utf-8-sig").set_index("sample_id")
    pred_by_id = pred.set_index("sample_id")
    opt_by_id = opt_targets.set_index("sample_id")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = train.ADS_DEVICE_LENGTH_SCALE
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    pred_cols = [f"pred_{col}" for col in train.multihead_target_columns(wrapper)]

    paths = []
    for idx, sample in dut_df.iterrows():
        sample_id = str(sample["sample_id"])
        target_s = sim.target_s[idx]
        direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[idx])))
        opt_p = opt_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
        opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[idx], omega, opt_p))
        pred_p = pred_by_id.loc[sample_id, pred_cols].to_numpy(dtype=np.float64)
        pred_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[idx], omega, pred_p))
        metric = metrics.loc[sample_id]

        fig, axes = base.plt.subplots(2, 2, figsize=(13, 8), dpi=150)
        fig.suptitle(
            f"{sample_id} | hardcap NN={metric['nn_nmse_s11_s21_ri']:.3e} | optimized={metric['optimized_nmse_s11_s21_ri']:.3e}",
            x=0.02,
            y=0.985,
            ha="left",
        )
        specs = [(0, 0, "S11 real", np.real), (0, 0, "S11 imag", np.imag), (1, 0, "S21 real", np.real), (1, 0, "S21 imag", np.imag)]
        for ax, (m, n, title, fn) in zip(axes.ravel(), specs):
            ax.plot(freq_ghz, fn(target_s[:, m, n]), label="HFSS", color="black", linewidth=1.8)
            ax.plot(freq_ghz, fn(direct_s[:, m, n]), label="ADS direct", color="#64748b", linestyle=":")
            ax.plot(freq_ghz, fn(opt_s[:, m, n]), label="Optimized shared", color="#16a34a", linestyle="--")
            ax.plot(freq_ghz, fn(pred_s[:, m, n]), label="Hardcap NN", color="#dc2626", linestyle="-.")
            ax.set_title(title)
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = plot_dir / f"{sample_id}.png"
        fig.savefig(out_path)
        base.plt.close(fig)
        paths.append(str(out_path))

    pd.DataFrame({"plot": paths}).to_csv(plot_dir / "hardcap_selected_sample_plots.csv", index=False, encoding="utf-8-sig")
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
