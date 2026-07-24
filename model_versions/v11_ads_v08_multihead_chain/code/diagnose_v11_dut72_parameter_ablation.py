# -*- coding: utf-8 -*-
"""Parameter ablation diagnosis for LHS400_Connection2_train_dut72.

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
RUN_LABEL = "v11_positive_symmetric_multihead_lc_fmin60_phase_log_adslen09"
SAMPLE_ID = "LHS400_Connection2_train_dut72"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ri_nmse(s_pred: np.ndarray, s_target: np.ndarray) -> float:
    pred_vec = np.concatenate(
        [s_pred[:, 0, 0].real, s_pred[:, 0, 0].imag, s_pred[:, 1, 0].real, s_pred[:, 1, 0].imag]
    )
    target_vec = np.concatenate(
        [s_target[:, 0, 0].real, s_target[:, 0, 0].imag, s_target[:, 1, 0].real, s_target[:, 1, 0].imag]
    )
    return float(np.mean((pred_vec - target_vec) ** 2) / np.mean(target_vec**2))


def wrapped_phase_mae_deg(s_pred: np.ndarray, s_target: np.ndarray) -> float:
    d11 = np.angle(s_pred[:, 0, 0] * np.conj(s_target[:, 0, 0]))
    d21 = np.angle(s_pred[:, 1, 0] * np.conj(s_target[:, 1, 0]))
    return float(np.mean(np.abs(np.concatenate([d11, d21]))) * 180.0 / np.pi)


def main() -> None:
    train = load_module(TRAIN_SCRIPT, "v11_dut72_ablation_train")
    source = train.load_module(train.SOURCE_SCRIPT, "v11_dut72_ablation_source")
    positive = train.load_module(train.POSITIVE_SCRIPT, "v11_dut72_ablation_positive")
    wrapper = train.load_module(train.WRAPPER_SCRIPT, "v11_dut72_ablation_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_dut72_ablation_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    output_dir = version_root / "results" / RUN_LABEL
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
    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = train.ADS_DEVICE_LENGTH_SCALE
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)
    omega = 2.0 * np.pi * sim.freq_hz
    target_s = sim.target_s[0]

    pred = pd.read_csv(output_dir / "positive_symmetric_multihead_lc_fmin60_phase_predictions.csv", encoding="utf-8-sig")
    pred_row = pred[pred["sample_id"].astype(str).eq(SAMPLE_ID)].iloc[0]
    pred_cols = [f"pred_{col}" for col in train.multihead_target_columns(wrapper)]
    pred_params = pred_row[pred_cols].to_numpy(dtype=np.float64)
    opt_shared = opt_targets.loc[0, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    opt_multi = np.tile(opt_shared, wrapper.CONNECTION_COUNT)

    def evaluate(params: np.ndarray) -> tuple[float, float]:
        s_pred = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[0], omega, params))
        return ri_nmse(s_pred, target_s), wrapped_phase_mae_deg(s_pred, target_s)

    names = list(wrapper.V08_PARAM_NAMES)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    experiments: list[dict[str, float | str]] = []

    def add(name: str, params: np.ndarray) -> None:
        nmse, phase_mae = evaluate(params)
        experiments.append({"experiment": name, "nmse_s11_s21_ri": nmse, "wrapped_phase_mae_deg": phase_mae})

    add("pred_all", pred_params.copy())
    add("opt_all", opt_multi.copy())
    replace_groups = [
        ["Cn3_scale"],
        ["Ln1_scale"],
        ["Cn3_scale", "Ln1_scale"],
        ["Rn1_scale", "Rn2_scale"],
        ["Rn3_scale"],
        ["Rn1_scale", "Rn2_scale", "Rn3_scale"],
        ["Cn1_scale", "Cn2_scale", "Cn3_scale", "Ln1_scale"],
        names,
    ]
    for group in replace_groups:
        arr = pred_params.copy().reshape(wrapper.CONNECTION_COUNT, len(names))
        opt = opt_multi.reshape(wrapper.CONNECTION_COUNT, len(names))
        for param in group:
            arr[:, name_to_idx[param]] = opt[:, name_to_idx[param]]
        add("pred_replace_" + "+".join(group), arr.ravel())

    reverse_groups = [
        ["Cn3_scale", "Ln1_scale"],
        ["Rn1_scale", "Rn2_scale", "Rn3_scale"],
        ["Cn3_scale", "Ln1_scale", "Rn1_scale", "Rn2_scale", "Rn3_scale"],
    ]
    for group in reverse_groups:
        arr = opt_multi.copy().reshape(wrapper.CONNECTION_COUNT, len(names))
        pred_arr = pred_params.copy().reshape(wrapper.CONNECTION_COUNT, len(names))
        for param in group:
            arr[:, name_to_idx[param]] = pred_arr[:, name_to_idx[param]]
        add("opt_plus_pred_" + "+".join(group), arr.ravel())

    result = pd.DataFrame(experiments).sort_values("nmse_s11_s21_ri")
    out_path = output_dir / "dut72_parameter_ablation.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False), flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
