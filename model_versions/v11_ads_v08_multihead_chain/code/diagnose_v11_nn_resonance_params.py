# -*- coding: utf-8 -*-
"""Diagnose which predicted v11 circuit parameters cause NN resonances."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = THIS_DIR / "optimize_v11_shared_connection_calibrated.py"
WRAPPER_SCRIPT = THIS_DIR / "train_v11_shared7_to_multihead12.py"
RESULT_LABEL = "v11_shared7_param_nns_all_goodstart"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    wrapper = load_module(WRAPPER_SCRIPT, "diag_resonance_wrapper")
    source = load_module(SOURCE_SCRIPT, "diag_resonance_source")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "diag_resonance_base")
    root = base.PROJECT_ROOT
    result_dir = root / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / RESULT_LABEL
    base.ADS_CACHE_DIR = root / "model_versions" / "v11_ads_v08_multihead_chain" / "results" / "v11_sharedopt_c30" / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.ADS_DEVICE_LENGTH_SCALE = 1.0
    dut_df = wrapper.collect_v11_samples(base)
    sim = base.load_single_device_simulation(dut_df, source.calibrated_ads_settings())
    omega = 2.0 * np.pi * sim.freq_hz
    freq_ghz = sim.freq_hz / 1e9
    pred = pd.read_csv(result_dir / "shared7_param_predictions.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(result_dir / "optimized_vs_shared7_nn_metrics.csv", encoding="utf-8-sig")

    def branch_diag(p):
        p = np.asarray(p, dtype=np.float64) * wrapper.V08_SCALE_FACTORS
        cn1, rn1, cn2, rn2, cn3, rn3, ln1 = p
        y1 = 1j * omega * cn1 + 1.0 / (rn1 + 1e-30)
        y2 = 1j * omega * cn2 + 1.0 / (rn2 + 1e-30)
        y3 = 1j * omega * cn3 + 1.0 / (rn3 + 1j * omega * ln1 + 1e-30)
        b = 1.0 / y3
        a = 1.0 + y2 / y3
        d = 1.0 + y1 / y3
        return {
            "min_abs_y1": float(np.min(np.abs(y1))),
            "freq_min_y1": float(freq_ghz[np.argmin(np.abs(y1))]),
            "min_abs_y2": float(np.min(np.abs(y2))),
            "freq_min_y2": float(freq_ghz[np.argmin(np.abs(y2))]),
            "min_abs_y3": float(np.min(np.abs(y3))),
            "freq_min_y3": float(freq_ghz[np.argmin(np.abs(y3))]),
            "max_abs_inv_y3": float(np.max(np.abs(b))),
            "freq_max_inv_y3": float(freq_ghz[np.argmax(np.abs(b))]),
            "max_abs_A": float(np.max(np.abs(a))),
            "freq_max_A": float(freq_ghz[np.argmax(np.abs(a))]),
            "max_abs_D": float(np.max(np.abs(d))),
            "freq_max_D": float(freq_ghz[np.argmax(np.abs(d))]),
        }

    rows = []
    for _, metric in metrics.iterrows():
        sample_id = str(metric["sample_id"])
        row = pred[pred["sample_id"].eq(sample_id)].iloc[0]
        target_p = [float(row[f"target_{name}"]) for name in wrapper.V08_PARAM_NAMES]
        nn_p = [float(row[f"pred_{name}"]) for name in wrapper.V08_PARAM_NAMES]
        rec = {
            "sample_id": sample_id,
            "split": metric["split"],
            "optimized_nmse_s11_s21_ri": float(metric["optimized_nmse_s11_s21_ri"]),
            "nn_nmse_s11_s21_ri": float(metric["nn_nmse_s11_s21_ri"]),
        }
        rec.update({f"target_{key}": value for key, value in branch_diag(target_p).items()})
        rec.update({f"nn_{key}": value for key, value in branch_diag(nn_p).items()})
        for idx, name in enumerate(wrapper.V08_PARAM_NAMES):
            rec[f"target_{name}"] = target_p[idx]
            rec[f"pred_{name}"] = nn_p[idx]
            rec[f"delta_{name}"] = nn_p[idx] - target_p[idx]
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = result_dir / "nn_resonance_parameter_diagnostics.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    worst = out.sort_values("nn_nmse_s11_s21_ri", ascending=False).head(12)
    print(worst[
        [
            "sample_id",
            "split",
            "nn_nmse_s11_s21_ri",
            "optimized_nmse_s11_s21_ri",
            "nn_min_abs_y3",
            "nn_freq_min_y3",
            "nn_max_abs_inv_y3",
            "target_min_abs_y3",
            "pred_Cn3_scale",
            "target_Cn3_scale",
            "pred_Rn3_scale",
            "target_Rn3_scale",
            "pred_Ln1_scale",
            "target_Ln1_scale",
            "pred_Rn1_scale",
            "target_Rn1_scale",
            "pred_Rn2_scale",
            "target_Rn2_scale",
        ]
    ].to_string(index=False))
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
