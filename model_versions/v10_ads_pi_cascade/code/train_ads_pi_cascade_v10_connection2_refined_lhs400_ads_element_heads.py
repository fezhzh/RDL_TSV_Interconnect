# -*- coding: utf-8 -*-
"""Train v10 pi-cascade with element-wise shared-trunk multi-head networks.

Run this file directly in VS Code. No command-line arguments are required.

The ADS settings and modeling flow match
``train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py``. The only model
change is the current ``PiConnectionNet`` architecture in
``train_ads_pi_cascade_v10.py``:
for each element type, a shared 9->30->30 trunk is followed by per-connection
30->20->1 heads.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10.py"
WRAPPER_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py"
SOURCE_CACHE_RUN_LABEL = "ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    base = load_module(BASE_SCRIPT, "v10_ads_pi_cascade_train_element_heads")
    wrapper = load_module(WRAPPER_SCRIPT, "v10_ads_pi_cascade_train_refined_wrapper")

    base.RUN_LABEL = "ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_element_heads"
    base.OUTPUT_DIR = base.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / base.RUN_LABEL
    base.ADS_CACHE_DIR = (
        base.PROJECT_ROOT
        / "model_versions"
        / "v10_ads_pi_cascade"
        / "results"
        / SOURCE_CACHE_RUN_LABEL
        / "ads_single_device_cache"
    )

    base.SIMULATION_BACKEND = "ads"
    base.LHS200_MODEL_COUNT = 150
    base.LHS200_TEST_COUNT = 50
    base.LHS200_RANDOM_SPLIT_SEED = 20260707
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0
    base.collect_samples = lambda: wrapper.collect_connection2_samples(base)

    sweep = {
        "freq_start_ghz": 0.1,
        "freq_stop_ghz": 100.0,
        "freq_step_ghz": 0.1,
    }
    base.BASE_ADS_SETTINGS = {
        "calibration_source": "ac_l400_ref2",
        "dataset": wrapper.DATASET_NAME,
        "single_device_calibration_dataset": "LHS400_Connection2/train",
        "network_architecture": "element-wise shared 9-30-30 trunks with 30-20-1 per-connection heads",
        "rdl_settings": {
            "er_si": 9.8,
            "cond": 5.8e7,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.65,
            "pitch_scale": 1.25,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 1.0,
            **sweep,
        },
        "tsv_settings": {
            "er_si": 11.9,
            "cond": 5.8e7,
            "tand": 0.005,
            "c1_scale": 1.0,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.2,
            "d_scale": 1.0,
            **sweep,
        },
    }
    base.RUN_MATERIAL_SWEEP = False
    base.main()


if __name__ == "__main__":
    main()
