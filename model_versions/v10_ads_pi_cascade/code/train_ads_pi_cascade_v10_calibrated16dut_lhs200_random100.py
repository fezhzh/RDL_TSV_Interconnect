# -*- coding: utf-8 -*-
"""Train v10 pi-cascade with 16-DUT calibrated ADS settings on LHS200 100/100.

Run this file directly in VS Code. No command-line arguments are required.
The modeling flow is unchanged from `train_ads_pi_cascade_v10.py`:
ADS single-device simulation -> pi optimization -> pi-parameter pretraining ->
S-parameter fine-tuning.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("v10_ads_pi_cascade_train", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base training script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    base = load_base_module()
    base.RUN_LABEL = "ads_pi_cascade_lhs200_random100train_100test_calibrated16dut"
    base.OUTPUT_DIR = (
        base.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / base.RUN_LABEL
    )
    base.ADS_CACHE_DIR = base.OUTPUT_DIR / "ads_single_device_cache"

    base.SIMULATION_BACKEND = "ads"
    base.LHS200_MODEL_COUNT = 100
    base.LHS200_TEST_COUNT = 100
    base.LHS200_RANDOM_SPLIT_SEED = 20260707
    base.USE_MODEL_SET_AS_VALIDATION = True

    # The 16-DUT calibration was performed without the older global 0.9 length
    # multiplier. Keep this run consistent with the calibrated geometry scales.
    base.ADS_DEVICE_LENGTH_SCALE = 1.0

    base.BASE_ADS_SETTINGS = {
        "calibration_source": "ads_single_device_calibration_16dut",
        "rdl_settings": {
            "er_si": 10.2,
            "cond": 5.8e7,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.85,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 0.8,
        },
        "tsv_settings": {
            "er_si": 11.9,
            "cond": 5.8e7,
            "tand": 0.005,
            "c1_scale": 1.0,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.1,
            "d_scale": 1.0,
        },
    }
    base.RUN_MATERIAL_SWEEP = False
    base.main()


if __name__ == "__main__":
    main()
