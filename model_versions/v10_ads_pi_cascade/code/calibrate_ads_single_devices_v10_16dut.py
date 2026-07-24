# -*- coding: utf-8 -*-
"""16-DUT ADS single-device calibration for v10.

Run this file directly in VS Code. No command-line arguments are required.
This pass reuses the refined candidate search but evaluates a wider, evenly
spaced subset of LHS200 DUTs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REFINED_SCRIPT = THIS_DIR / "calibrate_ads_single_devices_v10_refined.py"
PROJECT_ROOT = THIS_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / "ads_single_device_calibration_16dut"
SAMPLE_DUTS = [100, 113, 126, 140, 153, 166, 180, 193, 206, 219, 233, 246, 259, 273, 286, 299]


def load_refined_module():
    spec = importlib.util.spec_from_file_location("v10_ads_single_device_calibration_refined", REFINED_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load refined calibration script: {REFINED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    refined = load_refined_module()
    refined.OUTPUT_DIR = OUTPUT_DIR
    refined.SAMPLE_DUTS = SAMPLE_DUTS
    refined.main()


if __name__ == "__main__":
    main()
