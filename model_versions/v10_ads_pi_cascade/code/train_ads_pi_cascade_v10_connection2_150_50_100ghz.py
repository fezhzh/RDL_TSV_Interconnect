# -*- coding: utf-8 -*-
"""Train v10 pi-cascade on LHS150_50_Connection2 from 0.1 to 100 GHz.

Run this file directly in VS Code. No command-line arguments are required.
The modeling flow is unchanged from `train_ads_pi_cascade_v10.py`:
ADS single-device simulation -> pi optimization -> pi-parameter pretraining ->
S-parameter fine-tuning.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10.py"
DATASET_NAME = "LHS150_50_Connection2"


def load_base_module():
    spec = importlib.util.spec_from_file_location("v10_ads_pi_cascade_train", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base training script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_connection2_samples(base) -> pd.DataFrame:
    rows = []
    dataset_root = base.PROJECT_ROOT / "HFSS_sim" / DATASET_NAME
    for split in ["train", "test"]:
        split_dir = dataset_root / split
        csv_path = split_dir / "TSV_RDL_variations_record.csv"
        snp_dir = split_dir / "TSV_RDL"
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for _, row in df.sort_values("dut_index").iterrows():
            dut_index = int(row["dut_index"])
            rec = row.to_dict()
            rec["dut_index"] = dut_index
            rec["h_tmrdl"] = float(rec.pop("t_tmrdl"))
            rec["h_bsmrdl"] = float(rec.pop("t_bsmrdl"))
            rec["split"] = split
            rec["source_root"] = DATASET_NAME
            rec["sample_id"] = f"{DATASET_NAME}_{split}_dut{dut_index}"
            rec["file"] = f"dut{dut_index}.s2p"
            rec["snp_path"] = snp_dir / rec["file"]
            rows.append(rec)
    out = pd.DataFrame(rows)
    missing = [col for col in base.STRUCTURE_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"{DATASET_NAME} sample table is missing columns: {missing}")
    missing_files = [str(path) for path in out["snp_path"] if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing TSV_RDL S-parameter files: {missing_files[:5]}")
    return out.reset_index(drop=True)


def main():
    base = load_base_module()
    base.RUN_LABEL = "ads_pi_cascade_lhs150_50_connection2_100ghz_calibrated16dut"
    base.OUTPUT_DIR = (
        base.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / base.RUN_LABEL
    )
    base.ADS_CACHE_DIR = base.OUTPUT_DIR / "ads_single_device_cache"

    base.SIMULATION_BACKEND = "ads"
    base.LHS200_MODEL_COUNT = 150
    base.LHS200_TEST_COUNT = 50
    base.LHS200_RANDOM_SPLIT_SEED = 20260707
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = 1.0
    base.collect_samples = lambda: collect_connection2_samples(base)

    sweep = {
        "freq_start_ghz": 0.1,
        "freq_stop_ghz": 100.0,
        "freq_step_ghz": 0.1,
    }
    base.BASE_ADS_SETTINGS = {
        "calibration_source": "ads_single_device_calibration_16dut",
        "dataset": DATASET_NAME,
        "rdl_settings": {
            "er_si": 10.2,
            "cond": 5.8e7,
            "tand": 0.005,
            "l_scale": 1.0,
            "w_scale": 0.85,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.0,
            "h_rdl_scale": 0.8,
            **sweep,
        },
        "tsv_settings": {
            "er_si": 11.9,
            "cond": 5.8e7,
            "tand": 0.005,
            "c1_scale": 1.0,
            "pitch_scale": 1.0,
            "h_tsv_scale": 1.1,
            "d_scale": 1.0,
            **sweep,
        },
    }
    base.RUN_MATERIAL_SWEEP = False
    base.main()


if __name__ == "__main__":
    main()
