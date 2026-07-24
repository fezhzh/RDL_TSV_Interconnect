# -*- coding: utf-8 -*-
"""Train v10 ADS pi cascade with all 800 LHS800 samples.

Run this file directly in VS Code. No command-line arguments are required.

Dataset split:
- train: all 800 samples from HFSS_sim/LHS800/train/TSV_RDL
- test: the same fixed 50-sample LHS200 holdout used by the current v10 run
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_ads_pi_cascade_v10.py"
RUN_LABEL = "ads_pi_cascade_lhs800train_lhs200test_signed_pi_adslen09"
LHS800_TRAIN_COUNT = 800
LHS200_TEST_COUNT = 50
RANDOM_SPLIT_SEED = 20260707


def load_train_module():
    spec = importlib.util.spec_from_file_location("v10_train_ads_pi_cascade", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training script: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_lhs800_collect_samples(mod):
    def load_lhs800_rows() -> pd.DataFrame:
        split_dir = mod.PROJECT_ROOT / "HFSS_sim" / "LHS800" / "train"
        csv_path = split_dir / "TSV_RDL_variations_record.csv"
        design_dir = split_dir / "TSV_RDL"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing variation table: {csv_path}")
        if not design_dir.exists():
            raise FileNotFoundError(f"Missing S-parameter directory: {design_dir}")

        var_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        var_by_dut = {int(row["dut_index"]): row for _, row in var_df.iterrows()}
        rows = []
        for snp_path in sorted(design_dir.glob("dut*.s2p"), key=mod.lhs_base.natural_key):
            idx = mod.lhs_base.dut_index(snp_path)
            if idx not in var_by_dut:
                continue
            rec = var_by_dut[idx]
            row = {
                "sample_id": f"LHS800_train_dut{idx}",
                "source_root": "LHS800",
                "source_split": "train",
                "split": "train",
                "file": snp_path.name,
                "dut_index": int(idx),
                "variant": "lhs_sparam_only",
                "snp_path": str(snp_path),
            }
            for col in mod.STRUCTURE_COLUMNS:
                row[col] = float(rec[col])
            rows.append(row)
        return pd.DataFrame(rows)

    def collect_samples() -> pd.DataFrame:
        lhs800 = load_lhs800_rows()
        if len(lhs800) < LHS800_TRAIN_COUNT:
            raise ValueError(f"LHS800/train has {len(lhs800)} samples, but {LHS800_TRAIN_COUNT} are required.")
        train_df = lhs800.sort_values("dut_index").head(LHS800_TRAIN_COUNT).copy()

        df = mod.lhs_base.load_lhs_dataframe()
        df = df.sort_values(["split", "source_root", "dut_index"]).reset_index(drop=True)
        lhs200 = df[df["split"].eq("train") & df["source_root"].eq("LHS200")].copy()
        if len(lhs200) < mod.LHS200_MODEL_COUNT + LHS200_TEST_COUNT:
            raise ValueError("LHS200/train does not have enough samples for the fixed holdout test set.")
        shuffled = lhs200.sample(n=mod.LHS200_MODEL_COUNT + LHS200_TEST_COUNT, random_state=RANDOM_SPLIT_SEED)
        test_df = shuffled.iloc[mod.LHS200_MODEL_COUNT : mod.LHS200_MODEL_COUNT + LHS200_TEST_COUNT].copy()

        train_df["split"] = "train"
        test_df["split"] = "test"
        out = pd.concat([train_df, test_df.sort_values("dut_index")], ignore_index=True)
        missing = [col for col in mod.STRUCTURE_COLUMNS if col not in out.columns]
        if missing:
            raise ValueError(f"Sample table is missing columns: {missing}")
        return out.reset_index(drop=True)

    return collect_samples


def main():
    mod = load_train_module()
    output_dir = mod.PROJECT_ROOT / "model_versions" / "v10_ads_pi_cascade" / "results" / RUN_LABEL

    mod.RUN_LABEL = RUN_LABEL
    mod.OUTPUT_DIR = output_dir
    mod.ADS_CACHE_DIR = output_dir / "ads_single_device_cache"
    mod.collect_samples = make_lhs800_collect_samples(mod)

    # Keep the current best small v10 network and unbounded parameter output.
    # Increase S-parameter training enough to exploit the larger training set.
    mod.PARAM_EPOCHS = 180
    mod.PARAM_PATIENCE = 45
    mod.SPARAM_EPOCHS = 180
    mod.SPARAM_PATIENCE = 45
    mod.SPARAM_LR = 1e-5
    mod.BATCH_SIZE = 16
    mod.PARAM_ANCHOR_WEIGHT = 0.0
    mod.PLOT_SPLIT = "test"

    summary = mod.run_once(mod.BASE_ADS_SETTINGS)
    report_path = output_dir / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(
            {
                "dataset_note": "train uses all 800 LHS800/train samples; test uses the fixed 50-sample LHS200 holdout",
                "lhs800_train_count": LHS800_TRAIN_COUNT,
                "lhs200_test_count": LHS200_TEST_COUNT,
                "random_split_seed": RANDOM_SPLIT_SEED,
            }
        )
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(output_dir / "validation_archive.md", "a", encoding="utf-8") as f:
        f.write("\n## LHS800 Training Configuration\n\n")
        f.write(f"- Train: all `{LHS800_TRAIN_COUNT}` samples from `HFSS_sim/LHS800/train/TSV_RDL`\n")
        f.write(f"- Test: fixed `{LHS200_TEST_COUNT}`-sample LHS200 holdout using seed `{RANDOM_SPLIT_SEED}`\n")
        f.write("- Network: current small v10 PiConnectionNet\n")
        f.write("- Parameter output constraint: none after denormalization\n")
        f.write("\n")

    print("LHS800 run summary:", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
