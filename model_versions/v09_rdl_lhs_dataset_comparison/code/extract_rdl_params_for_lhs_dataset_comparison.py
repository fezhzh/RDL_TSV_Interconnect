# -*- coding: utf-8 -*-
"""Extract RDL circuit-parameter datasets for LHS data-size comparison.

Run directly in VS Code. No command-line arguments are required.

Each dataset uses its own train source and the same validation/test source:
LHS100/val and LHS100/test. This keeps the accuracy comparison consistent,
because LHS200/LHS400/LHS800 only contain train samples.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skrf as rf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_EXTRACTOR_PATH = (
    PROJECT_ROOT
    / "model_versions"
    / "v00_parameter_extraction_and_dataset_building"
    / "code"
    / "提参3.py"
)
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "results" / "extracted_params"

DEVICE_CONFIGS = {
    "TMRDL": {
        "features": ["pitch", "l_tmrdl", "w_tmrdl", "h_tmrdl"],
        "length_column": "l_tmrdl",
    },
    "BSMRDL": {
        "features": ["pitch", "l_bsmrdl", "w_bsmrdl", "h_bsmrdl"],
        "length_column": "l_bsmrdl",
    },
}
TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
LHS_ROOTS = {
    "LHS100": PROJECT_ROOT / "HFSS_sim" / "LHS100",
    "LHS200": PROJECT_ROOT / "HFSS_sim" / "LHS200",
    "LHS400": PROJECT_ROOT / "HFSS_sim" / "LHS400",
    "LHS800": PROJECT_ROOT / "HFSS_sim" / "LHS800",
}
DATASET_CONFIGS = {
    "lhs100": ["LHS100"],
    "lhs200": ["LHS200"],
    "lhs400": ["LHS400"],
    "lhs800": ["LHS800"],
    "lhs100_lhs200_lhs400_lhs800": ["LHS100", "LHS200", "LHS400", "LHS800"],
}
FORCE_REEXTRACT_UNIQUE = False
WRITE_PROGRESS_EVERY = 50


SPEC = importlib.util.spec_from_file_location("base_extractor", BASE_EXTRACTOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load base extractor: {BASE_EXTRACTOR_PATH}")
BASE_EXTRACTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE_EXTRACTOR)


def snp_path(root, split_name, device_name, dut_index):
    folder = root / split_name / device_name
    matches = sorted(folder.glob(f"dut{int(dut_index)}.s*p"))
    return matches[0] if matches else None


def load_variation_rows(root_label, split_name, device_name, model_split):
    root = LHS_ROOTS[root_label]
    record_file = root / split_name / f"{device_name}_variations_record.csv"
    if not record_file.exists():
        return []
    df = pd.read_csv(record_file, encoding="utf-8-sig")
    rows = []
    for row_number, row in df.iterrows():
        dut_index = int(row["dut_index"]) if "dut_index" in df.columns else int(row_number)
        path = snp_path(root, split_name, device_name, dut_index)
        if path is None:
            continue
        item = row.to_dict()
        item.update(
            {
                "source_root": root_label,
                "source_split": split_name,
                "split": model_split,
                "idx": dut_index,
                "snp_path": str(path),
            }
        )
        rows.append(item)
    return rows


def build_unique_rows(device_name):
    rows = []
    for root_label in LHS_ROOTS:
        rows.extend(load_variation_rows(root_label, "train", device_name, "train"))
    rows.extend(load_variation_rows("LHS100", "val", device_name, "val"))
    rows.extend(load_variation_rows("LHS100", "test", device_name, "test"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No Snp rows found for {device_name}")
    return df.drop_duplicates(subset=["snp_path"]).reset_index(drop=True)


def extract_circuit_params_from_snp(path, length_um):
    nw = rf.Network(str(path))
    s = nw.s
    freq = nw.f
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    a, b, c, d = BASE_EXTRACTOR.S_ABCD(s11, s12, s21, s22)
    r_l, l_l, g_l, c_l, _, _ = BASE_EXTRACTOR.ABCD_RLGC(a, b, c, d, freq, float(length_um) * 1e-6)
    result = BASE_EXTRACTOR.RLGC_SPICE_rlgc_way3(
        r_l,
        l_l,
        g_l,
        c_l,
        float(length_um) * 1e-6,
        freq,
        p1=0,
        p2=len(freq) - 1,
    )
    params = np.asarray(result[8], dtype=np.float64)
    fit_s = np.stack([result[0], result[1], result[2], result[3]], axis=-1).reshape(len(freq), 2, 2)
    return params, float(result[9]), float(np.mean(np.abs(fit_s - s) ** 2))


def extract_unique_device(device_name):
    out_file = OUTPUT_DIR / "_unique" / f"{device_name}_all_unique_circuit_params.csv"
    if out_file.exists() and not FORCE_REEXTRACT_UNIQUE:
        print(f"Loaded unique extraction table: {out_file}")
        return pd.read_csv(out_file, encoding="utf-8-sig")

    config = DEVICE_CONFIGS[device_name]
    df = build_unique_rows(device_name)
    rows = []
    failed_rows = []
    for n, row in df.iterrows():
        try:
            params, rlgc_rmse, s_mse = extract_circuit_params_from_snp(
                row["snp_path"],
                float(row[config["length_column"]]),
            )
            item = row.to_dict()
            for name, value in zip(TARGET_PARAMS, params):
                item[name] = float(value)
            item["extract_rlgc_rmse"] = rlgc_rmse
            item["extract_s_mse"] = s_mse
            rows.append(item)
        except Exception as exc:
            failed_rows.append(
                {
                    "device": device_name,
                    "source_root": row.get("source_root"),
                    "source_split": row.get("source_split"),
                    "idx": row.get("idx"),
                    "snp_path": row.get("snp_path"),
                    "error": str(exc),
                }
            )
            print(f"[extract skip] {device_name} {row.get('source_root')} dut{row.get('idx')}: {exc}", flush=True)
        if (n + 1) % WRITE_PROGRESS_EVERY == 0:
            print(f"  extracted {device_name}: {n + 1}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    if failed_rows:
        failed_file = OUTPUT_DIR / "_unique" / f"{device_name}_failed_rows.csv"
        pd.DataFrame(failed_rows).to_csv(failed_file, index=False, encoding="utf-8-sig")
    print(f"{device_name}: saved {len(out)} unique rows to {out_file}")
    return out


def write_dataset_tables(device_name, unique_df):
    summary_rows = []
    val_test_df = unique_df[
        (unique_df["source_root"].eq("LHS100")) & (unique_df["split"].isin(["val", "test"]))
    ].copy()

    for dataset_name, train_roots in DATASET_CONFIGS.items():
        train_df = unique_df[
            unique_df["source_root"].isin(train_roots) & unique_df["source_split"].eq("train")
        ].copy()
        train_df["split"] = "train"
        out = pd.concat([train_df, val_test_df], ignore_index=True)
        out = out.sort_values(["split", "source_root", "idx"]).reset_index(drop=True)
        dataset_dir = OUTPUT_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_file = dataset_dir / f"{device_name}_circuit_params.csv"
        out.to_csv(out_file, index=False, encoding="utf-8-sig")
        counts = out.groupby(["source_root", "split"]).size().reset_index(name="count")
        for _, row in counts.iterrows():
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "device": device_name,
                    "source_root": row["source_root"],
                    "split": row["split"],
                    "count": int(row["count"]),
                    "csv_path": str(out_file),
                }
            )
        print(f"{dataset_name} {device_name}: saved {len(out)} rows to {out_file}")
    return summary_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for device_name in DEVICE_CONFIGS:
        unique_df = extract_unique_device(device_name)
        summary_rows.extend(write_dataset_tables(device_name, unique_df))
    summary = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "rdl_extraction_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Summary saved to: {summary_file}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
