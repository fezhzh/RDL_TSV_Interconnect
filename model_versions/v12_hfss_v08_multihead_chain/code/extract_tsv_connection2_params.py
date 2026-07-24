# -*- coding: utf-8 -*-
"""Extract TSV equivalent-circuit parameters for LHS400_Connection2.

Run this file directly in VS Code. No command-line arguments are required.
The output is version-local under v12 results and does not overwrite the old
`training_datasets/TSV_TD_4.csv` dataset.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
BASE_DIR = PROJECT_ROOT / "model_versions" / "v00_parameter_extraction_and_dataset_building" / "code"
BASE_EXTRACTOR_PATH = next(path for path in BASE_DIR.glob("*3.py") if path.name != Path(__file__).name)
INPUT_DIR = PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train" / "TSV"
VARIATION_CSV = PROJECT_ROOT / "HFSS_sim" / "LHS400_Connection2" / "train" / "TSV_variations_record.csv"
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v12_hfss_v08_multihead_chain" / "results" / "tsv_connection2_extracted_params"
OUTPUT_CSV = OUTPUT_DIR / "TSV_connection2_circuit_params.csv"

CSV_HEADERS = [
    "dut_index",
    "file",
    "snp_path",
    "r_tsv",
    "h_tsv",
    "pitch",
    "R1",
    "R2",
    "R3",
    "L1",
    "L2",
    "L3",
    "Cox",
    "Csi",
    "Rsi",
    "extract_rmse",
    "scale_max",
    "scale_min",
]


def load_base_extractor():
    spec = importlib.util.spec_from_file_location("v12_base_extractor", BASE_EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base extractor: {BASE_EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_extractor()
ABCD_RLGC = BASE.ABCD_RLGC
RLGC_SPICE_rlgc_way3 = BASE.RLGC_SPICE_rlgc_way3
S_ABCD = BASE.S_ABCD
path_S2P = BASE.path_S2P


def parse_s2p_variables(path: Path) -> dict[str, float]:
    variables = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                break
            if not line.startswith("!") or "=" not in line:
                continue
            var_name, rest = line[1:].split("=", 1)
            match = re.search(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", rest)
            if match:
                variables[var_name.strip()] = float(match.group(1))
    return variables


def extract_one(path: Path, dut_index_from_name: int, variation_row: pd.Series | None) -> list[object]:
    variables = parse_s2p_variables(path)
    if variation_row is not None:
        r_tsv = float(variation_row["r_tsv"])
        h_tsv = float(variation_row["h_tsv"])
        pitch = float(variation_row["pitch"])
        dut_index = int(dut_index_from_name)
    else:
        r_tsv = float(variables["r_tsv"])
        h_tsv = float(variables.get("h_tsv", variables.get("h_sub")))
        pitch = float(variables["pitch"])
        dut_index = int(re.search(r"dut(\d+)", path.stem).group(1))

    s11, s12, s21, s22, _, _, _, _, freq = path_S2P(str(path))
    a, b, c, d = S_ABCD(s11, s12, s21, s22)
    length_m = h_tsv * 1e-6
    r_l, l_l, g_l, c_l, _, _ = ABCD_RLGC(a, b, c, d, freq, length_m)
    result = RLGC_SPICE_rlgc_way3(r_l, l_l, g_l, c_l, length_m, freq, p1=0, p2=len(freq) - 1)
    parameter_spice, rmse, scale_max, scale_min = result[8], result[9], result[10], result[11]
    return [
        dut_index,
        path.name,
        str(path),
        r_tsv,
        h_tsv,
        pitch,
        *[float(value) for value in parameter_spice],
        float(rmse),
        float(scale_max),
        float(scale_min),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variation_df = pd.read_csv(VARIATION_CSV, encoding="utf-8-sig").set_index("dut_index")
    rows = []
    failures = []
    for i, path in enumerate(sorted(INPUT_DIR.glob("dut*.s2p"), key=lambda p: int(re.search(r"dut(\d+)", p.stem).group(1))), start=1):
        dut_index = int(re.search(r"dut(\d+)", path.stem).group(1))
        try:
            row = variation_df.loc[dut_index] if dut_index in variation_df.index else None
            rows.append(extract_one(path, dut_index, row))
        except Exception as exc:
            failures.append({"dut_index": dut_index, "file": path.name, "error": str(exc)})
        if i == 1 or i % 50 == 0:
            print(f"extracted {i}/{len(list(INPUT_DIR.glob('dut*.s2p')))} TSV Connection2 files", flush=True)

    out = pd.DataFrame(rows, columns=CSV_HEADERS).sort_values("dut_index").reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    failure_df = pd.DataFrame(failures)
    failure_csv = OUTPUT_DIR / "TSV_connection2_extraction_failures.csv"
    failure_df.to_csv(failure_csv, index=False, encoding="utf-8-sig")
    summary = {
        "input_dir": str(INPUT_DIR),
        "variation_csv": str(VARIATION_CSV),
        "output_csv": str(OUTPUT_CSV),
        "n_success": int(len(out)),
        "n_failed": int(len(failure_df)),
        "r_tsv_min": float(out["r_tsv"].min()) if len(out) else None,
        "r_tsv_max": float(out["r_tsv"].max()) if len(out) else None,
        "extract_rmse_mean": float(out["extract_rmse"].mean()) if len(out) else None,
        "extract_rmse_median": float(out["extract_rmse"].median()) if len(out) else None,
    }
    (OUTPUT_DIR / "extraction_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 TSV Connection2 Parameter Extraction",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Input: `{INPUT_DIR}`",
                f"- Output CSV: `{OUTPUT_CSV}`",
                f"- Success: `{len(out)}`",
                f"- Failed: `{len(failure_df)}`",
                f"- Mean extraction RMSE: `{summary['extract_rmse_mean']}`",
                f"- Median extraction RMSE: `{summary['extract_rmse_median']}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
