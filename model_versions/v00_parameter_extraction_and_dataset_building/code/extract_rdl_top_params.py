# -*- coding: utf-8 -*-
"""Extract equivalent circuit parameters for RDL_Top_Snp files."""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from 提参3 import ABCD_RLGC, RLGC_SPICE_rlgc_way3, S_ABCD, path_S2P


CSV_HEADERS = [
    "l_rdl",
    "w_rdl",
    "t_rdl",
    "h_tsv",
    "p_rdl",
    "R1",
    "R2",
    "R3",
    "L1",
    "L2",
    "L3",
    "Cox",
    "Csi",
    "Rsi",
    "rmse",
    "scale_max",
    "scale_min",
]


def parse_s2p_variables(path):
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


def sorted_dut_files(input_dir):
    def dut_index(path):
        match = re.search(r"dut(\d+)\.s2p$", path.name, re.IGNORECASE)
        return int(match.group(1)) if match else 10**12

    return sorted(input_dir.glob("dut*.s2p"), key=dut_index)


def extract_one(momentum_path):
    variables = parse_s2p_variables(momentum_path)
    required = ["lrdl", "wrdl", "trdl", "htsv", "p1"]
    missing = [name for name in required if name not in variables]
    if missing:
        raise ValueError(f"{momentum_path} missing variables: {missing}")

    l_rdl = variables["lrdl"]
    w_rdl = variables["wrdl"]
    t_rdl = variables["trdl"]
    h_tsv = variables["htsv"]
    p_rdl = variables["p1"]
    parameter_l = l_rdl * 1e-6

    s11, s12, s21, s22, _, _, _, _, freq = path_S2P(str(momentum_path))
    p1 = 0
    p2 = len(freq) - 1

    a, b, c, d = S_ABCD(s11, s12, s21, s22)
    r_l, l_l, g_l, c_l, _, _ = ABCD_RLGC(a, b, c, d, freq, parameter_l)
    result = RLGC_SPICE_rlgc_way3(r_l, l_l, g_l, c_l, parameter_l, freq, p1=p1, p2=p2)
    parameter_spice, rmse, scale_max, scale_min = result[8], result[9], result[10], result[11]

    return [l_rdl, w_rdl, t_rdl, h_tsv, p_rdl] + list(parameter_spice) + [
        rmse,
        scale_max,
        scale_min,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=PROJECT_ROOT / "data" / "sparameters" / "RDL_Top_Snp",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "data" / "tables" / "RDL_Top_TD_4.csv",
        type=Path,
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    rows = []
    files = sorted_dut_files(args.input_dir)
    if args.limit is not None:
        files = files[: args.limit]

    for path in files:
        try:
            rows.append(extract_one(path))
            df = pd.DataFrame(rows, columns=CSV_HEADERS)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output, index=False)
            print(f"{path.name} parameters saved to: {args.output}")
        except Exception as exc:
            print(f"{path} failed, skipped: {exc}")


if __name__ == "__main__":
    main()
