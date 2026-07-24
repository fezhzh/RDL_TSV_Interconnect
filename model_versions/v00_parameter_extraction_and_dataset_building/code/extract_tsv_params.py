# -*- coding: utf-8 -*-
"""Extract equivalent circuit parameters for TSV_Snp files.

The extraction flow follows the shared functions from the RDL Bottom extractor:
S-parameters -> ABCD -> RLGC -> SPICE circuit parameters.
"""

import importlib.util
import os
import re
from pathlib import Path

import matplotlib
import pandas as pd

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_EXTRACTOR_PATH = next(path for path in SCRIPT_DIR.glob("*3.py") if path.name != Path(__file__).name)

# Configure these values before running directly from VS Code.
INPUT_DIR = PROJECT_ROOT / "snp_data" / "TSV_Snp"
OUTPUT_CSV = PROJECT_ROOT / "training_datasets" / "TSV_TD_4.csv"

WRITE_CSV = False
WRITE_PLOTS = True

# Use None for all files. Set to an integer, such as 10, for a quick subset.
LIMIT = None

# Number of extracted cases to show diagnostic plots for. Use 0 for all cases.
PLOT_LIMIT = 0

BASE_EXTRACTOR_SPEC = importlib.util.spec_from_file_location("base_extractor", BASE_EXTRACTOR_PATH)
if BASE_EXTRACTOR_SPEC is None or BASE_EXTRACTOR_SPEC.loader is None:
    raise ImportError(f"Cannot load base extractor: {BASE_EXTRACTOR_PATH}")
BASE_EXTRACTOR = importlib.util.module_from_spec(BASE_EXTRACTOR_SPEC)
BASE_EXTRACTOR_SPEC.loader.exec_module(BASE_EXTRACTOR)

ABCD_RLGC = BASE_EXTRACTOR.ABCD_RLGC
RLGC_SPICE_rlgc_way3 = BASE_EXTRACTOR.RLGC_SPICE_rlgc_way3
S_ABCD = BASE_EXTRACTOR.S_ABCD
path_S2P = BASE_EXTRACTOR.path_S2P
plot_extraction_comparison = BASE_EXTRACTOR.plot_extraction_comparison


CSV_HEADERS = [
    "d_tsv",
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


def show_extraction_plots(rlgcs, sparameters, freqs, names):
    plot_extraction_comparison(
        RLGCs=rlgcs,
        sparameters=sparameters,
        freqs=freqs,
        names=names,
        show=True,
    )


def extract_one(momentum_path, show_plots=False):
    variables = parse_s2p_variables(momentum_path)
    required = ["dtsv", "htsv", "p1"]
    missing = [name for name in required if name not in variables]
    if missing:
        raise ValueError(f"{momentum_path} missing variables: {missing}")

    d_tsv = variables["dtsv"]
    h_tsv = variables["htsv"]
    p_rdl = variables["p1"]
    parameter_l = h_tsv * 1e-6

    s11, s12, s21, s22, _, _, _, _, freq = path_S2P(str(momentum_path))
    sp_hfss = BASE_EXTRACTOR.np.array([[s11, s12], [s21, s22]])
    p1 = 0
    p2 = len(freq) - 1

    a, b, c, d = S_ABCD(s11, s12, s21, s22)
    r_l, l_l, g_l, c_l, _, _ = ABCD_RLGC(a, b, c, d, freq, parameter_l)
    rlgc_hfss = [r_l, l_l, g_l, c_l]
    result = RLGC_SPICE_rlgc_way3(r_l, l_l, g_l, c_l, parameter_l, freq, p1=p1, p2=p2)
    s11_fit, s12_fit, s21_fit, s22_fit = result[0], result[1], result[2], result[3]
    r_all, l_all, c_all, g_all = result[4], result[5], result[6], result[7]
    parameter_spice, rmse, scale_max, scale_min = result[8], result[9], result[10], result[11]

    if show_plots:
        sp_model = BASE_EXTRACTOR.np.array([[s11_fit, s12_fit], [s21_fit, s22_fit]])
        rlgc_model = [r_all, l_all, g_all, c_all]
        show_extraction_plots(
            rlgcs=[rlgc_hfss, rlgc_model],
            sparameters=[sp_hfss, sp_model],
            freqs=[freq, freq],
            names=["HFSS", "Model"],
        )

    return [d_tsv, h_tsv, p_rdl] + list(parameter_spice) + [
        rmse,
        scale_max,
        scale_min,
    ]


def main():
    if not WRITE_CSV and not WRITE_PLOTS:
        raise ValueError("At least one output must be enabled: set WRITE_CSV or WRITE_PLOTS to True.")

    os.chdir(PROJECT_ROOT)
    rows = []
    files = sorted_dut_files(INPUT_DIR)
    if LIMIT is not None:
        files = files[:LIMIT]

    for path in files:
        try:
            should_plot = WRITE_PLOTS and (PLOT_LIMIT == 0 or len(rows) < PLOT_LIMIT)
            rows.append(extract_one(path, show_plots=should_plot))
            if WRITE_CSV:
                df = pd.DataFrame(rows, columns=CSV_HEADERS)
                OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"{path.name} parameters saved to: {OUTPUT_CSV}")
            elif should_plot:
                print(f"{path.name} plots shown")
            else:
                print(f"{path.name} extracted")
        except Exception as exc:
            print(f"{path} failed, skipped: {exc}")


if __name__ == "__main__":
    main()
