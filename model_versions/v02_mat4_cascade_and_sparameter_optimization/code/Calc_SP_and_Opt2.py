# -*- coding: utf-8 -*-
"""Batch optimize cascaded RDL_TSV S-parameters with mat4 device models.

Run this file directly in VS Code. It reads every ``dut*.s2p`` under
``snp_data/RDL_TSV_Snp``, builds the direct RDL/TSV cascade from mat4 device
models, inserts correction networks between adjacent blocks, optimizes those
correction networks against the full HFSS S-parameters, and saves the resulting
Touchstone files locally.
"""

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import skrf as rf
from scipy.optimize import least_squares

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
S2P_DIR = PROJECT_ROOT / "snp_data" / "RDL_TSV_Snp"
MAT_DIR = PROJECT_ROOT / "model_versions" / "v01_matlab_mat_models" / "models" / "RDL_TSV_mat4"
OUTPUT_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "results" / "RDL_TSV_mat4_opt2"

# Configure these values before running directly from VS Code.
PROCESS_ALL_SAMPLES = True
START_DUT = 1
END_DUT = 10
MAX_SAMPLES = None  # Use an integer for a quick subset, or None for no limit.

SAVE_TOUCHSTONE = True
SAVE_DIRECT_CASCADE = True
SAVE_CONNECTION_PARAMS = True
WRITE_SUMMARY_CSV = True
SHOW_PLOTS = False

PARALLEL_OPTIMIZATION = True
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)
MAX_NFEV = 300

DEVICE_LENGTH_SCALE = 0.95
OPTIMIZATION_BOUNDS = (-1e5, 1e5)
WITH_CN3_P0 = [0.01, 1000.0, 0.01, 1000.0, 0.01, 0.01, 0.01]
WITH_CN3_LOWER = [0.0, 900.0, 0.0, 900.0, 0.0, 0.0, 0.0]
WITH_CN3_UPPER = [300.0, 1100.0, 300.0, 1100.0, 300.0, 200.0, 300.0]
WITHOUT_CN3_P0 = [0.01, 1000.0, 0.01, 1000.0, 0.01, 0.01]
WITHOUT_CN3_LOWER = [0.0, 900.0, 0.0, 900.0, 0.0, 0.0]
WITHOUT_CN3_UPPER = [300.0, 1100.0, 300.0, 1100.0, 200.0, 300.0]
WITHOUT_CN3_REG_WEIGHT = 0.0
BOUNDARY_RTOL = 1e-5
BOUNDARY_ATOL = 1e-8

BASE_DEVICE_SEQUENCE = [
    "RDL_Top",
    "TSV",
    "RDL_Bottom",
    "TSV",
    "RDL_Top",
    "TSV",
    "RDL_Bottom",
    "TSV",
    "RDL_Top",
]

STRUCTURE_PARAM_NAMES = ["lrdl", "wrdl", "trdl", "ldown", "wdown", "tdown", "dtsv", "htsv", "p1"]


def s2abcd(S, Z0=50.0):
    S11, S12 = S[:, 0, 0], S[:, 0, 1]
    S21, S22 = S[:, 1, 0], S[:, 1, 1]

    denom = 2 * S21 + 1e-15

    A = ((1 + S11) * (1 - S22) + S12 * S21) / denom
    B = Z0 * ((1 + S11) * (1 + S22) - S12 * S21) / denom
    C_mat = (1 / Z0) * ((1 - S11) * (1 - S22) - S12 * S21) / denom
    D = ((1 - S11) * (1 + S22) + S12 * S21) / denom

    ABCD = np.zeros_like(S, dtype=complex)
    ABCD[:, 0, 0], ABCD[:, 0, 1] = A, B
    ABCD[:, 1, 0], ABCD[:, 1, 1] = C_mat, D
    return ABCD


def abcd2s(ABCD, Z0=50.0):
    A, B = ABCD[:, 0, 0], ABCD[:, 0, 1]
    C_mat, D = ABCD[:, 1, 0], ABCD[:, 1, 1]

    denom = A + B / Z0 + C_mat * Z0 + D
    S = np.zeros_like(ABCD, dtype=complex)
    S[:, 0, 0] = (A + B / Z0 - C_mat * Z0 - D) / denom
    S[:, 0, 1] = 2 * (A * D - B * C_mat) / denom
    S[:, 1, 0] = 2 / denom
    S[:, 1, 1] = (-A + B / Z0 - C_mat * Z0 + D) / denom
    return S


def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).stem)
    return [int(part) if part.isdigit() else part for part in parts]


def dut_index(path):
    match = re.search(r"dut(\d+)\.s2p$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse DUT index from {path}")
    return int(match.group(1))


def s2p_files():
    if PROCESS_ALL_SAMPLES:
        files = sorted(S2P_DIR.glob("dut*.s2p"), key=natural_key)
    else:
        files = [S2P_DIR / f"dut{idx}.s2p" for idx in range(START_DUT, END_DUT + 1)]
        files = [path for path in files if path.exists()]

    if MAX_SAMPLES is not None:
        files = files[:MAX_SAMPLES]
    if not files:
        raise FileNotFoundError(f"No dut*.s2p files found under {S2P_DIR}")
    return files


def extract_device_params_RDL_TSV(filepath):
    params = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                break
            if line.startswith("!"):
                line = line[1:].strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    match = re.search(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
    return params


def predict_circuit_parameters(features, mat_dir, param_names, prefix):
    circuit_params = {}
    x = features.reshape(1, -1)
    for param in param_names:
        mat_filepath = Path(mat_dir) / f"{prefix}{param}.mat"
        if not mat_filepath.exists():
            circuit_params[param] = 1.0
            continue

        mat_data = sio.loadmat(mat_filepath)
        xmin, xmax = mat_data["psmin"], mat_data["psmax"]
        ymin, ymax = mat_data["outputmin"], mat_data["outputmax"]
        w1, b1 = mat_data["w1"], mat_data["theta1"]
        w2, b2 = mat_data["w2"], mat_data["theta2"]
        w3, b3 = mat_data["w3"], mat_data["theta3"]

        x_norm = 2.0 * (x - xmin) / (xmax - xmin + 1e-12) - 1.0
        a1 = np.tanh(np.dot(x_norm, w1) + b1)
        a2 = np.tanh(np.dot(a1, w2) + b2)
        y_norm = np.dot(a2, w3) + b3
        y_real = ymin + (y_norm + 1.0) * (ymax - ymin) / 2.0
        circuit_params[param] = float(y_real.flatten()[0])
    return circuit_params


def calculate_S_parameters(circuit_params, length_um, freqs):
    R1, R2, R3 = circuit_params["R1"], circuit_params["R2"], circuit_params["R3"]
    L1 = circuit_params["L1"] * 1e-9
    L2 = circuit_params["L2"] * 1e-9
    L3 = circuit_params["L3"] * 1e-9
    Cox, Csi = circuit_params["Cox"] * 1e-12, circuit_params["Csi"] * 1e-12
    Rsi = circuit_params["Rsi"]

    length_m = length_um * 1e-6
    omega = 2 * np.pi * freqs

    R_RLGC = (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2) / (
        (R1 + R2) ** 2 + omega**2 * L2**2
    ) + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    L_RLGC = (R1**2 * L2) / ((R1 + R2) ** 2 + omega**2 * L2**2) + L3 * R3**2 / (
        R3**2 + omega**2 * L3**2
    ) + L1
    G_RLGC = (omega**2 * Rsi * Cox**2) / (1 + omega**2 * Rsi**2 * (Cox + Csi) ** 2)
    C_RLGC = (Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)) / (
        1 + omega**2 * Rsi**2 * (Cox + Csi) ** 2
    )

    z0 = np.sqrt((R_RLGC + 1j * omega * L_RLGC) / (G_RLGC + 1j * omega * C_RLGC))
    gamma = np.sqrt((R_RLGC + 1j * omega * L_RLGC) * (G_RLGC + 1j * omega * C_RLGC))

    ABCD = np.zeros((len(freqs), 2, 2), dtype=complex)
    ABCD[:, 0, 0] = np.cosh(gamma * length_m)
    ABCD[:, 0, 1] = z0 * np.sinh(gamma * length_m)
    ABCD[:, 1, 0] = (1 / z0) * np.sinh(gamma * length_m)
    ABCD[:, 1, 1] = np.cosh(gamma * length_m)
    return abcd2s(ABCD)


def get_correction_abcd(p, omega):
    Cn1 = p[0] * 1e-14
    Rn1 = p[1] * 1e3
    Cn2 = p[2] * 1e-14
    Rn2 = p[3] * 1e3
    Cn3 = p[4] * 1e-14
    Rn3 = p[5] * 1.0
    Ln1 = p[6] * 1e-11

    Y1 = 1j * omega * Cn1 + 1.0 / Rn1
    Y2 = 1j * omega * Cn2 + 1.0 / Rn2
    Y3 = 1j * omega * Cn3 + 1.0 / (Rn3 + 1j * omega * Ln1)

    ABCD = np.zeros((len(omega), 2, 2), dtype=complex)
    ABCD[:, 0, 0] = 1.0 + Y2 / Y3
    ABCD[:, 0, 1] = 1.0 / Y3
    ABCD[:, 1, 0] = Y1 + Y2 + Y1 * Y2 / Y3
    ABCD[:, 1, 1] = 1.0 + Y1 / Y3
    return ABCD


def correction_component_values(p):
    return {
        "Cn1_F": float(p[0] * 1e-14),
        "Rn1_ohm": float(p[1] * 1e3),
        "Cn2_F": float(p[2] * 1e-14),
        "Rn2_ohm": float(p[3] * 1e3),
        "Cn3_F": float(p[4] * 1e-14),
        "Rn3_ohm": float(p[5] * 1.0),
        "Ln1_H": float(p[6] * 1e-11),
        "Cn1_scale": float(p[0]),
        "Rn1_scale": float(p[1]),
        "Cn2_scale": float(p[2]),
        "Rn2_scale": float(p[3]),
        "Cn3_scale": float(p[4]),
        "Rn3_scale": float(p[5]),
        "Ln1_scale": float(p[6]),
    }


def boundary_flags(values, lower, upper):
    values = np.asarray(values, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower_hit = values <= lower + np.maximum(BOUNDARY_ATOL, BOUNDARY_RTOL * np.maximum(np.abs(lower), 1.0))
    upper_hit = values >= upper - np.maximum(BOUNDARY_ATOL, BOUNDARY_RTOL * np.maximum(np.abs(upper), 1.0))
    return lower_hit, upper_hit


def boundary_summary(values, lower, upper):
    lower_hit, upper_hit = boundary_flags(values, lower, upper)
    return {
        "hit_lower_count": int(lower_hit.sum()),
        "hit_upper_count": int(upper_hit.sum()),
        "hit_any_count": int((lower_hit | upper_hit).sum()),
        "hit_any": bool((lower_hit | upper_hit).any()),
    }


def connection_param_rows(idx, file_name, structure_params, variant, p_all, include_cn3, lower_bounds, upper_bounds):
    rows = []
    structure_values = {f"structure_{name}": structure_params.get(name, np.nan) for name in STRUCTURE_PARAM_NAMES}

    for i in range(8):
        if include_cn3:
            p_i = p_all[i * 7 : (i + 1) * 7]
            lower_i = lower_bounds[i * 7 : (i + 1) * 7]
            upper_i = upper_bounds[i * 7 : (i + 1) * 7]
        else:
            p_i_6 = p_all[i * 6 : (i + 1) * 6]
            p_i = [p_i_6[0], p_i_6[1], p_i_6[2], p_i_6[3], 0.0, p_i_6[4], p_i_6[5]]
            lower_i_6 = lower_bounds[i * 6 : (i + 1) * 6]
            upper_i_6 = upper_bounds[i * 6 : (i + 1) * 6]
            lower_i = [lower_i_6[0], lower_i_6[1], lower_i_6[2], lower_i_6[3], 0.0, lower_i_6[4], lower_i_6[5]]
            upper_i = [upper_i_6[0], upper_i_6[1], upper_i_6[2], upper_i_6[3], 0.0, upper_i_6[4], upper_i_6[5]]
        lower_hit, upper_hit = boundary_flags(p_i, lower_i, upper_i)

        row = {
            "file": file_name,
            "dut_index": idx,
            "variant": variant,
            "connection_index": i + 1,
            "left_device": BASE_DEVICE_SEQUENCE[i],
            "right_device": BASE_DEVICE_SEQUENCE[i + 1],
        }
        row.update(correction_component_values(p_i))
        for name, hit_l, hit_u in zip(["Cn1", "Rn1", "Cn2", "Rn2", "Cn3", "Rn3", "Ln1"], lower_hit, upper_hit):
            row[f"{name}_hit_lower_bound"] = bool(hit_l)
            row[f"{name}_hit_upper_bound"] = bool(hit_u)
        row["connection_hit_any_bound"] = bool((lower_hit | upper_hit).any())
        row["connection_hit_bound_count"] = int((lower_hit | upper_hit).sum())
        row.update(structure_values)
        rows.append(row)

    return rows


def mse(ref_s, pred_s):
    return float(np.mean(np.abs(ref_s - pred_s) ** 2))


def build_base_abcds(params, freqs):
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    features_top = np.array([params["lrdl"], params["wrdl"], params["trdl"], params["htsv"], params["p1"]])
    features_bot = np.array([params["ldown"], params["wdown"], params["tdown"], params["htsv"], params["p1"]])
    features_tsv = np.array([params["dtsv"], params["htsv"], params["p1"]])

    cp_top = predict_circuit_parameters(features_top, MAT_DIR, target_params, prefix="RDL_Top_")
    cp_bot = predict_circuit_parameters(features_bot, MAT_DIR, target_params, prefix="RDL_Bottom_")
    cp_tsv = predict_circuit_parameters(features_tsv, MAT_DIR, target_params, prefix="TSV_")

    abcd_top = s2abcd(calculate_S_parameters(cp_top, params["lrdl"] * DEVICE_LENGTH_SCALE, freqs))
    abcd_bot = s2abcd(calculate_S_parameters(cp_bot, params["ldown"] * DEVICE_LENGTH_SCALE, freqs))
    abcd_tsv = s2abcd(calculate_S_parameters(cp_tsv, params["htsv"] * DEVICE_LENGTH_SCALE, freqs))

    return [abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top]


def cascade_direct(base_abcds):
    result = base_abcds[0]
    for next_abcd in base_abcds[1:]:
        result = np.matmul(result, next_abcd)
    return result


def cascade_with_corrections(base_abcds, omega, p_all, include_cn3):
    result = base_abcds[0]
    for i in range(8):
        if include_cn3:
            p_i = p_all[i * 7 : (i + 1) * 7]
        else:
            p_i_6 = p_all[i * 6 : (i + 1) * 6]
            p_i = [p_i_6[0], p_i_6[1], p_i_6[2], p_i_6[3], 0.0, p_i_6[4], p_i_6[5]]
        result = np.matmul(np.matmul(result, get_correction_abcd(p_i, omega)), base_abcds[i + 1])
    return result


def make_network(hfss_nw, s_matrix, name):
    return rf.Network(frequency=hfss_nw.frequency, s=s_matrix, name=name)


def save_network(nw, out_dir, dut_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / dut_name
    nw.write_touchstone(filename=filename)
    return filename.with_suffix(".s2p")


def plot_s_comparison(hfss_nw, direct_nw, with_cn3_nw, without_cn3_nw, title_suffix=""):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    hfss_nw.plot_s_db(m=0, n=0, color="blue", linewidth=2, label="HFSS")
    direct_nw.plot_s_db(m=0, n=0, color="gray", linestyle=":", label="Direct")
    without_cn3_nw.plot_s_db(m=0, n=0, color="orange", linestyle="-.", label="Opt w/o Cn3")
    with_cn3_nw.plot_s_db(m=0, n=0, color="red", linestyle="--", label="Opt with Cn3")
    plt.title(f"S11 Magnitude {title_suffix}")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    hfss_nw.plot_s_db(m=1, n=0, color="blue", linewidth=2, label="HFSS")
    direct_nw.plot_s_db(m=1, n=0, color="gray", linestyle=":", label="Direct")
    without_cn3_nw.plot_s_db(m=1, n=0, color="orange", linestyle="-.", label="Opt w/o Cn3")
    with_cn3_nw.plot_s_db(m=1, n=0, color="red", linestyle="--", label="Opt with Cn3")
    plt.title(f"S21 Magnitude {title_suffix}")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def optimize_one_snp_worker(s2p_file):
    return optimize_one_snp(Path(s2p_file))


def optimize_one_snp(s2p_file):
    idx = dut_index(s2p_file)
    dut_name = f"dut{idx}"

    print(f"\n==================================================")
    print(f">>> 开始优化 {s2p_file.name}")

    hfss_nw = rf.Network(str(s2p_file))
    freqs = hfss_nw.f
    omega = 2 * np.pi * freqs
    target_s = hfss_nw.s

    params = extract_device_params_RDL_TSV(s2p_file)
    base_abcds = build_base_abcds(params, freqs)

    direct_s = abcd2s(cascade_direct(base_abcds))
    direct_nw = make_network(hfss_nw, direct_s, "Direct_Cascade")

    def objective_with_cn3(p_all):
        pred_s = abcd2s(cascade_with_corrections(base_abcds, omega, p_all, include_cn3=True))
        error = pred_s - target_s
        return np.concatenate([error.real.ravel(), error.imag.ravel()])

    p0_7 = np.tile(WITH_CN3_P0, 8)
    lower_7 = np.full_like(p0_7, OPTIMIZATION_BOUNDS[0], dtype=np.float64)
    upper_7 = np.full_like(p0_7, OPTIMIZATION_BOUNDS[1], dtype=np.float64)
    print(">>> 正在优化包含 Cn3 的完整修正网络...")
    res_with_cn3 = least_squares(objective_with_cn3, p0_7, bounds=(lower_7, upper_7), max_nfev=MAX_NFEV)
    with_cn3_s = abcd2s(cascade_with_corrections(base_abcds, omega, res_with_cn3.x, include_cn3=True))
    with_cn3_nw = make_network(hfss_nw, with_cn3_s, "Opt_With_Cn3")
    with_cn3_bounds = boundary_summary(res_with_cn3.x, lower_7, upper_7)

    def objective_without_cn3(p_all_6):
        pred_s = abcd2s(cascade_with_corrections(base_abcds, omega, p_all_6, include_cn3=False))
        error = pred_s - target_s
        return np.concatenate([error.real.ravel(), error.imag.ravel()])

    p0_6 = np.tile(WITHOUT_CN3_P0, 8)
    lower_6 = np.full_like(p0_6, OPTIMIZATION_BOUNDS[0], dtype=np.float64)
    upper_6 = np.full_like(p0_6, OPTIMIZATION_BOUNDS[1], dtype=np.float64)
    print(">>> 正在优化剔除 Cn3 的简化修正网络...")
    res_without_cn3 = least_squares(objective_without_cn3, p0_6, bounds=(lower_6, upper_6), max_nfev=MAX_NFEV)
    without_cn3_s = abcd2s(cascade_with_corrections(base_abcds, omega, res_without_cn3.x, include_cn3=False))
    without_cn3_nw = make_network(hfss_nw, without_cn3_s, "Opt_Without_Cn3")
    without_cn3_bounds = boundary_summary(res_without_cn3.x, lower_6, upper_6)

    direct_mse = mse(target_s, direct_s)
    with_cn3_mse = mse(target_s, with_cn3_s)
    without_cn3_mse = mse(target_s, without_cn3_s)

    row = {
        "file": s2p_file.name,
        "dut_index": idx,
        "direct_mse": direct_mse,
        "optimized_with_cn3_mse": with_cn3_mse,
        "optimized_without_cn3_mse": without_cn3_mse,
        "with_cn3_improvement_pct": (direct_mse - with_cn3_mse) / direct_mse * 100 if direct_mse else np.nan,
        "without_cn3_improvement_pct": (direct_mse - without_cn3_mse) / direct_mse * 100 if direct_mse else np.nan,
        "with_cn3_cost": float(res_with_cn3.cost),
        "without_cn3_cost": float(res_without_cn3.cost),
        "with_cn3_nfev": int(res_with_cn3.nfev),
        "without_cn3_nfev": int(res_without_cn3.nfev),
        "with_cn3_success": bool(res_with_cn3.success),
        "without_cn3_success": bool(res_without_cn3.success),
        "with_cn3_hit_bound": with_cn3_bounds["hit_any"],
        "with_cn3_hit_bound_count": with_cn3_bounds["hit_any_count"],
        "with_cn3_hit_lower_count": with_cn3_bounds["hit_lower_count"],
        "with_cn3_hit_upper_count": with_cn3_bounds["hit_upper_count"],
        "without_cn3_hit_bound": without_cn3_bounds["hit_any"],
        "without_cn3_hit_bound_count": without_cn3_bounds["hit_any_count"],
        "without_cn3_hit_lower_count": without_cn3_bounds["hit_lower_count"],
        "without_cn3_hit_upper_count": without_cn3_bounds["hit_upper_count"],
        "device_length_scale": DEVICE_LENGTH_SCALE,
        "with_cn3_positive_bounds": False,
        "without_cn3_positive_bounds": False,
        "optimization_lower_bound": OPTIMIZATION_BOUNDS[0],
        "optimization_upper_bound": OPTIMIZATION_BOUNDS[1],
        "without_cn3_reg_weight": WITHOUT_CN3_REG_WEIGHT,
    }
    row.update({f"structure_{name}": params.get(name, np.nan) for name in STRUCTURE_PARAM_NAMES})

    component_rows = []
    component_rows.extend(
        connection_param_rows(
            idx,
            s2p_file.name,
            params,
            "optimized_with_cn3",
            res_with_cn3.x,
            include_cn3=True,
            lower_bounds=lower_7,
            upper_bounds=upper_7,
        )
    )
    component_rows.extend(
        connection_param_rows(
            idx,
            s2p_file.name,
            params,
            "optimized_without_cn3",
            res_without_cn3.x,
            include_cn3=False,
            lower_bounds=lower_6,
            upper_bounds=upper_6,
        )
    )

    if SAVE_TOUCHSTONE:
        if SAVE_DIRECT_CASCADE:
            row["direct_s2p"] = str(save_network(direct_nw, OUTPUT_DIR / "direct", dut_name))
        row["optimized_with_cn3_s2p"] = str(save_network(with_cn3_nw, OUTPUT_DIR / "optimized_with_cn3", dut_name))
        row["optimized_without_cn3_s2p"] = str(save_network(without_cn3_nw, OUTPUT_DIR / "optimized_without_cn3", dut_name))

    if SAVE_CONNECTION_PARAMS:
        connection_dir = OUTPUT_DIR / "connection_params"
        connection_dir.mkdir(parents=True, exist_ok=True)
        connection_csv = connection_dir / f"{dut_name}_connection_params.csv"
        pd.DataFrame(component_rows).to_csv(connection_csv, index=False, encoding="utf-8-sig")
        row["connection_params_csv"] = str(connection_csv)

    print(f">>> 直接级联 MSE: {direct_mse:.4e}")
    print(f">>> 优化后 MSE（含 Cn3）: {with_cn3_mse:.4e}")
    print(f">>> 优化后 MSE（无 Cn3）: {without_cn3_mse:.4e}")

    if SHOW_PLOTS:
        plot_s_comparison(hfss_nw, direct_nw, with_cn3_nw, without_cn3_nw, title_suffix=f"({s2p_file.name})")

    return row, component_rows


def collect_result(rows, all_component_rows, row, component_rows):
    rows.append(row)
    all_component_rows.extend(component_rows)


def write_boundary_hit_summary(component_df):
    param_names = ["Cn1", "Rn1", "Cn2", "Rn2", "Cn3", "Rn3", "Ln1"]
    rows = []
    for (variant, connection_index), group in component_df.groupby(["variant", "connection_index"], sort=True):
        for name in param_names:
            lower_col = f"{name}_hit_lower_bound"
            upper_col = f"{name}_hit_upper_bound"
            if lower_col not in group.columns or upper_col not in group.columns:
                continue
            lower_hits = group[lower_col].astype(bool)
            upper_hits = group[upper_col].astype(bool)
            any_hits = lower_hits | upper_hits
            rows.append(
                {
                    "variant": variant,
                    "connection_index": int(connection_index),
                    "parameter": name,
                    "n": int(len(group)),
                    "hit_lower_count": int(lower_hits.sum()),
                    "hit_upper_count": int(upper_hits.sum()),
                    "hit_any_count": int(any_hits.sum()),
                    "hit_lower_rate": float(lower_hits.mean()),
                    "hit_upper_rate": float(upper_hits.mean()),
                    "hit_any_rate": float(any_hits.mean()),
                }
            )
    summary_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "boundary_hit_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main():
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = s2p_files()
    print(f"将处理 {len(files)} 个 S 参数文件")
    print(f"HFSS 输入目录: {S2P_DIR}")
    print(f"mat4 模型目录: {MAT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    if PARALLEL_OPTIMIZATION and not SHOW_PLOTS and len(files) > 1 and MAX_WORKERS > 1:
        print(f"并行优化: 开启，worker 数量 = {MAX_WORKERS}")
    else:
        print("并行优化: 关闭，使用顺序执行")

    rows = []
    all_component_rows = []

    use_parallel = PARALLEL_OPTIMIZATION and not SHOW_PLOTS and len(files) > 1 and MAX_WORKERS > 1
    if use_parallel:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(optimize_one_snp_worker, str(s2p_file)): s2p_file for s2p_file in files}
            for n_done, future in enumerate(as_completed(future_to_file), start=1):
                s2p_file = future_to_file[future]
                try:
                    row, component_rows = future.result()
                    collect_result(rows, all_component_rows, row, component_rows)
                    print(f">>> 完成 {n_done}/{len(files)}: {s2p_file.name}")
                except Exception as exc:
                    print(f"[失败] {s2p_file.name}: {exc}")
                    rows.append({"file": s2p_file.name, "dut_index": dut_index(s2p_file), "error": str(exc)})
    else:
        for n_done, s2p_file in enumerate(files, start=1):
            try:
                row, component_rows = optimize_one_snp(s2p_file)
                collect_result(rows, all_component_rows, row, component_rows)
                print(f">>> 完成 {n_done}/{len(files)}: {s2p_file.name}")
            except Exception as exc:
                print(f"[失败] {s2p_file.name}: {exc}")
                rows.append({"file": s2p_file.name, "dut_index": dut_index(s2p_file), "error": str(exc)})

    rows.sort(key=lambda item: item.get("dut_index", 10**12))
    all_component_rows.sort(
        key=lambda item: (
            item.get("dut_index", 10**12),
            item.get("variant", ""),
            item.get("connection_index", 10**12),
        )
    )

    summary_df = pd.DataFrame(rows)
    if WRITE_SUMMARY_CSV:
        summary_csv = OUTPUT_DIR / "optimization_summary.csv"
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"\n汇总 CSV: {summary_csv}")

    if SAVE_CONNECTION_PARAMS and all_component_rows:
        connection_csv = OUTPUT_DIR / "connection_network_params.csv"
        component_df = pd.DataFrame(all_component_rows)
        component_df.to_csv(connection_csv, index=False, encoding="utf-8-sig")
        print(f"连接网络元件值 CSV: {connection_csv}")
        boundary_csv = write_boundary_hit_summary(component_df)
        print(f"边界命中统计 CSV: {boundary_csv}")

    valid_df = summary_df[summary_df.get("error").isna()] if "error" in summary_df else summary_df
    print("\n全部处理完成")
    print(f"有效样本: {len(valid_df)} / {len(summary_df)}")
    if len(valid_df):
        print(f"平均直接级联 MSE: {valid_df['direct_mse'].mean():.4e}")
        print(f"平均优化 MSE（含 Cn3）: {valid_df['optimized_with_cn3_mse'].mean():.4e}")
        print(f"平均优化 MSE（无 Cn3）: {valid_df['optimized_without_cn3_mse'].mean():.4e}")


if __name__ == "__main__":
    main()
