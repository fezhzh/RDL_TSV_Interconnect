# -*- coding: utf-8 -*-
"""Optimize connection-network parameters for LHS TSV_RDL cascade samples.

Run this file directly in VS Code. It uses the current v09 single-device base
models to build the direct cascade, then optimizes the eight inserted
connection networks against HFSS S-parameters from LHS100/LHS200/LHS400.

Main output:
    model_versions/v09_rdl_lhs_dataset_comparison/results/
    connection_network_lhs100_200_400_opt2/connection_network_params.csv

The CSV is the supervised target table for the preliminary cascade-network
model: one structure sample -> eight optimized connection-network parameter
sets.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import skrf as rf
from scipy.optimize import least_squares

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V09_CODE_DIR = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "code"
V02_CODE_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "code"
for path in [V09_CODE_DIR, V02_CODE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Calc_SP_and_Opt2 as opt2
import train_lhs_connection_multihead_sparam as lhs_base


OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "connection_network_lhs100_200_400_opt2"
)

# Configure these values before running directly from VS Code.
MAX_SAMPLES = None  # Use an integer for a quick debug subset, or None for all samples.
PARALLEL_OPTIMIZATION = True
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
MAX_NFEV = 300
RESUME_FROM_EXISTING = True

SAVE_TOUCHSTONE = False
SAVE_DIRECT_CASCADE = False
SAVE_PER_SAMPLE_CONNECTION_PARAMS = True
WRITE_SUMMARY_CSV = True

DEVICE_LENGTH_SCALE = opt2.DEVICE_LENGTH_SCALE
OPTIMIZATION_BOUNDS = (-1e5, 1e5)
WITH_CN3_P0 = np.array([0.01, 1000.0, 0.01, 1000.0, 0.01, 0.01, 0.01], dtype=np.float64)
WITHOUT_CN3_P0 = np.array([0.01, 1000.0, 0.01, 1000.0, 0.01, 0.01], dtype=np.float64)
WITHOUT_CN3_REG_WEIGHT = 0.0
BOUNDARY_RTOL = 1e-5
BOUNDARY_ATOL = 1e-8
Z_REF = 50.0

BASE_DEVICE_SEQUENCE = [
    "TMRDL",
    "TSV",
    "BSMRDL",
    "TSV",
    "TMRDL",
    "TSV",
    "BSMRDL",
    "TSV",
    "TMRDL",
]

LHS_STRUCTURE_COLUMNS = [
    "pitch",
    "r_tsv",
    "h_tsv",
    "l_tmrdl",
    "w_tmrdl",
    "h_tmrdl",
    "l_bsmrdl",
    "w_bsmrdl",
    "h_bsmrdl",
]

LEGACY_STRUCTURE_MAP = {
    "structure_lrdl": "l_tmrdl",
    "structure_wrdl": "w_tmrdl",
    "structure_trdl": "h_tmrdl",
    "structure_ldown": "l_bsmrdl",
    "structure_wdown": "w_bsmrdl",
    "structure_tdown": "h_bsmrdl",
    "structure_htsv": "h_tsv",
    "structure_p1": "pitch",
}

WITH_CN3_SCALE_COLUMNS = ["Cn1_scale", "Rn1_scale", "Cn2_scale", "Rn2_scale", "Cn3_scale", "Rn3_scale", "Ln1_scale"]
WITHOUT_CN3_SCALE_COLUMNS = ["Cn1_scale", "Rn1_scale", "Cn2_scale", "Rn2_scale", "Rn3_scale", "Ln1_scale"]


def sample_connection_csv(sample_id):
    return OUTPUT_DIR / "connection_params" / f"{sample_id}_connection_params.csv"


def sample_summary_json(sample_id):
    return OUTPUT_DIR / "sample_summaries" / f"{sample_id}_summary.json"


def safe_sample_id(value):
    return str(value).replace("\\", "_").replace("/", "_").replace(":", "_")


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def mse(a, b):
    return float(np.mean(np.abs(a - b) ** 2))


def s2abcd(s):
    return opt2.s2abcd(s)


def abcd2s(abcd):
    return opt2.abcd2s(abcd)


def make_network(freq_hz, s, name):
    nw = rf.Network(frequency=rf.Frequency.from_f(freq_hz, unit="hz"), s=s, z0=Z_REF)
    nw.name = name
    return nw


def save_network(freq_hz, s, out_dir, sample_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}.s2p"
    make_network(freq_hz, s, sample_id).write_touchstone(str(out_path.with_suffix("")))
    return out_path


def correction_abcd_one(p, omega, include_cn3):
    if include_cn3:
        cn1 = p[0] * 1e-14
        rn1 = p[1] * 1e3
        cn2 = p[2] * 1e-14
        rn2 = p[3] * 1e3
        cn3 = p[4] * 1e-14
        rn3 = p[5] * 1.0
        ln1 = p[6] * 1e-11
        y3 = 1j * omega * cn3 + 1.0 / (rn3 + 1j * omega * ln1)
    else:
        cn1 = p[0] * 1e-14
        rn1 = p[1] * 1e3
        cn2 = p[2] * 1e-14
        rn2 = p[3] * 1e3
        rn3 = p[4] * 1.0
        ln1 = p[5] * 1e-11
        y3 = 1.0 / (rn3 + 1j * omega * ln1)

    y1 = 1j * omega * cn1 + 1.0 / rn1
    y2 = 1j * omega * cn2 + 1.0 / rn2
    abcd = np.zeros((len(omega), 2, 2), dtype=complex)
    abcd[:, 0, 0] = 1.0 + y2 / y3
    abcd[:, 0, 1] = 1.0 / y3
    abcd[:, 1, 0] = y1 + y2 + y1 * y2 / y3
    abcd[:, 1, 1] = 1.0 + y1 / y3
    return abcd


def cascade_direct(base_abcds):
    result = np.array(base_abcds[0], copy=True)
    for abcd in base_abcds[1:]:
        result = np.matmul(result, abcd)
    return result


def cascade_with_corrections(base_abcds, omega, p_flat, include_cn3):
    n_per_conn = 7 if include_cn3 else 6
    p_all = np.asarray(p_flat, dtype=np.float64).reshape(8, n_per_conn)
    result = np.array(base_abcds[0], copy=True)
    for i in range(8):
        result = np.matmul(np.matmul(result, correction_abcd_one(p_all[i], omega, include_cn3)), base_abcds[i + 1])
    return result


def residual_vector(p_flat, base_abcds, target_s, omega, include_cn3):
    pred_s = abcd2s(cascade_with_corrections(base_abcds, omega, p_flat, include_cn3))
    diff = pred_s - target_s
    residual = np.concatenate([diff.real.ravel(), diff.imag.ravel()])
    if (not include_cn3) and WITHOUT_CN3_REG_WEIGHT > 0:
        residual = np.concatenate([residual, np.sqrt(WITHOUT_CN3_REG_WEIGHT) * np.asarray(p_flat, dtype=np.float64)])
    return residual


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
        "hit_any": bool(np.any(lower_hit | upper_hit)),
    }


def component_values(p, include_cn3):
    if include_cn3:
        return {
            "Cn1_F": float(p[0] * 1e-14),
            "Rn1_ohm": float(p[1] * 1e3),
            "Cn2_F": float(p[2] * 1e-14),
            "Rn2_ohm": float(p[3] * 1e3),
            "Cn3_F": float(p[4] * 1e-14),
            "Rn3_ohm": float(p[5]),
            "Ln1_H": float(p[6] * 1e-11),
            "Cn1_scale": float(p[0]),
            "Rn1_scale": float(p[1]),
            "Cn2_scale": float(p[2]),
            "Rn2_scale": float(p[3]),
            "Cn3_scale": float(p[4]),
            "Rn3_scale": float(p[5]),
            "Ln1_scale": float(p[6]),
        }
    return {
        "Cn1_F": float(p[0] * 1e-14),
        "Rn1_ohm": float(p[1] * 1e3),
        "Cn2_F": float(p[2] * 1e-14),
        "Rn2_ohm": float(p[3] * 1e3),
        "Rn3_ohm": float(p[4]),
        "Ln1_H": float(p[5] * 1e-11),
        "Cn1_scale": float(p[0]),
        "Rn1_scale": float(p[1]),
        "Cn2_scale": float(p[2]),
        "Rn2_scale": float(p[3]),
        "Rn3_scale": float(p[4]),
        "Ln1_scale": float(p[5]),
    }


def structure_columns(row):
    out = {legacy_col: float(row[lhs_col]) for legacy_col, lhs_col in LEGACY_STRUCTURE_MAP.items()}
    out["structure_dtsv"] = float(row["r_tsv"]) * 2.0
    for col in LHS_STRUCTURE_COLUMNS:
        out[f"structure_{col}"] = float(row[col])
    return out


def connection_param_rows(task, variant, p_flat, include_cn3, lower_bounds, upper_bounds):
    n_per_conn = 7 if include_cn3 else 6
    scale_columns = WITH_CN3_SCALE_COLUMNS if include_cn3 else WITHOUT_CN3_SCALE_COLUMNS
    p_all = np.asarray(p_flat, dtype=np.float64).reshape(8, n_per_conn)
    lower_all = np.asarray(lower_bounds, dtype=np.float64).reshape(8, n_per_conn)
    upper_all = np.asarray(upper_bounds, dtype=np.float64).reshape(8, n_per_conn)
    rows = []
    for conn_idx in range(8):
        lower_hit, upper_hit = boundary_flags(p_all[conn_idx], lower_all[conn_idx], upper_all[conn_idx])
        row = {
            "sample_id": task["sample_id"],
            "file": task["file"],
            "snp_path": task["snp_path"],
            "dut_index": int(task["sample_index"]),
            "source_dut_index": int(task["dut_index"]),
            "source_root": task["source_root"],
            "source_split": task["source_split"],
            "split": task["split"],
            "variant": variant,
            "connection_index": conn_idx + 1,
            "device_length_scale": DEVICE_LENGTH_SCALE,
        }
        row.update(task["structure"])
        row.update(component_values(p_all[conn_idx], include_cn3))
        for j, name in enumerate(scale_columns):
            base_name = name.replace("_scale", "")
            row[f"{base_name}_lower_bound"] = float(lower_all[conn_idx, j])
            row[f"{base_name}_upper_bound"] = float(upper_all[conn_idx, j])
            row[f"{base_name}_hit_lower_bound"] = bool(lower_hit[j])
            row[f"{base_name}_hit_upper_bound"] = bool(upper_hit[j])
        rows.append(row)
    return rows


def load_existing_result(task):
    summary_path = sample_summary_json(task["sample_id"])
    connection_path = sample_connection_csv(task["sample_id"])
    if not (RESUME_FROM_EXISTING and summary_path.exists() and connection_path.exists()):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        row = json.load(f)
    component_rows = pd.read_csv(connection_path, encoding="utf-8-sig").to_dict(orient="records")
    row["resumed_from_existing"] = True
    return row, component_rows


def write_existing_result(task, row, component_rows):
    if not SAVE_PER_SAMPLE_CONNECTION_PARAMS:
        return
    connection_path = sample_connection_csv(task["sample_id"])
    summary_path = sample_summary_json(task["sample_id"])
    connection_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(component_rows).to_csv(connection_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)


def optimize_task(task):
    existing = load_existing_result(task)
    if existing is not None:
        return existing

    base_abcds = task["base_abcds"]
    target_s = task["target_s"]
    freq_hz = task["freq_hz"]
    omega = 2.0 * np.pi * freq_hz
    sample_id = task["sample_id"]

    direct_s = abcd2s(cascade_direct(base_abcds))
    lower_7 = np.full(8 * 7, OPTIMIZATION_BOUNDS[0], dtype=np.float64)
    upper_7 = np.full(8 * 7, OPTIMIZATION_BOUNDS[1], dtype=np.float64)
    lower_6 = np.full(8 * 6, OPTIMIZATION_BOUNDS[0], dtype=np.float64)
    upper_6 = np.full(8 * 6, OPTIMIZATION_BOUNDS[1], dtype=np.float64)

    res_with = least_squares(
        residual_vector,
        np.tile(WITH_CN3_P0, 8),
        args=(base_abcds, target_s, omega, True),
        bounds=(lower_7, upper_7),
        max_nfev=MAX_NFEV,
    )
    res_without = least_squares(
        residual_vector,
        np.tile(WITHOUT_CN3_P0, 8),
        args=(base_abcds, target_s, omega, False),
        bounds=(lower_6, upper_6),
        max_nfev=MAX_NFEV,
    )

    with_s = abcd2s(cascade_with_corrections(base_abcds, omega, res_with.x, True))
    without_s = abcd2s(cascade_with_corrections(base_abcds, omega, res_without.x, False))
    direct_mse = mse(target_s, direct_s)
    with_mse = mse(target_s, with_s)
    without_mse = mse(target_s, without_s)
    with_bounds = boundary_summary(res_with.x, lower_7, upper_7)
    without_bounds = boundary_summary(res_without.x, lower_6, upper_6)

    row = {
        "sample_id": sample_id,
        "file": task["file"],
        "snp_path": task["snp_path"],
        "dut_index": int(task["sample_index"]),
        "source_dut_index": int(task["dut_index"]),
        "source_root": task["source_root"],
        "source_split": task["source_split"],
        "split": task["split"],
        "direct_mse": direct_mse,
        "optimized_with_cn3_mse": with_mse,
        "optimized_without_cn3_mse": without_mse,
        "with_cn3_improvement_pct": (direct_mse - with_mse) / direct_mse * 100 if direct_mse else np.nan,
        "without_cn3_improvement_pct": (direct_mse - without_mse) / direct_mse * 100 if direct_mse else np.nan,
        "with_cn3_cost": float(res_with.cost),
        "without_cn3_cost": float(res_without.cost),
        "with_cn3_nfev": int(res_with.nfev),
        "without_cn3_nfev": int(res_without.nfev),
        "with_cn3_success": bool(res_with.success),
        "without_cn3_success": bool(res_without.success),
        "with_cn3_hit_bound": with_bounds["hit_any"],
        "with_cn3_hit_bound_count": with_bounds["hit_any_count"],
        "with_cn3_hit_lower_count": with_bounds["hit_lower_count"],
        "with_cn3_hit_upper_count": with_bounds["hit_upper_count"],
        "without_cn3_hit_bound": without_bounds["hit_any"],
        "without_cn3_hit_bound_count": without_bounds["hit_any_count"],
        "without_cn3_hit_lower_count": without_bounds["hit_lower_count"],
        "without_cn3_hit_upper_count": without_bounds["hit_upper_count"],
        "device_length_scale": DEVICE_LENGTH_SCALE,
        "optimization_lower_bound": OPTIMIZATION_BOUNDS[0],
        "optimization_upper_bound": OPTIMIZATION_BOUNDS[1],
        "without_cn3_reg_weight": WITHOUT_CN3_REG_WEIGHT,
        "resumed_from_existing": False,
    }
    row.update(task["structure"])

    component_rows = []
    component_rows.extend(connection_param_rows(task, "optimized_with_cn3", res_with.x, True, lower_7, upper_7))
    component_rows.extend(connection_param_rows(task, "optimized_without_cn3", res_without.x, False, lower_6, upper_6))

    if SAVE_TOUCHSTONE:
        if SAVE_DIRECT_CASCADE:
            row["direct_s2p"] = str(save_network(freq_hz, direct_s, OUTPUT_DIR / "direct", sample_id))
        row["optimized_with_cn3_s2p"] = str(save_network(freq_hz, with_s, OUTPUT_DIR / "optimized_with_cn3", sample_id))
        row["optimized_without_cn3_s2p"] = str(save_network(freq_hz, without_s, OUTPUT_DIR / "optimized_without_cn3", sample_id))

    write_existing_result(task, row, component_rows)
    return row, component_rows


def build_tasks():
    dut_df = lhs_base.load_lhs_dataframe()
    if MAX_SAMPLES is not None:
        dut_df = dut_df.head(MAX_SAMPLES).copy()
    target_s, freq_hz = lhs_base.load_targets_and_freq(dut_df)
    base_abcds, _, _ = lhs_base.build_base_abcds(dut_df, freq_hz)

    tasks = []
    for sample_index, row in dut_df.reset_index(drop=True).iterrows():
        sample_id = safe_sample_id(row["sample_id"])
        task = {
            "sample_index": int(sample_index),
            "sample_id": sample_id,
            "file": row["file"],
            "snp_path": row["snp_path"],
            "dut_index": int(row["dut_index"]),
            "source_root": row["source_root"],
            "source_split": row["source_split"],
            "split": row["split"],
            "structure": structure_columns(row),
            "base_abcds": base_abcds[sample_index],
            "target_s": target_s[sample_index],
            "freq_hz": freq_hz,
        }
        tasks.append(task)
    return tasks


def write_boundary_hit_summary(component_df):
    rows = []
    param_names = ["Cn1", "Rn1", "Cn2", "Rn2", "Cn3", "Rn3", "Ln1"]
    for (variant, connection_index), group in component_df.groupby(["variant", "connection_index"], sort=True):
        for name in param_names:
            lower_col = f"{name}_hit_lower_bound"
            upper_col = f"{name}_hit_upper_bound"
            if lower_col not in group.columns or upper_col not in group.columns:
                continue
            lower_hits = group[lower_col].fillna(False).astype(bool)
            upper_hits = group[upper_col].fillna(False).astype(bool)
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
    out_path = OUTPUT_DIR / "boundary_hit_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def write_input_index(tasks):
    rows = []
    for task in tasks:
        row = {
            "sample_id": task["sample_id"],
            "file": task["file"],
            "snp_path": task["snp_path"],
            "dut_index": task["sample_index"],
            "source_dut_index": task["dut_index"],
            "source_root": task["source_root"],
            "source_split": task["source_split"],
            "split": task["split"],
        }
        row.update(task["structure"])
        rows.append(row)
    out_path = OUTPUT_DIR / "lhs_connection_optimization_input_index.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main():
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print("Input LHS data: LHS100 train/val/test + LHS200 train + LHS400 train TSV_RDL", flush=True)
    print(f"Parallel optimization: {PARALLEL_OPTIMIZATION}, workers={MAX_WORKERS}", flush=True)
    print(f"Resume from existing per-sample files: {RESUME_FROM_EXISTING}", flush=True)

    tasks = build_tasks()
    input_index = write_input_index(tasks)
    print(f"Optimization samples: {len(tasks)}", flush=True)
    print(f"Input index CSV: {input_index}", flush=True)

    rows = []
    all_component_rows = []
    use_parallel = PARALLEL_OPTIMIZATION and len(tasks) > 1 and MAX_WORKERS > 1
    if use_parallel:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(optimize_task, task): task for task in tasks}
            for n_done, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    row, component_rows = future.result()
                    rows.append(row)
                    all_component_rows.extend(component_rows)
                    source = "resume" if row.get("resumed_from_existing") else "opt"
                    print(
                        f"[{source}] {n_done}/{len(tasks)} {task['sample_id']} "
                        f"direct={row['direct_mse']:.3e}, with_cn3={row['optimized_with_cn3_mse']:.3e}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[failed] {task['sample_id']}: {exc}", flush=True)
                    rows.append(
                        {
                            "sample_id": task["sample_id"],
                            "file": task["file"],
                            "dut_index": task["sample_index"],
                            "source_dut_index": task["dut_index"],
                            "source_root": task["source_root"],
                            "source_split": task["source_split"],
                            "split": task["split"],
                            "error": str(exc),
                        }
                    )
    else:
        for n_done, task in enumerate(tasks, start=1):
            try:
                row, component_rows = optimize_task(task)
                rows.append(row)
                all_component_rows.extend(component_rows)
                source = "resume" if row.get("resumed_from_existing") else "opt"
                print(
                    f"[{source}] {n_done}/{len(tasks)} {task['sample_id']} "
                    f"direct={row['direct_mse']:.3e}, with_cn3={row['optimized_with_cn3_mse']:.3e}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[failed] {task['sample_id']}: {exc}", flush=True)
                rows.append(
                    {
                        "sample_id": task["sample_id"],
                        "file": task["file"],
                        "dut_index": task["sample_index"],
                        "source_dut_index": task["dut_index"],
                        "source_root": task["source_root"],
                        "source_split": task["source_split"],
                        "split": task["split"],
                        "error": str(exc),
                    }
                )

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
        print(f"Summary CSV: {summary_csv}", flush=True)

    if all_component_rows:
        component_df = pd.DataFrame(all_component_rows)
        connection_csv = OUTPUT_DIR / "connection_network_params.csv"
        component_df.to_csv(connection_csv, index=False, encoding="utf-8-sig")
        boundary_csv = write_boundary_hit_summary(component_df)
        print(f"Connection parameter CSV: {connection_csv}", flush=True)
        print(f"Boundary hit CSV: {boundary_csv}", flush=True)

    valid_df = summary_df[summary_df["error"].isna()] if "error" in summary_df.columns else summary_df
    print("\nOptimization complete", flush=True)
    print(f"Valid samples: {len(valid_df)} / {len(summary_df)}", flush=True)
    if len(valid_df):
        print(f"Mean direct MSE: {valid_df['direct_mse'].mean():.6e}", flush=True)
        print(f"Mean optimized MSE with Cn3: {valid_df['optimized_with_cn3_mse'].mean():.6e}", flush=True)
        print(f"Mean optimized MSE without Cn3: {valid_df['optimized_without_cn3_mse'].mean():.6e}", flush=True)


if __name__ == "__main__":
    main()
