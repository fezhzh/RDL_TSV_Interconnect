# -*- coding: utf-8 -*-
"""Batch HFSS simulations for the base 100/20/20 TSV/RDL dataset.

Run this file directly in VS Code with the PyAnsys conda environment.
No command-line arguments are required.
"""

import os
import random
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ansys.aedt.core import Hfss
    from ansys.aedt.core.desktop import Desktop
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyAEDT is required. Run this script in the PyAnsys conda environment."
    ) from exc


WORK_DIR = Path(__file__).resolve().parent

# =========================
# Direct VS Code settings
# =========================
AEDT_VERSION = "2026.1"
ORIGINAL_PROJECT = Path(r"C:\ffzhzh\LocalFiles\Ansys_Project_Files\TSV_RDL_Connection.aedt")
SOURCE_PROJECT_DIR = ORIGINAL_PROJECT.parent
SOLUTION_SETUP = "Auto1"
SOLUTION_SWEEP = "Sweep"

TARGET_DESIGNS = ["TMRDL", "BSMRDL", "TSV", "TSV_RDL"]

SPLIT_SAMPLE_COUNTS = {
    "train": 100,
    "val": 20,
    "test": 20,
}

NUM_PARALLEL = 5
CORES_PER_SIM = 4
RANDOM_SEED = 20260701

BASE_DATA_OUT_PATH = WORK_DIR / "LHS100"
DATA_OUT_PATH = WORK_DIR / "LHS100"
BASE_SPLIT_NAME = "train"
INCREMENTAL_AUGMENT_MODE = False
CANDIDATE_MULTIPLIER = 50
MIN_CANDIDATE_COUNT = 5000

# All values are in um.
VAR_RANGES_BY_DESIGN = {
    "TMRDL": {
        "pitch": (0.0, 60.0),
        "l_tmrdl": (100.0, 1000.0),
        "w_tmrdl": (10.0, 30.0),
        "h_tmrdl": (2.0, 6.0),
    },
    "BSMRDL": {
        "pitch": (0.0, 60.0),
        "l_bsmrdl": (100.0, 1000.0),
        "w_bsmrdl": (10.0, 30.0),
        "h_bsmrdl": (2.0, 6.0),
    },
    "TSV": {
        "pitch": (0.0, 60.0),
        "r_tsv": (2.0, 15.0),
        "h_tsv": (50.0, 100.0),
    },
    "TSV_RDL": {
        "pitch": (0.0, 60.0),
        "r_tsv": (2.0, 15.0),
        "h_tsv": (50.0, 100.0),
        "l_tmrdl": (100.0, 1000.0),
        "w_tmrdl": (10.0, 30.0),
        "h_tmrdl": (2.0, 6.0),
        "l_bsmrdl": (100.0, 1000.0),
        "w_bsmrdl": (10.0, 30.0),
        "h_bsmrdl": (2.0, 6.0),
    },
}

PITCH_MARGIN_UM = 1.0


def sample_count(split_name):
    return SPLIT_SAMPLE_COUNTS[split_name]


def parse_design_variables(project_path):
    """Read design variable names from the AEDT text file."""
    text = project_path.read_text(errors="ignore")
    design_vars = {}
    design_names = re.findall(r"\$begin '([^']+)'", text)
    for design in sorted(set(design_names)):
        marker = f"Name='{design}'"
        start = text.find(marker)
        if start < 0:
            continue
        next_start = text.find("Name='", start + len(marker))
        block = text[start : next_start if next_start > start else len(text)]
        vars_found = re.findall(r"VariableProp\('([^']+)'\s*,\s*'UD'\s*,\s*''\s*,\s*'([^']*)'\)", block)
        if vars_found:
            design_vars[design] = {name: value for name, value in vars_found}
    return design_vars


def is_valid_variation(design_name, values):
    """Reject pitch values that are too close to the active width/radius."""
    pitch = values.get("pitch")
    if pitch is None:
        return True

    return pitch > required_pitch_min(design_name, values)


def required_pitch_min(design_name, values):
    """Return the minimum pitch required by the current sampled dimensions."""
    if design_name == "TMRDL":
        return values["w_tmrdl"] + PITCH_MARGIN_UM
    if design_name == "BSMRDL":
        return values["w_bsmrdl"] + PITCH_MARGIN_UM
    if design_name == "TSV":
        return 2.0 * values["r_tsv"] + PITCH_MARGIN_UM
    if design_name == "TSV_RDL":
        return max(2.0 * values["r_tsv"], values["w_tmrdl"], values["w_bsmrdl"]) + PITCH_MARGIN_UM
    return float("-inf")


def validate_configuration():
    if not ORIGINAL_PROJECT.exists():
        raise FileNotFoundError(f"AEDT project not found: {ORIGINAL_PROJECT}")

    design_vars = parse_design_variables(ORIGINAL_PROJECT)
    missing_designs = [name for name in TARGET_DESIGNS if name not in design_vars]
    if missing_designs:
        raise ValueError(f"Missing designs in AEDT project: {missing_designs}. Found: {sorted(design_vars)}")

    for design_name in TARGET_DESIGNS:
        missing_vars = [name for name in VAR_RANGES_BY_DESIGN[design_name] if name not in design_vars[design_name]]
        if missing_vars:
            raise ValueError(f"{design_name} is missing variables: {missing_vars}")

    warnings = []
    for design_name, ranges in VAR_RANGES_BY_DESIGN.items():
        pitch_range = ranges.get("pitch")
        if not pitch_range:
            continue
        pitch_min, pitch_max = pitch_range
        if design_name == "TMRDL":
            required_min = ranges["w_tmrdl"][1] + PITCH_MARGIN_UM
        elif design_name == "BSMRDL":
            required_min = ranges["w_bsmrdl"][1] + PITCH_MARGIN_UM
        elif design_name == "TSV":
            required_min = 2.0 * ranges["r_tsv"][1] + PITCH_MARGIN_UM
        elif design_name == "TSV_RDL":
            required_min = max(2.0 * ranges["r_tsv"][1], ranges["w_tmrdl"][1], ranges["w_bsmrdl"][1]) + PITCH_MARGIN_UM
        else:
            continue
        if pitch_min <= required_min:
            warnings.append(
                f"{design_name}: pitch range starts at {pitch_min:g} um. "
                f"Valid samples require pitch > about {required_min:g} um for the current max width/radius."
            )
        if pitch_max <= required_min:
            raise ValueError(f"{design_name}: pitch max {pitch_max:g} um is too small; must exceed {required_min:g} um")

    print("Configuration check passed.")
    if warnings:
        print("Range warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def lhs_unit_values(n_samples, rng):
    values = [(i + rng.random()) / n_samples for i in range(n_samples)]
    rng.shuffle(values)
    return values


def scale_lhs_values(unit_values, lo, hi):
    return [round(lo + value * (hi - lo), 2) for value in unit_values]


def lhs_variations_for_design(design_name, split_name, rng, n_samples=None):
    ranges = VAR_RANGES_BY_DESIGN[design_name]
    n_samples = sample_count(split_name) if n_samples is None else int(n_samples)
    rows = [{} for _ in range(n_samples)]

    for name, (lo, hi) in ranges.items():
        if name == "pitch":
            continue
        values = scale_lhs_values(lhs_unit_values(n_samples, rng), lo, hi)
        for row, value in zip(rows, values):
            row[name] = value

    pitch_lo, pitch_hi = ranges["pitch"]
    pitch_units = lhs_unit_values(n_samples, rng)
    for row, unit_value in zip(rows, pitch_units):
        min_pitch = max(pitch_lo, required_pitch_min(design_name, row) + 0.01)
        if min_pitch >= pitch_hi:
            raise RuntimeError(
                f"{design_name}: no valid pitch range for sampled row {row}. "
                f"Need pitch > {required_pitch_min(design_name, row):.2f} um, max is {pitch_hi:.2f} um."
            )
        row["pitch"] = round(min_pitch + unit_value * (pitch_hi - min_pitch), 2)

    ordered_names = list(ranges)
    rows = [{name: row[name] for name in ordered_names} for row in rows]
    invalid_rows = [row for row in rows if not is_valid_variation(design_name, row)]
    if invalid_rows:
        raise RuntimeError(f"{design_name}: LHS generated invalid pitch rows: {invalid_rows[:3]}")
    return rows


def normalized_design_vector(design_name, row):
    values = []
    for name, (lo, hi) in VAR_RANGES_BY_DESIGN[design_name].items():
        values.append((float(row[name]) - lo) / (hi - lo))
    return values


def existing_train_record_file(design_name):
    return BASE_DATA_OUT_PATH / BASE_SPLIT_NAME / f"{design_name}_variations_record.csv"


def load_existing_base_rows(design_name):
    record_file = existing_train_record_file(design_name)
    if not record_file.exists():
        print(f"No base train record found for {design_name}; incremental sampling will use only new LHS candidates.")
        return []
    rows = read_variation_record(record_file, design_name, validate_pitch=False)
    print(f"Loaded {len(rows)} base train variations for {design_name}: {record_file}")
    return rows


def generate_incremental_variations_for_design(design_name, split_name, rng):
    n_new = sample_count(split_name)
    base_rows = load_existing_base_rows(design_name)
    existing_vectors = [normalized_design_vector(design_name, row) for row in base_rows]
    n_candidates = max(MIN_CANDIDATE_COUNT, n_new * CANDIDATE_MULTIPLIER)
    candidate_rows = lhs_variations_for_design(design_name, split_name, rng, n_samples=n_candidates)
    candidate_vectors = np.asarray([normalized_design_vector(design_name, row) for row in candidate_rows], dtype=float)

    if existing_vectors:
        existing = np.asarray(existing_vectors, dtype=float)
        min_dist = np.min(np.sum((candidate_vectors[:, None, :] - existing[None, :, :]) ** 2, axis=2), axis=1)
    else:
        min_dist = np.full(len(candidate_rows), np.inf)

    selected = []
    used = np.zeros(len(candidate_rows), dtype=bool)
    for _ in range(n_new):
        best_idx = int(np.argmax(np.where(used, -np.inf, min_dist)))
        chosen = candidate_rows[best_idx]
        selected.append(chosen)
        used[best_idx] = True
        new_dist = np.sum((candidate_vectors - candidate_vectors[best_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    start_idx = len(base_rows)
    for local_idx, row in enumerate(selected):
        row["dut_index"] = start_idx + local_idx
    return selected


def read_variation_record(record_file, design_name, validate_pitch=True):
    expected_columns = list(VAR_RANGES_BY_DESIGN[design_name])
    df = pd.read_csv(record_file, encoding="utf-8-sig")
    missing_columns = [name for name in expected_columns if name not in df.columns]
    if missing_columns:
        raise ValueError(f"{record_file} is missing columns: {missing_columns}")

    keep_columns = expected_columns + (["dut_index"] if "dut_index" in df.columns else [])
    df = df[keep_columns].copy()
    for name in expected_columns:
        df[name] = pd.to_numeric(df[name], errors="raise")
    if "dut_index" in df.columns:
        df["dut_index"] = pd.to_numeric(df["dut_index"], errors="raise").astype(int)

    rows = df.to_dict("records")
    if validate_pitch:
        invalid_rows = [idx for idx, row in enumerate(rows) if not is_valid_variation(design_name, row)]
        if invalid_rows:
            raise ValueError(
                f"{record_file} contains invalid pitch rows under the current constraints. "
                f"First invalid row index: {invalid_rows[0]}"
            )
    return rows


def generate_variations():
    DATA_OUT_PATH.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    variations_by_split = {}

    for split_name in SPLIT_SAMPLE_COUNTS:
        split_out_dir = DATA_OUT_PATH / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)
        variations_by_split[split_name] = {}

        for design_name in TARGET_DESIGNS:
            record_file = split_out_dir / f"{design_name}_variations_record.csv"
            if record_file.exists():
                rows = read_variation_record(record_file, design_name)
                print(f"Loaded {len(rows)} existing {split_name} variations for {design_name}: {record_file}")
                if len(rows) != sample_count(split_name):
                    print(
                        f"  Note: existing {split_name} record has {len(rows)} rows; "
                        f"configured count is {sample_count(split_name)}. The existing record is kept unchanged."
                    )
            else:
                if INCREMENTAL_AUGMENT_MODE:
                    rows = generate_incremental_variations_for_design(design_name, split_name, rng)
                else:
                    rows = lhs_variations_for_design(design_name, split_name, rng)
                write_dict_rows(record_file, rows)
                print(f"Generated {len(rows)} {split_name} LHS variations for {design_name}: {record_file}")
            variations_by_split[split_name][design_name] = rows

    return variations_by_split


def write_dict_rows(path, rows):
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8-sig")


def remove_path(path):
    path = Path(path)
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception:
        pass


def validate_result_file(path):
    if not path.exists():
        return False
    return path.stat().st_size > 0


def run_single_simulation(args):
    split_name, idx, var_dict, design_name = args
    var_dict = dict(var_dict)
    var_dict.pop("dut_index", None)

    design_out_dir = DATA_OUT_PATH / split_name / design_name
    design_out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(design_out_dir.glob(f"dut{idx}.s*p"))
    if existing_files:
        print(f"[{split_name} / {design_name} - Task {idx}] result exists, skipped.")
        return {"split": split_name, "design": design_name, "idx": idx, "status": "skipped", "file": str(existing_files[0])}

    print(f"[{split_name} / {design_name} - Task {idx}] simulation started.")

    temp_prefix = f"temp_{split_name}_{design_name}_{idx}_{int(time.time())}_{os.getpid()}"
    temp_project_path = SOURCE_PROJECT_DIR / f"{temp_prefix}.aedt"
    temp_results_folder = SOURCE_PROJECT_DIR / f"{temp_prefix}.aedtresults"
    temp_pyaedt_folder = SOURCE_PROJECT_DIR / f"{temp_prefix}.pyaedt"
    temp_snp_export = SOURCE_PROJECT_DIR / f"{temp_prefix}_SNP"

    try:
        shutil.copy2(ORIGINAL_PROJECT, temp_project_path)
    except FileNotFoundError:
        msg = f"Original project not found: {ORIGINAL_PROJECT}"
        print(f"[{split_name} / {design_name} - Task {idx}] {msg}")
        return {"split": split_name, "design": design_name, "idx": idx, "status": "failed", "error": msg}

    desktop = None
    app = None
    try:
        temp_snp_export.mkdir(parents=True, exist_ok=True)
        desktop = Desktop(version=AEDT_VERSION, non_graphical=True, new_desktop=True, close_on_exit=True)
        app = Hfss(project=str(temp_project_path), design=design_name)
        app.modeler.model_units = "um"

        for var, value in var_dict.items():
            app[var] = f"{value}um"

        valid = app.validate_simple()
        is_valid = valid[0] if isinstance(valid, (tuple, list)) else bool(valid)
        if not is_valid:
            print(f"[{split_name} / {design_name} - Task {idx}] validation failed: {valid}")
            return {"split": split_name, "design": design_name, "idx": idx, "status": "validation_failed", "details": str(valid)}

        solved = app.analyze(cores=CORES_PER_SIM)
        if not solved:
            msg = f"HFSS analyze failed for setup {SOLUTION_SETUP}."
            print(f"[{split_name} / {design_name} - Task {idx}] {msg}")
            return {"split": split_name, "design": design_name, "idx": idx, "status": "solve_failed", "error": msg}

        output_file = temp_snp_export / f"{temp_prefix}.s2p"
        exported = app.export_touchstone(
            setup=SOLUTION_SETUP,
            sweep=SOLUTION_SWEEP,
            output_file=str(output_file),
            impedance=50,
            gamma_impedance_comments=True,
        )

        if not exported:
            exported = app.export_touchstone(
                setup=SOLUTION_SETUP,
                output_file=str(output_file),
                impedance=50,
                gamma_impedance_comments=True,
            )

        if not exported or not validate_result_file(output_file):
            msg = "No exported Touchstone file found."
            print(f"[{split_name} / {design_name} - Task {idx}] {msg}")
            return {"split": split_name, "design": design_name, "idx": idx, "status": "failed", "error": msg}

        dst_file = design_out_dir / f"dut{idx}{output_file.suffix}"
        if dst_file.exists():
            dst_file.unlink()
        shutil.move(str(output_file), dst_file)
        ok = validate_result_file(dst_file)
        print(f"[{split_name} / {design_name} - Task {idx}] simulation completed: {dst_file}")
        return {
            "split": split_name,
            "design": design_name,
            "idx": idx,
            "status": "completed" if ok else "empty_result",
            "file": str(dst_file),
        }

    except Exception as exc:
        error = str(exc)
        if "GetDesignType" in error or "requested resource is not permitted" in error:
            error = (
                f"{error}; AEDT did not provide an editable HFSS design object. "
                "Check the Ansys license server and whether the copied project opened read-only."
            )
        print(f"[{split_name} / {design_name} - Task {idx}] exception: {error}")
        return {"split": split_name, "design": design_name, "idx": idx, "status": "failed", "error": error}

    finally:
        if desktop:
            try:
                desktop.release_desktop(close_projects=True, close_on_exit=True)
            except Exception:
                pass
        time.sleep(1)
        for path in [
            temp_project_path,
            Path(str(temp_project_path) + ".lock"),
            temp_results_folder,
            temp_pyaedt_folder,
            temp_snp_export,
        ]:
            remove_path(path)


def build_tasks(variations_by_split):
    tasks = []
    for split_name in SPLIT_SAMPLE_COUNTS:
        for design_name in TARGET_DESIGNS:
            for idx, var_dict in enumerate(variations_by_split[split_name][design_name]):
                dut_index = int(var_dict.get("dut_index", idx))
                tasks.append((split_name, dut_index, var_dict, design_name))
    return tasks


def main():
    print("=====================================")
    print("HFSS batch simulation")
    print(f"AEDT project: {ORIGINAL_PROJECT}")
    print(f"AEDT version: {AEDT_VERSION}")
    print(f"Target designs: {', '.join(TARGET_DESIGNS)}")
    print(f"Sample splits: {SPLIT_SAMPLE_COUNTS}")
    print("=====================================")

    validate_configuration()
    variations_by_split = generate_variations()
    tasks = build_tasks(variations_by_split)

    print(f"Total tasks: {len(tasks)}")
    print(f"Parallel processes: {NUM_PARALLEL}")
    print(f"Cores per simulation: {CORES_PER_SIM}")
    print(f"Output directory: {DATA_OUT_PATH}")
    print("=====================================\n")

    if NUM_PARALLEL <= 1:
        results = [run_single_simulation(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=NUM_PARALLEL) as executor:
            results = list(executor.map(run_single_simulation, tasks))

    summary_file = DATA_OUT_PATH / "simulation_summary.csv"
    write_dict_rows(summary_file, results)
    print(f"\nAll tasks finished. Summary: {summary_file}")


if __name__ == "__main__":
    main()
