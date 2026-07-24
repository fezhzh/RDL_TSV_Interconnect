# -*- coding: utf-8 -*-
"""Batch HFSS simulations for TSV_RDL Connection2 LHS datasets.

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
ORIGINAL_PROJECT = Path(r"C:\ffzhzh\LocalFiles\Ansys_Project_Files\TSV_RDL_Connection2.aedt")
SOURCE_PROJECT_DIR = ORIGINAL_PROJECT.parent
SOLUTION_SETUP = "Auto1"
SOLUTION_SWEEP = "Sweep"

NUM_PARALLEL = 3
CORES_PER_SIM = 4
RANDOM_SEED = 20260707
PITCH_MARGIN_UM = 1.0

# The AEDT project names RDL thickness variables as h_*.
HFSS_VARIABLE_BY_RECORD_COLUMN = {
    "t_tmrdl": "h_tmrdl",
    "t_bsmrdl": "h_bsmrdl",
}

# User-facing variable names and ranges. All values are in um.
TSV_RDL_RANGES = {
    "h_tsv": (50.0, 100.0),
    "r_tsv": (5.0, 15.0),
    "pitch": (40.0, 60.0),
    "l_tmrdl": (100.0, 700.0),
    "w_tmrdl": (10.0, 30.0),
    "t_tmrdl": (2.0, 5.0),
    "l_bsmrdl": (100.0, 700.0),
    "w_bsmrdl": (10.0, 30.0),
    "t_bsmrdl": (2.0, 5.0),
}

RDL_RANGES = {
    "pitch": (40.0, 60.0),
    "l_tmrdl": (100.0, 700.0),
    "w_tmrdl": (10.0, 30.0),
    "t_tmrdl": (2.0, 5.0),
}

TSV_RANGES = {
    "h_tsv": (50.0, 100.0),
    "r_tsv": (5.0, 15.0),
    "pitch": (40.0, 60.0),
}

DATASET_CONFIGS = {
    "rdl_tsv_lhs400": {
        "output_path": WORK_DIR / "LHS400_Connection2",
        "designs": {
            "RDL": {
                "splits": {"train": 400},
                "dut_index_start": {"train": 0},
                "ranges": RDL_RANGES,
            },
            "TSV": {
                "splits": {"train": 400},
                "dut_index_start": {"train": 0},
                "ranges": TSV_RANGES,
            },
            "TSV_RDL": {
                "splits": {"train": 400},
                "dut_index_start": {"train": 0},
                "ranges": TSV_RDL_RANGES,
            },
        },
    },
    "tsv_rdl_150_50": {
        "output_path": WORK_DIR / "LHS150_50_Connection2",
        "designs": {
            "TSV_RDL": {
                "splits": {"train": 150, "test": 50},
                "dut_index_start": {"train": 0, "test": 150},
                "ranges": TSV_RDL_RANGES,
            },
        },
    },
}


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


def hfss_variable_name(record_column):
    return HFSS_VARIABLE_BY_RECORD_COLUMN.get(record_column, record_column)


def iter_design_configs():
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        output_path = dataset_config["output_path"]
        for design_name, design_config in dataset_config["designs"].items():
            yield dataset_name, output_path, design_name, design_config


def required_pitch_min(design_name, values):
    if design_name == "RDL":
        return values["w_tmrdl"] + PITCH_MARGIN_UM
    if design_name == "TSV":
        return 2.0 * values["r_tsv"] + PITCH_MARGIN_UM
    if design_name == "TSV_RDL":
        return max(2.0 * values["r_tsv"], values["w_tmrdl"], values["w_bsmrdl"]) + PITCH_MARGIN_UM
    return float("-inf")


def is_valid_variation(design_name, values):
    pitch = values.get("pitch")
    if pitch is None:
        return True
    return pitch > required_pitch_min(design_name, values)


def validate_configuration():
    if not ORIGINAL_PROJECT.exists():
        raise FileNotFoundError(f"AEDT project not found: {ORIGINAL_PROJECT}")

    design_vars = parse_design_variables(ORIGINAL_PROJECT)
    missing_designs = [design_name for _, _, design_name, _ in iter_design_configs() if design_name not in design_vars]
    if missing_designs:
        raise ValueError(f"Missing designs in AEDT project: {missing_designs}. Found: {sorted(design_vars)}")

    for _, _, design_name, design_config in iter_design_configs():
        ranges = design_config["ranges"]
        required_hfss_vars = [hfss_variable_name(name) for name in ranges]
        missing_vars = [name for name in required_hfss_vars if name not in design_vars[design_name]]
        if missing_vars:
            raise ValueError(f"{design_name} is missing HFSS variables: {missing_vars}")

        pitch_range = ranges.get("pitch")
        if pitch_range:
            pitch_max = pitch_range[1]
            max_values = {name: hi for name, (_, hi) in ranges.items()}
            required_at_max = required_pitch_min(design_name, max_values)
            if pitch_max <= required_at_max:
                raise ValueError(
                    f"{design_name}: pitch max {pitch_max:g} um is too small; "
                    f"it must exceed {required_at_max:g} um."
                )

    print("Configuration check passed.")
    print("HFSS thickness mapping: t_tmrdl -> h_tmrdl, t_bsmrdl -> h_bsmrdl")
    for dataset_name, output_path, design_name, design_config in iter_design_configs():
        print(
            f"{dataset_name} / {design_name}: splits={design_config['splits']}, "
            f"columns={list(design_config['ranges'])}, output={output_path}"
        )


def lhs_unit_values(n_samples, rng):
    values = [(idx + rng.random()) / n_samples for idx in range(n_samples)]
    rng.shuffle(values)
    return values


def scale_lhs_values(unit_values, lo, hi):
    return [round(lo + value * (hi - lo), 2) for value in unit_values]


def lhs_variations(design_name, design_config, split_name, rng):
    ranges = design_config["ranges"]
    n_samples = int(design_config["splits"][split_name])
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

    start_idx = int(design_config["dut_index_start"][split_name])
    for local_idx, row in enumerate(rows):
        row["dut_index"] = start_idx + local_idx
    return rows


def read_variation_record(record_file, design_name, design_config, validate_pitch=True):
    expected_columns = list(design_config["ranges"])
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
                f"{record_file} contains invalid pitch rows. "
                f"First invalid row index: {invalid_rows[0]}"
            )
    return rows


def write_dict_rows(path, rows):
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8-sig")


def generate_variations():
    rng = random.Random(RANDOM_SEED)
    variations = {}

    for dataset_name, output_path, design_name, design_config in iter_design_configs():
        output_path.mkdir(parents=True, exist_ok=True)
        variations[(dataset_name, design_name)] = {}

        for split_name, split_count in design_config["splits"].items():
            split_out_dir = output_path / split_name
            split_out_dir.mkdir(parents=True, exist_ok=True)
            record_file = split_out_dir / f"{design_name}_variations_record.csv"

            if record_file.exists():
                rows = read_variation_record(record_file, design_name, design_config)
                print(f"Loaded {len(rows)} existing {dataset_name} / {split_name} variations for {design_name}: {record_file}")
                if len(rows) != split_count:
                    print(
                        f"  Note: existing {split_name} record has {len(rows)} rows; "
                        f"configured count is {split_count}. The existing record is kept unchanged."
                    )
            else:
                rows = lhs_variations(design_name, design_config, split_name, rng)
                write_dict_rows(record_file, rows)
                print(f"Generated {len(rows)} {dataset_name} / {split_name} LHS variations for {design_name}: {record_file}")

            variations[(dataset_name, design_name)][split_name] = rows

    return variations


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
    return path.exists() and path.stat().st_size > 0


def run_single_simulation(args):
    output_path, split_name, idx, record_row, design_name = args
    record_row = dict(record_row)
    record_row.pop("dut_index", None)

    design_out_dir = output_path / split_name / design_name
    design_out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(design_out_dir.glob(f"dut{idx}.s*p"))
    if existing_files:
        print(f"[{split_name} / {design_name} - Task {idx}] result exists, skipped.")
        return {"output_path": str(output_path), "split": split_name, "design": design_name, "idx": idx, "status": "skipped", "file": str(existing_files[0])}

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
        return {"output_path": str(output_path), "split": split_name, "design": design_name, "idx": idx, "status": "failed", "error": msg}

    desktop = None
    try:
        temp_snp_export.mkdir(parents=True, exist_ok=True)
        desktop = Desktop(version=AEDT_VERSION, non_graphical=True, new_desktop=True, close_on_exit=True)
        app = Hfss(project=str(temp_project_path), design=design_name)
        app.modeler.model_units = "um"

        for column, value in record_row.items():
            app[hfss_variable_name(column)] = f"{value}um"

        valid = app.validate_simple()
        is_valid = valid[0] if isinstance(valid, (tuple, list)) else bool(valid)
        if not is_valid:
            print(f"[{split_name} / {design_name} - Task {idx}] validation failed: {valid}")
            return {
                "output_path": str(output_path),
                "split": split_name,
                "design": design_name,
                "idx": idx,
                "status": "validation_failed",
                "details": str(valid),
            }

        solved = app.analyze(cores=CORES_PER_SIM)
        if not solved:
            msg = f"HFSS analyze failed for setup {SOLUTION_SETUP}."
            print(f"[{split_name} / {design_name} - Task {idx}] {msg}")
            return {"output_path": str(output_path), "split": split_name, "design": design_name, "idx": idx, "status": "solve_failed", "error": msg}

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
            return {"output_path": str(output_path), "split": split_name, "design": design_name, "idx": idx, "status": "failed", "error": msg}

        dst_file = design_out_dir / f"dut{idx}{output_file.suffix}"
        if dst_file.exists():
            dst_file.unlink()
        shutil.move(str(output_file), dst_file)
        ok = validate_result_file(dst_file)
        print(f"[{split_name} / {design_name} - Task {idx}] simulation completed: {dst_file}")
        return {
            "output_path": str(output_path),
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
        return {"output_path": str(output_path), "split": split_name, "design": design_name, "idx": idx, "status": "failed", "error": error}

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


def build_tasks(variations):
    tasks = []
    for dataset_name, output_path, design_name, design_config in iter_design_configs():
        for split_name in design_config["splits"]:
            for record_row in variations[(dataset_name, design_name)][split_name]:
                dut_index = int(record_row["dut_index"])
                tasks.append((output_path, split_name, dut_index, record_row, design_name))
    return tasks


def write_summary_files(results):
    results_by_output = {}
    for result in results:
        results_by_output.setdefault(result["output_path"], []).append(result)

    for output_path, output_results in results_by_output.items():
        summary_file = Path(output_path) / "simulation_summary.csv"
        write_dict_rows(summary_file, output_results)
        print(f"Summary written: {summary_file}")


def main():
    print("=====================================")
    print("HFSS batch simulation")
    print(f"AEDT project: {ORIGINAL_PROJECT}")
    print(f"AEDT version: {AEDT_VERSION}")
    print("=====================================")

    validate_configuration()
    variations = generate_variations()
    tasks = build_tasks(variations)

    print(f"Total tasks: {len(tasks)}")
    print(f"Parallel processes: {NUM_PARALLEL}")
    print(f"Cores per simulation: {CORES_PER_SIM}")
    print("=====================================\n")

    if NUM_PARALLEL <= 1:
        results = [run_single_simulation(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=NUM_PARALLEL) as executor:
            results = list(executor.map(run_single_simulation, tasks))

    write_summary_files(results)
    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()
