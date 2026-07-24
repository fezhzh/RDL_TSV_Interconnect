# -*- coding: utf-8 -*-
"""ADS RDL single-device S-parameter simulation helper for v10.

Run directly in VS Code for a one-sample RDL smoke test, or import
``simulate_single_device`` from the v10 training script.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_NETLIST = SCRIPT_DIR / "sim.net"
RUN_NETLIST_DIR = SCRIPT_DIR / "generated_netlists"
SNP_DIR = SCRIPT_DIR / "Snp"
LOG_DIR = SCRIPT_DIR / "logs"

HPEESOF_DIR = Path(os.environ.get("HPEESOF_DIR", r"C:\Keysight\ADS2026_Update1.2"))
HPEESOF_SIM = HPEESOF_DIR / "bin" / "hpeesofsim.exe"
SIM_TIMEOUT_SEC = 900

DEFAULT_ADS_SETTINGS = {
    "substrate_thickness_um": 100.0,
    "substrate_er": 11.9,
    "substrate_loss_tangent": 0.005,
    "metal_thickness_um": 3.0,
    "metal_conductivity_s_per_m": 5.8e7,
    "l_scale": 1.0,
    "w_scale": 1.0,
    "pitch_scale": 1.0,
    "h_tsv_scale": 1.0,
    "h_rdl_scale": 1.0,
    "freq_start_ghz": 0.1,
    "freq_stop_ghz": 20.0,
    "freq_step_ghz": 0.1,
}

DEVICE_COLUMNS = {
    "TMRDL": ["l_tmrdl", "w_tmrdl", "h_tmrdl", "pitch", "h_tsv"],
    "BSMRDL": ["l_bsmrdl", "w_bsmrdl", "h_bsmrdl", "pitch", "h_tsv"],
}


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _to_dict(values: Mapping[str, object] | pd.Series) -> dict[str, object]:
    return values.to_dict() if isinstance(values, pd.Series) else dict(values)


def _required(device_name: str, structure: Mapping[str, object]) -> None:
    if device_name not in DEVICE_COLUMNS:
        raise ValueError(f"RDL ADS helper only supports TMRDL/BSMRDL, got {device_name}")
    missing = [name for name in DEVICE_COLUMNS[device_name] if name not in structure]
    if missing:
        raise ValueError(f"{device_name} ADS simulation is missing columns: {missing}")


def _settings(ads_settings: Mapping[str, object] | None) -> dict[str, float]:
    settings = dict(DEFAULT_ADS_SETTINGS)
    if ads_settings:
        settings.update(dict(ads_settings))
    if "er_si" in settings:
        settings["substrate_er"] = settings["er_si"]
    if "cond" in settings:
        settings["metal_conductivity_s_per_m"] = settings["cond"]
    if "tand" in settings:
        settings["substrate_loss_tangent"] = settings["tand"]
    return {key: float(value) for key, value in settings.items()}


def ads_variables_for_device(
    device_name: str,
    structure: Mapping[str, object] | pd.Series,
    ads_settings: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Map v10 RDL geometry in micrometers to ADS netlist variables in SI."""

    row = _to_dict(structure)
    _required(device_name, row)
    settings = _settings(ads_settings)
    if device_name == "TMRDL":
        length_um = float(row["l_tmrdl"])
        width_um = float(row["w_tmrdl"])
        metal_thickness_um = float(row["h_tmrdl"])
    else:
        length_um = float(row["l_bsmrdl"])
        width_um = float(row["w_bsmrdl"])
        metal_thickness_um = float(row["h_bsmrdl"])

    return {
        "l_rdl": length_um * settings["l_scale"] * 1e-6,
        "w_rdl": width_um * settings["w_scale"] * 1e-6,
        "pitch": float(row["pitch"]) * settings["pitch_scale"] * 1e-6,
        "h_tsv": float(row.get("h_tsv", settings["substrate_thickness_um"])) * settings["h_tsv_scale"] * 1e-6,
        "er_si": settings["substrate_er"],
        "h_rdl": metal_thickness_um * settings["h_rdl_scale"] * 1e-6,
        "cond": settings["metal_conductivity_s_per_m"],
        "tand": settings["substrate_loss_tangent"],
    }


def _format_ads_number(value: float) -> str:
    return f"{float(value):.12g}"


def _replace_assignment(line: str, variables: Mapping[str, float]) -> str | None:
    stripped = line.strip()
    for name, value in variables.items():
        if stripped.startswith(f"{name}="):
            return f"{name}={_format_ads_number(value)}\n"
    return None


def _replace_sweep_plan(line: str, settings: Mapping[str, float]) -> str | None:
    if not line.strip().startswith("SweepPlan: SP1_stim"):
        return None
    return (
        "SweepPlan: SP1_stim "
        f"Start={_format_ads_number(settings['freq_start_ghz'])} GHz "
        f"Stop={_format_ads_number(settings['freq_stop_ghz'])} GHz "
        f"Step={_format_ads_number(settings['freq_step_ghz'])} GHz \n"
    )


def write_netlist(
    device_name: str,
    sample_id: str,
    structure: Mapping[str, object] | pd.Series,
    ads_settings: Mapping[str, object] | None = None,
    output_base: Path | None = None,
) -> tuple[Path, Path, dict[str, float]]:
    if not TEMPLATE_NETLIST.exists():
        raise FileNotFoundError(f"Missing RDL ADS template: {TEMPLATE_NETLIST}")
    row = _to_dict(structure)
    settings = _settings(ads_settings)
    variables = ads_variables_for_device(device_name, row, ads_settings)
    safe_sample = _safe_name(sample_id)
    safe_device = _safe_name(device_name)
    output_base = Path(output_base) if output_base is not None else SNP_DIR / f"{safe_sample}_{safe_device}"
    output_base = output_base.resolve()
    out_s2p = output_base.with_suffix(".s2p")

    RUN_NETLIST_DIR.mkdir(parents=True, exist_ok=True)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    run_netlist = (RUN_NETLIST_DIR / f"{safe_sample}_{safe_device}.net").resolve()
    rendered = []
    for line in TEMPLATE_NETLIST.read_text(encoding="utf-8").splitlines(keepends=True):
        replacement = _replace_assignment(line, variables)
        if replacement is not None:
            rendered.append(replacement)
        elif (replacement := _replace_sweep_plan(line, settings)) is not None:
            rendered.append(replacement)
        elif line.lstrip().startswith("Argument[1]"):
            ads_output_base = str(output_base).replace("\\", "/")
            rendered.append(f'Argument[1] = "{ads_output_base}" \\ \n')
        else:
            rendered.append(line)
    run_netlist.write_text("".join(rendered), encoding="utf-8")
    return run_netlist, out_s2p, variables


def _ads_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["HPEESOF_DIR"] = str(HPEESOF_DIR)
    env["COMPL_DIR"] = str(HPEESOF_DIR)
    env["SIMARCH"] = "win32_64"
    env["PATH"] = ";".join(
        [
            str(HPEESOF_DIR / "bin"),
            str(HPEESOF_DIR / "adsptolemy" / "lib.win32_64"),
            str(HPEESOF_DIR / "tools" / "python"),
            env.get("PATH", ""),
        ]
    )
    return env


def _cleanup_ads_runtime_files() -> None:
    for name in ["cell_1.ds", "cell_2.ds", "spectra.raw"]:
        path = SCRIPT_DIR / name
        if path.exists() and path.is_file():
            path.unlink()


def run_ads_netlist(netlist_path: Path, log_path: Path | None = None) -> None:
    if not HPEESOF_SIM.exists():
        raise FileNotFoundError(f"ADS simulator executable was not found: {HPEESOF_SIM}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path or LOG_DIR / f"{netlist_path.stem}.log"
    _cleanup_ads_runtime_files()
    result = subprocess.run(
        [str(HPEESOF_SIM), str(netlist_path)],
        cwd=str(SCRIPT_DIR),
        env=_ads_environment(),
        capture_output=True,
        text=True,
        timeout=SIM_TIMEOUT_SEC,
        check=False,
    )
    log_path.write_text(result.stdout + "\n\nSTDERR:\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ADS RDL simulation failed for {netlist_path.name}; see {log_path}")


def simulate_single_device(
    device_name: str,
    sample_id: str,
    structure: Mapping[str, object] | pd.Series,
    ads_settings: Mapping[str, object] | None = None,
    output_base: Path | None = None,
    reuse_existing: bool = True,
) -> Path:
    netlist, out_s2p, variables = write_netlist(device_name, sample_id, structure, ads_settings, output_base)
    settings = _settings(ads_settings)
    out_s2p.parent.mkdir(parents=True, exist_ok=True)
    out_s2p.with_suffix(".json").write_text(
        json.dumps(
            {
                "device_name": device_name,
                "sample_id": sample_id,
                "ads_settings": settings,
                "ads_variables_si": variables,
                "netlist": str(netlist),
                "touchstone": str(out_s2p),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if reuse_existing and out_s2p.exists() and out_s2p.stat().st_size > 0:
        return out_s2p
    run_ads_netlist(netlist)
    if not out_s2p.exists() or out_s2p.stat().st_size == 0:
        raise FileNotFoundError(f"ADS RDL simulation did not write {out_s2p}")
    return out_s2p


def demo_structure() -> dict[str, float]:
    return {
        "pitch": 45.0,
        "r_tsv": 5.0,
        "h_tsv": 80.0,
        "l_tmrdl": 100.0,
        "w_tmrdl": 20.0,
        "h_tmrdl": 2.0,
        "l_bsmrdl": 110.0,
        "w_bsmrdl": 18.0,
        "h_bsmrdl": 2.0,
    }


def main() -> None:
    for device_name in ["TMRDL", "BSMRDL"]:
        out = simulate_single_device(device_name, "rdl_ads_smoke", demo_structure(), DEFAULT_ADS_SETTINGS)
        print(f"{device_name}: {out}", flush=True)


if __name__ == "__main__":
    main()
