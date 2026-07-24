# -*- coding: utf-8 -*-
"""ADS TSV single-device S-parameter simulation helper for v10."""

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
    "substrate_er": 11.9,
    "substrate_loss_tangent": 0.005,
    "metal_conductivity_s_per_m": 5.8e7,
    "c1_scale": 1.0,
    "pitch_scale": 1.0,
    "h_tsv_scale": 1.0,
    "d_scale": 1.0,
    "freq_start_ghz": 0.1,
    "freq_stop_ghz": 20.0,
    "freq_step_ghz": 0.1,
}

REQUIRED_COLUMNS = ["r_tsv", "h_tsv", "pitch"]


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _to_dict(values: Mapping[str, object] | pd.Series) -> dict[str, object]:
    return values.to_dict() if isinstance(values, pd.Series) else dict(values)


def _settings(ads_settings: Mapping[str, object] | None) -> dict[str, float]:
    settings = dict(DEFAULT_ADS_SETTINGS)
    if ads_settings:
        settings.update({key: value for key, value in dict(ads_settings).items() if key in settings})
        incoming = dict(ads_settings)
        if "er_si" in incoming:
            settings["substrate_er"] = incoming["er_si"]
        if "cond" in incoming:
            settings["metal_conductivity_s_per_m"] = incoming["cond"]
        if "tand" in incoming:
            settings["substrate_loss_tangent"] = incoming["tand"]
    return {key: float(value) for key, value in settings.items()}


def ads_variables_for_device(
    device_name: str,
    structure: Mapping[str, object] | pd.Series,
    ads_settings: Mapping[str, object] | None = None,
) -> dict[str, float]:
    if device_name != "TSV":
        raise ValueError(f"TSV ADS helper only supports TSV, got {device_name}")
    row = _to_dict(structure)
    missing = [name for name in REQUIRED_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"TSV ADS simulation is missing columns: {missing}")
    settings = _settings(ads_settings)
    return {
        "h_tsv": float(row["h_tsv"]) * settings["h_tsv_scale"] * 1e-6,
        "d_tsv": 2.0 * float(row["r_tsv"]) * settings["d_scale"] * 1e-6,
        "er_si": settings["substrate_er"],
        "cond": settings["metal_conductivity_s_per_m"],
        "tand": settings["substrate_loss_tangent"],
        "pitch": float(row["pitch"]) * settings["pitch_scale"] * 1e-6,
        "c1_scale": settings["c1_scale"],
    }


def _format_ads_number(value: float) -> str:
    return f"{float(value):.12g}"


def _replace_assignment(line: str, variables: Mapping[str, float]) -> str | None:
    stripped = line.strip()
    if stripped.startswith("c1=") and "c1_scale" in variables:
        return f"c1={_format_ads_number(variables['c1_scale'])}*(pi*er_si*er_0/acosh(pitch/d_tsv)*h_tsv)\n"
    for name, value in variables.items():
        if name == "c1_scale":
            continue
        if stripped.startswith(f"{name}="):
            suffix = "um" if name in {"h_tsv", "d_tsv", "pitch"} else ""
            if suffix:
                return f"{name}={_format_ads_number(value / 1e-6)}{suffix}\n"
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
        raise FileNotFoundError(f"Missing TSV ADS template: {TEMPLATE_NETLIST}")
    settings = _settings(ads_settings)
    variables = ads_variables_for_device(device_name, structure, ads_settings)
    safe_sample = _safe_name(sample_id)
    output_base = Path(output_base) if output_base is not None else SNP_DIR / f"{safe_sample}_TSV"
    output_base = output_base.resolve()
    out_s2p = output_base.with_suffix(".s2p")

    RUN_NETLIST_DIR.mkdir(parents=True, exist_ok=True)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    run_netlist = (RUN_NETLIST_DIR / f"{safe_sample}_TSV.net").resolve()
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
        raise RuntimeError(f"ADS TSV simulation failed for {netlist_path.name}; see {log_path}")


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
        raise FileNotFoundError(f"ADS TSV simulation did not write {out_s2p}")
    return out_s2p


def demo_structure() -> dict[str, float]:
    return {"pitch": 45.0, "r_tsv": 5.0, "h_tsv": 80.0}


def main() -> None:
    out = simulate_single_device("TSV", "tsv_ads_smoke", demo_structure(), DEFAULT_ADS_SETTINGS)
    print(f"TSV: {out}", flush=True)


if __name__ == "__main__":
    main()
