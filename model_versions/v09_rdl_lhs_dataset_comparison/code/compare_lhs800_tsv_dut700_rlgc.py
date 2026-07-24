# -*- coding: utf-8 -*-
"""Compare RLGC extracted from two LHS800 TSV Touchstone files.

Run this file directly in VS Code. No command-line arguments are required.

The two inputs are:
- HFSS_sim/LHS800/train/TSV/dut700.s2p
- HFSS_sim/LHS800/train/TSV/dut_700.s2p

Outputs are written to:
model_versions/v09_rdl_lhs_dataset_comparison/results/rlgc_compare_dut700/
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_A = PROJECT_ROOT / "HFSS_sim" / "LHS800" / "train" / "TSV" / "dut700.s2p"
INPUT_B = PROJECT_ROOT / "HFSS_sim" / "LHS800" / "train" / "TSV" / "dut_700.s2p"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v09_rdl_lhs_dataset_comparison"
    / "results"
    / "rlgc_compare_dut700"
)

BASE_EXTRACTOR_DIR = (
    PROJECT_ROOT / "model_versions" / "v00_parameter_extraction_and_dataset_building" / "code"
)
BASE_EXTRACTOR_PATH = next(BASE_EXTRACTOR_DIR.glob("*3.py"))
EPS = 1e-30


def load_base_extractor():
    spec = importlib.util.spec_from_file_location("base_extractor", BASE_EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base extractor: {BASE_EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_EXTRACTOR = load_base_extractor()


def parse_touchstone_variables(path: Path) -> dict[str, float]:
    variables: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith("#"):
                break
            if not stripped.startswith("!") or "=" not in stripped:
                continue
            name, value = stripped[1:].split("=", 1)
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
            if match:
                variables[name.strip().lower()] = float(match.group(0))
    return variables


def tsv_length_m(*paths: Path) -> float:
    for path in paths:
        variables = parse_touchstone_variables(path)
        for key in ("h_tsv", "htsv"):
            if key in variables:
                return variables[key] * 1e-6
    raise ValueError("Cannot find h_tsv/htsv in the input Touchstone headers.")


def load_network(path: Path) -> rf.Network:
    if not path.exists():
        raise FileNotFoundError(path)
    return rf.Network(str(path))


def extract_rlgc(path: Path, length_m: float) -> pd.DataFrame:
    network = load_network(path)
    s = network.s
    freq_hz = network.f
    s11, s12, s21, s22 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    a, b, c, d = BASE_EXTRACTOR.S_ABCD(s11, s12, s21, s22)
    r, l, g, c_rlgc, z0, gamma = BASE_EXTRACTOR.ABCD_RLGC(a, b, c, d, freq_hz, length_m)
    return pd.DataFrame(
        {
            "freq_hz": freq_hz,
            "freq_ghz": freq_hz / 1e9,
            "R_ohm_per_m": np.real(r),
            "L_h_per_m": np.real(l),
            "G_s_per_m": np.real(g),
            "C_f_per_m": np.real(c_rlgc),
            "Z0_real": np.real(z0),
            "Z0_imag": np.imag(z0),
            "gamma_real": np.real(gamma),
            "gamma_imag": np.imag(gamma),
        }
    )


def align_by_frequency(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(right, on=["freq_hz", "freq_ghz"], suffixes=("_dut700", "_dut_700"))
    if merged.empty:
        raise ValueError("The two files have no exactly matching frequency points.")
    return merged


def add_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in ("R_ohm_per_m", "L_h_per_m", "G_s_per_m", "C_f_per_m"):
        a = out[f"{name}_dut700"].to_numpy(dtype=np.float64)
        b = out[f"{name}_dut_700"].to_numpy(dtype=np.float64)
        out[f"{name}_delta"] = b - a
        out[f"{name}_abs_delta"] = np.abs(b - a)
        out[f"{name}_rel_delta_pct"] = np.where(np.abs(a) > EPS, (b - a) / a * 100.0, np.nan)
        out[f"{name}_abs_rel_delta_pct"] = np.abs(out[f"{name}_rel_delta_pct"])
    return out


def finite_metrics(values: pd.Series) -> tuple[float, float, float]:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.max(arr)), float(np.sqrt(np.mean(arr**2)))


def summarize_comparison(df: pd.DataFrame, length_m: float) -> dict[str, object]:
    parameters = {}
    for name, label in (
        ("R_ohm_per_m", "R"),
        ("L_h_per_m", "L"),
        ("G_s_per_m", "G"),
        ("C_f_per_m", "C"),
    ):
        mean_abs, max_abs, rms_abs = finite_metrics(df[f"{name}_abs_delta"])
        mean_rel, max_rel, rms_rel = finite_metrics(df[f"{name}_abs_rel_delta_pct"])
        max_idx = int(df[f"{name}_abs_delta"].idxmax())
        parameters[label] = {
            "mean_abs_delta": mean_abs,
            "max_abs_delta": max_abs,
            "rms_abs_delta": rms_abs,
            "mean_abs_rel_delta_pct": mean_rel,
            "max_abs_rel_delta_pct": max_rel,
            "rms_abs_rel_delta_pct": rms_rel,
            "max_delta_freq_ghz": float(df.loc[max_idx, "freq_ghz"]),
        }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_a": str(INPUT_A),
        "input_b": str(INPUT_B),
        "length_m": length_m,
        "frequency_points": int(len(df)),
        "frequency_min_ghz": float(df["freq_ghz"].min()),
        "frequency_max_ghz": float(df["freq_ghz"].max()),
        "parameters": parameters,
    }


def write_plots(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    specs = [
        ("R_ohm_per_m", "R (ohm/m)", axes[0, 0]),
        ("L_h_per_m", "L (H/m)", axes[0, 1]),
        ("G_s_per_m", "G (S/m)", axes[1, 0]),
        ("C_f_per_m", "C (F/m)", axes[1, 1]),
    ]
    for name, ylabel, ax in specs:
        ax.plot(df["freq_ghz"], df[f"{name}_dut700"], label="dut700.s2p", linewidth=1.8)
        ax.plot(df["freq_ghz"], df[f"{name}_dut_700"], label="dut_700.s2p", linewidth=1.4, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Frequency (GHz)")
    axes[1, 1].set_xlabel("Frequency (GHz)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("LHS800 TSV dut700 RLGC comparison")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "rlgc_curves.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for name, ylabel, ax in specs:
        ax.plot(df["freq_ghz"], df[f"{name}_rel_delta_pct"], linewidth=1.6)
        ax.set_ylabel(f"{ylabel} diff (%)")
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Frequency (GHz)")
    axes[1, 1].set_xlabel("Frequency (GHz)")
    fig.suptitle("dut_700.s2p relative to dut700.s2p")
    fig.tight_layout()
    diff_path = OUTPUT_DIR / "rlgc_relative_difference.png"
    fig.savefig(diff_path, dpi=200)
    plt.close(fig)
    return out_path


def write_verification(summary: dict[str, object], output_files: list[Path]) -> Path:
    path = OUTPUT_DIR / f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    lines = [
        "RLGC comparison verification",
        f"Created at: {summary['created_at']}",
        f"Input A exists: {INPUT_A.exists()} - {INPUT_A}",
        f"Input B exists: {INPUT_B.exists()} - {INPUT_B}",
        f"Length used: {summary['length_m']} m",
        f"Frequency points: {summary['frequency_points']}",
        f"Frequency range: {summary['frequency_min_ghz']} to {summary['frequency_max_ghz']} GHz",
        "Output files:",
    ]
    lines.extend(f"- {path}" for path in output_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    length = tsv_length_m(INPUT_A, INPUT_B)
    rlgc_a = extract_rlgc(INPUT_A, length)
    rlgc_b = extract_rlgc(INPUT_B, length)
    rlgc_a.to_csv(OUTPUT_DIR / "dut700_rlgc.csv", index=False, encoding="utf-8-sig")
    rlgc_b.to_csv(OUTPUT_DIR / "dut_700_rlgc.csv", index=False, encoding="utf-8-sig")

    comparison = add_diff_columns(align_by_frequency(rlgc_a, rlgc_b))
    comparison_path = OUTPUT_DIR / "rlgc_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    summary = summarize_comparison(comparison, length)
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_path = write_plots(comparison)
    output_files = [
        OUTPUT_DIR / "dut700_rlgc.csv",
        OUTPUT_DIR / "dut_700_rlgc.csv",
        comparison_path,
        summary_path,
        plot_path,
        OUTPUT_DIR / "rlgc_relative_difference.png",
    ]
    verification_path = write_verification(summary, output_files)

    print(f"RLGC comparison saved to: {OUTPUT_DIR}")
    print(f"Verification archived to: {verification_path}")
    for name, metrics in summary["parameters"].items():
        print(
            f"{name}: mean abs rel diff={metrics['mean_abs_rel_delta_pct']:.6g}% "
            f"max abs rel diff={metrics['max_abs_rel_delta_pct']:.6g}%"
        )


if __name__ == "__main__":
    main()
