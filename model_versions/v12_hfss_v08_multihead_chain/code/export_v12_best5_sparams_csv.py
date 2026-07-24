# -*- coding: utf-8 -*-
"""Export S11/S21 real/imag CSVs for the five best current v12 model samples.

Run this file directly in VS Code. No command-line arguments are required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_v12_hfss_v08_symmetric_multihead as v12


VERSION_ROOT = PROJECT_ROOT / "model_versions" / "v12_hfss_v08_multihead_chain"
CURRENT_BEST_DIR = VERSION_ROOT / "results" / "current_best_model"
CURRENT_BEST_CHECKPOINT = CURRENT_BEST_DIR / "v08_connection_multihead_current_best.pt"
ROUND3_DIR = VERSION_ROOT / "results" / "hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round3"
ROUND3_METRICS = ROUND3_DIR / "v08_sparam_metrics.csv"
OUTPUT_DIR = VERSION_ROOT / "results" / "best5_sparameter_csv_current_best_round3"

# Keep this as a code-defined default so the script can be run directly in VS Code.
# Use "all" to select across train+test, or "test" to select held-out samples only.
SELECT_SPLIT = "test"
BEST_N = 5


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            cells.append(f"{value:.8g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def force_original_train_test_split(dut_df: pd.DataFrame) -> pd.DataFrame:
    out = dut_df.copy()
    out["split"] = np.where(out["sample_id"].astype(str).str.contains("_train_"), "train", "test")
    return out


def prepare_arrays(dut_df: pd.DataFrame, sim, metadata: dict):
    x_raw = dut_df[v12.base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    x_mean = np.asarray(metadata["x_mean"], dtype=np.float64)
    x_std = np.maximum(np.asarray(metadata["x_std"], dtype=np.float64), 1e-30)
    x_norm = (x_raw - x_mean) / x_std
    y_mean = np.asarray(metadata["y_mean"], dtype=np.float64)
    y_std = np.maximum(np.asarray(metadata["y_std"], dtype=np.float64), 1e-30)
    return x_norm, y_mean, y_std


def s_to_source_columns(prefix: str, s: np.ndarray) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_s11_real": np.real(s[:, 0, 0]),
        f"{prefix}_s11_imag": np.imag(s[:, 0, 0]),
        f"{prefix}_s21_real": np.real(s[:, 1, 0]),
        f"{prefix}_s21_imag": np.imag(s[:, 1, 0]),
    }


def s_to_wide_rows(sample_id: str, split: str, dut_index: int, freq_hz: np.ndarray, hfss_s: np.ndarray, direct_s: np.ndarray, model_s: np.ndarray):
    rows = []
    source_cols = {}
    source_cols.update(s_to_source_columns("hfss", hfss_s))
    source_cols.update(s_to_source_columns("direct_cascade", direct_s))
    source_cols.update(s_to_source_columns("cascade_model", model_s))
    for freq_idx, freq in enumerate(freq_hz):
        row = {
            "sample_id": sample_id,
            "split": split,
            "dut_index": int(dut_index),
            "freq_hz": float(freq),
            "freq_ghz": float(freq / 1e9),
        }
        for key, values in source_cols.items():
            row[key] = float(values[freq_idx])
        rows.append(row)
    return rows


def selected_metrics() -> pd.DataFrame:
    if not ROUND3_METRICS.exists():
        raise FileNotFoundError(f"Missing round3 metrics: {ROUND3_METRICS}")
    metrics = pd.read_csv(ROUND3_METRICS, encoding="utf-8-sig")
    if SELECT_SPLIT != "all":
        metrics = metrics[metrics["split"].eq(SELECT_SPLIT)].copy()
    if len(metrics) < BEST_N:
        raise ValueError(f"Only {len(metrics)} samples available for split={SELECT_SPLIT!r}; need {BEST_N}.")
    return metrics.sort_values("v08_nn_nmse_s11_s21_ri", ascending=True).head(BEST_N).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CURRENT_BEST_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing current best checkpoint: {CURRENT_BEST_CHECKPOINT}")

    selected = selected_metrics()

    v12.base.RUN_LABEL = OUTPUT_DIR.name
    v12.base.OUTPUT_DIR = OUTPUT_DIR
    v12.base.SIMULATION_BACKEND = "hfss_equivalent_circuit"
    v12.base.USE_MODEL_SET_AS_VALIDATION = True
    v12.base.collect_samples = lambda: v12.shared.collect_v11_samples(v12.base)
    v12.base.load_single_device_simulation = v12.load_hfss_equivalent_simulation

    checkpoint = torch.load(CURRENT_BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    dut_df = force_original_train_test_split(v12.base.collect_samples())
    sim = v12.load_hfss_equivalent_simulation(
        dut_df,
        {
            "version": "v12",
            "source_checkpoint": str(CURRENT_BEST_CHECKPOINT),
            "export": "best five S11/S21 real/imag wide CSV",
            "selection_split": SELECT_SPLIT,
        },
    )
    x_norm, y_mean, y_std = prepare_arrays(dut_df, sim, metadata)

    device = torch.device("cpu")
    model = v12.SymmetricV08ConnectionNet(input_dim=len(metadata["feature_columns"])).to(
        dtype=v12.base.REAL_DTYPE, device=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    omega_np = 2.0 * np.pi * sim.freq_hz
    omega_t = torch.tensor(omega_np, dtype=v12.base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=v12.base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=v12.base.REAL_DTYPE, device=device)

    sample_dir = OUTPUT_DIR / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    export_rows = []

    for rank, metric in selected.iterrows():
        sample_id = str(metric["sample_id"])
        idx_values = dut_df.index[dut_df["sample_id"].eq(sample_id)].to_list()
        if not idx_values:
            raise ValueError(f"Selected sample not found in collected samples: {sample_id}")
        idx = int(idx_values[0])
        row = dut_df.iloc[idx]

        target_s = sim.target_s[idx]
        direct_s = v12.base.abcd2s(v12.base.opt2.cascade_direct(list(sim.base_abcds[idx])))
        x_b = torch.tensor(x_norm[idx : idx + 1], dtype=v12.base.REAL_DTYPE, device=device)
        base_b = torch.tensor(sim.base_abcds[idx : idx + 1], dtype=v12.base.COMPLEX_DTYPE, device=device)
        with torch.no_grad():
            p_flat = v12.base.denormalize_params(model(x_b), y_mean_t, y_std_t)
            p_all = p_flat.reshape(-1, v12.shared.CONNECTION_COUNT, len(v12.shared.V08_PARAM_NAMES))
            model_s = v12.base.abcd2s_torch(v12.shared.cascade_with_v08_torch(v12.base, base_b, p_all, omega_t)).cpu().numpy()[0]

        sample_rows = s_to_wide_rows(sample_id, row["split"], int(row["dut_index"]), sim.freq_hz, target_s, direct_s, model_s)
        sample_df = pd.DataFrame(sample_rows)
        sample_csv = sample_dir / f"{rank + 1:02d}_{sample_id}_sparameters.csv"
        sample_df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
        all_rows.extend(sample_rows)

        export_rows.append(
            {
                "rank": int(rank + 1),
                "sample_id": sample_id,
                "split": row["split"],
                "dut_index": int(row["dut_index"]),
                "source_snp": str(row["snp_path"]),
                "csv_file": str(sample_csv),
                "direct_nmse_s11_s21_ri": float(metric["direct_nmse_s11_s21_ri"]),
                "cascade_model_nmse_s11_s21_ri": float(metric["v08_nn_nmse_s11_s21_ri"]),
                "direct_mse_vs_target": float(metric["direct_mse_vs_target"]),
                "cascade_model_mse_vs_target": float(metric["v08_nn_mse_vs_target"]),
                "cascade_model_s11_db_mae": float(metric["v08_nn_s11_db_mae"]),
                "cascade_model_s21_db_mae": float(metric["v08_nn_s21_db_mae"]),
            }
        )

    selected_export = pd.DataFrame(export_rows)
    combined = pd.DataFrame(all_rows)
    selected_export.to_csv(OUTPUT_DIR / "selected_best5_samples.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT_DIR / "best5_sparameters_combined.csv", index=False, encoding="utf-8-sig")

    report = {
        "entry": Path(__file__).name,
        "current_best_checkpoint": str(CURRENT_BEST_CHECKPOINT),
        "source_metrics": str(ROUND3_METRICS),
        "selection_split": SELECT_SPLIT,
        "best_n": BEST_N,
        "output_dir": str(OUTPUT_DIR),
        "sample_csv_dir": str(sample_dir),
        "selected_samples": selected_export.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 Best-5 S-Parameter CSV Export",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Current best checkpoint: `{CURRENT_BEST_CHECKPOINT}`",
                f"- Source metrics: `{ROUND3_METRICS}`",
                f"- Selection: best `{BEST_N}` samples by `v08_nn_nmse_s11_s21_ri` in split `{SELECT_SPLIT}`.",
                "- Exported columns per sample: one frequency column set plus S11/S21 real/imag columns for `hfss`, `direct_cascade`, and `cascade_model`.",
                f"- Per-sample CSV directory: `{sample_dir}`",
                f"- Combined CSV: `{OUTPUT_DIR / 'best5_sparameters_combined.csv'}`",
                "",
                "## Selected Samples",
                "",
                dataframe_to_markdown(selected_export),
            ]
        ),
        encoding="utf-8",
    )
    print(selected_export.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
