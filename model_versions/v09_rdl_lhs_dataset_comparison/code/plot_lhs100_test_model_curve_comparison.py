# -*- coding: utf-8 -*-
"""Plot curve-level comparison of five RDL models on LHS100/test data.

Run directly in VS Code. No command-line arguments are required.
"""

import os
import sys
from pathlib import Path

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "results"
FINETUNE_ROOT = RESULT_ROOT / "sparam_finetuned_models"
PARAM_TABLE_ROOT = RESULT_ROOT / "extracted_params"
SUMMARY_CSV = FINETUNE_ROOT / "summary_metrics.csv"
OUTPUT_DIR = RESULT_ROOT / "lhs100_test_model_curve_comparison"

SAVE_PLOTS = True
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1").strip().lower() not in {"0", "false", "no"}
DPI = 180
MAX_PLOTS_PER_DEVICE = 5
SELECT_BY = "worst_all_model_mean_mse"  # uses LHS100/test samples.

DATASET_ORDER = ["lhs100", "lhs200", "lhs400", "lhs800", "lhs100_lhs200_lhs400_lhs800"]
DATASET_LABELS = {
    "lhs100": "LHS100",
    "lhs200": "LHS200",
    "lhs400": "LHS400",
    "lhs800": "LHS800",
    "lhs100_lhs200_lhs400_lhs800": "All",
}
DEVICE_CONFIGS = {
    "TMRDL": {
        "length_column": "l_tmrdl",
    },
    "BSMRDL": {
        "length_column": "l_bsmrdl",
    },
}
TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
Z_REF = 50.0
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def circuit_params_to_s_np(params, length_um, freqs_hz):
    params_t = torch.tensor(np.asarray(params, dtype=np.float64).reshape(1, -1), dtype=REAL_DTYPE)
    length_t = torch.tensor([float(length_um)], dtype=REAL_DTYPE)
    with torch.no_grad():
        return circuit_params_to_s_torch(params_t, length_t, freqs_hz).cpu().numpy()[0]


def circuit_params_to_s_torch(params, length_um, freqs_hz):
    r1, r2, r3 = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    l1, l2, l3 = params[:, 3:4] * 1e-9, params[:, 4:5] * 1e-9, params[:, 5:6] * 1e-9
    cox, csi, rsi = params[:, 6:7] * 1e-12, params[:, 7:8] * 1e-12, params[:, 8:9]
    omega = torch.tensor(2.0 * np.pi * freqs_hz, dtype=REAL_DTYPE, device=params.device)[None, :]
    length_m = length_um[:, None].to(params.device) * 1e-6
    j = torch.complex(
        torch.tensor(0.0, dtype=REAL_DTYPE, device=params.device),
        torch.tensor(1.0, dtype=REAL_DTYPE, device=params.device),
    )
    r_rlgc = (r1**2 * r2 + r1 * r2**2 + omega**2 * r1 * l2**2) / (
        (r1 + r2) ** 2 + omega**2 * l2**2 + 1e-30
    ) + (omega**2 * l3**2 * r3) / (r3**2 + omega**2 * l3**2 + 1e-30)
    l_rlgc = (r1**2 * l2) / ((r1 + r2) ** 2 + omega**2 * l2**2 + 1e-30) + l3 * r3**2 / (
        r3**2 + omega**2 * l3**2 + 1e-30
    ) + l1
    g_rlgc = (omega**2 * rsi * cox**2) / (1.0 + omega**2 * rsi**2 * (cox + csi) ** 2 + 1e-30)
    c_rlgc = (cox + omega**2 * csi * rsi**2 * cox * (cox + csi)) / (
        1.0 + omega**2 * rsi**2 * (cox + csi) ** 2 + 1e-30
    )
    z0 = torch.sqrt(
        (r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE))
        / (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE) + 1e-300)
    )
    gamma = torch.sqrt(
        (r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE))
        * (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE))
    )
    gl = gamma * length_m.to(COMPLEX_DTYPE)
    a = torch.cosh(gl)
    b = z0 * torch.sinh(gl)
    c = torch.sinh(gl) / (z0 + 1e-300)
    d = torch.cosh(gl)
    denom = a + b / Z_REF + c * Z_REF + d + 1e-30
    s = torch.zeros((*a.shape, 2, 2), dtype=COMPLEX_DTYPE, device=params.device)
    s[..., 0, 0] = (a + b / Z_REF - c * Z_REF - d) / denom
    s[..., 0, 1] = 2.0 * (a * d - b * c) / denom
    s[..., 1, 0] = 2.0 / denom
    s[..., 1, 1] = (-a + b / Z_REF - c * Z_REF + d) / denom
    return s


def load_prediction_table(dataset_name, device_name):
    csv_path = FINETUNE_ROOT / dataset_name / device_name / "finetuned_predicted_circuit_params.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predicted parameter CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    length_col = DEVICE_CONFIGS[device_name]["length_column"]
    if length_col not in df.columns:
        param_path = PARAM_TABLE_ROOT / dataset_name / f"{device_name}_circuit_params.csv"
        param_df = pd.read_csv(param_path, encoding="utf-8-sig")
        key_cols = ["source_root", "source_split", "split", "idx", "snp_path"]
        df = df.merge(param_df[key_cols + [length_col]], on=key_cols, how="left")
    return df[df["split"].eq("test") & df["source_root"].eq("LHS100")].copy().reset_index(drop=True)


def build_sample_errors(device_name):
    rows = []
    first_table = load_prediction_table(DATASET_ORDER[0], device_name)
    for _, base_row in first_table.iterrows():
        idx = int(base_row["idx"])
        hfss = rf.Network(str(base_row["snp_path"]))
        mse_values = []
        for dataset_name in DATASET_ORDER:
            table = load_prediction_table(dataset_name, device_name)
            match = table[table["idx"].eq(idx)]
            if match.empty:
                continue
            row = match.iloc[0]
            params = row[[f"pred_{name}" for name in TARGET_PARAMS]].to_numpy(dtype=np.float64)
            pred_s = circuit_params_to_s_np(params, float(row[DEVICE_CONFIGS[device_name]["length_column"]]), hfss.f)
            mse_values.append(float(np.mean(np.abs(pred_s - hfss.s) ** 2)))
        rows.append({"idx": idx, "snp_path": base_row["snp_path"], "mean_mse": float(np.mean(mse_values))})
    return pd.DataFrame(rows).sort_values("mean_mse", ascending=False).reset_index(drop=True)


def collect_curves(device_name, dut_index):
    curves = {}
    hfss_network = None
    length_col = DEVICE_CONFIGS[device_name]["length_column"]
    for dataset_name in DATASET_ORDER:
        table = load_prediction_table(dataset_name, device_name)
        match = table[table["idx"].eq(int(dut_index))]
        if match.empty:
            raise ValueError(f"Missing {device_name} dut{dut_index} in {dataset_name}")
        row = match.iloc[0]
        if hfss_network is None:
            hfss_network = rf.Network(str(row["snp_path"]))
        params = row[[f"pred_{name}" for name in TARGET_PARAMS]].to_numpy(dtype=np.float64)
        curves[dataset_name] = circuit_params_to_s_np(params, float(row[length_col]), hfss_network.f)
    return hfss_network.f, hfss_network.s, curves


def save_or_show(fig, filename):
    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR / filename}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_one_sample(device_name, dut_index):
    freq, hfss_s, curves = collect_curves(device_name, dut_index)
    freq_ghz = freq / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"LHS100/test {device_name} dut{dut_index}: Five Model Comparison", x=0.02, ha="left", fontsize=13)
    ports = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]
    cmap = plt.get_cmap("tab10")
    for ax, (m, n, label) in zip(axes.ravel(), ports):
        ax.plot(freq_ghz, db20(hfss_s[:, m, n]), color="black", linewidth=2.0, label="HFSS")
        for i, dataset_name in enumerate(DATASET_ORDER):
            ax.plot(
                freq_ghz,
                db20(curves[dataset_name][:, m, n]),
                linewidth=1.25,
                color=cmap(i),
                label=DATASET_LABELS[dataset_name],
            )
        ax.set_title(label)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_or_show(fig, f"{device_name.lower()}_dut{int(dut_index)}_five_model_curve_compare.png")


def write_selected_samples(selected):
    rows = []
    for device_name, df in selected.items():
        out = df.copy()
        out.insert(0, "device", device_name)
        rows.append(out)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "selected_lhs100_test_samples.csv"
    pd.concat(rows, ignore_index=True).to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_file}")


def main():
    selected = {}
    for device_name in DEVICE_CONFIGS:
        errors = build_sample_errors(device_name)
        chosen = errors.head(MAX_PLOTS_PER_DEVICE).copy()
        selected[device_name] = chosen
        for dut_index in chosen["idx"]:
            plot_one_sample(device_name, int(dut_index))
    write_selected_samples(selected)
    print(f"Done. Plot directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
