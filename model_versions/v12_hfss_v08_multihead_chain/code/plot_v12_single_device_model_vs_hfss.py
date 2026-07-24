# -*- coding: utf-8 -*-
"""Plot v12 single-device model predictions against HFSS simulations.

Run this file directly in VS Code. No command-line arguments are required.
It evaluates the available independent LHS400 single-device HFSS data for
TMRDL, BSMRDL, and TSV, then writes metrics and comparison plots.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio
import skrf as rf
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_versions"
    / "v12_hfss_v08_multihead_chain"
    / "results"
    / "single_device_model_vs_hfss_lhs400"
)

V03_SCRIPT = PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "code" / "train_single_device_sparam_model.py"
V09_SCRIPT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "code" / "finetune_matlab_rdl_models_on_sparams.py"
RDL_MODEL_DIR = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "models" / "matlab_param_nns" / "lhs400"
TSV_CHECKPOINT = (
    PROJECT_ROOT
    / "model_versions"
    / "v03_single_device_sparam_finetune"
    / "results"
    / "single_device_sparam_TSV_mat4_init_sparam_noanchor"
    / "single_device_sparam_net.pt"
)

DEVICE_CONFIGS = {
    "TMRDL": {
        "csv": PROJECT_ROOT / "HFSS_sim" / "LHS400" / "train" / "TMRDL_variations_record.csv",
        "snp_dir": PROJECT_ROOT / "HFSS_sim" / "LHS400" / "train" / "TMRDL",
        "features": ["pitch", "l_tmrdl", "w_tmrdl", "h_tmrdl"],
        "length": "l_tmrdl",
        "model_prefix": "TMRDL",
    },
    "BSMRDL": {
        "csv": PROJECT_ROOT / "HFSS_sim" / "LHS400" / "train" / "BSMRDL_variations_record.csv",
        "snp_dir": PROJECT_ROOT / "HFSS_sim" / "LHS400" / "train" / "BSMRDL",
        "features": ["pitch", "l_bsmrdl", "w_bsmrdl", "h_bsmrdl"],
        "length": "l_bsmrdl",
        "model_prefix": "BSMRDL",
    },
    "TSV": {
        "csv": PROJECT_ROOT / "HFSS_sim" / "LHS400" / "train" / "TSV_variations_record.csv",
        "snp_dir": PROJECT_ROOT / "HFSS_sim" / "LHS400" / "train" / "TSV",
        "features": ["r_tsv", "h_tsv", "pitch"],
        "length": "h_tsv",
        "model_prefix": "TSV",
    },
}

TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
REAL_DTYPE = torch.float64
PLOT_RANDOM_COUNT = 6
PLOT_WORST_COUNT = 6
RANDOM_SEED = 20260712


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v03 = load_module(V03_SCRIPT, "v12_plot_v03_single_device")
v09 = load_module(V09_SCRIPT, "v12_plot_v09_rdl")


class MatlabSingleParamNet(nn.Module):
    def __init__(self, mat_path: Path):
        super().__init__()
        data = sio.loadmat(mat_path)
        self.register_buffer("psmin", torch.tensor(data["psmin"], dtype=REAL_DTYPE).reshape(1, -1))
        self.register_buffer("psmax", torch.tensor(data["psmax"], dtype=REAL_DTYPE).reshape(1, -1))
        self.register_buffer("outputmin", torch.tensor(data["outputmin"], dtype=REAL_DTYPE).reshape(1, 1))
        self.register_buffer("outputmax", torch.tensor(data["outputmax"], dtype=REAL_DTYPE).reshape(1, 1))
        self.register_buffer("w1", torch.tensor(data["w1"], dtype=REAL_DTYPE))
        self.register_buffer("theta1", torch.tensor(data["theta1"], dtype=REAL_DTYPE))
        self.register_buffer("w2", torch.tensor(data["w2"], dtype=REAL_DTYPE))
        self.register_buffer("theta2", torch.tensor(data["theta2"], dtype=REAL_DTYPE))
        self.register_buffer("w3", torch.tensor(data["w3"], dtype=REAL_DTYPE))
        self.register_buffer("theta3", torch.tensor(data["theta3"], dtype=REAL_DTYPE))

    def forward(self, x_raw):
        x = 2.0 * (x_raw - self.psmin) / torch.clamp(self.psmax - self.psmin, min=1e-30) - 1.0
        y = torch.tanh(x @ self.w1 + self.theta1)
        y = torch.tanh(y @ self.w2 + self.theta2)
        y = y @ self.w3 + self.theta3
        return (self.outputmin + (y + 1.0) * (self.outputmax - self.outputmin) / 2.0).squeeze(-1)


class RdlParamModel(nn.Module):
    def __init__(self, prefix: str):
        super().__init__()
        self.nets = nn.ModuleList([MatlabSingleParamNet(RDL_MODEL_DIR / f"{prefix}_{name}.mat") for name in TARGET_PARAMS])

    def forward(self, x_raw):
        return torch.stack([net(x_raw) for net in self.nets], dim=1)


class TsvParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint = torch.load(TSV_CHECKPOINT, map_location="cpu", weights_only=False)
        self.metadata = checkpoint["metadata"]
        self.model = v03.Mat4InitializedDeviceNet(v03.DEVICE_CONFIGS["TSV"], self.metadata).to(dtype=REAL_DTYPE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def forward(self, x_raw):
        x_mean = torch.tensor(self.metadata["x_mean"], dtype=REAL_DTYPE).reshape(1, -1)
        x_std = torch.tensor(self.metadata["x_std"], dtype=REAL_DTYPE).reshape(1, -1)
        y_mean = torch.tensor(self.metadata["y_log_mean"], dtype=REAL_DTYPE).reshape(1, -1)
        y_std = torch.tensor(self.metadata["y_log_std"], dtype=REAL_DTYPE).reshape(1, -1)
        x_norm = (x_raw - x_mean) / torch.clamp(x_std, min=1e-30)
        y_norm = self.model(x_norm)
        return torch.exp(y_norm * y_std + y_mean)


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def wrapped_phase_delta_deg(pred, target):
    delta = np.angle(pred) - np.angle(target)
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    return np.rad2deg(delta)


def s11_s21_real_imag_y(s_params):
    s11 = s_params[:, 0, 0]
    s21 = s_params[:, 1, 0]
    return np.column_stack([s11.real, s11.imag, s21.real, s21.imag]).ravel()


def nmse_s11_s21_ri(target, pred):
    y_true = s11_s21_real_imag_y(target)
    y_pred = s11_s21_real_imag_y(pred)
    return float(np.sum((y_true - y_pred) ** 2) / max(np.sum((y_true - y_true.mean()) ** 2), 1e-30))


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def load_device_table(device_name: str) -> pd.DataFrame:
    cfg = DEVICE_CONFIGS[device_name]
    df = pd.read_csv(cfg["csv"], encoding="utf-8-sig").sort_values("dut_index").reset_index(drop=True)
    df["snp_path"] = df["dut_index"].map(lambda idx: cfg["snp_dir"] / f"dut{int(idx)}.s2p")
    missing = [str(path) for path in df["snp_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{device_name} missing S-parameter files: {missing[:5]}")
    return df


def predict_device_s(device_name: str, model: nn.Module, row: pd.Series, freq_hz: np.ndarray):
    cfg = DEVICE_CONFIGS[device_name]
    if device_name == "TSV":
        x = np.array([[float(row["r_tsv"]), row["h_tsv"], row["pitch"]]], dtype=np.float64)
    else:
        x = row[cfg["features"]].to_numpy(dtype=np.float64).reshape(1, -1)
    with torch.no_grad():
        params = model(torch.tensor(x, dtype=REAL_DTYPE)).cpu().numpy()[0]
    return v09.circuit_params_to_s_np(params, float(row[cfg["length"]]), freq_hz), params


def plot_one(device_name: str, row: pd.Series, hfss_s: np.ndarray, pred_s: np.ndarray, out_path: Path):
    freq_ghz = row["_freq_hz"] / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle(f"{device_name} dut{int(row['dut_index'])} model vs HFSS", x=0.02, y=0.98, ha="left")
    specs = [
        (0, 0, "S11 magnitude (dB)", lambda s: db20(s[:, 0, 0])),
        (1, 0, "S21 magnitude (dB)", lambda s: db20(s[:, 1, 0])),
        (0, 0, "S11 phase (deg)", lambda s: np.rad2deg(np.unwrap(np.angle(s[:, 0, 0])))),
        (1, 0, "S21 phase (deg)", lambda s: np.rad2deg(np.unwrap(np.angle(s[:, 1, 0])))),
    ]
    for ax, (m, n, title, fn) in zip(axes.ravel(), specs):
        ax.plot(freq_ghz, fn(hfss_s), label="HFSS", color="black", linewidth=1.8)
        ax.plot(freq_ghz, fn(pred_s), label="Model", color="#dc2626", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_device(device_name: str, model: nn.Module) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_device_table(device_name)
    metric_rows = []
    param_rows = []
    plot_cache = {}
    for i, row in df.iterrows():
        hfss = rf.Network(str(row["snp_path"]))
        pred_s, params = predict_device_s(device_name, model, row, hfss.f)
        metric_rows.append(
            {
                "device": device_name,
                "dut_index": int(row["dut_index"]),
                "s_mse": float(np.mean(np.abs(pred_s - hfss.s) ** 2)),
                "nmse_s11_s21_ri": nmse_s11_s21_ri(hfss.s, pred_s),
                "s11_db_mae": float(np.mean(np.abs(db20(pred_s[:, 0, 0]) - db20(hfss.s[:, 0, 0])))),
                "s21_db_mae": float(np.mean(np.abs(db20(pred_s[:, 1, 0]) - db20(hfss.s[:, 1, 0])))),
                "s11_phase_mae_deg": float(np.mean(np.abs(wrapped_phase_delta_deg(pred_s[:, 0, 0], hfss.s[:, 0, 0])))),
                "s21_phase_mae_deg": float(np.mean(np.abs(wrapped_phase_delta_deg(pred_s[:, 1, 0], hfss.s[:, 1, 0])))),
            }
        )
        param_row = {"device": device_name, "dut_index": int(row["dut_index"])}
        param_row.update({f"pred_{name}": float(params[j]) for j, name in enumerate(TARGET_PARAMS)})
        param_rows.append(param_row)
        plot_cache[int(row["dut_index"])] = (hfss.s, pred_s, hfss.f)
        if (i + 1) % 100 == 0:
            print(f"{device_name}: evaluated {i + 1}/{len(df)}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    params = pd.DataFrame(param_rows)
    plot_dir = OUTPUT_DIR / "plots" / device_name
    rng = np.random.default_rng(RANDOM_SEED)
    random_ids = metrics["dut_index"].sample(n=min(PLOT_RANDOM_COUNT, len(metrics)), random_state=RANDOM_SEED).tolist()
    worst_ids = metrics.sort_values("nmse_s11_s21_ri", ascending=False).head(PLOT_WORST_COUNT)["dut_index"].tolist()
    selected = [("random", random_ids), ("worst_nmse", worst_ids)]
    for group, dut_ids in selected:
        for dut_idx in dut_ids:
            table_row = df[df["dut_index"].eq(dut_idx)].iloc[0].copy()
            hfss_s, pred_s, freq = plot_cache[int(dut_idx)]
            table_row["_freq_hz"] = freq
            plot_one(device_name, table_row, hfss_s, pred_s, plot_dir / group / f"{device_name}_dut{dut_idx}_{group}.png")
    return metrics, params


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("device", as_index=False)
        .agg(
            count=("dut_index", "count"),
            s_mse_mean=("s_mse", "mean"),
            s_mse_median=("s_mse", "median"),
            nmse_s11_s21_ri_mean=("nmse_s11_s21_ri", "mean"),
            nmse_s11_s21_ri_median=("nmse_s11_s21_ri", "median"),
            s11_db_mae_mean=("s11_db_mae", "mean"),
            s21_db_mae_mean=("s21_db_mae", "mean"),
            s11_phase_mae_deg_mean=("s11_phase_mae_deg", "mean"),
            s21_phase_mae_deg_mean=("s21_phase_mae_deg", "mean"),
        )
        .sort_values("device")
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        "TMRDL": RdlParamModel("TMRDL").to(dtype=REAL_DTYPE).eval(),
        "BSMRDL": RdlParamModel("BSMRDL").to(dtype=REAL_DTYPE).eval(),
        "TSV": TsvParamModel().to(dtype=REAL_DTYPE).eval(),
    }
    all_metrics = []
    all_params = []
    for device_name, model in models.items():
        metrics, params = evaluate_device(device_name, model)
        all_metrics.append(metrics)
        all_params.append(params)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    params_df = pd.concat(all_params, ignore_index=True)
    summary_df = summarize(metrics_df)
    metrics_df.to_csv(OUTPUT_DIR / "single_device_model_vs_hfss_metrics.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(OUTPUT_DIR / "single_device_model_predicted_params.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "single_device_model_vs_hfss_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "entry": str(Path(__file__).name),
        "output_dir": str(OUTPUT_DIR),
        "dataset": "HFSS_sim/LHS400/train independent single-device data",
        "rdl_model_dir": str(RDL_MODEL_DIR),
        "tsv_checkpoint": str(TSV_CHECKPOINT),
        "summary": summary_df.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "single_device_model_vs_hfss_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "validation_archive.md").write_text(
        "\n".join(
            [
                "# v12 Single-Device Model vs HFSS Validation",
                "",
                f"- Entry: `{Path(__file__).name}`",
                "- Dataset: `HFSS_sim/LHS400/train/TMRDL|BSMRDL|TSV`.",
                f"- Output: `{OUTPUT_DIR}`",
                f"- Metrics CSV: `{OUTPUT_DIR / 'single_device_model_vs_hfss_metrics.csv'}`",
                f"- Summary CSV: `{OUTPUT_DIR / 'single_device_model_vs_hfss_summary.csv'}`",
                f"- Plots: `{OUTPUT_DIR / 'plots'}`",
                "",
                "## Summary",
                "",
                dataframe_to_markdown(summary_df),
            ]
        ),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False), flush=True)
    print(f"Done: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
