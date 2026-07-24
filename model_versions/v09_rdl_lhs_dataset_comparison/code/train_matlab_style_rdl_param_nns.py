# -*- coding: utf-8 -*-
"""Train MATLAB-style RDL parameter NNs and export MATLAB-compatible .mat files.

This is a fallback/direct-run Python implementation of the model structure used
by nn_train_3.m:

    min-max input normalization -> tanh(20) -> tanh(20) -> linear output
    min-max output denormalization

The exported .mat files contain psmin, psmax, outputmin, outputmax, w1/theta1,
w2/theta2, and w3/theta3, so the S-parameter fine-tuning script can read them
the same way as MATLAB-exported files.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAM_TABLE_ROOT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "results" / "extracted_params"
MODEL_ROOT = PROJECT_ROOT / "model_versions" / "v09_rdl_lhs_dataset_comparison" / "models" / "matlab_param_nns"

DATASET_NAMES = ["lhs100", "lhs200", "lhs400", "lhs800", "lhs100_lhs200_lhs400_lhs800"]
DEVICE_CONFIGS = {
    "TMRDL": {
        "features": ["pitch", "l_tmrdl", "w_tmrdl", "h_tmrdl"],
        "model_prefix": "TMRDL_",
    },
    "BSMRDL": {
        "features": ["pitch", "l_bsmrdl", "w_bsmrdl", "h_bsmrdl"],
        "model_prefix": "BSMRDL_",
    },
}
TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]

RANDOM_SEED = 20260706
USE_CUDA_IF_AVAILABLE = True
REAL_DTYPE = torch.float64
EPOCHS = 1800
PATIENCE = 220
BATCH_SIZE = 128
LR = 8e-4
WEIGHT_DECAY = 1e-6
PRINT_EVERY = 100


class SingleParamNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, 20)
        self.w2 = nn.Linear(20, 20)
        self.w3 = nn.Linear(20, 1)

    def forward(self, x):
        y = torch.tanh(self.w1(x))
        y = torch.tanh(self.w2(y))
        return self.w3(y).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def minmax_normalize(values):
    vmin = values.min(axis=0, keepdims=True)
    vmax = values.max(axis=0, keepdims=True)
    denom = np.maximum(vmax - vmin, 1e-30)
    return 2.0 * (values - vmin) / denom - 1.0, vmin.squeeze(0), vmax.squeeze(0)


def train_val_split(n_rows):
    indices = np.arange(n_rows)
    np.random.shuffle(indices)
    n_val = max(1, int(round(n_rows * 0.15)))
    return indices[n_val:], indices[:n_val]


def train_one_param(x_norm, y_norm, device):
    train_idx, val_idx = train_val_split(len(x_norm))
    x_train = torch.tensor(x_norm[train_idx], dtype=REAL_DTYPE)
    y_train = torch.tensor(y_norm[train_idx], dtype=REAL_DTYPE)
    x_val = torch.tensor(x_norm[val_idx], dtype=REAL_DTYPE, device=device)
    y_val = torch.tensor(y_norm[val_idx], dtype=REAL_DTYPE, device=device)

    model = SingleParamNet(x_norm.shape[1]).to(dtype=REAL_DTYPE, device=device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=60, factor=0.5)
    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            n_seen += len(xb)

        train_loss = total / max(n_seen, 1)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(x_val) - y_val) ** 2).item()
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"    epoch={epoch}, train={train_loss:.6e}, val={val_loss:.6e}", flush=True)
        if stale >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def export_matlab_style(model, x_raw, y_raw, mat_file, valname):
    psmin = x_raw.min(axis=0, keepdims=True)
    psmax = x_raw.max(axis=0, keepdims=True)
    outputmin = np.array([[float(y_raw.min())]], dtype=np.float64)
    outputmax = np.array([[float(y_raw.max())]], dtype=np.float64)
    state = model.state_dict()
    data = {
        "psmax": psmax.astype(np.float64),
        "psmin": psmin.astype(np.float64),
        "w1": state["w1.weight"].detach().cpu().numpy().T.astype(np.float64),
        "theta1": state["w1.bias"].detach().cpu().numpy().reshape(1, -1).astype(np.float64),
        "w2": state["w2.weight"].detach().cpu().numpy().T.astype(np.float64),
        "theta2": state["w2.bias"].detach().cpu().numpy().reshape(1, -1).astype(np.float64),
        "w3": state["w3.weight"].detach().cpu().numpy().T.astype(np.float64),
        "theta3": state["w3.bias"].detach().cpu().numpy().reshape(1, -1).astype(np.float64),
        "outputmax": outputmax,
        "outputmin": outputmin,
        "valname": np.array([valname], dtype=object),
    }
    mat_file.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(mat_file, data)


def evaluate_relative_error(model, x_norm, y_norm, y_min, y_max, y_raw, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_norm), 512):
            xb = torch.tensor(x_norm[start : start + 512], dtype=REAL_DTYPE, device=device)
            preds.append(model(xb).cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0)
    pred = y_min + (pred_norm + 1.0) * (y_max - y_min) / 2.0
    err = np.abs((pred - y_raw) / np.maximum(np.abs(y_raw), 1e-30))
    return float(np.max(err)), float(np.mean(err))


def train_dataset_device(dataset_name, device_name, device):
    config = DEVICE_CONFIGS[device_name]
    csv_path = PARAM_TABLE_ROOT / dataset_name / f"{device_name}_circuit_params.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["split"].eq("train")].dropna(subset=config["features"] + TARGET_PARAMS).reset_index(drop=True)
    x_raw = df[config["features"]].to_numpy(dtype=np.float64)
    x_norm, _, _ = minmax_normalize(x_raw)
    rows = []
    model_dir = MODEL_ROOT / dataset_name

    print(f"\nTraining MATLAB-style param NNs: {dataset_name} / {device_name}, samples={len(df)}", flush=True)
    for target_name in TARGET_PARAMS:
        y_raw = df[target_name].to_numpy(dtype=np.float64)
        valid = np.isfinite(y_raw) & (y_raw > 0.0) & np.all(np.isfinite(x_raw), axis=1)
        x_j = x_raw[valid]
        x_norm_j, _, _ = minmax_normalize(x_j)
        y_j = y_raw[valid]
        y_norm, y_min_arr, y_max_arr = minmax_normalize(y_j.reshape(-1, 1))
        y_norm = y_norm[:, 0]
        y_min = float(y_min_arr[0])
        y_max = float(y_max_arr[0])

        print(f"  {target_name}: samples={len(y_j)}", flush=True)
        model, best_val = train_one_param(x_norm_j, y_norm, device)
        max_err, avg_err = evaluate_relative_error(model, x_norm_j, y_norm, y_min, y_max, y_j, device)
        mat_file = model_dir / f"{config['model_prefix']}{target_name}.mat"
        export_matlab_style(model, x_j, y_j, mat_file, target_name)
        rows.append(
            {
                "dataset": dataset_name,
                "device": device_name,
                "parameter": target_name,
                "samples": int(len(y_j)),
                "trainer": "python_matlab_style",
                "best_internal_val_mse": best_val,
                "max_relative_error": max_err,
                "average_relative_error": avg_err,
                "mat_file": str(mat_file),
            }
        )
        print(f"    saved {mat_file}", flush=True)
    return rows


def main():
    set_seed(RANDOM_SEED)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    summary_rows = []
    for dataset_name in DATASET_NAMES:
        for device_name in DEVICE_CONFIGS:
            summary_rows.extend(train_dataset_device(dataset_name, device_name, device))

    summary = pd.DataFrame(summary_rows)
    summary_file = MODEL_ROOT / "matlab_training_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    with open(MODEL_ROOT / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "workflow": "python_fallback_for_nn_train_3_matlab_style_param_nns",
                "reason": "MATLAB CLI crashed before executing nn_train_3.m on this machine",
                "datasets": DATASET_NAMES,
                "devices": list(DEVICE_CONFIGS.keys()),
                "summary_file": str(summary_file),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nDone. Summary saved to {summary_file}", flush=True)


if __name__ == "__main__":
    main()
