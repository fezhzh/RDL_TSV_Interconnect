# -*- coding: utf-8 -*-
"""Train one RDL/TSV device model with S-parameter loss.

Run this file directly in VS Code. The default experiment trains one
``RDL_Bottom`` model in Python/PyTorch, using the existing CSV circuit-parameter
targets for pretraining and the HFSS single-device S-parameters for fine-tuning.
"""

import json
import random
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
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Configure these values before running directly from VS Code.
DEVICE_NAME = "RDL_Bottom"  # Used when DEVICE_NAMES_TO_TRAIN is empty.
DEVICE_NAMES_TO_TRAIN = ["RDL_Top", "TSV"]  # Direct VS Code run list.
RANDOM_SEED = 20260629
TRAIN_RATIO = 0.8
BATCH_SIZE_PARAM = 128
BATCH_SIZE_SPARAM = 32
PARAM_PRETRAIN_EPOCHS = 0
SPARAM_FINETUNE_EPOCHS = 800
PARAM_LR = 8e-4
SPARAM_LR = 2e-5
WEIGHT_DECAY = 1e-8
PARAM_PATIENCE = 180
SPARAM_PATIENCE = 160
PRINT_EVERY = 50
USE_CUDA_IF_AVAILABLE = True
PARAM_ANCHOR_WEIGHT = 0.0
PLOT_VAL_LIMIT = 40
INITIALIZE_FROM_MAT4 = True
RUN_PARAM_PRETRAIN = False
MAT4_INIT_OUTPUT_SUFFIX = "_mat4_init_sparam_noanchor"

TARGET_PARAMS = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
Z_REF = 50.0
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128

DEVICE_CONFIGS = {
    "RDL_Bottom": {
        "snp_dir": PROJECT_ROOT / "snp_data" / "RDL_Bottom_Snp",
        "csv": PROJECT_ROOT / "training_datasets" / "RDL_Bottom_TD_4.csv",
        "mat_dir": PROJECT_ROOT / "model_versions" / "v01_matlab_mat_models" / "models" / "RDL_TSV_mat4",
        "mat_prefix": "RDL_Bottom_",
        "feature_order": ["ldown", "wdown", "tdown", "htsv", "p1"],
        "csv_feature_order": ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl"],
        "length_header": "ldown",
        "output_dir": PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "results" / "single_device_sparam_RDL_Bottom",
    },
    "RDL_Top": {
        "snp_dir": PROJECT_ROOT / "snp_data" / "RDL_Top_Snp",
        "csv": PROJECT_ROOT / "training_datasets" / "RDL_Top_TD_4.csv",
        "mat_dir": PROJECT_ROOT / "model_versions" / "v01_matlab_mat_models" / "models" / "RDL_TSV_mat4",
        "mat_prefix": "RDL_Top_",
        "feature_order": ["lrdl", "wrdl", "trdl", "htsv", "p1"],
        "csv_feature_order": ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl"],
        "length_header": "lrdl",
        "output_dir": PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "results" / "single_device_sparam_RDL_Top",
    },
    "TSV": {
        "snp_dir": PROJECT_ROOT / "snp_data" / "TSV_Snp",
        "csv": PROJECT_ROOT / "training_datasets" / "TSV_TD_4.csv",
        "mat_dir": PROJECT_ROOT / "model_versions" / "v01_matlab_mat_models" / "models" / "RDL_TSV_mat4",
        "mat_prefix": "TSV_",
        "feature_order": ["dtsv", "htsv", "p1"],
        "csv_feature_order": ["d_tsv", "h_tsv", "p_rdl"],
        "length_header": "htsv",
        "output_dir": PROJECT_ROOT / "model_versions" / "v03_single_device_sparam_finetune" / "results" / "single_device_sparam_TSV",
    },
}


class SingleDeviceParamNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Mat4SingleParamNet(nn.Module):
    def __init__(self, mat_path):
        super().__init__()
        data = sio.loadmat(mat_path)
        self.register_buffer("psmin", torch.tensor(np.asarray(data["psmin"], dtype=np.float64).reshape(1, -1)))
        self.register_buffer("psmax", torch.tensor(np.asarray(data["psmax"], dtype=np.float64).reshape(1, -1)))
        self.register_buffer("outputmin", torch.tensor(float(np.asarray(data["outputmin"]).reshape(-1)[0]), dtype=REAL_DTYPE))
        self.register_buffer("outputmax", torch.tensor(float(np.asarray(data["outputmax"]).reshape(-1)[0]), dtype=REAL_DTYPE))
        self.w1 = nn.Parameter(torch.tensor(np.asarray(data["w1"], dtype=np.float64)))
        self.b1 = nn.Parameter(torch.tensor(np.asarray(data["theta1"], dtype=np.float64).reshape(1, -1)))
        self.w2 = nn.Parameter(torch.tensor(np.asarray(data["w2"], dtype=np.float64)))
        self.b2 = nn.Parameter(torch.tensor(np.asarray(data["theta2"], dtype=np.float64).reshape(1, -1)))
        self.w3 = nn.Parameter(torch.tensor(np.asarray(data["w3"], dtype=np.float64).reshape(-1, 1)))
        self.b3 = nn.Parameter(torch.tensor(float(np.asarray(data["theta3"]).reshape(-1)[0]), dtype=REAL_DTYPE))

    def forward(self, x_raw):
        x_norm = 2.0 * (x_raw - self.psmin) / (self.psmax - self.psmin + 1e-30) - 1.0
        y = torch.tanh(x_norm @ self.w1 + self.b1)
        y = torch.tanh(y @ self.w2 + self.b2)
        y_norm = y @ self.w3 + self.b3
        y = self.outputmin + (y_norm.squeeze(-1) + 1.0) * (self.outputmax - self.outputmin) / 2.0
        return y


class Mat4InitializedDeviceNet(nn.Module):
    def __init__(self, config, metadata):
        super().__init__()
        self.register_buffer("x_mean", torch.tensor(metadata["x_mean"], dtype=REAL_DTYPE).reshape(1, -1))
        self.register_buffer("x_std", torch.tensor(metadata["x_std"], dtype=REAL_DTYPE).reshape(1, -1))
        self.register_buffer("y_mean", torch.tensor(metadata["y_log_mean"], dtype=REAL_DTYPE).reshape(1, -1))
        self.register_buffer("y_std", torch.tensor(metadata["y_log_std"], dtype=REAL_DTYPE).reshape(1, -1))
        self.param_nets = nn.ModuleList(
            [Mat4SingleParamNet(config["mat_dir"] / f"{config['mat_prefix']}{name}.mat") for name in TARGET_PARAMS]
        )

    def forward(self, x_norm):
        x_raw = x_norm * self.x_std + self.x_mean
        params = torch.stack([net(x_raw) for net in self.param_nets], dim=1)
        log_params = torch.log(torch.clamp(params, min=1e-300))
        return (log_params - self.y_mean) / self.y_std


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).stem)
    return [int(part) if part.isdigit() else part for part in parts]


def dut_index(path):
    match = re.search(r"dut(\d+)\.s2p$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse DUT index from {path}")
    return int(match.group(1))


def parse_touchstone_variables(path):
    params = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                break
            if not line.startswith("!") or "=" not in line:
                continue
            key, value = line[1:].split("=", 1)
            match = re.search(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", value)
            if match:
                params[key.strip()] = float(match.group(1))
    return params


def make_key(values):
    return tuple(round(float(v), 6) for v in values)


def load_dataset(config):
    csv_df = pd.read_csv(config["csv"])
    csv_map = {}
    for _, row in csv_df.iterrows():
        key = make_key(row[name] for name in config["csv_feature_order"])
        csv_map[key] = {name: float(row[name]) for name in TARGET_PARAMS}

    rows = []
    s_rows = []
    for path in sorted(config["snp_dir"].glob("dut*.s2p"), key=natural_key):
        variables = parse_touchstone_variables(path)
        missing = [name for name in config["feature_order"] + [config["length_header"]] if name not in variables]
        if missing:
            continue
        key = make_key(variables[name] for name in config["feature_order"])
        if key not in csv_map:
            continue
        nw = rf.Network(str(path))
        rows.append(
            {
                "file": path.name,
                "dut_index": dut_index(path),
                "length_um": float(variables[config["length_header"]]),
                **{f"feature_{name}": float(variables[name]) for name in config["feature_order"]},
                **{f"target_{name}": csv_map[key][name] for name in TARGET_PARAMS},
            }
        )
        s_rows.append(nw.s)

    df = pd.DataFrame(rows).sort_values("dut_index").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No matched training samples for {DEVICE_NAME}")
    s_target = np.stack(s_rows, axis=0)[df.index.to_numpy()]
    return df, s_target


def split_by_dut(df):
    rng = np.random.default_rng(RANDOM_SEED)
    ids = df["dut_index"].to_numpy()
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * TRAIN_RATIO))))
    train_ids = set(shuffled[:n_train].tolist())
    train_mask = df["dut_index"].isin(train_ids).to_numpy()
    return train_mask, ~train_mask


def normalize(values, train_mask):
    mean = values[train_mask].mean(axis=0)
    std = np.maximum(values[train_mask].std(axis=0), 1e-12)
    return (values - mean) / std, mean, std


def prepare_arrays(df, s_target, config):
    feature_cols = [f"feature_{name}" for name in config["feature_order"]]
    target_cols = [f"target_{name}" for name in TARGET_PARAMS]
    x_raw = df[feature_cols].to_numpy(dtype=np.float64)
    y_raw = df[target_cols].to_numpy(dtype=np.float64)
    length_um = df["length_um"].to_numpy(dtype=np.float64)
    train_mask, val_mask = split_by_dut(df)

    x_norm, x_mean, x_std = normalize(x_raw, train_mask)
    y_log = np.log(np.maximum(y_raw, 1e-300))
    y_norm, y_mean, y_std = normalize(y_log, train_mask)
    metadata = {
        "device_name": DEVICE_NAME,
        "feature_columns": config["feature_order"],
        "target_params": TARGET_PARAMS,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_log_mean": y_mean.tolist(),
        "y_log_std": y_std.tolist(),
    }
    return x_norm, y_norm, y_raw, length_um, train_mask, val_mask, metadata


def abcd2s_torch(a, b, c, d):
    denom = a + b / Z_REF + c * Z_REF + d + 1e-30
    s = torch.zeros((*a.shape, 2, 2), dtype=COMPLEX_DTYPE, device=a.device)
    s[..., 0, 0] = (a + b / Z_REF - c * Z_REF - d) / denom
    s[..., 0, 1] = 2.0 * (a * d - b * c) / denom
    s[..., 1, 0] = 2.0 / denom
    s[..., 1, 1] = (-a + b / Z_REF - c * Z_REF + d) / denom
    return s


def circuit_params_to_s_torch(params, length_um, freqs_hz):
    r1, r2, r3 = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    l1, l2, l3 = params[:, 3:4] * 1e-9, params[:, 4:5] * 1e-9, params[:, 5:6] * 1e-9
    cox, csi, rsi = params[:, 6:7] * 1e-12, params[:, 7:8] * 1e-12, params[:, 8:9]
    omega = torch.tensor(2.0 * np.pi * freqs_hz, dtype=REAL_DTYPE, device=params.device)[None, :]
    length_m = (length_um[:, None] * 1e-6).to(params.device)

    r_rlgc = (r1**2 * r2 + r1 * r2**2 + omega**2 * r1 * l2**2) / (
        (r1 + r2) ** 2 + omega**2 * l2**2
    ) + (omega**2 * l3**2 * r3) / (r3**2 + omega**2 * l3**2)
    l_rlgc = (r1**2 * l2) / ((r1 + r2) ** 2 + omega**2 * l2**2) + l3 * r3**2 / (
        r3**2 + omega**2 * l3**2
    ) + l1
    g_rlgc = (omega**2 * rsi * cox**2) / (1.0 + omega**2 * rsi**2 * (cox + csi) ** 2)
    c_rlgc = (cox + omega**2 * csi * rsi**2 * cox * (cox + csi)) / (
        1.0 + omega**2 * rsi**2 * (cox + csi) ** 2
    )

    j = torch.complex(
        torch.tensor(0.0, dtype=REAL_DTYPE, device=params.device),
        torch.tensor(1.0, dtype=REAL_DTYPE, device=params.device),
    )
    z0 = torch.sqrt((r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE)) / (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE)))
    gamma = torch.sqrt(
        (r_rlgc.to(COMPLEX_DTYPE) + j * omega * l_rlgc.to(COMPLEX_DTYPE))
        * (g_rlgc.to(COMPLEX_DTYPE) + j * omega * c_rlgc.to(COMPLEX_DTYPE))
    )
    gl = gamma * length_m.to(COMPLEX_DTYPE)
    a = torch.cosh(gl)
    b = z0 * torch.sinh(gl)
    c = torch.sinh(gl) / z0
    d = torch.cosh(gl)
    return abcd2s_torch(a, b, c, d)


def s_loss(pred_s, target_s):
    return torch.mean((pred_s.real - target_s.real) ** 2 + (pred_s.imag - target_s.imag) ** 2)


def train_model(model, arrays, freqs_hz, device):
    x_norm, y_norm, _, length_um, train_mask, val_mask, metadata, s_target = arrays
    x_train = torch.tensor(x_norm[train_mask], dtype=REAL_DTYPE)
    y_train = torch.tensor(y_norm[train_mask], dtype=REAL_DTYPE)
    x_val = torch.tensor(x_norm[val_mask], dtype=REAL_DTYPE, device=device)
    y_val = torch.tensor(y_norm[val_mask], dtype=REAL_DTYPE, device=device)
    length_val = torch.tensor(length_um[val_mask], dtype=REAL_DTYPE, device=device)
    s_val = torch.tensor(s_target[val_mask], dtype=COMPLEX_DTYPE, device=device)

    y_mean_t = torch.tensor(metadata["y_log_mean"], dtype=REAL_DTYPE, device=device)
    y_std_t = torch.tensor(metadata["y_log_std"], dtype=REAL_DTYPE, device=device)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE_PARAM, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=60, factor=0.5)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, PARAM_PRETRAIN_EPOCHS + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
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
        history.append({"stage": "param_pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[param] epoch={epoch}, train={train_loss:.6e}, val={val_loss:.6e}", flush=True)
        if stale >= PARAM_PATIENCE:
            print(f"[param] 早停: epoch={epoch}, best_val={best_val:.6e}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    s_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_norm[train_mask], dtype=REAL_DTYPE),
            torch.tensor(y_norm[train_mask], dtype=REAL_DTYPE),
            torch.tensor(length_um[train_mask], dtype=REAL_DTYPE),
            torch.tensor(s_target[train_mask], dtype=COMPLEX_DTYPE),
        ),
        batch_size=BATCH_SIZE_SPARAM,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPARAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    for epoch in range(1, SPARAM_FINETUNE_EPOCHS + 1):
        model.train()
        train_s_sum = 0.0
        train_anchor_sum = 0.0
        n_seen = 0
        for xb, yb, lb, sb in s_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lb = lb.to(device)
            sb = sb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)
            log_params = pred_norm * y_std_t + y_mean_t
            params = torch.exp(torch.clamp(log_params, min=-40.0, max=40.0))
            pred_s = circuit_params_to_s_torch(params, lb, freqs_hz)
            loss_s = s_loss(pred_s, sb)
            loss_anchor = torch.mean((pred_norm - yb) ** 2)
            loss = loss_s + PARAM_ANCHOR_WEIGHT * loss_anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_s_sum += float(loss_s.detach().cpu()) * len(xb)
            train_anchor_sum += float(loss_anchor.detach().cpu()) * len(xb)
            n_seen += len(xb)
        train_s = train_s_sum / n_seen
        train_anchor = train_anchor_sum / n_seen
        model.eval()
        with torch.no_grad():
            val_norm = model(x_val)
            val_params = torch.exp(torch.clamp(val_norm * y_std_t + y_mean_t, min=-40.0, max=40.0))
            val_s_pred = circuit_params_to_s_torch(val_params, length_val, freqs_hz)
            val_s = s_loss(val_s_pred, s_val).item()
            val_anchor = torch.mean((val_norm - y_val) ** 2).item()
            val_loss = val_s + PARAM_ANCHOR_WEIGHT * val_anchor
        scheduler.step(val_loss)
        history.append(
            {
                "stage": "sparam_finetune",
                "epoch": epoch,
                "train_s_loss": train_s,
                "train_anchor_loss": train_anchor,
                "val_s_loss": val_s,
                "val_anchor_loss": val_anchor,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"[sparam] epoch={epoch}, train_s={train_s:.6e}, val_s={val_s:.6e}, anchor={val_anchor:.6e}", flush=True)
        if stale >= SPARAM_PATIENCE:
            print(f"[sparam] 早停: epoch={epoch}, best_val={best_val:.6e}", flush=True)
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


def load_mat4_parameters(features, config):
    out = {}
    x = features.reshape(1, -1)
    for name in TARGET_PARAMS:
        data = sio.loadmat(config["mat_dir"] / f"{config['mat_prefix']}{name}.mat")
        xmin = np.asarray(data["psmin"], dtype=float)
        xmax = np.asarray(data["psmax"], dtype=float)
        ymin = float(np.asarray(data["outputmin"]).squeeze())
        ymax = float(np.asarray(data["outputmax"]).squeeze())
        w1, b1 = np.asarray(data["w1"], dtype=float), np.asarray(data["theta1"], dtype=float)
        w2, b2 = np.asarray(data["w2"], dtype=float), np.asarray(data["theta2"], dtype=float)
        w3, b3 = np.asarray(data["w3"], dtype=float), np.asarray(data["theta3"], dtype=float)
        xn = 2.0 * (x - xmin) / (xmax - xmin + 1e-12) - 1.0
        y = np.tanh(xn @ w1 + b1)
        y = np.tanh(y @ w2 + b2)
        yn = y @ w3 + b3
        out[name] = float((ymin + (yn + 1.0) * (ymax - ymin) / 2.0).squeeze())
    return np.array([out[name] for name in TARGET_PARAMS], dtype=np.float64)


def circuit_params_to_s_np(params, length_um, freqs_hz):
    p = torch.tensor(params.reshape(1, -1), dtype=REAL_DTYPE)
    l = torch.tensor([length_um], dtype=REAL_DTYPE)
    with torch.no_grad():
        return circuit_params_to_s_torch(p, l, freqs_hz).numpy()[0]


def evaluate_and_save(model, df, arrays, freqs_hz, config, output_dir, device):
    x_norm, y_norm, y_raw, length_um, train_mask, val_mask, metadata, s_target = arrays
    y_mean = torch.tensor(metadata["y_log_mean"], dtype=REAL_DTYPE, device=device)
    y_std = torch.tensor(metadata["y_log_std"], dtype=REAL_DTYPE, device=device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(x_norm, dtype=REAL_DTYPE, device=device))
        pred_params = torch.exp(torch.clamp(pred_norm * y_std + y_mean, min=-40.0, max=40.0)).cpu().numpy()

    rows = []
    for i, row in df.iterrows():
        pred_s = circuit_params_to_s_np(pred_params[i], length_um[i], freqs_hz)
        mat4_params = load_mat4_parameters(
            row[[f"feature_{name}" for name in config["feature_order"]]].to_numpy(dtype=np.float64),
            config,
        )
        mat4_s = circuit_params_to_s_np(mat4_params, length_um[i], freqs_hz)
        target_s = s_target[i]
        rows.append(
            {
                "file": row["file"],
                "dut_index": int(row["dut_index"]),
                "split": "train" if train_mask[i] else "val",
                "python_sparam_mse": float(np.mean(np.abs(pred_s - target_s) ** 2)),
                "mat4_mse": float(np.mean(np.abs(mat4_s - target_s) ** 2)),
                "python_vs_mat4_mse": float(np.mean(np.abs(pred_s - mat4_s) ** 2)),
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "single_device_sparam_metrics.csv", index=False, encoding="utf-8-sig")

    pred_df = df[["file", "dut_index", "length_um"] + [f"feature_{name}" for name in config["feature_order"]]].copy()
    for idx, name in enumerate(TARGET_PARAMS):
        pred_df[f"target_{name}"] = y_raw[:, idx]
        pred_df[f"pred_{name}"] = pred_params[:, idx]
    pred_df.to_csv(output_dir / "single_device_param_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics_df, pred_df


def save_plots(metrics_df, df, pred_df, s_target, freqs_hz, config, output_dir):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    val_df = metrics_df[metrics_df["split"] == "val"].sort_values("python_sparam_mse", ascending=False)
    for _, metric_row in val_df.head(PLOT_VAL_LIMIT).iterrows():
        i = int(df.index[df["dut_index"] == metric_row["dut_index"]][0])
        pred_params = pred_df[pred_df["dut_index"] == metric_row["dut_index"]][[f"pred_{name}" for name in TARGET_PARAMS]].to_numpy(dtype=np.float64)[0]
        mat4_params = load_mat4_parameters(df.loc[i, [f"feature_{name}" for name in config["feature_order"]]].to_numpy(dtype=np.float64), config)
        pred_s = circuit_params_to_s_np(pred_params, df.loc[i, "length_um"], freqs_hz)
        mat4_s = circuit_params_to_s_np(mat4_params, df.loc[i, "length_um"], freqs_hz)
        target_s = s_target[i]
        freq_ghz = freqs_hz / 1e9
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        fig.suptitle(f"{DEVICE_NAME} {metric_row['file']} S-parameter target training ({metric_row['split']})", x=0.02, y=0.985, ha="left")
        fig.text(
            0.02,
            0.955,
            f"mat4 MSE={metric_row['mat4_mse']:.3e} | Python S-target MSE={metric_row['python_sparam_mse']:.3e}",
            ha="left",
            va="top",
            fontsize=9,
        )
        for ax, (m, n, label) in zip(axes.ravel(), [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]):
            ax.plot(freq_ghz, 20 * np.log10(np.maximum(np.abs(target_s[:, m, n]), 1e-30)), label="HFSS", color="black", linewidth=2.0)
            ax.plot(freq_ghz, 20 * np.log10(np.maximum(np.abs(mat4_s[:, m, n]), 1e-30)), label="mat4", color="#dc2626", linestyle="--")
            ax.plot(freq_ghz, 20 * np.log10(np.maximum(np.abs(pred_s[:, m, n]), 1e-30)), label="Python S-target", color="#16a34a", linestyle="-.")
            ax.set_title(label)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.9, wspace=0.2, hspace=0.3)
        fig.savefig(plot_dir / f"{Path(metric_row['file']).stem}_sparam_model_compare.png", dpi=150)
        plt.close(fig)
    return plot_dir


def run_device(device_name):
    global DEVICE_NAME
    DEVICE_NAME = device_name
    set_seed(RANDOM_SEED)
    config = DEVICE_CONFIGS[DEVICE_NAME]
    output_dir = config["output_dir"]
    if INITIALIZE_FROM_MAT4:
        output_dir = output_dir.with_name(output_dir.name + MAT4_INIT_OUTPUT_SUFFIX)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")

    print(f"设备: {device}")
    print(f"训练器件: {DEVICE_NAME}")
    print("建议: S 参数 target 训练使用 Python/PyTorch；MATLAB mat4 保留作为基线。")

    df, s_target = load_dataset(config)
    freqs_hz = rf.Network(str(next(iter(sorted(config["snp_dir"].glob("dut*.s2p"), key=natural_key))))).f
    arrays = (*prepare_arrays(df, s_target, config), s_target)
    _, _, _, _, train_mask, val_mask, metadata, _ = arrays
    print(f"匹配样本数: {len(df)}, 训练: {int(train_mask.sum())}, 验证: {int(val_mask.sum())}")

    if INITIALIZE_FROM_MAT4:
        model = Mat4InitializedDeviceNet(config, metadata).to(dtype=REAL_DTYPE, device=device)
        print("initial model: MATLAB mat4 weights")
    else:
        model = SingleDeviceParamNet(input_dim=len(config["feature_order"]), output_dim=len(TARGET_PARAMS)).to(dtype=REAL_DTYPE, device=device)
        print("initial model: random PyTorch weights")
    history_df = train_model(model, arrays, freqs_hz, device)

    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        output_dir / "single_device_sparam_net.pt",
    )
    history_df.to_csv(output_dir / "single_device_sparam_history.csv", index=False, encoding="utf-8-sig")
    metrics_df, pred_df = evaluate_and_save(model, df, arrays, freqs_hz, config, output_dir, device)
    plot_dir = save_plots(metrics_df, df, pred_df, s_target, freqs_hz, config, output_dir)

    val = metrics_df[metrics_df["split"] == "val"]
    report = {
        "device_name": DEVICE_NAME,
        "output_dir": str(output_dir),
        "n_samples": int(len(df)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "device": str(device),
        "initialize_from_mat4": bool(INITIALIZE_FROM_MAT4),
        "run_param_pretrain": bool(RUN_PARAM_PRETRAIN),
        "param_pretrain_epochs": int(PARAM_PRETRAIN_EPOCHS),
        "mat4_val_mean_mse": float(val["mat4_mse"].mean()),
        "python_sparam_val_mean_mse": float(val["python_sparam_mse"].mean()),
        "mat4_val_median_mse": float(val["mat4_mse"].median()),
        "python_sparam_val_median_mse": float(val["python_sparam_mse"].median()),
        "plot_dir": str(plot_dir),
    }
    with open(output_dir / "single_device_sparam_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n单器件 S 参数目标训练完成")
    print(f"模型文件: {output_dir / 'single_device_sparam_net.pt'}")
    print(f"指标 CSV: {output_dir / 'single_device_sparam_metrics.csv'}")
    print("验证集 MSE:")
    print(val[["mat4_mse", "python_sparam_mse", "python_vs_mat4_mse"]].mean().to_string())


def main():
    device_names = DEVICE_NAMES_TO_TRAIN or [DEVICE_NAME]
    for i, device_name in enumerate(device_names, start=1):
        print(f"\n===== {i}/{len(device_names)}: {device_name} =====")
        run_device(device_name)


if __name__ == "__main__":
    main()
