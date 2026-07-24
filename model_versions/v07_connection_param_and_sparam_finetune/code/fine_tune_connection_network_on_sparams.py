# -*- coding: utf-8 -*-
"""Fine-tune the connection-parameter NN directly on full-structure S-parameters.

Run this file directly in VS Code after ``train_connection_network_params.py``.
It loads the trained ``connection_param_net.pt`` as the initial model, predicts
all connection-network scale parameters from DUT geometry, cascades them with
the mat4 RDL/TSV device models, and continues training against the HFSS complex
S-parameters.
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skrf as rf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONNECTION_CODE_DIR = PROJECT_ROOT / "model_versions" / "v07_connection_param_and_sparam_finetune" / "code"
SPARAM_CODE_DIR = PROJECT_ROOT / "model_versions" / "v02_mat4_cascade_and_sparameter_optimization" / "code"
for path in [CONNECTION_CODE_DIR, SPARAM_CODE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Calc_SP_and_Opt2 as opt2
import train_connection_network_params as param_train


SOURCE_MODEL_DIR = PROJECT_ROOT / "model_versions" / "v07_connection_param_and_sparam_finetune" / "results" / "connection_network_sparam_finetune_optimized_with_cn3"
CHECKPOINT_PATH = SOURCE_MODEL_DIR / "connection_param_net_sparam_finetuned.pt"
OUTPUT_DIR = (
    PROJECT_ROOT / "model_versions" / "v07_connection_param_and_sparam_finetune" / "results" / "connection_network_sparam_finetune_optimized_with_cn3_with_device_scales"
)

# Configure these values before running directly from VS Code.
TARGET_VARIANT = "optimized_with_cn3"
RANDOM_SEED = 20260628
BATCH_SIZE = 16
EPOCHS = 500
LEARNING_RATE = 5e-7
WEIGHT_DECAY = 1e-8
PATIENCE = 120
PRINT_EVERY = 10
USE_CUDA_IF_AVAILABLE = True
PARAMETER_ANCHOR_WEIGHT = 0.0  # 0 means S-parameter loss only.
SCALE_CENTER = 0.95
SCALE_HALF_RANGE = 0.10  # scale = 0.95 +/- 0.10, constrained by tanh.
SCALE_REG_WEIGHT = 1e-4
PLOT_SPLIT = "val"  # "val", "train", or "all"
PLOT_DUT_LIMIT = 80
FILE_READ_RETRIES = 5
FILE_READ_RETRY_SECONDS = 2.0

Z_REF = 50.0
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128


DEVICE_SEQUENCE = [
    "RDL_Top",
    "TSV",
    "RDL_Bottom",
    "TSV",
    "RDL_Top",
    "TSV",
    "RDL_Bottom",
    "TSV",
    "RDL_Top",
]


class SParamDataset(Dataset):
    def __init__(self, indices, x_norm, y_norm, base_abcds, segment_rlgc, segment_lengths_m, target_s, dut_indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.base_abcds = base_abcds
        self.segment_rlgc = segment_rlgc
        self.segment_lengths_m = segment_lengths_m
        self.target_s = target_s
        self.dut_indices = dut_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        return (
            self.dut_indices[idx],
            self.x_norm[idx],
            self.y_norm[idx],
            self.base_abcds[idx],
            self.segment_rlgc[idx],
            self.segment_lengths_m[idx],
            self.target_s[idx],
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def abcd2s_torch_batched(abcd):
    a = abcd[:, :, 0, 0]
    b = abcd[:, :, 0, 1]
    c = abcd[:, :, 1, 0]
    d = abcd[:, :, 1, 1]

    denom = a + b / Z_REF + c * Z_REF + d + 1e-30
    s = torch.zeros_like(abcd)
    s[:, :, 0, 0] = (a + b / Z_REF - c * Z_REF - d) / denom
    s[:, :, 0, 1] = 2.0 * (a * d - b * c) / denom
    s[:, :, 1, 0] = 2.0 / denom
    s[:, :, 1, 1] = (-a + b / Z_REF - c * Z_REF + d) / denom
    return s


def rlgc_to_abcd_torch_batched(segment_rlgc, segment_lengths_m, segment_scales, omega):
    r = segment_rlgc[..., 0].to(COMPLEX_DTYPE)
    l = segment_rlgc[..., 1].to(COMPLEX_DTYPE)
    g = segment_rlgc[..., 2].to(COMPLEX_DTYPE)
    c = segment_rlgc[..., 3].to(COMPLEX_DTYPE)

    omega_b = omega[None, None, :].to(COMPLEX_DTYPE)
    length = (segment_lengths_m * segment_scales).to(COMPLEX_DTYPE)[:, :, None]
    j = torch.complex(
        torch.tensor(0.0, dtype=REAL_DTYPE, device=segment_rlgc.device),
        torch.tensor(1.0, dtype=REAL_DTYPE, device=segment_rlgc.device),
    )

    z0 = torch.sqrt((r + j * omega_b * l) / (g + j * omega_b * c))
    gamma = torch.sqrt((r + j * omega_b * l) * (g + j * omega_b * c))
    gl = gamma * length

    abcd = torch.zeros((*segment_rlgc.shape[:3], 2, 2), dtype=COMPLEX_DTYPE, device=segment_rlgc.device)
    abcd[..., 0, 0] = torch.cosh(gl)
    abcd[..., 0, 1] = z0 * torch.sinh(gl)
    abcd[..., 1, 0] = (1.0 / z0) * torch.sinh(gl)
    abcd[..., 1, 1] = torch.cosh(gl)
    return abcd


def correction_abcd_torch_batched(p_all, omega):
    batch_size = p_all.shape[0]
    n_conn = p_all.shape[1]
    n_freq = omega.numel()

    p = p_all[:, :, None, :]
    omega_b = omega[None, None, :]

    cn1 = p[..., 0] * 1e-14
    rn1 = p[..., 1] * 1e3
    cn2 = p[..., 2] * 1e-14
    rn2 = p[..., 3] * 1e3
    cn3 = p[..., 4] * 1e-14
    rn3 = p[..., 5] * 1.0
    ln1 = p[..., 6] * 1e-11

    j = torch.complex(torch.tensor(0.0, dtype=REAL_DTYPE, device=p_all.device), torch.tensor(1.0, dtype=REAL_DTYPE, device=p_all.device))
    y1 = j * omega_b * cn1.to(COMPLEX_DTYPE) + 1.0 / rn1.to(COMPLEX_DTYPE)
    y2 = j * omega_b * cn2.to(COMPLEX_DTYPE) + 1.0 / rn2.to(COMPLEX_DTYPE)
    y3 = j * omega_b * cn3.to(COMPLEX_DTYPE) + 1.0 / (rn3.to(COMPLEX_DTYPE) + j * omega_b * ln1.to(COMPLEX_DTYPE))

    abcd = torch.zeros((batch_size, n_conn, n_freq, 2, 2), dtype=COMPLEX_DTYPE, device=p_all.device)
    one = torch.ones_like(y1)
    abcd[:, :, :, 0, 0] = one + y2 / y3
    abcd[:, :, :, 0, 1] = 1.0 / y3
    abcd[:, :, :, 1, 0] = y1 + y2 + y1 * y2 / y3
    abcd[:, :, :, 1, 1] = one + y1 / y3
    return abcd


def cascade_with_corrections_torch(base_abcds, p_all, omega):
    corrections = correction_abcd_torch_batched(p_all, omega)
    result = base_abcds[:, 0]
    for i in range(param_train.CONNECTION_COUNT):
        result = torch.matmul(torch.matmul(result, corrections[:, i]), base_abcds[:, i + 1])
    return abcd2s_torch_batched(result)


def cascade_with_scales_and_corrections_torch(segment_rlgc, segment_lengths_m, segment_scales, p_all, omega):
    base_abcds = rlgc_to_abcd_torch_batched(segment_rlgc, segment_lengths_m, segment_scales, omega)
    corrections = correction_abcd_torch_batched(p_all, omega)
    result = base_abcds[:, 0]
    for i in range(param_train.CONNECTION_COUNT):
        result = torch.matmul(torch.matmul(result, corrections[:, i]), base_abcds[:, i + 1])
    return abcd2s_torch_batched(result)


class ConnectionParamAndScaleModel(nn.Module):
    def __init__(self, param_model, input_dim):
        super().__init__()
        self.param_model = param_model
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, len(DEVICE_SEQUENCE)),
        )
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)

    def forward(self, x):
        param_norm = self.param_model(x)
        scale_raw = self.scale_net(x)
        scales = SCALE_CENTER + SCALE_HALF_RANGE * torch.tanh(scale_raw)
        return param_norm, scales


def sparam_loss(pred_s, target_s):
    return torch.mean((pred_s.real - target_s.real) ** 2 + (pred_s.imag - target_s.imag) ** 2)


def load_initialized_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"未找到初始模型: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    hidden_layers = checkpoint.get("hidden_layers", param_train.HIDDEN_LAYERS)
    metadata = checkpoint["metadata"]

    if metadata.get("target_variant") != TARGET_VARIANT:
        raise ValueError(f"初始模型 variant={metadata.get('target_variant')}，当前 TARGET_VARIANT={TARGET_VARIANT}")

    param_model = param_train.DutConnectionParamNet(
        input_dim=len(metadata["feature_columns"]),
        output_dim=len(metadata["target_columns"]),
        hidden_layers=hidden_layers,
    ).to(dtype=REAL_DTYPE, device=device)
    param_model.load_state_dict(checkpoint["model_state_dict"])
    model = ConnectionParamAndScaleModel(param_model, input_dim=len(metadata["feature_columns"])).to(
        dtype=REAL_DTYPE, device=device
    )
    return model, metadata, hidden_layers


def load_hfss_network_with_retry(path):
    last_exc = None
    for attempt in range(1, FILE_READ_RETRIES + 1):
        try:
            nw = rf.Network(str(path))
            params = opt2.extract_device_params_RDL_TSV(path)
            return nw, params
        except PermissionError as exc:
            last_exc = exc
            if attempt == FILE_READ_RETRIES:
                break
            print(f"[文件占用] {path.name}: {exc}; {FILE_READ_RETRY_SECONDS:.1f}s 后重试 {attempt}/{FILE_READ_RETRIES}", flush=True)
            time.sleep(FILE_READ_RETRY_SECONDS)
    raise last_exc


def circuit_params_to_rlgc(circuit_params, freqs):
    r1, r2, r3 = circuit_params["R1"], circuit_params["R2"], circuit_params["R3"]
    l1 = circuit_params["L1"] * 1e-9
    l2 = circuit_params["L2"] * 1e-9
    l3 = circuit_params["L3"] * 1e-9
    cox = circuit_params["Cox"] * 1e-12
    csi = circuit_params["Csi"] * 1e-12
    rsi = circuit_params["Rsi"]
    omega = 2.0 * np.pi * freqs

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
    return np.stack([r_rlgc, l_rlgc, g_rlgc, c_rlgc], axis=-1)


def build_segment_rlgc_and_lengths(params, freqs):
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    features_top = np.array([params["lrdl"], params["wrdl"], params["trdl"], params["htsv"], params["p1"]])
    features_bot = np.array([params["ldown"], params["wdown"], params["tdown"], params["htsv"], params["p1"]])
    features_tsv = np.array([params["dtsv"], params["htsv"], params["p1"]])

    cp_top = opt2.predict_circuit_parameters(features_top, opt2.MAT_DIR, target_params, prefix="RDL_Top_")
    cp_bot = opt2.predict_circuit_parameters(features_bot, opt2.MAT_DIR, target_params, prefix="RDL_Bottom_")
    cp_tsv = opt2.predict_circuit_parameters(features_tsv, opt2.MAT_DIR, target_params, prefix="TSV_")

    rlgc_top = circuit_params_to_rlgc(cp_top, freqs)
    rlgc_bot = circuit_params_to_rlgc(cp_bot, freqs)
    rlgc_tsv = circuit_params_to_rlgc(cp_tsv, freqs)

    segment_map = {
        "RDL_Top": (rlgc_top, params["lrdl"] * 1e-6),
        "RDL_Bottom": (rlgc_bot, params["ldown"] * 1e-6),
        "TSV": (rlgc_tsv, params["htsv"] * 1e-6),
    }
    segment_rlgc = []
    segment_lengths = []
    for name in DEVICE_SEQUENCE:
        rlgc, length_m = segment_map[name]
        segment_rlgc.append(rlgc)
        segment_lengths.append(length_m)
    return np.stack(segment_rlgc, axis=0), np.asarray(segment_lengths, dtype=np.float64)


def build_training_arrays():
    connection_df = param_train.load_connection_dataframe()
    dut_df = param_train.build_dut_dataframe(connection_df)
    x_norm, y_norm, y_raw, train_mask, val_mask, metadata = param_train.build_matrices(dut_df)

    base_rows = []
    segment_rlgc_rows = []
    segment_length_rows = []
    target_rows = []
    for n_done, row in enumerate(dut_df.sort_values("dut_index").itertuples(index=False), start=1):
        dut_idx = int(row.dut_index)
        hfss_path = opt2.S2P_DIR / f"dut{dut_idx}.s2p"
        hfss_nw, params = load_hfss_network_with_retry(hfss_path)
        base_rows.append(np.stack(opt2.build_base_abcds(params, hfss_nw.f), axis=0))
        segment_rlgc, segment_lengths = build_segment_rlgc_and_lengths(params, hfss_nw.f)
        segment_rlgc_rows.append(segment_rlgc)
        segment_length_rows.append(segment_lengths)
        target_rows.append(hfss_nw.s)
        if n_done == 1 or n_done % 200 == 0:
            print(f"预计算级联数据 {n_done}/{len(dut_df)}", flush=True)

    base_abcds = np.stack(base_rows, axis=0)
    segment_rlgc = np.stack(segment_rlgc_rows, axis=0)
    segment_lengths = np.stack(segment_length_rows, axis=0)
    target_s = np.stack(target_rows, axis=0)
    dut_indices = dut_df.sort_values("dut_index")["dut_index"].to_numpy(dtype=np.int64)

    order = np.argsort(dut_df["dut_index"].to_numpy(dtype=np.int64))
    x_norm = x_norm[order]
    y_norm = y_norm[order]
    y_raw = y_raw[order]
    train_mask = train_mask[order]
    val_mask = val_mask[order]
    dut_df = dut_df.iloc[order].reset_index(drop=True)

    return (
        dut_df,
        x_norm,
        y_norm,
        y_raw,
        train_mask,
        val_mask,
        metadata,
        base_abcds,
        segment_rlgc,
        segment_lengths,
        target_s,
        dut_indices,
    )


def train_sparam_finetune(model, arrays, device):
    (
        dut_df,
        x_norm,
        y_norm,
        _,
        train_mask,
        val_mask,
        metadata,
        base_abcds,
        segment_rlgc,
        segment_lengths,
        target_s,
        dut_indices,
    ) = arrays
    omega = 2.0 * np.pi * rf.Network(str(opt2.S2P_DIR / f"dut{int(dut_indices[0])}.s2p")).f

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    train_ds = SParamDataset(train_indices, x_norm, y_norm, base_abcds, segment_rlgc, segment_lengths, target_s, dut_indices)
    val_ds = SParamDataset(val_indices, x_norm, y_norm, base_abcds, segment_rlgc, segment_lengths, target_s, dut_indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    y_mean_t = torch.tensor(metadata["y_mean"], dtype=REAL_DTYPE, device=device)
    y_std_t = torch.tensor(metadata["y_std"], dtype=REAL_DTYPE, device=device)
    omega_t = torch.tensor(omega, dtype=REAL_DTYPE, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    history = []

    def run_loader(loader, training):
        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_s_loss = 0.0
        total_anchor = 0.0
        n_seen = 0

        for _, x_b, y_b, _, segment_rlgc_b, segment_lengths_b, target_b in loader:
            x_b = x_b.to(device=device, dtype=REAL_DTYPE)
            y_b = y_b.to(device=device, dtype=REAL_DTYPE)
            segment_rlgc_b = segment_rlgc_b.to(device=device, dtype=REAL_DTYPE)
            segment_lengths_b = segment_lengths_b.to(device=device, dtype=REAL_DTYPE)
            target_b = target_b.to(device=device, dtype=COMPLEX_DTYPE)

            with torch.set_grad_enabled(training):
                y_pred_norm, segment_scales = model(x_b)
                p_all = (y_pred_norm * y_std_t + y_mean_t).reshape(-1, param_train.CONNECTION_COUNT, len(param_train.SCALE_COLUMNS))
                pred_s = cascade_with_scales_and_corrections_torch(
                    segment_rlgc_b,
                    segment_lengths_b,
                    segment_scales,
                    p_all,
                    omega_t,
                )
                loss_s = sparam_loss(pred_s, target_b)
                loss_anchor = torch.mean((y_pred_norm - y_b) ** 2)
                loss_scale_reg = torch.mean((segment_scales - SCALE_CENTER) ** 2)
                loss = loss_s + PARAMETER_ANCHOR_WEIGHT * loss_anchor + SCALE_REG_WEIGHT * loss_scale_reg

                if not torch.isfinite(loss):
                    raise FloatingPointError("S 参数微调出现 NaN/Inf")

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()

            batch_n = len(x_b)
            total_loss += float(loss.detach().cpu()) * batch_n
            total_s_loss += float(loss_s.detach().cpu()) * batch_n
            total_anchor += float(loss_anchor.detach().cpu()) * batch_n
            n_seen += batch_n

        return total_loss / n_seen, total_s_loss / n_seen, total_anchor / n_seen

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_s, train_anchor = run_loader(train_loader, training=True)
        with torch.no_grad():
            val_loss, val_s, val_anchor = run_loader(val_loader, training=False)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_s_loss": train_s,
                "train_anchor_loss": train_anchor,
                "val_loss": val_loss,
                "val_s_loss": val_s,
                "val_anchor_loss": val_anchor,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(
                f"epoch={epoch}, train_s={train_s:.6e}, val_s={val_s:.6e}, "
                f"anchor_val={val_anchor:.6e}, lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if stale >= PATIENCE:
            print(f"早停: epoch={epoch}, best_val_s_loss={best_val:.6e}", flush=True)
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_all(model, arrays, device):
    (
        dut_df,
        x_norm,
        y_norm,
        y_raw,
        train_mask,
        val_mask,
        metadata,
        base_abcds,
        segment_rlgc,
        segment_lengths,
        target_s,
        dut_indices,
    ) = arrays
    omega = 2.0 * np.pi * rf.Network(str(opt2.S2P_DIR / f"dut{int(dut_indices[0])}.s2p")).f
    y_mean_t = torch.tensor(metadata["y_mean"], dtype=REAL_DTYPE, device=device)
    y_std_t = torch.tensor(metadata["y_std"], dtype=REAL_DTYPE, device=device)
    omega_t = torch.tensor(omega, dtype=REAL_DTYPE, device=device)

    rows = []
    pred_params = []
    pred_scales = []
    model.eval()
    for start in range(0, len(dut_df), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(dut_df))
        x_b = torch.tensor(x_norm[start:stop], dtype=REAL_DTYPE, device=device)
        segment_rlgc_b = torch.tensor(segment_rlgc[start:stop], dtype=REAL_DTYPE, device=device)
        segment_lengths_b = torch.tensor(segment_lengths[start:stop], dtype=REAL_DTYPE, device=device)
        target_b = torch.tensor(target_s[start:stop], dtype=COMPLEX_DTYPE, device=device)
        with torch.no_grad():
            y_pred_norm, segment_scales = model(x_b)
            p_all = (y_pred_norm * y_std_t + y_mean_t).reshape(-1, param_train.CONNECTION_COUNT, len(param_train.SCALE_COLUMNS))
            pred_s = cascade_with_scales_and_corrections_torch(
                segment_rlgc_b,
                segment_lengths_b,
                segment_scales,
                p_all,
                omega_t,
            )
        pred_params.append(p_all.detach().cpu().numpy().reshape(stop - start, -1))
        pred_scales.append(segment_scales.detach().cpu().numpy())
        pred_s_np = pred_s.detach().cpu().numpy()
        for local_i, dut_idx in enumerate(dut_indices[start:stop]):
            hfss_path = opt2.S2P_DIR / f"dut{int(dut_idx)}.s2p"
            optimized_path = opt2.OUTPUT_DIR / TARGET_VARIANT / f"dut{int(dut_idx)}.s2p"
            optimized_s = rf.Network(str(optimized_path)).s
            direct_s = opt2.abcd2s(opt2.cascade_direct(list(base_abcds[start + local_i])))
            rows.append(
                {
                    "dut_index": int(dut_idx),
                    "split": "train" if train_mask[start + local_i] else "val",
                    "direct_mse_vs_hfss": opt2.mse(target_b[local_i].detach().cpu().numpy(), direct_s),
                    "optimized_mse_vs_hfss": opt2.mse(target_b[local_i].detach().cpu().numpy(), optimized_s),
                    "sparam_finetune_mse_vs_hfss": opt2.mse(target_b[local_i].detach().cpu().numpy(), pred_s_np[local_i]),
                    "sparam_finetune_mse_vs_optimized": opt2.mse(optimized_s, pred_s_np[local_i]),
                }
            )

    pred_param_df = dut_df[["file", "dut_index", "variant"] + param_train.STRUCTURE_COLUMNS].copy()
    pred_param_arr = np.vstack(pred_params)
    pred_scale_arr = np.vstack(pred_scales)
    for i, col in enumerate(param_train.TARGET_COLUMNS):
        pred_param_df[f"target_{col}"] = y_raw[:, i]
        pred_param_df[f"pred_{col}"] = pred_param_arr[:, i]
    for i, name in enumerate(DEVICE_SEQUENCE, start=1):
        pred_param_df[f"pred_device{i}_{name}_length_scale"] = pred_scale_arr[:, i - 1]
    return pd.DataFrame(rows), pred_param_df


def db20(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def save_plots(metrics_df, pred_param_df, arrays):
    _, _, _, _, train_mask, _, _, base_abcds, segment_rlgc, segment_lengths, _, dut_indices = arrays
    plot_df = metrics_df if PLOT_SPLIT == "all" else metrics_df[metrics_df["split"] == PLOT_SPLIT]
    plot_dir = OUTPUT_DIR / "sparam_finetune_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_plotted = 0
    for _, metric_row in plot_df.sort_values("dut_index").iterrows():
        if PLOT_DUT_LIMIT is not None and n_plotted >= PLOT_DUT_LIMIT:
            break
        dut_idx = int(metric_row["dut_index"])
        array_idx = int(np.where(dut_indices == dut_idx)[0][0])
        hfss_nw = rf.Network(str(opt2.S2P_DIR / f"dut{dut_idx}.s2p"))
        optimized_nw = rf.Network(str(opt2.OUTPUT_DIR / TARGET_VARIANT / f"dut{dut_idx}.s2p"))
        direct_s = opt2.abcd2s(opt2.cascade_direct(list(base_abcds[array_idx])))

        p_row = pred_param_df[pred_param_df["dut_index"] == dut_idx].iloc[0]
        p_all = []
        for conn_idx in range(1, param_train.CONNECTION_COUNT + 1):
            for name in param_train.SCALE_COLUMNS:
                p_all.append(p_row[f"pred_conn{conn_idx}_{name}"])
        scale_values = [
            p_row[f"pred_device{i}_{name}_length_scale"]
            for i, name in enumerate(DEVICE_SEQUENCE, start=1)
        ]
        with torch.no_grad():
            pred_s = (
                cascade_with_scales_and_corrections_torch(
                    torch.tensor(segment_rlgc[array_idx : array_idx + 1], dtype=REAL_DTYPE),
                    torch.tensor(segment_lengths[array_idx : array_idx + 1], dtype=REAL_DTYPE),
                    torch.tensor([scale_values], dtype=REAL_DTYPE),
                    torch.tensor(np.array(p_all, dtype=np.float64).reshape(1, param_train.CONNECTION_COUNT, -1), dtype=REAL_DTYPE),
                    torch.tensor(2.0 * np.pi * hfss_nw.f, dtype=REAL_DTYPE),
                )
                .detach()
                .cpu()
                .numpy()[0]
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        fig.suptitle(f"dut{dut_idx}.s2p S-parameter fine-tuned NN ({metric_row['split']})", x=0.02, y=0.985, ha="left")
        fig.text(
            0.02,
            0.955,
            (
                f"Direct={metric_row['direct_mse_vs_hfss']:.3e} | "
                f"Optimized={metric_row['optimized_mse_vs_hfss']:.3e} | "
                f"S-finetune NN={metric_row['sparam_finetune_mse_vs_hfss']:.3e}"
            ),
            ha="left",
            va="top",
            fontsize=9,
            color="#475569",
        )
        freq_ghz = hfss_nw.f / 1e9
        for ax, (m, n, label) in zip(axes.ravel(), [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]):
            ax.plot(freq_ghz, db20(hfss_nw.s[:, m, n]), label="HFSS", color="black", linewidth=2.0)
            ax.plot(freq_ghz, db20(direct_s[:, m, n]), label="Direct", color="#64748b", linestyle=":", linewidth=1.6)
            ax.plot(freq_ghz, db20(optimized_nw.s[:, m, n]), label="Optimized", color="#dc2626", linestyle="--", linewidth=1.6)
            ax.plot(freq_ghz, db20(pred_s[:, m, n]), label="S-finetuned NN", color="#16a34a", linestyle="-.", linewidth=1.6)
            ax.set_title(f"{label} magnitude")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.9, wspace=0.2, hspace=0.3)
        fig.savefig(plot_dir / f"dut{dut_idx}_{metric_row['split']}_sparam_finetune.png", dpi=150)
        plt.close(fig)
        n_plotted += 1

    return plot_dir, n_plotted


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)
    print(f"初始模型: {CHECKPOINT_PATH}", flush=True)

    model, metadata, hidden_layers = load_initialized_model(device)
    arrays = build_training_arrays()
    dut_df, _, _, _, train_mask, val_mask, _, _, _, _, _, _ = arrays
    print(f"DUT 样本数: {len(dut_df)}", flush=True)
    print(f"训练 DUT 数: {int(train_mask.sum())}, 验证 DUT 数: {int(val_mask.sum())}", flush=True)
    print(f"输入维度: {len(metadata['feature_columns'])}, 输出维度: {len(metadata['target_columns'])}", flush=True)

    model, history_df = train_sparam_finetune(model, arrays, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "hidden_layers": hidden_layers,
            "source_checkpoint": str(CHECKPOINT_PATH),
            "sparam_finetune": True,
            "parameter_anchor_weight": PARAMETER_ANCHOR_WEIGHT,
            "learn_device_length_scales": True,
            "device_sequence": DEVICE_SEQUENCE,
            "scale_center": SCALE_CENTER,
            "scale_half_range": SCALE_HALF_RANGE,
            "scale_reg_weight": SCALE_REG_WEIGHT,
        },
        OUTPUT_DIR / "connection_param_net_sparam_finetuned.pt",
    )
    history_df.to_csv(OUTPUT_DIR / "sparam_finetune_history.csv", index=False, encoding="utf-8-sig")

    metrics_df, pred_param_df = predict_all(model, arrays, device)
    metrics_df.to_csv(OUTPUT_DIR / "sparam_finetune_metrics.csv", index=False, encoding="utf-8-sig")
    pred_param_df.to_csv(OUTPUT_DIR / "sparam_finetune_param_predictions.csv", index=False, encoding="utf-8-sig")
    plot_dir, n_plots = save_plots(metrics_df, pred_param_df, arrays)

    report = {
        "source_checkpoint": str(CHECKPOINT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "target_variant": TARGET_VARIANT,
        "n_dut": int(len(dut_df)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "device": str(device),
        "final_epoch": int(history_df["epoch"].iloc[-1]) if len(history_df) else 0,
        "best_val_s_loss": float(history_df["val_s_loss"].min()) if len(history_df) else None,
        "parameter_anchor_weight": PARAMETER_ANCHOR_WEIGHT,
        "learn_device_length_scales": True,
        "device_sequence": DEVICE_SEQUENCE,
        "scale_center": SCALE_CENTER,
        "scale_half_range": SCALE_HALF_RANGE,
        "scale_reg_weight": SCALE_REG_WEIGHT,
        "n_plots": int(n_plots),
        "plot_dir": str(plot_dir),
    }
    with open(OUTPUT_DIR / "sparam_finetune_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    val_metrics = metrics_df[metrics_df["split"] == "val"]
    print("\nS 参数微调完成", flush=True)
    print(f"模型文件: {OUTPUT_DIR / 'connection_param_net_sparam_finetuned.pt'}", flush=True)
    print(f"指标 CSV: {OUTPUT_DIR / 'sparam_finetune_metrics.csv'}", flush=True)
    print(f"对比图目录: {plot_dir}", flush=True)
    print("验证集平均 MSE:", flush=True)
    print(
        val_metrics[
            [
                "direct_mse_vs_hfss",
                "optimized_mse_vs_hfss",
                "sparam_finetune_mse_vs_hfss",
                "sparam_finetune_mse_vs_optimized",
            ]
        ]
        .mean()
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
