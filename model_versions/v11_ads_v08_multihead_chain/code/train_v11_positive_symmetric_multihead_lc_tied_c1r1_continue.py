# -*- coding: utf-8 -*-
"""Continue positive symmetric multi-head NN with tied Cn1/Cn2 and Rn1/Rn2.

Run this file directly in VS Code. No command-line arguments are required.

The 13-device cascade is left-right symmetric. This model learns only the first
six connection-network heads and mirrors them to the last six positions:

    1,2,3,4,5,6,6,5,4,3,2,1

This variant starts from the ultra-tight hardcap checkpoint, hard-caps the NN
circuit output before cascade evaluation, and then enforces the simplified
connection-network topology:

    Cn1/Cn2/Cn3 <= 100, equivalent to C <= 1e-12 F
    Ln1_scale <= 1.0, equivalent to Ln1 <= 1e-11 H
    Cn2 = Cn1
    Rn2 = Rn1

The S-parameter objective includes real/imag loss plus a wrapped phase loss on
S11 and S21. The existing same-sample C/L and f0 anchors are kept.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


THIS_DIR = Path(__file__).resolve().parent
BASE_TRAIN_SCRIPT = THIS_DIR / "train_v11_positive_multihead_sparam_from_shared.py"

RUN_LABEL = "v11_positive_symmetric_multihead_lc_tied_c1r1_continue_log_adslen09"
INIT_RUN_LABEL = "v11_positive_symmetric_multihead_lc_ultratight_hardcap_continue_log_adslen09"
INIT_CHECKPOINT_NAME = "positive_symmetric_multihead_lc_sample_anchor.pt"
UNIQUE_HEAD_COUNT = 6
C_LIMIT_F = 1e-12
L_LIMIT_H = 1e-11
CN_SCALE_HARD_CAP = C_LIMIT_F / 1e-14
LN1_SCALE_HARD_CAP = L_LIMIT_H / 1e-11
LC_FMIN_HZ = 60e9
C_PENALTY_WEIGHT = 1.0
L_PENALTY_WEIGHT = 0.35
LC_FMIN_PENALTY_WEIGHT = 0.8
R_UPPER_SCALE = 5e4
R_PENALTY_WEIGHT = 0.03
REFERENCE_PENALTY_WEIGHT = 0.04
HEAD_SMOOTHNESS_WEIGHT = 0.02
PHASE_LOSS_WEIGHT = 0.12
SAMPLE_CL_ANCHOR_WEIGHT = 0.035
SAMPLE_F0_ANCHOR_WEIGHT = 0.06
REFERENCE_LOG_TOL = 0.45
HEAD_LOG_STD_TOL = 0.35
SAMPLE_CL_LOG_TOL = 0.22
SAMPLE_F0_LOG_TOL = 0.12
JOINT_EPOCHS = 180
JOINT_PATIENCE = 35
JOINT_LR = 1e-5
BATCH_SIZE = 8
PRINT_EVERY = 10
TIE_C1_C2_R1_R2 = True


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            vals.append(f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


class SymmetricMultiHeadV08ConnectionNet(nn.Module):
    """Seven parameter networks with six heads mirrored to 12 connections."""

    def __init__(self, wrapper, input_dim: int):
        super().__init__()
        self.wrapper = wrapper
        self.element_nets = nn.ModuleDict(
            {
                name: nn.ModuleDict(
                    {
                        "trunk": nn.Sequential(
                            nn.Linear(input_dim, 30),
                            nn.Tanh(),
                            nn.Linear(30, 30),
                            nn.Tanh(),
                        ),
                        "heads": nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(30, 20),
                                    nn.Tanh(),
                                    nn.Linear(20, 1),
                                )
                                for _ in range(UNIQUE_HEAD_COUNT)
                            ]
                        ),
                    }
                )
                for name in wrapper.V08_PARAM_NAMES
            }
        )

    def initialize_from_shared(self, shared_model) -> None:
        for name in self.wrapper.V08_PARAM_NAMES:
            src = shared_model.param_nets[name]
            dst = self.element_nets[name]
            dst["trunk"][0].load_state_dict(src[0].state_dict())
            dst["trunk"][2].load_state_dict(src[2].state_dict())
            for head in dst["heads"]:
                head[0].load_state_dict(src[4].state_dict())
                head[2].load_state_dict(src[6].state_dict())

    def forward(self, x):
        unique_outputs = [[] for _ in range(UNIQUE_HEAD_COUNT)]
        for name in self.wrapper.V08_PARAM_NAMES:
            z = self.element_nets[name]["trunk"](x)
            for head_idx, head in enumerate(self.element_nets[name]["heads"]):
                unique_outputs[head_idx].append(head(z))
        unique_flat = [torch.cat(values, dim=1) for values in unique_outputs]
        mirrored = unique_flat + list(reversed(unique_flat))
        return torch.cat(mirrored, dim=1)


def load_symmetric_initialized_model(base, wrapper, shared_ckpt_path: Path, device):
    shared_model = wrapper.SharedV08ParamNet(input_dim=len(base.STRUCTURE_COLUMNS)).to(dtype=base.REAL_DTYPE, device=device)
    checkpoint = torch.load(shared_ckpt_path, map_location=device)
    shared_model.load_state_dict(checkpoint["model_state_dict"])
    model = SymmetricMultiHeadV08ConnectionNet(wrapper, input_dim=len(base.STRUCTURE_COLUMNS)).to(dtype=base.REAL_DTYPE, device=device)
    model.initialize_from_shared(shared_model)
    return shared_model, model, checkpoint


def lc_limit_loss(base, wrapper, p_all):
    scales = torch.tensor(wrapper.V08_SCALE_FACTORS, dtype=base.REAL_DTYPE, device=p_all.device)
    physical = p_all * scales
    c_vals = physical[..., [0, 2, 4]]
    l_vals = physical[..., [6]]
    cn3_vals = physical[..., 4]
    ln1_vals = physical[..., 6]
    r_vals = p_all[..., [1, 3]]
    c_penalty = torch.mean(torch.relu(c_vals / C_LIMIT_F - 1.0) ** 2)
    l_penalty = torch.mean(torch.relu(l_vals / L_LIMIT_H - 1.0) ** 2)
    lc_f0 = 1.0 / (2.0 * torch.pi * torch.sqrt(torch.clamp(cn3_vals * ln1_vals, min=1e-60)))
    fmin_penalty = torch.mean(torch.relu(LC_FMIN_HZ / lc_f0 - 1.0) ** 2)
    r_penalty = torch.mean(torch.relu(r_vals / R_UPPER_SCALE - 1.0) ** 2)
    total = (
        C_PENALTY_WEIGHT * c_penalty
        + L_PENALTY_WEIGHT * l_penalty
        + LC_FMIN_PENALTY_WEIGHT * fmin_penalty
        + R_PENALTY_WEIGHT * r_penalty
    )
    return total, c_penalty, l_penalty, fmin_penalty, r_penalty


def apply_tied_c1r1_constraint(p_all):
    """Return parameters with Cn2=Cn1 and Rn2=Rn1."""
    if not TIE_C1_C2_R1_R2:
        return p_all
    parts = [p_all[..., idx] for idx in range(p_all.shape[-1])]
    parts[2] = parts[0]
    parts[3] = parts[1]
    return torch.stack(parts, dim=-1)


def apply_tied_c1r1_constraint_np(p_all: np.ndarray) -> np.ndarray:
    if not TIE_C1_C2_R1_R2:
        return p_all
    tied = np.array(p_all, copy=True)
    tied[..., 2] = tied[..., 0]
    tied[..., 3] = tied[..., 1]
    return tied


def denorm_log_to_hardcapped_params(base_train, base, wrapper, pred_norm, y_mean_t, y_std_t):
    p_flat, p_all, log_flat = base_train.denorm_log_to_positive_params(
        base,
        pred_norm,
        y_mean_t,
        y_std_t,
        wrapper.CONNECTION_COUNT,
        len(wrapper.V08_PARAM_NAMES),
    )
    parts = []
    for param_idx in range(len(wrapper.V08_PARAM_NAMES)):
        values = p_all[..., param_idx]
        if param_idx in (0, 2, 4):
            values = torch.clamp(values, max=CN_SCALE_HARD_CAP)
        elif param_idx == 6:
            values = torch.clamp(values, max=LN1_SCALE_HARD_CAP)
        parts.append(values)
    capped = apply_tied_c1r1_constraint(torch.stack(parts, dim=-1))
    capped_flat = capped.reshape_as(p_flat)
    capped_log_flat = torch.log10(torch.clamp(capped_flat, min=base_train.POSITIVE_LOWER))
    return capped_flat, capped, capped_log_flat


def reference_and_smoothness_loss(log_all, ref_log_all):
    # Constrain the parameters that caused the remaining sample-level issues:
    # Cn1/Cn2/Cn3 and Ln1, with Cn3/Ln1 receiving the strongest effect through
    # both reference-deviation and head-smoothness terms.
    watched = [0, 2, 4, 6]
    drift = torch.abs(log_all[..., watched] - ref_log_all[..., watched])
    reference_penalty = torch.mean(torch.relu(drift - REFERENCE_LOG_TOL) ** 2)
    head_std = torch.std(log_all[..., watched], dim=1, unbiased=False)
    smoothness_penalty = torch.mean(torch.relu(head_std - HEAD_LOG_STD_TOL) ** 2)
    return reference_penalty, smoothness_penalty


def sample_anchor_loss(base, wrapper, p_all, log_all, opt_log_all):
    watched = [0, 2, 4, 6]
    weights = torch.tensor([0.7, 0.7, 1.0, 1.0], dtype=base.REAL_DTYPE, device=log_all.device)
    drift = torch.abs(log_all[..., watched] - opt_log_all[..., watched])
    cl_anchor = torch.mean(torch.relu(drift - SAMPLE_CL_LOG_TOL) ** 2 * weights)

    opt_scales = torch.pow(torch.tensor(10.0, dtype=base.REAL_DTYPE, device=log_all.device), opt_log_all)
    scale_factors = torch.tensor(wrapper.V08_SCALE_FACTORS, dtype=base.REAL_DTYPE, device=log_all.device)
    pred_physical = p_all * scale_factors
    opt_physical = opt_scales * scale_factors
    pred_f0 = 1.0 / (2.0 * torch.pi * torch.sqrt(torch.clamp(pred_physical[..., 4] * pred_physical[..., 6], min=1e-60)))
    opt_f0 = 1.0 / (2.0 * torch.pi * torch.sqrt(torch.clamp(opt_physical[..., 4] * opt_physical[..., 6], min=1e-60)))
    f0_drift = torch.abs(torch.log10(torch.clamp(pred_f0, min=1.0)) - torch.log10(torch.clamp(opt_f0, min=1.0)))
    f0_anchor = torch.mean(torch.relu(f0_drift - SAMPLE_F0_LOG_TOL) ** 2)
    return cl_anchor, f0_anchor


def wrapped_phase_loss(pred_s, target_s):
    pred_ports = torch.stack([pred_s[:, :, 0, 0], pred_s[:, :, 1, 0]], dim=-1)
    target_ports = torch.stack([target_s[:, :, 0, 0], target_s[:, :, 1, 0]], dim=-1)
    phase_diff = torch.angle(pred_ports * torch.conj(target_ports))
    target_mag = torch.abs(target_ports)
    weights = torch.clamp(target_mag, min=0.05, max=1.0)
    return torch.mean(weights * phase_diff.pow(2))


def train_sparam_lc_limited(base_train, base, wrapper, model, x_norm, masks, sim, y_mean, y_std, ref_log_all_np, opt_log_all_np, device):
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    train_ds = TensorDataset(torch.tensor(train_idx, dtype=torch.long), torch.tensor(x_norm[train_idx], dtype=base.REAL_DTYPE))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    omega_t = torch.tensor(2.0 * np.pi * sim.freq_hz, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    ri_scale = base_train.target_ri_scale(base, sim.target_s, masks["train"], device)
    val_x = torch.tensor(x_norm[val_idx], dtype=base.REAL_DTYPE, device=device)
    val_base = torch.tensor(sim.base_abcds[val_idx], dtype=base.COMPLEX_DTYPE, device=device)
    val_target = torch.tensor(sim.target_s[val_idx], dtype=base.COMPLEX_DTYPE, device=device)
    ref_log_all_t = torch.tensor(ref_log_all_np, dtype=base.REAL_DTYPE, device=device)
    opt_log_all_t = torch.tensor(opt_log_all_np, dtype=base.REAL_DTYPE, device=device)
    val_ref_log = ref_log_all_t[val_idx]
    val_opt_log = opt_log_all_t[val_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=JOINT_LR, weight_decay=1e-8)
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_val = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, JOINT_EPOCHS + 1):
        model.train()
        total_s = 0.0
        total_phase = 0.0
        total_lc = 0.0
        total_anchor = 0.0
        total = 0.0
        seen = 0
        for idx_b, xb in loader:
            idx_np = idx_b.numpy()
            idx_t = idx_b.to(device)
            xb = xb.to(device)
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            _, p_all, log_flat = denorm_log_to_hardcapped_params(base_train, base, wrapper, model(xb), y_mean_t, y_std_t)
            log_all = log_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            pred_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t))
            s_loss = base_train.multihead_sparam_loss(pred_s, target_b, ri_scale)
            phase_loss = wrapped_phase_loss(pred_s, target_b)
            lc_loss, _, _, _, _ = lc_limit_loss(base, wrapper, p_all)
            ref_loss, smooth_loss = reference_and_smoothness_loss(log_all, ref_log_all_t[idx_t])
            cl_anchor_loss, f0_anchor_loss = sample_anchor_loss(base, wrapper, p_all, log_all, opt_log_all_t[idx_t])
            loss = (
                s_loss
                + PHASE_LOSS_WEIGHT * phase_loss
                + lc_loss
                + REFERENCE_PENALTY_WEIGHT * ref_loss
                + HEAD_SMOOTHNESS_WEIGHT * smooth_loss
                + SAMPLE_CL_ANCHOR_WEIGHT * cl_anchor_loss
                + SAMPLE_F0_ANCHOR_WEIGHT * f0_anchor_loss
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(xb)
            total_s += float(s_loss.detach().cpu()) * n
            total_phase += float(phase_loss.detach().cpu()) * n
            total_lc += float(lc_loss.detach().cpu()) * n
            total_anchor += float((cl_anchor_loss + f0_anchor_loss).detach().cpu()) * n
            total += float(loss.detach().cpu()) * n
            seen += n

        model.eval()
        with torch.no_grad():
            _, val_all, val_log_flat = denorm_log_to_hardcapped_params(base_train, base, wrapper, model(val_x), y_mean_t, y_std_t)
            val_log_all = val_log_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            val_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, val_base, val_all, omega_t))
            val_s_loss = base_train.multihead_sparam_loss(val_s, val_target, ri_scale)
            val_phase_loss = wrapped_phase_loss(val_s, val_target)
            val_lc_loss, val_c_loss, val_l_loss, val_fmin_loss, val_r_loss = lc_limit_loss(base, wrapper, val_all)
            val_ref_loss, val_smooth_loss = reference_and_smoothness_loss(val_log_all, val_ref_log)
            val_cl_anchor_loss, val_f0_anchor_loss = sample_anchor_loss(base, wrapper, val_all, val_log_all, val_opt_log)
            val_loss = (
                val_s_loss
                + PHASE_LOSS_WEIGHT * val_phase_loss
                + val_lc_loss
                + REFERENCE_PENALTY_WEIGHT * val_ref_loss
                + HEAD_SMOOTHNESS_WEIGHT * val_smooth_loss
                + SAMPLE_CL_ANCHOR_WEIGHT * val_cl_anchor_loss
                + SAMPLE_F0_ANCHOR_WEIGHT * val_f0_anchor_loss
            )

        row = {
            "stage": "symmetric_multihead_sparam_lc",
            "epoch": epoch,
            "train_total_loss": float(total / max(seen, 1)),
            "train_ri_loss": float(total_s / max(seen, 1)),
            "train_phase_loss": float(total_phase / max(seen, 1)),
            "train_lc_loss": float(total_lc / max(seen, 1)),
            "train_anchor_loss": float(total_anchor / max(seen, 1)),
            "val_total_loss": float(val_loss.detach().cpu()),
            "val_ri_loss": float(val_s_loss.detach().cpu()),
            "val_phase_loss": float(val_phase_loss.detach().cpu()),
            "val_lc_loss": float(val_lc_loss.detach().cpu()),
            "val_c_loss": float(val_c_loss.detach().cpu()),
            "val_l_loss": float(val_l_loss.detach().cpu()),
            "val_fmin_loss": float(val_fmin_loss.detach().cpu()),
            "val_r_loss": float(val_r_loss.detach().cpu()),
            "val_reference_loss": float(val_ref_loss.detach().cpu()),
            "val_head_smoothness_loss": float(val_smooth_loss.detach().cpu()),
            "val_cl_anchor_loss": float(val_cl_anchor_loss.detach().cpu()),
            "val_f0_anchor_loss": float(val_f0_anchor_loss.detach().cpu()),
        }
        rows.append(row)
        if row["val_total_loss"] < best_val:
            best_val = row["val_total_loss"]
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(
                f"[multihead-lc] epoch={epoch}, train_ri={row['train_ri_loss']:.4e}, "
                f"train_phase={row['train_phase_loss']:.4e}, train_lc={row['train_lc_loss']:.4e}, "
                f"val_ri={row['val_ri_loss']:.4e}, val_phase={row['val_phase_loss']:.4e}, "
                f"val_lc={row['val_lc_loss']:.4e}, val_fmin={row['val_fmin_loss']:.4e}, "
                f"val_cl_anchor={row['val_cl_anchor_loss']:.4e}, val_f0_anchor={row['val_f0_anchor_loss']:.4e}, "
                f"val_ref={row['val_reference_loss']:.4e}, "
                f"val_smooth={row['val_head_smoothness_loss']:.4e}",
                flush=True,
            )
        if stale >= JOINT_PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def lc_limit_stats(pred_table: pd.DataFrame, wrapper) -> pd.DataFrame:
    rows = []
    for name in ["Cn1_scale", "Cn2_scale", "Cn3_scale", "Ln1_scale", "Rn1_scale", "Rn2_scale"]:
        cols = [f"pred_conn{idx}_{name}" for idx in range(1, wrapper.CONNECTION_COUNT + 1)]
        values = pred_table[cols].to_numpy(dtype=np.float64).ravel()
        if name.startswith("Cn"):
            scale = 1e-14
            limit = C_LIMIT_F
        elif name.startswith("Ln"):
            scale = 1e-11
            limit = L_LIMIT_H
        else:
            scale = 1.0
            limit = R_UPPER_SCALE
        physical = values * scale
        rows.append(
            {
                "parameter": name,
                "scale_limit": float(limit / scale),
                "physical_limit": limit,
                "scale_min": float(np.min(values)),
                "scale_p95": float(np.quantile(values, 0.95)),
                "scale_p99": float(np.quantile(values, 0.99)),
                "scale_max": float(np.max(values)),
                "physical_max": float(np.max(physical)),
                "exceed_count": int(np.sum(physical > limit)),
                "total_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def lc_frequency_stats(pred_table: pd.DataFrame, wrapper) -> pd.DataFrame:
    values = []
    for idx in range(1, wrapper.CONNECTION_COUNT + 1):
        cn3 = pred_table[f"pred_conn{idx}_Cn3_scale"].to_numpy(dtype=np.float64) * 1e-14
        ln1 = pred_table[f"pred_conn{idx}_Ln1_scale"].to_numpy(dtype=np.float64) * 1e-11
        f0 = 1.0 / (2.0 * np.pi * np.sqrt(np.maximum(cn3 * ln1, 1e-60))) / 1e9
        values.extend(f0.tolist())
    arr = np.asarray(values, dtype=np.float64)
    return pd.DataFrame(
        [
            {
                "fmin_limit_ghz": LC_FMIN_HZ / 1e9,
                "f0_min_ghz": float(np.min(arr)),
                "f0_p01_ghz": float(np.quantile(arr, 0.01)),
                "f0_p05_ghz": float(np.quantile(arr, 0.05)),
                "f0_p10_ghz": float(np.quantile(arr, 0.10)),
                "f0_median_ghz": float(np.median(arr)),
                "f0_max_ghz": float(np.max(arr)),
                "count_below_60ghz": int(np.sum(arr < 60.0)),
                "count_below_80ghz": int(np.sum(arr < 80.0)),
                "total_count": int(len(arr)),
            }
        ]
    )


def evaluate_multihead_hardcapped(base_train, base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_mean, y_std, device):
    omega = 2.0 * np.pi * sim.freq_hz
    omega_t = torch.tensor(omega, dtype=base.REAL_DTYPE, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=base.REAL_DTYPE, device=device)
    y_std_t = torch.tensor(y_std, dtype=base.REAL_DTYPE, device=device)
    opt_by_id = opt_targets.set_index("sample_id")
    rows = []
    pred_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dut_df), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(dut_df))
            x_b = torch.tensor(x_norm[start:stop], dtype=base.REAL_DTYPE, device=device)
            base_b = torch.tensor(sim.base_abcds[start:stop], dtype=base.COMPLEX_DTYPE, device=device)
            p_flat, p_all, log_flat = denorm_log_to_hardcapped_params(base_train, base, wrapper, model(x_b), y_mean_t, y_std_t)
            pred_s_batch = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t)).cpu().numpy()
            pred_params = p_flat.cpu().numpy()
            pred_logs = log_flat.cpu().numpy()
            for local_i in range(stop - start):
                i = start + local_i
                sample = dut_df.iloc[i]
                sample_id = str(sample["sample_id"])
                target_s = sim.target_s[i]
                direct_s = base.abcd2s(base.opt2.cascade_direct(list(sim.base_abcds[i])))
                opt_p = opt_by_id.loc[sample_id, wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
                opt_s = base.abcd2s(wrapper.cascade_with_v08_np(base, sim.base_abcds[i], omega, opt_p))
                nn_s = pred_s_batch[local_i]
                direct = base_train.metric_dict(base, target_s, direct_s)
                opt = base_train.metric_dict(base, target_s, opt_s)
                nn = base_train.metric_dict(base, target_s, nn_s)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": sample["split"],
                        "file": sample["file"],
                        "dut_index": int(sample["dut_index"]),
                        "direct_nmse_s11_s21_ri": direct["nmse_s11_s21_ri"],
                        "optimized_nmse_s11_s21_ri": opt["nmse_s11_s21_ri"],
                        "nn_nmse_s11_s21_ri": nn["nmse_s11_s21_ri"],
                        "direct_mse_all_s": direct["mse_all_s"],
                        "optimized_mse_all_s": opt["mse_all_s"],
                        "nn_mse_all_s": nn["mse_all_s"],
                        "direct_s11_db_mae": direct["s11_db_mae"],
                        "optimized_s11_db_mae": opt["s11_db_mae"],
                        "nn_s11_db_mae": nn["s11_db_mae"],
                        "direct_s21_db_mae": direct["s21_db_mae"],
                        "optimized_s21_db_mae": opt["s21_db_mae"],
                        "nn_s21_db_mae": nn["s21_db_mae"],
                    }
                )
                pred_row = {"sample_id": sample_id, "split": sample["split"]}
                for col_idx, col_name in enumerate(base_train.multihead_target_columns(wrapper)):
                    pred_row[f"pred_{col_name}"] = float(pred_params[local_i, col_idx])
                    pred_row[f"pred_log10_{col_name}"] = float(pred_logs[local_i, col_idx])
                pred_rows.append(pred_row)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def main() -> None:
    base_train = load_module(BASE_TRAIN_SCRIPT, "v11_positive_multihead_base_train_lc")
    source = base_train.load_module(base_train.SOURCE_SCRIPT, "v11_lc_source")
    positive = base_train.load_module(base_train.POSITIVE_SCRIPT, "v11_lc_positive")
    wrapper = base_train.load_module(base_train.WRAPPER_SCRIPT, "v11_lc_wrapper")
    base = wrapper.load_module(wrapper.BASE_SCRIPT, "v11_lc_base")

    version_root = base.PROJECT_ROOT / "model_versions" / "v11_ads_v08_multihead_chain"
    opt_dir = version_root / "results" / base_train.OPT_RESULT_LABEL
    shared_nn_dir = version_root / "results" / base_train.SHARED_NN_LABEL
    source_ads_dir = version_root / "results" / base_train.SOURCE_ADS_LABEL
    output_dir = version_root / "results" / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    base.RUN_LABEL = RUN_LABEL
    base.OUTPUT_DIR = output_dir
    base.ADS_CACHE_DIR = source_ads_dir / "ads_cache"
    base.SIMULATION_BACKEND = "ads"
    base.USE_MODEL_SET_AS_VALIDATION = True
    base.ADS_DEVICE_LENGTH_SCALE = base_train.ADS_DEVICE_LENGTH_SCALE
    base.set_seed(base.RANDOM_SEED)

    opt_targets_all = pd.read_csv(opt_dir / base_train.OPT_TARGET_FILE, encoding="utf-8-sig")
    dut_all = positive.collect_lhs400_rdl_tsv_samples(base)
    target_ids = set(opt_targets_all["sample_id"].astype(str))
    excluded_unoptimized = dut_all[~dut_all["sample_id"].astype(str).isin(target_ids)].copy()
    dut_df = dut_all[dut_all["sample_id"].astype(str).isin(target_ids)].reset_index(drop=True)
    opt_targets = dut_df[["sample_id"]].merge(opt_targets_all, on="sample_id", how="left")

    settings = source.calibrated_ads_settings()
    settings["ads_device_length_scale"] = base_train.ADS_DEVICE_LENGTH_SCALE
    settings["ads_geometry_scale_note"] = "Applied to l_tmrdl, l_bsmrdl, and h_tsv by the v11 base ADS runner."
    sim = positive.load_single_device_simulation_with_retries(base, dut_df, settings)

    masks = base_train.split_masks(opt_targets)
    x_raw = opt_targets[base.STRUCTURE_COLUMNS].to_numpy(dtype=np.float64)
    y_linear = opt_targets[wrapper.V08_PARAM_NAMES].to_numpy(dtype=np.float64)
    y_log_shared = np.log10(np.clip(y_linear, base_train.POSITIVE_LOWER, base_train.POSITIVE_UPPER))
    y_log_multi = base_train.repeat_shared_targets(y_log_shared, wrapper)
    x_norm, x_mean, x_std = base_train.normalize_by_train(x_raw, masks["train"])
    y_log_multi_norm, y_log_multi_mean, y_log_multi_std = base_train.normalize_by_train(y_log_multi, masks["train"])

    device = torch.device("cuda" if base.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    shared_ckpt_path = shared_nn_dir / "positive_shared7_param_nns_log.pt"
    _, model, shared_checkpoint = load_symmetric_initialized_model(base, wrapper, shared_ckpt_path, device)
    init_ckpt_path = version_root / "results" / INIT_RUN_LABEL / INIT_CHECKPOINT_NAME
    init_checkpoint = torch.load(init_ckpt_path, map_location=device)
    model.load_state_dict(init_checkpoint["model_state_dict"])

    initial_metrics, initial_pred = evaluate_multihead_hardcapped(
        base_train, base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device
    )
    initial_summary = base_train.summarize(initial_metrics)
    with torch.no_grad():
        y_mean_t = torch.tensor(y_log_multi_mean, dtype=base.REAL_DTYPE, device=device)
        y_std_t = torch.tensor(y_log_multi_std, dtype=base.REAL_DTYPE, device=device)
        x_all_t = torch.tensor(x_norm, dtype=base.REAL_DTYPE, device=device)
        _, _, ref_log_flat = denorm_log_to_hardcapped_params(base_train, base, wrapper, model(x_all_t), y_mean_t, y_std_t)
        ref_log_all_np = ref_log_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES)).detach().cpu().numpy()
    opt_scales_multi = np.power(10.0, y_log_multi.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES)))
    opt_scales_multi = apply_tied_c1r1_constraint_np(opt_scales_multi)
    opt_log_all_np = np.log10(np.clip(opt_scales_multi, base_train.POSITIVE_LOWER, base_train.POSITIVE_UPPER))
    history = train_sparam_lc_limited(
        base_train,
        base,
        wrapper,
        model,
        x_norm,
        masks,
        sim,
        y_log_multi_mean,
        y_log_multi_std,
        ref_log_all_np,
        opt_log_all_np,
        device,
    )
    metrics, pred_table = evaluate_multihead_hardcapped(
        base_train, base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device
    )
    summary = base_train.summarize(metrics)
    sign_stats = base_train.parameter_sign_stats(pred_table, wrapper)
    lc_stats = lc_limit_stats(pred_table, wrapper)
    lc_freq_stats = lc_frequency_stats(pred_table, wrapper)
    plot_dir, plot_paths = base_train.save_plots(base, wrapper, output_dir, history.rename(columns={"train_total_loss": "train_ri_loss", "val_total_loss": "val_ri_loss"}), metrics, dut_df, sim, opt_targets, pred_table)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": RUN_LABEL,
                "architecture": "positive symmetric multi-head initialized from shared 30-30-20 networks; 6 learned heads mirrored to 12 connections; Cn2 is tied to Cn1 and Rn2 is tied to Rn1 before cascade evaluation",
                "training_objective": "S11/S21 real/imag and wrapped phase losses plus hard L/C caps, tied Cn1/Cn2 and Rn1/Rn2 topology, R upper, initial-reference, head-smoothness, same-sample C/L anchor, and same-sample Cn3/Ln1 f0 anchor penalties",
                "lc_limits": {
                    "C_limit_F": C_LIMIT_F,
                    "L_limit_H": L_LIMIT_H,
                    "LC_fmin_Hz": LC_FMIN_HZ,
                    "C_penalty_weight": C_PENALTY_WEIGHT,
                    "L_penalty_weight": L_PENALTY_WEIGHT,
                    "LC_fmin_penalty_weight": LC_FMIN_PENALTY_WEIGHT,
                    "R_upper_scale": R_UPPER_SCALE,
                    "R_penalty_weight": R_PENALTY_WEIGHT,
                    "reference_penalty_weight": REFERENCE_PENALTY_WEIGHT,
                    "head_smoothness_weight": HEAD_SMOOTHNESS_WEIGHT,
                    "phase_loss_weight": PHASE_LOSS_WEIGHT,
                    "sample_cl_anchor_weight": SAMPLE_CL_ANCHOR_WEIGHT,
                    "sample_f0_anchor_weight": SAMPLE_F0_ANCHOR_WEIGHT,
                    "reference_log_tolerance": REFERENCE_LOG_TOL,
                    "head_log_std_tolerance": HEAD_LOG_STD_TOL,
                    "sample_cl_log_tolerance": SAMPLE_CL_LOG_TOL,
                    "sample_f0_log_tolerance": SAMPLE_F0_LOG_TOL,
                },
                "equivalent_scale_limits": {"Cn_scale_limit": C_LIMIT_F / 1e-14, "Ln1_scale_limit": L_LIMIT_H / 1e-11},
                "symmetry": {"unique_head_count": UNIQUE_HEAD_COUNT, "connection_map": [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]},
                "tied_connection_topology": {"Cn2_scale": "Cn1_scale", "Rn2_scale": "Rn1_scale"},
                "feature_columns": base.STRUCTURE_COLUMNS,
                "target_columns": base_train.multihead_target_columns(wrapper),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_log_multi_mean": y_log_multi_mean.tolist(),
                "y_log_multi_std": y_log_multi_std.tolist(),
                "source_shared_checkpoint": str(shared_ckpt_path),
                "source_initial_checkpoint": str(init_ckpt_path),
                "source_optimized_targets": str(opt_dir / base_train.OPT_TARGET_FILE),
                "ads_device_length_scale": base_train.ADS_DEVICE_LENGTH_SCALE,
                "excluded_unoptimized_sample_ids": excluded_unoptimized["sample_id"].astype(str).tolist(),
                "shared_checkpoint_metadata": shared_checkpoint.get("metadata", {}),
                "initial_checkpoint_metadata": init_checkpoint.get("metadata", {}),
            },
        },
        output_dir / "positive_symmetric_multihead_lc_sample_anchor.pt",
    )

    history.to_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_history.csv", index=False, encoding="utf-8-sig")
    initial_metrics.to_csv(output_dir / "initial_shared_expanded_metrics.csv", index=False, encoding="utf-8-sig")
    initial_summary.to_csv(output_dir / "initial_shared_expanded_summary.csv", index=False, encoding="utf-8-sig")
    initial_pred.to_csv(output_dir / "initial_shared_expanded_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_summary.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_predictions.csv", index=False, encoding="utf-8-sig")
    sign_stats.to_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_parameter_sign_stats.csv", index=False, encoding="utf-8-sig")
    lc_stats.to_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_limit_stats.csv", index=False, encoding="utf-8-sig")
    lc_freq_stats.to_csv(output_dir / "positive_symmetric_multihead_lc_sample_anchor_frequency_stats.csv", index=False, encoding="utf-8-sig")
    excluded_unoptimized.to_csv(output_dir / "excluded_unoptimized_samples.csv", index=False, encoding="utf-8-sig")

    report = {
        "run_label": RUN_LABEL,
        "output_dir": str(output_dir),
        "samples": int(len(dut_df)),
        "excluded_unoptimized_samples": int(len(excluded_unoptimized)),
        "source_shared_checkpoint": str(shared_ckpt_path),
        "source_initial_checkpoint": str(init_ckpt_path),
        "lc_limits": {
            "C_limit_F": C_LIMIT_F,
            "L_limit_H": L_LIMIT_H,
            "LC_fmin_Hz": LC_FMIN_HZ,
            "C_penalty_weight": C_PENALTY_WEIGHT,
            "L_penalty_weight": L_PENALTY_WEIGHT,
            "LC_fmin_penalty_weight": LC_FMIN_PENALTY_WEIGHT,
            "R_upper_scale": R_UPPER_SCALE,
            "R_penalty_weight": R_PENALTY_WEIGHT,
            "reference_penalty_weight": REFERENCE_PENALTY_WEIGHT,
            "head_smoothness_weight": HEAD_SMOOTHNESS_WEIGHT,
            "phase_loss_weight": PHASE_LOSS_WEIGHT,
            "sample_cl_anchor_weight": SAMPLE_CL_ANCHOR_WEIGHT,
            "sample_f0_anchor_weight": SAMPLE_F0_ANCHOR_WEIGHT,
            "reference_log_tolerance": REFERENCE_LOG_TOL,
            "head_log_std_tolerance": HEAD_LOG_STD_TOL,
            "sample_cl_log_tolerance": SAMPLE_CL_LOG_TOL,
            "sample_f0_log_tolerance": SAMPLE_F0_LOG_TOL,
        },
        "equivalent_scale_limits": {"Cn_scale_limit": C_LIMIT_F / 1e-14, "Ln1_scale_limit": L_LIMIT_H / 1e-11},
        "tied_connection_topology": {"Cn2_scale": "Cn1_scale", "Rn2_scale": "Rn1_scale"},
        "epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "best_val_total_loss": float(history["val_total_loss"].min()) if len(history) else None,
        "initial_summary": initial_summary.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "lc_stats": lc_stats.to_dict(orient="records"),
        "lc_frequency_stats": lc_freq_stats.to_dict(orient="records"),
        "parameter_sign_stats": sign_stats.to_dict(orient="records"),
        "plot_dir": str(plot_dir),
        "plot_files": plot_paths,
    }
    (output_dir / "positive_symmetric_multihead_lc_sample_anchor_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "positive_symmetric_multihead_lc_sample_anchor_report.md").write_text(
        "\n".join(
            [
                "# V11 Positive Symmetric Multi-Head Training With Tied Cn1/Cn2 and Rn1/Rn2",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source shared checkpoint: `{shared_ckpt_path}`",
                f"- Source initial checkpoint: `{init_ckpt_path}`",
                "- Architecture: six learned connection-position heads mirrored to twelve connection networks as `1,2,3,4,5,6,6,5,4,3,2,1`.",
                "- Training target: cascaded `S11/S21` real/imag loss plus wrapped phase loss, hard ultra-tight L/C caps, tied `Cn1=Cn2` and `Rn1=Rn2` topology, R upper, initial-reference, head-smoothness, same-sample C/L anchor, and same-sample Cn3/Ln1 f0 anchor penalties.",
                "- Connection topology: `Cn2_scale` is overwritten with `Cn1_scale`, and `Rn2_scale` is overwritten with `Rn1_scale` before cascade evaluation and before writing predictions.",
                f"- C limit: `{C_LIMIT_F}` F, equivalent to `Cn*_scale < {C_LIMIT_F / 1e-14:.6g}`.",
                f"- L limit: `{L_LIMIT_H}` H, equivalent to `Ln1_scale < {L_LIMIT_H / 1e-11:.6g}`.",
                f"- Cn3/Ln1 LC frequency lower limit: `{LC_FMIN_HZ / 1e9:.6g}` GHz.",
                f"- C penalty weight: `{C_PENALTY_WEIGHT}`",
                f"- L penalty weight: `{L_PENALTY_WEIGHT}`",
                f"- LC frequency penalty weight: `{LC_FMIN_PENALTY_WEIGHT}`",
                f"- Wrapped phase loss weight: `{PHASE_LOSS_WEIGHT}`",
                f"- Same-sample C/L anchor weight: `{SAMPLE_CL_ANCHOR_WEIGHT}`, log10 tolerance `{SAMPLE_CL_LOG_TOL}`.",
                f"- Same-sample Cn3/Ln1 f0 anchor weight: `{SAMPLE_F0_ANCHOR_WEIGHT}`, log10 tolerance `{SAMPLE_F0_LOG_TOL}`.",
                f"- R upper scale: `{R_UPPER_SCALE}`, penalty weight `{R_PENALTY_WEIGHT}`",
                f"- Reference drift penalty weight: `{REFERENCE_PENALTY_WEIGHT}`, log10 tolerance `{REFERENCE_LOG_TOL}`.",
                f"- Head smoothness penalty weight: `{HEAD_SMOOTHNESS_WEIGHT}`, log10 std tolerance `{HEAD_LOG_STD_TOL}`.",
                f"- Epochs completed: `{report['epochs_completed']}`",
                "",
                "## Constraint Rationale",
                "",
                "- The previous ultra-tight run satisfied `C <= 1e-12 F` and `L <= 1e-11 H`, but several samples sat on incompatible C/L/R combinations.",
                "- This run tests whether the simplified symmetric branch topology `Cn1=Cn2` and `Rn1=Rn2` removes one source of branch-placement freedom.",
                "- The R upper penalty discourages the L/C error from being transferred into `Rn1_scale` upper-bound saturation.",
                "- The LC frequency lower-bound penalty controls very-low-frequency Cn3/Ln1 outliers.",
                "- The same-sample C/L and f0 anchors target the later diagnosis: resonant samples stayed below global optimized maxima but moved far away from their own optimized parameter combination.",
                "- Symmetric heads encode the physical left-right symmetry of the 13-device cascade and reduce the trainable output heads by half.",
                "",
                "## Initial Shared-Expanded Summary",
                "",
                dataframe_to_markdown(initial_summary),
                "",
                "## L/C-Limited Multi-Head Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## L/C Limit Stats",
                "",
                dataframe_to_markdown(lc_stats),
                "",
                "## Cn3/Ln1 LC Frequency Stats",
                "",
                dataframe_to_markdown(lc_freq_stats),
                "",
                "## Parameter Sign Summary",
                "",
                dataframe_to_markdown(sign_stats),
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "validation_archive.md").write_text(
        "\n".join(
            [
                "# Validation Archive",
                "",
                f"- Entry: `{Path(__file__).name}`",
                "- Status: completed",
                f"- Samples: `{len(dut_df)}`",
                f"- Train/val/test: `{int(masks['train'].sum())}` / `{int(masks['val'].sum())}` / `{int(masks['test'].sum())}`",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- C limit: `{C_LIMIT_F}` F",
                f"- L limit: `{L_LIMIT_H}` H",
                f"- Cn3/Ln1 LC frequency lower limit: `{LC_FMIN_HZ / 1e9:.6g}` GHz",
                f"- C penalty weight: `{C_PENALTY_WEIGHT}`",
                f"- L penalty weight: `{L_PENALTY_WEIGHT}`",
                f"- LC frequency penalty weight: `{LC_FMIN_PENALTY_WEIGHT}`",
                f"- Wrapped phase loss weight: `{PHASE_LOSS_WEIGHT}`",
                f"- Same-sample C/L anchor weight: `{SAMPLE_CL_ANCHOR_WEIGHT}`, log10 tolerance `{SAMPLE_CL_LOG_TOL}`",
                f"- Same-sample Cn3/Ln1 f0 anchor weight: `{SAMPLE_F0_ANCHOR_WEIGHT}`, log10 tolerance `{SAMPLE_F0_LOG_TOL}`",
                f"- R upper scale: `{R_UPPER_SCALE}`",
                f"- R penalty weight: `{R_PENALTY_WEIGHT}`",
                f"- Reference drift penalty weight: `{REFERENCE_PENALTY_WEIGHT}`",
                f"- Head smoothness penalty weight: `{HEAD_SMOOTHNESS_WEIGHT}`",
                f"- Comparison plots: `{len(plot_paths)}`",
                "",
                "## L/C-Limited Summary",
                "",
                dataframe_to_markdown(summary),
                "",
                "## L/C Limit Stats",
                "",
                dataframe_to_markdown(lc_stats),
                "",
                "## Cn3/Ln1 LC Frequency Stats",
                "",
                dataframe_to_markdown(lc_freq_stats),
            ]
        ),
        encoding="utf-8",
    )
    print("L/C-limited summary:", flush=True)
    print(dataframe_to_markdown(summary), flush=True)
    print("L/C limit stats:", flush=True)
    print(dataframe_to_markdown(lc_stats), flush=True)
    print(f"Done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
