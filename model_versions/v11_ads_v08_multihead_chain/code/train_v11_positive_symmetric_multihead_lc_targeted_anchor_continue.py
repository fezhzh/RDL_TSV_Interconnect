# -*- coding: utf-8 -*-
"""Continue positive symmetric multi-head NN with targeted C/L anchors.

Run this file directly in VS Code. No command-line arguments are required.

The 13-device cascade is left-right symmetric. This model learns only the first
six connection-network heads and mirrors them to the last six positions:

    1,2,3,4,5,6,6,5,4,3,2,1

The L/C limits are deliberately milder than the previous strong-LC run, and
the Cn3/Ln1 branch is constrained so its LC frequency stays mostly above the
model band:

    Cn1/Cn2/Cn3 < 1e-11 F
    Ln1_scale < 4.0, equivalent to Ln1 < 4e-11 H
    f0(Cn3, Ln1) >= 60 GHz, where f0 = 1 / (2*pi*sqrt(Cn3*Ln1))

This variant starts from the previous sample-anchor checkpoint and adds stronger
targeted constraints for the remaining difficult samples. It emphasizes Cn3,
Ln1, and the same-sample Cn3/Ln1 resonance frequency rather than global L/C
limits.
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

RUN_LABEL = "v11_positive_symmetric_multihead_lc_targeted_anchor_continue_log_adslen09"
INIT_RUN_LABEL = "v11_positive_symmetric_multihead_lc_sample_anchor_continue_log_adslen09"
INIT_CHECKPOINT_NAME = "positive_symmetric_multihead_lc_sample_anchor.pt"
UNIQUE_HEAD_COUNT = 6
C_LIMIT_F = 1e-11
L_LIMIT_H = 4e-11
LC_FMIN_HZ = 60e9
C_PENALTY_WEIGHT = 1.0
L_PENALTY_WEIGHT = 0.35
LC_FMIN_PENALTY_WEIGHT = 0.8
R_UPPER_SCALE = 5e4
R_PENALTY_WEIGHT = 0.03
REFERENCE_PENALTY_WEIGHT = 0.04
HEAD_SMOOTHNESS_WEIGHT = 0.02
PHASE_LOSS_WEIGHT = 0.12
SAMPLE_CL_ANCHOR_WEIGHT = 0.04
SAMPLE_F0_ANCHOR_WEIGHT = 0.08
TARGETED_BRANCH_ANCHOR_WEIGHT = 0.25
TARGETED_F0_ANCHOR_WEIGHT = 0.35
TARGETED_SPARAM_WEIGHT = 12.0
TARGETED_SAMPLE_IDS = {
    "LHS400_Connection2_train_dut72",
    "LHS400_Connection2_train_dut123",
    "LHS400_Connection2_train_dut253",
    "LHS400_Connection2_train_dut258",
}
REFERENCE_LOG_TOL = 0.45
HEAD_LOG_STD_TOL = 0.35
SAMPLE_CL_LOG_TOL = 0.20
SAMPLE_F0_LOG_TOL = 0.10
TARGETED_BRANCH_LOG_TOL = 0.12
TARGETED_F0_LOG_TOL = 0.06
HIGH_OPT_F0_HZ = 150e9
HIGH_OPT_F0_PRED_FLOOR_HZ = 120e9
HIGH_OPT_F0_FLOOR_WEIGHT = 0.10
JOINT_EPOCHS = 180
JOINT_PATIENCE = 45
JOINT_LR = 8e-6
BATCH_SIZE = 8
PRINT_EVERY = 10


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


def sample_anchor_loss(base, wrapper, p_all, log_all, opt_log_all, sample_weights=None):
    watched = [0, 2, 4, 6]
    weights = torch.tensor([0.7, 0.7, 1.0, 1.0], dtype=base.REAL_DTYPE, device=log_all.device)
    drift = torch.abs(log_all[..., watched] - opt_log_all[..., watched])
    cl_per_sample = torch.mean(torch.relu(drift - SAMPLE_CL_LOG_TOL) ** 2 * weights, dim=(1, 2))

    opt_scales = torch.pow(torch.tensor(10.0, dtype=base.REAL_DTYPE, device=log_all.device), opt_log_all)
    scale_factors = torch.tensor(wrapper.V08_SCALE_FACTORS, dtype=base.REAL_DTYPE, device=log_all.device)
    pred_physical = p_all * scale_factors
    opt_physical = opt_scales * scale_factors
    pred_f0 = 1.0 / (2.0 * torch.pi * torch.sqrt(torch.clamp(pred_physical[..., 4] * pred_physical[..., 6], min=1e-60)))
    opt_f0 = 1.0 / (2.0 * torch.pi * torch.sqrt(torch.clamp(opt_physical[..., 4] * opt_physical[..., 6], min=1e-60)))
    f0_drift = torch.abs(torch.log10(torch.clamp(pred_f0, min=1.0)) - torch.log10(torch.clamp(opt_f0, min=1.0)))
    f0_per_sample = torch.mean(torch.relu(f0_drift - SAMPLE_F0_LOG_TOL) ** 2, dim=1)

    branch_drift = torch.abs(log_all[..., [4, 6]] - opt_log_all[..., [4, 6]])
    branch_per_sample = torch.mean(torch.relu(branch_drift - TARGETED_BRANCH_LOG_TOL) ** 2, dim=(1, 2))
    f0_tight_per_sample = torch.mean(torch.relu(f0_drift - TARGETED_F0_LOG_TOL) ** 2, dim=1)
    opt_high_f0 = (opt_f0 > HIGH_OPT_F0_HZ).to(dtype=base.REAL_DTYPE)
    high_f0_floor_per_sample = torch.mean(opt_high_f0 * torch.relu(HIGH_OPT_F0_PRED_FLOOR_HZ / pred_f0 - 1.0) ** 2, dim=1)

    if sample_weights is None:
        sample_weights = torch.ones_like(cl_per_sample)
    sample_weights = sample_weights / torch.mean(sample_weights)
    cl_anchor = torch.mean(sample_weights * cl_per_sample)
    f0_anchor = torch.mean(sample_weights * f0_per_sample)
    branch_anchor = torch.mean(sample_weights * branch_per_sample)
    f0_tight_anchor = torch.mean(sample_weights * f0_tight_per_sample)
    high_f0_floor = torch.mean(sample_weights * high_f0_floor_per_sample)
    return cl_anchor, f0_anchor, branch_anchor, f0_tight_anchor, high_f0_floor


def wrapped_phase_loss(pred_s, target_s):
    pred_ports = torch.stack([pred_s[:, :, 0, 0], pred_s[:, :, 1, 0]], dim=-1)
    target_ports = torch.stack([target_s[:, :, 0, 0], target_s[:, :, 1, 0]], dim=-1)
    phase_diff = torch.angle(pred_ports * torch.conj(target_ports))
    target_mag = torch.abs(target_ports)
    weights = torch.clamp(target_mag, min=0.05, max=1.0)
    return torch.mean(weights * phase_diff.pow(2))


def weighted_sparam_loss(pred_s, target_s, ri_scale, sample_weights):
    pred_ri = torch.stack(
        [pred_s[:, :, 0, 0].real, pred_s[:, :, 0, 0].imag, pred_s[:, :, 1, 0].real, pred_s[:, :, 1, 0].imag],
        dim=-1,
    )
    target_ri = torch.stack(
        [target_s[:, :, 0, 0].real, target_s[:, :, 0, 0].imag, target_s[:, :, 1, 0].real, target_s[:, :, 1, 0].imag],
        dim=-1,
    )
    per_sample = torch.mean(((pred_ri - target_ri) / ri_scale) ** 2, dim=(1, 2))
    sample_weights = sample_weights / torch.mean(sample_weights)
    return torch.mean(sample_weights * per_sample)


def weighted_phase_loss(pred_s, target_s, sample_weights):
    pred_ports = torch.stack([pred_s[:, :, 0, 0], pred_s[:, :, 1, 0]], dim=-1)
    target_ports = torch.stack([target_s[:, :, 0, 0], target_s[:, :, 1, 0]], dim=-1)
    phase_diff = torch.angle(pred_ports * torch.conj(target_ports))
    weights = torch.clamp(torch.abs(target_ports), min=0.05, max=1.0)
    per_sample = torch.mean(weights * phase_diff.pow(2), dim=(1, 2))
    sample_weights = sample_weights / torch.mean(sample_weights)
    return torch.mean(sample_weights * per_sample)


def train_sparam_lc_limited(base_train, base, wrapper, model, x_norm, masks, sim, y_mean, y_std, ref_log_all_np, opt_log_all_np, sample_weight_np, device):
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    train_ds = TensorDataset(
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(x_norm[train_idx], dtype=base.REAL_DTYPE),
        torch.tensor(sample_weight_np[train_idx], dtype=base.REAL_DTYPE),
    )
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
    sample_weight_t = torch.tensor(sample_weight_np, dtype=base.REAL_DTYPE, device=device)
    val_ref_log = ref_log_all_t[val_idx]
    val_opt_log = opt_log_all_t[val_idx]
    val_weight = sample_weight_t[val_idx]

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
        for idx_b, xb, wb in loader:
            idx_np = idx_b.numpy()
            idx_t = idx_b.to(device)
            xb = xb.to(device)
            wb = wb.to(device)
            base_b = torch.tensor(sim.base_abcds[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            target_b = torch.tensor(sim.target_s[idx_np], dtype=base.COMPLEX_DTYPE, device=device)
            optimizer.zero_grad(set_to_none=True)
            _, p_all, log_flat = base_train.denorm_log_to_positive_params(
                base,
                model(xb),
                y_mean_t,
                y_std_t,
                wrapper.CONNECTION_COUNT,
                len(wrapper.V08_PARAM_NAMES),
            )
            log_all = log_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            pred_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, base_b, p_all, omega_t))
            s_loss = weighted_sparam_loss(pred_s, target_b, ri_scale, wb)
            phase_loss = weighted_phase_loss(pred_s, target_b, wb)
            lc_loss, _, _, _, _ = lc_limit_loss(base, wrapper, p_all)
            ref_loss, smooth_loss = reference_and_smoothness_loss(log_all, ref_log_all_t[idx_t])
            cl_anchor_loss, f0_anchor_loss, branch_anchor_loss, f0_tight_loss, high_f0_floor_loss = sample_anchor_loss(
                base, wrapper, p_all, log_all, opt_log_all_t[idx_t], wb
            )
            loss = (
                s_loss
                + PHASE_LOSS_WEIGHT * phase_loss
                + lc_loss
                + REFERENCE_PENALTY_WEIGHT * ref_loss
                + HEAD_SMOOTHNESS_WEIGHT * smooth_loss
                + SAMPLE_CL_ANCHOR_WEIGHT * cl_anchor_loss
                + SAMPLE_F0_ANCHOR_WEIGHT * f0_anchor_loss
                + TARGETED_BRANCH_ANCHOR_WEIGHT * branch_anchor_loss
                + TARGETED_F0_ANCHOR_WEIGHT * f0_tight_loss
                + HIGH_OPT_F0_FLOOR_WEIGHT * high_f0_floor_loss
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
            total_anchor += float((cl_anchor_loss + f0_anchor_loss + branch_anchor_loss + f0_tight_loss + high_f0_floor_loss).detach().cpu()) * n
            total += float(loss.detach().cpu()) * n
            seen += n

        model.eval()
        with torch.no_grad():
            _, val_all, val_log_flat = base_train.denorm_log_to_positive_params(
                base,
                model(val_x),
                y_mean_t,
                y_std_t,
                wrapper.CONNECTION_COUNT,
                len(wrapper.V08_PARAM_NAMES),
            )
            val_log_all = val_log_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
            val_s = base.abcd2s_torch(wrapper.cascade_with_v08_torch(base, val_base, val_all, omega_t))
            val_s_loss = weighted_sparam_loss(val_s, val_target, ri_scale, val_weight)
            val_phase_loss = weighted_phase_loss(val_s, val_target, val_weight)
            val_lc_loss, val_c_loss, val_l_loss, val_fmin_loss, val_r_loss = lc_limit_loss(base, wrapper, val_all)
            val_ref_loss, val_smooth_loss = reference_and_smoothness_loss(val_log_all, val_ref_log)
            val_cl_anchor_loss, val_f0_anchor_loss, val_branch_anchor_loss, val_f0_tight_loss, val_high_f0_floor_loss = sample_anchor_loss(
                base, wrapper, val_all, val_log_all, val_opt_log, val_weight
            )
            val_loss = (
                val_s_loss
                + PHASE_LOSS_WEIGHT * val_phase_loss
                + val_lc_loss
                + REFERENCE_PENALTY_WEIGHT * val_ref_loss
                + HEAD_SMOOTHNESS_WEIGHT * val_smooth_loss
                + SAMPLE_CL_ANCHOR_WEIGHT * val_cl_anchor_loss
                + SAMPLE_F0_ANCHOR_WEIGHT * val_f0_anchor_loss
                + TARGETED_BRANCH_ANCHOR_WEIGHT * val_branch_anchor_loss
                + TARGETED_F0_ANCHOR_WEIGHT * val_f0_tight_loss
                + HIGH_OPT_F0_FLOOR_WEIGHT * val_high_f0_floor_loss
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
            "val_branch_anchor_loss": float(val_branch_anchor_loss.detach().cpu()),
            "val_f0_tight_loss": float(val_f0_tight_loss.detach().cpu()),
            "val_high_f0_floor_loss": float(val_high_f0_floor_loss.detach().cpu()),
        }
        rows.append(row)
        selection_loss = row["train_total_loss"]
        if selection_loss < best_val:
            best_val = selection_loss
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
                f"val_branch={row['val_branch_anchor_loss']:.4e}, val_f0tight={row['val_f0_tight_loss']:.4e}, "
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
    sample_ids = opt_targets["sample_id"].astype(str).to_numpy()
    sample_weight_np = np.ones(len(sample_ids), dtype=np.float64)
    sample_weight_np[np.isin(sample_ids, list(TARGETED_SAMPLE_IDS))] = TARGETED_SPARAM_WEIGHT
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

    initial_metrics, initial_pred = base_train.evaluate_multihead(base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device)
    initial_summary = base_train.summarize(initial_metrics)
    with torch.no_grad():
        y_mean_t = torch.tensor(y_log_multi_mean, dtype=base.REAL_DTYPE, device=device)
        y_std_t = torch.tensor(y_log_multi_std, dtype=base.REAL_DTYPE, device=device)
        x_all_t = torch.tensor(x_norm, dtype=base.REAL_DTYPE, device=device)
        _, _, ref_log_flat = base_train.denorm_log_to_positive_params(
            base,
            model(x_all_t),
            y_mean_t,
            y_std_t,
            wrapper.CONNECTION_COUNT,
            len(wrapper.V08_PARAM_NAMES),
        )
        ref_log_all_np = ref_log_flat.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES)).detach().cpu().numpy()
    opt_log_all_np = y_log_multi.reshape(-1, wrapper.CONNECTION_COUNT, len(wrapper.V08_PARAM_NAMES))
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
        sample_weight_np,
        device,
    )
    metrics, pred_table = base_train.evaluate_multihead(base, wrapper, model, dut_df, sim, opt_targets, x_norm, y_log_multi_mean, y_log_multi_std, device)
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
                "architecture": "positive symmetric multi-head initialized from shared 30-30-20 networks; 6 learned heads mirrored to 12 connections",
                "training_objective": "S11/S21 real/imag and wrapped phase losses plus moderate L/C, R upper, initial-reference, head-smoothness, same-sample C/L anchor, and same-sample Cn3/Ln1 f0 anchor penalties",
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
                    "targeted_branch_anchor_weight": TARGETED_BRANCH_ANCHOR_WEIGHT,
                    "targeted_f0_anchor_weight": TARGETED_F0_ANCHOR_WEIGHT,
                    "targeted_sparam_weight": TARGETED_SPARAM_WEIGHT,
                    "targeted_sample_ids": sorted(TARGETED_SAMPLE_IDS),
                    "targeted_branch_log_tolerance": TARGETED_BRANCH_LOG_TOL,
                    "targeted_f0_log_tolerance": TARGETED_F0_LOG_TOL,
                    "high_opt_f0_hz": HIGH_OPT_F0_HZ,
                    "high_opt_f0_pred_floor_hz": HIGH_OPT_F0_PRED_FLOOR_HZ,
                    "high_opt_f0_floor_weight": HIGH_OPT_F0_FLOOR_WEIGHT,
                    "reference_log_tolerance": REFERENCE_LOG_TOL,
                    "head_log_std_tolerance": HEAD_LOG_STD_TOL,
                    "sample_cl_log_tolerance": SAMPLE_CL_LOG_TOL,
                    "sample_f0_log_tolerance": SAMPLE_F0_LOG_TOL,
                },
                "equivalent_scale_limits": {"Cn_scale_limit": C_LIMIT_F / 1e-14, "Ln1_scale_limit": L_LIMIT_H / 1e-11},
                "symmetry": {"unique_head_count": UNIQUE_HEAD_COUNT, "connection_map": [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]},
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
            "targeted_branch_anchor_weight": TARGETED_BRANCH_ANCHOR_WEIGHT,
            "targeted_f0_anchor_weight": TARGETED_F0_ANCHOR_WEIGHT,
            "targeted_sparam_weight": TARGETED_SPARAM_WEIGHT,
            "targeted_sample_ids": sorted(TARGETED_SAMPLE_IDS),
            "targeted_branch_log_tolerance": TARGETED_BRANCH_LOG_TOL,
            "targeted_f0_log_tolerance": TARGETED_F0_LOG_TOL,
            "high_opt_f0_hz": HIGH_OPT_F0_HZ,
            "high_opt_f0_pred_floor_hz": HIGH_OPT_F0_PRED_FLOOR_HZ,
            "high_opt_f0_floor_weight": HIGH_OPT_F0_FLOOR_WEIGHT,
            "reference_log_tolerance": REFERENCE_LOG_TOL,
            "head_log_std_tolerance": HEAD_LOG_STD_TOL,
            "sample_cl_log_tolerance": SAMPLE_CL_LOG_TOL,
            "sample_f0_log_tolerance": SAMPLE_F0_LOG_TOL,
        },
        "equivalent_scale_limits": {"Cn_scale_limit": C_LIMIT_F / 1e-14, "Ln1_scale_limit": L_LIMIT_H / 1e-11},
        "epochs_completed": int(history["epoch"].iloc[-1]) if len(history) else 0,
        "best_weighted_train_total_loss": float(history["train_total_loss"].min()) if len(history) else None,
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
                "# V11 Positive Symmetric Multi-Head Targeted C/L Anchor Continuation",
                "",
                f"- Entry: `{Path(__file__).name}`",
                f"- Output: `{output_dir}`",
                f"- Source shared checkpoint: `{shared_ckpt_path}`",
                f"- Source initial checkpoint: `{init_ckpt_path}`",
                "- Architecture: six learned connection-position heads mirrored to twelve connection networks as `1,2,3,4,5,6,6,5,4,3,2,1`.",
                "- Training target: cascaded `S11/S21` real/imag loss plus wrapped phase loss, moderate L/C limit, R upper, initial-reference, head-smoothness, same-sample C/L anchor, same-sample Cn3/Ln1 f0 anchor, and targeted Cn3/Ln1 branch penalties.",
                f"- C limit: `{C_LIMIT_F}` F, equivalent to `Cn*_scale < {C_LIMIT_F / 1e-14:.6g}`.",
                f"- L limit: `{L_LIMIT_H}` H, equivalent to `Ln1_scale < {L_LIMIT_H / 1e-11:.6g}`.",
                f"- Cn3/Ln1 LC frequency lower limit: `{LC_FMIN_HZ / 1e9:.6g}` GHz.",
                f"- C penalty weight: `{C_PENALTY_WEIGHT}`",
                f"- L penalty weight: `{L_PENALTY_WEIGHT}`",
                f"- LC frequency penalty weight: `{LC_FMIN_PENALTY_WEIGHT}`",
                f"- Wrapped phase loss weight: `{PHASE_LOSS_WEIGHT}`",
                f"- Same-sample C/L anchor weight: `{SAMPLE_CL_ANCHOR_WEIGHT}`, log10 tolerance `{SAMPLE_CL_LOG_TOL}`.",
                f"- Same-sample Cn3/Ln1 f0 anchor weight: `{SAMPLE_F0_ANCHOR_WEIGHT}`, log10 tolerance `{SAMPLE_F0_LOG_TOL}`.",
                f"- Targeted samples: `{', '.join(sorted(TARGETED_SAMPLE_IDS))}`.",
                f"- Targeted S-parameter sample weight: `{TARGETED_SPARAM_WEIGHT}`.",
                f"- Targeted Cn3/Ln1 branch anchor weight: `{TARGETED_BRANCH_ANCHOR_WEIGHT}`, log10 tolerance `{TARGETED_BRANCH_LOG_TOL}`.",
                f"- Targeted f0 anchor weight: `{TARGETED_F0_ANCHOR_WEIGHT}`, log10 tolerance `{TARGETED_F0_LOG_TOL}`.",
                f"- High optimized f0 floor: if optimized f0 is above `{HIGH_OPT_F0_HZ / 1e9:.6g}` GHz, penalize predicted f0 below `{HIGH_OPT_F0_PRED_FLOOR_HZ / 1e9:.6g}` GHz with weight `{HIGH_OPT_F0_FLOOR_WEIGHT}`.",
                f"- R upper scale: `{R_UPPER_SCALE}`, penalty weight `{R_PENALTY_WEIGHT}`",
                f"- Reference drift penalty weight: `{REFERENCE_PENALTY_WEIGHT}`, log10 tolerance `{REFERENCE_LOG_TOL}`.",
                f"- Head smoothness penalty weight: `{HEAD_SMOOTHNESS_WEIGHT}`, log10 std tolerance `{HEAD_LOG_STD_TOL}`.",
                f"- Epochs completed: `{report['epochs_completed']}`",
                "",
                "## Constraint Rationale",
                "",
                "- The previous `Ln1_scale < 2` strong-LC run reduced spike count but degraded most samples. This run uses a milder `Ln1_scale < 4` limit.",
                "- The C limit remains `1e-11 F` to control `Cn3_scale` overshoot.",
                "- The R upper penalty discourages the L/C error from being transferred into `Rn1_scale` and `Rn2_scale` upper-bound saturation.",
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
                f"- Targeted samples: `{', '.join(sorted(TARGETED_SAMPLE_IDS))}`",
                f"- Targeted S-parameter sample weight: `{TARGETED_SPARAM_WEIGHT}`",
                f"- Targeted Cn3/Ln1 branch anchor weight: `{TARGETED_BRANCH_ANCHOR_WEIGHT}`",
                f"- Targeted f0 anchor weight: `{TARGETED_F0_ANCHOR_WEIGHT}`",
                f"- High optimized f0 floor: `{HIGH_OPT_F0_PRED_FLOOR_HZ / 1e9:.6g}` GHz when optimized f0 exceeds `{HIGH_OPT_F0_HZ / 1e9:.6g}` GHz",
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
