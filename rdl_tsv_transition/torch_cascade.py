# -*- coding: utf-8 -*-
"""PyTorch 端到端级联和单 DUT HFSS 微调。"""

import copy
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .constants import Z_REF
from .devices import DeviceBlock
from .model import Normalizer, TransitionElementNN
from .utils import abcd2s_torch


def transition_abcd_torch(values: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """values shape = [n_freq, 6]，顺序 [L1, R1, L2, R2, C1, G1]。"""
    L1 = values[:, 0]
    R1 = values[:, 1]
    L2 = values[:, 2]
    R2 = values[:, 3]
    C1 = values[:, 4]
    G1 = values[:, 5]

    Z1 = torch.complex(R1, omega * L1)
    Z2 = torch.complex(R2, omega * L2)
    Y = torch.complex(G1, omega * C1)

    one = torch.ones_like(Z1)
    A = one + Z1 * Y
    B = Z1 + Z2 + Z1 * Z2 * Y
    C_abcd = Y
    D = one + Z2 * Y

    ABCD = torch.zeros((omega.numel(), 2, 2), dtype=torch.complex128, device=omega.device)
    ABCD[:, 0, 0] = A
    ABCD[:, 0, 1] = B
    ABCD[:, 1, 0] = C_abcd
    ABCD[:, 1, 1] = D
    return ABCD


def cascade_with_transition_values_torch(
    base_abcds_torch: Sequence[torch.Tensor],
    transition_values_torch: torch.Tensor,
    omega_torch: torch.Tensor,
) -> torch.Tensor:
    """
    transition_values_torch shape = [n_trans, n_freq, 6]
    返回 S 参数 shape = [n_freq, 2, 2]
    """
    n_trans = transition_values_torch.shape[0]
    if len(base_abcds_torch) != n_trans + 1:
        raise ValueError("base_abcds_torch 数量必须比 transition_values_torch 多 1")

    abcd_curr = base_abcds_torch[0]
    for i in range(n_trans):
        trans_abcd = transition_abcd_torch(transition_values_torch[i], omega_torch)
        abcd_curr = torch.matmul(torch.matmul(abcd_curr, trans_abcd), base_abcds_torch[i + 1])
    return abcd2s_torch(abcd_curr, Z0=Z_REF)


def fine_tune_transition_nn_on_hfss(
    model: TransitionElementNN,
    normalizer: Normalizer,
    blocks: List[DeviceBlock],
    freqs_hz: np.ndarray,
    base_abcds_np: Sequence[np.ndarray],
    target_s_np: np.ndarray,
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    epochs: int = 300,
    lr: float = 2e-4,
    reg_weight: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> TransitionElementNN:
    """以 HFSS 整体 S 参数为目标，端到端微调过渡结构 NN。"""
    if device is None:
        device = next(model.parameters()).device

    n_trans = len(blocks) - 1
    n_freq = len(freqs_hz)

    X_norm = (X_raw - normalizer.x_mean) / normalizer.x_std
    logY = np.log(np.maximum(Y_raw, 1e-300))
    Y_norm = (logY - normalizer.y_mean) / normalizer.y_std

    X_t = torch.tensor(X_norm, dtype=torch.float64, device=device)
    Y_t = torch.tensor(Y_norm, dtype=torch.float64, device=device)
    y_mean_t = torch.tensor(normalizer.y_mean, dtype=torch.float64, device=device)
    y_std_t = torch.tensor(normalizer.y_std, dtype=torch.float64, device=device)

    omega_t = torch.tensor(2.0 * np.pi * freqs_hz, dtype=torch.float64, device=device)
    target_s_t = torch.tensor(target_s_np, dtype=torch.complex128, device=device)
    base_abcds_t = [torch.tensor(a, dtype=torch.complex128, device=device) for a in base_abcds_np]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-8)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        y_norm_pred = model(X_t)
        log_values = y_norm_pred * y_std_t + y_mean_t
        log_values = torch.clamp(log_values, min=-100.0, max=100.0)
        values = torch.exp(log_values).reshape(n_trans, n_freq, 6)

        pred_s = cascade_with_transition_values_torch(base_abcds_t, values, omega_t)

        loss_s = torch.mean(torch.abs(pred_s - target_s_t) ** 2)
        loss_reg = F.mse_loss(y_norm_pred, Y_t)
        loss = loss_s + reg_weight * loss_reg

        if not torch.isfinite(loss):
            raise FloatingPointError(f"端到端训练出现 NaN/Inf，epoch={epoch}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(
                f"    [FineTune] epoch={epoch:04d}, "
                f"loss={loss_val:.6e}, "
                f"loss_s={float(loss_s.detach().cpu()):.6e}, "
                f"loss_reg={float(loss_reg.detach().cpu()):.6e}"
            )

    model.load_state_dict(best_state)
    return model
