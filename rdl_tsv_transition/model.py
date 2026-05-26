# -*- coding: utf-8 -*-
"""过渡结构神经网络、特征构造、监督训练和预测。"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import KIND_TO_ONEHOT
from .devices import DeviceBlock


@dataclass
class Normalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def transition_input_vector(left: DeviceBlock, right: DeviceBlock, freq_hz: float) -> np.ndarray:
    """
    输入特征：
        left_type_onehot(3), right_type_onehot(3), left_geom5, right_geom5, freq_GHz
    共 17 维。
    """
    return np.concatenate(
        [
            KIND_TO_ONEHOT[left.kind],
            KIND_TO_ONEHOT[right.kind],
            left.geom5.astype(np.float64),
            right.geom5.astype(np.float64),
            np.array([float(freq_hz) / 1e9], dtype=np.float64),
        ]
    )


def build_transition_training_data(
    blocks: List[DeviceBlock],
    freqs_hz: np.ndarray,
    transition_values: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []

    for i in range(len(blocks) - 1):
        left, right = blocks[i], blocks[i + 1]
        values_i = np.asarray(transition_values[i], dtype=np.float64)
        for k, f in enumerate(freqs_hz):
            X_rows.append(transition_input_vector(left, right, float(f)))
            Y_rows.append(values_i[k])

    X = np.vstack(X_rows).astype(np.float64)
    Y = np.vstack(Y_rows).astype(np.float64)
    return X, Y


class TransitionElementNN(nn.Module):
    """过渡结构元件值网络。输出是标准化后的 log 元件值。"""

    def __init__(self, in_features: int = 17, out_features: int = 6, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_normalizer(X_raw: np.ndarray, Y_raw: np.ndarray) -> Tuple[Normalizer, np.ndarray, np.ndarray]:
    X_raw = np.asarray(X_raw, dtype=np.float64)
    Y_raw = np.asarray(Y_raw, dtype=np.float64)

    logY = np.log(np.maximum(Y_raw, 1e-300))

    x_mean = X_raw.mean(axis=0, keepdims=True)
    x_std = X_raw.std(axis=0, keepdims=True) + 1e-12
    y_mean = logY.mean(axis=0, keepdims=True)
    y_std = logY.std(axis=0, keepdims=True) + 1e-12

    X_norm = (X_raw - x_mean) / x_std
    Y_norm = (logY - y_mean) / y_std

    return Normalizer(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std), X_norm, Y_norm


def train_supervised_transition_nn(
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    epochs: int = 1000,
    lr: float = 2e-3,
    hidden: int = 128,
    batch_size: int = 8192,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[TransitionElementNN, Normalizer, Dict[str, List[float]]]:
    """
    用多个 DUT 合并后的过渡结构提取值监督训练共享 NN。

    输出目标使用 log 元件值并标准化，避免 R/L/C/G 量纲差异过大。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer, X_norm, Y_norm = make_normalizer(X_raw, Y_raw)

    X_t = torch.tensor(X_norm, dtype=torch.float64, device=device)
    Y_t = torch.tensor(Y_norm, dtype=torch.float64, device=device)

    model = TransitionElementNN(in_features=X_t.shape[1], out_features=Y_t.shape[1], hidden=hidden)
    model = model.to(device=device, dtype=torch.float64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    n_samples = X_t.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size > n_samples:
        batch_size = n_samples

    best_loss = float("inf")
    best_state = None
    history: Dict[str, List[float]] = {"epoch": [], "loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        epoch_loss_sum = 0.0
        epoch_count = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_t[idx]
            yb = Y_t[idx]

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = int(xb.shape[0])
            epoch_loss_sum += float(loss.detach().cpu()) * bs
            epoch_count += bs

        loss_val = epoch_loss_sum / max(epoch_count, 1)
        history["epoch"].append(float(epoch))
        history["loss"].append(float(loss_val))

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"    [Supervised] epoch={epoch:04d}, loss={loss_val:.6e}, samples={n_samples}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, normalizer, history


def predict_transition_values_np(
    model: TransitionElementNN,
    normalizer: Normalizer,
    blocks: List[DeviceBlock],
    freqs_hz: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device

    n_trans = len(blocks) - 1
    n_freq = len(freqs_hz)

    X_rows: List[np.ndarray] = []
    for i in range(n_trans):
        for f in freqs_hz:
            X_rows.append(transition_input_vector(blocks[i], blocks[i + 1], float(f)))
    X_raw = np.vstack(X_rows).astype(np.float64)
    X_norm = (X_raw - normalizer.x_mean) / normalizer.x_std

    X_t = torch.tensor(X_norm, dtype=torch.float64, device=device)
    y_mean_t = torch.tensor(normalizer.y_mean, dtype=torch.float64, device=device)
    y_std_t = torch.tensor(normalizer.y_std, dtype=torch.float64, device=device)

    model.eval()
    with torch.no_grad():
        y_norm = model(X_t)
        log_values = y_norm * y_std_t + y_mean_t
        log_values = torch.clamp(log_values, min=-100.0, max=100.0)
        values = torch.exp(log_values).detach().cpu().numpy()

    return values.reshape(n_trans, n_freq, 6)
