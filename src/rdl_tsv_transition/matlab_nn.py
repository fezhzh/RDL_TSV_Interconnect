# -*- coding: utf-8 -*-
"""调用 MATLAB 导出的 .mat 神经网络并生成器件 RLGC。"""

import os
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio

from .circuit import circuit_params_to_rlgc
from .constants import CIRCUIT_PARAM_NAMES, MAT_PREFIX
from .devices import DeviceBlock


def _as_row(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(1, -1)


def _matlab_layer_forward(a: np.ndarray, W: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
    """
    MATLAB 导出时使用：
        w1 = net.iw{1,1}'
        theta1 = net.b{1}'
        output = tansig(tansig(input*w1 + theta1)*w2 + theta2)*w3 + theta3

    因此通常 W.shape = [in_features, out_features]。
    这里额外兼容 W 被保存成转置的情况。
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(1, -1)

    if W.shape[0] == a.shape[1]:
        z = a @ W
    elif W.shape[1] == a.shape[1]:
        z = a @ W.T
    else:
        raise ValueError(f"{name} 维度不匹配: input={a.shape}, W={W.shape}, b={b.shape}")

    if b.shape[1] != z.shape[1]:
        if b.shape[0] == z.shape[1]:
            b = b.T
        else:
            raise ValueError(f"{name} bias 维度不匹配: z={z.shape}, b={b.shape}")

    return z + b


def predict_one_matlab_nn(features: np.ndarray, mat_filepath: str) -> float:
    """调用单个 .mat 文件中的 MATLAB 神经网络。"""
    if not os.path.exists(mat_filepath):
        raise FileNotFoundError(f"缺少本地神经网络参数文件: {mat_filepath}")

    data = sio.loadmat(mat_filepath)
    required = ["psmin", "psmax", "w1", "theta1", "w2", "theta2", "w3", "theta3", "outputmax", "outputmin"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{mat_filepath} 缺少字段: {missing}")

    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    psmin = _as_row(data["psmin"])
    psmax = _as_row(data["psmax"])

    if psmin.shape[1] != x.shape[1]:
        raise ValueError(f"{mat_filepath} 输入维度不匹配: features={x.shape}, psmin={psmin.shape}")

    x_norm = 2.0 * (x - psmin) / (psmax - psmin + 1e-30) - 1.0

    a1 = np.tanh(_matlab_layer_forward(x_norm, data["w1"], data["theta1"], "layer1"))
    a2 = np.tanh(_matlab_layer_forward(a1, data["w2"], data["theta2"], "layer2"))
    y_norm = _matlab_layer_forward(a2, data["w3"], data["theta3"], "layer3")

    output_min = float(np.asarray(data["outputmin"]).reshape(-1)[0])
    output_max = float(np.asarray(data["outputmax"]).reshape(-1)[0])
    y = output_min + (y_norm + 1.0) * (output_max - output_min) / 2.0

    return float(np.asarray(y).reshape(-1)[0])


def predict_circuit_parameters(features: np.ndarray, mat_dir: str, prefix: str) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for name in CIRCUIT_PARAM_NAMES:
        mat_filepath = os.path.join(mat_dir, f"{prefix}{name}.mat")
        params[name] = predict_one_matlab_nn(features, mat_filepath)
    return params


def attach_circuit_params_to_blocks(blocks: List[DeviceBlock], freqs_hz: np.ndarray, mat_dir: str) -> None:
    """给每个器件块预测等效电路参数，并计算对应的单位长度 RLGC。"""
    cache: Dict[
        Tuple[str, Tuple[float, ...]],
        Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ] = {}

    for block in blocks:
        key = (block.kind, tuple(np.asarray(block.features, dtype=np.float64).tolist()))
        if key in cache:
            cp, rlgc = cache[key]
        else:
            cp = predict_circuit_parameters(block.features, mat_dir, MAT_PREFIX[block.kind])
            rlgc = circuit_params_to_rlgc(cp, freqs_hz)
            cache[key] = (cp, rlgc)

        block.circuit_params = cp
        block.rlgc = rlgc
