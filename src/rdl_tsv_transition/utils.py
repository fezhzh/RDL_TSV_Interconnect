# -*- coding: utf-8 -*-
"""路径、Network 和矩阵转换工具。"""

import os
from typing import Optional, Sequence

import numpy as np
import skrf as rf
import torch

from .constants import Z_REF


def script_base_dir() -> str:
    """兼容 .py 脚本和 notebook/交互环境。"""
    if "__file__" in globals():
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.getcwd()


def as_abs_path(path: str, base_dir: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = script_base_dir()
    return os.path.abspath(os.path.join(base_dir, path))


def frequency_from_hz(freqs_hz: np.ndarray) -> rf.Frequency:
    return rf.Frequency.from_f(np.asarray(freqs_hz, dtype=float), unit="Hz")


def network_from_s(freqs_hz: np.ndarray, s: np.ndarray, name: str) -> rf.Network:
    return rf.Network(frequency=frequency_from_hz(freqs_hz), s=s, z0=Z_REF, name=name)


def network_from_abcd(freqs_hz: np.ndarray, abcd: np.ndarray, name: str) -> rf.Network:
    return network_from_s(freqs_hz, abcd2s_np(abcd, Z0=Z_REF), name=name)


def load_hfss_network(s2p_file: str, max_points: Optional[int] = None) -> rf.Network:
    nw = rf.Network(s2p_file)
    if max_points is None:
        nw.name = os.path.basename(s2p_file)
        return nw

    freqs = nw.f[:max_points]
    s = nw.s[:max_points]
    return network_from_s(freqs, s, name=os.path.basename(s2p_file))


def s2abcd_np(S: np.ndarray, Z0: float = Z_REF) -> np.ndarray:
    S = np.asarray(S, dtype=np.complex128)
    S11, S12 = S[:, 0, 0], S[:, 0, 1]
    S21, S22 = S[:, 1, 0], S[:, 1, 1]

    denom = 2.0 * S21 + 1e-30
    A = ((1 + S11) * (1 - S22) + S12 * S21) / denom
    B = Z0 * ((1 + S11) * (1 + S22) - S12 * S21) / denom
    C = (1.0 / Z0) * ((1 - S11) * (1 - S22) - S12 * S21) / denom
    D = ((1 - S11) * (1 + S22) + S12 * S21) / denom

    ABCD = np.zeros_like(S, dtype=np.complex128)
    ABCD[:, 0, 0] = A
    ABCD[:, 0, 1] = B
    ABCD[:, 1, 0] = C
    ABCD[:, 1, 1] = D
    return ABCD


def abcd2s_np(ABCD: np.ndarray, Z0: float = Z_REF) -> np.ndarray:
    ABCD = np.asarray(ABCD, dtype=np.complex128)
    A, B = ABCD[:, 0, 0], ABCD[:, 0, 1]
    C, D = ABCD[:, 1, 0], ABCD[:, 1, 1]

    denom = A + B / Z0 + C * Z0 + D + 1e-30
    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S21 = 2.0 / denom
    S22 = (-A + B / Z0 - C * Z0 + D) / denom

    S = np.zeros_like(ABCD, dtype=np.complex128)
    S[:, 0, 0] = S11
    S[:, 0, 1] = S12
    S[:, 1, 0] = S21
    S[:, 1, 1] = S22
    return S


def cascade_abcd_np(abcd_list: Sequence[np.ndarray]) -> np.ndarray:
    if len(abcd_list) == 0:
        raise ValueError("abcd_list 不能为空")

    out = np.array(abcd_list[0], dtype=np.complex128, copy=True)
    for abcd in abcd_list[1:]:
        out = np.matmul(out, abcd)
    return out


def abcd2s_torch(ABCD: torch.Tensor, Z0: float = Z_REF) -> torch.Tensor:
    A, B = ABCD[:, 0, 0], ABCD[:, 0, 1]
    C, D = ABCD[:, 1, 0], ABCD[:, 1, 1]

    denom = A + B / Z0 + C * Z0 + D + 1e-30
    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S21 = 2.0 / denom
    S22 = (-A + B / Z0 - C * Z0 + D) / denom

    S = torch.zeros_like(ABCD)
    S[:, 0, 0] = S11
    S[:, 0, 1] = S12
    S[:, 1, 0] = S21
    S[:, 1, 1] = S22
    return S
