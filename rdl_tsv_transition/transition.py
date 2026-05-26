# -*- coding: utf-8 -*-
"""过渡结构元件提取、ABCD 构造和级联。"""

from typing import List, Sequence

import numpy as np

from .devices import DeviceBlock


def _positive(x: np.ndarray, floor: float) -> np.ndarray:
    """过渡结构元件值需要为正。对很小/负的数做保护。"""
    arr = np.asarray(np.real(x), dtype=np.float64)
    return np.maximum(arr, floor)


def transition_values_from_blocks(left: DeviceBlock, right: DeviceBlock) -> np.ndarray:
    """
    由左右器件单位长度 RLGC 提取过渡结构元件值。

    输出 shape = [n_freq, 6]，顺序：
        [L1, R1, L2, R2, C1, G1]
    """
    if left.rlgc is None or right.rlgc is None:
        raise ValueError("左右器件必须先计算 RLGC")

    R_l, L_l, G_l, C_l = left.rlgc
    R_r, L_r, G_r, C_r = right.rlgc

    left_len = 0.1 * left.length_m
    right_len = 0.1 * right.length_m

    L1 = _positive(L_l * left_len, 1e-24)
    R1 = _positive(R_l * left_len, 1e-12)
    L2 = _positive(L_r * right_len, 1e-24)
    R2 = _positive(R_r * right_len, 1e-12)
    C1 = _positive(C_l * left_len + C_r * right_len, 1e-24)
    G1 = _positive(G_l * left_len + G_r * right_len, 1e-18)

    return np.stack([L1, R1, L2, R2, C1, G1], axis=1)


def transition_abcd_from_values(values: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """由过渡结构元件值构造 ABCD。values shape=[n_freq, 6]。"""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"values 应为 [n_freq, 6]，实际 {values.shape}")

    L1, R1, L2, R2, C1, G1 = [values[:, i] for i in range(6)]
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)

    Z1 = R1 + 1j * omega * L1
    Z2 = R2 + 1j * omega * L2
    Y = G1 + 1j * omega * C1

    A = 1.0 + Z1 * Y
    B = Z1 + Z2 + Z1 * Z2 * Y
    C_abcd = Y
    D = 1.0 + Z2 * Y

    ABCD = np.zeros((len(freqs_hz), 2, 2), dtype=np.complex128)
    ABCD[:, 0, 0] = A
    ABCD[:, 0, 1] = B
    ABCD[:, 1, 0] = C_abcd
    ABCD[:, 1, 1] = D
    return ABCD


def build_transition_values_for_structure(blocks: List[DeviceBlock]) -> List[np.ndarray]:
    return [transition_values_from_blocks(blocks[i], blocks[i + 1]) for i in range(len(blocks) - 1)]


def cascade_with_transitions_np(
    base_abcds: Sequence[np.ndarray],
    transition_values: Sequence[np.ndarray],
    freqs_hz: np.ndarray,
) -> np.ndarray:
    if len(base_abcds) != len(transition_values) + 1:
        raise ValueError(
            f"base_abcds 数量必须比 transition_values 多 1，"
            f"实际 {len(base_abcds)} vs {len(transition_values)}"
        )

    abcd_curr = np.array(base_abcds[0], dtype=np.complex128, copy=True)
    for i, values in enumerate(transition_values):
        trans_abcd = transition_abcd_from_values(values, freqs_hz)
        abcd_curr = np.matmul(np.matmul(abcd_curr, trans_abcd), base_abcds[i + 1])
    return abcd_curr
