# -*- coding: utf-8 -*-
"""等效电路参数到 RLGC、ABCD 和 Network 的转换。"""

from typing import Dict, Tuple

import numpy as np
import skrf as rf

from .devices import DeviceBlock
from .utils import network_from_abcd


def circuit_params_to_rlgc(
    circuit_params: Dict[str, float],
    freqs_hz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据提参代码中的等效电路公式计算单位长度 RLGC。

    .mat 输出约定：
        R1,R2,R3: Ohm/m 等效公式中的电阻参数
        L1,L2,L3: nH/m，代码中乘 1e-9
        Cox,Csi: pF/m，代码中乘 1e-12
        Rsi: Ohm
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    omega = 2.0 * np.pi * freqs_hz

    R1 = float(circuit_params["R1"])
    R2 = float(circuit_params["R2"])
    R3 = float(circuit_params["R3"])

    L1 = float(circuit_params["L1"]) * 1e-9
    L2 = float(circuit_params["L2"]) * 1e-9
    L3 = float(circuit_params["L3"]) * 1e-9

    Cox = float(circuit_params["Cox"]) * 1e-12
    Csi = float(circuit_params["Csi"]) * 1e-12
    Rsi = float(circuit_params["Rsi"])

    R_RLGC = (
        (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2)
        / ((R1 + R2) ** 2 + omega**2 * L2**2 + 1e-30)
        + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2 + 1e-30)
    )

    L_RLGC = (
        (R1**2 * L2) / ((R1 + R2) ** 2 + omega**2 * L2**2 + 1e-30)
        + L3 * R3**2 / (R3**2 + omega**2 * L3**2 + 1e-30)
        + L1
    )

    G_RLGC = (
        omega**2 * Rsi * Cox**2
        / (1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2 + 1e-30)
    )

    C_RLGC = (
        Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)
    ) / (1.0 + omega**2 * Rsi**2 * (Cox + Csi) ** 2 + 1e-30)

    return (
        np.asarray(R_RLGC, dtype=np.float64),
        np.asarray(L_RLGC, dtype=np.float64),
        np.asarray(G_RLGC, dtype=np.float64),
        np.asarray(C_RLGC, dtype=np.float64),
    )


def rlgc_to_abcd(
    rlgc: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    length_m: float,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """传输线 RLGC 模型转 ABCD。"""
    R, L, G, C = rlgc
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    omega = 2.0 * np.pi * freqs_hz

    Z_series = R + 1j * omega * L
    Y_shunt = G + 1j * omega * C

    Zc = np.sqrt(Z_series / (Y_shunt + 1e-300))
    gamma = np.sqrt(Z_series * Y_shunt)

    gl = gamma * length_m
    A = np.cosh(gl)
    B = Zc * np.sinh(gl)
    C_abcd = (1.0 / (Zc + 1e-300)) * np.sinh(gl)
    D = np.cosh(gl)

    ABCD = np.zeros((len(freqs_hz), 2, 2), dtype=np.complex128)
    ABCD[:, 0, 0] = A
    ABCD[:, 0, 1] = B
    ABCD[:, 1, 0] = C_abcd
    ABCD[:, 1, 1] = D
    return ABCD


def block_to_abcd(block: DeviceBlock, freqs_hz: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    if block.rlgc is None:
        raise ValueError(f"{block.name} 尚未计算 RLGC")
    return rlgc_to_abcd(block.rlgc, block.length_m * length_scale, freqs_hz)


def block_to_network(block: DeviceBlock, freqs_hz: np.ndarray, length_scale: float = 1.0) -> rf.Network:
    return network_from_abcd(freqs_hz, block_to_abcd(block, freqs_hz, length_scale), name=block.name)
