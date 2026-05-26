# -*- coding: utf-8 -*-
"""器件块定义和几何参数组装。"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import DEVICE_SEQUENCE


def require_keys(params: Dict[str, float], keys: Sequence[str], context: str = "") -> None:
    missing = [k for k in keys if k not in params or params[k] is None]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}缺少必要参数 {missing}")


@dataclass
class DeviceBlock:
    kind: str
    index: int
    length_um: float
    features: np.ndarray
    geom5: np.ndarray
    circuit_params: Optional[Dict[str, float]] = None
    rlgc: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.index}"

    @property
    def length_m(self) -> float:
        return float(self.length_um) * 1e-6


def make_device_block(kind: str, index: int, params: Dict[str, float]) -> DeviceBlock:
    """根据整体结构 s2p 头部参数构造单个器件块。"""
    require_keys(params, ["htsv", "p1"], context=kind)

    if kind == "RDL_Top":
        require_keys(params, ["lrdl", "wrdl", "trdl"], context=kind)
        features = np.array(
            [params["lrdl"], params["wrdl"], params["trdl"], params["htsv"], params["p1"]],
            dtype=np.float64,
        )
        length_um = params["lrdl"]
        geom5 = features.copy()

    elif kind == "RDL_Bottom":
        require_keys(params, ["ldown", "wdown", "tdown"], context=kind)
        features = np.array(
            [params["ldown"], params["wdown"], params["tdown"], params["htsv"], params["p1"]],
            dtype=np.float64,
        )
        length_um = params["ldown"]
        geom5 = features.copy()

    elif kind == "TSV":
        require_keys(params, ["dtsv"], context=kind)
        features = np.array([params["dtsv"], params["htsv"], params["p1"]], dtype=np.float64)
        length_um = params["htsv"]
        geom5 = np.array([params["dtsv"], params["htsv"], params["p1"], 0.0, 0.0], dtype=np.float64)

    else:
        raise ValueError(f"未知器件类型: {kind}")

    return DeviceBlock(kind=kind, index=index, length_um=length_um, features=features, geom5=geom5)


def build_structure_blocks(params: Dict[str, float]) -> List[DeviceBlock]:
    return [make_device_block(kind, i, params) for i, kind in enumerate(DEVICE_SEQUENCE)]


def shortened_length_scales(n_blocks: int) -> List[float]:
    """
    过渡结构插入后：
    - 首尾 RDL 只有一个连接端参与过渡，保留 0.9*Length；
    - 中间器件左右两端都参与过渡，保留 0.8*Length。
    """
    if n_blocks < 2:
        raise ValueError("至少需要两个器件块")
    return [0.9 if i in (0, n_blocks - 1) else 0.8 for i in range(n_blocks)]
