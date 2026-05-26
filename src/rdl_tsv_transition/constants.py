# -*- coding: utf-8 -*-
"""全局常量和模型约定。"""

import numpy as np

Z_REF = 50.0

CIRCUIT_PARAM_NAMES = [
    "R1", "R2", "R3",
    "L1", "L2", "L3",
    "Cox", "Csi", "Rsi",
]

DEVICE_SEQUENCE = [
    "RDL_Top", "TSV", "RDL_Bottom", "TSV",
    "RDL_Top", "TSV", "RDL_Bottom", "TSV",
    "RDL_Top", "TSV", "RDL_Bottom", "TSV",
    "RDL_Top",
]

MAT_PREFIX = {
    "RDL_Top": "RDL_Top_",
    "RDL_Bottom": "RDL_Bottom_",
    "TSV": "TSV_",
}

KIND_ORDER = ["RDL_Top", "RDL_Bottom", "TSV"]
KIND_TO_ONEHOT = {
    kind: np.eye(len(KIND_ORDER), dtype=np.float64)[i]
    for i, kind in enumerate(KIND_ORDER)
}

# 过渡结构 NN 输出顺序：
# Port1 -- L1 -- R1 -- node -- L2 -- R2 -- Port2
#                              |
#                             C1 || G1
#                              |
#                             GND
TRANSITION_VALUE_NAMES = ["L1", "R1", "L2", "R2", "C1", "G1"]

CURVE_STYLES = {
    "HFSS": {"color": "black", "linestyle": "-", "linewidth": 3.0, "marker": None},
    "Direct full cascade": {"linestyle": ":", "linewidth": 2.2, "marker": None},
    "Extracted transition": {"linestyle": "--", "linewidth": 2.0, "marker": None},
    "NN supervised transition": {"linestyle": "-.", "linewidth": 2.0, "marker": "o"},
    "NN fine-tuned transition": {"linestyle": (0, (5, 1, 1, 1)), "linewidth": 2.2, "marker": "s"},
}

FALLBACK_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
FALLBACK_MARKERS = [None, "o", "s", "^", "D", "x"]
