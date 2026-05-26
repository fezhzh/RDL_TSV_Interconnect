# -*- coding: utf-8 -*-
"""S 参数文件输入和头部参数解析。"""

import re
from typing import Dict


def parse_s2p_header_params(filepath: str) -> Dict[str, float]:
    """读取 s2p 文件开头注释行中的 key=value 参数。"""
    params: Dict[str, float] = {}
    number_re = re.compile(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("#"):
                break
            if not line.startswith("!"):
                continue

            line = line[1:].strip()
            if "=" not in line:
                continue

            key, val = line.split("=", 1)
            match = number_re.search(val.strip())
            if match:
                params[key.strip()] = float(match.group(1))

    return params
