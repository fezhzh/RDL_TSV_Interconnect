# -*- coding: utf-8 -*-
"""关键中间结果保存，便于后续分析和复用。"""

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import skrf as rf
import torch

from .devices import DeviceBlock
from .model import Normalizer, TransitionElementNN


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _block_metadata(block: DeviceBlock) -> Dict[str, object]:
    rlgc_shapes = None
    if block.rlgc is not None:
        rlgc_shapes = [list(arr.shape) for arr in block.rlgc]
    return {
        "name": block.name,
        "kind": block.kind,
        "index": block.index,
        "length_um": block.length_um,
        "length_m": block.length_m,
        "features": block.features,
        "geom5": block.geom5,
        "circuit_params": block.circuit_params,
        "rlgc_shapes": rlgc_shapes,
    }


def _network_s(nw: rf.Network) -> np.ndarray:
    return np.asarray(nw.s, dtype=np.complex128)


def save_structure_sample(sample, out_dir: str) -> Optional[str]:
    """保存单个 DUT 的样本准备结果。sample 使用 duck typing，避免和 dataset.py 循环导入。"""
    if not out_dir:
        return None

    dut_dir = ensure_dir(os.path.join(out_dir, "intermediate", f"dut{sample.idx:03d}"))

    meta = {
        "idx": sample.idx,
        "s2p_file": sample.s2p_file,
        "header_params": sample.header_params,
        "blocks": [_block_metadata(block) for block in sample.blocks],
    }
    with open(os.path.join(dut_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

    arrays = {
        "freqs_hz": sample.freqs_hz,
        "hfss_s": _network_s(sample.hfss_nw),
        "direct_full_s": _network_s(sample.direct_full_nw),
        "extracted_transition_s": _network_s(sample.extracted_transition_nw),
        "X_raw": sample.X_raw,
        "Y_raw": sample.Y_raw,
    }
    for i, arr in enumerate(sample.full_abcds):
        arrays[f"full_abcd_{i:02d}"] = arr
    for i, arr in enumerate(sample.shortened_abcds):
        arrays[f"shortened_abcd_{i:02d}"] = arr
    for i, arr in enumerate(sample.transition_values_extracted):
        arrays[f"transition_values_extracted_{i:02d}"] = arr
    for i, block in enumerate(sample.blocks):
        if block.rlgc is None:
            continue
        arrays[f"block_{i:02d}_R"] = block.rlgc[0]
        arrays[f"block_{i:02d}_L"] = block.rlgc[1]
        arrays[f"block_{i:02d}_G"] = block.rlgc[2]
        arrays[f"block_{i:02d}_C"] = block.rlgc[3]

    np.savez_compressed(os.path.join(dut_dir, "sample_arrays.npz"), **arrays)
    return dut_dir


def save_training_dataset(X_all: np.ndarray, Y_all: np.ndarray, samples: Sequence, out_dir: str) -> Optional[str]:
    if not out_dir:
        return None

    dataset_dir = ensure_dir(os.path.join(out_dir, "intermediate", "dataset"))
    np.savez_compressed(
        os.path.join(dataset_dir, "transition_training_dataset.npz"),
        X_all=X_all,
        Y_all=Y_all,
        sample_indices=np.array([sample.idx for sample in samples], dtype=np.int64),
        sample_row_counts=np.array([sample.X_raw.shape[0] for sample in samples], dtype=np.int64),
    )
    return dataset_dir


def save_normalizer(normalizer: Normalizer, out_dir: str) -> Optional[str]:
    if not out_dir:
        return None

    model_dir = ensure_dir(os.path.join(out_dir, "intermediate", "models"))
    path = os.path.join(model_dir, "transition_normalizer.npz")
    np.savez_compressed(path, x_mean=normalizer.x_mean, x_std=normalizer.x_std, y_mean=normalizer.y_mean, y_std=normalizer.y_std)
    return path


def save_model_checkpoint(
    model: TransitionElementNN,
    normalizer: Normalizer,
    out_dir: str,
    name: str,
    extra: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    if not out_dir:
        return None

    model_dir = ensure_dir(os.path.join(out_dir, "intermediate", "models"))
    path = os.path.join(model_dir, f"{name}.pth")
    payload = {
        "model_state_dict": model.state_dict(),
        "normalizer": {
            "x_mean": normalizer.x_mean,
            "x_std": normalizer.x_std,
            "y_mean": normalizer.y_mean,
            "y_std": normalizer.y_std,
        },
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def save_evaluation_result(
    sample,
    compare_networks: Dict[str, rf.Network],
    values_nn_supervised: np.ndarray,
    values_nn_fine: np.ndarray,
    mse_row: Dict[str, float],
    out_dir: str,
) -> Optional[str]:
    if not out_dir:
        return None

    dut_dir = ensure_dir(os.path.join(out_dir, "intermediate", f"dut{sample.idx:03d}"))
    arrays = {
        "freqs_hz": sample.freqs_hz,
        "transition_values_nn_supervised": values_nn_supervised,
        "transition_values_nn_fine_tuned": values_nn_fine,
    }
    for name, nw in compare_networks.items():
        key = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        arrays[f"{key}_s"] = _network_s(nw)
    np.savez_compressed(os.path.join(dut_dir, "evaluation_arrays.npz"), **arrays)

    with open(os.path.join(dut_dir, "mse.json"), "w", encoding="utf-8") as f:
        json.dump(mse_row, f, ensure_ascii=False, indent=2, default=_json_default)
    return dut_dir


def save_mse_summary(rows: Sequence[Dict[str, float]], out_dir: str) -> Optional[str]:
    if not out_dir:
        return None

    summary_dir = ensure_dir(os.path.join(out_dir, "intermediate", "dataset"))
    path = os.path.join(summary_dir, "mse_summary.csv")
    if not rows:
        return path

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_loss_history(history: Dict[str, Sequence[float]], out_dir: str, name: str) -> Optional[str]:
    """保存训练 loss 历史为 CSV。"""
    if not out_dir or not history:
        return None

    loss_dir = ensure_dir(os.path.join(out_dir, "intermediate", "loss_curves"))
    path = os.path.join(loss_dir, f"{name}.csv")
    keys = list(history.keys())
    n_rows = max((len(history[key]) for key in keys), default=0)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(n_rows):
            row = {}
            for key in keys:
                values = history[key]
                row[key] = values[i] if i < len(values) else ""
            writer.writerow(row)
    return path


def save_error_analysis(analysis: Dict[str, object], out_dir: str) -> Optional[str]:
    """保存误差分析 JSON 和 Markdown 报告。"""
    if not out_dir:
        return None

    dataset_dir = ensure_dir(os.path.join(out_dir, "intermediate", "dataset"))
    json_path = os.path.join(dataset_dir, "error_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, default=_json_default)

    md_path = os.path.join(dataset_dir, "error_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Error Analysis\n\n")
        best_model = analysis.get("best_model_by_mean_mse")
        if best_model:
            f.write(f"Best model by mean MSE: `{best_model}`\n\n")

        stats = analysis.get("model_stats", {})
        if stats:
            f.write("## Model Statistics\n\n")
            f.write("| Model | Mean | Std | Min | Max |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            for name, row in stats.items():
                f.write(
                    f"| {name} | {row['mean']:.6e} | {row['std']:.6e} | "
                    f"{row['min']:.6e} | {row['max']:.6e} |\n"
                )
            f.write("\n")

        ranking = analysis.get("sample_ranking_by_final_mse", [])
        if ranking:
            f.write("## Worst Samples By Final MSE\n\n")
            f.write("| Rank | DUT | MSE |\n")
            f.write("| ---: | ---: | ---: |\n")
            for rank, item in enumerate(ranking, start=1):
                f.write(f"| {rank} | dut{item['idx']} | {item['mse']:.6e} |\n")
            f.write("\n")

        recs = analysis.get("recommendations", [])
        if recs:
            f.write("## Recommendations\n\n")
            for i, text in enumerate(recs, start=1):
                f.write(f"{i}. {text}\n")

    return md_path
