# -*- coding: utf-8 -*-
"""评估、画图和结果汇总。"""

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

from .constants import CURVE_STYLES, FALLBACK_LINESTYLES, FALLBACK_MARKERS


def complex_mse(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - pred) ** 2))


def print_mse_table(hfss: rf.Network, networks: Dict[str, rf.Network]) -> None:
    print("\nMSE against HFSS:")
    print("-" * 72)
    for name, nw in networks.items():
        print(f"{name:<35s}: {complex_mse(hfss.s, nw.s):.6e}")
    print("-" * 72)


def print_dataset_mse_summary(rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return

    names = [k for k in rows[0].keys() if k != "idx"]
    print("\n" + "=" * 88)
    print("数据集 MSE 汇总：")
    print("=" * 88)
    print(f"{'Model':<35s} {'Mean MSE':>16s} {'Median MSE':>16s} {'Max MSE':>16s}")
    print("-" * 88)
    for name in names:
        vals = np.array([row[name] for row in rows], dtype=np.float64)
        print(f"{name:<35s} {np.mean(vals):>16.6e} {np.median(vals):>16.6e} {np.max(vals):>16.6e}")
    print("=" * 88)


def plot_loss_history(
    history: Dict[str, Sequence[float]],
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制训练 loss 收敛曲线。"""
    epochs = np.asarray(history.get("epoch", []), dtype=np.float64)
    loss_keys = [key for key in history.keys() if key != "epoch"]
    if epochs.size == 0 or not loss_keys:
        return

    plt.figure(figsize=(9, 5.5))
    for key in loss_keys:
        values = np.asarray(history[key], dtype=np.float64)
        if values.size != epochs.size:
            continue
        plt.plot(epochs, values, label=key, linewidth=2.0)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    positive_values = [
        np.asarray(history[key], dtype=np.float64)
        for key in loss_keys
        if np.all(np.asarray(history[key], dtype=np.float64) > 0)
    ]
    if positive_values:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss 曲线已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_mse_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    """对所有 sample 的 MSE 做统计、排序和改进建议。"""
    if not rows:
        return {"model_stats": {}, "sample_ranking": [], "recommendations": ["没有可分析的 MSE 数据。"]}

    model_names = [key for key in rows[0].keys() if key != "idx"]
    stats: Dict[str, Dict[str, float]] = {}
    for name in model_names:
        values = np.array([row[name] for row in rows], dtype=np.float64)
        stats[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    final_name = "NN fine-tuned transition" if "NN fine-tuned transition" in model_names else model_names[-1]
    sample_ranking = sorted(
        [{"idx": int(row["idx"]), "mse": float(row[final_name])} for row in rows],
        key=lambda item: item["mse"],
        reverse=True,
    )

    best_by_mean = min(model_names, key=lambda name: stats[name]["mean"])
    recommendations: List[str] = []

    direct = stats.get("Direct full cascade")
    extracted = stats.get("Extracted transition")
    supervised = stats.get("NN supervised transition")
    fine = stats.get("NN fine-tuned transition")

    if fine and supervised:
        improvement = (supervised["mean"] - fine["mean"]) / max(supervised["mean"], 1e-30)
        if improvement > 0.05:
            recommendations.append(
                f"端到端微调相对监督预训练平均 MSE 降低 {improvement:.2%}，建议保留 HFSS 微调阶段。"
            )
        elif improvement > 0:
            recommendations.append(
                f"端到端微调仅相对监督预训练平均 MSE 降低 {improvement:.2%}，可增加 fine_epochs 或调大 fine_reg_weight 网格搜索。"
            )
        else:
            recommendations.append(
                f"端到端微调未改善平均 MSE（变化 {improvement:.2%}），建议降低 fine_lr、提高 fine_reg_weight，或检查 HFSS 目标与提参模型的一致性。"
            )

    if extracted and direct:
        extracted_gain = (direct["mean"] - extracted["mean"]) / max(direct["mean"], 1e-30)
        if extracted_gain < 0.01:
            recommendations.append(
                "提取过渡结构相对直接级联改善很小，说明 0.1*Length RLGC 过渡假设可能不足；建议引入连接处几何特征或增加可学习的耦合/寄生项。"
            )

    if fine and direct:
        nn_gain = (direct["mean"] - fine["mean"]) / max(direct["mean"], 1e-30)
        if nn_gain > 0:
            recommendations.append(f"最终 NN 相对直接级联平均 MSE 降低 {nn_gain:.2%}，当前学习型过渡模型有效。")
        else:
            recommendations.append(
                "最终 NN 未优于直接级联，建议先减少模型自由度或只做监督预训练，避免微调把物理先验破坏。"
            )

    if sample_ranking:
        worst = sample_ranking[0]
        recommendations.append(
            f"误差最大的样本是 dut{worst['idx']}，建议优先检查该样本的 s2p 头部几何参数、HFSS 端口定义和频段内异常点。"
        )

    if len(rows) < 5:
        recommendations.append("当前样本数较少，建议增加不同尺寸 DUT，避免共享 NN 只记住少量几何组合。")

    recommendations.append("建议按频段分别统计 S11/S21 幅度和相位误差，用于判断误差主要来自反射、插损还是相位延迟。")
    recommendations.append("可对 hidden、supervised_epochs、fine_epochs、fine_lr、fine_reg_weight 做小规模网格搜索，并以验证 DUT 的 MSE 选择参数。")

    return {
        "model_stats": stats,
        "best_model_by_mean_mse": best_by_mean,
        "sample_ranking_by_final_mse": sample_ranking,
        "final_model_name": final_name,
        "recommendations": recommendations,
    }


def print_error_analysis(analysis: Dict[str, object]) -> None:
    """打印误差分析和改进建议。"""
    print("\n" + "=" * 88)
    print("误差分析与改进建议：")
    print("=" * 88)
    best_model = analysis.get("best_model_by_mean_mse")
    if best_model:
        print(f"平均 MSE 最优模型: {best_model}")

    stats = analysis.get("model_stats", {})
    if stats:
        print("\n各模型误差统计：")
        print(f"{'Model':<35s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
        print("-" * 88)
        for name, row in stats.items():
            print(
                f"{name:<35s} "
                f"{row['mean']:>12.4e} {row['std']:>12.4e} "
                f"{row['min']:>12.4e} {row['max']:>12.4e}"
            )

    ranking = analysis.get("sample_ranking_by_final_mse", [])
    if ranking:
        top = ranking[: min(5, len(ranking))]
        print("\n最终模型误差最高的样本：")
        for item in top:
            print(f"  dut{item['idx']}: MSE={item['mse']:.6e}")

    recs = analysis.get("recommendations", [])
    if recs:
        print("\n改进建议：")
        for i, text in enumerate(recs, start=1):
            print(f"  {i}. {text}")
    print("=" * 88)


def _curve_style(name: str, idx: int, n_points: int) -> Dict[str, object]:
    """为每条曲线生成稳定的线型/标记，保证对比图中曲线类型不同。"""
    style = dict(CURVE_STYLES.get(name, {}))
    if not style:
        style = {
            "linestyle": FALLBACK_LINESTYLES[idx % len(FALLBACK_LINESTYLES)],
            "marker": FALLBACK_MARKERS[idx % len(FALLBACK_MARKERS)],
            "linewidth": 2.0,
        }

    marker = style.get("marker", None)
    if marker is not None:
        style.setdefault("markersize", 4)
        style.setdefault("markevery", max(1, n_points // 25))
    return style


def _plot_trace(ax, freq_ghz: np.ndarray, y: np.ndarray, label: str, idx: int) -> None:
    ax.plot(freq_ghz, y, label=label, **_curve_style(label, idx, len(freq_ghz)))


def plot_s_comparison(
    hfss: rf.Network,
    networks: Dict[str, rf.Network],
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    freq_ghz = hfss.f / 1e9

    plt.figure(figsize=(14, 9))
    curves = [("HFSS", hfss)] + list(networks.items())

    ax = plt.subplot(2, 2, 1)
    for idx, (name, nw) in enumerate(curves):
        y = 20 * np.log10(np.maximum(np.abs(nw.s[:, 0, 0]), 1e-30))
        _plot_trace(ax, freq_ghz, y, name, idx)
    ax.set_title(f"S11 Magnitude {title_suffix}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("dB")
    ax.grid(True)
    ax.legend()

    ax = plt.subplot(2, 2, 2)
    for idx, (name, nw) in enumerate(curves):
        y = 20 * np.log10(np.maximum(np.abs(nw.s[:, 1, 0]), 1e-30))
        _plot_trace(ax, freq_ghz, y, name, idx)
    ax.set_title(f"S21 Magnitude {title_suffix}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("dB")
    ax.grid(True)
    ax.legend()

    ax = plt.subplot(2, 2, 3)
    for idx, (name, nw) in enumerate(curves):
        y = np.unwrap(np.angle(nw.s[:, 0, 0])) * 180 / np.pi
        _plot_trace(ax, freq_ghz, y, name, idx)
    ax.set_title(f"S11 Phase {title_suffix}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("deg")
    ax.grid(True)
    ax.legend()

    ax = plt.subplot(2, 2, 4)
    for idx, (name, nw) in enumerate(curves):
        y = np.unwrap(np.angle(nw.s[:, 1, 0])) * 180 / np.pi
        _plot_trace(ax, freq_ghz, y, name, idx)
    ax.set_title(f"S21 Phase {title_suffix}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("deg")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
