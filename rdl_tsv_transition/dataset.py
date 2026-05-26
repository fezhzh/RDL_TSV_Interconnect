# -*- coding: utf-8 -*-
"""多 DUT 数据集准备、共享训练、端到端微调和评估入口。"""

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import skrf as rf
import torch
import torch.nn.functional as F

from .circuit import block_to_abcd
from .constants import TRANSITION_VALUE_NAMES
from .devices import DeviceBlock, build_structure_blocks, shortened_length_scales
from .io import parse_s2p_header_params
from .matlab_nn import attach_circuit_params_to_blocks
from .metrics_plot import (
    analyze_mse_rows,
    complex_mse,
    plot_loss_history,
    plot_s_comparison,
    print_dataset_mse_summary,
    print_error_analysis,
    print_mse_table,
)
from .model import (
    Normalizer,
    TransitionElementNN,
    build_transition_training_data,
    predict_transition_values_np,
    train_supervised_transition_nn,
)
from .persistence import (
    save_evaluation_result,
    save_model_checkpoint,
    save_mse_summary,
    save_normalizer,
    save_error_analysis,
    save_loss_history,
    save_structure_sample,
    save_training_dataset,
)
from .torch_cascade import cascade_with_transition_values_torch
from .transition import build_transition_values_for_structure, cascade_with_transitions_np
from .utils import as_abs_path, cascade_abcd_np, load_hfss_network, network_from_abcd, script_base_dir


@dataclass
class StructureSample:
    """一个整体结构 DUT 样本，包含 HFSS 目标、器件级联基准和过渡结构监督目标。"""

    idx: int
    s2p_file: str
    hfss_nw: rf.Network
    freqs_hz: np.ndarray
    header_params: Dict[str, float]
    blocks: List[DeviceBlock]
    full_abcds: List[np.ndarray]
    shortened_abcds: List[np.ndarray]
    transition_values_extracted: List[np.ndarray]
    X_raw: np.ndarray
    Y_raw: np.ndarray
    direct_full_nw: rf.Network
    extracted_transition_nw: rf.Network


def prepare_structure_sample(
    idx: int,
    s2p_dir_abs: str,
    mat_dir_abs: str,
    max_points: Optional[int] = None,
    verbose: bool = True,
) -> Optional[StructureSample]:
    """
    准备单个 DUT 样本，但不在这里训练 NN。

    完成：HFSS 读取、头部几何参数解析、MATLAB .mat 提参、RLGC/ABCD 级联、
    过渡结构提取，以及监督训练样本 X/Y 生成。
    """
    s2p_file = os.path.join(s2p_dir_abs, f"dut{idx}.s2p")
    if not os.path.exists(s2p_file):
        if verbose:
            print(f"[跳过] 文件不存在: {s2p_file}")
        return None

    if verbose:
        print(f"\n>>> 准备数据样本 dut{idx}: {s2p_file}")

    hfss_nw = load_hfss_network(s2p_file, max_points=max_points)
    hfss_nw.name = "HFSS"
    freqs_hz = np.asarray(hfss_nw.f, dtype=np.float64)

    header_params = parse_s2p_header_params(s2p_file)
    blocks = build_structure_blocks(header_params)

    attach_circuit_params_to_blocks(blocks, freqs_hz, mat_dir_abs)

    full_abcds = [block_to_abcd(block, freqs_hz, length_scale=1.0) for block in blocks]
    direct_full_abcd = cascade_abcd_np(full_abcds)
    direct_full_nw = network_from_abcd(freqs_hz, direct_full_abcd, name="Direct full cascade")

    scales = shortened_length_scales(len(blocks))
    shortened_abcds = [
        block_to_abcd(block, freqs_hz, length_scale=scales[i])
        for i, block in enumerate(blocks)
    ]

    transition_values_extracted = build_transition_values_for_structure(blocks)
    extracted_trans_abcd = cascade_with_transitions_np(shortened_abcds, transition_values_extracted, freqs_hz)
    extracted_transition_nw = network_from_abcd(freqs_hz, extracted_trans_abcd, name="Extracted transition")

    X_raw, Y_raw = build_transition_training_data(blocks, freqs_hz, transition_values_extracted)

    return StructureSample(
        idx=idx,
        s2p_file=s2p_file,
        hfss_nw=hfss_nw,
        freqs_hz=freqs_hz,
        header_params=header_params,
        blocks=blocks,
        full_abcds=full_abcds,
        shortened_abcds=shortened_abcds,
        transition_values_extracted=transition_values_extracted,
        X_raw=X_raw,
        Y_raw=Y_raw,
        direct_full_nw=direct_full_nw,
        extracted_transition_nw=extracted_transition_nw,
    )


def collect_structure_samples(
    start_idx: int,
    end_idx: int,
    s2p_dir: str,
    mat_dir: str,
    max_points: Optional[int] = None,
    verbose: bool = True,
) -> List[StructureSample]:
    base_dir = script_base_dir()
    s2p_dir_abs = as_abs_path(s2p_dir, base_dir)
    mat_dir_abs = as_abs_path(mat_dir, base_dir)

    samples: List[StructureSample] = []
    for idx in range(start_idx, end_idx + 1):
        sample = prepare_structure_sample(
            idx=idx,
            s2p_dir_abs=s2p_dir_abs,
            mat_dir_abs=mat_dir_abs,
            max_points=max_points,
            verbose=verbose,
        )
        if sample is not None:
            samples.append(sample)

    if not samples:
        raise RuntimeError(f"没有找到可用 DUT 样本。请检查 s2p_dir={s2p_dir_abs}, idx={start_idx}..{end_idx}")

    return samples


def evaluate_sample_with_transition_model(
    sample: StructureSample,
    model: TransitionElementNN,
    normalizer: Normalizer,
    name: str,
    device: Optional[torch.device] = None,
) -> Tuple[rf.Network, np.ndarray]:
    """用共享过渡结构 NN 预测某个 DUT 的过渡元件值，并级联得到整体 Network。"""
    if device is None:
        device = next(model.parameters()).device

    transition_values = predict_transition_values_np(
        model=model,
        normalizer=normalizer,
        blocks=sample.blocks,
        freqs_hz=sample.freqs_hz,
        device=device,
    )
    abcd = cascade_with_transitions_np(sample.shortened_abcds, transition_values, sample.freqs_hz)
    nw = network_from_abcd(sample.freqs_hz, abcd, name=name)
    return nw, transition_values


def fine_tune_transition_nn_on_dataset(
    model: TransitionElementNN,
    normalizer: Normalizer,
    samples: Sequence[StructureSample],
    epochs: int = 300,
    lr: float = 2e-4,
    reg_weight: float = 1e-4,
    sample_batch_size: int = 2,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[TransitionElementNN, Dict[str, List[float]]]:
    """
    以多个 DUT 的 HFSS 整体 S 参数为共同目标，端到端微调同一个共享过渡结构 NN。

    loss = mean_over_DUT(MSE(S_pred, S_HFSS))
           + reg_weight * mean_over_DUT(MSE(log_element_pred_norm, log_element_extracted_norm))
    """
    if not samples:
        raise ValueError("samples 不能为空")

    if device is None:
        device = next(model.parameters()).device

    y_mean_t = torch.tensor(normalizer.y_mean, dtype=torch.float64, device=device)
    y_std_t = torch.tensor(normalizer.y_std, dtype=torch.float64, device=device)

    items = []
    for sample in samples:
        X_norm = (sample.X_raw - normalizer.x_mean) / normalizer.x_std
        logY = np.log(np.maximum(sample.Y_raw, 1e-300))
        Y_norm = (logY - normalizer.y_mean) / normalizer.y_std

        item = {
            "idx": sample.idx,
            "n_trans": len(sample.blocks) - 1,
            "n_freq": len(sample.freqs_hz),
            "X_t": torch.tensor(X_norm, dtype=torch.float64, device=device),
            "Y_t": torch.tensor(Y_norm, dtype=torch.float64, device=device),
            "omega_t": torch.tensor(2.0 * np.pi * sample.freqs_hz, dtype=torch.float64, device=device),
            "target_s_t": torch.tensor(sample.hfss_nw.s, dtype=torch.complex128, device=device),
            "base_abcds_t": [torch.tensor(a, dtype=torch.complex128, device=device) for a in sample.shortened_abcds],
        }
        items.append(item)

    if sample_batch_size is None or sample_batch_size <= 0 or sample_batch_size > len(items):
        sample_batch_size = len(items)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-8)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(seed=2026)
    history: Dict[str, List[float]] = {"epoch": [], "loss": [], "loss_s": [], "loss_reg": []}

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(items))

        epoch_loss_sum = 0.0
        epoch_s_sum = 0.0
        epoch_reg_sum = 0.0
        epoch_count = 0

        for start in range(0, len(items), sample_batch_size):
            batch_ids = order[start:start + sample_batch_size]

            optimizer.zero_grad(set_to_none=True)
            loss_s_total = torch.zeros((), dtype=torch.float64, device=device)
            loss_reg_total = torch.zeros((), dtype=torch.float64, device=device)

            for bid in batch_ids:
                item = items[int(bid)]
                y_norm_pred = model(item["X_t"])
                log_values = y_norm_pred * y_std_t + y_mean_t
                log_values = torch.clamp(log_values, min=-100.0, max=100.0)
                values = torch.exp(log_values).reshape(item["n_trans"], item["n_freq"], 6)

                pred_s = cascade_with_transition_values_torch(item["base_abcds_t"], values, item["omega_t"])
                loss_s_total = loss_s_total + torch.mean(torch.abs(pred_s - item["target_s_t"]) ** 2)
                loss_reg_total = loss_reg_total + F.mse_loss(y_norm_pred, item["Y_t"])

            batch_n = max(len(batch_ids), 1)
            loss_s = loss_s_total / batch_n
            loss_reg = loss_reg_total / batch_n
            loss = loss_s + reg_weight * loss_reg

            if not torch.isfinite(loss):
                raise FloatingPointError(f"端到端数据集训练出现 NaN/Inf，epoch={epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            loss_s_val = float(loss_s.detach().cpu())
            loss_reg_val = float(loss_reg.detach().cpu())

            epoch_loss_sum += loss_val * batch_n
            epoch_s_sum += loss_s_val * batch_n
            epoch_reg_sum += loss_reg_val * batch_n
            epoch_count += batch_n

        epoch_loss = epoch_loss_sum / max(epoch_count, 1)
        epoch_s = epoch_s_sum / max(epoch_count, 1)
        epoch_reg = epoch_reg_sum / max(epoch_count, 1)

        history["epoch"].append(float(epoch))
        history["loss"].append(float(epoch_loss))
        history["loss_s"].append(float(epoch_s))
        history["loss_reg"].append(float(epoch_reg))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(
                f"    [FineTune Dataset] epoch={epoch:04d}, "
                f"loss={epoch_loss:.6e}, loss_s={epoch_s:.6e}, "
                f"loss_reg={epoch_reg:.6e}, dut_samples={len(items)}"
            )

    model.load_state_dict(best_state)
    return model, history


def run_dataset_training(
    start_idx: int = 1,
    end_idx: int = 10,
    s2p_dir: str = "./RDL_TSV_Snp",
    mat_dir: str = "./RDL_TSV_mat2",
    max_points: Optional[int] = None,
    supervised_epochs: int = 1000,
    fine_epochs: int = 300,
    supervised_lr: float = 2e-3,
    fine_lr: float = 2e-4,
    fine_reg_weight: float = 1e-4,
    hidden: int = 128,
    supervised_batch_size: int = 8192,
    fine_sample_batch_size: int = 2,
    plot: bool = True,
    save_plot: bool = False,
    out_dir: str = "./RDL_TSV_results",
    save_intermediate: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    推荐主入口：使用多个不同尺寸 DUT 共同训练一个可缩放过渡结构模型。

    训练流程：
    1. 对 start_idx..end_idx 中存在的 dut*.s2p 逐个提取器件参数、RLGC、过渡元件；
    2. 合并所有 DUT 的过渡结构提参数据 X/Y，监督训练一个共享 NN；
    3. 使用同一个共享 NN 对所有 DUT 进行预测和级联评估；
    4. 使用所有 DUT 的 HFSS 整体 S 参数作为共同目标，端到端微调共享 NN；
    5. 保存关键中间结果，便于后续分析和调用。
    """
    base_dir = script_base_dir()
    out_dir_abs = as_abs_path(out_dir, base_dir)
    persist_dir = out_dir_abs if save_intermediate else ""

    print("\n" + "=" * 88)
    print(">>> Step 1/4: 收集多个 DUT，构建可缩放训练数据集")
    print("=" * 88)

    samples = collect_structure_samples(
        start_idx=start_idx,
        end_idx=end_idx,
        s2p_dir=s2p_dir,
        mat_dir=mat_dir,
        max_points=max_points,
        verbose=verbose,
    )

    if save_intermediate:
        for sample in samples:
            save_structure_sample(sample, persist_dir)

    X_all = np.vstack([sample.X_raw for sample in samples])
    Y_all = np.vstack([sample.Y_raw for sample in samples])

    if save_intermediate:
        save_training_dataset(X_all, Y_all, samples, persist_dir)

    print("\n数据集统计：")
    print(f"  DUT 数量             : {len(samples)}")
    print(f"  过渡结构训练样本数   : {X_all.shape[0]}")
    print(f"  输入维度             : {X_all.shape[1]}")
    print(f"  输出维度             : {Y_all.shape[1]} ({TRANSITION_VALUE_NAMES})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备             : {device}")
    if save_intermediate:
        print(f"  中间结果目录         : {os.path.join(out_dir_abs, 'intermediate')}")

    print("\n" + "=" * 88)
    print(">>> Step 2/4: 监督训练共享过渡结构 NN")
    print("=" * 88)

    model, normalizer, supervised_loss_history = train_supervised_transition_nn(
        X_all,
        Y_all,
        epochs=supervised_epochs,
        lr=supervised_lr,
        hidden=hidden,
        batch_size=supervised_batch_size,
        device=device,
        verbose=verbose,
    )

    model_supervised = copy.deepcopy(model).to(device=device, dtype=torch.float64)
    model_supervised.eval()

    if save_intermediate:
        save_loss_history(supervised_loss_history, persist_dir, "supervised_pretrain_loss")
        plot_loss_history(
            supervised_loss_history,
            title="Supervised Pretrain Loss",
            save_path=os.path.join(out_dir_abs, "intermediate", "loss_curves", "supervised_pretrain_loss.png"),
            show=plot,
        )

    if save_intermediate:
        save_normalizer(normalizer, persist_dir)
        save_model_checkpoint(
            model_supervised,
            normalizer,
            persist_dir,
            "transition_model_supervised",
            extra={"stage": "supervised", "hidden": hidden, "epochs": supervised_epochs},
        )

    print("\n" + "=" * 88)
    print(">>> Step 3/4: 以所有 DUT 的 HFSS 整体 S 参数为共同目标端到端微调")
    print("=" * 88)

    if fine_epochs > 0:
        model, fine_tune_loss_history = fine_tune_transition_nn_on_dataset(
            model=model,
            normalizer=normalizer,
            samples=samples,
            epochs=fine_epochs,
            lr=fine_lr,
            reg_weight=fine_reg_weight,
            sample_batch_size=fine_sample_batch_size,
            device=device,
            verbose=verbose,
        )
    else:
        print("fine_epochs <= 0，跳过端到端微调。")
        fine_tune_loss_history = {"epoch": [], "loss": [], "loss_s": [], "loss_reg": []}

    model_fine_tuned = copy.deepcopy(model).to(device=device, dtype=torch.float64)
    model_fine_tuned.eval()

    if save_intermediate:
        save_loss_history(fine_tune_loss_history, persist_dir, "hfss_fine_tune_loss")
        plot_loss_history(
            fine_tune_loss_history,
            title="HFSS Fine-tune Loss",
            save_path=os.path.join(out_dir_abs, "intermediate", "loss_curves", "hfss_fine_tune_loss.png"),
            show=plot,
        )
        save_model_checkpoint(
            model_fine_tuned,
            normalizer,
            persist_dir,
            "transition_model_fine_tuned",
            extra={
                "stage": "fine_tuned",
                "hidden": hidden,
                "supervised_epochs": supervised_epochs,
                "fine_epochs": fine_epochs,
                "fine_reg_weight": fine_reg_weight,
            },
        )

    print("\n" + "=" * 88)
    print(">>> Step 4/4: 对每个 DUT 评估并绘图")
    print("=" * 88)

    results: Dict[int, Dict[str, object]] = {}
    mse_rows: List[Dict[str, float]] = []

    for sample in samples:
        nw_nn_supervised, values_nn_supervised = evaluate_sample_with_transition_model(
            sample=sample,
            model=model_supervised,
            normalizer=normalizer,
            name="NN supervised transition",
            device=device,
        )
        nw_nn_fine, values_nn_fine = evaluate_sample_with_transition_model(
            sample=sample,
            model=model_fine_tuned,
            normalizer=normalizer,
            name="NN fine-tuned transition",
            device=device,
        )

        compare_networks = {
            "Direct full cascade": sample.direct_full_nw,
            "Extracted transition": sample.extracted_transition_nw,
            "NN supervised transition": nw_nn_supervised,
            "NN fine-tuned transition": nw_nn_fine,
        }

        print("\n" + "-" * 88)
        print(f"dut{sample.idx} 结果：")
        print_mse_table(sample.hfss_nw, compare_networks)

        mse_row = {"idx": float(sample.idx)}
        for name, nw in compare_networks.items():
            mse_row[name] = complex_mse(sample.hfss_nw.s, nw.s)
        mse_rows.append(mse_row)

        if save_intermediate:
            save_evaluation_result(
                sample,
                compare_networks,
                values_nn_supervised,
                values_nn_fine,
                mse_row,
                persist_dir,
            )

        if plot or save_plot:
            save_path = None
            if save_plot:
                save_path = os.path.join(out_dir_abs, f"dut{sample.idx}_comparison.png")
            plot_s_comparison(
                hfss=sample.hfss_nw,
                networks=compare_networks,
                title_suffix=f"dut{sample.idx}",
                save_path=save_path,
                show=plot,
            )

        results[sample.idx] = {
            "hfss": sample.hfss_nw,
            "blocks": sample.blocks,
            "direct_full": sample.direct_full_nw,
            "extracted_transition": sample.extracted_transition_nw,
            "nn_supervised_transition": nw_nn_supervised,
            "nn_fine_tuned_transition": nw_nn_fine,
            "transition_values_extracted": sample.transition_values_extracted,
            "transition_values_nn_supervised": values_nn_supervised,
            "transition_values_nn_fine_tuned": values_nn_fine,
            "mse": {name: mse_row[name] for name in compare_networks.keys()},
        }

    print_dataset_mse_summary(mse_rows)
    error_analysis = analyze_mse_rows(mse_rows)
    print_error_analysis(error_analysis)
    if save_intermediate:
        save_mse_summary(mse_rows, persist_dir)
        save_error_analysis(error_analysis, persist_dir)

    return {
        "samples": samples,
        "results": results,
        "transition_model_supervised": model_supervised,
        "transition_model_fine_tuned": model_fine_tuned,
        "transition_normalizer": normalizer,
        "mse_rows": mse_rows,
        "supervised_loss_history": supervised_loss_history,
        "fine_tune_loss_history": fine_tune_loss_history,
        "error_analysis": error_analysis,
    }


def run_one_dut(idx: int, **kwargs) -> Dict[str, object]:
    """
    单 DUT 调试入口。注意：为了保持模型可缩放，这里也调用数据集训练入口，
    只是数据集里只有一个 DUT。实际建模建议使用 run_dataset_training。
    """
    output = run_dataset_training(start_idx=idx, end_idx=idx, **kwargs)
    return output["results"].get(idx, {})


def run_batch(start_idx: int = 1, end_idx: int = 10, **kwargs) -> Dict[str, object]:
    """兼容旧入口；现在默认执行多 DUT 共享模型训练。"""
    return run_dataset_training(start_idx=start_idx, end_idx=end_idx, **kwargs)
