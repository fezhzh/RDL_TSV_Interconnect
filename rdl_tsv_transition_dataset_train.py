# -*- coding: utf-8 -*-
"""
RDL/TSV 级联 + 过渡结构建模 + 共享过渡结构神经网络训练入口。

该文件保留为兼容入口；具体实现已按功能拆分到 rdl_tsv_transition 包：
    constants.py      全局常量和器件/过渡结构约定
    utils.py          路径、Network、S/ABCD 转换和级联工具
    io.py             s2p 头部几何参数解析
    devices.py        RDL/TSV 器件块构造与长度缩放
    matlab_nn.py      调用 MATLAB 导出的 .mat 神经网络
    circuit.py        等效电路参数 -> RLGC -> ABCD/Network
    transition.py     过渡结构元件提取、ABCD 构造和级联
    model.py          过渡结构 NN、特征构造、监督训练和预测
    torch_cascade.py  PyTorch 端到端级联和单 DUT 微调
    metrics_plot.py   MSE 评估、汇总和绘图
    persistence.py    关键中间结果保存
    dataset.py        多 DUT 数据集训练主流程

运行后默认在 ./RDL_TSV_results/intermediate 下保存：
    dutXXX/metadata.json
    dutXXX/sample_arrays.npz
    dutXXX/evaluation_arrays.npz
    dutXXX/mse.json
    dataset/transition_training_dataset.npz
    dataset/mse_summary.csv
    models/transition_normalizer.npz
    models/transition_model_supervised.pth
    models/transition_model_fine_tuned.pth
"""

from rdl_tsv_transition import run_batch, run_dataset_training, run_one_dut
from rdl_tsv_transition.circuit import block_to_abcd, block_to_network, circuit_params_to_rlgc, rlgc_to_abcd
from rdl_tsv_transition.constants import (
    CIRCUIT_PARAM_NAMES,
    CURVE_STYLES,
    DEVICE_SEQUENCE,
    FALLBACK_LINESTYLES,
    FALLBACK_MARKERS,
    KIND_ORDER,
    KIND_TO_ONEHOT,
    MAT_PREFIX,
    TRANSITION_VALUE_NAMES,
    Z_REF,
)
from rdl_tsv_transition.dataset import (
    StructureSample,
    collect_structure_samples,
    evaluate_sample_with_transition_model,
    fine_tune_transition_nn_on_dataset,
    prepare_structure_sample,
)
from rdl_tsv_transition.devices import DeviceBlock, build_structure_blocks, make_device_block, shortened_length_scales
from rdl_tsv_transition.io import parse_s2p_header_params
from rdl_tsv_transition.matlab_nn import (
    attach_circuit_params_to_blocks,
    predict_circuit_parameters,
    predict_one_matlab_nn,
)
from rdl_tsv_transition.metrics_plot import (
    analyze_mse_rows,
    complex_mse,
    plot_loss_history,
    plot_s_comparison,
    print_dataset_mse_summary,
    print_error_analysis,
    print_mse_table,
)
from rdl_tsv_transition.model import (
    Normalizer,
    TransitionElementNN,
    build_transition_training_data,
    make_normalizer,
    predict_transition_values_np,
    train_supervised_transition_nn,
    transition_input_vector,
)
from rdl_tsv_transition.persistence import (
    save_evaluation_result,
    save_error_analysis,
    save_loss_history,
    save_model_checkpoint,
    save_mse_summary,
    save_normalizer,
    save_structure_sample,
    save_training_dataset,
)
from rdl_tsv_transition.torch_cascade import (
    cascade_with_transition_values_torch,
    fine_tune_transition_nn_on_hfss,
    transition_abcd_torch,
)
from rdl_tsv_transition.transition import (
    build_transition_values_for_structure,
    cascade_with_transitions_np,
    transition_abcd_from_values,
    transition_values_from_blocks,
)
from rdl_tsv_transition.utils import (
    abcd2s_np,
    abcd2s_torch,
    as_abs_path,
    cascade_abcd_np,
    frequency_from_hz,
    load_hfss_network,
    network_from_abcd,
    network_from_s,
    s2abcd_np,
    script_base_dir,
)

__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":
    # 推荐用多个不同尺寸 DUT 共同训练共享过渡模型。
    # 若只想使用提参脚本中前 300 个频点，可设置 max_points=300。
    run_dataset_training(
        start_idx=1,
        end_idx=10,
        s2p_dir="./RDL_TSV_Snp",
        mat_dir="./RDL_TSV_mat2",
        max_points=None,
        supervised_epochs=2000,
        fine_epochs=1000,
        supervised_lr=2e-3,
        fine_lr=2e-4,
        fine_reg_weight=1e-4,
        hidden=128,
        supervised_batch_size=8192,
        fine_sample_batch_size=2,
        plot=True,
        save_plot=False,
        out_dir="./RDL_TSV_results",
        save_intermediate=True,
        verbose=True,
    )
