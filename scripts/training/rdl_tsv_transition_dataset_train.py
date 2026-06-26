# -*- coding: utf-8 -*-
"""
RDL/TSV 绾ц仈 + 杩囨浮缁撴瀯寤烘ā + 鍏变韩杩囨浮缁撴瀯绁炵粡缃戠粶璁粌鍏ュ彛銆?
璇ユ枃浠朵繚鐣欎负鍏煎鍏ュ彛锛涘叿浣撳疄鐜板凡鎸夊姛鑳芥媶鍒嗗埌 rdl_tsv_transition 鍖咃細
    constants.py      鍏ㄥ眬甯搁噺鍜屽櫒浠?杩囨浮缁撴瀯绾﹀畾
    utils.py          璺緞銆丯etwork銆丼/ABCD 杞崲鍜岀骇鑱斿伐鍏?    io.py             s2p 澶撮儴鍑犱綍鍙傛暟瑙ｆ瀽
    devices.py        RDL/TSV 鍣ㄤ欢鍧楁瀯閫犱笌闀垮害缂╂斁
    matlab_nn.py      璋冪敤 MATLAB 瀵煎嚭鐨?.mat 绁炵粡缃戠粶
    circuit.py        绛夋晥鐢佃矾鍙傛暟 -> RLGC -> ABCD/Network
    transition.py     杩囨浮缁撴瀯鍏冧欢鎻愬彇銆丄BCD 鏋勯€犲拰绾ц仈
    model.py          杩囨浮缁撴瀯 NN銆佺壒寰佹瀯閫犮€佺洃鐫ｈ缁冨拰棰勬祴
    torch_cascade.py  PyTorch 绔埌绔骇鑱斿拰鍗?DUT 寰皟
    metrics_plot.py   MSE 璇勪及銆佹眹鎬诲拰缁樺浘
    persistence.py    鍏抽敭涓棿缁撴灉淇濆瓨
    dataset.py        澶?DUT 鏁版嵁闆嗚缁冧富娴佺▼

杩愯鍚庨粯璁ゅ湪 ./model_results/training/RDL_TSV_results/intermediate 涓嬩繚瀛橈細
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

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    # Train the shared transition model across multiple DUT sizes.
    # Set max_points=300 to use only the first 300 frequency points.
    run_dataset_training(
        start_idx=1,
        end_idx=10,
        s2p_dir="./snp_data/RDL_TSV_Snp",
        mat_dir="./device_models/RDL_TSV_mat2",
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
        out_dir="./model_results/training/RDL_TSV_results",
        save_intermediate=True,
        verbose=True,
    )

