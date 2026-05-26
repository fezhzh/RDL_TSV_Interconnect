# rdl_tsv_transition

RDL/TSV 级联、过渡结构建模、共享过渡结构神经网络训练和 HFSS 端到端微调工具包。

原始单文件脚本已拆分为多个模块。兼容入口仍保留在上一级目录：

```text
../rdl_tsv_transition_dataset_train.py
```

推荐直接从该入口运行，或从本包导入 `run_dataset_training`。

## 依赖

运行完整流程需要：

```text
numpy
scipy
scikit-rf
matplotlib
torch
```

如果导入时报错 `ModuleNotFoundError: No module named 'skrf'`，需要先安装 `scikit-rf`。

## 数据目录约定

默认假设数据和 MATLAB 导出的网络参数位于入口脚本同级目录：

```text
data/sparameters/RDL_TSV_Snp/
  dut1.s2p
  dut2.s2p
  ...

data/matlab_models/RDL_TSV_mat2/
  RDL_Top_R1.mat
  RDL_Top_R2.mat
  ...
  RDL_Bottom_R1.mat
  ...
  TSV_R1.mat
  ...
```

`.s2p` 文件头部注释中需要包含几何参数，例如：

```text
! lrdl=...
! wrdl=...
! trdl=...
! ldown=...
! wdown=...
! tdown=...
! dtsv=...
! htsv=...
! p1=...
# ...
```

## 模块说明

### `constants.py`

定义全局常量和模型约定：

- `Z_REF`：参考阻抗，默认 50 Ohm。
- `CIRCUIT_PARAM_NAMES`：MATLAB 网络输出的等效电路参数名。
- `DEVICE_SEQUENCE`：整体结构中的 RDL/TSV 级联顺序。
- `MAT_PREFIX`：不同器件类型对应的 `.mat` 文件名前缀。
- `KIND_TO_ONEHOT`：过渡结构 NN 输入中的器件类型编码。
- `TRANSITION_VALUE_NAMES`：过渡结构 NN 输出顺序 `[L1, R1, L2, R2, C1, G1]`。
- 绘图曲线样式。

### `utils.py`

基础工具函数：

- 路径处理：`script_base_dir`、`as_abs_path`
- `skrf.Network` 构造：`network_from_s`、`network_from_abcd`
- HFSS `.s2p` 读取：`load_hfss_network`
- S 参数和 ABCD 转换：`s2abcd_np`、`abcd2s_np`、`abcd2s_torch`
- ABCD 级联：`cascade_abcd_np`

### `io.py`

输入文件解析：

- `parse_s2p_header_params(filepath)`：读取 `.s2p` 文件开头注释行中的 `key=value` 几何参数。

### `devices.py`

器件结构定义和几何参数组装：

- `DeviceBlock`：单个 RDL/TSV 器件块，包含类型、长度、几何特征、等效参数和 RLGC。
- `make_device_block`：根据 `.s2p` 头部参数创建单个器件块。
- `build_structure_blocks`：按 `DEVICE_SEQUENCE` 创建完整 13 段器件结构。
- `shortened_length_scales`：插入过渡结构后，计算每段器件保留长度比例。

长度缩放规则：

- 首尾器件保留 `0.9 * Length`
- 中间器件保留 `0.8 * Length`
- 被扣除的 `0.1 * Length` 用于相邻过渡结构建模

### `matlab_nn.py`

调用 MATLAB 导出的 `.mat` 神经网络：

- `predict_one_matlab_nn`：调用单个 `.mat` 网络预测一个电路参数。
- `predict_circuit_parameters`：对一个器件预测全部 9 个等效电路参数。
- `attach_circuit_params_to_blocks`：为所有器件块附加电路参数和 RLGC。

该模块假设 `.mat` 文件包含：

```text
psmin, psmax,
w1, theta1,
w2, theta2,
w3, theta3,
outputmax, outputmin
```

### `circuit.py`

等效电路参数到电磁网络参数的转换：

- `circuit_params_to_rlgc`：根据等效电路公式计算单位长度 `R/L/G/C`。
- `rlgc_to_abcd`：将传输线 RLGC 模型转换为 ABCD 矩阵。
- `block_to_abcd`：将单个 `DeviceBlock` 转换为 ABCD。
- `block_to_network`：将单个 `DeviceBlock` 转换为 `skrf.Network`。

### `transition.py`

过渡结构提取和级联：

- `transition_values_from_blocks`：由左右相邻器件的 `0.1 * Length` RLGC 提取过渡结构元件。
- `transition_abcd_from_values`：将 `[L1, R1, L2, R2, C1, G1]` 转换为 ABCD。
- `build_transition_values_for_structure`：为完整结构中所有相邻器件生成过渡元件。
- `cascade_with_transitions_np`：执行“缩短器件 + 过渡结构”的整体级联。

过渡结构拓扑：

```text
Port1 -- L1 -- R1 -- node -- L2 -- R2 -- Port2
                           |
                         C1 || G1
                           |
                          GND
```

### `model.py`

过渡结构神经网络及监督训练：

- `Normalizer`：保存输入特征和输出 log 元件值的标准化参数。
- `transition_input_vector`：构造 NN 输入特征。
- `build_transition_training_data`：生成监督训练集 `X_raw/Y_raw`。
- `TransitionElementNN`：过渡结构元件值预测网络。
- `make_normalizer`：生成标准化器。
- `train_supervised_transition_nn`：用提取的过渡结构元件值监督训练 NN。
- `predict_transition_values_np`：用训练好的 NN 预测完整结构中的所有过渡元件。

NN 输入维度为 17：

```text
left_type_onehot(3)
right_type_onehot(3)
left_geom5(5)
right_geom5(5)
freq_GHz(1)
```

NN 输出维度为 6：

```text
[L1, R1, L2, R2, C1, G1]
```

训练时输出目标使用 `log(Y)` 后标准化，以减小不同量纲造成的数值差异。

### `torch_cascade.py`

PyTorch 版本的过渡结构级联，用于端到端微调：

- `transition_abcd_torch`：过渡结构元件值转 ABCD。
- `cascade_with_transition_values_torch`：可微分级联并输出 S 参数。
- `fine_tune_transition_nn_on_hfss`：单 DUT HFSS 目标微调入口。

多 DUT 端到端微调的主函数在 `dataset.py` 中。

### `metrics_plot.py`

评估和绘图：

- `complex_mse`：计算复数 S 参数 MSE。
- `print_mse_table`：打印单个 DUT 的模型对比 MSE。
- `print_dataset_mse_summary`：打印数据集 MSE 汇总。
- `plot_s_comparison`：绘制 HFSS、直接级联、提取过渡、NN 监督、NN 微调结果对比。

### `persistence.py`

保存关键中间结果，便于后续分析和调用：

- `save_structure_sample`：保存单 DUT 的样本准备结果。
- `save_training_dataset`：保存合并后的监督训练集。
- `save_normalizer`：保存标准化参数。
- `save_model_checkpoint`：保存 NN 权重和标准化器。
- `save_evaluation_result`：保存每个 DUT 的预测结果、S 参数和 MSE。
- `save_mse_summary`：保存全数据集 MSE 汇总。

默认保存目录：

```text
outputs/training/RDL_TSV_results/
  intermediate/
    dut001/
      metadata.json
      sample_arrays.npz
      evaluation_arrays.npz
      mse.json
    dut002/
      ...
    dataset/
      transition_training_dataset.npz
      mse_summary.csv
      error_analysis.json
      error_analysis.md
    models/
      transition_normalizer.npz
      transition_model_supervised.pth
      transition_model_fine_tuned.pth
    loss_curves/
      supervised_pretrain_loss.csv
      supervised_pretrain_loss.png
      hfss_fine_tune_loss.csv
      hfss_fine_tune_loss.png
```

### `dataset.py`

完整多 DUT 工作流主模块：

- `StructureSample`：单个 DUT 的完整样本数据结构。
- `prepare_structure_sample`：准备一个 DUT 的 HFSS、器件块、RLGC、ABCD、过渡结构和训练样本。
- `collect_structure_samples`：收集多个 DUT。
- `evaluate_sample_with_transition_model`：用过渡结构 NN 评估一个 DUT。
- `fine_tune_transition_nn_on_dataset`：以多个 DUT 的 HFSS S 参数为共同目标微调共享 NN。
- `run_dataset_training`：推荐主入口。
- `run_one_dut`：单 DUT 调试入口。
- `run_batch`：兼容旧入口。

## 整体工作流程

完整流程由 `run_dataset_training` 驱动。

### Step 1：收集 DUT 并构建训练数据

对 `start_idx..end_idx` 中存在的 `dut*.s2p`：

1. 读取 HFSS 整体结构 S 参数。
2. 解析 `.s2p` 头部几何参数。
3. 按固定序列创建 RDL/TSV 器件块。
4. 调用 `.mat` 神经网络预测每个器件的等效电路参数。
5. 计算每个器件的单位长度 RLGC。
6. 构造完整长度直接级联结果。
7. 构造缩短器件段。
8. 从相邻器件提取过渡结构元件。
9. 生成监督训练样本 `X_raw/Y_raw`。

### Step 2：监督训练共享过渡结构 NN

合并所有 DUT 的 `X_raw/Y_raw`：

```text
X_all = vstack(sample.X_raw)
Y_all = vstack(sample.Y_raw)
```

使用提取出来的过渡结构元件值作为监督目标，训练一个共享 `TransitionElementNN`。

输出：

- `transition_model_supervised`
- `transition_normalizer`
- `supervised_loss_history`
- `loss_curves/supervised_pretrain_loss.png`
- `loss_curves/supervised_pretrain_loss.csv`

### Step 3：HFSS 端到端微调

以多个 DUT 的 HFSS 整体 S 参数为共同目标，继续微调同一个共享 NN。

损失函数：

```text
loss = mean(MSE(S_pred, S_HFSS))
       + fine_reg_weight * mean(MSE(predicted_transition_norm, extracted_transition_norm))
```

第二项用于约束微调后的过渡元件不要过度偏离由 `0.1 * Length` RLGC 提取得到的初始物理估计。

输出：

- `transition_model_fine_tuned`
- `fine_tune_loss_history`
- `loss_curves/hfss_fine_tune_loss.png`
- `loss_curves/hfss_fine_tune_loss.csv`

### Step 4：评估和保存结果

对每个 DUT 输出以下模型对比：

```text
Direct full cascade
Extracted transition
NN supervised transition
NN fine-tuned transition
```

并计算相对 HFSS 的复数 S 参数 MSE。

训练完成后还会对所有 sample 的误差做统计分析：

- 计算每个模型的 mean、median、std、min、max MSE。
- 找出平均 MSE 最优模型。
- 按最终模型 MSE 对 DUT 排序，定位误差最大的 sample。
- 根据直接级联、提取过渡、监督 NN、微调 NN 的相对改善情况生成改进建议。

分析结果保存到：

```text
outputs/training/RDL_TSV_results/intermediate/dataset/error_analysis.json
outputs/training/RDL_TSV_results/intermediate/dataset/error_analysis.md
```

如果开启 `plot=True` 或 `save_plot=True`，会绘制：

- S11 magnitude
- S21 magnitude
- S11 phase
- S21 phase

## 使用方法

### 方式 1：运行兼容入口脚本

在 `Temp` 目录下运行：

```bash
python rdl_tsv_transition_dataset_train.py
```

该入口使用默认参数：

```python
run_dataset_training(
    start_idx=1,
    end_idx=10,
    s2p_dir="./data/sparameters/RDL_TSV_Snp",
    mat_dir="./data/matlab_models/RDL_TSV_mat2",
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
    out_dir="./outputs/training/RDL_TSV_results",
    save_intermediate=True,
    verbose=True,
)
```

### 方式 2：在 Python 中调用

```python
from rdl_tsv_transition import run_dataset_training

output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    s2p_dir="./data/sparameters/RDL_TSV_Snp",
    mat_dir="./data/matlab_models/RDL_TSV_mat2",
    max_points=300,
    supervised_epochs=500,
    fine_epochs=200,
    plot=False,
    save_plot=True,
    out_dir="./outputs/training/RDL_TSV_results",
    save_intermediate=True,
)
```

返回值是一个字典：

```python
{
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
```

### 单 DUT 调试

```python
from rdl_tsv_transition import run_one_dut

result = run_one_dut(
    idx=1,
    s2p_dir="./data/sparameters/RDL_TSV_Snp",
    mat_dir="./data/matlab_models/RDL_TSV_mat2",
    max_points=300,
    supervised_epochs=200,
    fine_epochs=100,
    plot=False,
)
```

### 跳过端到端微调

```python
output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    fine_epochs=0,
)
```

### 不保存中间结果

```python
output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    save_intermediate=False,
)
```

### 只保存图，不弹出图窗

```python
output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    plot=False,
    save_plot=True,
)
```

## 中间结果读取示例

读取合并后的监督训练集：

```python
import numpy as np

data = np.load("./outputs/training/RDL_TSV_results/intermediate/dataset/transition_training_dataset.npz")
X_all = data["X_all"]
Y_all = data["Y_all"]
```

读取某个 DUT 的样本数组：

```python
sample = np.load("./outputs/training/RDL_TSV_results/intermediate/dut001/sample_arrays.npz")
freqs_hz = sample["freqs_hz"]
hfss_s = sample["hfss_s"]
direct_full_s = sample["direct_full_s"]
extracted_transition_s = sample["extracted_transition_s"]
X_raw = sample["X_raw"]
Y_raw = sample["Y_raw"]
```

读取模型：

```python
import torch
from rdl_tsv_transition.model import TransitionElementNN, Normalizer

ckpt = torch.load("./outputs/training/RDL_TSV_results/intermediate/models/transition_model_fine_tuned.pth")

model = TransitionElementNN(hidden=ckpt["extra"]["hidden"]).to(dtype=torch.float64)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

n = ckpt["normalizer"]
normalizer = Normalizer(
    x_mean=n["x_mean"],
    x_std=n["x_std"],
    y_mean=n["y_mean"],
    y_std=n["y_std"],
)
```

读取 loss 历史：

```python
import pandas as pd

pretrain_loss = pd.read_csv("./outputs/training/RDL_TSV_results/intermediate/loss_curves/supervised_pretrain_loss.csv")
fine_loss = pd.read_csv("./outputs/training/RDL_TSV_results/intermediate/loss_curves/hfss_fine_tune_loss.csv")
```

读取误差分析：

```python
import json

with open("./outputs/training/RDL_TSV_results/intermediate/dataset/error_analysis.json", "r", encoding="utf-8") as f:
    analysis = json.load(f)

print(analysis["best_model_by_mean_mse"])
print(analysis["recommendations"])
```

## 常见参数说明

| 参数 | 说明 |
| --- | --- |
| `start_idx`, `end_idx` | DUT 编号范围，对应 `dut{idx}.s2p` |
| `s2p_dir` | HFSS 整体结构 `.s2p` 目录 |
| `mat_dir` | MATLAB 导出的器件级 `.mat` 网络参数目录 |
| `max_points` | 只使用前 N 个频点；`None` 表示使用全部频点 |
| `supervised_epochs` | 过渡结构 NN 监督训练轮数 |
| `fine_epochs` | HFSS 端到端微调轮数；设为 0 可跳过 |
| `supervised_lr` | 监督训练学习率 |
| `fine_lr` | 端到端微调学习率 |
| `fine_reg_weight` | 微调时约束过渡元件偏离提取值的正则权重 |
| `hidden` | 过渡结构 NN 隐藏层宽度 |
| `supervised_batch_size` | 监督训练 batch size |
| `fine_sample_batch_size` | 端到端微调时每个 batch 包含的 DUT 数量 |
| `plot` | 是否显示对比图 |
| `save_plot` | 是否保存对比图 |
| `out_dir` | 输出目录 |
| `save_intermediate` | 是否保存关键中间结果 |
| `verbose` | 是否打印详细训练日志 |

## 注意事项

1. `s2p_dir` 和 `mat_dir` 的相对路径基准是入口脚本所在目录。
2. `.mat` 文件名必须满足 `MAT_PREFIX + CIRCUIT_PARAM_NAME + ".mat"` 的规则。
3. `TSV` 的几何输入只有 3 维，代码会 padding 到 5 维，并通过 one-hot 类型区分含义。
4. 训练和级联默认使用 `float64/complex128`，以降低高频级联时的数值误差。
5. 如果数据点很多且 GPU/内存不足，优先减小 `max_points`、`supervised_batch_size` 或 `fine_sample_batch_size`。
