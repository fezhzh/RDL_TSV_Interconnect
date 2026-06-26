# RDL/TSV 参数提取与 S 参数建模项目

本项目用于 RDL Top、RDL Bottom、TSV、RDL+TSV 级联结构的 S 参数数据处理、等效电路参数提取、MATLAB/PyTorch 模型训练，以及模型结果与 HFSS 仿真结果对比。

根目录按文件用途划分。找文件时优先看根目录的语义目录，不再使用旧的 `data/`、`outputs/`、`models/` 目录。

## 快速定位

| 想找的内容 | 位置 |
| --- | --- |
| `.s2p` / Snp / HFSS 仿真数据 | `snp_data/` |
| MATLAB 导出的 `.mat` 器件模型 | `device_models/` |
| 参数提取结果 / 训练 CSV 数据集 | `training_datasets/` |
| 模型对比结果、图、报告、训练产物 | `model_results/` |
| 可运行脚本 | `scripts/` |
| 可复用 Python 模块 | `src/rdl_tsv_transition/` |
| 旧数据、备份、废弃实验 | `archive/` |

## 根目录结构

| 路径 | 当前内容和用途 |
| --- | --- |
| `snp_data/` | S 参数输入数据。当前包含 `RDL_Bottom_Snp`、`RDL_Top_Snp`、`TSV_Snp`、`RDL_TSV_Snp`、`RDL_TSV_NN_Snp`。 |
| `device_models/` | MATLAB 神经网络器件模型。当前按 `RDL_TSV_mat1` 到 `RDL_TSV_mat4` 归档。 |
| `training_datasets/` | 参数提取结果和训练表格，例如 `RDL_Bottom_TD_4.csv`、`RDL_Top_TD_4.csv`、`TSV_TD_4.csv`。 |
| `model_results/` | 本地模型结果。当前主要是 `comparison/` 下的 RDL Bottom、RDL Top、TSV 对比报告和图。该目录默认不纳入 Git。 |
| `scripts/` | 参数提取、训练、模型对比和早期 S 参数计算脚本。 |
| `src/rdl_tsv_transition/` | RDL/TSV 级联与过渡结构建模的模块化 Python 包。 |
| `archive/` | 备份、缓存、历史数据、临时实验和废弃脚本。该目录默认不纳入 Git。 |

## 数据文件

### Snp 数据

| 路径 | 内容 | 当前数量 |
| --- | --- | --- |
| `snp_data/RDL_Bottom_Snp/` | RDL Bottom HFSS `.s2p` 数据 | 1400 |
| `snp_data/RDL_Top_Snp/` | RDL Top HFSS `.s2p` 数据 | 1398 |
| `snp_data/TSV_Snp/` | TSV HFSS `.s2p` 数据 | 1400 |
| `snp_data/RDL_TSV_Snp/` | RDL+TSV 级联结构 `.s2p` 数据 | 1400 |
| `snp_data/RDL_TSV_NN_Snp/` | NN 预测或中间 `.s2p` 数据 | 208 |

### 器件模型

| 路径 | 内容 | 当前数量 |
| --- | --- | --- |
| `device_models/RDL_TSV_mat1/` | MATLAB `.mat` 模型集合 | 9 |
| `device_models/RDL_TSV_mat2/` | MATLAB `.mat` 模型集合，常作为已有参考模型 | 27 |
| `device_models/RDL_TSV_mat3/` | MATLAB `.mat` 模型集合 | 9 |
| `device_models/RDL_TSV_mat4/` | 新训练或补充的 MATLAB `.mat` 模型集合 | 27 |

模型文件命名按器件和参数区分，例如 `RDL_Top_R1.mat`、`RDL_Bottom_Cox.mat`、`TSV_Rsi.mat`。

### 训练数据集

| 文件 | 用途 |
| --- | --- |
| `training_datasets/RDL_Bottom_TD.csv` | RDL Bottom 早期参数表。 |
| `training_datasets/RDL_Bottom_TD_2.csv` | RDL Bottom 中间版本参数表。 |
| `training_datasets/RDL_Bottom_TD_4.csv` | RDL Bottom 当前主要训练/对比参数表。 |
| `training_datasets/RDL_Bottom_TD_4_.csv` | RDL Bottom `TD_4` 的变体或备份。 |
| `training_datasets/RDL_Bottom_TD_dp_trend.csv` | RDL Bottom dp trend 筛选结果。 |
| `training_datasets/RDL_Bottom_TD_parametric_trend.csv` | RDL Bottom parametric trend 筛选结果。 |
| `training_datasets/RDL_top_td.csv` | RDL Top 早期参数表。 |
| `training_datasets/RDL_Top_TD_4.csv` | RDL Top 当前主要训练/对比参数表。 |
| `training_datasets/TSV_td.csv` | TSV 早期参数表。 |
| `training_datasets/TSV_TD_4.csv` | TSV 当前主要训练/对比参数表。 |

## 脚本入口

### 参数提取

| 脚本 | 默认输入 | 默认输出 / 行为 |
| --- | --- | --- |
| `scripts/parameter_extraction/提参3.py` | `snp_data/RDL_Bottom_Snp/` | `training_datasets/RDL_Bottom_TD_4.csv` |
| `scripts/parameter_extraction/提参2.py` | RDL Bottom 旧版流程 | `training_datasets/RDL_Bottom_TD_3.csv` 或旧实验结果 |
| `scripts/parameter_extraction/extract_rdl_top_params.py` | `snp_data/RDL_Top_Snp/` | `training_datasets/RDL_Top_TD_4.csv` |
| `scripts/parameter_extraction/extract_tsv_params.py` | `snp_data/TSV_Snp/` | `training_datasets/TSV_TD_4.csv`；可显示提参诊断图 |

`extract_tsv_params.py` 顶部可直接改运行开关：

```python
WRITE_CSV = False     # 是否写出 CSV
WRITE_PLOTS = True    # 是否显示诊断图
LIMIT = None          # None 表示处理全部 DUT；整数表示只处理前 N 个
PLOT_LIMIT = 0        # 0 表示显示全部已处理 DUT 的诊断图
```

### 模型训练

| 脚本 | 用途 |
| --- | --- |
| `scripts/training/nn_train_3.m` | MATLAB RDL Top 训练脚本，读取 `training_datasets/RDL_Top_TD_4.csv`，导出 `device_models/RDL_TSV_mat4/RDL_Top_*.mat`。 |
| `scripts/training/nn_train_tsv.m` | MATLAB TSV 训练脚本，读取 `training_datasets/TSV_TD_4.csv`，导出 `device_models/RDL_TSV_mat4/TSV_*.mat`。 |
| `scripts/training/rdl_tsv_transition_dataset_train.py` | Python RDL/TSV 级联 + 过渡结构共享 NN 训练入口，默认输出到 `model_results/training/RDL_TSV_results/`。 |
| `scripts/training/train_rdl_bottom_trend_then_s.py` | RDL Bottom trend 筛选、监督训练和 S 参数微调流程。 |

### 模型对比

| 脚本 | 默认对比对象 | 默认输出 |
| --- | --- | --- |
| `scripts/comparison/compare_rdl_bottom_models.py` | RDL Bottom `.mat` / PyTorch 模型 vs HFSS | `model_results/comparison/RDL_Bottom_model_compare/` |
| `scripts/comparison/compare_rdl_top_models.py` | RDL Top `.mat` 模型 vs HFSS | `model_results/comparison/RDL_Top_model_compare/` |
| `scripts/comparison/compare_tsv_models.py` | TSV `.mat` 模型 vs HFSS | `model_results/comparison/TSV_model_compare/` |

对比输出通常包含：

- `compare_report.json`
- `*_model_compare_summary.csv`
- `*_model_compare_aggregate.csv`
- `*_model_compare_compact.csv`
- `summary_error_trends.png`
- `plots/` 下的前几个 DUT 对比图和 worst-case 图

### 早期 S 参数计算脚本

`scripts/sparameter_calculation/` 保存早期计算、级联、优化和 NN 比较脚本：

- `Calc_SP.py`
- `Calc_SP_and_Opt.py`
- `Calc_SP_and_Opt2.py`
- `Calc_SP_NN.py`
- `CONN.py`
- `CONN1.py`

这些脚本保留用于复查旧流程或复用函数，新工作优先使用 `scripts/parameter_extraction/`、`scripts/training/`、`scripts/comparison/` 下的入口。

## Python 模块

`src/rdl_tsv_transition/` 是可复用包，主要模块如下：

| 文件 | 作用 |
| --- | --- |
| `dataset.py` | 多 DUT 数据准备、共享过渡网络训练、端到端微调和评估主流程。 |
| `devices.py` | RDL/TSV 器件块构造与长度缩放。 |
| `matlab_nn.py` | 读取 MATLAB `.mat` 网络并预测等效电路参数。 |
| `circuit.py` | 等效电路参数到 RLGC、ABCD、Network 的转换。 |
| `transition.py` | 过渡结构参数构造与级联。 |
| `torch_cascade.py` | PyTorch 端到端级联和微调计算。 |
| `model.py` | 过渡结构 NN、特征构造、归一化与监督训练。 |
| `metrics_plot.py` | MSE 统计、误差分析和绘图。 |
| `plotting.py` | 对比图、诊断图等通用绘图工具。 |
| `persistence.py` | 中间数据、模型、loss、评估结果保存。 |
| `io.py`、`utils.py`、`constants.py` | 数据读取、路径/Network 工具和常量。 |

包入口：

```python
from rdl_tsv_transition import run_dataset_training, run_one_dut, run_batch
```

## 常用命令

从项目根目录运行：

```powershell
python scripts\parameter_extraction\extract_rdl_top_params.py
python scripts\parameter_extraction\extract_tsv_params.py

python scripts\comparison\compare_rdl_bottom_models.py --no-plots
python scripts\comparison\compare_rdl_top_models.py --no-plots
python scripts\comparison\compare_tsv_models.py

python scripts\training\rdl_tsv_transition_dataset_train.py
```

MATLAB 中运行：

```matlab
run("scripts/training/nn_train_3.m")
run("scripts/training/nn_train_tsv.m")
```

如果直接导入 Python 包，先把项目根目录下的 `src/` 加入 `PYTHONPATH`。

## 结果目录

当前 `model_results/` 中已有：

- `model_results/comparison/RDL_Bottom_model_compare/`
- `model_results/comparison/RDL_Top_model_compare/`
- `model_results/comparison/TSV_model_compare/`

后续训练或对比输出建议继续放在：

- `model_results/comparison/`：模型对比报告、CSV、图。
- `model_results/training/`：训练日志、loss、checkpoint、中间数据。
- `model_results/figures/`：临时或汇总图。
- `model_results/checkpoints/`：单独保存的 `.pth` / `.pt` 权重。

## Git 与文件管理

当前 Git 主要管理：

- `src/`
- `scripts/`
- `snp_data/`
- `device_models/`
- `training_datasets/`
- `README.md`
- `.gitignore`

默认忽略：

- `archive/`
- `model_results/`
- `__pycache__/`
- `*.pyc`
- `*.pth`
- `*.pt`
- `*.png`

注意事项：

- 不要恢复旧的根目录 `data/`、`outputs/`、`models/`；新文件应放入上面的语义目录。
- 包含中文注释或中文文件名的脚本统一按 UTF-8 打开和保存，避免用 ANSI/GBK 另存。
- `archive/` 仅用于归档，不作为当前流程入口。
