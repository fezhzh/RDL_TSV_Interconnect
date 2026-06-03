# RDL/TSV 参数提取与 S 参数建模项目

本项目用于 RDL、TSV、RDL+TSV 级联结构的 S 参数数据处理、等效电路参数提取、神经网络训练和模型结果对比。目录已按主要功能重新整理，根目录只保留项目说明、Git 配置和几个一级功能目录。

## 目录结构

| 路径 | 作用 |
| --- | --- |
| `src/rdl_tsv_transition/` | 模块化后的 RDL/TSV 级联、过渡结构建模、共享过渡结构 NN 训练和 HFSS 端到端微调工具包。 |
| `src/rdl_tsv_transition/plotting.py` | 脚本复用的本地绘图模块，集中提供 S 参数模型对比图、参数提取标准诊断图、`R/G/L/C` 排布和坐标轴刻度朝内的通用绘图样式函数。 |
| `scripts/parameter_extraction/` | 参数提取脚本，包括 `提参2.py`、`提参3.py` 和 trend 参数提取脚本。 |
| `scripts/sparameter_calculation/` | 早期 S 参数计算、级联、优化和 NN 比较脚本。 |
| `scripts/training/` | 训练入口脚本，包括 RDL Bottom trend+sparam 训练和 RDL/TSV 过渡模型训练。 |
| `scripts/comparison/` | RDL Bottom MATLAB 模型、新模型和 HFSS 结果对比脚本。 |
| `data/sparameters/` | `.s2p` 输入数据，按 `RDL_Top_Snp`、`RDL_Bottom_Snp`、`TSV_Snp`、`RDL_TSV_Snp` 等子目录存放。 |
| `data/matlab_models/` | MATLAB 导出的 `.mat` 神经网络参数，按 `RDL_TSV_mat1` 到 `RDL_TSV_mat4` 归档。 |
| `data/tables/` | 参数提取和筛选生成的 CSV 表格。 |
| `outputs/` | 本地实验输出、训练结果、对比图和验证图。该目录默认不纳入 Git。 |
| `models/` | 本地模型权重和 checkpoint。该目录默认不纳入 Git。 |
| `archive/` | 缓存、备份、临时实验和废弃脚本的归档区。该目录默认不纳入 Git。 |

## 常用入口

| 路径 | 作用 |
| --- | --- |
| `scripts/training/rdl_tsv_transition_dataset_train.py` | RDL/TSV 级联 + 过渡结构建模主入口，默认读取 `data/sparameters/RDL_TSV_Snp/` 和 `data/matlab_models/RDL_TSV_mat2/`。 |
| `scripts/training/train_rdl_bottom_trend_then_s.py` | RDL Bottom 参数趋势筛选、监督训练和 S 参数微调流程。 |
| `scripts/comparison/compare_rdl_bottom_models.py` | 对比 `RDL_TSV_mat2`、`RDL_TSV_mat3`、`RDL_TSV_mat4` 的 RDL Bottom 预测结果。 |
| `scripts/comparison/compare_rdl_bottom_new_vs_mat.py` | 对比新训练 RDL Bottom 模型与 MATLAB 模型输出。 |
| `scripts/parameter_extraction/extract_params_parametric_trend.py` | 生成 `data/tables/RDL_Bottom_TD_parametric_trend.csv`。 |
| `scripts/parameter_extraction/extract_params_dp_trend.py` | 生成 `data/tables/RDL_Bottom_TD_dp_trend.csv`。 |
| `scripts/parameter_extraction/提参3.py` | 最近版本的 RDL Bottom 参数提取脚本，默认输出 `data/tables/RDL_Bottom_TD_4.csv`。 |

## 输入数据

| 路径 | 内容 |
| --- | --- |
| `data/sparameters/RDL_Top_Snp/` | RDL Top 的 `.s2p` 原始/仿真数据。 |
| `data/sparameters/RDL_Bottom_Snp/` | RDL Bottom 的 `.s2p` 原始/仿真数据。 |
| `data/sparameters/TSV_Snp/` | TSV 的 `.s2p` 原始/仿真数据。 |
| `data/sparameters/RDL_TSV_Snp/` | RDL+TSV 级联结构 `.s2p` 数据。 |
| `data/sparameters/RDL_TSV_NN_Snp/` | NN 预测/中间 `.s2p` 数据。 |
| `data/matlab_models/RDL_TSV_mat2/` | 当前默认 MATLAB 网络参数目录。 |
| `data/tables/*.csv` | 参数提取、筛选和 trend 数据表。 |

## 输出和归档

`outputs/` 中保留了本地已有实验结果，包括：

- `outputs/training/RDL_TSV_results/`
- `outputs/training/RDL_TSV_results_analysis_run/`
- `outputs/training/RDL_Bottom_TD4_trend_sparam_training*/`
- `outputs/comparison/RDL_Bottom_model_compare/`
- `outputs/figures/*.png`

`archive/` 中保留了之前整理出的缓存、备份、临时 smoke/test 输出、历史数据和废弃文件。归档只是移动，没有删除；需要恢复时从 `archive/` 移回即可。

## Git 管理

当前 Git 主要管理：

- `src/` 模块源码
- `scripts/` 可执行脚本
- `data/` 中的小型/中型输入数据和表格
- `README.md`、`.gitignore`

默认忽略：

- `archive/`
- `outputs/`
- `models/`
- Python 缓存
- `*.pth`、`*.pt`
- `*.png`

## 示例命令

从项目根目录运行：

```powershell
python scripts\training\rdl_tsv_transition_dataset_train.py
python scripts\comparison\compare_rdl_bottom_models.py --no-plots
python scripts\parameter_extraction\extract_params_dp_trend.py
```

如果直接导入包，先把 `src/` 加入 `PYTHONPATH`，或在脚本中添加项目根目录下的 `src` 路径。
