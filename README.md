# RDL/TSV 参数提取与 S 参数建模工作目录

本目录用于 RDL、TSV、RDL+TSV 级联结构的 S 参数数据处理、等效电路参数提取、神经网络训练和模型结果对比。当前整理原则是：源码和输入数据保留在根目录及功能目录中；缓存、备份、smoke/test 运行结果和明显无效文件已移动到 `archive/`；Git 主要管理源码、说明和可复现实验所需的小型输入文件。

## 主要入口

| 路径 | 作用 |
| --- | --- |
| `rdl_tsv_transition/` | 模块化后的 RDL/TSV 级联、过渡结构建模、共享过渡结构 NN 训练和 HFSS 微调工具包。详细用法见 `rdl_tsv_transition/README.md`。 |
| `rdl_tsv_transition_dataset_train.py` | `rdl_tsv_transition` 包的兼容入口脚本，默认读取 `RDL_TSV_Snp/` 和 `RDL_TSV_mat2/`，输出到 `RDL_TSV_results/`。 |
| `train_rdl_bottom_trend_then_s.py` | RDL Bottom 参数趋势筛选、监督训练和 S 参数微调流程。 |
| `compare_rdl_bottom_models.py` | 对比不同 RDL Bottom MATLAB 参数模型的结果，并生成对比图和统计。 |
| `compare_rdl_bottom_new_vs_mat.py` | 对比新训练 RDL Bottom 模型与 MATLAB 模型输出。 |
| `extract_params_parametric_trend.py` | 从 RDL Bottom 数据中提取参数化趋势结果。 |
| `extract_params_dp_trend.py` | 从 RDL Bottom 数据中提取 DP trend 结果。 |
| `Calc_SP.py`, `Calc_SP_and_Opt.py`, `Calc_SP_and_Opt2.py`, `Calc_SP_NN.py` | 早期 S 参数计算、优化和 NN 相关脚本。 |
| `CONN.py`, `CONN1.py` | 连接/级联网络相关早期脚本。 |
| `提参2.py`, `提参3.py` | 参数提取脚本，`提参3.py` 是最近修改版本。 |
| `nn_train_3.m` | MATLAB 训练脚本。 |

## 输入数据与模型参数

| 路径 | 作用 |
| --- | --- |
| `RDL_Top_Snp/` | RDL Top 的 `.s2p` 原始/仿真数据。 |
| `RDL_Bottom_Snp/` | RDL Bottom 的 `.s2p` 原始/仿真数据。 |
| `TSV_Snp/` | TSV 的 `.s2p` 原始/仿真数据。 |
| `RDL_TSV_Snp/` | RDL+TSV 级联结构的 `.s2p` 数据，供 `rdl_tsv_transition` 默认流程使用。 |
| `RDL_TSV_NN_Snp/` | NN 相关的 RDL+TSV `.s2p` 数据子集。 |
| `RDL_TSV_mat1/` | 早期 RDL Top MATLAB 网络参数。 |
| `RDL_TSV_mat2/` | 当前默认 RDL Top、RDL Bottom、TSV MATLAB 网络参数目录。 |
| `RDL_TSV_mat3/` | RDL Bottom MATLAB 参数版本 3。 |
| `RDL_TSV_mat4/` | RDL Bottom MATLAB 参数版本 4。 |
| `SNN_Cascade_to_HFSS.pth` | 已训练 PyTorch 模型权重，属于二进制模型产物，默认不纳入 Git。 |

## 表格与验证图

| 路径 | 作用 |
| --- | --- |
| `RDL_top_td.csv` | RDL Top 时域/参数提取结果表。 |
| `RDL_Bottom_TD.csv` | RDL Bottom 基础参数结果表。 |
| `RDL_Bottom_TD_2.csv`, `RDL_Bottom_TD_4.csv` | RDL Bottom 参数结果的不同版本。 |
| `RDL_Bottom_TD_parametric_trend.csv` | 参数化趋势筛选/拟合后的 RDL Bottom 结果。 |
| `RDL_Bottom_TD_dp_trend.csv` | DP trend 筛选/拟合后的 RDL Bottom 结果。 |
| `TSV_td.csv` | TSV 参数结果表。 |
| `parametric_trend_validation.png` | 参数化趋势验证图，属于可再生成输出，默认不纳入 Git。 |
| `dp_trend_validation.png` | DP trend 验证图，属于可再生成输出，默认不纳入 Git。 |

## 当前保留的输出目录

| 路径 | 作用 |
| --- | --- |
| `RDL_TSV_results/` | `rdl_tsv_transition` 默认完整输出目录。 |
| `RDL_TSV_results_analysis_run/` | RDL/TSV 分析运行输出。 |
| `RDL_Bottom_TD4_trend_sparam_training/` | RDL Bottom TD4 trend+sparam 训练输出。 |
| `RDL_Bottom_TD4_trend_sparam_training_long/` | TD4 trend+sparam 长训练输出。 |
| `RDL_Bottom_TD4_trend_sparam_training_loose/` | TD4 trend+sparam 宽松筛选训练输出。 |
| `RDL_Bottom_model_compare/` | RDL Bottom 模型对比输出和图表。 |

这些目录主要是实验产物，体积和变动都比较大，已在 `.gitignore` 中排除。需要复现实验时保留在本地即可。

## 归档目录

| 路径 | 内容 |
| --- | --- |
| `archive/cache/` | Python `__pycache__` 缓存。 |
| `archive/backup/` | 备份数据，例如 `RDL_TSV_Snp_bak/`。 |
| `archive/experiments/` | smoke/test/loss_test 等临时实验输出，以及空的 matplotlib SVG 测试文件。 |
| `archive/legacy_data/` | 原名不明确的历史数据目录，例如 `新建文件夹/`。 |
| `archive/obsolete/` | 自动保存文件和风险较高的旧清理脚本，例如 `nn_train_3.asv`、`delete.py`。 |

归档只是移动，没有删除。如需恢复，直接从 `archive/` 移回原路径即可。

## Git 管理

已加入 `.gitignore`，默认排除：

- `archive/`
- Python 缓存
- 训练 checkpoint：`*.pth`, `*.pt`
- 可再生成实验输出：`RDL_TSV_results*/`, `RDL_Bottom_*training*/`, `RDL_Bottom_model_compare/`, `*.png`

建议后续把源码、README、CSV 输入表、`.s2p` 和 `.mat` 输入数据纳入 Git；把大模型权重、训练输出和临时归档只保留在本地或另行用数据/模型仓库管理。
