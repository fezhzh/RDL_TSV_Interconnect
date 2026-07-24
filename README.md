# RDL/TSV 参数提取与 S 参数建模项目

本项目用于 RDL Top、RDL Bottom、TSV、RDL+TSV 级联结构的 S 参数数据处理、等效电路参数提取、MATLAB/PyTorch 模型训练，以及模型结果与 HFSS 仿真结果对比。

根目录按文件用途划分。找文件时优先看根目录的语义目录，不再使用旧的 `data/`、`outputs/`、`models/` 目录。

## 快速定位

| 想找的内容 | 位置 |
| --- | --- |
| `.s2p` / Snp / HFSS 仿真数据 | `snp_data/` |
| MATLAB 导出的 `.mat` 器件模型 | `model_versions/v01_matlab_mat_models/models/` |
| 参数提取结果 / 训练 CSV 数据集 | `training_datasets/` |
| 模型对比结果、图、报告、训练产物 | `model_versions/*/results/` |
| 可运行脚本 | `model_versions/*/code/` |
| 按模型版本归类的代码快照和结果索引 | `model_versions/` |
| HFSS 批量仿真脚本和本地仿真输出 | `HFSS_sim/` |
| 可复用 Python 模块 | `src/rdl_tsv_transition/` |
| 旧数据、备份、废弃实验 | `archive/` |

## 根目录结构

| 路径 | 当前内容和用途 |
| --- | --- |
| `snp_data/` | S 参数输入数据。当前包含 `RDL_Bottom_Snp`、`RDL_Top_Snp`、`TSV_Snp`、`RDL_TSV_Snp`、`RDL_TSV_NN_Snp`。 |
| `model_versions/v01_matlab_mat_models/models/` | MATLAB 神经网络器件模型。当前按 `RDL_TSV_mat1` 到 `RDL_TSV_mat4` 归档。 |
| `training_datasets/` | 参数提取结果和训练表格，例如 `RDL_Bottom_TD_4.csv`、`RDL_Top_TD_4.csv`、`TSV_TD_4.csv`。 |
| `model_versions/` | 按模型版本整理的索引目录。每个版本目录包含 `code/` 代码入口快照和 `results/` 结果目录；模型结果已按版本整理到各版本的 `results/` 中。 |
| `model_versions/*/code/` | 参数提取、训练、模型对比和早期 S 参数计算脚本。 |
| `HFSS_sim/` | PyAEDT/HFSS 批量仿真脚本。`batchsim_100samples.py` 默认打开 `C:\ffzhzh\LocalFiles\Ansys_Project_Files\TSV_RDL_Connection.aedt`，对 `TMRDL`、`BSMRDL`、`TSV`、`TSV_RDL` 生成基础 `train=100`、`val=20`、`test=20` 数据集，输出到 `HFSS_sim/LHS100/`。`batchsim_200samples.py` 是增量数据集脚本，对四个设计各生成 `train=200` 组新增样本，参考 `HFSS_sim/LHS100/train/` 中已有样本做补洞采样，输出到 `HFSS_sim/LHS200/train/<design>/`。`batchsim_400samples.py` 是独立 400 组 LHS 脚本，对四个设计各生成 `train=400` 组样本，输出到 `HFSS_sim/LHS400/train/<design>/`；Snp 文件编号固定从 `dut300` 开始，到 `dut699` 结束。`batchsim_800samples.py` 是独立 800 组 LHS 脚本，对四个设计各生成 `train=800` 组样本，输出到 `HFSS_sim/LHS800/train/<design>/`；Snp 文件编号固定从 `dut700` 开始，到 `dut1499` 结束。四个脚本均过滤 `TSV: pitch <= 2*r_tsv + 1um`、`TSV_RDL: pitch <= max(2*r_tsv, w_tmrdl, w_bsmrdl) + 1um` 等无效组合，临时 AEDT 工程和求解目录在源工程目录下创建，求解后指定 `Auto1/Sweep` 导出 Snp 文件。 |
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
| `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat1/` | MATLAB `.mat` 模型集合 | 9 |
| `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat2/` | MATLAB `.mat` 模型集合，常作为已有参考模型 | 27 |
| `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat3/` | MATLAB `.mat` 模型集合 | 9 |
| `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat4/` | 新训练或补充的 MATLAB `.mat` 模型集合 | 27 |

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

如果是按实验/模型版本查找代码和结果，优先看 `model_versions/README.md`：

| 版本目录 | 主题 |
| --- | --- |
| `model_versions/v01_matlab_mat_models/` | MATLAB `.mat` 基线模型和 RDL Bottom/Top/TSV 对比。 |
| `model_versions/v02_mat4_cascade_and_sparameter_optimization/` | mat4 直接级联和连接网络 least-squares 优化。 |
| `model_versions/v03_single_device_sparam_finetune/` | 单器件 S 参数目标微调。 |
| `model_versions/v04_hfss_split_direct_sparam/` | HFSS split 直接 S 参数代理模型。 |
| `model_versions/v05_hfss_split_circuit_param/` | HFSS split 电路参数模型和连接级联。 |
| `model_versions/v07_connection_param_and_sparam_finetune/` | 连接网络参数模型和整体 S 参数微调。 |
| `model_versions/v08_connection_multihead/` | 多头连接网络和 k-fold 验证。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/` | RDL LHS100/200/400/800/组合数据集提参、参数网络训练、S 参数微调和精度对比。 |
| `model_versions/v10_ads_pi_cascade/` | ADS 单器件仿真驱动的 v10 pi 级联网络；先优化 pi 元件值，再训练参数网络并以整体 S 参数微调。 |
| `model_versions/v12_hfss_v08_multihead_chain/` | v12 长链 HFSS 等效电路 + v08 7 参数 pi 连接网络；使用 LHS400/HFSS 派生单器件模型，先训练共享连接网络，再扩展为 12 个输出头建模 12 个连接位置。 |
| `model_versions/v99_legacy_and_shared/` | 旧版/共享 S 参数计算脚本。 |

`model_versions/*/code/` 是版本入口快照，便于按版本查找；实际运行仍建议使用 `model_versions/*/code/` 下的原始入口，避免硬编码路径被相对位置影响。

### 参数提取

| 脚本 | 默认输入 | 默认输出 / 行为 |
| --- | --- | --- |
| `model_versions/v00_parameter_extraction_and_dataset_building/code/提参3.py` | `snp_data/RDL_Bottom_Snp/` | `training_datasets/RDL_Bottom_TD_4.csv` |
| `model_versions/v00_parameter_extraction_and_dataset_building/code/提参2.py` | RDL Bottom 旧版流程 | `training_datasets/RDL_Bottom_TD_3.csv` 或旧实验结果 |
| `model_versions/v00_parameter_extraction_and_dataset_building/code/extract_rdl_top_params.py` | `snp_data/RDL_Top_Snp/` | `training_datasets/RDL_Top_TD_4.csv` |
| `model_versions/v00_parameter_extraction_and_dataset_building/code/extract_tsv_params.py` | `snp_data/TSV_Snp/` | `training_datasets/TSV_TD_4.csv`；可显示提参诊断图 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/extract_rdl_params_for_lhs_dataset_comparison.py` | `HFSS_sim/LHS100`、`HFSS_sim/LHS200`、`HFSS_sim/LHS400`、`HFSS_sim/LHS800` 中的 `TMRDL`、`BSMRDL` Snp | 为 `lhs100`、`lhs200`、`lhs400`、`lhs800`、`lhs100_lhs200_lhs400_lhs800` 五组数据提取 RDL 等效电路参数，输出到 `model_versions/v09_rdl_lhs_dataset_comparison/results/extracted_params/` |

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
| `model_versions/v01_matlab_mat_models/code/nn_train_3.m` | MATLAB RDL Top 训练脚本，读取 `training_datasets/RDL_Top_TD_4.csv`，导出 `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat4/RDL_Top_*.mat`。 |
| `model_versions/v01_matlab_mat_models/code/nn_train_tsv.m` | MATLAB TSV 训练脚本，读取 `training_datasets/TSV_TD_4.csv`，导出 `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat4/TSV_*.mat`。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/nn_train_3.m` | MATLAB RDL 提参结果训练脚本，遍历五组 LHS 数据和 `TMRDL`、`BSMRDL`，训练 `input -> 20 -> 20 -> 1` 参数网络并导出 MATLAB `.mat` 到 `model_versions/v09_rdl_lhs_dataset_comparison/models/matlab_param_nns/`。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/train_matlab_style_rdl_param_nns.py` | MATLAB CLI 崩溃时使用的 Python 等价训练入口，导出与 `nn_train_3.m` 相同字段的 MATLAB-compatible `.mat` 参数网络；本轮结果由该脚本生成。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/finetune_matlab_rdl_models_on_sparams.py` | 读取五组数据的 `.mat` 参数网络作为初始模型，以复数 S 参数为目标再次训练，输出 before/after 指标、微调模型和 test 最差样本图到 `model_versions/v09_rdl_lhs_dataset_comparison/results/sparam_finetuned_models/`。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/plot_lhs_dataset_model_comparison.py` | 读取 `summary_metrics.csv`，绘制五组不同样本数 RDL 模型的 test 精度对比图，默认保存并显示到 `model_versions/v09_rdl_lhs_dataset_comparison/results/dataset_model_comparison_plots/`。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/plot_lhs100_test_model_curve_comparison.py` | 固定使用 `LHS100/test` 数据，绘制 HFSS 与五个不同训练数据模型在同一 DUT 上的 S 参数曲线对比图，默认保存并显示到 `model_versions/v09_rdl_lhs_dataset_comparison/results/lhs100_test_model_curve_comparison/`。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/Connected_Network_Opt2.py` | LHS 级联连接网络优化提参入口；默认读取 `LHS100/train|val|test`、`LHS200/train`、`LHS400/train` 的 `TSV_RDL` Snp，使用 v09 RDL 单器件模型和 v03 TSV 单器件模型建立基础级联，再分别优化 with-Cn3 / without-Cn3 的 8 个连接网络参数；输出初步连接网络监督训练数据集到 `model_versions/v09_rdl_lhs_dataset_comparison/results/connection_network_lhs100_200_400_opt2/connection_network_params.csv`，并支持按样本文件续跑。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/train_lhs_connection_multihead_sparam.py` | 非 k-fold 的 LHS 级联模型训练入口；默认使用 `LHS100/train + LHS200/train + LHS400/train` 的 `TSV_RDL` 作为训练集，`LHS100/val` 和 `LHS100/test` 作为验证/测试集；基础级联中的 `TMRDL`、`BSMRDL` 读取 v09 `lhs100_lhs200_lhs400_lhs800` 单器件模型，`TSV` 读取 v03 单器件模型；优先读取 `connection_network_lhs100_200_400_opt2/connection_network_params.csv` 做连接参数预训练，再以整体结构复数 S 参数为目标微调；默认输出到 `model_versions/v09_rdl_lhs_dataset_comparison/results/connection_multihead_lhs100_200_400_v09_rdl_all_param_pretrain_sparam/`。 |
| `model_versions/v09_rdl_lhs_dataset_comparison/code/continue_lhs_connection_multihead_sparam.py` | 读取上一轮 `connection_multihead_lhs100_200_400_v09_rdl_all_param_pretrain_sparam/connection_param_multihead_net.pt` 作为初始模型，跳过连接参数预训练，仅以整体结构复数 S 参数为目标继续微调；默认输出到 `model_versions/v09_rdl_lhs_dataset_comparison/results/connection_multihead_lhs100_200_400_v09_rdl_all_sparam_continue/`。 |
| `model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10.py` | v10 直接运行入口；ADS 后端通过 `rdl_ads_sim/ADS_Sim.py` 和 `tsv_ads_sim/ADS_Sim.py` 分别生成 `TMRDL`、`BSMRDL`、`TSV` 单器件 S 参数，级联后逐样本优化 8 个 pi 网络元件值并保存 `pi_optimized_targets.csv`，再训练结构参数到 pi 元件值的初步模型 `pi_connection_net_param_pretrain.pt`，最后仅以整体 `TSV_RDL` 复数 S 参数为目标微调得到 `pi_connection_net.pt`；误差按 `S11/S21` 实部和虚部组成的 y 向量计算 NMSE；当前默认 signed-pi 版本不限制连接网络元件 scale 为正，优化器使用 `[-1e5, 1e5]` signed bounds，NN 输出反归一化后不再 clamp；默认使用 `LHS200/train` 随机 150 个样本建模、剩余 50 个样本测试，ADS 仿真时 `l_tmrdl/l_bsmrdl/h_tsv` 乘 `0.9`，输出到 `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09/`。 |
| `model_versions/v10_ads_pi_cascade/code/calibrate_ads_single_devices_v10.py` | v10 ADS 单器件小样本校准入口；使用少量 HFSS RDL/TSV 数据扫描 `er_si`、`cond`、`tand`、TSV `c1_scale` 及几何 scale，输出校准明细、汇总、推荐设置和对比图到 `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_small/`。 |
| `model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10_lhs800.py` | v10 LHS800 训练入口；使用 `HFSS_sim/LHS800/train/TSV_RDL` 全部 800 组样本训练当前小型 Pi-NN，并使用固定 50 组 LHS200 holdout 测试；输出和验证归档到 `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs800train_lhs200test_signed_pi_adslen09/`。 |
| `model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10_v09net.py` | v10 的 v09-style 大型 multi-head 网络试验入口；复用当前 150/50 ADS cache 和 `pi_optimized_targets.csv`，使用 `9 -> 256 -> 256 -> 128` trunk 及 8 个 `128 -> 64 -> 4` head 训练 v10 Pi 参数；结果归档到 `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_v09net/`。 |
| `model_versions/v10_ads_pi_cascade/code/continue_sparam_unbounded_v10.py` | 读取当前 150/50 signed-pi ADS length 0.9 checkpoint，取消 NN 输出参数范围限制，仅以整体复数 S 参数为目标继续学习；输出和验证归档到 `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_unbounded_sparam_continue/`。 |
| `model_versions/v10_ads_pi_cascade/code/regenerate_comparison_plots_v10.py` | v10 对比图刷新入口；读取已保存的 `pi_connection_net.pt`、ADS 单器件缓存和指标 CSV，不重新训练，覆盖生成 `S11/S21` 实部和虚部对比图。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/extract_rdl_connection2_params.py` | v12 RDL Connection2 专用提参入口；从 `HFSS_sim/LHS400_Connection2/train/RDL` 重新提取通用 RDL 等效电路参数，输出到 `model_versions/v12_hfss_v08_multihead_chain/results/rdl_connection2_extracted_params/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/train_rdl_connection2_sparam_model.py` | v12 RDL Connection2 单器件训练入口；读取重新提参 CSV，训练通用 RDL 参数网络并以同批 HFSS S 参数微调，输出模型、指标和对比图到 `model_versions/v12_hfss_v08_multihead_chain/results/rdl_connection2_sparam_model/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/extract_tsv_connection2_params.py` | v12 TSV Connection2 专用提参入口；从 `HFSS_sim/LHS400_Connection2/train/TSV` 重新提取 TSV 等效电路参数，输出到 `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_extracted_params/`，不覆盖旧 `training_datasets/TSV_TD_4.csv`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/train_tsv_connection2_sparam_model.py` | v12 TSV Connection2 单器件训练入口；读取重新提参 CSV，训练 TSV 参数网络并以同批 HFSS S 参数微调，输出模型、指标和对比图到 `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_sparam_model/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/continue_tsv_connection2_sparam_model.py` | v12 TSV Connection2 单器件 S 参数继续训练入口；读取已训练 checkpoint，仅以复数 S 参数为目标继续优化，输出 before/after 指标、对比图和验证归档到 `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_sparam_continue/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/train_v12_hfss_v08_symmetric_multihead.py` | v12 直接运行入口；读取 `LHS150_50_Connection2/train|test/TSV_RDL` 整体目标，使用 LHS400_Connection2/HFSS 派生的新 RDL checkpoint 和新 TSV checkpoint 构建 13 段基础级联，逐样本优化共享 v08 7 参数 pi 电路，再训练共享参数网络并展开为 12 个输出头，以 `S11/S21` 幅值和 wrapped phase 微调；当前新 RDL/TSV backend 默认输出到 `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_new_rdl_tsv/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/continue_v12_all_train_sparam.py` | v12 整体模型 S 参数继续训练入口；读取新 RDL/TSV 整体模型 checkpoint，禁用优化质量筛选，使用原始 150 个 train 样本全部继续训练，并在原始 50 个 test 样本上评估，输出到 `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/continue_v12_all_train_sparam_round2.py` | v12 整体模型第二轮 S 参数继续训练入口；读取第一轮 all150 checkpoint，以较低学习率继续用原始 150 个 train 样本训练，并在原始 50 个 test 样本上评估，输出到 `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round2/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/continue_v12_all_train_sparam_round3.py` | v12 整体模型第三轮 S 参数继续训练入口；读取第二轮 all150 checkpoint，以 `1e-6` 学习率继续用原始 150 个 train 样本训练，并在原始 50 个 test 样本上评估，输出到 `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round3/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/continue_v12_all_train_sparam_round4.py` | v12 整体模型第四轮 S 参数继续训练入口；读取第三轮 all150 checkpoint，以 `5e-7` 学习率继续训练并检查继续优化是否过拟合，输出到 `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round4/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/recompute_v12_paper_nmse.py` | v12 论文风格误差复算入口；按照 Ye 2026 论文的 NMSE 公式，在 `Re/Im(S11,S21)` 线性曲线上重新汇总整体模型误差，并输出小数和百分数结果到 `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_new_rdl_tsv/paper_nmse_recalculation/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/recompute_single_device_paper_nmse.py` | v12 单器件论文风格误差复算入口；按照 Ye 2026 论文的 NMSE 公式，在 `Re/Im(S11,S21)` 线性曲线上重新汇总当前 RDL 和 TSV 单器件模型误差，输出到 `model_versions/v12_hfss_v08_multihead_chain/results/single_device_paper_nmse_recalculation/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/export_v12_best5_sparams_csv.py` | v12 最优样本 S 参数导出入口；按当前 round3 最佳模型在 test 集的 `v08_nn_nmse_s11_s21_ri` 选取前 5 个样本，将 HFSS 仿真、直接级联和级联模型的 `S11/S21` 实部和虚部以宽表 CSV 导出到 `model_versions/v12_hfss_v08_multihead_chain/results/best5_sparameter_csv_current_best_round3/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/plot_v12_single_device_model_vs_hfss.py` | v12 单器件对比图入口；读取 `HFSS_sim/LHS400/train/TMRDL|BSMRDL|TSV` 的独立单器件 HFSS 数据，分别调用当前 v12 使用的 RDL/TSV 等效电路单器件模型，输出全样本指标、汇总和随机/最差样本曲线图到 `model_versions/v12_hfss_v08_multihead_chain/results/single_device_model_vs_hfss_lhs400/`。 |
| `model_versions/v12_hfss_v08_multihead_chain/code/plot_v12_tsv_connection2_model_vs_hfss.py` | v12 TSV Connection2 对比图入口；读取 `HFSS_sim/LHS400_Connection2/train/TSV`，使用 `[r_tsv, h_tsv, pitch]` 直接输入 TSV 单器件模型，输出全样本指标、汇总和随机/最差样本曲线图到 `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_model_vs_hfss/`。 |
| `model_versions/v07_connection_param_and_sparam_finetune/code/train_connection_network_params.py` | 读取 `connection_network_params.csv`，默认训练 `optimized_with_cn3` 的 DUT 级 56 输出 scale 参数模型；可在脚本顶部把 `TARGET_VARIANT` 改为 `optimized_without_cn3` 训练 48 输出模型，并输出训练后整体结构 S 参数预测效果图；默认输出到 `model_versions/v07_connection_param_and_sparam_finetune/results/connection_network_param_model_optimized_with_cn3/`。 |
| `model_versions/v07_connection_param_and_sparam_finetune/code/fine_tune_connection_network_on_sparams.py` | 读取上一轮 S 参数微调得到的 `connection_param_net_sparam_finetuned.pt` 作为连接网络初始模型，同时学习 9 段器件 length scale 和 8 个连接网络元件值，并以整体结构 HFSS 复数 S 参数为目标训练；默认输出到 `model_versions/v07_connection_param_and_sparam_finetune/results/connection_network_sparam_finetune_optimized_with_cn3_with_device_scales/`。 |
| `model_versions/v08_connection_multihead/code/train_connection_network_multihead_sparam.py` | 使用共享 trunk + 8 个连接位置 head 的多头网络训练 with-Cn3 连接网络参数，固定器件长度缩放为 0.95；当前默认剔除整体结构 HFSS 中 `S21 mean < -15 dB` 的异常 DUT，使用严格 `train/val/test = 70/15/15` 划分，基础级联使用已微调的单器件 S 参数模型，先用优化连接参数预训练，再以整体结构 HFSS 复数 S 参数为主目标训练；默认输出到 `model_versions/v08_connection_multihead/results/connection_network_multihead_sparam_with_cn3_rigorous_unconstrained/`，对比图默认保存 test 集 `multihead_mse_vs_hfss` 最差的 10 个 DUT。 |
| `model_versions/v08_connection_multihead/code/train_connection_network_multihead_sparam_kfold.py` | 对当前严格无约束 multi-head 级联建模流程做 5-fold 验证；每个 fold 中一份作为 test、下一份作为 val、剩余三份作为 train，剔除同一批 `S21 mean < -15 dB` 异常 DUT，输出每个 fold 的模型、指标和 test 最差 10 图；默认输出到 `model_versions/v08_connection_multihead/results/connection_network_multihead_sparam_with_cn3_kfold/`。 |
| `model_versions/v08_connection_multihead/code/plot_best_kfold_multihead_sparam.py` | 从 `connection_network_multihead_sparam_with_cn3_kfold` 的 test 指标中选取 `multihead_mse_vs_hfss` 最小的样本，绘制效果较好的 HFSS / Direct / Optimized / Multi-head 对比图；输出到 `model_versions/v08_connection_multihead/results/connection_network_multihead_sparam_with_cn3_kfold/best_multihead_sparam_plots/`。 |
| `model_versions/v03_single_device_sparam_finetune/code/train_single_device_sparam_model.py` | 单器件 Python/PyTorch S 参数目标训练试验；当前默认批量训练 `RDL_Top` 和 `TSV`，读取 MATLAB `mat4` 权重作为初始模型，跳过 CSV 参数预训练，直接以 HFSS 单器件复数 S 参数为目标继续微调，并和 MATLAB `mat4` 基线比较；输出到 `model_versions/v03_single_device_sparam_finetune/results/single_device_sparam_<器件名>_mat4_init_sparam_noanchor/`，汇总 CSV 为 `model_versions/v03_single_device_sparam_finetune/results/single_device_mat4_init_sparam_summary.csv`。 |
| `model_versions/v04_hfss_split_direct_sparam/code/train_hfss_split_sparam_models.py` | 读取 `HFSS_sim/simdata_TSV_RDL_Connection/train|val|test/` 新 HFSS split 数据，直接训练 `TMRDL`、`BSMRDL`、`TSV` 单器件 S 参数代理模型，再用 `TMRDL -> TSV -> BSMRDL` 级联基线加 S 参数残差网络训练 `TSV_RDL` 整体模型；默认输出到 `model_versions/v04_hfss_split_direct_sparam/results/hfss_split_sparam_models/`，并保存 test 集最差样本对比图。 |
| `model_versions/v05_hfss_split_circuit_param/code/train_hfss_split_circuit_param_models.py` | 新 HFSS split 数据的物理参数流程：先对 `TMRDL`、`BSMRDL`、`TSV` 单器件 Snp 提取 `R/L/C/G` 等效电路参数，再训练结构参数到电路参数的 NN，并通过电路公式以 S 参数目标继续微调；输出到 `model_versions/v05_hfss_split_circuit_param/results/hfss_split_circuit_param_models/`。 |
| `model_versions/v05_hfss_split_circuit_param/code/train_hfss_split_connection_cascade.py` | 在物理单器件模型基础上，对 `TSV_RDL` 每个样本优化两段连接网络元件值，再训练整体结构参数到连接网络参数的 NN，并以级联 S 参数目标微调；当前使用正元件边界版本，输出到 `model_versions/v05_hfss_split_circuit_param/results/hfss_split_connection_cascade_positive_bounds/`。 |

### 模型对比

| 脚本 | 默认对比对象 | 默认输出 |
| --- | --- | --- |
| `model_versions/v01_matlab_mat_models/code/compare_rdl_bottom_models.py` | RDL Bottom `.mat` / PyTorch 模型 vs HFSS | `model_versions/v01_matlab_mat_models/results/RDL_Bottom_model_compare/` |
| `model_versions/v01_matlab_mat_models/code/compare_rdl_top_models.py` | RDL Top `.mat` 模型 vs HFSS | `model_versions/v01_matlab_mat_models/results/RDL_Top_model_compare/` |
| `model_versions/v01_matlab_mat_models/code/compare_tsv_models.py` | TSV `.mat` 模型 vs HFSS | `model_versions/v01_matlab_mat_models/results/TSV_model_compare/` |
| `model_versions/v02_mat4_cascade_and_sparameter_optimization/code/compare_model_cascade_results.py` | RDL_TSV 整体结构 HFSS vs `RDL_TSV_mat4` 单器件直接级联；遍历每个 DUT 并直接显示对比图，不保存图文件 | `model_versions/v02_mat4_cascade_and_sparameter_optimization/results/RDL_TSV_mat4_cascade_compare/` |
| `model_versions/v02_mat4_cascade_and_sparameter_optimization/code/compare_optimized_sparameter_results.py` | 查看 `Calc_SP_and_Opt2.py` 保存的优化后 S 参数效果；默认显示所有已保存 DUT 的对比图 | `model_versions/v02_mat4_cascade_and_sparameter_optimization/results/RDL_TSV_mat4_opt2/` |

`compare_rdl_tsv_mat4_cascade.py` 直接在 VS Code 中运行时会使用交互式绘图后端；每读取并计算一个 `dut*.s2p` 后立即显示一张对比图。

对比输出通常包含：

- `compare_report.json`
- `*_model_compare_summary.csv`
- `*_model_compare_aggregate.csv`
- `*_model_compare_compact.csv`
- `summary_error_trends.png`
- `plots/` 下的前几个 DUT 对比图和 worst-case 图

### 早期 S 参数计算脚本

`model_versions/v99_legacy_and_shared/code/` 保存早期计算、级联、优化和 NN 比较脚本：

- `Calc_SP.py`
- `Calc_SP_and_Opt.py`
- `Calc_SP_and_Opt2.py`：默认读取 `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat4/`，遍历 `RDL_TSV_Snp` 中所有 DUT，对直接级联结果插入修正网络并做 least-squares 优化；基础 RDL/TSV 段长度按 `DEVICE_LENGTH_SCALE=0.95` 缩放，with/without-Cn3 连接网络均使用原始宽 bounds `(-1e5, 1e5)`，不添加正元件限制或正则项；默认多进程并行优化多个 DUT，可在脚本顶部用 `PARALLEL_OPTIMIZATION` 和 `MAX_WORKERS` 调整；优化后的 Touchstone、连接网络元件值、器件结构参数和边界命中统计保存到 `model_versions/v02_mat4_cascade_and_sparameter_optimization/results/RDL_TSV_mat4_opt2/`。
- `Calc_SP_NN.py`
- `CONN.py`
- `CONN1.py`

这些脚本保留用于复查旧流程或复用函数，新工作优先使用对应版本目录下的 `code/` 入口。

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

## VS Code 入口

直接在 VS Code 中打开对应版本目录下的 `code/` 脚本运行；不需要额外命令行参数。常用入口按版本查找：

- `model_versions/v00_parameter_extraction_and_dataset_building/code/`：参数提取和训练 CSV 构建。
- `model_versions/v01_matlab_mat_models/code/`：MATLAB `.mat` 模型训练和 RDL/TSV 基线对比。
- `model_versions/v02_mat4_cascade_and_sparameter_optimization/code/`：mat4 级联对比和 S 参数优化。
- `model_versions/v03_single_device_sparam_finetune/code/`：单器件 S 参数微调。
- `model_versions/v04_hfss_split_direct_sparam/code/`：HFSS split 直接 S 参数模型。
- `model_versions/v05_hfss_split_circuit_param/code/`：HFSS split 电路参数模型和连接级联。
- `model_versions/v07_connection_param_and_sparam_finetune/code/`：连接网络参数模型和整体 S 参数微调。
- `model_versions/v08_connection_multihead/code/`：multi-head 连接网络、k-fold 和优秀样本对比图绘制。
- `model_versions/v09_rdl_lhs_dataset_comparison/code/`：RDL LHS 数据规模对比和 LHS 级联训练，包含提参、参数网络训练、S 参数微调、连接网络优化提参和级联 multi-head 训练。
- `model_versions/v10_ads_pi_cascade/code/`：ADS 单器件仿真驱动的 pi 型级联网络训练入口。
- `model_versions/v12_hfss_v08_multihead_chain/code/`：v12 HFSS 等效电路单器件 + v08 7 参数 pi 长链共享优化与 12 输出头训练入口。
- `model_versions/v99_legacy_and_shared/code/`：早期 S 参数计算、级联、优化和 NN 比较脚本。

如果直接导入 Python 包，先在 VS Code 中把项目根目录下的 `src/` 加入解释器搜索路径。

## 结果目录

当前模型结果已按版本整理到 `model_versions/*/results/`。主要入口：

- `model_versions/v01_matlab_mat_models/results/RDL_Bottom_model_compare/`
- `model_versions/v01_matlab_mat_models/results/RDL_Top_model_compare/`
- `model_versions/v01_matlab_mat_models/results/TSV_model_compare/`

后续训练或对比输出建议继续放在对应版本目录：

- `model_versions/v01_matlab_mat_models/results/`：MATLAB `.mat` 基线模型对比报告、CSV、图。
- `model_versions/v02_mat4_cascade_and_sparameter_optimization/results/`：mat4 级联对比和 S 参数优化结果。
- `model_versions/v03_single_device_sparam_finetune/results/`：单器件 S 参数微调结果。
- `model_versions/v04_hfss_split_direct_sparam/results/` 到 `model_versions/v08_connection_multihead/results/`：HFSS split、RDL refinement、连接网络和 multi-head 训练结果。
- `archive/model_results_unmatched_20260703/`：没有明确对应模型版本的旧结果归档。
- `archive/model_versions_v06_rdl_mat4style_and_lhs_refinement_20260706_155014/`：旧 v06 RDL mat4-style 和 LHS refinement 代码、模型、结果归档。

## Git 与文件管理

当前 Git 主要管理：

- `src/`
- `model_versions/*/code/`
- `model_versions/`
- `snp_data/`
- `model_versions/v01_matlab_mat_models/models/`
- `training_datasets/`
- `README.md`
- `.gitignore`

默认忽略：

- `archive/`
- `model_versions/*/results/` 中除 `README.md` 外的大体量结果文件
- `model_results/`（废弃的旧结果根目录）
- `__pycache__/`
- `*.pyc`
- `*.pth`
- `*.pt`
- `*.png`

注意事项：

- 不要恢复旧的根目录 `data/`、`outputs/`、`models/`；新文件应放入上面的语义目录。
- 新模型版本优先新增 `model_versions/vXX_<name>/`，并在该版本的 `README.md` 与 `results/README.md` 中登记代码入口和结果路径。
- 包含中文注释或中文文件名的脚本统一按 UTF-8 打开和保存，避免用 ANSI/GBK 另存。
- `archive/` 仅用于归档，不作为当前流程入口。

## 本次补充：LHS800 TSV dut700 RLGC 对比

- 入口脚本：`model_versions/v09_rdl_lhs_dataset_comparison/code/compare_lhs800_tsv_dut700_rlgc.py`
- 默认输入：`HFSS_sim/LHS800/train/TSV/dut700.s2p` 和 `HFSS_sim/LHS800/train/TSV/dut_700.s2p`
- 默认输出：`model_versions/v09_rdl_lhs_dataset_comparison/results/rlgc_compare_dut700/`
- 输出内容：`dut700_rlgc.csv`、`dut_700_rlgc.csv`、`rlgc_comparison.csv`、`summary.json`、RLGC 曲线图、相对误差图和验证归档文本。

## 本次补充：Connection2 TSV_RDL 150/50 LHS HFSS 数据

- 入口脚本：`HFSS_sim/batchsim_150train_50test_connection2.py`
- HFSS 工程：`C:\ffzhzh\LocalFiles\Ansys_Project_Files\TSV_RDL_Connection2.aedt`
- Design name：`TSV_RDL`
- 默认输出：`HFSS_sim/LHS150_50_Connection2/`
- 数据划分：`train=150`，输出到 `HFSS_sim/LHS150_50_Connection2/train/TSV_RDL/dut0.s2p` 到 `dut149.s2p`；`test=50`，输出到 `HFSS_sim/LHS150_50_Connection2/test/TSV_RDL/dut150.s2p` 到 `dut199.s2p`
- 采样范围：`h_tsv=50~100`，`r_tsv=5~15`，`pitch=40~60`，`l_tmrdl=100~700`，`w_tmrdl=10~30`，`t_tmrdl=2~5`，`l_bsmrdl=100~700`，`w_bsmrdl=10~30`，`t_bsmrdl=2~5`
- 说明：采样记录 CSV 中保留 `t_tmrdl`、`t_bsmrdl` 列名；仿真写入 HFSS 时映射到工程变量 `h_tmrdl`、`h_bsmrdl`
- 稳定性设置：默认 `NUM_PARALLEL=3`，用于降低多个 AEDT 进程并行时的资源压力；脚本会跳过已有 `.s2p` 并续跑缺失 DUT
- 验证归档：`HFSS_sim/validation/`

同一入口脚本也生成 Connection2 单器件 LHS400 数据：

- RDL design name：`RDL`
- TSV design name：`TSV`
- 默认输出：`HFSS_sim/LHS400_Connection2/train/RDL/`、`HFSS_sim/LHS400_Connection2/train/TSV/` 和 `HFSS_sim/LHS400_Connection2/train/TSV_RDL/`
- RDL 数据：`train=400`，`dut0.s2p` 到 `dut399.s2p`，采样范围为 `pitch=40~60`，`l_tmrdl=100~700`，`w_tmrdl=10~30`，`t_tmrdl=2~5`
- TSV 数据：`train=400`，`dut0.s2p` 到 `dut399.s2p`，采样范围为 `h_tsv=50~100`，`r_tsv=5~15`，`pitch=40~60`
- TSV_RDL 数据：`train=400`，`dut0.s2p` 到 `dut399.s2p`，采样范围为 `h_tsv=50~100`，`r_tsv=5~15`，`pitch=40~60`，`l_tmrdl=100~700`，`w_tmrdl=10~30`，`t_tmrdl=2~5`，`l_bsmrdl=100~700`，`w_bsmrdl=10~30`，`t_bsmrdl=2~5`
- 厚度映射：RDL 采样记录 CSV 中保留 `t_tmrdl` 列名；仿真写入 HFSS 时映射到工程变量 `h_tmrdl`

## v10 ADS Refined Single-Device Calibration

- Entry script: `model_versions/v10_ads_pi_cascade/code/calibrate_ads_single_devices_v10_refined.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_refined/`
- Scope: six LHS200 DUTs, refined ADS-vs-HFSS single-device calibration around the first calibration result.
- Best RDL: `er_si=10.2`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`, `w_scale=0.85`, `pitch_scale=1.0`, `h_tsv_scale=1.2`, `h_rdl_scale=1.0`, NMSE mean `0.008500`.
- Best TSV: baseline settings, `er_si=11.9`, `cond=5.8e7`, `tand=0.005`, `c1_scale=1.0`, `pitch_scale=1.0`, `h_tsv_scale=1.0`, `d_scale=1.0`, NMSE mean `0.002484`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_refined/validation_archive.md`
## v10 ADS 16-DUT Single-Device Calibration

- Entry script: `model_versions/v10_ads_pi_cascade/code/calibrate_ads_single_devices_v10_16dut.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_16dut/`
- Scope: 16 evenly spaced LHS200 DUTs: `100, 113, 126, 140, 153, 166, 180, 193, 206, 219, 233, 246, 259, 273, 286, 299`.
- Best RDL: `er_si=10.2`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`, `w_scale=0.85`, `pitch_scale=1.0`, `h_tsv_scale=1.0`, `h_rdl_scale=0.8`, NMSE mean `0.006183`.
- Best TSV: `er_si=11.9`, `cond=5.8e7`, `tand=0.005`, `c1_scale=1.0`, `pitch_scale=1.0`, `h_tsv_scale=1.1`, `d_scale=1.0`, NMSE mean `0.001169`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_16dut/validation_archive.md`
## v10 ADS 16-DUT Additional Comparison Plots

- Entry script: `model_versions/v10_ads_pi_cascade/code/plot_ads_single_device_calibration_16dut.py`
- Output plots: `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_16dut/plots_all_best/`
- Summary CSV: `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_16dut/plots_all_best_summary.csv`
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_single_device_calibration_16dut/plots_all_best_validation.md`
- Plot count: `48`, covering TMRDL, BSMRDL, and TSV for all 16 calibration DUTs.
## v10 ADS Calibrated 16-DUT LHS200 100/100 Training

- Entry script: `model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10_calibrated16dut_lhs200_random100.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random100train_100test_calibrated16dut/`
- Split: `LHS200/train`, random seed `20260707`, `train=100`, `test=100`, overlap `0`.
- ADS settings: 16-DUT calibrated RDL/TSV settings from `ads_single_device_calibration_16dut`; global ADS length multiplier set to `1.0` for consistency with calibration.
- Final test metrics: direct NMSE mean `0.254311`, final Pi-NN NMSE mean `0.046348`, S11 MAE `2.952143 dB`, S21 MAE `0.406668 dB`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random100train_100test_calibrated16dut/validation_archive.md`
## v10 ADS Calibrated 16-DUT Best-Test Plots

- Entry script: `model_versions/v10_ads_pi_cascade/code/plot_best_calibrated16dut_lhs200_random100.py`
- Output plots: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random100train_100test_calibrated16dut/comparison_plots/best_test/`
- Summary CSV: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random100train_100test_calibrated16dut/best_test_comparison_plots.csv`
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs200_random100train_100test_calibrated16dut/best_test_comparison_plots_validation.md`
- Plot count: `8`; best test sample is `LHS200_train_dut240` with final Pi-NN NMSE `0.003328`.
## v10 ADS LHS150_50_Connection2 0.1-100 GHz Training

- Entry script: `model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10_connection2_150_50_100ghz.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_calibrated16dut/`
- Dataset: `HFSS_sim/LHS150_50_Connection2/train` for modeling (`150` samples), `HFSS_sim/LHS150_50_Connection2/test` for testing (`50` samples).
- Frequency grid: `0.1` to `100.0` GHz, `1000` points; ADS generated netlists use `Start=0.1 GHz Stop=100 GHz Step=0.1 GHz`.
- Modeling flow: ADS single-device simulation, pi optimization, structure-to-pi parameter pretraining, and S-parameter fine-tuning.
- Final test metrics: direct NMSE mean `0.922865`, optimized pi NMSE mean `0.247965`, final Pi-NN NMSE mean `0.823552`, S11 MAE `5.965828 dB`, S21 MAE `23.157745 dB`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_calibrated16dut/validation_archive.md`

## v10 ADS RDL Netlist Update Calibration

- Entry script: `model_versions/v10_ads_pi_cascade/code/calibrate_ads_single_devices_v10_16dut_rdl_net_update.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_cal_rdl_update16/`
- Scope: RDL-only 16-DUT recalibration after the RDL ADS netlist update; TSV settings are carried over from `ads_single_device_calibration_16dut`.
- Best RDL settings: `er_si=10.2`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`, `w_scale=0.85`, `pitch_scale=1.1`, `h_tsv_scale=1.0`, `h_rdl_scale=1.0`.
- Best RDL mean NMSE: `0.027345`; previous 16-DUT RDL mean NMSE was `0.006183`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_cal_rdl_update16/validation_archive.md`

## v10 ADS LHS400_Connection2 Random-10 RDL Calibration

- Entry script: `model_versions/v10_ads_pi_cascade/code/calibrate_ads_rdl_lhs400_connection2_random10.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_cal_rdl_lhs400c2_rand10/`
- Dataset: `HFSS_sim/LHS400_Connection2/train/RDL`, random seed `20260708`, DUTs `12, 45, 116, 150, 187, 242, 269, 298, 335, 373`.
- ADS RDL netlist: restored original `MCLIN` template with `MSUB H=h_tsv`; fixed `h_tsv=100um` for this standalone RDL dataset; sweep `0.1-100 GHz`.
- Best RDL settings: `er_si=12.5`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`, `w_scale=0.8`, `pitch_scale=1.0`, `h_tsv_scale=1.0`, `h_rdl_scale=1.0`.
- Best RDL mean NMSE: `0.129976`; validation archive: `model_versions/v10_ads_pi_cascade/results/ads_cal_rdl_lhs400c2_rand10/validation_archive.md`

## v10 ADS LHS400_Connection2 Random-10 RDL Calibration, MLIN

- Entry script: `model_versions/v10_ads_pi_cascade/code/calibrate_ads_rdl_lhs400_connection2_random10.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_cal_rdl_lhs400c2_rand10_mlin/`
- Dataset: `HFSS_sim/LHS400_Connection2/train/RDL`, random seed `20260708`, DUTs `12, 45, 116, 150, 187, 242, 269, 298, 335, 373`.
- ADS RDL netlist: updated `MLIN2` template with `MSUB H=pitch-w_rdl`; sweep `0.1-100 GHz`.
- Best RDL settings: `er_si=9.8`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`, `w_scale=0.8`, `pitch_scale=1.1`, `h_tsv_scale=1.0`, `h_rdl_scale=1.0`.
- Best RDL mean NMSE: `0.082460`; validation archive: `model_versions/v10_ads_pi_cascade/results/ads_cal_rdl_lhs400c2_rand10_mlin/validation_archive.md`

## v10 ADS LHS400_Connection2 Random-10 RDL and TSV Calibration

- Entry script: `model_versions/v10_ads_pi_cascade/code/calibrate_ads_lhs400_connection2_rdl_tsv_random10.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ac_l400_rdl_tsv10/`
- Dataset: `HFSS_sim/LHS400_Connection2/train`, RDL seed `20260708`, TSV seed `20260709`.
- RDL DUTs: `12, 45, 116, 150, 187, 242, 269, 298, 335, 373`.
- TSV DUTs: `20, 85, 95, 143, 191, 217, 267, 316, 330, 338`.
- ADS templates: RDL updated `MLIN2`; TSV updated `d_tsv`; sweep `0.1-100 GHz`.
- Best RDL settings: `er_si=9.4`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`, `w_scale=0.75`, `pitch_scale=1.15`, `h_tsv_scale=1.0`, `h_rdl_scale=1.0`; mean NMSE `0.062834`.
- Best TSV settings: `er_si=11.9`, `cond=5.8e7`, `tand=0.005`, `c1_scale=1.0`, `pitch_scale=0.9`, `h_tsv_scale=1.1`, `d_scale=1.0`; mean NMSE `0.009993`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ac_l400_rdl_tsv10/validation_archive.md`

## v10 ADS LHS400_Connection2 Refined RDL and TSV Calibration

- Entry script: `model_versions/v10_ads_pi_cascade/code/calibrate_ads_lhs400_connection2_rdl_tsv_random10_refined.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ac_l400_ref2/`
- Dataset: same random-10 RDL and TSV samples as `ac_l400_rdl_tsv10`; sweep `0.1-100 GHz`.
- Best RDL settings: `er_si=9.8`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`, `w_scale=0.65`, `pitch_scale=1.25`, `h_tsv_scale=1.0`, `h_rdl_scale=1.0`; mean NMSE `0.039423`.
- Best TSV settings: `er_si=11.9`, `cond=5.8e7`, `tand=0.005`, `c1_scale=1.0`, `pitch_scale=1.0`, `h_tsv_scale=1.2`, `d_scale=1.0`; mean NMSE `0.007154`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ac_l400_ref2/validation_archive.md`

## v10 ADS LHS150_50_Connection2 Cascade With Refined LHS400 ADS

- Entry script: `model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads/`
- Dataset: `HFSS_sim/LHS150_50_Connection2`, using 150 train and 50 test samples over `0.1-100 GHz`.
- ADS settings: refined single-device settings from `ac_l400_ref2`; modeling flow unchanged.
- Per-sample optimized pi test NMSE mean: `0.020980`; final Pi-NN test NMSE mean: `0.549241`.
- Final test S11 MAE: `4.878773 dB`; final test S21 MAE: `5.348503 dB`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads/validation_archive.md`

## v10 ADS LHS150_50_Connection2 S-Parameter Continuation

- Entry script: `model_versions/v10_ads_pi_cascade/code/continue_sparam_connection2_refined_lhs400_ads.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_sparam_continue/`
- Source checkpoint: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads/pi_connection_net.pt`
- Training target: pure complex S-parameter loss only, `160` epochs, learning rate `8e-6`.
- Test NMSE mean improved from `0.549241` to `0.476691`; test NMSE median improved from `0.466522` to `0.324145`.
- Final test S11 MAE: `4.810463 dB`; final test S21 MAE: `4.261821 dB`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_sparam_continue/validation_archive.md`

## v10 ADS LHS150_50_Connection2 Filtered Training Trial

- Entry script: `model_versions/v10_ads_pi_cascade/code/train_filtered_connection2_refined_lhs400_ads.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_filtered_train20/`
- Filter rule: exclude the 20 highest-error train samples from the S-parameter continuation run; test samples are not filtered.
- Final filtered test NMSE mean: `0.520745`; test NMSE median: `0.405490`.
- Kept-train NMSE mean: `0.178825`; excluded-train NMSE mean: `0.912100`.
- Result: better than the pre-continuation full-data model (`0.549241`) but worse than S-parameter continuation (`0.476691`).
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_filtered_train20/validation_archive.md`

## v10 ADS LHS150_50_Connection2 Element-Wise Multi-Head NN

- Entry script: `model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10_connection2_refined_lhs400_ads_element_heads.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_element_heads/`
- Network: for each pi element type, shared `9->30->30` trunk; for each connection position, `30->20->1` head. Output remains `8*4=32`.
- Final test NMSE mean: `0.460406`; test NMSE median: `0.415037`.
- Final test S11 MAE: `4.968888 dB`; final test S21 MAE: `3.660034 dB`.
- Result: currently better than S-parameter continuation (`0.476691`) and filtered retraining (`0.520745`).
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_element_heads/validation_archive.md`

## v10 ADS LHS150_50_Connection2 Optimized-Pi Filtered NN

- Entry script: `model_versions/v10_ads_pi_cascade/code/train_optfilter_connection2_refined_lhs400_ads.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_optfilter005/`
- Filter rule: exclude train and test samples with per-sample `optimized_pi_nmse_s11_s21_ri > 0.05` before neural-network training/evaluation.
- Excluded samples: 2 train samples and 3 test samples; active split is 148 train and 47 test.
- Active test NMSE mean: `0.444260`; active test NMSE median: `0.398114`.
- Excluded test NMSE mean: `0.799328`.
- Note: active test error is lower than the full 50-sample element-wise run (`0.460406`), but three hard-to-fit test samples were removed.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_optfilter005/validation_archive.md`

## v10 ADS LHS150_50_Connection2 V08-Circuit Shared-to-Multihead NN

- Entry script: `model_versions/v10_ads_pi_cascade/code/train_ads_v08circuit_shared_to_multihead_connection2_refined_lhs400_ads.py`
- Output: `model_versions/v10_ads_pi_cascade/results/ads_v08circuit_shared_to_multihead_lhs150_50_connection2_100ghz_lhs400_refined_ads/`
- Circuit: v08 with-Cn3 connection network with 7 scale parameters, replacing the v10 four-element pi network for this trial.
- Optimization: each sample fits one shared 7-parameter connection circuit and inserts the same circuit at all eight connection positions.
- Filter rule: before neural-network training, exclude samples with `optimized_v08_shared_nmse_s11_s21_ri > 0.3`; excluded samples are reported separately in `excluded_optimized_v08_shared_samples.csv`.
- Excluded samples: 33 train samples and 11 test samples; active split is 117 train and 39 test.
- Network: first train seven independent `9->30->30->20->1` scalar networks; then expand each parameter network into eight `30->20->1` connection-position heads and fine-tune with `S11`/`S21` magnitude and wrapped phase loss.
- Optimized shared-circuit test NMSE mean: `0.176276`; test NMSE median: `0.135372`.
- Final active-test NMSE mean: `0.389323`; active-test NMSE median: `0.300856`.
- Final active-test magnitude-phase MSE mean: `0.680871`.
- Final active-test S11 MAE: `5.334023 dB`; active-test S21 MAE: `3.570800 dB`.
- Excluded-test NMSE mean: `0.574713`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_v08circuit_shared_to_multihead_lhs150_50_connection2_100ghz_lhs400_refined_ads/validation_archive.md`

## v11 ADS Long-Chain V08-Circuit Shared-to-Multihead Setup

- Version path: `model_versions/v11_ads_v08_multihead_chain/`
- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_shared7_to_multihead12.py`
- Structure: `TMRDL-TSV-BSMRDL-TSV-TMRDL-TSV-BSMRDL-TSV-TMRDL-TSV-BSMRDL-TSV-TMRDL`, with 13 device blocks and 12 connection positions.
- Circuit: Appendix-1 7-parameter pi/connection network (`Cn1/Rn1/Cn2/Rn2/Cn3/Rn3/Ln1` scale outputs).
- Network: first trains seven independent `9->30->30->20->1` scalar networks, then expands each parameter network into 12 `30->20->1` connection heads for S-parameter fine-tuning.
- Data status: `HFSS_sim/LHS400_Connection2/train/RDL` and `TSV` can support ADS single-device calibration, but the full-chain v11 HFSS target is not present yet.
- Required target inputs before training can proceed: `HFSS_sim/LHS400_Connection2/train/V11_RDL_TSV_CHAIN_variations_record.csv` and `HFSS_sim/LHS400_Connection2/train/V11_RDL_TSV_CHAIN/dut*.s2p`.
- If the target is missing, the entry archives the blocked validation result under `model_versions/v11_ads_v08_multihead_chain/results/ads_v08circuit_shared_to_multihead12_lhs400_connection2/` instead of training against the wrong target.
- Current blocked-run artifacts: `model_report.md`, `training_report.json`, `validation_archive.md`, `data_readiness_summary.csv`, and `data_readiness_plot.png` under `model_versions/v11_ads_v08_multihead_chain/results/ads_v08circuit_shared_to_multihead12_lhs400_connection2/`.

## v11 ADS LHS150_50_Connection2 Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_shared7_to_multihead12.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/ads_v08circuit_shared_to_multihead12_lhs150_50_connection2/`
- Dataset: according to `建模流程.md` Appendix 6, train target uses `HFSS_sim/LHS150_50_Connection2/train/TSV_RDL`, test target uses `HFSS_sim/LHS150_50_Connection2/test/TSV_RDL`.
- ADS cache: reuses `model_versions/v10_ads_pi_cascade/results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads/ads_single_device_cache/`.
- Samples: 200 loaded; active after optimized-NMSE filter is 104 train, 12 val, and 45 test; 39 samples are excluded by `optimized_v08_shared_nmse_s11_s21_ri > 0.3`.
- Training: parameter pretrain completed 31 epochs; S-parameter fine-tune completed 90 epochs.
- Final active-test result: ADS direct cascade NMSE mean `0.296451`; final v11 NN NMSE mean `0.622891`; final v11 NN median NMSE `0.690376`; S11/S21 MAE means are `5.325428 dB` and `5.715562 dB`.
- Diagnosis: S-parameter fine-tuning improves over parameter pretraining (`1.137043` to `0.622891` test NMSE mean), but the final NN is still worse than direct ADS cascade.
- Report and plots: `model_report.md`, `metric_summary.png`, `training_loss_curves.png`, `optimization_nmse_histogram.png`, and `comparison_plots/` under the output directory.

## v11 Direct Cascade 9-vs-13 Block Diagnostic

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/compare_direct_cascade_9_vs_13.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/direct_cascade_9_vs_13_lhs150_50_connection2/`
- Purpose: compare 9-block direct cascade and 13-block direct cascade on the same `HFSS_sim/LHS150_50_Connection2/train|test/TSV_RDL` HFSS targets, without pi correction, optimization, or NN prediction.
- 9-block sequence: `TMRDL-TSV-BSMRDL-TSV-TMRDL-TSV-BSMRDL-TSV-TMRDL`.
- 13-block sequence: `TMRDL-TSV-BSMRDL-TSV-TMRDL-TSV-BSMRDL-TSV-TMRDL-TSV-BSMRDL-TSV-TMRDL`.
- Result: 13-block direct cascade is better for all 200 samples. All-sample NMSE mean is `0.295709` for 13-block vs `1.86893` for 9-block; test NMSE mean is `0.304390` for 13-block vs `1.88979` for 9-block.
- Archived outputs: `direct_9_vs_13_per_sample.csv`, `direct_9_vs_13_summary.csv`, `direct_9_vs_13_report.md`, `direct_nmse_9_vs_13.png`, and selected curve plots under `comparison_plots/`.

## v11 Shared 7-Parameter Optimization S-Parameter Plots

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_shared_optimization_sparams.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/shared_v08_optimization_sparam_plots_lhs150_50_connection2/`
- Purpose: visualize the first v11 optimization stage before NN training. For each sample, one optimized 7-parameter connection circuit is repeated at all 12 v11 connection positions, then compared against HFSS target and 13-block direct cascade.
- Result: all-sample NMSE mean improves from 13-block direct `0.295709` to shared-optimized `0.136277`; test NMSE mean improves from `0.304390` to `0.100448`.
- Coverage: generated 200 per-sample S-parameter comparison plots under `plots/all_samples/`, plus selected best/worst test plots under `plots/selected/`.
- Archived outputs: `shared_optimization_sparam_metrics.csv`, `shared_optimization_sparam_summary.csv`, `shared_optimization_sparam_report.md`, and `shared_optimization_nmse_summary.png`.

## v11 ADS-vs-HFSS LHS400_Connection2 Single-Device Plots

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/plot_ads_vs_hfss_lhs400_connection2_single_devices.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/ads_vs_hfss_lhs400_connection2_single_device_50/`
- Dataset: first 50 `RDL` samples and first 50 `TSV` samples from `HFSS_sim/LHS400_Connection2/train`, sorted by `dut_index`.
- ADS settings: same `ac_l400_ref2` single-device settings used by the current v11 training flow; RDL is simulated as ADS `TMRDL` with `t_tmrdl` mapped to `h_tmrdl`, TSV is simulated as ADS `TSV`.
- Result: RDL NMSE mean `0.0416688`, TSV NMSE mean `0.00626028`; RDL S11/S21 dB MAE means are `3.97278 dB` and `0.617988 dB`, TSV S11/S21 dB MAE means are `5.70775 dB` and `0.0320798 dB`.
- Archived outputs: 100 per-sample ADS/HFSS S-parameter plots under `plots/RDL/` and `plots/TSV/`, `ads_vs_hfss_single_device_metrics.csv`, `ads_vs_hfss_single_device_summary.csv`, `ads_vs_hfss_single_device_report.md`, and `ads_vs_hfss_single_device_metric_summary.png`.

## v11 ADS LHS400_Connection2 Random-30 Single-Device Calibration

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/calibrate_ads_lhs400_connection2_random30.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/ads_single_device_calibration_lhs400_connection2_random30/`
- Dataset: random 30 `RDL` samples and random 30 `TSV` samples from `HFSS_sim/LHS400_Connection2/train`, using random seed `20260709`.
- Method: one-variable-at-a-time ADS candidate scan around the current v11 `ac_l400_ref2` settings.
- Best RDL candidate: `w_scale=0.55`, NMSE mean `0.0334426`; random-30 RDL baseline was `0.0442668`.
- Best TSV candidate: `pitch_scale=1.1`, NMSE mean `0.00533247`; random-30 TSV baseline was `0.00550278`.
- Archived outputs: 600 ADS `.s2p` cache files, `ads_calibration_random30_detail.csv`, `ads_calibration_random30_summary.csv`, `ads_calibration_random30_best_per_sample.csv`, `ads_calibration_random30_report.md`, 60 best ADS/HFSS comparison plots, and `ads_calibration_random30_metric_summary.png`.

## v11 ADS Random-30 Calibrated Shared-Connection Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/optimize_v11_shared_connection_calibrated.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30/`
- Dataset: `HFSS_sim/LHS150_50_Connection2/train|test/TSV_RDL`, with `135` train, `15` validation, and `50` test samples after the standard v11 split.
- ADS settings: random-30 LHS400_Connection2 calibration (`RDL w_scale=0.55`, `TSV pitch_scale=1.1`), 0.1-100 GHz sweep.
- Method: cascade 13 ADS device blocks, insert one optimized 7-parameter connection network at all 12 connection positions, and compare against the direct 13-device cascade.
- Result: all-sample NMSE mean improves from direct `0.196623` to optimized `0.105893`; test NMSE mean improves from `0.203091` to `0.0715929`.
- Diagnostic filter: `43` of `200` samples have optimized NMSE greater than `0.1`; test split has `8` such samples.
- Archived outputs: 600 ADS `.s2p` cache files, 200 optimization JSON files, 200 all-sample comparison plots, 12 selected test plots, `direct_vs_optimized_summary.csv`, `optimization_report.md`, `direct_vs_optimized_nmse_summary.png`, and `validation_archive.md`.
- ADS cache reuse update: v11 ADS helper scripts now return an existing `.s2p` file before rewriting sidecar JSON metadata, so interrupted VS Code runs can resume cleanly from cache.

## v11 ADS Re-Optimization For First-Pass Worse Samples

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/reoptimize_v11_worse_shared_connection.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_reopt_worse/`
- Source result: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30/`
- Scope: only the 39 samples where the first shared-connection optimization had higher `S11/S21` real/imag NMSE than direct cascade.
- Method: reuse the 600 ADS single-device cache files, run multi-start least-squares on normalized `S11/S21` real/imag residuals, and keep the lowest-NMSE result per sample.
- Result: all-sample NMSE mean improves from first optimized `0.105893` to re-optimized `0.0500447`; test NMSE mean improves from `0.0715929` to `0.056977`.
- Samples better than direct increase from `161/200` to `186/200`; `14` samples remain worse than direct after re-optimization.
- Archived outputs: 39 re-optimization comparison plots, 429 attempt rows, `v08_shared_reoptimized_targets.csv`, `reopt_direct_first_reopt_summary.csv`, `still_worse_than_direct_after_reopt.csv`, `reopt_worse_nmse_summary.png`, `reoptimization_report.md`, and `validation_archive.md`.
- ADS cache reuse update: the v11 base ADS runner now checks the `.s2p` cache before calling helper scripts, so continuation runs do not rewrite generated netlists.

## v11 ADS Good-Sample Initial-Value Re-Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/reoptimize_v11_remaining_with_good_starts.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_remaining/`
- Source result: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_reopt_worse/`
- Scope: the 14 samples that were still worse than direct cascade after the previous re-optimization.
- Method: use the 186 already-good optimized samples as an initial-value pool; for each remaining bad sample, try the globally best good-sample circuit parameters and geometry-nearest good-sample circuit parameters as least-squares starts.
- Result: all 14 remaining bad samples become better than direct cascade; samples better than direct are now `200/200`.
- All-sample NMSE mean improves from previous re-optimized `0.0500447` to `0.0288596`; test NMSE mean improves from `0.056977` to `0.0221681`.
- Conclusion: the poor fits were mainly caused by local minima from weak initial values, not by the 7-parameter shared connection circuit being unable to fit those samples.
- Archived outputs: 14 good-start comparison plots, 643 attempt rows, `v08_shared_goodstart_targets.csv`, `goodstart_remaining_summary.csv`, `still_worse_than_direct_after_goodstart.csv`, `goodstart_remaining_nmse_summary.png`, `goodstart_report.md`, and `validation_archive.md`.

## v11 ADS Worst Final Optimized Sample Plots

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_goodstart_worst_samples.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_remaining/comparison_plots/worst_final_samples/`
- Source result: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_remaining/`
- Ranking metric: final `goodstart_nmse_s11_s21_ri` descending.
- Plots: 12 worst final optimized samples, each showing HFSS, direct cascade, first optimized, and final optimized curves.
- Worst final samples: `LHS150_50_Connection2_train_dut45` (`0.426598`), `train_dut35` (`0.347798`), and `train_dut136` (`0.284660`); each remains better than its direct-cascade NMSE.
- Archived outputs: `worst_final_sample_plots.csv`, `worst_final_sample_plots_report.md`, `worst_final_sample_plots_report.json`, and 12 plot PNGs.

## v11 ADS Worst-10% Good-Sample Initial-Value Re-Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/reoptimize_v11_worst10pct_with_good_starts.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_worst10pct/`
- Source result: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_remaining/`
- Scope: the 20 samples with the largest final `goodstart_nmse_s11_s21_ri`, i.e. the worst 10% of the 200-sample set.
- Method: reuse the 600 ADS single-device cache files; use the other 180 samples as the good-start pool; try globally best and geometry-nearest good-sample circuit parameters as initial values.
- Result: all 20 target samples improve; all 200 samples remain better than direct cascade.
- All-sample NMSE mean improves from `0.0288596` to `0.0222568`; test NMSE mean improves from `0.0221681` to `0.0217764`.
- Archived outputs: 20 comparison plots, 949 attempt rows, `v08_shared_worst10pct_goodstart_targets.csv`, `worst10pct_goodstart_summary.csv`, `still_worse_than_direct_after_worst10pct.csv`, `worst10pct_goodstart_nmse_summary.png`, `worst10pct_goodstart_report.md`, and `validation_archive.md`.

## v11 ADS All-Sample Good-Sample Initial-Value Re-Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/reoptimize_v11_all_with_good_starts.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_all/`
- Source result: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_worst10pct/`
- Scope: all 200 samples.
- Method: for each sample, keep the current final parameters and also try first-pass, unit, globally better-sample, and geometry-nearest better-sample initial values; retain only the lowest-NMSE result.
- Result: 198/200 samples improve further; all 200 samples remain better than direct cascade.
- All-sample NMSE mean improves from `0.0222568` to `0.0200875`; test NMSE mean improves from `0.0217764` to `0.0193815`.
- Archived outputs: 3515 attempt rows, `v08_shared_all_goodstart_targets.csv`, `all_goodstart_metrics.csv`, `all_goodstart_summary.csv`, `still_worse_than_direct_after_all_goodstart.csv`, `all_goodstart_nmse_summary.png`, `all_goodstart_report.md`, and `validation_archive.md`.

## v11 ADS Shared 7-Parameter NN Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_shared7_param_nns_from_all_goodstart.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_shared7_param_nns_all_goodstart/`
- Source target: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_goodstart_all/v08_shared_all_goodstart_targets.csv`
- Network: seven independent `input -> 30 -> 30 -> 20 -> 1` neural networks, one for each 7-parameter connection-circuit element.
- Split: `135` train, `15` validation, `50` test; training stopped after `97` epochs.
- Result: parameter-supervised NN does not reproduce the optimized cascades. Test optimized NMSE mean is `0.0193815`, but NN cascade NMSE mean is `1.17482`; all-sample optimized NMSE mean is `0.0200875`, but NN cascade mean is `0.892640`.
- Diagnosis: the optimized circuit parameters are likely non-unique and discontinuous across nearby structures, so direct `structure -> optimized parameter` regression is a poor target even when the optimized S-parameter fit is good.
- Archived outputs: `shared7_param_nns.pt`, `shared7_param_training_history.csv`, `shared7_param_predictions.csv`, `optimized_vs_shared7_nn_metrics.csv`, `optimized_vs_shared7_nn_summary.csv`, `training_and_nmse_summary.png`, `parameter_prediction_scatter.png`, comparison plots, `shared7_param_nn_report.md`, and `validation_archive.md`.

## v11 ADS 0.9-Scale Shared-Connection Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/optimize_v11_shared_connection_adslen09.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_adslen09/`
- Dataset: `HFSS_sim/LHS150_50_Connection2/train|test/TSV_RDL`, with `135` train, `15` validation, and `50` test samples after the standard v11 split.
- ADS geometry scale: `ADS_DEVICE_LENGTH_SCALE=0.9`, applied to ADS `l_tmrdl`, `l_bsmrdl`, and `h_tsv` before single-device simulation.
- Method: cascade 13 ADS device blocks, insert one optimized 7-parameter connection network at all 12 connection positions, and compare against the direct 13-device cascade.
- Result: all-sample NMSE mean improves from direct `0.467333` to optimized `0.0948387`; test NMSE mean improves from `0.470152` to `0.115027`.
- Diagnostic filter: `19` of `200` samples have optimized NMSE greater than `0.1`; test split has `6` such samples.
- Archived outputs: 600 ADS `.s2p` cache files, 200 optimization JSON files, 200 all-sample comparison plots, 12 selected test plots, `direct_vs_optimized_summary.csv`, `optimization_report.md`, `direct_vs_optimized_nmse_summary.png`, and `validation_archive.md`.

## v11 ADS 0.9-Scale Bad-Sample Good-Start Re-Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/reoptimize_v11_adslen09_bad_with_good_starts.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_adslen09_goodstart_bad/`
- Source result: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_adslen09/`
- Scope: the `19` first-pass samples with optimized NMSE greater than `0.1`.
- Method: reuse the 0.9-scale ADS cache, try globally best and geometry-nearest good-sample circuit parameters as initial values, and keep the lowest-NMSE result per sample.
- Result: all `19` target samples improve, and all `200/200` samples become better than direct cascade. All-sample NMSE mean improves from first-pass optimized `0.0948387` to `0.0270039`; test mean improves from `0.115027` to `0.0258311`.
- Parameter sign diagnostics: final signed parameters still include many negative values, especially `Cn3_scale` (`124/200`) and `Rn3_scale` (`176/200`). Among the 19 re-optimized targets, negative counts are `Cn1_scale=17`, `Rn1_scale=10`, `Cn2_scale=1`, `Rn2_scale=9`, `Cn3_scale=15`, `Rn3_scale=4`, `Ln1_scale=3`.
- Archived outputs: `v08_shared_adslen09_goodstart_bad_targets.csv`, `goodstart_bad_metrics.csv`, `goodstart_bad_attempts.csv`, `parameter_sign_stats.csv`, `samples_with_negative_parameters.csv`, 19 comparison plots, `goodstart_bad_nmse_and_parameter_signs.png`, `goodstart_bad_report.md`, and `validation_archive.md`.

## v11 ADS 0.9-Scale Shared 7-Parameter NN Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_shared7_param_nns_adslen09_goodstart.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_shared7_param_nns_adslen09_goodstart/`
- Source target: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_adslen09_goodstart_bad/v08_shared_adslen09_goodstart_bad_targets.csv`
- Network: seven independent `input -> 30 -> 30 -> 20 -> 1` neural networks, one for each 7-parameter connection-circuit element.
- Split: `135` train, `15` validation, `50` test; training stopped after `97` epochs.
- Result: direct parameter-supervised NN still fails to reproduce the optimized cascade. Test optimized NMSE mean is `0.0258311`, while NN cascade NMSE mean is `1.43895`; all-sample optimized NMSE mean is `0.0270039`, while NN cascade mean is `1.42759`. NN is better than direct for only `36/200` samples and better than optimized for `0/200`.
- Parameter sign diagnostics: targets contain many negative `Cn3_scale` (`124/200`) and `Rn3_scale` (`176/200`) values; NN predictions also contain many negative `Cn1_scale` (`164/200`), `Cn3_scale` (`113/200`), and `Rn3_scale` (`194/200`) values, but the S-parameter cascade remains poor.
- Archived outputs: `shared7_param_nns.pt`, `shared7_param_training_history.csv`, `shared7_param_predictions.csv`, `optimized_vs_shared7_nn_metrics.csv`, `optimized_vs_shared7_nn_summary.csv`, `target_and_predicted_parameter_sign_stats.csv`, comparison plots, `shared7_param_nn_report.md`, and `validation_archive.md`.

## v11 ADS 0.9-Scale Joint Parameter/S-Parameter Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_shared7_param_sparam_joint_adslen09.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_shared7_param_sparam_joint_adslen09/`
- Source target: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_adslen09_goodstart_bad/v08_shared_adslen09_goodstart_bad_targets.csv`
- Network: seven independent `input -> 30 -> 30 -> 20 -> 1` neural networks. The model is parameter-pretrained first, then fine-tuned with combined loss: normalized `S11/S21` real/imag loss + `0.15` normalized parameter MSE + `0.02` magnitude/phase loss.
- Training: parameter pretrain stops after `97` epochs; joint fine-tuning stops after `72` epochs.
- Result: S-parameter loss helps but does not solve the modeling problem. All-sample NN NMSE mean improves from parameter-only `1.42759` to joint `0.502775`; test NN NMSE mean improves from `1.43895` to `0.481906`. The optimized-parameter cascade remains much better: all-sample `0.0270039`, test `0.0258311`.
- Diagnostic: joint NN is better than direct cascade for `120/200` samples, but better than optimized for `0/200`. This indicates the S-parameter term pulls the model in the right direction, while the smooth 7-output shared-parameter network is still not expressive/stable enough to reproduce the optimized circuit behavior.
- Archived outputs: `shared7_param_sparam_joint.pt`, `param_pretrain_history.csv`, `joint_training_history.csv`, `param_only_metrics.csv`, `joint_metrics.csv`, `joint_summary.csv`, `joint_predictions.csv`, `target_and_joint_predicted_parameter_sign_stats.csv`, comparison plots, `joint_training_report.md`, and `validation_archive.md`.

## v11 ADS 0.9-Scale Resonance-Filtered Multi-Head Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_multihead_exclude_resonance_adslen09.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_multihead_exclude_resonance_adslen09/`
- Source target: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_adslen09_goodstart_bad/v08_shared_adslen09_goodstart_bad_targets.csv`
- Resonance filter: use the previous joint NN predictions and exclude samples with `nn_db_d1 > 12.0` or `nn_max_d1 > 0.2`.
- Scope after filtering: `77/200` samples are excluded; `123` samples remain, split as `81` train, `10` validation, and `32` test.
- Network: shared trunk plus 12 connection-position heads for the 13-device chain. Training uses shared-parameter pretraining, multi-head parameter pretraining, then joint S-parameter fine-tuning.
- Result: retained all-sample multi-head NN NMSE mean is `0.332941`; on the same retained samples, the previous shared-joint NN was better at `0.236783`. Retained test NN NMSE mean is `0.585927`, also worse than the previous shared-joint retained-test `0.254827` and much worse than optimized cascade `0.027617`.
- Archived outputs: `resonance_diagnostics_all_samples.csv`, `excluded_resonance_samples.csv`, `multihead_exclude_resonance.pt`, `multihead_joint_metrics.csv`, `multihead_joint_summary.csv`, `multihead_joint_predictions.csv`, `multihead_training_loss.png`, `multihead_filtered_nmse_summary.png`, comparison plots, `multihead_exclude_resonance_report.md`, and `validation_archive.md`.

## v11 ADS 0.9-Scale Resonance-Filtered Multi-Head S-Parameter-Only Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_multihead_exclude_resonance_sparam_only_adslen09.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_multihead_exclude_resonance_sparam_only_adslen09/`
- Source target: `model_versions/v11_ads_v08_multihead_chain/results/v11_sharedopt_c30_adslen09_goodstart_bad/v08_shared_adslen09_goodstart_bad_targets.csv`
- Resonance filter: same as the prior multi-head run, excluding samples with `nn_db_d1 > 12.0` or `nn_max_d1 > 0.2`.
- Scope after filtering: `77/200` samples are excluded; `123` samples remain, split as `81` train, `10` validation, and `32` test.
- Network: shared trunk plus 12 connection-position heads for the 13-device chain. The optimizer uses only cascaded `S11/S21` real/imag loss; no parameter MSE anchor and no magnitude/phase auxiliary loss are used.
- Result: retained all-sample NN NMSE mean improves from the prior multi-head `0.332941` to `0.248288`; retained test improves slightly from `0.585927` to `0.561180`. On the same retained samples, the previous shared-joint NN is still better (`0.236783` all, `0.254827` test), and the optimized cascade remains much better (`0.0287767` all, `0.027617` test).
- Archived outputs: `resonance_diagnostics_all_samples.csv`, `excluded_resonance_samples.csv`, `multihead_exclude_resonance.pt`, `multihead_sparam_only_history.csv`, `multihead_joint_metrics.csv`, `multihead_joint_summary.csv`, `multihead_joint_predictions.csv`, `multihead_training_loss.png`, `multihead_filtered_nmse_summary.png`, comparison plots, `multihead_exclude_resonance_report.md`, and `validation_archive.md`.

## v11 ADS 0.9-Scale Multi-Head Resonance Diagnosis

- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_sparam_only_resonance.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_multihead_exclude_resonance_sparam_only_adslen09/sparam_only_resonance_diagnosis_report.md`
- Finding: excluding resonant samples before training is not enough. The S-parameter-only multi-head model creates new resonance-like spikes on `76/123` retained samples, including `22/32` test samples.
- Parameter diagnosis: the 12 independent heads generate highly nonphysical signed parameters, especially negative `Cn1_scale` (`1152/1476`), negative `Cn3_scale` (`978/1476`), and negative `Rn3_scale` (`1282/1476`).
- Smoothness experiment: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_multihead_exclude_resonance_sparam_smooth_adslen09.py` adds S-parameter first- and second-difference losses, but the result is worse: `81/123` retained samples and `25/32` test samples are still resonant, with test NN NMSE `0.656841`.
- Current conclusion: the free 12-head circuit-parameter output is too unstable for the current 123-sample retained set. The safer direction is to use the shared joint model, or change multi-head to `shared baseline + bounded small residual heads` instead of predicting 84 fully free circuit parameters.

## v11 ADS 0.9-Scale Positive-Parameter LHS400 Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/optimize_v11_positive_shared_connection_lhs400_adslen09.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_sharedopt_lhs400_connection2_adslen09/`
- Dataset: `HFSS_sim/LHS400_Connection2/train/TSV_RDL`
- Scope: `TSV_RDL_variations_record.csv` has `400` rows; `280` `.s2p` files currently exist on disk and all were optimized.
- Constraint: all seven connection-circuit scale parameters are constrained to positive bounds `[1e-9, 1e5]`.
- Method: ADS device length scale remains `0.9`; one positive 7-parameter connection circuit is optimized per sample and repeated at all `12` connection positions.
- Result: all-sample direct NMSE mean is `0.452023`; positive-optimized NMSE mean is `0.0351993`; optimized is better than direct for `280/280` samples.
- Sign check: `positive_parameter_sign_summary.csv` reports `0` nonpositive values across all seven parameters.
- Archived outputs: 840 ADS single-device `.s2p` cache files, 280 optimization JSON files, `v08_positive_shared_optimized_targets.csv`, `direct_vs_positive_optimized_metrics.csv`, `direct_vs_positive_optimized_summary.csv`, `positive_parameter_sign_summary.csv`, selected comparison plots, `positive_optimization_report.md`, and `validation_archive.md`.

## v11 Positive-Parameter LHS400 All-Sample Good-Start Re-Optimization

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/reoptimize_v11_positive_lhs400_all_good_starts.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_sharedopt_lhs400_connection2_goodstart_all_adslen09/`
- Source result: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_sharedopt_lhs400_connection2_adslen09/`
- Scope: `298` existing `HFSS_sim/LHS400_Connection2/train/TSV_RDL` `.s2p` samples were processed. `280` already had previous positive optimized targets; `18` additional samples appeared on disk and were optimized directly from good-sample starts.
- Method: keep positive bounds `[1e-9, 1e5]`, reuse the ADS 0.9-scale cache when available, and for each sample try the source positive result, unit vector, up to `6` globally best optimized samples, and up to `4` geometry-nearest good optimized samples as initial values.
- Result: on the previous `280` samples, NMSE mean improves from source positive `0.0351993` to good-start `0.0278140`, and all `280/280` remain better than direct cascade. On all `298` current samples, direct NMSE mean is `0.453240`, good-start NMSE mean is `0.0279288`, and `298/298` are better than direct cascade.
- Sign check: `positive_goodstart_parameter_sign_summary.csv` reports `0` nonpositive values across all seven parameters. Several parameters sit on bounds for many samples (`Cn1_scale` and `Rn3_scale` lower bound, `Rn1_scale` and `Rn2_scale` upper bound), so these optimized parameters should still be treated as bounded fitting variables rather than unique physical values.
- Archived outputs: `v08_positive_goodstart_targets.csv`, `positive_goodstart_metrics.csv`, `positive_goodstart_summary.csv`, `positive_goodstart_attempts.csv` (`3534` rows), `positive_goodstart_parameter_sign_summary.csv`, `positive_goodstart_nmse_summary.png`, 12 selected comparison plots, `positive_goodstart_report.md`, and `validation_archive.md`.

## v11 Positive-Parameter Good-Start 30-30-20 Parameter NN

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_goodstart_param_nns.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_goodstart_shared7_param_nns_log_adslen09/`
- Source target: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_sharedopt_lhs400_connection2_goodstart_all_adslen09/v08_positive_goodstart_targets.csv`
- Scope: uses the `298` samples with optimized positive circuit parameters. Current disk also contains `dut248` and `dut300`, but they do not have optimized circuit-parameter targets, so they are excluded and recorded in `excluded_unoptimized_samples.csv`.
- Network: seven independent `input -> 30 -> 30 -> 20 -> 1` parameter networks. The model trains on `log10(parameter)` and converts predictions back to positive scale clipped to `[1e-9, 1e5]`.
- Training: stopped after `141` epochs. Predicted parameter sign check reports `0` nonpositive values.
- Result: parameter-only NN prediction is still much worse than per-sample optimized parameters after cascading. All-sample optimized NMSE mean is `0.0279288`, while NN cascade NMSE mean is `0.343877`; validation optimized mean is `0.0265976`, while validation NN mean is `0.399673`. NN is better than direct cascade for `248/298` samples but better than optimized for `0/298`.
- Archived outputs: `positive_shared7_param_nns_log.pt`, `positive_shared7_param_training_history.csv`, `positive_shared7_param_predictions.csv`, `optimized_vs_positive_shared7_nn_metrics.csv`, `optimized_vs_positive_shared7_nn_summary.csv`, `target_and_predicted_parameter_sign_stats.csv`, `excluded_unoptimized_samples.csv`, summary plots, comparison plots, report, and `validation_archive.md`.

## v11 Positive-Parameter Multi-Head S-Parameter Training From Shared NN

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_multihead_sparam_from_shared.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_multihead_sparam_from_shared_log_adslen09/`
- Source shared checkpoint: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_goodstart_shared7_param_nns_log_adslen09/positive_shared7_param_nns_log.pt`
- Source optimized target: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_sharedopt_lhs400_connection2_goodstart_all_adslen09/v08_positive_goodstart_targets.csv`
- Method: expand the current seven `input -> 30 -> 30 -> 20 -> 1` parameter networks into 12 connection-position heads. Each head is initialized with the shared network weights and biases. The model outputs normalized `log10(parameter)` values, converts them back to positive circuit parameters clipped to `[1e-9, 1e5]`, and trains only on cascaded `S11/S21` real/imag loss.
- Scope: uses the same `298` optimized-target samples; current-disk `dut248` and `dut300` remain excluded because they have no optimized parameter targets.
- Training: ran for `320` epochs; best validation RI loss is `0.106159`.
- Result: S-parameter-only multi-head training improves all-sample NN NMSE mean from the shared-expanded baseline `0.343877` to `0.0532449`; validation NN NMSE mean improves from `0.399673` to `0.0562088`. The model is better than direct cascade for `294/298` samples and better than the per-sample optimized shared circuit for `79/298` samples, but the optimized shared-circuit target is still better on mean NMSE (`0.0279288` all, `0.0265976` val).
- Sign check: all `3576` predicted multi-head circuit-parameter values are positive; nonpositive count is `0`.
- Archived outputs: `positive_multihead_sparam_from_shared.pt`, `positive_multihead_sparam_history.csv`, `initial_shared_expanded_metrics.csv`, `positive_multihead_sparam_metrics.csv`, `positive_multihead_sparam_summary.csv`, `positive_multihead_sparam_predictions.csv`, `positive_multihead_parameter_sign_stats.csv`, `excluded_unoptimized_samples.csv`, `multihead_sparam_training_summary.png`, comparison plots, report, and `validation_archive.md`.

## v11 Positive Multi-Head Resonance and L/C Diagnosis

- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_multihead_resonance_params.py`
- Output report: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_multihead_sparam_from_shared_log_adslen09/positive_multihead_resonance_lc_diagnosis_report.md`
- Scope: diagnoses the 298-sample positive multi-head S-parameter-only result.
- Resonance rule: mark a sample if adjacent-frequency `S11/S21` dB jump exceeds `12 dB` or real/imag jump exceeds `0.2`.
- Finding: `20/298` predictions are resonance-like (`19` train, `1` validation). Resonant samples have mean NN NMSE `0.120236`, versus `0.048425` for non-resonant samples.
- L/C diagnosis: the main extreme output is capacitance, not inductance. `Cn3_scale` reaches p99 `11626.8` and max `84489.8` across all heads; several worst resonance samples have their largest capacitance at `Cn3_scale` head 2/4/7/8. `Ln1_scale` max is only `6.796`, and its resonant/non-resonant distribution is not the main separator.
- Interpretation: positive constraints remove negative nonphysical values, but unconstrained upper ranges still let the 12-head S-parameter-only model create very large C values, especially `Cn3_scale`, which can introduce sharp resonant behavior. The next modeling fix should add tighter per-parameter upper bounds or a regularization/penalty on large `Cn3` and possibly smoothness on head-to-head parameter variation.
- Archived outputs: `positive_multihead_resonance_lc_diagnostics.csv`, `positive_multihead_resonance_lc_summary.csv`, `positive_multihead_parameter_range_diagnostic.csv`, `positive_multihead_resonant_parameter_range_diagnostic.csv`, `positive_multihead_worst_resonance_samples.csv`, `positive_multihead_large_lc_samples.csv`, `positive_multihead_top_lc_parameter_outputs.csv`, and the Markdown/JSON diagnosis reports.

## v11 Positive Multi-Head S-Parameter Training With L/C Limit Loss

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_multihead_sparam_lc_limited.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_multihead_sparam_lc_limited_log_adslen09/`
- Constraint requested: physical `L < 1e-8 H` and `C < 1e-11 F`.
- Scale conversion: current v08 circuit uses `Cn*_scale * 1e-14 = C(F)` and `Ln1_scale * 1e-11 = L(H)`, so the requested limits are equivalent to `Cn*_scale < 1000` and `Ln1_scale < 1000`.
- Reasonableness: `C < 1e-11 F` is reasonable as a stabilizing loss because previous resonance diagnosis showed `Cn3_scale` p99 `11626.8` and max `84489.8`. `L < 1e-8 H` is too loose for this model because previous `Ln1_scale` max was only about `6.8`, i.e. `6.8e-11 H`, far below the requested limit.
- Training method: initialize from the same positive shared `30-30-20` checkpoint, train with cascaded `S11/S21` real/imag loss plus `0.25 * L/C_limit_penalty`. The penalty is soft, so it discourages but does not guarantee strict satisfaction of the limits.
- Result: all-sample NN NMSE mean improves from the previous positive multi-head `0.0532449` to `0.0473491`; validation mean improves from `0.0562088` to `0.0468867`. NN is better than direct cascade for `294/298` samples and better than the per-sample optimized shared circuit for `89/298` samples.
- Limit effect: `Cn3_scale` max drops from `84489.8` to `1752.3`; `Cn3_scale` p99 drops from `11626.8` to `1026.64`. There are still `40/3576` C values above `1e-11 F`, because the limit is in the loss rather than a hard clamp. `Ln1_scale` remains far below the equivalent limit (`max=6.8659 << 1000`), confirming the L limit is not active.
- Resonance check: resonance-like predictions reduce from `20/298` to `18/298` with the same jump thresholds. The remaining resonances are less tied to extremely huge C values, so a hard cap or stronger `Cn3` penalty may be needed if strict suppression is required.
- Archived outputs: `positive_multihead_sparam_lc_limited.pt`, history/metrics/predictions CSVs, `positive_multihead_lc_limit_stats.csv`, `lc_limited_resonance_diagnostics.csv`, `lc_limited_resonance_summary.csv`, comparison plots, report, and `validation_archive.md`.

## v11 L/C-Limited Multi-Head Specific Resonance Diagnosis

- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_multihead_sparam_lc_limited_log_adslen09/`
- Scope: checked the user-reported resonant samples `LHS400_Connection2_train_dut72`, `dut79`, `dut123`, `dut253`, and `dut258` using the latest L/C-limited positive multi-head model.
- Finding by sample: `dut72` is mainly driven by oversized `Cn3_scale` at multiple heads, with max `Cn3_scale=1752.30` (`1.75e-11 F`) and about `10.6x` its optimized shared target. `dut123` is also `Cn3_scale` dominated, with many heads above the requested C limit, but its optimized shared target already has `Cn3_scale=2161.37`, so the C limit conflicts with the optimized target for that sample. `dut253` is not C-limit driven; its suspicious output is inflated `Ln1_scale` across heads, especially head 12 at `2.98`, about `17.5x` its optimized target, while still far below the loose `1e-8 H` physical L limit. `dut258` has a large relative `Cn3_scale` jump at head 9 (`403.03`, about `107x` target) but remains below the physical C limit; the strongest curve spike is therefore more consistent with unconstrained head-to-head variation than absolute L/C overflow. `dut79` has no obvious large L/C culprit; its C/L values are moderate and the visible error is likely from S-parameter-only multi-head freedom rather than an individual oversized parameter.
- Conclusion: remaining resonances are not all solved by the current physical `C < 1e-11 F`, `L < 1e-8 H` soft loss. The next constraint should add a hard or stronger `Cn3` cap for `dut72`-like cases, plus a relative-to-shared or head-smoothness penalty for `Ln1_scale`/`Cn3_scale` deviations to handle `dut253` and `dut258`.
- Archived diagnostics: `specific_resonance_samples_metrics.csv`, `specific_resonance_samples_max_lc.csv`, `specific_resonance_samples_top_lc_outputs.csv`, and `specific_resonance_samples_all_heads_params.csv`.

## v11 Positive Multi-Head Strong L/C Constraint Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_multihead_sparam_strong_lc.py`
- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_strong_lc_resonance_params.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_multihead_sparam_strong_lc_log_adslen09/`
- Constraint change: keep `C < 1e-11 F` but increase the C penalty, tighten L from the previous ineffective `Ln1_scale < 1000` to `Ln1_scale < 2` (`L < 2e-11 H`), and add reference-drift/head-smoothness penalties for C/L outputs.
- Result: resonance-like samples reduce from the previous L/C-limited `18/298` to `10/298`. `Cn3_scale` max drops from `1752.3` to `1145.8`, and `Ln1_scale` max drops from `6.8659` to `2.6351`.
- Tradeoff: all-sample NN NMSE worsens from `0.0473491` to `0.0582432`; NN better-than-direct count drops from `294/298` to `290/298`, and NN better-than-optimized count drops from `89/298` to `58/298`.
- User-flagged samples: `dut72`, `dut79`, `dut123`, and `dut253` are no longer resonance-like under the original jump rule; `dut258` remains resonance-like (`pred_db_d1=17.3255`, `pred_max_d1=0.3551`) and its NN NMSE worsens to `0.646978`.
- Archived outputs: `positive_multihead_sparam_strong_lc.pt`, history/metrics/predictions CSVs, `positive_multihead_strong_lc_limit_stats.csv`, `strong_lc_resonance_lc_diagnostics.csv`, `strong_lc_resonance_lc_summary.csv`, `strong_lc_worst_resonance_samples.csv`, comparison plots, reports, and `validation_archive.md`.

## v11 Positive Symmetric Multi-Head L/C Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc.py`
- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_resonance_params.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_log_adslen09/`
- Architecture: uses the left-right symmetry of the 13-device cascade. The model learns six connection heads and mirrors them to twelve positions as `1,2,3,4,5,6,6,5,4,3,2,1`; measured max mirrored-parameter difference is `0`.
- Constraint change: keeps `C < 1e-11 F`, uses a milder `Ln1_scale < 4` (`L < 4e-11 H`) instead of the too-tight `Ln1_scale < 2`, and adds an R upper penalty to reduce `Rn1/Rn2` compensation.
- Result: original strong-resonance count drops to `0/298`. Lower-threshold counts also improve: with `db>6` or `ri>0.05`, previous L/C-limited has `85`, strong-LC has `38`, and symmetric-LC has `5`.
- Tradeoff: all-sample NN NMSE is `0.0528428`, worse than previous L/C-limited `0.0473491` but better than strong-LC `0.0582432`. NN is better than direct for `292/298` samples and better than optimized for `87/298`.
- Archived outputs: `positive_symmetric_multihead_lc.pt`, history/metrics/predictions CSVs, `positive_symmetric_multihead_lc_limit_stats.csv`, `symmetric_lc_resonance_lc_diagnostics.csv`, `symmetric_lc_resonance_lc_summary.csv`, `symmetric_lc_worst_resonance_samples.csv`, comparison plots, reports, and `validation_archive.md`.

### v11 Symmetric-LC DUT72 Phase Plot

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_symmetric_phase_dut72.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_log_adslen09/phase_comparison_plots/LHS400_Connection2_train_dut72_phase_wrapped_unwrapped.png`
- Scope: compares HFSS simulation, ADS direct cascade, optimized shared cascade, and symmetric-LC model phase for `S11` and `S21`, with both wrapped and unwrapped phase.
- Metrics: direct NMSE `0.177281`, optimized shared NMSE `0.019674`, symmetric model NMSE `0.202946`.
- Observation: the symmetric model S11 phase deviates onto a different unwrapped phase branch after about `25 GHz`; S21 phase keeps a smoother trend but has a visible slope offset. The optimized shared cascade remains the closest phase match for this sample.

## v11 Positive Symmetric Multi-Head LC fmin60 Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_fmin.py`
- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_fmin_resonance_params.py`
- DUT72 phase script: `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_symmetric_fmin_phase_dut72.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_fmin60_log_adslen09/`
- Constraint change: keep the symmetric six-head mirrored architecture and add a soft lower-bound loss for the local `Cn3/Ln1` resonance frequency, targeting `f0 >= 60 GHz`.
- Result: all-sample NN NMSE mean is `0.0532894`, direct cascade mean is `0.453240`, and optimized shared-circuit mean is `0.0279288`. The model is better than direct for `291/298` samples and better than optimized for `88/298`.
- f0 effect: predicted `Cn3/Ln1` f0 minimum is `34.796 GHz`; `68/3576` positions remain below `60 GHz` and `112/3576` remain below `80 GHz`, so the current loss is only a soft suppression and not a hard guarantee.
- Resonance diagnosis: original strong-resonance rule flags `3/298` samples (`dut123`, `dut127`, `dut47`), mainly correlated with larger `Cn3_scale`.
- DUT72 phase check: fmin60 model NMSE is `0.268276`, worse than ADS direct `0.177281` and previous symmetric-LC model `0.202946`; optimized shared cascade remains best at `0.019674`.
- Archived outputs: checkpoint, history/metrics/predictions CSVs, f0 stats, L/C/R stats, resonance diagnostics, phase plot, reports, and `validation_archive.md`.

## v11 Positive Symmetric Multi-Head LC fmin60 Phase-Loss Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_fmin_phase.py`
- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_fmin_phase_resonance_params.py`
- DUT72 phase script: `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_symmetric_fmin_phase_loss_dut72.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_fmin60_phase_log_adslen09/`
- Loss change: keeps real/imag `S11/S21` loss and adds wrapped phase loss on `S11/S21`, computed as `angle(pred * conj(target))`, with weight `0.12`.
- Result: all-sample NN NMSE mean improves from fmin60-only `0.0532894` to `0.0512387`; direct cascade mean is `0.453240`, optimized shared-circuit mean is `0.0279288`. The model is better than direct for `292/298` samples and better than optimized for `89/298`.
- Resonance diagnosis: original strong-resonance rule flags `1/298` sample (`dut123`), improved from `3/298` in the fmin60-only run. f0 below `60 GHz` improves from `68/3576` to `54/3576`.
- DUT72 phase check: phase-loss model NMSE is `0.295838`, worse than ADS direct `0.177281`, fmin60-only `0.268276`, and previous symmetric-LC `0.202946`; optimized shared cascade remains best at `0.019674`.
- Conclusion: adding aggregate wrapped phase loss helps overall metrics and resonance count, but it does not solve the specific dut72 mid-band phase branch jump.
- Archived outputs: checkpoint, history/metrics/predictions CSVs, f0 stats, L/C/R stats, resonance diagnostics, phase plot, reports, and `validation_archive.md`.

## v11 DUT72 Parameter Ablation Diagnosis

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_dut72_parameter_ablation.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_fmin60_phase_log_adslen09/dut72_parameter_ablation.csv`
- Finding: dut72's bad phase result is dominated by C/L branch placement, not R damping. The phase-loss model baseline is NMSE `0.295391`; replacing only `Cn3_scale + Ln1_scale` with optimized values improves NMSE to `0.121469`, and replacing all C parameters plus `Ln1_scale` improves NMSE to `0.021997`.
- R check: replacing `Rn1_scale + Rn2_scale` alone leaves NMSE essentially unchanged at `0.295439`; replacing all R parameters worsens NMSE to `0.430718`.
- Interpretation: the optimized shared target has `Cn3/Ln1` local f0 about `112 GHz`, while the model predicts about `41-59 GHz`, causing the mid-band phase branch error visible in the dut72 phase plot.

## v11 Positive Symmetric Multi-Head LC fmin100 Phase-Loss Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_fmin100_phase.py`
- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_fmin100_phase_resonance_params.py`
- DUT72 phase script: `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_symmetric_fmin100_phase_dut72.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_fmin100_phase_log_adslen09/`
- Loss change: raises the soft `Cn3/Ln1` local resonance frequency target from `60 GHz` to `100 GHz`, while keeping the wrapped phase loss weight `0.12`.
- Result: all-sample NN NMSE mean is `0.0525081`; direct cascade mean is `0.453240`, optimized shared-circuit mean is `0.0279288`. This is worse than fmin60 phase-loss (`0.0512387`) but better than fmin60-only (`0.0532894`).
- Frequency effect: `f0 < 60 GHz` drops from `54/3576` to `12/3576`, and `f0 < 80 GHz` drops from `94/3576` to `40/3576`; however, `64/3576` positions remain below `100 GHz` because the limit is still a soft penalty.
- Resonance diagnosis: strong-resonance samples increase from `1/298` to `2/298`; `dut72` becomes resonant.
- DUT72 phase check: fmin100 phase-loss model NMSE is `0.546193`, worse than direct `0.177281`, fmin60 phase-loss `0.295838`, and optimized shared `0.019674`. The stricter global f0 target creates stronger 60-75 GHz phase oscillation for this sample.
- Conclusion: a global `f0 > 100 GHz` soft loss suppresses low-f0 outputs overall, but it is not the right fix for dut72. The next fix should target sample-specific C/L branch matching to optimized parameters rather than only raising the global lower bound.

## v11 Positive Symmetric Multi-Head LC fmin60 Phase-Spike Training

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_fmin60_phase_spike.py`
- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_fmin60_phase_spike_resonance_params.py`
- DUT72 phase script: `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_symmetric_fmin60_phase_spike_dut72.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_fmin60_phase_spike_log_adslen09/`
- Loss change: keeps fmin60 phase-loss and adds unwanted-resonance penalties on excess adjacent-frequency jumps relative to HFSS. RI spike weight is `1.5` with margin `0.025`; phase spike weight is `0.25` with margin `0.18 rad`.
- Result: all-sample NN NMSE mean is `0.0524206`; this is worse than fmin60 phase-loss `0.0512387` and slightly better than fmin60-only `0.0532894`.
- Resonance diagnosis: strong-resonance samples are `2/298`, worse than fmin60 phase-loss `1/298`. The new resonant samples are `dut123` and `dut127`.
- DUT72 phase check: phase-spike model NMSE is `0.284100`, slightly better than fmin60 phase-loss `0.295838`, but still worse than fmin60-only `0.268276` and much worse than optimized shared `0.019674`.
- Conclusion: spike loss can suppress some local wiggles, but it does not solve the main dut72 branch-placement error. The better next direction remains parameter/branch supervision for C/L, especially `Cn3/Ln1`, rather than only stronger curve smoothness.

## v11 Resonant NN Output vs Optimized Parameter Max

- Entry script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_resonant_param_exceed_optimized.py`
- Output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_resonant_param_exceed_analysis/`
- Optimized global maxima: C max is `3.523e-11 F` from `Cn3_scale=3523.01`; L max is `3.041e-11 H` from `Ln1_scale=3.04073`; R max is `1e5 ohm`.
- Checked resonant runs: fmin60, fmin60 phase, fmin100 phase, and fmin60 phase-spike.
- Finding: no resonant NN sample exceeds the optimized global maximum for any R/L/C parameter. Each run has `0` samples with C/R/L above the optimized global maxima.
- Interpretation: the resonances are not caused by NN outputs exceeding the global optimized parameter range. They are caused by wrong parameter placement for the specific sample, especially same-sample C/L deviations. For example, resonant `dut123` predicts `Cn2_scale` roughly `872-1013x` its own optimized value and `Ln1_scale` roughly `56-70x`, while still remaining below the global optimized maxima.

## v11 Modeling Flow Paper Draft

- Draft: `model_versions/v11_ads_v08_multihead_chain/v11建模流程小论文.md`
- Validation archive: `model_versions/v11_ads_v08_multihead_chain/results/paper_draft_validation_archive.md`
- Scope: Markdown paper draft based on `建模流程.md` and the v11 ADS/pi-network result summaries. It explains ADS single-device calibration, 13-device cascade construction, 7-parameter pi connection optimization, positive/shared/multi-head NN training, physical constraints, S11/S21 metrics, and current result interpretation.

## v12 Modeling Flow Paper Draft

- Draft: `model_versions/v12_hfss_v08_multihead_chain/v12建模流程小论文.md`
- Validation archive: `model_versions/v12_hfss_v08_multihead_chain/results/paper_method_revision_validation_archive.md`
- Scope: Markdown paper draft revised from the v12 HFSS-equivalent-circuit model files. It explains the HFSS-derived RDL/TSV single-device backend, 13-device cascade, v08 7-parameter pi connection network, shared optimization, twelve-head multi-head training, and TSV-RDL3 validation metrics.
- Scope: Markdown paper draft based on `寤烘ā娴佺▼.md` and the v11 ADS/pi-network result summaries. It explains ADS single-device calibration, 13-device cascade construction, 7-parameter pi connection optimization, positive/shared/multi-head NN training, physical constraints, S11/S21 metrics, and current result interpretation.

## v11 Positive Symmetric Multi-Head Sample-Anchor Training

- Entry scripts:
  - `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_sample_anchor.py`
  - `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_sample_anchor_continue.py`
- Targeted experiment scripts:
  - `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_targeted_anchor_continue.py`
  - `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_targeted_anchor_continue_resonance_params.py`
- Hardcap continuation scripts:
  - `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_hardcap_continue.py`
  - `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_hardcap_continue_resonance_params.py`
  - `model_versions/v11_ads_v08_multihead_chain/code/plot_v11_hardcap_selected_samples.py`
- Tighter hardcap scripts:
  - `model_versions/v11_ads_v08_multihead_chain/code/train_v11_positive_symmetric_multihead_lc_tighter_hardcap_continue.py`
  - `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_tighter_hardcap_continue_resonance_params.py`
- Diagnostic script: `model_versions/v11_ads_v08_multihead_chain/code/diagnose_v11_positive_symmetric_lc_sample_anchor_continue_resonance_params.py`
- Final output: `model_versions/v11_ads_v08_multihead_chain/results/v11_positive_symmetric_multihead_lc_sample_anchor_continue_log_adslen09/`
- Constraint change: continue from the fmin60 phase-loss checkpoint and add same-sample C/L anchors on `Cn1/Cn2/Cn3/Ln1` plus a same-sample `Cn3/Ln1` f0 anchor. This follows the diagnosis that resonant samples did not exceed the global optimized R/L/C maxima, but deviated from their own optimized C/L branch.
- Result: all-sample NN NMSE mean is `0.0467274`, better than fmin60 phase-loss `0.0512387`, direct cascade `0.453240`, and still above optimized shared-circuit `0.0279288`. The model is better than direct for `293/298` samples and better than optimized for `105/298`.
- L/C status: `Cn3_scale` max is `887.006`, so Cn3 stays below `1e-11 F`; `Ln1_scale` max is `4.76938`, with `52/3576` positions above the soft `Ln1_scale < 4` target.
- Resonance diagnosis: strong-resonance rule flags `2/298` samples, `dut123` and `dut72`. The first sample-anchor attempt from shared initialization is archived separately at `results/v11_positive_symmetric_multihead_lc_sample_anchor_log_adslen09/` and is not the selected result because its NN NMSE worsened to `0.142346`.
- Targeted-anchor follow-up: `results/v11_positive_symmetric_multihead_lc_targeted_anchor_continue_log_adslen09/` increases the difficult-sample S-parameter weight and strengthens `Cn3/Ln1/f0` anchors for `dut72/123/253/258`. It improves `dut72` (`0.2592 -> 0.2166`), `dut123` (`0.3822 -> 0.3200`), and `dut258` (`0.3895 -> 0.2132`), but worsens `dut253` (`0.1230 -> 0.1775`), worsens all-sample NN NMSE (`0.0467274 -> 0.0477937`), increases strong-resonance samples (`2 -> 3`), and reintroduces `Cn3` C-limit violations (`6/3576`). It is archived as a tradeoff experiment rather than replacing the selected sample-anchor model.
- Hardcap follow-up: `results/v11_positive_symmetric_multihead_lc_hardcap_continue_log_adslen09/` starts from the selected sample-anchor checkpoint and hard-caps NN outputs before cascade evaluation: `Cn1/Cn2/Cn3 <= 1000` (`C <= 1e-11 F`) and `Ln1 <= 4` (`L <= 4e-11 H`). This improves all-sample NN NMSE to `0.0455798`, better than sample-anchor continue `0.0467274`, with `293/298` better than direct and `115/298` better than optimized. L/C constraints are strictly satisfied: C exceed count `0/3576`, L exceed count `0/3576`, `Cn3_scale` max `925.442`, `Ln1_scale` max `4.0`. Strong-resonance count remains `2/298` (`dut123`, `dut72`), but `dut123`, `dut253`, and `dut258` NMSE improve relative to sample-anchor continue.
- Hardcap selected-sample plots: `plot_v11_hardcap_selected_samples.py` regenerates hardcap comparison plots for `dut72`, `dut123`, `dut253`, and `dut258` under `results/v11_positive_symmetric_multihead_lc_hardcap_continue_log_adslen09/comparison_plots/`.
- Tighter hardcap follow-up: `results/v11_positive_symmetric_multihead_lc_tighter_hardcap_continue_log_adslen09/` starts from hardcap and tightens output caps to `Cn1/Cn2/Cn3 <= 500` (`C <= 5e-12 F`) and `Ln1 <= 3` (`L <= 3e-11 H`). All-sample NN NMSE improves further to `0.0445784`, with `119/298` better than optimized and strict L/C cap satisfaction. However strong-resonance count worsens from `2/298` to `4/298`, adding `dut144`, `dut253`, and `dut233`; `dut72` is no longer strong-resonant and `dut258` improves, but `dut253` crosses the dB-jump threshold. This is a useful tradeoff result, not a clean replacement for the `Cn<=1000/Ln1<=4` hardcap model.
- Ultra-tight hardcap follow-up: `results/v11_positive_symmetric_multihead_lc_ultratight_hardcap_continue_log_adslen09/` starts from tighter hardcap and enforces the requested `Cn1/Cn2/Cn3 <= 100` (`C <= 1e-12 F`) and `Ln1 <= 1` (`L <= 1e-11 H`). The caps are strictly satisfied with C/L exceed count `0/3576`, but the model becomes over-constrained: all-sample NN NMSE worsens to `0.0545040`, better-than-optimized drops to `84/298`, and strong-resonance count is `3/298` (`dut123`, `dut253`, `dut294`). Use this as a strict-bound archive result, not as the current recommended default.
- Tied-topology follow-up: `results/v11_positive_symmetric_multihead_lc_tied_c1r1_continue_log_adslen09/` starts from ultra-tight and enforces `Cn1=Cn2` plus `Rn1=Rn2` before cascade evaluation. This removes the strong-resonance flag from `dut253` (`pred_db_d1=5.61736`, `pred_max_d1=0.0957354`) but worsens its NN NMSE from `0.545175` to `0.646391`, and the all-sample NN NMSE worsens sharply to `0.230818` with only `5/298` samples better than optimized. The tied topology is archived as a negative experiment rather than a replacement model.
- Tied-topology circuit re-optimization: `results/v11_positive_tied_c1r1_sharedopt_lhs400_connection2_goodstart_all_adslen09/` re-optimizes the connection circuit itself under `Cn1=Cn2` and `Rn1=Rn2`, using five positive free variables (`Cn1/Rn1/Cn3/Rn3/Ln1`) and writing the equivalent seven-parameter table. Current disk state contains `399` existing `LHS400_Connection2/train/TSV_RDL` samples, all optimized successfully. The all-sample tied optimized NMSE mean is `0.028715`, all `399/399` samples are better than direct, and equality checks give `max |Cn1-Cn2| = 0`, `max |Rn1-Rn2| = 0`. For `dut253`, tied optimization reaches NMSE `0.062966`, effectively matching the previous untied optimized target while satisfying the tied topology.
- Tied-topology top-start continuation: `results/v11_positive_tied_c1r1_sharedopt_lhs400_connection2_topstarts_refine_adslen09/` starts from the tied optimized targets and retries every sample with the globally best tied parameter sets as initial values. Current disk state contains `400` existing `LHS400_Connection2/train/TSV_RDL` samples; all are optimized and all are better than direct. All-sample tied optimized NMSE mean is `0.0287208`; on the `399` samples common with the previous tied run, `62` improve, `0` worsen, and mean NMSE changes from `0.0287150` to `0.0286971`. The added sample `dut353` improves from initial tied NMSE `1.218604` to `0.038158`. Equality and positivity checks pass (`max |Cn1-Cn2| = 0`, `max |Rn1-Rn2| = 0`, nonpositive count `0`), so this is the current best tied-circuit parameter target set.
- Tied-topology parameter NN: `results/v11_positive_tied_c1r1_param_nns_log_adslen09/` trains an `input -> 30 -> 30 -> 20 -> 5` network on the tied optimized free parameters (`Cn1/Rn1/Cn3/Rn3/Ln1`) and expands predictions to exact `Cn2=Cn1`, `Rn2=Rn1` before cascade validation. Equality and positivity checks pass (`max |pred Cn1-Cn2| = 0`, `max |pred Rn1-Rn2| = 0`, nonpositive count `0`), but the supervised NN is still much worse than per-sample tied optimization: all-sample NN NMSE mean is `0.120038` versus optimized tied `0.028715`; NN is better than direct for `377/399` samples but better than optimized for `0/399`. `dut253` NN NMSE is `0.227737` versus optimized tied `0.062966`; worst NN samples include `dut306`, `dut308`, `dut370`, `dut72`, and `dut123`.
- Tied-topology parameter NN with per-sample ratio clamp: `results/v11_positive_tied_c1r1_param_nns_ratio_clamped_log_adslen09/` uses the refined `400`-sample tied optimized target set and hard-clamps each predicted free parameter to `0.2x~5x` of that sample's optimized value before expanding to `Cn2=Cn1`, `Rn2=Rn1`. Equality, positivity, and ratio checks pass (`max |pred Cn1-Cn2| = 0`, `max |pred Rn1-Rn2| = 0`, ratio range `0.2~5`, nonpositive count `0`). The clamp changes `438/2000` tied free-parameter predictions, mainly `Cn3_scale` (`295/400`). It improves some samples such as `dut253` (`NN NMSE 0.227737 -> 0.110988` versus unclamped), but worsens aggregate accuracy: all-sample NN NMSE is `0.135644` versus optimized tied `0.0287208`, and common-sample mean is worse than the unclamped NN (`0.135830` vs `0.120038`). This is archived as a bounded-output diagnostic rather than the best-accuracy NN.

## v12 Current Best Model

- Current selected v12 overall model: round3 all-150 S-parameter continuation.
- Stable checkpoint copy: `model_versions/v12_hfss_v08_multihead_chain/results/current_best_model/v08_connection_multihead_current_best.pt`
- Source checkpoint: `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round3/v08_connection_multihead_all150_sparam_continue.pt`
- Reason: round3 test paper-style NMSE mean is `3.63163252939746%`; round4 increased test paper-style NMSE mean to `3.909791747799181%`, so round4 is retained only as an overfit-check archive.

## v12 paper document update (2026-07-12)

- `model_versions/v12_hfss_v08_multihead_chain/v12建模流程小论文.md`: English Markdown paper draft revised from the original Chinese draft with the current v12 method, metrics, and placeholder figure captions.
- `model_versions/v12_hfss_v08_multihead_chain/v12建模流程小论文.docx`: English Word small-paper version generated from `v12建模流程小论文.md` using `Manuscript.doc` as the template source.
- `model_versions/v12_hfss_v08_multihead_chain/code/build_v12_paper_docx.py`: VS Code-runnable build entry for regenerating the English Word paper without command-line arguments.
- `model_versions/v12_hfss_v08_multihead_chain/results/v12_paper_docx_validation_archive.md`: validation archive for generated content, English revision, metric sources, PDF/PNG render, and visual QA.
