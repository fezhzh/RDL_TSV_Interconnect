# 模型版本索引

本目录按模型演进版本归类代码、模型文件和结果。每个版本目录通常包含：

- `code/`：该版本的可运行入口。
- `models/`：该版本的模型文件。
- `results/`：该版本对应的模型结果。

| 版本 | 主题 | 主要内容 |
| --- | --- | --- |
| `v00_parameter_extraction_and_dataset_building/` | 参数提取和数据集构建 | 训练前的 RDL/TSV 参数提取脚本，输出到 `training_datasets/`。 |
| `v01_matlab_mat_models/` | MATLAB `.mat` 基线模型 | `RDL_TSV_mat1` 到 `RDL_TSV_mat4` 器件模型训练与 RDL Bottom/Top/TSV 对比。 |
| `v02_mat4_cascade_and_sparameter_optimization/` | mat4 直接级联与 S 参数优化 | RDL+TSV mat4 直接级联对比、连接网络 least-squares 优化结果。 |
| `v03_single_device_sparam_finetune/` | 单器件 S 参数微调 | RDL Bottom/Top/TSV 单器件 PyTorch S 参数目标微调。 |
| `v04_hfss_split_direct_sparam/` | HFSS split 直接 S 参数代理 | `TMRDL`、`BSMRDL`、`TSV` 单器件和 `TSV_RDL` 级联残差 S 参数模型。 |
| `v05_hfss_split_circuit_param/` | HFSS split 电路参数模型 | 先提取等效电路参数，再训练参数模型和连接级联模型。 |
| `v07_connection_param_and_sparam_finetune/` | 连接网络参数模型与 S 参数微调 | 使用优化连接网络参数训练连接参数模型，再用整体 S 参数微调。 |
| `v08_connection_multihead/` | 多头连接网络 | 共享 trunk + 多连接位置 head 的连接网络建模和 k-fold 验证。 |
| `v09_rdl_lhs_dataset_comparison/` | RDL LHS 数据规模对比 | 对 LHS100、LHS200、LHS400、LHS800、组合数据集分别提参、训练参数网络、以 S 参数目标微调，并对 LHS100+200+400 级联结构做连接网络优化提参和 multi-head 训练。 |
| `v10_ads_pi_cascade/` | ADS 单器件仿真 + pi 级联网络 | 通过 Python 调用 ADS 生成 RDL/TSV 单器件 S 参数，级联后优化 8 个 pi 网络元件值，再训练 pi 参数网络并以整体 S 参数微调。 |
| `v11_ads_v08_multihead_chain/` | v11 长链 ADS + 7 参数 pi 连接网络 | 面向 13 段 `TMRDL-TSV-BSMRDL-...-TMRDL` 长链结构；复用 v10 ADS helper 和 shared-to-multihead 训练流程，改为 12 个连接位置、7 参数 Appendix 1 电路。当前入口会先检查 full-chain HFSS 目标数据是否存在。 |
| `v12_hfss_v08_multihead_chain/` | v12 长链 HFSS 等效电路 + v08 多头连接网络 | 面向同一 13 段长链结构；单器件基础级联改用 LHS400/HFSS 派生的 RDL/TSV 等效电路模型，连接处使用 v08 Appendix 1 七参数 pi 电路，并将 12 个连接位置约束为 6 个镜像对称多头。 |
| `v99_legacy_and_shared/` | 旧版/共享脚本 | 早期 S 参数计算、级联、优化、NN 比较脚本。 |

已归档版本：

| 归档路径 | 内容 |
| --- | --- |
| `archive/model_versions_v06_rdl_mat4style_and_lhs_refinement_20260706_155014/` | 旧 v06 RDL mat4-style 和 LHS refinement 代码、模型、结果。 |

维护规则：

- 新版本实验优先新增一个 `vXX_<name>/` 目录，并在本文件登记。
- `code/` 中放与该版本直接相关的可运行入口脚本。
- 大体量训练和对比结果统一放在对应版本的 `results/`。
- 没有明确对应模型版本的旧结果归档到 `archive/`。
