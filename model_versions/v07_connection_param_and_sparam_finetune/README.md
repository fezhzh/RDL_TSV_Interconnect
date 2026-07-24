# v07 连接网络参数模型与 S 参数微调

定位：从 `RDL_TSV_mat4_opt2` 的优化连接网络参数出发，训练连接网络参数模型，再以整体结构 HFSS 复数 S 参数目标继续微调。

代码入口：

- `code/train_connection_network_params.py`
- `code/fine_tune_connection_network_on_sparams.py`
