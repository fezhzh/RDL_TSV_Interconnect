# v02 mat4 直接级联与 S 参数优化

定位：使用 `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat4/` 对 RDL+TSV 整体结构做直接级联对比，并在直接级联基础上插入连接网络做 least-squares 优化。

代码入口：

- `code/compare_model_cascade_results.py`
- `code/compare_optimized_sparameter_results.py`
- `code/Calc_SP_and_Opt2.py`
主要模型：

- `model_versions/v01_matlab_mat_models/models/RDL_TSV_mat4/`
