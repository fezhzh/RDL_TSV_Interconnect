# v99 结果路径

旧版脚本的结果可能散落在历史输出或归档目录中，优先检查：

- `archive/`
- `model_versions/v99_legacy_and_shared/results/`
- 脚本内部硬编码的输出目录

新实验不建议继续把结果写到根目录；应统一写入对应的 `model_versions/vXX_*/results/`，没有对应模型版本的结果归档到 `archive/model_results_unmatched_20260703/`。
