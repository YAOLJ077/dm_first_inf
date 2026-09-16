# FSIM Sobol 分析运行说明

运行类型：正式抽样；指数能否解释仍需查看收敛检查。

主响应为固定六站点年份RMSE，辅以MAE和预测产生率；逐站点年份另算有符号误差、绝对误差和是否产生预测。误差沿用预测窗口到观测显症区间的距离，未使用任何最优值选择。

| 模型 | 抽样种子 | 响应 | 状态 | 缺失响应数 |
|---|---:|---|---|---:|
| M0 | 20260913 | prediction_rate | ok | 0 |
| M0 | 20260913 | RMSE_days | unavailable_missing_predictions | 15 |
| M0 | 20260913 | MAE_days | unavailable_missing_predictions | 15 |
| M0 | 20260913 | accurate_prediction_rate_7d | ok | 0 |
| M0 | 20260914 | prediction_rate | ok | 0 |
| M0 | 20260914 | RMSE_days | unavailable_missing_predictions | 17 |
| M0 | 20260914 | MAE_days | unavailable_missing_predictions | 17 |
| M0 | 20260914 | accurate_prediction_rate_7d | ok | 0 |
| M1 | 20260913 | prediction_rate | ok | 0 |
| M1 | 20260913 | RMSE_days | unavailable_missing_predictions | 265 |
| M1 | 20260913 | MAE_days | unavailable_missing_predictions | 265 |
| M1 | 20260913 | accurate_prediction_rate_7d | ok | 0 |
| M1 | 20260914 | prediction_rate | ok | 0 |
| M1 | 20260914 | RMSE_days | unavailable_missing_predictions | 271 |
| M1 | 20260914 | MAE_days | unavailable_missing_predictions | 271 |
| M1 | 20260914 | accurate_prediction_rate_7d | ok | 0 |

状态含义：

- `ok`：本次抽样具备估计条件；不自动意味着已经收敛。
- `unavailable_missing_predictions`：响应存在无预测造成的缺失；不删除样本、不填惩罚值，整项Sobol估计跳过。
- `undefined_constant_response`：响应没有变异，指数没有定义，不等同于所有参数效应为零。
- `undefined_base_sample_variance`：Saltelli的A/B基样本无变异。
- `unstable_numerical_estimate`：存在非有限估计或置信区间，不能据此给出稳定排名。

## 解释与限制

独立均匀取样是本次参数空间探索假设，不是生物学参数分布。Sobol指数描述所选范围内响应变异的贡献；它不表示参数增大必然改善误差。
各站点年份分别计算，不能把模型运行次数当作独立田间样本。RMSE强调大误差；MAE作为补充。重叠率不作为本轮响应。
逐样本六站点年份中只要一个无预测，汇总RMSE和MAE即记为缺失。不能以较少站点的误差替代。对于含缺失的完整设计，脚本只报告响应可用性及预测产生率的合法Sobol估计，不自动缩小范围或构造条件Sobol。
置信区间是指数估计的bootstrap区间，不是六个站点年份泛化性能区间。有限样本指数可能小于0或大于1，原值保留。
收敛文件比较递增N下的指数变化，以及独立scramble种子的差异；阈值只是预设诊断标准，不能代替数值和科学判断。
此前发现的重复年末天气时间戳在分析副本中保留末次记录，按配置复算保留首次记录；原始天气不修改。若两种策略结果不同，须在正式解释前解决输入不确定性。

## 输出

- `sobol_design.csv`：按原始顺序保存的Saltelli设计。
- `predictions.csv`：逐抽样、逐站点年份预测，包括无预测记录。
- `response_vectors.csv`：与抽样行严格对齐的响应。
- `response_diagnostics.csv`：每个响应在各N和种子下是否能估计。
- `sobol_indices.csv`：S1、ST、S2及置信区间半宽；不输出最佳参数。
- `convergence.csv`、`seed_stability.csv`：样本量和种子稳定性检查。
- `run_metadata.json`、`source_snapshot/`：配置、版本、源代码及输入指纹。

## 新增容许误差响应

容许误差（天）：[7.0]。
accurate_prediction_rate 为产生预测且绝对区间误差不超过容许天数的比例，分母固定为六。无预测或超出容许误差记为未满足条件；程序错误不能记为未满足条件。逐站点响应为同一条件的0/1指标。
这是新增的准确预测比例，不是原始RMSE或区间重叠率。10天是探索设置，不是已验证的管理标准；响应完整也不保证方差非零或指数收敛。
