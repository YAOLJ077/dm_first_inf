# M0 四参数 TPE LOSYO（单种子1024，尚未正式运行）

入口：`src/calibrate_m0_losyo_tpe.py`。设置：`config/fsim_m0_losyo_tpe.json`。
已在项目环境 Optuna 4.7.0 下检查。使用多变量约束 TPE，这是参数优化，不是后验采样。

## 参数与预算

| 参数 | 区间 | 采样 |
|---|---|---|
| VPD_w | 1–8 hPa | 连续均匀搜索空间 |
| I_req | 40–450 °C h | 连续均匀搜索空间 |
| RH_sp | 75–95% | 连续参数 |
| MP_sp | 3–7 h | 整数，间隔1小时 |

RH区间是连续探索范围，MP为整数探索范围；文献参考设置仍为80%、3小时。固定参数表不改角色；本实验通过明确的运行配置覆盖这四项。
MP通过模型运行配置传入，缺省调用仍使用原始3小时。温度条件、降雨/叶湿替代条件、过程方程和M0结构保持原设置。
每次试验建立独立模型状态，不跨RH/MP复用上游状态。

每折单个种子1024次，包含128次初始探索和其中1次已有同折M0参考试验。
种子为20260914，六折：6,144次参数试验，30,720次主要训练模型评估。
两个存在重复时间戳的site-year继续检查keep_first与keep_last，增加10,240次训练评估。
最终选定组合再计算6次留出预测和2次天气核查，合计最多40,968次。
该预算只包含本次四参数实验；不自动运行两参数TPE对照。
中断时未完整保存的试验需要重算，因此重启后的实际累计调用可能高于此预算。

## 目标与选择

每折只使用另外五个site-year的观测端点，最小化既有定义的窗口间绝对距离MAE。
仅五个训练site-year全部产生预测的组合可选；无预测数作为TPE约束，必须等于0。
无预测组合的training_MAE_days/training_RMSE_days为null，不虚构误差。
Optuna 4.7需要有限的目标反馈：对不可行组合使用单独记录的optimizer_value=0，
由正约束将其排除出可行目标排名；该占位值不能作为真实MAE分析或绘图。
模型异常和两种重复天气处理产生不同预测时直接停止，并保留已完成试验。

每折首个参考点来自已完成三轮grid的**同一训练折**选择，另加RH=80、MP=3。
核对原输入和训练折身份，重新计算参考点的训练误差；不复用旧预测或旧损失。
两个种子完成后，仅按训练MAE联合选取该折参数。
误差在1e-9天内并列时，先选最接近MP=3者，再选最接近RH=80者，
随后按RH、VPD、I、seed、trial升序。选择写入selection.json后才计算留出预测。
输出7/10天内预测比例、RMSE与无预测情况，但不以这些留出指标选参。
近优集合为有限搜索样本中的训练近优点，不是后验样本或可信区间。

## 命令

在项目根目录运行：

```bash
dm_first_os/bin/python src/calibrate_m0_losyo_tpe.py --settings config/fsim_m0_losyo_tpe.json --workers 4 --output-dir result/fsim_m0_losyo_tpe_4p_n1024
```

只核查配置、输入、参考点与预算，不计算模型、不创建结果目录：

```bash
dm_first_os/bin/python src/calibrate_m0_losyo_tpe.py --workers 4 --design-only
```

中断后恢复：

```bash
dm_first_os/bin/python src/calibrate_m0_losyo_tpe.py --settings config/fsim_m0_losyo_tpe.json --workers 4 --output-dir result/fsim_m0_losyo_tpe_4p_n1024 --resume
```

## 并行和恢复

使用4个独立进程评估一次试验涉及的训练site-year及天气核查，试验之间顺序ask/tell。
`tqdm`显示折、种子、试验数、当前最佳可行训练MAE和可行试验比例。
结果以固定提交顺序汇总，避免进程完成顺序改变TPE采样路径。

每个完整试验原子写入独立JSON。恢复时顺序重放ask/tell并逐项核对采样参数，
已保存试验不重复计算。中断中的未保存试验会重新计算；不会跳过或当作不可行。
恢复要求源码、配置、输入、参考点和主要依赖版本一致；不能在原目录直接把512改为更大预算。
不加载pickle状态。输出目录加独占锁，防止两个运行同时写入。

## 输出

- `effective_settings.json`、`run_identity.json`、`source_snapshot/`：实际设置、参考点、源码/输入指纹与软件版本。
- `run_metadata.json`：运行状态与预算；完成后的计算次数包含所有保存的训练和留出评估。
- `folds/<site-year>/seed_<seed>/trial_*.json`：参数、训练site-year、主要/核查预测、真实误差与约束。
- 每折 `trials.csv`、`convergence.csv`、`checkpoints.csv`：完整试验汇总与64/128/256/512节点训练收敛。
- 每折 `selection.json`、`effective_runtime_config.json`、`training_predictions.csv`、`held_out_prediction.csv`。
- 每折 `near_optimal_parameters.csv`、`tied_best_parameters.csv`：训练近优与并列组合。
- 根目录 `fold_summary.csv`、`oof_predictions.csv`、`oof_summary.json`：六折参数和留出评价。

如果一折没有可行训练组合，记录calibration_failed，不计算该折留出预测；固定六折分母保留。
全部六折都有预测时才报告完整MAE/RMSE，否则保留null并另列有预测子集的条件误差。

本次模型源码变动是MP运行接口；历史结果及源码快照保留。旧脚本若严格核对历史源码指纹，
会正常提示源码改变，不能绕过该检查而把新运行宣称为旧结果的完全复算。
