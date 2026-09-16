# Joblib 粗网格与局部加密

入口：`src/calibrate_fsim_joblib_local.py`。配置：`config/fsim_unified_joblib_local.json`。独立结果目录：`result/fsim_unified_joblib_local/`。保留 worker、原 joblib 和完整全局 Grid 的源码与结果；本版不读取它们的最优参数，不复用不同运行身份的检查点。

## 同步后的参数空间

| 参数 | 范围 | Grid | TPE |
| --- | --- | --- | --- |
| VPD_w | 1–8 | 粗网格后局部细分 | 连续采样 |
| I_req | 40–450 | 粗网格后局部细分 | 连续采样 |
| RH_sp | 80–95 | 80、85、90、95，步长 5 | **连续采样** |
| MP_sp | 3–6 | 3、4、5、6，步长 1 | **整数小时** |

两参数方案继续固定 RH_sp=80、MP_sp=3，只校准 VPD_w 与 I_req；四参数方案搜索全部四项。冻结 FSIM/M0 模型、气象重复策略检查、显症窗口损失和并列规则不变，不包含 FSIM-S 的 beta_sp 校准。

## 粗网格与加密规则

初始 VPD_w：`[1, 2, 4, 6, 8]`；初始 I_req：`[40, 100, 200, 300, 400, 450]`。两参数粗网格为 30 个候选；四参数粗网格为 `5×6×4×4=480` 个候选。它们覆盖范围端点及共同的初始候选 `(2,200,80,3)`，不是先跑完整四维网格再追加加密。

每折最多三轮加密：

1. 只根据本折训练集的可行候选，保留距最佳训练 MAE 不超过 1 天的近优点；无可行点时在预算内扩大细分覆盖。
2. 找出包含近优点的 VPD/I 网格单元，在单元中心和边中点提出新候选；RH/MP 保持上述离散网格，不增加小数小时或 RH 中间值。
3. 多个竞争区域通过固定种子的单元顺序、轮流分配候选得到覆盖。预算不足时不能保证所有近优单元都充分探索；已完整采样的单元拆成四个子单元，后续继续细分。
4. 每折 Grid 最多 **1,024** 个不同候选，包含粗网格和全部加密轮次。剩余预算分摊到剩余轮次；达到三轮或没有新候选时可提前结束，不为凑足次数重复计算。

每轮各折独立生成自己的计划，然后把所有计划交给同一全局 joblib 队列，跨折、跨方案去重并行。没有每折提前四个候选的限制。下一轮必须等待本轮训练反馈，因为新区域由训练损失决定。所有加密轮次结束后才进行最终留出评价，不允许留出误差影响区域选择。

Grid 先运行，之后 TPE 各折独立顺序更新，共享最多 64 个计算进程。TPE 每折仍为 **1,024 次＝128 次启动＋896 次自适应**。设计输出中 Grid 的 trials_per_fold、总候选和总模型计数表示上限，不是必须完成的数量；缓存复用也会降低实际模型调用数。

## 结果比较

`view=final` 使用各折实际完成的候选数量，`fold_summary.csv` 保留逐折 n_trials；`method_summary.csv` 增加 min_trials_per_fold/max_trials_per_fold，只有各折次数一致时才给统一的 n_trials。

`view=common` 使用前 630 个候选，仅当一套方案所有折都达到 630 次时才输出。若局部 Grid 提前结束、某折不足 630，不会用最终结果假冒“630 次结果”，对应公共预算方法摘要会缺席。

`comparison_summary.csv` 的 same_candidate_budget 按每个配对站点年份的所属折逐一核对，而不是只比较第一折。局部 Grid 不等于穷举最优，可能漏掉粗网格未发现的狭窄最优区域。不能把搜索方法、搜索分辨率及实际预算差异都解释成优化器本身的优劣。

每折 `grid_round_0.json` 至 `grid_round_3.json` 记录预先冻结的本轮候选和训练键；`trials.csv` 增加 round 列。`grid_rounds/round_N/grid_execution.json` 记录该轮去重计算量及缓存命中。缓存逐模型写入，trial 逐候选写入；续跑会重新推导并核验相同加密计划。

## Linux 运行

先正常停止旧任务并等待完全退出。保留旧结果，不加 --resume 启动新空间：

```bash
cd /home/dell/ZG_Team/yaolinjia/dm_first_inf
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python src/calibrate_fsim_joblib_local.py --workers 64 --design-only
python src/calibrate_fsim_joblib_local.py --workers 64
```

依赖与上一 joblib 版本相同，无需新增包。输出是新的 `result/fsim_unified_joblib_local/`，不会覆盖任何旧结果目录。本版本自己的续跑：

```bash
python src/calibrate_fsim_joblib_local.py --workers 64 --resume
```

可把 workers 改为 32 等支持值续跑；参数空间、加密规则、数据及代码变动则必须另开目录。64 进程不保证优于 32，实际速度取决于硬件及背景负载。

自动测试：`dm_first_os/bin/python tests/check_joblib_local.py`。真实数据小规模实现检查：`python src/calibrate_fsim_joblib_local.py --smoke --workers 4 --no-progress`，使用独立 smoke 目录且不会形成正式校准结论。
