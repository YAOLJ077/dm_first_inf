# 孢子观测层开发：方案1

## 第一步：固定 B 参数重放

`replay_b_fixed.py` 读取 `result/fsim_abc_tpe_seed123/B` 已保存的逐折参数和完整运行配置，并调用该次运行的冻结源码快照。不会启动 TPE、重新选参或引入孢子观测。固定参数是每折各自的参数，不是所有折共用一组参数。

默认重放 LOSYO 6 折、留一地点 3 折、留一年份 3 折，每折两种选择：

- `min_train`：最低训练 MAE，正式基线。
- `train_plus1_test_selected`：训练 MAE 不超过最低值加 1 天后，再按测试 MAE 择优的**原有探索性选择**。这一版本使用了测试结果，不能作为独立测试性能或正式增益基线。

每种选择均重放训练和留出案例，共 144 次案例模拟，其中 36 次为留出预测。1024 是原校准总 trial 数（包含 128 次启动），不是本脚本的搜索预算。

## Linux 终端

```bash
conda activate /home/dell/miniforge-fsim/envs/fsim-tpe
cd ~/ZG_Team/yaolinjia/dm_first_inf
python src/spore_observation/replay_b_fixed.py --design-only
python src/spore_observation/replay_b_fixed.py --workers 4
```

中断后使用相同参数并追加 `--resume`。允许改变 `--workers`，不允许混用不同代码、输入或任务设计。脚本检查源文件、数据及已完成案例的哈希，并拒绝同时向同一结果目录运行两个进程。

```bash
python src/spore_observation/replay_b_fixed.py --workers 4 --resume
```

为保持基线可复现性，要求与原运行一致的 Python、NumPy、pandas 版本，版本不匹配时会明确报错，不自动修改环境。还需要 matplotlib。

## 输出

- `result/spore_observation/plan1_b_fixed/`：独立结果目录。
- `fig/spore_observation/plan1_b_fixed/`：六面板 PNG/PDF 对照图。
- 案例路径：`选择规则/划分方式/折名/train或test/site-year/`。

每个案例保存完整配置、选择记录、首个病斑窗口及评分，以及附着孢子、传播队列、沉降、离体存活、原始感染链、寄主门控后感染链、折叠后全部候选窗口等 CSV。`cohort_outcomes.csv` 记录离体存活队列是否完成感染链、通过寄主门控、被合并或保留；冻结引擎没有暴露每个早期退出的详细原因，不能将 `primary_chain_not_completed` 直接解释为某个特定生物学失败原因，也不能将其视为所有上游队列的完整失败清单。

根目录 `predictions.csv` 保存训练及测试预测；`fold_summary.csv` 检查训练 MAE；`reproduction_checks.csv` 逐项核对留出预测与原记录；`method_summary.csv` 汇总留出结果。`run_metadata.json` 中 `status=complete` 且 `baseline_reproduced=true` 表示全部选定任务通过复现检查。

未产生预测的案例保留为失败结果，误差不补零。存在失败时完整组 MAE 留空，另外报告成功预测案例的条件 MAE及预测数；图中同样标记这一口径。

## 小规模检查与后续扩展

可用 `--schemes`、`--rules`、`--folds` 限定任务，用 `--case-role test` 仅输出留出案例。改变设计时应同时指定新的 `--output-dir` 和 `--fig-dir`，避免与正式重放混淆。例如：

```bash
python src/spore_observation/replay_b_fixed.py --folds DA_MPT_2022 DA 2024 --workers 4 --output-dir result/spore_observation/smoke_b_fixed --fig-dir fig/spore_observation/smoke_b_fixed
```

这个检查包含三种划分、两种选参规则及留一地点中原本无预测的案例。后续孢子采样区间映射、观测支持度和窗口生成方案另建脚本；本脚本产物只作为固定机制与参数下的诊断基线，不预先改变病斑输出规则。方案依据见根目录 `SPORE_OBSERVATION_NOTES.md` 第 10–11 节。

本地实际验证结果保存在 `result/spore_observation/smoke_b_fixed_v2/`：36次模拟完成，12条留出记录（含无预测案例）全部与原结果匹配，6组选定参数的训练 MAE 全部复现；PNG/PDF 已生成。该目录是部分折的开发检查，不是全折正式结果。

## 第二步：采样区间与队列关联

`link_spore_cohorts.py` 读取已核验的孢子审计区间和 B 重放输出，直接构建关联表，不拟合参数、不赋权、不改变原预测。

```bash
python src/spore_observation/link_spore_cohorts.py
python src/spore_observation/link_spore_cohorts.py --plot-only
```

也可用 `--plot` 一次生成表格和图件。默认表格位于 `result/spore_observation/plan1_b_links/`，六个正式 LOSYO 案例的 PNG/PDF 位于 `fig/spore_observation/plan1_b_links/`。

| 表格 | 含义 |
| --- | --- |
| `cohort_observation_links.csv` | 每次运行每个已导出传播队列一行，关联采样区间、正值／零值／无观测、后续过程及折叠事件编号 |
| `candidate_observation_links.csv` | 原规则折叠后保留的候选；观测状态对应被保留队列本身，不代表所有贡献队列的综合支持 |
| `event_interval_links.csv` | 每个折叠后感染事件与各采样区间的关联，保留合并前贡献队列数量，避免只查看被保留队列而遗漏证据 |
| `interval_cohort_summary.csv` | 每次运行每个采样区间一行，包括没有传播队列的区间；计数只在此粒度使用，不能跨重复运行累加 |
| `case_diagnostics.csv` | 全部运行的候选覆盖及观测对应诊断 |
| `losyo_primary_diagnostics.csv` | 最低训练 MAE 版本的六个 LOSYO 留出案例主诊断 |
| `link_metadata.json` | 文件哈希、处理约定和解释限制 |

采样仅记录日期，使用 `前次采样日 < 传播日期（归一为日历日）≤ 本次采样日`；首段为3天。这是日尺度关联约定，不代表已知采样小时。区间计数在多个队列行中重复出现，字段名为 `interval_count_shared_not_additive`，不得加总，也不等于每个队列的剂量。没有观测不编码成零。`retained_event_Num` 可追溯同感染时间折叠前的不同贡献队列。

所有 `posthoc_*` 列和候选与实测病害窗口的重叠数仅用于事后定位问题，不能据此挑选预测候选。正值／零值是证据标签，不是已经标定的支持度或惩罚权重。表格从已导出的传播队列起步，并未重建产孢之前或未传播的失败过程。

## 第三步：定位感染—潜育链终止环节

```bash
python src/spore_observation/diagnose_chain_termination.py
```

默认只运行正式最低训练 MAE 的六个 LOSYO 留出案例，输出到 `result/spore_observation/plan1_chain_diagnostics/`。不重新校准。脚本以 Python 行执行跟踪读取冻结 `stage_primary_chain` 实际执行的退出分支，不更改函数内部计算；仅跟踪该函数，不记录其他调用的局部变量。每个队列记录一个终止原因，并与关联表合并。

每次诊断须逐项复现原预测，并比较七张阶段表：日期、队列编号和状态完全一致，浮点数量仅允许绝对差异不超过 `1e-12`（跨平台数值库可能存在机器精度差别），否则报错并将运行标记为失败。核验表同时记录是否字节级一致及最大浮点差异。`cohort_termination_diagnostics.csv` 保存完整诊断；`termination_summary.csv` 按观测状态和退出原因汇总；`positive_interval_termination.csv` 按正捕获区间汇总。案例子目录同时保留重放阶段表和 `termination_trace.csv`。

终止原因包括：无孢子囊存活窗口、存活窗口内无湿润小时、未达到释放条件、释放后无存活窗口、无寄主合格的湿润小时、未达到感染条件、潜育未完成、完整链完成。记录的原因是本次参数和输入下最先阻止模型继续执行的分支，并非对田间真实失败机制的直接证明；一个早期退出也可能掩盖下游限制。

`host_ready_at_or_after_suz_end` 用于辨别寄主就绪时间是否不早于游动孢子窗口结束。逐步变量只在当前队列确实到达相应阶段时记录，其他阶段留空，不沿用上一队列值。小时数按原小时序列中的步数统计。表中的区间计数仍为共享证据，不得跨队列加总。潜育未完成需区分输入时间序列结束导致的右截断，不能直接解释为生物学上不发病。

需要扩展到全部三种划分、两个版本的留出案例时使用另一个目录：

```bash
python src/spore_observation/diagnose_chain_termination.py --schemes losyo site year --rules min_train train_plus1_test_selected --output-dir result/spore_observation/plan1_chain_diagnostics_all_test
```

默认顺序执行；同一设计可重复运行，重建诊断结果，不实现断点缓存。代码或任务设计改变时须换新目录。不要在调试器或 coverage 中运行，因为脚本需要独占 Python trace 接口。

## 捕获率与折内标准化的每日压力

```bash
python src/spore_observation/build_spore_pressure.py
```

输出目录：`result/spore_observation/plan1_pressure/`。基于已核验的102个采样区间计算，不修改原工作簿或模型。没有读取病害端点参与压力计算，也没有拟合 beta。

- 捕获率 `r = 区间计数 / 区间天数`，单位为计数/天，不是空气体积浓度。
- 对数值 `x = log(1 + r / r_ref)`，默认 `r_ref=1计数/天`。
- 每折用训练组求均值 m 和总体标准差 s；训练 site-year 等权，其内部区间按持续天数加权。相对压力 `z = (x-m)/s`。因此，同一案例在不同折下 z 可以不同。
- 采样时刻和报告延迟没有记录：默认采样日期的次日零时起可使用。仅向后延续最近可用记录，绝不把尚未返回的区间计数回填到过去。
- 距采样日期不超过7天（含第7天）可用，超过标为 `stale`；这只是预设的新鲜度规则，不是孢子寿命。末次记录也依同样规则自然过时，不根据“已知监测终止日期”提前截断。
- `zero` 是有效零捕获；`unobserved` 是尚无可用记录；`stale` 是记录过时。后两者 z 留空，候选固定映射系数回退1；不能将它们当作零捕获。
- `fixed_mapping_factor = exp(z)` 仅为后续试验提供候选输入；没有限制系数边界、没有宣称它是有效感染概率，也未接入 FSIM。

主要文件：`capture_intervals.csv`（102条捕获率）、`daily_capture_background.csv`（6例逐日可用背景）、`fold_scalers.csv`（12折训练尺度）、`daily_pressure_by_fold.csv`（每折训练/测试压力）、`losyo_oof_daily_pressure.csv`（6例正式 LOSYO 留出压力）、`pressure_summary.csv`（状态覆盖和取值范围）。`interval_pressure_by_fold.csv` 是区间诊断表，其 z 在 available_date 之前不能用于预测。

全年背景覆盖各案例气象输入起止日期。逐日值是最近已知捕获率的估计，不是日观测，也不应跨日累加为实测总孢子量。训练历史季节视为部署前已获得；LOSYO／留年折包含未来年份训练时仍只是分组内部验证，不是严格的逐年向前预测验证。

修改报告延迟、新鲜度或参考量时使用新目录，例如：

```bash
python src/spore_observation/build_spore_pressure.py --report-delay-days 2 --max-age-days 3 --output-dir result/spore_observation/pressure_delay2_age3
```

这些规则需在增益试验前固定或列为明确敏感性分析，不应根据留出病害日期选择。

## 固定映射和仅 beta 校准：统一调度

三个入口：

- `run_pressure_experiments.py`：依次调用下列两个独立脚本，最后比较 B／固定映射／beta-only。
- `evaluate_fixed_pressure.py`：固定 beta=1，不拟合参数。
- `calibrate_pressure_tpe.py`：只校准 beta∈[0,1]，原 B 两参数始终使用各折原值。

两个实验共用 `pressure_model_core.py` 的模型计算和 `pressure_experiment_runner.py` 的校验／并行逻辑。不提供三参数联合校准，也不重新校准 B。基线仅使用最低训练 MAE 选参记录，不使用测试结果择优的探索性版本。

Linux 完整运行：

```bash
conda activate /home/dell/miniforge-fsim/envs/fsim-tpe
cd ~/ZG_Team/yaolinjia/dm_first_inf
python -u src/spore_observation/run_pressure_experiments.py --workers 64 --trials 1024 --startup-trials 128
```

默认覆盖6个 LOSYO、3个留一地点、3个留一年份折。1024是每折总 trial 数，包含 beta=0 和 beta=1 两个显式端点；其余由 Optuna TPE 建议。没有 beta 以外的可调参数。采样时延、背景过时规则、标准化和 `exp(beta*z)` 均沿用已生成的压力输入，试验期间不调节。系数不做事后裁剪，应将强映射的结果作为待评估假设。

各队列在 `dispersal_datetime` 所在日读取对应折的可用压力；z缺测时用于计算的z取0，因此因子为1；将该系数绑定到队列，在原有湿润／寄主有效时间内逐小时判断 `WD*TWD*exp(beta*z) >= 原I`。重新计算感染与潜育，再应用原门控、折叠和最早窗口规则。空气输入、沉降和孢子囊存活读取原 B 缓存，不重算上游，也不通过事后挪动原预测日期产生新结果。

`--workers 64` 是一个实验内共享进程池的最大进程数。调度器顺序运行两个实验，不会各自再建立64进程；同一输出根目录有调度锁。校准父进程管理12个独立 study，轮流向共享池提交各折 trial；worker只评价当前训练组，不读取测试误差进行选参。64进程需要相应内存，服务器资源紧张时可降低 workers。每个进程最多缓存8组案例上下文。

训练目标沿用 B：要求所有训练案例均产生预测，否则 trial 标为不可行；仅在可行 trial 中按最低训练区间 MAE 选择。并列（1e-10天以内）选较小 beta，再选较早 trial。无预测误差不补零：优化器内部占位值与不可行约束一起使用，不作为 MAE 报告。

两种实验均先验证 beta=0复现各案例B预测及三张下游阶段表，训练MAE也必须复现；beta=1作为固定映射和校准的共同端点。统一比较再次核对两个独立入口的端点预测一致。任何不一致报错终止。

输出目录：

```text
result/spore_observation/plan1_pressure_experiments/
  fixed/                  固定映射，含各折训练/留出输出
  beta_tpe/               单参数TPE，含各折trials.json/csv和selection.json
  comparison/             三组汇总、配对误差变化、beta选择与核验记录
fig/spore_observation/plan1_pressure_experiments/
```

每个最终案例输出三张感染链表、预测记录和 `cohort_changes.csv`（新增／消失／保留感染及感染时刻变化）。三组训练和留出结果保存在 `comparison/predictions.csv`，完整组与条件MAE及覆盖率保存在 `method_summary.csv`。配对误差差值仅在两者均预测时定义。图中星号表示缺失预测后的条件MAE，不能忽略覆盖率比较。

续算使用相同指令追加 `--resume`：已完成trial保留，TPE由历史记录重新建立study并继续；采样器轨迹可能与不中断运行不同。固定映射和端点检查会重算。每个trial结束后原子保存历史，尚未保存的trial需要重算。修改代码、输入、预算或任务设计时须换新的 `--output-root`。

```bash
python -u src/spore_observation/run_pressure_experiments.py --workers 64 --trials 1024 --startup-trials 128 --resume
```

独立运行时同样支持 workers、schemes、folds、output-root 和 resume：

```bash
python -u src/spore_observation/evaluate_fixed_pressure.py --workers 64
python -u src/spore_observation/calibrate_pressure_tpe.py --workers 64 --trials 1024 --startup-trials 128
```

开发检查用独立目录及小预算，不作为正式科学结果：

```bash
python -u src/spore_observation/run_pressure_experiments.py --workers 4 --trials 4 --startup-trials 2 --folds DA_MPT_2024 DA 2024 --output-root result/spore_observation/smoke_pressure_three_schemes
```

定制输出根目录的图保存在该目录的 `fig/`，避免覆盖正式图。当前阶段评价观测引入的经验性影响，不证明孢子数量改变单孢子的生理湿润要求；没有开展三参数联合校准或同复杂度无孢子阈值对照。

开发验证已完成：完整12折输入设计预检查；`smoke_pressure_three_schemes` 中三类划分各1折、每折4个trial的端到端运行，包含留一地点的无预测案例；beta=0及beta=1端点一致性通过，比较图已检查。`smoke_pressure_experiments` 中还验证了已完成任务的续算，保留4个trial且不重复追加。以上仅为功能检查，不代表完整预算的科学结果。
