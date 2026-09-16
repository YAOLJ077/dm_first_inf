# A → B → C：seed 123 的 TPE 探索

入口：`src/calibrate_fsim_abc_tpe.py`；配置：`config/fsim_abc_tpe_seed123.json`。
仅运行 TPE，不运行 grid 或 4p，也不接续旧参数空间的 trial。

| 顺序 | 维数 | VPD_w | I_req | u_tr | 风速门控 |
|---|---|---|---|---|---|
| A | 2p | 1–4.5 | 120–350 | 固定 0 | 关闭 |
| B | 2p | 1–6 | 100–600 | 固定 0 | 关闭 |
| C | 3p | 1–6 | 100–600 | 0–6 | 开启 |

全部采用连续均匀搜索空间，固定 RH_sp=80、MP_sp=3。每个研究独立使用
seed=123，首个候选点为 VPD_w=2、I_req=200（C 的 u_tr=0）。每个方案同时
推进 6 个 LOSYO、3 个留出 site、3 个留出 year 研究；每个研究 1024 次，
其中包含 128 次启动候选。三个方案共 36 个研究、36864 个 trial，模型调用数
与 trial 数不同。C 的代码内部标签 M1 表示启用风速门控的 FSIM，不是 FSIM-S。

## 64 并行的实现

- A 完成训练、基准留出评估及探索性选优后进入 B，再进入 C；全过程共享一个 64 进程池。
- 调度单位是一个候选参数在一个 site-year 上的模型计算。12 个研究轮流向
  同一队列提供任务，避免每个折单独建立进程池、只能占用少量进程。
- 启动阶段先产生全部 128 个候选，再反馈结果。自适应阶段每个研究一次
  ask 两个候选，使用 constant_liar，两者均完成后按 trial 编号 tell。
  不同研究无需互相等待，但一个研究内部保持固定的反馈顺序。
- 最多提交 256 个尚未回收结果的任务；精确重复的输入/参数组合共享结果，
  包括进行中的任务。训练评分始终只使用各研究自己的训练案例。
- 每个模型进程将底层数值库线程数限制为 1。进程池不会为每个 trial 重建。

这是一套统一的批量 TPE 设计，与旧版逐个反馈的 TPE 轨迹不同。新的 A/B/C
可在相同预算、种子、反馈规则和输入条件下比较，不能把旧 trial 拼入新实验。
改变进程数或队列长度不会改变候选反馈顺序；同一运行环境下应得到相同科学结果。
64 是进程数上限，不承诺始终满载：重复计算命中缓存、批次同步和最后几个研究
收尾时都可能降低占用率。

## 输入与选优

使用 `result/fsim_inputs_seed123_exploration/site_year_manifest.csv` 及该目录下
修正坐标后的观测副本；以气象坐标统一，原始观测文件不覆盖。坐标变更可查
`coordinate_audit.json`。气象文件、冻结模型实现及评分定义与既定探索保持一致。
训练阶段仍仅按训练区间 MAE 更新 TPE。训练完成后输出两套结果：

1. **基准结果**：保持原规则，选择最低训练 MAE 参数，再评价留出案例。
2. **探索性结果（本次采用的选优规则）**：保留训练案例全部产生预测且训练
   MAE ≤ 本折最低训练 MAE＋1 天的全部已采样参数，精确去重；评价这些参数
   在测试折上的表现，从测试预测完整的候选中选择测试 MAE 最低者。

该规则名为 `training_tolerance_then_test_MAE`，容差配置为
`selection_tolerance_days: 1.0`。它是固定 1 天容差规则，**不是按标准误计算的
One-SE 规则**。探索性结果使用测试数据选参数，不是独立测试性能；文件带有
`selection_uses_test_data=true` 和 `evaluation_role=exploratory_test_selected`。

同一留出折使用一套参数；测试 MAE 按该折所有 site-year 的区间绝对误差平均，
不逐案例分别挑参数。测试缺预测的候选保留诊断记录，但不能用成功案例的
条件 MAE 与完整预测候选竞争。如果所有候选均缺预测，明确输出选优失败，
不补零，不自动放宽训练容差；基准结果仍保留。测试 MAE 并列（容差 1e-9 天）时，
依次按较低训练 MAE、训练 RMSE、固定参数字典序和 trial 编号确定结果。

候选测试评估复用同一进程池及精确缓存，不计入 1024 次训练 trial，但会增加
模型计算量。测试结果不反馈给 TPE；扩到 2048 时，仍只使用训练评分继续搜索，
随后分别保留 1024 和 2048 预算的两套选优结果。

## Linux 执行

先将新入口、配置、本说明、`src/fsim_joblib_pool.py`、
`src/calibrate_fsim_joblib_local.py`、它们使用的冻结模型和修正输入目录同步到 Linux。
运行前检查 Python 与依赖版本；入口会严格验证配置中的版本和冻结模型哈希，
不一致时停止，不自动安装依赖或替换科学代码。

```bash
cd ~/ZG_Team/yaolinjia/dm_first_inf
conda activate fsim-tpe
python src/calibrate_fsim_abc_tpe.py --design-only
```

正式运行（前台）：

```bash
python -u src/calibrate_fsim_abc_tpe.py --workers 64
```

或使用后台运行，二选一，不要重复启动同一输出目录：

```bash
nohup python -u src/calibrate_fsim_abc_tpe.py --workers 64 > run/fsim_abc_tpe_seed123.log 2>&1 &
tail -f run/fsim_abc_tpe_seed123.log
```

中断后续跑：

```bash
python -u src/calibrate_fsim_abc_tpe.py --workers 64 --resume
```

检查 1024 结果后，决定扩展时执行以下指令。2048 是总次数，新增 1024 次；
保留原 1024 次的结果和独立预算汇总，按 A→B→C 扩展：

```bash
python -u src/calibrate_fsim_abc_tpe.py --workers 64 --trials 2048 --resume
```

已扩展到 2048 的运行，再次续跑或重新导出时也需指定 `--trials 2048`。
相同预算重新导出使用 `--report-only`。模型缓存与成批 trial 检查点存入 SQLite；
未完成批次恢复时重建相同候选并复用已落盘的模型计算。恢复要求科学配置、
代码、输入和依赖版本一致；允许改变 workers、queue-capacity 和总预算。
本次修改了科学选优配置及代码，旧版脚本建立的检查点不能直接使用 `--resume`；
新版运行应使用空输出目录，新版自身中断后可正常续跑。

小规模检查（每个研究 8 次，不能作为科学结果）：

```bash
python src/calibrate_fsim_abc_tpe.py --smoke --workers 2 --queue-capacity 16
```

## 结果与诊断

默认输出 `result/fsim_abc_tpe_seed123/`；smoke 使用单独目录。

绘制三行两列选优对照图，并在终端输出分折报告：

```bash
python src/plot_fsim_abc_selection_comparison.py --arm A
```

三行依次为 LOSYO、留出 site、留出 year；左列为最低训练 MAE，右列为
1 天训练容差内的最低测试 MAE（探索性）。每折以并列柱显示训练 MAE（灰色）
和测试 MAE（蓝色/橙色），均为非负绝对误差；面板标题另报总体测试 MAE
和预测完整性。各案例有符号偏差继续保留在终端和 TXT 报告中。
输出 PNG、PDF、分折 TXT、分折/案例 CSV 及输入哈希审计到 `fig/`。
只读取已有结果，不补算模型。每个方案需两套完整导出，可在其他方案仍运行时绘图；
后续用 `--arm B` / `--arm C`，扩展预算用 `--budget 2048`。

直观进度面板（独立终端运行，现有校准无需重启）：

```bash
python src/watch_fsim_abc_progress.py
```

该面板仅读取输出目录的 JSON，每 5 秒刷新总训练百分比、A/B/C 分方案百分比、
当前方案各折 trial 数和当前阶段。主任务约每 15 秒写一次进度，面板刷新不会
加快数据更新。`--once` 仅输出一次，`--output-dir` 可指定其他运行目录。
启动批次尚未整体反馈时，百分比可能暂时不变；模型计算完成数仍会更新。
旧版评估/选优阶段没有完整任务分母，不显示推算百分比。
训练完成后仍需等待评估及选优；元数据的运行状态 `complete` 才表示全部完成。
退出监控面板不会中断校准。只需将新监控脚本同步到 Linux，无需替换正在运行的校准代码。

- `oof_predictions.csv`、`fold_summary.csv`：全部留出预测、失败和选优参数。
- `method_summary.csv`：方案和分折方式的误差、偏差及预测完整性汇总。
- `paired_predictions.csv`、`comparison_summary.csv`：同一 site-year 的配对比较。
- 上述无前缀文件保留原最低训练 MAE 基准。**本次新增选优的结果**见
  `exploratory_predictions.csv`、`exploratory_fold_summary.csv`、
  `exploratory_method_summary.csv`、`exploratory_paired_predictions.csv` 和
  `exploratory_comparison_summary.csv`，均显式标明测试集参与选优。
- 各折 `trials.csv`、`convergence.csv`、`training_checkpoints.csv`：训练完整性、
  搜索轨迹与预算检查点。
- 各折 `budget_1024/`（扩展后另有 `budget_2048/`）：选优参数、实际运行配置、
  并列最优解、训练 MAE 距最优不超过 0.5/1 天的近优参数，以及留出预测。
- 每个预算目录的 `selection_candidates.csv` / `candidate_rule.json` 记录
  去重后的 1 天容差候选及筛选规则；其 `exploratory/` 子目录保存新选优的
  `selection.json`、实际运行配置、`selected_predictions.csv`，以及所有候选的
  `candidate_metrics.csv` / `candidate_predictions.csv`（包括缺预测情况）。
- `run_identity.json`、`source_snapshot/`：版本、代码、输入哈希与源代码副本。
- `live_progress.json`、`scheduler_metrics.csv`：约每 15 秒更新任务量、完成数、
  缓存命中、进程 PID、吞吐和耗时。in-flight 包含已提交但仍在队列内的任务，
  不等于实际忙碌进程数；利用率估算来自已完成模型的耗时，不能替代系统 CPU 监测。

正式探索不会自动扩到 2048。是否扩展应结合训练预测完整性、收敛轨迹、
近优参数分布及边界聚集情况判断；不能保证预设空间一定产生完整留出预测。

## 本地验证（2026-09-15）

新增探索性筛选版本已通过 9 项测试，包括 1 天边界、候选去重、测试缺失、
整折共同参数选择、并列排序，以及使用模拟模型完成 A/B/C 共 36 个研究后的
54 条探索性预测导出与缓存恢复；基准结果保持一致。64 进程配置预检通过。
该版本未启动真实数据校准，也未对已有研究结果进行补算。

以下为增加探索性选优之前的进程调度版本验证记录：

`dm_first_os/bin/python tests/check_abc_tpe.py` 的 5 项检查通过，覆盖研究拆分、
队列上限、跨并发度的确定性、中断恢复、追加预算和风速门控缓存隔离。
真实数据以 2 进程完成 36 个研究各 8 次的 smoke，输出 54 条留出预测、36 条
折汇总、9 条方法汇总。改用 1 进程恢复后，43 个科学结果 CSV 的 SHA256
全部一致，新增模型计算为 0。共同初始参数下，六个真实案例的零风速阈值
与关闭门控预测一致。64 进程的配置预检通过；尚未在 Linux 上测量 64 进程
的实际吞吐或满载率，也未启动正式 1024 次探索。
