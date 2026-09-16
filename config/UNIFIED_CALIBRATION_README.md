# FSIM 统一校准与对比

入口：`src/calibrate_fsim_unified.py`；配置：`config/fsim_unified_calibration.json`。

本入口比较冻结 M0/FSIM 的参数数量和搜索方法，使用与此前四参数 TPE 相同的 `src/tpe_m0_reference` 模型快照；不校准 FSIM-S 的 `beta_sp`。site 和 year 分组留一法分别运行，均为三个留出折，每套方法生成六个站点年份的折外预测。两套分组结果不是十二个独立样本，也不能代替论文计划的主要 LOSYO 评价。

## 统一设置

| 参数 | 范围 | 完整网格间隔 | 两参数方案 |
| --- | --- | --- | --- |
| VPD_w | 1–8 | 0.5 | 校准 |
| I_req | 40–450 | 10 | 校准 |
| RH_sp | 75–95 | 5 | 固定 80 |
| MP_sp | 3–7，整数 | 1 | 固定 3 |

四个方案统一输入、冻结模型、训练/留出划分、重复气象记录处理、其他运行参数、损失函数、缺失预测处理和并列最优判据。训练目标是显症窗口区间距离 MAE；训练站点年份必须全部产生预测才可参与最优选择。运行错误会中止任务，不会被归为普通无预测。

| 方案 | 每折完整候选预算 | 公共预算 |
| --- | ---: | ---: |
| 2p_grid | 630 | 630 |
| 4p_grid | 15,750 | 630 |
| 2p_tpe | 1,024 | 630 |
| 4p_tpe | 1,024 | 630 |

默认 TPE 种子为 20260914，启动阶段 128 次（计入总预算）。所有方法首个候选固定为 `(VPD_w=2, I_req=200, RH_sp=80, MP_sp=3)`，不读取旧校准最优值。网格剩余候选按预先固定种子打乱；完整运行穷举整个网格，不另做局部加密。TPE 对前三个参数连续采样、对 MP_sp 整数采样，因此这是规定网格与连续 TPE 的比较，不是完全相同离散候选集合内的比较。

`view=common` 比较每折前 630 个候选。四参数网格的这一结果只是固定顺序的部分网格，不是完整网格最优值。`view=final` 使用每个方案的完整预算；跨参数数量比较时须同时检查 `same_candidate_budget`。两参数空间是四参数空间中 RH_sp=80、MP_sp=3 的切片。

两参数和四参数 TPE 均为每折总计 1,024 次，包含 128 次启动采样及随后 896 次自适应采样。默认全部方案总计 110,568 次训练候选评估，缓存前包含重复气象策略检查的模型评估计数为 589,696；留出评估另计。同一参数、输入和气象策略的确定性结果共享缓存，减少实际计算，但每个优化器只能收到当前折的训练结果。各折 TPE 严格依序 ask/tell，工作进程数不会改变候选序列。默认单个种子不能量化 TPE 的随机波动；需要时可在首次正式运行前配置多个种子，分别报告，不能按留出误差选最佳种子。

## Linux 运行

默认依次执行 Grid、TPE 两阶段，前一阶段所有折及其留出评价完成后，自动进入后一阶段，最后合并对比表。全局默认上限为 16 个模型进程，两个阶段复用同一个进程池与缓存。

Grid 每折最多提前提交 4 个候选，候选内的各站点年份也可并行；结果按原有候选顺序写入逐次检查点，乱序完成不改变公共预算前 630 个候选、并列选参或最终结果。所有折共享精确模型缓存，避免相同计算重复提交。TPE 阶段各折共同推进，每折仍在获得当前候选的完整训练结果后才提出下一个候选。

Grid 提前计算的结果可能在中断时已缓存但尚未写入逐次记录；同版本续跑会复用这些缓存。该调度升级改变了源码身份，不支持直接续跑升级前的结果；本次按用户要求清理后重新运行，不迁移旧检查点。

当前 `fsim-tpe` 环境已经具备依赖；SQLite 是 Python 标准库，无需额外安装。脚本核验 Python 3.12.7、numpy 2.4.0、pandas 2.3.3、scipy 1.16.3、optuna 4.7.0、tqdm 4.67.1。所有项目路径基于脚本位置解析，支持 SMB 复制后的 Linux 路径。

```bash
cd /home/dell/ZG_Team/yaolinjia/dm_first_inf
source /home/dell/miniforge-fsim/etc/profile.d/conda.sh
conda activate fsim-tpe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python src/calibrate_fsim_unified.py --workers 16 --design-only
```

正式前可在 Linux 做小规模实现检查（使用真实数据，但结果不能作为正式分析）：

```bash
python src/calibrate_fsim_unified.py --workers 4 --smoke --no-progress
```

正式前台运行，默认结果目录为 `result/fsim_unified_calibration`：

```bash
python src/calibrate_fsim_unified.py --workers 16
```

上述数字是全局模型进程上限，不是每个折各开 16 个进程。当前服务器有 48 个逻辑 CPU；16 个单线程模型进程满载时约对应 33% 的逻辑 CPU 时间，主进程调度等另有开销，不能视为严格配额或保证提速。保留上面的数值库单线程环境变量，避免嵌套线程争抢 CPU。实际全机占用还取决于其他工作负载。

后台运行（与前台命令二选一，不要同时启动）：

```bash
mkdir -p run
nohup python -u src/calibrate_fsim_unified.py --workers 16 --no-progress > run/fsim_unified.log 2>&1 &
echo $!
tail -f run/fsim_unified.log
```

记录 `echo $!` 打印的 PID。`tail -f` 中 Ctrl+C 只停止看日志。需要停止后台任务时，先用 `ps -p 实际PID -o pid,args` 核对目标，再执行 `kill -INT 实际PID`。脚本显式处理 SIGINT/SIGTERM 并保存逐候选检查点；等待当前计算退出后再续跑。后台模式关闭动态进度条，但每 128 个新候选打印一次进度，也可从结果目录的 `run_metadata.json` 查看已完成的折。

本次新版本运行中断后续跑（允许调整支持的进程数）：

```bash
python src/calibrate_fsim_unified.py --workers 16 --resume
```

后台续跑：

```bash
nohup python -u src/calibrate_fsim_unified.py --workers 16 --resume --no-progress >> run/fsim_unified.log 2>&1 &
echo $!
```

续跑会重放已存 TPE 提案并核对缓存，以恢复相同随机序列；这一步可能需要时间，但已缓存模型不重新计算。修改参数范围、种子、预算、模型源码、输入或依赖版本后不可复用旧检查点，须使用新的 `--output-dir`。单独运行某方案时可用 `--scheme site` 或 `--scheme year`、`--arms 2p_grid 2p_tpe`，并给不同结果目录；续跑时保留原选项。

## 结果读取

- `method_summary.csv`：各方法、分组、种子、预算的 MAE、RMSE、预测覆盖率、失败数量；只有六个站点年份全部预测成功才给完整 MAE。
- `oof_predictions.csv`：六个站点年份逐一折外预测，保留无预测与校准失败。
- `paired_predictions.csv`：相同站点年份的方法配对差值，`delta_absolute_error_days = 右侧绝对误差 − 左侧绝对误差`；负数表示右侧更好。
- `comparison_summary.csv`：配对覆盖、平均差值与候选预算是否相同。缺失预测时仅报告明确标为 conditional 的条件均值。
- `fold_summary.csv`：训练选参、训练误差及所用预算。
- `comparison_report.md`：比较口径与已完成方法摘要。
- 每折 `trials.csv`、`convergence.csv`、`training_checkpoints.csv` 与 `budget_*/`：训练轨迹、并列/近优参数及留出结果。
- `checkpoints.sqlite`：精确模型缓存、逐候选记录及留出结果；`run_identity.json`、`source_snapshot/` 保存身份和源码。

训练完全结束后，先固定公共/完整预算各自的训练最优参数，再计算留出结果；不会用留出结果反选预算或参数。汇总文件随完整方法的折结果就绪更新；只完成部分折时，尚未出现对应六站点年份摘要是正常的。运行耗时包含共享进程池等待及缓存命中，不能作为独立算法速度基准。

停止运行后可重新生成已有结果的对比表：

```bash
python src/calibrate_fsim_unified.py --report-only
```

## 旧结果与清理

本次用户要求删除旧统一校准的部分结果。在 Linux 确认旧任务已停止后，以下命令会永久删除该目录，包括缓存与逐折结果；不删除输入、模型或更早的其他结果目录。只在重新开始前执行一次，不在新运行启动后执行：

```bash
rm -r -- /home/dell/ZG_Team/yaolinjia/dm_first_inf/result/fsim_unified_calibration
```

其他旧结果无需删除。新入口不读取此前 `fsim_m0_grouped_tpe_4p_linux` 的最优值或检查点，也不覆盖该目录。保留输入、`src/tpe_m0_reference`、环境和旧参考结果。不要删除正在使用的新 `checkpoints.sqlite` 或复制活动数据库。

如果只是想把此前中断目录标为历史结果，确认旧任务已经停止后，可执行以下可恢复的重命名；不是运行新脚本的必要步骤：

```bash
test ! -e result/fsim_m0_grouped_tpe_4p_linux_interrupted_20260914 && mv result/fsim_m0_grouped_tpe_4p_linux result/fsim_m0_grouped_tpe_4p_linux_interrupted_20260914
```

新结果目录非空时脚本拒绝覆盖。要完全重新做一轮，请指定新目录，例如 `--output-dir result/fsim_unified_calibration_v2`，而不是删除已有记录。
