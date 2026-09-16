# 全局 Grid 队列版本

入口 `src/calibrate_fsim_joblib_global.py`；配置 `config/fsim_unified_joblib_global.json`。旧 worker 入口、旧 joblib 入口及其结果全部保留。

## 调度变化

Grid 不再有“每折提前 4 个候选”的限制，也不再等待每个折提出计算请求。先将完整网格拆成唯一的“参数组合 × 站点年份 × 气象处理策略”任务，跨两参数/四参数方案和所有折去重。所有未缓存计算进入一次持续的 joblib Parallel 调用；计算进程完成任务后即可领取后续任务，不等待一个固定批次全部结束。

`--workers 64` 控制同时执行的模型计算上限；`pre_dispatch=128` 只控制提前送入后台的任务缓冲，不按折限制候选，也不是 128 个计算进程。参数候选和站点年份均可混合调度。任务耗时、序列化、结果写入、机器负载和队列末尾仍可能使部分进程短暂空闲，不承诺始终 64 个进程满载。

完整默认 Grid 包含 15,750 个四参数候选，包含全部 630 个两参数切片候选。六个站点年份和两个重复气象处理检查合计每组参数 8 个计算任务，所以完整去重 Grid 为 **126,000 个模型计算**，不是每折再各算 126,000 次，也没有增加候选预算。

Grid 首先显示全局模型计算进度 `GRID unique model computations`。所有结果缓存齐备后，各折从缓存中读取自己的训练部分，按原候选顺序记录 trial、确定最优参数，再生成留出评价与对比表。预计算各站点年份的确定性模型结果不使留出误差进入本折选参。训练损失、并列规则、公共预算前 630 次、固定参数和模型版本与旧版一致。

Grid 模型计算阶段即使尚未出现每折 trials.csv，也已经逐模型保存数据库检查点。中断后可复用所有已提交缓存，继续缺少的计算，随后重新组装各折；不需要等到一个折全部结束才能保存进度。

Grid 和折报告全部完成后再运行 TPE。TPE 的 12 个独立任务共同推进，每折仍需在完整训练反馈到达后才提出下一个候选，这是保留原 TPE 序列的算法要求。TPE 继续使用单个共享 joblib 调度器，不会因为创建多个折而各开 64 个进程。TPE 候选预算仍为 1,024 次（128 次启动 + 896 次自适应），本次没有改成批量 TPE。

## 运行

先正常停止旧运行，等待退出，保留全部旧结果。在已安装 joblib 1.5.3 的 fsim-tpe 环境运行：

```bash
cd /home/dell/ZG_Team/yaolinjia/dm_first_inf
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python src/calibrate_fsim_joblib_global.py --workers 64
```

如果尚未安装 joblib：`python -m pip install joblib==1.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple`。

新结果目录：`result/fsim_unified_joblib_global/`。无需删除旧目录。新版本不能直接续跑旧 worker 或旧 joblib 检查点，新运行从独立缓存开始；本版本中断后续跑：

```bash
python src/calibrate_fsim_joblib_global.py --workers 64 --resume
```

后台启动（与前台二选一）：

```bash
mkdir -p run
nohup python -u src/calibrate_fsim_joblib_global.py --workers 64 --no-progress > run/fsim_joblib_global.log 2>&1 &
echo $!
```

后台模式每完成 256 个新模型计算输出一次进度。`grid_execution.json` 保存 Grid 唯一任务数、本次开始时的缓存数、新完成数、模型耗时合计、实际经过时间及出现过的工作进程 PID。模型耗时合计可以超过实际经过时间，PID 数也不等于瞬间并行度。

对照表字段与旧版相同，主要看 `method_summary.csv`、`paired_predictions.csv` 和 `comparison_summary.csv`。速度比较应保持相同机器、进程数、未缓存工作量和背景负载；本版各折报告期间主要读取缓存，每折 s/trial 已不代表模型运算速度，应查看全局模型计算阶段的实际耗时。

测试：`dm_first_os/bin/python tests/check_joblib_global.py`。小规模真实数据检查：`python src/calibrate_fsim_joblib_global.py --smoke --workers 4 --no-progress`，使用独立 smoke 目录，不构成正式校准结果。
