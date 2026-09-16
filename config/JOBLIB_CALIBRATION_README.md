# 独立 joblib 校准版本

入口 `src/calibrate_fsim_joblib.py`，配置 `config/fsim_unified_joblib.json`，调度适配器 `src/fsim_joblib_pool.py`。旧版 `src/calibrate_fsim_unified.py`、旧配置和旧结果不修改。

默认结果目录：`result/fsim_unified_joblib/`。缓存、检查点、源码快照及对比表全部独立创建；不导入 worker 版本缓存，也不续跑 worker 版本的结果。显式指定旧默认结果目录会被拒绝；其他非空目录也不能覆盖。

## 任务与进程

默认先 Grid、后 TPE。Grid 阶段有 12 个折任务（两种分组 × 两种维数 × 三折）共同推进，每折最多提前提交 4 个候选；不同候选的模型计算可以同时执行。TPE 阶段同样有 12 个独立校准任务，各折内部依序 ask/tell，不会提前提出 64 个依赖尚未反馈结果的候选。

`--workers 64` 是整个程序的计算进程上限。使用一个 joblib `Parallel` 对象和 `loky` 后端，不为每折创建一个 64 进程池。一个 trial 需要多次“参数组合 × 站点年份 × 气象处理策略”的模型计算；进程池执行这些子计算。不同折请求完全相同的计算时，共享正在执行的任务或已完成缓存，不重复运行。训练选参仅使用本折训练数据。

适配器由单个调度线程调用公开 joblib API，将就绪任务分成最多 `2 × workers` 的小批次，`batch_size=1`、完成结果乱序返回，各折按原候选顺序保存检查点。数值库内部线程通过 `inner_max_num_threads=1` 限制。有限批次在批次边界仍可能等待较慢任务，不能宣称这是最高吞吐或恒定 64 个繁忙进程。主进程协调线程负责等候、缓存和记录，不是额外的模型进程池。

可选 `--schedule concurrent` 会让 Grid/TPE 两类任务同时推进；默认 `staged` 符合用户要求，也与保留的 worker 版本阶段顺序一致。不要在续跑时改变 schedule。

## 科学设置

与 worker 版本一致：冻结 M0/FSIM 引擎、六个站点年份、site/year 三折、VPD_w=1–8、I_req=40–450、RH_sp=75–95、MP_sp=3–7。两参数固定 RH_sp=80、MP_sp=3。Grid 每折两参数 630 点、四参数 15,750 点；TPE 每折均为 1,024 次，包含 128 次启动采样。相同初始点、种子、缺失处理、MAE、并列判据及公共预算 630 次；未新增 FSIM-S 校准。

## Linux 指令

先正常停止旧任务并等待其退出，保留全部旧结果。当前 `fsim-tpe` 环境原有依赖继续使用，只需补充固定版本 joblib：

```bash
cd /home/dell/ZG_Team/yaolinjia/dm_first_inf
source /home/dell/miniforge-fsim/etc/profile.d/conda.sh
conda activate fsim-tpe
python -m pip install joblib==1.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python src/calibrate_fsim_joblib.py --workers 64 --design-only
python src/calibrate_fsim_joblib.py --workers 64
```

单独的后台启动方式（不要同时执行前台命令）：

```bash
mkdir -p run
nohup python -u src/calibrate_fsim_joblib.py --workers 64 --no-progress > run/fsim_joblib.log 2>&1 &
echo $!
tail -f run/fsim_joblib.log
```

前台 Ctrl+C 或对核实后的主 PID 执行 `kill -INT 实际PID`，会停止新任务并等待已提交任务退出。不要只关闭终端或反复发送中断。joblib 版本自己的断点续跑：

```bash
python src/calibrate_fsim_joblib.py --workers 64 --resume
```

同版本续跑允许修改 workers 为 16/32/64 等受支持值；参数、输入、源码和依赖版本必须一致。更换输出目录时，在启动和续跑命令中都保留相同的 `--output-dir`。

64 是进程数，不是新增的 CPU。服务器有 48 个逻辑 CPU，64 进程可能出现超额调度，不保证优于 32；若比较框架本身，应在同样的 workers 下比较。

## 两版比较

旧版结果在 `result/fsim_unified_calibration/`；新版在 `result/fsim_unified_joblib/`。两版输出相同字段的 `method_summary.csv`、`comparison_summary.csv`、`paired_predictions.csv`、`oof_predictions.csv` 和逐折训练记录。只有完整对应的预算、折和种子才适合直接比结果；旧版中断期间尚未完成的方法可能没有汇总行。

速度对照需固定输入、完整工作量、进程数、冷/热缓存、机器和背景负载。旧版已经运行且有缓存、新版从空目录运行时，不能仅凭单条进度的 s/trial 宣称哪个后端更快。每折 elapsed 包含等候且会重叠，不应相加当作整机总耗时。

测试入口：`dm_first_os/bin/python tests/check_joblib_calibration.py`。小规模真实数据检查：`python src/calibrate_fsim_joblib.py --smoke --workers 4 --no-progress`，结果在 `result/fsim_unified_joblib_smoke/`，不属于正式校准。

实现参考：[joblib Parallel](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html) 与 [parallel_config](https://joblib.readthedocs.io/en/stable/generated/joblib.parallel_config.html)。
