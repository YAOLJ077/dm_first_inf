# 统一 LOSYO 2p / 3p / 4p 校准

入口：`src/calibrate_fsim_losyo_joblib.py`；配置：`config/fsim_losyo_joblib.json`。
在项目根目录使用已配置的 Python 环境运行。依赖版本与冻结模型均在启动时校验；
当前项目的 `dm_first_os/bin/python` 满足版本要求。迁移到 Linux 时应重建相同依赖环境。

```bash
# 只检查配置、模型、输入和六折设计，不启动校准
dm_first_os/bin/python src/calibrate_fsim_losyo_joblib.py --design-only

# 正式校准：新目录，64 个工作进程上限
dm_first_os/bin/python -u src/calibrate_fsim_losyo_joblib.py --workers 64

# 中断后续跑（保持原参数、输入、源码和依赖版本；允许改变 workers）
dm_first_os/bin/python -u src/calibrate_fsim_losyo_joblib.py --workers 64 --resume

# 小规模真实数据验证，独立于正式结果
dm_first_os/bin/python src/calibrate_fsim_losyo_joblib.py --smoke --workers 2 --no-progress
```

默认正式输出为 `result/fsim_losyo_joblib/`；可用 `--output-dir` 指定新目录。
断点续跑必须使用原输出目录和相同的 `--smoke` 状态。
不会读取或拼接旧 LOSYO 的 Grid/TPE 结果；不覆盖已有非空结果目录。
`--report-only` 可重建报告；如果共同预算对应的留出预测尚未缓存，会计算该预测，
但不会重新校准或改变已冻结的训练选择。

## 固定运行顺序与并行方式

`2p Grid → 2p TPE → 3p Grid → 3p TPE → 4p Grid → 4p TPE`。
每阶段完成全部六折后进入下一阶段。Grid 使用现有的全局去重任务队列；
TPE 的六个训练折共享 Joblib 自带的 loky executor，各折保持顺序 ask/tell。
任务直接进入持续调度的共享进程池；某折完成训练评价后，可立即提交下一 trial，
不用等其他折或此前整批任务结束。单个 trial 仍须等待自己的全部训练案例完成。
首次 TPE 提交时才建立 executor，并释放 Grid 遗留的闲置 worker，避免两个进程池叠加。
64 是模型工作进程上限，不是 64 个并发 TPE trial；TPE 的实际利用率受六折待执行任务数限制。
不计重复天气检查，每折一个在途 trial、每 trial 五个训练案例，最多约 30 个模型任务；
相同参数与案例会去重，实际并发数可能更少。进度条以完整 trial 为单位，
共享计算完成后多个折仍可能同时更新；这不代表仍在等待整批任务。
不为每折创建 64 个进程。每个参数组使用独立 SQLite 缓存，避免风速门控配置相互污染。

## 旧运行升级持续调度器后续跑

服务器先按 Ctrl+C 并等待旧进程退出，再同步以下两个文件：
`src/fsim_joblib_pool.py` 和 `src/migrate_fsim_losyo_scheduler.py`。
在已激活的 `fsim-tpe` 环境、项目根目录执行：

```bash
# 仅检查兼容性，不改检查点
python src/migrate_fsim_losyo_scheduler.py

# 核对通过后记录调度器迁移，再继续原运行（进度条默认开启）
python src/migrate_fsim_losyo_scheduler.py --apply
python -u src/calibrate_fsim_losyo_joblib.py --workers 64 --resume
```

仅接受固定旧版调度器到本次持续调度器的源码哈希变化；其他源码、输入、参数或依赖版本
变化均会拒绝迁移。正式 trial、模型缓存和原 `source_snapshot/` 保持原样。
原身份文件、新身份文件、新调度器快照和时间记录保存于 `scheduler_migrations/`；
最后原子更新 `run_identity.json`。无需修改原校准入口或绕过正常 resume 校验。
自定义输出目录需对两个命令使用同一 `--output-dir`；自定义设置和 smoke 状态也须一致。
新输出目录不需要迁移。仅需纯文本日志时可加 `--no-progress`。

调度器使用 [loky executor API](https://loky.readthedocs.io/en/stable/API.html)，
不增加依赖；worker 内部数值库线程仍限制为 1。

## 数据、参数和优化条件

六个 site-year；每折训练五个、测试一个。输入、基础参数表和模型实现均使用现有冻结版本。
新脚本不修改模型过程，也不修改现有校准入口。

| 参数组 | 校准参数 | 固定设置 |
|---|---|---|
| 2p | VPD_w: 1–8；I_req: 40–450 | RH=80，MP=3，关闭风速门控 |
| 3p | VPD_w: 1–8；I_req: 40–450；u_tr: 0–8 | RH=80，MP=3，开启风速门控 |
| 4p | VPD_w: 1–8；I_req: 40–450；RH: 80–95；MP: 3–6 | 关闭风速门控 |

采用当前 Joblib 配置的参数范围，而不是旧 LOSYO 4p 的 RH=75–95、MP=3–7。
3p 与 4p 检验不同扩展，不构成逐项嵌套的参数链。3p 的内部模型标记 M1 表示风速门控 FSIM，
不是 FSIM-S。

TPE 每折 1024 次、启动阶段 128 次，种子 20260914；Grid 每折上限 1024 个候选，
使用三轮训练驱动的局部细化，近优容差 1 天。初始粗网格分别为 30、270、480 个候选。
Grid 的 RH、MP、风速层沿用既有离散层级；只细化 VPD/I。
TPE 的 RH、风速参数连续，MP 为整数。
初始 VPD=2、I=200、RH=80、MP=3；3p 初始 u_tr=4。
TPE 不读取前一阶段 Grid 的最优点，初始点独立于疾病观测结果。

目标是训练集的观测区间—预测窗口距离 MAE，重叠时误差为零；
要求全部五个训练案例产生预测，才视为可行参数。
训练目标、Grid 细化、TPE 提案、并列最优选择均不读取本折留出结果。

## 结果和比较口径

- `run_metadata.json`：阶段、各折完成状态和运行记录。
- `run_identity.json` / `source_snapshot/`：模型、脚本、输入哈希和依赖版本。
- `split_design.json`：36 个任务的训练与测试键。
- `2p/`、`3p/`、`4p/`：SQLite 断点、训练轨迹、参数选择和留出预测。
- `oof_predictions.csv` / `fold_summary.csv`：逐 site-year 预测和逐折参数。
- `method_summary.csv`：六折 MAE、RMSE、预测率和容差命中率。
- `paired_predictions.csv` / `comparison_summary.csv`：配对误差差值，保留失败案例。
- `comparison_design.json` / `comparison_report.md`：共同预算和解释边界。

`view=final` 使用各组实际完成的全部试验。Grid 可能早于上限结束；TPE 固定运行到预算。
不能把“相同上限”写成“相同实际次数”。最终 TPE 各参数组有相同试验次数；
最终 Grid/TPE 比较必须检查 `same_candidate_budget`。

`view=matched` 对所有组、所有折、所有种子取实际训练试验数的最小值，
然后只在该截断位置之前的训练记录中选择参数并计算留出预测。
截断位置不依赖留出误差，所有配对具有相同训练候选次数。
若截断位置不超过 128，比较仅涉及 TPE 启动阶段，不能据此评价自适应 TPE 的优势。
Grid 的早停和两种算法不同的候选空间仍应在解释中保留；相同次数不保证同等收敛。

正的配对 `delta_absolute_error_days` 表示右侧方法误差更大。
缺失预测不赋零误差，完整 MAE 要求六个案例均有预测；另给出条件 MAE。
六个 site-year 是独立比较单位，方法和种子不能当作新增田间样本。
该结果用于 FSIM 参数化及优化方法比较，不是 FSIM 与 FSIM-S 的主比较。
