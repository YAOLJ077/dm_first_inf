# VPD、I、beta 三参数联合校准

脚本：`calibrate_joint_pressure_tpe.py`。这是现有压力阈值映射实验的扩展，不是背景剂量注入版 FSIM-S。

## 设计

- 默认 seed=123，每折总计 1024 次，包含 128 次启动探索和 896 次后续试验，与历史基础 B 和 beta-only 的预算口径一致。`--trials` 表示总数，不是启动后的追加次数。
- VPD、I 的范围直接读取基础 B 保存的运行配置：分别为 [1,6]、[100,600]，线性连续采样。
- beta 线性采样 [0,1]，允许训练数据选择不采用孢子调节，不使用对数先验。
- 映射保留 `H * exp(beta * z_at_dispersal_date) >= I`，缺少压力时中性处理。beta=0 既用于复现基线，也是优化候选。
- 全部三种划分：LOSYO 六折、留 site 三折、留 year 三折。压力归一化仍使用各折训练集合。
- 每组候选重新运行完整过程，包括孢子产生、传播、沉降、存活、感染、潜伏和寄主门控。不会只对最终误差乘系数。
- 每折独立 Optuna 多变量 constrained TPE；候选每折成对提出，结果按固定顺序反馈；全局进程池默认 64 个 worker，按案例分配任务。
- 两个初始候选为该折基础 B 的 VPD/I 加 beta=0 或 1，计入启动预算。训练缺失预测的候选不可入选；仅按最低训练 MAE 选择，平局优先较小 beta，再优先较早 trial。
- 每次运行先以原 B 参数、beta=0 全过程复现所有所选折的训练/测试结果，并检查逐队列输出和训练目标。
- 联合空间包含 B，并明确评价原 B 参数、beta=0 的候选。因此所选训练 MAE 在数值容差内不会劣于原 B；这不保证测试误差改善。相同预算、种子和空间边界也不意味着二参数与三参数的搜索轨迹完全相同。

## Linux 运行

```bash
conda activate /home/dell/miniforge-fsim/envs/fsim-tpe
cd ~/ZG_Team/yaolinjia/dm_first_inf
python -u src/spore_observation/calibrate_joint_pressure_tpe.py \
  --workers 64 --seed 123 --startup-trials 128 --trials 1024
```

断点恢复：同一指令末尾追加 `--resume`。保留已完成试验，由历史重建 TPE；恢复后的随机搜索轨迹可能与连续运行不同。不得在恢复时更改代码、输入、搜索空间或预算；改变设计请使用新的 `--output-root`。

默认输出为 `result/spore_observation/plan1_joint_pressure_tpe/`：

- `run_metadata.json`：阶段、完成试验数、运行状态。
- `joint_run_identity.json`：模型/数据来源、哈希、空间、预算和版本。
- `baseline_predictions.csv`：复现的原 B 结果。
- `predictions.csv`、`method_summary.csv`：B 与 joint_tpe 的训练/测试预测及测试汇总。
- `minimum/`：按训练集最低 MAE 选择的版本，含 B 对照的 `predictions.csv` 和 `method_summary.csv`。
- `tolerance_1day/`：训练集 MAE 不超过最低值 1 天、再按完整测试集 MAE 进行探索性选择的版本，含同一 B 对照。
- `versions_vs_B_predictions.csv`、`versions_method_summary.csv`：两版与 B 的逐案例配对误差及汇总，便于直接比较。
- `selected_parameters.csv`：逐折所选三参数。
- `folds/<scheme>/<fold>/trials.json`：逐试验更新的恢复检查点。
- `folds/<scheme>/<fold>/trials.csv`、`selection.json`：最终试验表和仅训练选参记录。
- 各折 `train/`、`test/`：所选参数的全部过程表、压力系数、配置和预测窗口。

脚本不会重校准对照 B，不会修改已完成的固定映射或 beta-only 输出，也不会自动启动额外二参数对照。当前总预算与原 B 相同；多种子稳定性仍是后续独立实验。此前 1152 次、beta 正下限的版本与当前设计不兼容，不可混用其断点结果；若已运行旧版，请为新版指定新的输出目录。

## 结果绘图与同规则 B 对比

运行 `python src/spore_observation/plot_joint_pressure_results.py` 可生成三行两列的训练/测试 MAE 图及同规则 B 对照图（PNG/PDF，位于 `fig/`）。该脚本读取已完成输出，不重新校准。`matched_B_*` 表使用 B 的对应选择版本；原 `versions_*` 表在两版中均使用最低训练 MAE 的 B，比较口径不同。近最优候选诊断写入 `joint_near_optimal_diagnostics.csv`；范围不代表置信区间。
