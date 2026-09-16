# FSIM Sobol 敏感性分析

本入口分析参数对误差和预测产生率变异的贡献，不选择最优参数，不输出近优集合，也不执行校准。历史网格结果不作为Sobol样本。

## 配置

- `config/fsim_sobol_parameters.csv`：模型读取的专用参数表，不含网格点数。独立均匀探索范围为VPD_w 1–8 hPa、I_req 45–300 °C h、M1专属u_tr 1–8 m/s。M0没有u_tr自由参数。默认值保留历史配置，仅抽样值随运行变化。
- `config/fsim_sobol_analysis.json`：首轮设置采用两个独立scramble种子；每个模型、种子的基样本量N=128；检查N=32、64、128的嵌套前缀。计算S1、ST和S2；bootstrap 500次，95%置信水平。
- SALib的Sobol/Saltelli设计不是全因子网格。启用二阶指数时，每个模型和种子有N(2D+2)个参数组合；N不是田间独立样本数。
- 首轮设计共有3,584行参数组合，六站点年份主分析21,504次，另有7,168次重复天气记录策略复算，总计28,672次模型评估。N=32、64使用同一设计的前缀，不增加模型评估次数。
- 上述N是起始的正式诊断设置，不保证收敛。`convergence.csv`比较递增N的指数变化（预设绝对差0.05仅作诊断），`seed_stability.csv`检查独立种子差异。是否需要更大N由数值稳定性决定，不按希望得到的排名决定。

## 响应定义

主响应RMSE = sqrt(sum(e_j²)/6)，辅以MAE = sum(|e_j|)/6；e_j为预测显症窗口与观测(L,U]区间的有符号边界距离，重叠时为零。另分析六站点年份的预测产生率、各站点年份的有符号误差、绝对误差和是否产生预测。重叠率不进入本次Sobol响应。

同一个参数组合在任一站点年份无预测时，其汇总RMSE和MAE均为缺失；不能改变分母。某响应只要在整套Sobol设计中存在缺失，该响应的Sobol分解即标为不可用，不能删除行或插入惩罚值。其他定义完整的响应（如预测产生率）照常检查和分析。常量响应的指数为未定义，不是零。

小N下没有遇到无预测，不证明整个参数空间都可预测；更大N可能出现缺失，因此每个N都独立检查响应完整性。含缺失的参数空间若要定义其他误差响应或改变分析域，必须作为新的明确方法决定，脚本不自动处理。

经用户确认，现设置 `accuracy_tolerances_days: [10, 7]`：当且仅当产生预测且绝对区间误差不超过相应容许天数，站点年份准确性指标记为1，否则为0；六站点汇总为指标之和除以6。程序错误仍阻止分析。响应名为 `accurate_prediction_rate_10d` 和 `accurate_prediction_rate_7d`，不会修改原始 RMSE/MAE 或将无预测的误差补为0。容许天数仅为探索设置，不是已验证的管理标准；响应完整不保证方差非零或指数收敛。10天在前以保留已有10天响应的 bootstrap 种子。

已有运行可直接追加分析，无需重新计算模型。结果和新图分别保存在新的目录中：

```bash
dm_first_os/bin/python src/analyze_sobol_accuracy.py --tolerance-days 10 --input-dir result/fsim_sobol_n128 --output-dir result/fsim_sobol_n128_accuracy10 --figure-dir fig/fsim_sobol_n128_accuracy10
```

上述目录已有完成结果；再次运行须指定新的空目录。默认后续模型运行会读取配置中的容许误差，自动追加相同响应。原有响应列顺序及其 bootstrap 种子保留。

7天试算已经完成，独立结果目录为 `result/fsim_sobol_n128_accuracy7/`，图件为 `fig/fsim_sobol_n128_accuracy7/`。其复现命令使用 `--tolerance-days 7` 并指定这两个目录（重复运行需使用新的空目录）。

## 完全匹配的结构比较

`src/compare_fsim_paired.py --workers 4` 读取原始 M1 设计，对完全相同的 VPD_w、I_req 关闭风速门控补算 M0；只缓存完全相同的共同参数，展开后保留原始 M1 Saltelli 行序。模型代码、固定参数及输入均与原运行核验。`src/report_fsim_paired.py` 输出配对准确性、预测可用性、双方有预测时的时间差，以及准确比例差的 Sobol 指数。

已完成结果为 `result/fsim_paired_n128/`，图件为 `fig/fsim_paired_n128/`。新增 M0 主评估6,144次、重复天气策略复算2,048次，匹配到2,048个M1设计行及六站点年份。描述性主汇总只使用A/B基础行，完整Saltelli汇总作为补充；Sobol始终使用完整设计。模拟次数不是田间独立样本数。该比较隔离门控结构差异，不进行寻优，也不构成LOSYO验证。

置信区间反映指数估计的抽样不确定性，不是六站点年份的泛化性能区间。负的一阶估计、超过1的估计及数值警告保留用于诊断，不进行裁剪。

## 命令

仅生成正式设计及规模说明，不运行模型：

```bash
dm_first_os/bin/python src/sobol_fsim.py --design-only --output-dir result/fsim_sobol_design_new
```

小样本端到端检查（N=8、一个种子；不能用于解释参数排名）：

```bash
dm_first_os/bin/python src/sobol_fsim.py --smoke --output-dir result/fsim_sobol_smoke_new
```

正式运行（所指定目录须为空）：

```bash
dm_first_os/bin/python src/sobol_fsim.py --output-dir result/fsim_sobol_formal
```

命令行可选2或4个工作进程，覆盖设置文件的`workers`，并写入本次有效设置和运行元数据：

```bash
dm_first_os/bin/python src/sobol_fsim.py --workers 2 --output-dir result/fsim_sobol_formal
dm_first_os/bin/python src/sobol_fsim.py --workers 4 --output-dir result/fsim_sobol_formal
```

以上命令二选一；未指定`--workers`时使用配置值，当前为4。

运行时默认显示两个`tqdm`进度条：

- `Model evaluations`：已完成的站点年份模型评估数、百分比、耗时、速度与预计剩余时间；包括重复天气策略复算，并显示并行进程数、无预测记录数和程序错误数。按每个任务组完成后的实际记录数更新，不按任务组数量估算完成率。
- `Sobol responses`：各模型、种子、样本量和响应的分析进度；无法计算指数的响应也会完成诊断并计入进度。

进度条由主进程统一显示，不会由2或4个工作进程重复刷屏。预计剩余时间基于当前速度估计，可能因任务耗时不同而波动。日志或批处理可加`--no-progress`关闭进度条，例如：

```bash
dm_first_os/bin/python src/sobol_fsim.py --workers 2 --no-progress --output-dir result/fsim_sobol_formal
```

运行其他设置：`--settings config/自定义设置.json`。参数表路径和输入manifest路径由设置文件指定，项目内相对路径从仓库根目录解析。默认使用4个工作进程；中间过程按相同风速条件分组缓存，结果仍恢复为原始Sobol行序。源码、参数表、有效设置和输入哈希自动留存。

两个2024年输入的年末重复天气记录在分析副本中保留末次记录，并按设置对保留首次记录进行复算。若策略改变显症输出，正式指数解释被阻止；原始JSON不修改。程序异常与模型无预测分别处理。

## 主要输出

| 文件 | 内容 |
|---|---|
| `sobol_design.csv` | 完整配套抽样设计及行号 |
| `predictions.csv` | 六站点年份逐抽样预测和无预测记录 |
| `response_vectors.csv` | 严格按抽样行对齐的响应 |
| `response_diagnostics.csv` | 各响应的缺失、常量或数值问题 |
| `sobol_indices.csv` | S1、ST、S2及置信区间半宽 |
| `convergence.csv` | 随N变化的指数差异 |
| `seed_stability.csv` | 独立scramble种子间的差异 |
| `README.md` | 本次响应能否进行合法分解的摘要 |
| `run_metadata.json` | 运行状态、配置、版本和指纹 |

## 已完成结果的绘图

```bash
dm_first_os/bin/python src/plot_sobol_fsim.py --input-dir result/fsim_sobol_n128 --output-dir fig/fsim_sobol_n128
```

该入口只读取已完成的分析结果，核对评估数量、响应行号、程序错误和重复天气策略差异，生成预测可用性、汇总预测产生率敏感性、逐站点预测敏感性、可估计的绝对误差敏感性和收敛诊断图。输出为 PNG、PDF，并附中文解读、绘图数据和输入指纹。

N=128 结果中，两个模型的汇总 RMSE/MAE 均有缺失，不能绘制这些响应的 Sobol 指数。部分预测产生率的 bootstrap 半宽虽然是有限值，却达到约 1e28–1e33；原始 `status=ok` 不代表可靠区间。绘图保留这些原始估计，并对极端区间明确标注，不能按它们给出可靠排名。详见 `fig/fsim_sobol_n128/README.md`。

## 方法依据

采用SALib 1.5.2中`SALib.sample.sobol.sample`和`SALib.analyze.sobol.analyze`配套实现。

- [SALib抽样文档](https://salib.readthedocs.io/en/latest/api/SALib.sample.html)
- [SALib分析文档](https://salib.readthedocs.io/en/stable/api/SALib.analyze.html)

Hui Wang等（2022）分别描述了Sobol总效应敏感性筛查和后续误差目标校准。本项目保留二者分开的原则，但本轮明确把RMSE、MAE和预测产生率作为响应，不声称这是该文已明确报告的同一标量输出定义。
