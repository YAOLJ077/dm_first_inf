# 版本与参数配置登记

统一 site/year 校准对比入口：`src/calibrate_fsim_unified.py`，支持两/四参数 × grid/TPE、I_req=40–450、缓存续跑和配对表。默认先 Grid 多候选并行、再 TPE 每折顺序更新，全局模型进程上限 16；TPE 每折 1,024 次（含 128 次启动采样）。设计及 Linux 指令见 [UNIFIED_CALIBRATION_README.md](UNIFIED_CALIBRATION_README.md)。此入口使用冻结 M0/FSIM 快照，不包含 FSIM-S 校准。

`project_registry.json`是版本及参数元数据清单。它登记当前入口、旧版脚本、延期组件和参数分类，**不是新的运行时参数文件**。当前模型通过`model/parameter_table.py`读取`config/fsim_runtime_parameters.csv`；`model/config.py`保留旧调用兼容入口。

“固定”按项目讨论解释为“本轮采用的默认设置”，不自动等同于不可修改。登记使用以下分类：

- `open_candidate`：开放候选项；登记不执行筛选或拟合。
- `literature_attributed_default`：项目已注明文献来源的默认关系；本轮未完成系数逐项原文核验。
- `definition_default`：定义性的默认设置。潜育完成判据登记为1，与现有共享常量一致。
- `study_or_computational_convention`：研究或计算约定，不称为试验常数。

参数含义继续维护在`data/fsim_parameters.csv`的原有五列中。本清单仅用缩写关联，不复制完整数值表，避免形成互相冲突的运行时默认值。运行数值以新增CSV表为准，五列表仍承担中文解释与来源性质说明。登记的定义默认值仅作记录，模型不会读取本文件覆盖参数。

## 历史版本的标记范围

21个已确认的旧脚本已在源码开头添加`LEGACY / 旧版`标记，并在清单中列出。包括旧分析入口、其结果汇总和制图脚本，以及相关历史探索脚本。

本次选择用户允许的“标记”方式：文件没有移动，原有导入路径与执行行为保留；标记不是执行拦截。这些脚本也不是可直接重现历史结果的完整快照，因为其导入的模型模块可能已经更新。旧参数与旧结果不能仅凭文件仍存在就视为适用于当前版本。

FSIM-S核心被列为`deferred_components`，而不是删除或判定废弃。与当前FSIM任务无关的物候、气象及论文绘图工具没有仅凭文件年龄被标为旧版。

当前入口登记只表示版本身份，不表示相关分析已经运行、验证通过或适合直接用于最终结论。

## 本次检查

对21个标记文件逐一比较修改前后的Python语法树，确认仅增加注释。对登记文件检查JSON格式、路径存在性、参数缩写完整性及记录的潜育完成值与共享常量的一致性。未运行模型、敏感性分析或校准。

## 新运行时参数表与分析（2026-09-13）

`fsim_runtime_parameters.csv`是当前FSIM运行值与分析空间的来源。列含参数缩写、配置键、JSON值、角色、版本作用域、单位、上下界、网格点数和说明。系数组采用JSON单元格，避免拆散已固定的文献关系。

- M0读取共同候选VPD_w和I_req；M1增加u_tr。版本开关独立于候选参数。
- `model.config.config`在导入时读默认表；`load_config(path)`读取一份新配置；`CropSeasonFirst(rb)`省略配置时自动读默认表。
- 固定方程与时段常量在进程启动时从默认表加载；修改默认表后必须重启Python。替代表只允许修改候选默认值/范围及结构开关，不能偷偷改变固定关系。
- `BBCH_min=15`和`t_init=01-01`是受支持的计算约定，加载器拒绝其他值。物候日期由输入提供；完整年天气起点由新分析脚本校验。通用核心入口仍允许短时间序列用于诊断测试。
- 原有候选默认值保留为历史对照，未改写为本网格最优值。固定项的来源分类不等同于已经逐项核验原文。
- 表内两个legacy雨水键只为延期的旧FSIM-S入口保留；本次没有宣称FSIM-S全链同步。

运行新分析（结果目录须为空，避免覆盖既有结果）：

```bash
dm_first_os/bin/python src/analyze_fsim_parameters.py --output-dir result/fsim_parameter_analysis_next
dm_first_os/bin/python src/report_fsim_parameter_analysis.py --result-dir result/fsim_parameter_analysis_next
```

`--config-table`允许显式指定替代表；默认网格由表内`grid_levels`读取。`--workers`控制并发，`--grid-levels 3`可用于小网格检查。小网格不能冒充本轮完整设计。

已完成结果位于`result/fsim_parameter_analysis_v1/analysis_report.md`：630个网格组合、12次站点年份默认值对照；另对两个2024年输入中的重复年末天气记录进行了1264次策略对照，首次显症输出无差异。原始输入未修改。

`result/fsim_parameter_analysis/`保存最初被输入检查拦截的记录，不是有效分析结果。新脚本的网格差分敏感性不是旧Morris结果，也不是LOSYO证据。

## M0 LOSYO 校准

`src/calibrate_m0_losyo.py` 使用 `config/fsim_m0_losyo.json`，每折仅以五个训练站点的区间距离MAE校准VPD_w和I_req，要求训练站点全部产生预测，采用粗网格和两轮局部加密。支持2/4进程、tqdm、跨折确定性计算缓存，输出逐折损失面、并列/近优参数集合及六个留出结果。详细设置、命令和缺失处理见 `config/LOSYO_README.md`。已完成六折小规模实现检查，正式校准尚未运行。

## I_req 上界扩展试验

`config/fsim_Ireq200_exploration.csv`是用户指定的独立探索配置：I_req上界为200，其他范围和所有默认值保持原样。可选列`grid_values_json`指定有序、唯一、覆盖上下界的显式网格；未提供时继续按`grid_levels`等距生成。显式网格保留原45–90的全部7点，并增加100–200、间隔10的11点，避免扩展范围时丢失原基准点。`--grid-levels`仍可覆盖显式网格用于烟雾测试。

```bash
dm_first_os/bin/python src/analyze_fsim_parameters.py --config-table config/fsim_Ireq200_exploration.csv --output-dir result/fsim_Ireq200_exploration
dm_first_os/bin/python src/report_fsim_parameter_analysis.py --result-dir result/fsim_Ireq200_exploration
dm_first_os/bin/python src/compare_fsim_Ireq_expansion.py --baseline result/fsim_parameter_analysis_v1 --expanded result/fsim_Ireq200_exploration
```

比较脚本核对旧网格预测复现，生成保持其他参数不变的扩展比较、逐站点年份最优网格配对误差和嵌套上界诊断。该探索不修改主运行表的默认值，也不构成校准或LOSYO验证。范围归一化的敏感性值不宜直接跨两轮比较，因为I_req范围宽度和网格分布改变。

条件分辨率检查：`dm_first_os/bin/python src/refine_fsim_Ireq_profile.py`固定扩展网格最优组合的其他参数，测试I_req=180–200、步长1。结果保存在扩展目录的`conditional_refinement/`，该子目录存在时拒绝覆盖。随后重跑比较脚本可将细化检查纳入`Ireq_expansion_report.md`。这不是联合细网格或正式校准。

## Sobol 专用入口

新的敏感性入口为`src/sobol_fsim.py`，专用参数表为`fsim_sobol_parameters.csv`，运行设置为`fsim_sobol_analysis.json`。参数表允许不提供网格列；Sobol只读取连续范围，不读取既往网格结果。详见[SOBOL_README.md](SOBOL_README.md)。本轮主响应为RMSE，补充MAE和预测产生率，不进行寻优。
