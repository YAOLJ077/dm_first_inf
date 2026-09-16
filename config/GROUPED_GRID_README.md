# M0 两参数 grid：按地点/年份整体留出

入口 `src/calibrate_m0_grouped_grid.py`，配置 `config/fsim_m0_grouped_grid.json`。
正式校准尚未启动。数据仍为六个广西site-year，两个方案分别独立校准，均为三折。

| 方案 | 留出组 | 训练site-year数 | 测试site-year数 |
|---|---|---:|---:|
| leave-one-site-out | DA全部年份 | 3 | 3 |
| leave-one-site-out | GL全部年份 | 5 | 1 |
| leave-one-site-out | NN全部年份 | 4 | 2 |
| leave-one-year-out | 2022全部地点 | 4 | 2 |
| leave-one-year-out | 2023全部地点 | 4 | 2 |
| leave-one-year-out | 2024全部地点 | 4 | 2 |

分组使用manifest的place_code和Year字段，不从字符串猜测。每个方案中六个site-year各测试一次。
模型不把年份或地点作为新特征；分组仅决定哪些观测用于训练和测试。

## 与此前版本的可比性

对照为 `result/fsim_m0_losyo_Ireq40_400_refine3`。
使用该运行的模型与grid辅助代码快照，保存在 `src/grid_m0_reference/`，复制时按原源码哈希核对。
该目录独立于当前model目录。所需默认运行参数表取自已保存的TPE源码快照，固定值通过参数表加载器一致性校验。
原LOSYO模型和结果不修改。每次启动核对快照哈希及原LOSYO输入哈希。

- VPD_w：1–8 hPa；I_req：40–400 °C h；固定RH_sp=80%、MP_sp=3 h。
- 初始步长：VPD 0.5、I 10，共15×37=555个粗网格点。
- 三轮局部加密，I步长5、2.5、1.25；VPD步长0.25、0.125、0.0625。
- 加密所有触及训练最优MAE+1天候选的网格区域；不是仅追踪一个最优点。
- 每折候选上限35000，全域最细格32657点；实际只评估训练规则选中的局部加密点。
- 所有训练site-year均须可预测；MAE分母是该折实际训练数3、4或5，不再硬编码5。
- 目标仍是观测区间与预测窗口的边界距离绝对值均值；相交或边界距离为零时为0。
- 并列优先较低VPD，再较低I；训练选择写入selection.json后才请求整个留出组的预测。
- 不把旧LOSYO最优参数直接应用为新折参数，不使用新折测试误差选参。

每个scheme内可复用同一参数、同一site-year的确定性模型计算。缓存只减少计算，不改变该折训练集成员或损失计算。
两个scheme分别使用缓存。每个参数组合可复用于多个fold，实际计算次数不等于候选数×折数×训练数。
天气重复时间戳采用原keep_last主方案与keep_first核查；差异或程序异常会停止，不能伪装为无预测。

## 运行命令

同时运行地点和年份留出（依次运行，各自使用4进程）：

```bash
dm_first_os/bin/python src/calibrate_m0_grouped_grid.py --workers 4 --scheme both --output-dir result/fsim_m0_grouped_grid
```

只运行地点：

```bash
dm_first_os/bin/python src/calibrate_m0_grouped_grid.py --workers 4 --scheme site --output-dir result/fsim_m0_loso_grid
```

只运行年份：

```bash
dm_first_os/bin/python src/calibrate_m0_grouped_grid.py --workers 4 --scheme year --output-dir result/fsim_m0_loyo_grid
```

只检查划分、设置和输入：加 `--design-only`，不创建结果目录或运行模型。
有tqdm折进度及模型评估进度。输出目录必须为空；本grid入口不支持断点恢复，不使用TPE的--resume。

## 输出与解释

根目录：split_design.json、effective_settings.json、effective_runtime_config.json、run_metadata.json、输入与源码快照。
`site/`和`year/`各自包含：

- fold_summary.csv：三折训练数、候选数量、最优训练误差、参数和停止原因。
- oof_predictions.csv：六个site-year的独立留出预测、方向误差及失败状态。
- oof_summary.json：六个site-year等权MAE/RMSE、固定六个分母的可预测比例及7/10天准确比例。
- test_group_metrics.csv：每个留出组的测试MAE/RMSE和预测数量。
- comparison_with_LOSYO.csv：逐site-year与此前LOSYO预测的配对差异；delta_absolute_error_days>0表示本方案误差更大。
- folds/下：各阶段训练损失面、训练预测、近优与并列参数、选择及整个测试组预测。

汇总的n_folds=3，n_site_years=6。除六个site-year等权MAE，还报告equal_group_MAE_days（先算各留出组MAE，再对三个组等权）。
地点测试组大小不同，这两个平均可能不同；不要混淆。完整MAE/RMSE要求六个site-year均可预测，否则为null，另列条件误差。
没有可行训练组合时，该组所有测试site-year记为calibration_failed，不跳过组，也不计算任意参数的测试结果。

这些评价测试现有数据内的跨地点/跨年份表现。地点与品种混杂、地点年份不完整交叉仍然存在；年份留出允许使用之后年份训练，是回顾性年份迁移评价。
历史LOSYO的训练集通常更大，与新方案比较应同时报告训练规模和失败情况，不能仅凭平均误差差异归因于模型结构。
