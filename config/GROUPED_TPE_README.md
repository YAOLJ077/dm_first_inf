# 四参数TPE：地点/年份整体留出，单种子1024次（含128次初始探索）

脚本：`src/calibrate_m0_grouped_tpe.py`；配置：`config/fsim_m0_grouped_tpe_4p.json`。
每折总计1024次：128次初始探索（包含1个已有同训练折grid参考点）+896次TPE引导试验。
单种子20260914；六个留出折并发推进，共用一个32进程模型池。每折内部保持顺序ask/tell，带tqdm和确定性重放恢复。

| 参数 | 范围 | 类型 |
|---|---|---|
| VPD_w | 1–8 hPa | 连续 |
| I_req | 40–450 °C h | 连续 |
| RH_sp | 75–95% | 连续 |
| MP_sp | 3–7 h | 整数，步长1 |

使用之前四参数TPE运行的独立模型快照 `src/tpe_m0_reference/`；不修改现有M0/FSIM-S模型代码。
启动核对快照、输入、依赖版本及四参数范围。历史结果中的Mac绝对输入路径按项目内相对路径映射到当前项目目录后逐文件核对SHA-256；不修改历史记录，也不跳过输入一致性检查。新运行的输入和源码指纹使用项目相对路径，便于在项目整体搬迁后恢复。地点DA/GL/NN各一折，年份2022/2023/2024各一折。
按地点划分训练site-year数3/5/4，按年份均为4；每种方案六个site-year各留出一次。

训练目标：全部训练site-year可预测条件下的区间边界MAE，分母为当前训练集数量。
无预测组合为不可行；其真实MAE为null，optimizer_value的占位值不用于误差解释。
模型异常或重复天气策略产生不同预测会停止。并列规则沿用四参数TPE：优先MP接近3，
再RH接近80，随后按RH、VPD、I、seed、trial顺序。选择冻结后才评估完整留出组。
参考点来自 `result/fsim_m0_grouped_grid/<scheme>/folds/<group>/selection.json`，
严格核对相同训练/测试成员，再加RH=80、MP=3重新评估；不使用LOSYO折或全数据最优参数初始化。

## 命令

在Linux服务器上先进入项目并激活Linux本地创建的Python环境（不能使用从Mac复制的`dm_first_os`）；历史四参数运行要求Python 3.12.7，以及optuna 4.7.0、numpy 2.4.0、pandas 2.3.3、tqdm 4.67.1、scipy 1.16.3。地点与年份共六折同时推进，共用32进程（不是每折各开32进程）：

```bash
cd /home/dell/ZG_Team/yaolinjia/dm_first_inf
source .venv-linux/bin/activate
python src/calibrate_m0_grouped_tpe.py --workers 32 --scheme both --design-only
python src/calibrate_m0_grouped_tpe.py --workers 32 --scheme both --output-dir result/fsim_m0_grouped_tpe_4p_linux
```

仅地点：`--scheme site`；仅年份：`--scheme year`，建议使用不同输出目录。
只做设计/数据核对：加 `--design-only`，不运行模型、不创建结果目录。
中断后在原命令末尾加 `--resume`；配置、源码、输入、分组和软件版本必须保持一致。
输出目录独占锁防止同时写入。完成试验原子保存；未完成试验在恢复时重算。

每个方案3072次参数试验，12288次主要训练评估，4096次天气核查，最多8次留出及其核查，
共16392次模型评估；两个方案共32784次，不包含已有grid参考搜索的历史计算成本。
已保存试验恢复时不重复计算；中断时未完成试验可能额外重算。

## 输出

`site/`、`year/`分别保存fold_summary.csv、oof_predictions.csv、oof_summary.json、
test_group_metrics.csv、comparison_with_grouped_grid.csv。
每折保存训练试验、128/256/512/1024节点收敛、并列与近优集合、选择、运行配置、留出预测。
原始trial JSON保存参数、训练成员、预测与天气核查；根目录保存实际设置、split_design、指纹和源码快照。

汇总同时报告六个site-year等权MAE和三个留出组等权MAE。完整误差要求六个均可预测，否则null并另列条件误差。
失败折保留全部测试site-year，不删除；group数3与site-year数6分开报告。
配对差异delta_absolute_error_days>0表示相对同分组两参数grid恶化。

与两参数grid对照时，参数数量、搜索算法、I上界及预算不同；本次比较不是单一因素实验。
使用同训练折grid参考点也意味着利用已有搜索成果，1024次是新增TPE试验预算。
已完成设计核对、四维采样、分组训练/测试隔离、恢复及四进程合成数据检查；未启动正式校准。

并行结构：每折由独立协调线程运行自己的TPE study，所有模型计算提交至唯一的32进程池。
六折同时进行时，每轮合计最多32个训练/天气核查任务可排队，能够让32个进程同时执行模型任务；实际速度还受CPU、内存和存储吞吐限制。
协调线程不共享参数搜索历史或训练损失；每个study仍使用原种子和顺序ask/tell。
进度条固定位置显示scheme/group，最多六条；汇总由主线程写入，防止元数据并发覆盖。
当部分折已完成、保存结果或搜索器安排下一次试验时，实际占用可能下降，不保证全程CPU满载。
单独运行site或year时合计最多16个训练/天气任务可同时排队，因此32进程不保证在单方案运行时全部忙碌。

验证包括：原16进程运行中16个独立worker PID同时就绪；并发与顺序执行的六折合成试验参数及评分逐条相同；
并发恢复不重算已完成试验；训练/测试隔离、四参数采样及无预测约束检查。

本次32进程变更改变了源码与设置指纹；旧16进程运行不能加--resume继续，必须先停止旧进程，并在清理或移走旧输出目录后重新开始。
新调度运行中断后，可用完全相同命令加--resume恢复。正式校准尚未启动。
