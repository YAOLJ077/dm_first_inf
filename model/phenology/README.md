# 物候模型主版本

当前主版本按纬度 32°N 分区：北方和南方均使用独立参数的 Force-only_Tb0。广西 5 个病害 site-year 使用实测物候日期，不参与区域参数校准和交叉验证。

## 固定配置

- 北方：[`config_north.json`](config_north.json)，从当年 1 月 1 日开始积温。
- 南方：[`config_south.json`](config_south.json)，从当年 1 月 1 日开始积温。
- [`config.json`](config.json) 是北方配置的默认兼容入口；区域模拟应根据纬度显式选择 `config_north.json` 或 `config_south.json`。
- 病害物候调节配置 [`susceptibility_modulation.json`](susceptibility_modulation.json) 已自动按纬度选择上述两个固定配置。

三个阶段按 BBCH09 → 65 → 81 顺序计算，每个阶段达到阈值后清零，下一阶段从次日重新累积。

## Table/Vine 分层校准配置

当前用于广西分层物候模拟的最终配置为 [`profile_config_table.json`](profile_config_table.json) 和 [`profile_config_vine.json`](profile_config_vine.json)，索引文件为 [`profile_configs.json`](profile_configs.json)。它们使用北京时间日平均气温、线性 GDD 响应和分阶段增量阈值；Guilin site-year 不再参与该配置。可复现校准脚本为 [`calibrate_table_vine_no_guilin.py`](calibrate_table_vine_no_guilin.py)，拟合核心为 [`profile_calibration_core.py`](profile_calibration_core.py)；候选基温上限固定为 10 °C，并以留一site-year验证选择最终模型。

## 主校准与验证

区域 BBCH53 是辅助阶段：从 BBCH09 次日开始，积累达到 BBCH09→65 增量积温阈值的 30% 时判定。比例统一由 `susceptibility_modulation.json` 的 `auxiliary_stage_rules.53.forcing_fraction` 控制，区域模拟与校准辅助计算均读取该配置。BBCH09、65、81 的已校准阈值不受此规则影响。

历史物候校准入口为 [`src/calibrate_historical_phenology.py`](../../src/calibrate_historical_phenology.py)。它按完整 site-year 分组，北方和南方分别进行 5 折交叉验证，支持 4 进程和进度监控。默认输入位于 `data/phenology_data/historical_data/`，结果保存在 `result/historical_calibration_phenology/`。

结果目录中的 `cv_summary_by_stage.csv` 和 `cv_predictions.csv` 是留出 site-year 的验证结果，`final_config_north.json` 与 `final_config_south.json` 是使用全部有效校准数据重新拟合的参数。
