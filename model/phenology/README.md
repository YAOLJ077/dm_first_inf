# 物候模型主版本

当前主版本按纬度 32°N 分区：北方和南方均使用独立参数的 Force-only_Tb0。广西 5 个病害 site-year 使用实测物候日期，不参与区域参数校准和交叉验证。

## 固定配置

- 北方：[`config_north.json`](config_north.json)，从当年 1 月 1 日开始积温。
- 南方：[`config_south.json`](config_south.json)，从当年 1 月 1 日开始积温。
- [`config.json`](config.json) 是北方配置的默认兼容入口；区域模拟应根据纬度显式选择 `config_north.json` 或 `config_south.json`。
- 病害物候调节配置 [`susceptibility_modulation.json`](susceptibility_modulation.json) 已自动按纬度选择上述两个固定配置。

三个阶段按 BBCH09 → 65 → 81 顺序计算，每个阶段达到阈值后清零，下一阶段从次日重新累积。

## 主校准与验证

历史物候校准入口为 [`src/calibrate_historical_phenology.py`](../../src/calibrate_historical_phenology.py)。它按完整 site-year 分组，北方和南方分别进行 5 折交叉验证，支持 4 进程和进度监控。默认输入位于 `data/phenology_data/historical_data/`，结果保存在 `result/historical_calibration_phenology/`。

结果目录中的 `cv_summary_by_stage.csv` 和 `cv_predictions.csv` 是留出 site-year 的验证结果，`final_config_north.json` 与 `final_config_south.json` 是使用全部有效校准数据重新拟合的参数。
