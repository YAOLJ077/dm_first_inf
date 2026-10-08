# Phenology calibration data

These are the prepared inputs for the final Table/Vine profile calibration.

- `phenology_calibration_records.csv` retains the merged historical and
  `phenology.xlsx` observations.
- `phenology_calibration_targets.csv` contains BBCH09, BBCH53, BBCH65 and
  BBCH81 targets. Exact dates, interval censoring and right/left censoring are
  explicit. For a right-censored BBCH09 target, `calibration_lower` is a
  conservative 14-day lower bound used only to prevent implausibly early
  solutions; `observed_start` remains empty.
- `phenology_weather_hourly_bjt.csv` converts historical UTC timestamps to
  `Asia/Shanghai` before grouping. The repository JSON weather for Guilin
  2022 is included because it is the only measured workbook site-year absent
  from the historical ERA5 index.
- `phenology_weather_daily_bjt.csv` is the 24-hour Beijing-time daily mean
  table used by the calibration scripts.
- `weather_support_audit.csv` confirms complete January--December target-year
  coverage for every retained site-year.

The final entry point is `model/phenology/calibrate_table_vine_no_guilin.py`.
It uses the validated Table/Vine profiles and excludes Guilin from the target
and calibration records.
