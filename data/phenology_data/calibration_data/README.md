# Local phenology calibration inputs

Only the local Nanning Table and Hechi Wine observations used by the accepted
profile calibration are retained here. Literature observations and merged
historical weather archives have been removed.

- `phenology_calibration_records.csv`: local records derived from
  `../phenology.xlsx`.
- `phenology_calibration_targets.csv`: BBCH09, BBCH53, BBCH65 and BBCH81
  targets with explicit exact, interval- and one-sided censoring rules.
- `site_year_temperature_daily.csv`: 1,827 target-year daily temperatures
  for the five calibrated site-years, grouped in Asia/Shanghai time, with
  24 hourly values per day. These are the exact daily values used by the
  accepted calibration, extracted from the former merged table without
  rounding or replacing temperatures.

The calibration entry point is `src/calibrate_phenology_profiles.py`.
It retains the accepted Nanning BBCH09 assumptions and site selection and
writes its adjusted targets, predictions and parameter estimates to
`result/phenology_calibration/`. The six-site-year observed workbook remains
at `data/phenology_data/phenology.xlsx` for disease-model evaluation.
