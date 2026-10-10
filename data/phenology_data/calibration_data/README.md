# Local phenology calibration inputs

These files contain the local Nanning Table and Hechi Wine observations used
by the profile calibration, together with the corresponding daily weather.

- `phenology_calibration_records.csv`: local records derived from
  `../phenology.xlsx`.
- `phenology_calibration_targets.csv`: BBCH09, BBCH53, BBCH65 and BBCH81
  targets with explicit exact, interval- and one-sided censoring rules.
- `site_year_temperature_daily.csv`: 1,827 target-year daily temperatures
  for the five calibrated site-years, grouped in Asia/Shanghai time, with
  24 hourly values per day.

The calibration entry point is `src/calibrate_phenology_profiles.py`.
The six-site-year observed workbook remains at
`data/phenology_data/phenology.xlsx` for disease-model evaluation.
