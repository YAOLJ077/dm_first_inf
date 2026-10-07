# Historical phenology calibration inputs

These files were migrated from the curated phenology calibration workflow in
the companion repository. The observations cover 2010--2024 and are filtered
by the calibration script to BBCH09, BBCH65, and BBCH81 anchor events.

- `phenology_model_observations.csv`: curated phenology observations.
- `phenology_model_site_years.csv`: site-year metadata and weather-file index.
- `phenology_weather_hourly.csv`: hourly weather used by the calibration.

The five Guangxi disease site-years used as observed phenology anchors in the
main study are excluded by default in `src/calibrate_historical_phenology.py`.
