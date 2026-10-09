# Phenology data

- `phenology.xlsx`: original local field observations used as phenological
  anchors in the six Guangxi disease-model site-years.
- `calibration_data/`: local Table/Wine profile calibration records,
  censored stage targets and the required daily temperatures.

The active calibration script is `src/calibrate_phenology_profiles.py`;
the accepted profile configurations are in `model/phenology/` and their
results are in `result/phenology_calibration/`.
