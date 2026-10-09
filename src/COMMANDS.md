# Model Command Lines

Run the following commands from the repository root:

## Current script layout

Reusable preparation, calibration, regional simulation and plotting scripts
are kept in `src/`. A `tmp/` directory may be created as needed for transient
downloads and scratch outputs; no analysis entry points belong there.

## Phenology profile calibration

```bash
python src/calibrate_phenology_profiles.py --overwrite
```

The calibration uses the local Table/Wine observations and retained daily
temperatures in `data/phenology_data/calibration_data/`. Literature records
and merged historical weather archives are not inputs. Results go to
`result/phenology_calibration/`. The accepted runtime profiles remain in
`model/phenology/profile_config_{table,vine}.json`; recalibration writes
results without automatically replacing those configurations.

## Guangxi Table/Vine regional scenarios

```bash
python src/run_guangxi_profiles.py --workers 16 --overwrite
```

The command runs the all-Table scenario and then the all-Vine scenario for
342 grids and 2000-2024. Both use the same hourly weather archive at
`D:\yaolinjia\Grape\input_weather_json\guangxi`. The calibrated profile
configurations in `model/phenology/` supply each scenario's start DOY, base
temperature, and incremental BBCH09/65/81 requirements. BBCH53 retains the
operational susceptibility configuration's 30% forcing fallback. The regional
runner and calibration diagnostic both read this fraction from
`model/phenology/susceptibility_modulation.json`.

Daily temperature and phenology JSON files go under
`D:\yaolinjia\Grape\guangxi_phenology\{table,vine}\`; first symptom window
JSON files go under `D:\yaolinjia\Grape\guangxi_disease_window\{table,vine}\`.
Phenology and onset CSVs retain the existing
`result/guangxi_2000_2024_onset_windows/` directory; savings CSVs retain
`result/guangxi_2000_2024_fungicides/`. New CSVs and summaries have `_table`
or `_vine` suffixes. Savings use the existing 2022 Duan spray schedule and
the seven days before the first symptom. Superseded unlabelled onset and
savings CSVs and summaries have been removed from `result/`.
Omit `--overwrite` to reuse existing simulation
JSON files after an interrupted run with the same inputs and configuration.

## Guangxi annual maps: Table/Wine

```bash
python src/plot_table_vine_annual_maps.py
```

This command generates only four PNG figures in `fig/guangxi_2000_2024/`:
annual first symptom dates (`01_annual_first_symptom_maps_{table,wine}.png`)
and annual potential spray savings (`02_annual_spray_savings_maps_{table,wine}.png`).
Each figure contains 25 annual panels for 2000-2024. Wine labels use the
existing `_vine.csv` source data. Missing display cells are filled from the
nearest valid source grid and clipped to the province boundary; source data
are unchanged.

## Guangxi spring weather background

```bash
python src/plot_guangxi_spring_weather_background.py
```

The command writes `fig/figure_12.png`
at 180 mm width and 600 dpi, with Times New Roman text. The upper panels
show 2000-2024 grid means of March-May temperature, precipitation and
nighttime high-humidity hours (RH >=85%, 20:00-07:59 local time). The lower
panels show equally weighted regional grid means by year and spatial
25th-75th percentiles, not confidence intervals. Dashed lines show ordinary
least-squares trends, with changes per decade annotated in each panel.
Boundary padding is only
used for map display. `--width-mm` adjusts the width to a journal's specification.

Weather summaries in `result/guangxi_2000_2024_weather/` retain only annual
and fixed spring statistics. `src/build_guangxi_weather_statistics.py` no
longer reads phenology files or generates the obsolete BBCH09-65 statistics.

The weather-archive preparation script is `src/build_guangxi_weather_json.py`.
Earth Engine export scripts are `src/gee_era5_hourly_guangxi_grid.js` and
`src/gee_era5_hourly_site_year_weather.js`. Exported CSV batches may be staged
in `tmp/`, or supplied with the weather builder's `--gee-dir` option.
`src/plot_guangxi_weather_trends.py` retains the separate annual-weather plot.

## Manuscript figures and observation audit

```bash
python src/plot_figure03_autumn_winter.py
python src/plot_figure08_observation_timeline.py
python src/plot_sobol_sensitivity.py
python src/plot_fsim_fsim_s_error_metrics.py
python src/plot_graphical_abstract.py
python src/plot_table_vine_summary.py
python src/plot_data_audit.py
```

The error plots read `result/fsim_calibration/` for FSIM and
`result/fsim_s_calibration/` for FSIM-S. The Table/Wine summary script writes
`fig/figure_13.png` and `fig/figure_14.png` for manuscript Figures 13 and 14.
All manuscript figures use `fig/figure_01.png` through `fig/figure_14.png`,
matching their order in the manuscript. See `fig/README.md` for the index.
Figure 1 and the conceptual framework also have editable source artwork
under `fig/`.

## Sobol sensitivity: FSIM/S

Run several Sobol base sample sizes sequentially. The script creates one
output directory per sample size under `result/sobol_sensitivity/`.

```bash
PYTHONPATH=. python src/sobol_sensitivity.py \
  --route S \
  --sporangia-config model/FSIM/config.json \
  --N 256 512 1024 \
  --workers 4 \
  --seed 20261006 \
  --resamples 100 \
  --confidence 0.95
```

`tqdm` progress monitoring is enabled by default. The three runs are executed
in sequence and write to `result/sobol_sensitivity/S_N256`,
`result/sobol_sensitivity/S_N512`, and `result/sobol_sensitivity/S_N1024`.

After a failed run, restart the command with only unfinished `--N` values.
For example, if N=256 completed, use `--N 512 1024` and retain the same seed
and other settings. The script does not resume partial samples within an N.
Restart processes launched before the model-directory rename so they load
the current imports and configuration path.

For a quick smoke test using the six site-years with a small Sobol design:

```bash
PYTHONPATH=. python src/sobol_sensitivity.py \
  --route S \
  --sporangia-config model/FSIM/config.json \
  --N 2 \
  --workers 4 \
  --resamples 2 \
  --no-second-order \
  --seed 20261006 \
  --output-dir /tmp/sobol_s_smoke
```

The current script supports only the FSIM/S route (`--route S`). Its inputs are
the six Guangxi weather seasons, disease-onset workbook, observed phenology
workbook, and `model/FSIM/config.json`.

## Nested FSIM-S site-year calibration

Run six primary outer site-year leave-one-out folds. Within each fold, three
open FSIM parameters (`infection_degree_hours`,
`effective_deposition_min_rain_mm`, and `sporulation_rh_min_pct`) are fitted on
the five training site-years with the prespecified Cartesian grid.
`wind_min_m_s=0.5`, `heavy_rain_washoff_mm=5.0`, and
`minimum_moist_hours=7.0` remain fixed. The
selected FSIM configuration is then fixed while `beta_sp` is fitted on the
same training site-years, and both FSIM and FSIM-S are evaluated on the held-out
site-year.

Pressure-enabled evaluations use the observed-zero weather-cohort gate: a
numeric zero-spore collection interval suppresses weather-generated cohorts;
blank observations remain missing and do not activate the gate.

```bash
PYTHONPATH=. python src/calibrate_spore_pressure_s.py \
  --s-config model/FSIM/config.json \
  --spore-config model/FSIM-S/config.json \
  --workers 4 \
  --beta-fine-passes 1 \
  --cv-scheme site_year \
  --output-dir result/fsim_s_calibration \
  --overwrite
```

Optional grouped stress tests can be run in the same command. `site` holds out
all site-years from one site, while `year` holds out all site-years from one
calendar year. The output tables include a `cv_scheme` column so these results
remain separate from the primary site-year evaluation:

```bash
PYTHONPATH=. python src/calibrate_spore_pressure_s.py \
  --s-config model/FSIM/config.json \
  --spore-config model/FSIM-S/config.json \
  --workers 4 \
  --beta-fine-passes 1 \
  --cv-scheme site_year site year \
  --output-dir result/fsim_grouped_validation \
  --overwrite
```

The output contains fold-specific S parameters and `beta_sp`, held-out
predictions, and aggregate MAE, RMSE, bias, and symptom-window coverage for
both `FSIM` and `FSIM-S`. Candidate selection uses the minimum training
penalized MAE, then the normalized median of candidates within
`--mae-tolerance 0.5` days of that minimum; configuration defaults are
recorded as provenance only and do not guide calibration. `tqdm` progress
monitoring is enabled by default.

## Deployment parameter selection

After the accepted site-year fold results are available, select one global
deployment configuration by retaining parameters fixed across folds, taking the
unique fold mode for variable parameters, and using the median only when the
mode is tied:

```bash
PYTHONPATH=. python src/select_deployment_parameters.py --overwrite
```

The command writes `deployment_parameters.json`, a fold-level selection audit,
and runnable FSIM/FSIM-S configuration files under
`result/deployment_parameters/`.
