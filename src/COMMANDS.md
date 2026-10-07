# Model Command Lines

Run the following commands from the repository root:

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
same training site-years, and both FSIM and FSIM-S are evaluated on the held-

Pressure-enabled evaluations use the observed-zero weather-cohort gate: a
numeric zero-spore collection interval suppresses weather-generated cohorts;
blank observations remain missing and do not activate the gate.

```bash
PYTHONPATH=. python src/calibrate_spore_pressure_s.py \
  --s-config model/FSIM/config.json \
  --spore-config model/FSIM-S/config.json \
  --workers 4 \
  --coarse-passes 1 \
  --fine-passes 1 \
  --cv-scheme site_year \
  --output-dir result/cv_spore_pressure_s \
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
  --coarse-passes 1 \
  --fine-passes 1 \
  --cv-scheme site_year site year \
  --output-dir result/cv_spore_pressure_s_all_cv \
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
