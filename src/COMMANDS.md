# Public workflow commands

Run commands from the repository root. The six site-year weather files and
the field workbooks are included under `data/`. The larger 342-grid regional
weather archive is an external input; pass its location explicitly when
running regional simulations.

## Site-year disease simulation

Run the weather-driven FSIM route for one or more included weather files:

```bash
python run/run_sporangia.py \
  --weather data/weather_data/site_year/23_83_108_1_2022.json \
  --config model/FSIM/config.json \
  --output-dir result/FSIM/example
```

Run the observation-informed FSIM-S route:

```bash
PYTHONPATH=. python run/run_spore_plus.py \
  --weather data/weather_data/site_year/23_83_108_1_2022.json \
  --spore-file data/spore_data/guangxi_site_years_spore_data.xlsx \
  --enable \
  --output result/FSIM-S/example.json
```

The model configuration files under `model/` are the portable source
configurations. A separately calibrated deployment configuration can be passed
with `--config` when available.

## Phenology calibration

The local Table/Vine phenology calibration uses the inputs under
`data/phenology_data/calibration_data/`:

```bash
python src/calibrate_phenology_profiles.py --overwrite
```

The accepted runtime profiles are kept in `model/phenology/`.

## FSIM-S parameter analysis

The nested site-year evaluation and pressure-parameter analysis can be run
with:

```bash
PYTHONPATH=. python src/calibrate_spore_pressure_s.py \
  --s-config model/FSIM/config.json \
  --spore-config model/FSIM-S/config.json \
  --weather-dir data/weather_data/site_year \
  --spore-file data/spore_data/guangxi_site_years_spore_data.xlsx \
  --cv-scheme site_year \
  --output-dir result/fsim_s_calibration \
  --overwrite
```

This evaluates six held-out site-year groups. The observations from a held-out
group are excluded from its training pressure field.

## Regional scenarios

Regional runs require an hourly weather archive with one directory per year
and 342 grid files per year. The archive is not part of this repository. The
preparation script accepts external source files and an optional Earth Engine
export directory:

```bash
python src/build_guangxi_weather_json.py \
  --source-root /path/to/weather/source \
  --target-root data/weather_data/regional \
  --gee-dir /path/to/gee_exports
```

After the archive is available, run one cultivar profile at a time. The
following commands keep generated intermediate files under `result/`:

```bash
python src/run_guangxi_phenology.py \
  --profile table \
  --weather-root data/weather_data/regional \
  --output-root result/regional_phenology

python src/run_guangxi_disease_windows.py \
  --profile table \
  --weather-root data/weather_data/regional \
  --phenology-root result/regional_phenology \
  --output-root result/regional_disease_window

python src/build_guangxi_phenology_long_table.py \
  --profile table \
  --phenology-root result/regional_phenology

python src/build_guangxi_onset_long_table.py \
  --profile table \
  --disease-root result/regional_disease_window
```

Replace `table` with `vine` for the wine-grape profile. The regional weather
statistics command uses the same archive:

```bash
python src/build_guangxi_weather_statistics.py \
  --weather-root data/weather_data/regional
```

The scripts write analysis tables and figures to the existing `result/` and
`fig/` locations. Those generated artifacts are not required to use the core
FSIM and FSIM-S model packages.
