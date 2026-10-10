# Grapevine Downy Mildew Symptom-Onset Modelling

This is a research-oriented repository for modelling grapevine downy mildew in Guangxi and other subtropical production environments. The project focuses on a specific question: given hourly weather, host phenology and spore-monitoring information, when is the first visible symptom window likely to occur during a growing season?

The models represent disease development as a sequence of processes. Weather conditions can generate sporangial cohorts, which survive, disperse with wind, deposit under suitable rainfall, release zoospores, infect leaves, incubate and produce visible symptoms. The repository includes a weather-driven baseline model and an extension that adds field-based spore pressure.

## Intended Uses

The repository is intended for:

- studying the timing and environmental drivers of first visible downy mildew symptoms in Guangxi;
- comparing the effects of weather, airborne spore pressure and host phenology;
- testing alternative assumptions for sporulation, dispersal, deposition and infection with hourly weather data;
- comparing Table and Wine grape phenology scenarios using regional Guangxi weather;
- conducting parameter sensitivity analysis, cross-validation and research data analysis.

This is a research model, not a real-time diagnostic service or a substitute for local disease scouting, cultivar trials or plant-protection decisions. The current parameters were calibrated primarily with six Guangxi site-years. Applying the model to another region, cultivar or production system requires checking the weather contract, phenology assumptions and parameter values.

## Repository Contents

### `data/`

This directory contains the source and prepared data used by the models:

- `disease_data/`: field disease observations and first symptom dates;
- `weather_data/`: hourly weather files for the site-years and selected autumn-winter weather series;
- `spore_data/`: field airborne-sporangia monitoring data and observation notes;
- `phenology_data/`: phenology observations, calibration targets and daily temperatures;
- `management_data/`: management records used for potential spray-savings analysis;
- `shapefile/`: Guangxi boundaries and regional grid files used for spatial analysis.

Weather timestamps use Beijing time (`Asia/Shanghai`). In the spore-monitoring data, a blank value means missing information, while a numeric zero means that zero target sporangia were observed. FSIM-S treats these cases differently.

### `model/`

This directory contains the model implementations:

- `model/FSIM/`: the baseline FSIM process model, from weather-triggered sporangial cohorts to the first symptom window;
- `model/FSIM-S/`: the FSIM-S extension, which adds observed spore pressure and numeric zero-count gating;
- `model/phenology/`: separate Table and Wine phenology profiles and host-susceptibility modulation across developmental stages.

FSIM and FSIM-S share the same weather drivers and post-infection processes. FSIM-S adds an observation-derived inoculum pressure, allowing weather-only and weather-plus-observation scenarios to be compared within the same process framework.

### `src/` and `run/`

`run/` contains direct entry points for individual site-year simulations. `src/` contains data preparation, phenology calibration, model calibration, regional simulation, sensitivity analysis and plotting-data scripts. Public command examples are collected in [`src/COMMANDS.md`](src/COMMANDS.md).

## Basic Usage

Run commands from the repository root. A single included weather file can be passed to the baseline FSIM route:

```bash
python run/run_sporangia.py \
  --weather data/weather_data/site_year/23_83_108_1_2022.json \
  --config model/FSIM/config.json \
  --output-dir /tmp/fsim_example
```

FSIM-S additionally requires the spore-monitoring workbook and must be enabled explicitly:

```bash
PYTHONPATH=. python run/run_spore_plus.py \
  --weather data/weather_data/site_year/23_83_108_1_2022.json \
  --spore-file data/spore_data/guangxi_site_years_spore_data.xlsx \
  --enable \
  --output /tmp/fsim_s_example.json
```

New analyses can use the portable configurations under `model/`, or pass a separately calibrated configuration through the command line. `result/` and `fig/` are local destinations for generated analysis products and figures; they are not required inputs for the core model packages.

## Scope and Interpretation

The model predicts a first complete symptom window. It does not predict lesion counts, epidemic severity or yield loss. Regional scenarios cover 342 grid cells from 2000 to 2024 and assume either Table or Wine grapes throughout the region for comparison. They are scenario analyses, not a map of actual vineyard distribution in Guangxi.

Rainfall thresholds, wind thresholds, infection degree-hours and phenology parameters represent the current study design. They are useful for sensitivity analysis and methodological comparison, but should not be treated as universal biological constants without local validation.
