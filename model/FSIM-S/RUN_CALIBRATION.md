# FSIM-S evaluation

The current implementation is a nested retrospective extension, not an
operational forecast calibration. Keep the six disease-onset site-years as
the independent units. The corrected CK workbook contains observations for
all six site-years, including Guilin 2022. Construct the pressure field from
the training-period observations and vary only `pressure_adjustment.beta_sp` in
leave-one-site-year-out evaluation; a held-out site-year must not contribute
to its training pressure field.

The deployment configuration uses an interval-rate field with measured-period
`zscore` normalization, three-day smoothing, reference level 0.5, and
`beta_sp=0.005`. It also enables the observed-zero weather-cohort gate, which
blocks weather-generated cohorts on dates covered by numeric zero-spore
intervals; blank observations remain missing and do not activate the gate. The first
collection interval is `(t_first - 3 days, t_first]`; numeric zeroes are valid
observations and blank cells are missing results.

The command-line smoke entry point is:

```bash
PYTHONPATH=. python run/run_spore_plus.py \
  --weather data/weather_data/site_year/23_83_108_1_2022.json \
  --spore-file data/spore_data/guangxi_site_years_spore_data.xlsx \
  --enable --pressure-config result/deployment_parameters/fsim_s_deployment_config.json \
  --output result/FSIM-S/23_83_108_1_2022.json
```
