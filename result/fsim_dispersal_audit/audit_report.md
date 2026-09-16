# FSIM dispersal and deposition audit

## Scope

This is a structural diagnostic, not a calibration or validation result. All parameters outside sporangial release and deposition were held at the current values. The six optimized site-year onset intervals and the BBCH15 output gate were unchanged.

## Unit audit

| Quantity | Code interpretation | Audit status |
|---|---|---|
| `precip_acc_period_adjusted` | Precipitation accumulated over the hourly record; thresholds therefore operate as mm per preceding hour | Consistent with the hourly Open-Meteo field definition and the values in the model inputs |
| `wind_speed` | Wind speed in m s-1 | An explicit m s-1 query to the official historical API for the same coordinate and date returned the same magnitude as the stored values; the JSON files nevertheless do not preserve the original unit request, which should be recorded when inputs are rebuilt |
| VPD | hPa-scale values | Brischetto et al. (2020) reports VPD in hPa and uses values of approximately 10 in its mortality equations; using kPa values without converting the coefficients would be incompatible, although the 2021 variable table labels VPD as kPa |

Across 52,656 hourly records, precipitation had a median of 0, a 99th percentile of 4.60, and a 99.9th percentile of 11.70 mm per hour. Wind speed had a median of 2.61, a 99th percentile of 6.58, and a 99.9th percentile of 8.33 m s-1. The current release threshold of 10.79 mm per hour was exceeded in approximately 0.14% of hours, the deposition threshold of 6.45 mm per hour in 0.58%, and the wind-release threshold of 6.42 m s-1 in 1.31%.

## Biological audit

The three current thresholds are not supported as transferable constants for airborne asexual sporangia. Caffi et al. (2013) detected airborne sporangia on 91.2% of sampling days under a broad range of conditions, including rainless periods. The same paper cited release above 9 m s-1 in a wind-tunnel experiment, but this does not establish a universal field threshold. Brischetto et al. (2021) therefore represented dispersal and deposition jointly with a relative rate of one whenever viable sporangia were available, rather than with separate hard rain and wind thresholds.

The present deposition rule is especially difficult to defend biologically. A wind-released cohort cannot enter the host-surface compartment until rainfall reaches 6.45 mm in one hour, although dry and wet deposition are not separately represented. Surface moisture is already evaluated later for zoospore release, survival, and infection. Requiring intense rainfall at deposition may therefore duplicate a later moisture requirement and delay or remove otherwise viable cohorts.

The former mortality implementation was inconsistent with Brischetto et al. (2020): it divided the VPD polynomial by 24 and applied the result only during daylight. FSIM and FSIM-S now use the reciprocal of 24 times the corresponding VPD polynomial during every hour. At the median weather conditions across the six site-years, the former equation would give one-day survival of approximately 0.007 for attached and 0.019 for detached sporangia if applied for 24 h, compared with approximately 0.816 and 0.778 under the published rates. Historical parameter estimates obtained with the former implementation are therefore obsolete.

## Scenario results

| Scenario | Predictions | Observed-window overlaps | Mean interval distance (d) | Mean absolute midpoint error (d) | Mean dispersal / deposited cohorts |
|---|---:|---:|---:|---:|---:|
| Pre-fix hard-threshold reference | 6/6 | 0/6 | 14.32 | 16.52 | 30.0 / 12.2 |
| Pre-fix release, immediate deposition | 6/6 | 0/6 | 31.41 | 33.79 | 30.0 / 30.0 |
| Measurable rain or 9 m s-1 wind, immediate deposition | 6/6 | 0/6 | 39.76 | 42.16 | 186.2 / 186.2 |
| Broad-condition release and deposition | 6/6 | 0/6 | 39.83 | 42.22 | 201.3 / 201.3 |
| Historical thresholds, corrected reciprocal mortality | 6/6 | 0/6 | 32.15 | 34.36 | 30.3 / 23.3 |
| Broad release, published reciprocal mortality | 6/6 | 0/6 | 39.83 | 42.22 | 201.7 / 201.7 |

Removing the deposition threshold generated many more deposited cohorts and substantially earlier first post-BBCH15 symptom windows. Correcting the mortality-rate equation had a similar effect under the legacy thresholds because more cohorts survived until deposition. This does not validate the current threshold or mortality implementation. Instead, it demonstrates structural compensation: restrictive deposition and excessive mortality suppress early candidates produced by the current inoculum-production, survival, and infection settings. The absence of overlap in every scenario also means that none of these fixed configurations is an acceptable final model.

## Decision for the next analysis

Do not tune the three historical release/deposition thresholds against six onset intervals. With the mortality formulation now corrected, select a biologically defensible transport structure before recalibration. The Brischetto-style broad-condition transfer is the strongest literature-based starting structure, while the 08:00 transfer used here remains an operational detail requiring a robustness check. Then re-audit the production, host-surface moisture, zoospore-release, infection, and dose-to-detection conditions that permit very early candidates. Only after this structural step should Morris screening be rerun and a parsimonious parameter subset be calibrated within grouped training folds. See `brischetto_fsim_process_audit.md` for the complete process mapping.

## Supporting literature

- Caffi T, Gilardi G, Monchiero M, Rossi V. Production and release of asexual sporangia in *Plasmopara viticola*. *Phytopathology*. 2013;103:64-73. https://doi.org/10.1094/PHYTO-04-12-0082-R
- Brischetto C, Bove F, Fedele G, Rossi V. A weather-driven model for predicting infections of grapevines by sporangia of *Plasmopara viticola*. *Frontiers in Plant Science*. 2021;12:636607. https://doi.org/10.3389/fpls.2021.636607
