# Fixed-Parameter FSIM Diagnostic

## Scope

The diagnostic used the current FSIM implementation without modification, the parameter values in `model/config.py`, the six site-year weather JSON files, and the observed disease-onset intervals in `data/site_year_onset_endpoints.csv`.

## Site-year results

| Site-year | Observed onset interval | Predicted symptom window | Boundary error (days) | Midpoint error (days) | Interpretation |
|---|---|---|---:|---:|---|
| DA_MPT_2022 | 2022-05-05 to 2022-05-08 | 2022-05-13 17:00 to 2022-05-15 00:00 | 5.71 | 7.85 | Late |
| DA_MPT_2023 | 2023-05-25 to 2023-05-28 | 2023-05-28 07:00 to 2023-05-29 13:00 | 0.29 | 2.42 | Slightly late |
| DA_MPT_2024 | 2024-04-24 to 2024-04-27 | 2024-04-22 20:00 to 2024-04-24 00:00 | 0.00 | -2.08 | Overlap |
| GL_WK_2022 | 2022-05-27 to 2022-05-30 | 2022-05-02 23:00 to 2022-05-04 16:00 | -22.33 | -24.69 | Early |
| NN_JF_2023 | 2023-06-07 to 2023-06-10 | 2023-04-25 06:00 to 2023-04-26 16:00 | -41.33 | -43.54 | Early |
| NN_JF_2024 | 2024-04-19 to 2024-04-22 | 2024-04-24 22:00 to 2024-04-26 05:00 | 2.92 | 5.06 | Late |

All six simulations produced a symptom window. One predicted window overlapped its observed interval. Across site-years, the mean absolute boundary error was 12.10 days and the mean absolute midpoint error was 14.27 days. The mean predicted-window width was 1.35 days.

## Process diagnostics

Every simulation produced sporangial dispersal, deposition, survival, infection, incubation, and symptom-appearance outputs. The first symptom window before and after the BBCH53 gate was identical in all six site-years. Consequently, the current prediction errors were not caused by removal of an earlier candidate event at the BBCH53 filtering step.

The prediction errors were not consistent in direction. FSIM was substantially early for GL_WK_2022 and NN_JF_2023, moderately late for DA_MPT_2022 and NN_JF_2024, and close to the observed interval for DA_MPT_2023 and DA_MPT_2024. A uniform date correction would therefore be inappropriate.

The current controller begins the weather simulation seven days before the RB-file BBCH53 date. DA_MPT_2022 is a notable boundary case because its BBCH53 date and first-positive observation are both 2022-05-08; the model cannot generate a symptom window preceding that phenological gate under its current controller design. This is a design issue to examine separately from parameter calibration.

## Decision for the next stage

The fixed-parameter model is computationally viable for all study site-years, so the next stage should focus on model response rather than error handling. Sensitivity analysis should quantify effects on interval boundary error, midpoint error, and successful symptom-window generation. Parameter calibration should remain parsimonious and should not attempt to remove the site-year-specific errors through a single temporal offset.
