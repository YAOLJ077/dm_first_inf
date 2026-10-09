# Accepted analysis results

| Directory | Contents |
| --- | --- |
| `fsim_calibration/` | FSIM fold parameters, predictions, candidate scores and calibration audits. |
| `fsim_s_calibration/` | Accepted FSIM-S spore-constraint calibration, fold parameters and predictions. |
| `deployment_parameters/` | Consensus parameters, runnable configurations and six-site-year evaluation. |
| `phenology_calibration/` | Local Table/Wine model comparison, predictions, adjusted targets and final parameters. |
| `sobol_sensitivity/` | Sobol designs, model outputs and sensitivity indices for N=256, 512 and 1024. |
| `guangxi_2000_2024_onset_windows/` | Table/Wine regional phenology and symptom-window tables. |
| `guangxi_2000_2024_fungicides/` | Table/Wine potential early-spray savings and summaries. |
| `guangxi_2000_2024_weather/` | Annual and spring weather statistics and summaries. |

For model comparisons, use the FSIM rows in `fsim_calibration/` and the
FSIM-S rows in `fsim_s_calibration/`. The baseline calibration archive also
retains other evaluations from that run as provenance; they do not replace
the accepted FSIM-S result. Candidate scores, manifests, selection audits
and Sobol sample outputs are retained to reproduce and audit the analyses.
`sobol_sensitivity/figure09_source_data.csv` combines the first- and total-order
indices and confidence interval half-widths plotted in Figure 9 for the six
site-year designs.

Current entry points are documented in `src/COMMANDS.md`.
