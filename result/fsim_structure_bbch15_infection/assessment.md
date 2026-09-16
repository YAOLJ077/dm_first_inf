# FSIM infection-gate and transport structure assessment

Reproduce: `python src/evaluate_fsim_host_gate_structures.py`.

## Scope

Six current site-years, fixed `model/config.py` parameters, corrected Rossi
release parentheses, separate physical VPD and Brischetto mortality index,
and BBCH15 gating infection accumulation. No calibration or FSIM-S evaluation
was performed. Scores are descriptive in-sample diagnostics, not independent
validation or evidence that the lowest-error structure is biologically correct.

BBCH15 is treated as a study-specific host eligibility approximation, not a
claim that grape tissue has no susceptibility at any earlier stage. Infection
accumulation begins at the later of release and BBCH15, without extending the
zoospore survival window. Earlier wetness does not contribute. Date-only gate
metadata means eligibility from midnight; missing dates represent no gate.

## Fixed-parameter results

| Structure | Predictions | Overlapping observed intervals | Mean interval distance (d) |
|---|---:|---:|---:|
| S0: legacy release and rainfall-dependent deposition | 6/6 | 1/6 | 15.10 |
| S1: legacy release, deposition after 1 h without rain requirement | 6/6 | 0/6 | 27.74 |
| S2: measurable-rain or wind-event release, deposition after 1 h | 6/6 | 0/6 | 32.37 |
| S3: broad daily release, deposition after 1 h | 6/6 | 0/6 | 32.69 |

All S1-S3 first windows precede observed onset intervals. S2 and S3 give
identical first windows for five site-years. Average full-weather-series
deposited cohort counts rise from 47.17 (S0) to 88 (S1), while release counts
remain 88. These are not exclusively pre-onset cohort counts. The removal of
the secondary rainfall requirement strongly changes availability and onset.
This demonstrates structural sensitivity, not the validity of that requirement.

Transport-delay perturbations of 0, 1 and 3 h give mean interval distances
27.75, 27.74 and 27.73 d for S1, and 32.06, 32.69 and 32.69 d for S3.
The early-onset tendency is stable across these tested delays. This limited
check does not establish robustness to all timing, wetness or weather choices.

## Literature and identifiability

Brischetto et al. (2021), dispersal/deposition compartment, uses relative
D&DR=1 and permits transfer when viable sporangia are available. It does not
establish the three legacy release/deposition thresholds or a universal 1 h
transport delay. S3 is a daily discretization inspired by that abstraction,
not an exact reproduction: fixed 08:00 transfer and full-pool clearance are
additional assumptions.

Rossi et al. (2008) concerns soil-to-host splash of zoospores after release;
its rainfall criterion cannot establish a mandatory rainfall requirement for
deposition of airborne asexual sporangia. Source pathways and propagule stages
must not be conflated.

Six onset intervals and weather alone cannot reliably separate source supply,
release, transport, deposition and surface wetness parameters. Similar S2/S3
endpoints despite different cohort counts reinforce that practical limitation;
this is not a formal structural-identifiability proof. No physical transport
parameters should be fitted from these scores alone.

## Decision

Prefer a parsimonious relative availability/transfer formulation without a
mandatory heavy-rain deposition threshold as the next main structural candidate.
Retain S0 as a labelled legacy sensitivity case. Do not select S0 merely for
its lower current error, and do not call S3 a validated replacement.

Before freezing that candidate, inspect host-surface wetness definition, regional
source availability and the mapping from first feasible infection to detectable
onset. The present binary infection sequence does not weight onset by positive
cohort dose magnitude. Consequently, multiplying all positive doses by a fixed
positive transfer fraction cannot identify a physical deposition efficiency
from timing alone. This does not justify restoring a minimum-spore cutoff.

The run emitted exponential-overflow warnings for some low-temperature release
calculations. The resulting infinite wetness requirement prevents release for
those values, but the temperature-domain handling remains a numerical review
item. It was not silently changed during the transport comparison.

References: https://doi.org/10.3389/fpls.2021.636607 ;
https://doi.org/10.1016/j.ecolmodel.2007.10.046 .

Files `scenario_predictions.csv`, `scenario_summary.csv` and
`scenario_definitions.csv` provide site-year results and all tested definitions.
