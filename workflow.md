# Modelling Workflow

This document defines the provisional analytical framework for the manuscript. It should be updated when the revised analyses establish the final parameter set and validation results. Planned analyses are distinguished from completed structural diagnostics; legacy manuscript performance values are not results of the revised workflow.

## 1. Study objective and analytical structure

The study evaluates whether a process-based model representing early-season sporangial inoculum can estimate the onset of grapevine downy mildew in subtropical vineyards, and whether field-observed spore pressure improves those estimates. The analysis comprises three linked components:

1. multi-site field observations;
2. construction of FSIM and its spore-informed extension, FSIM-S;
3. parsimonious parameterization, grouped out-of-fold evaluation, and focused robustness analysis.

The principal comparison is the paired out-of-fold performance of FSIM-S and FSIM at the site-year level. The primary evaluation is grouped leave-one-site-year-out (six folds), with each complete site-year held out once. Leave-one-site-out and leave-one-year-out analyses are secondary stress tests. Regional simulations and fungicide-saving scenarios are excluded from the current paper. Preventive-management relevance is discussed conceptually, without estimating avoided applications or implying established field efficacy.

## 2. Data structure and evaluation unit

The field dataset covers three Guangxi vineyards, six site-years from 2022 to 2024, three cultivars, and CK and TT treatments. It contains sequential disease assessments, airborne-sporangia counts, hourly meteorological records, dated BBCH observations, and management records.

The site-year is the independent unit for model calibration and evaluation. Repeated dates, plots, and vines are observations nested within site-years and are not treated as independent validation replicates. Site and cultivar effects cannot be separated reliably because cultivar is largely confounded with site.

## 3. Disease-onset endpoint

The model output is the predicted timing of first visible symptoms, not the timing of infection or the duration of the latent period. The biological simulation nevertheless represents the sequence from inoculum production to symptom appearance.

- **Last disease-negative assessment (`L`)**: the last assessment with no confirmed downy mildew lesions.
- **First disease-positive assessment (`U`)**: the first assessment with confirmed symptoms.
- **Observed disease-onset interval (`(L, U]`)**: the interval within which first symptom appearance is inferred.
- **Predicted disease-onset window (`[P_start, P_end]`)**: the period in which FSIM or FSIM-S predicts first symptom appearance.

The current endpoint table contains six bounded onset intervals, each three days wide. This status reflects the current optimized disease observations and should be treated as the fixed endpoint definition for the next analysis.

## 4. Process-based models

FSIM represents sporangium production and loss, cohort formation, dispersal, deposition, zoospore release and survival, infection, incubation, and symptom appearance. FSIM-S retains the same process structure and incorporates a field-derived spore-pressure field into the infection process.

Spore pressure is a relative inoculum index derived from measured airborne-sporangia counts. Positive and negative values represent deviations above and below a reference level, respectively; they do not represent positive or negative spore counts. Its calculation distinguishes missing observations from zero counts and uses only information available up to each prediction time. Out-of-fold evaluation therefore uses trailing smoothing, forward-only imputation, and training-fold normalization.

## 5. Parameter analysis and calibration

Parameters are organized into three groups:

1. biological parameters fixed or constrained by published evidence;
2. a small set of locally uncertain and influential parameters estimated from field data; and
3. a parsimonious spore-pressure response term introduced only in FSIM-S.

Morris global sensitivity analysis is used to identify influential processes and assess the stability of parameter rankings. It is a screening method rather than evidence that individual parameters are identifiable. The provisional calibration set contains only a few parameters relative to the six independent site-years, while predefined ranges propagate uncertainty in influential fixed parameters.

Within each leave-one-site-year-out training fold, shared FSIM parameters are calibrated first. These parameters are then retained when estimating the additional spore-response term for FSIM-S. This staged comparison limits the extent to which apparent improvement can arise from additional parameter flexibility. Joint recalibration of both models may be retained as a sensitivity analysis. The primary calibration target is the interval distance between the predicted disease-onset window and the observed `(L,U]` onset interval; midpoint error is secondary and exact-date error is sensitivity analysis only.

Near-optimal parameter sets characterize equifinality and prediction uncertainty. Parameter selection first prioritizes successful onset predictions and then minimizes interval-based timing error. Missing predictions are reported separately rather than incorporated into the principal MAE through an arbitrary large penalty.

### 5.1 Dispersal and deposition structure audit

FSIM now applies BBCH15 inside infection accumulation: only wet hours at or
after the host eligibility date contribute to the infection requirement.
Earlier sporangium production, transport and zoospore release remain possible,
but a pre-gate zoospore cohort must survive continuously to the gate to infect.
The gate neither restarts zoospore survival nor counts pre-gate wet hours.
The final eligibility flag checks infection time rather than symptom time.
Missing BBCH15 explicitly represents an ungated diagnostic case. This change
currently applies to FSIM; synchronization of FSIM-S is deferred to its review.
The active fixed-parameter comparison is reproducible with
`python src/evaluate_fsim_host_gate_structures.py`; its outputs are in
`result/fsim_merged_transfer_vpd/`. The former `result/fsim_merged_transfer/`,
`result/fsim_dispersal_audit/`
and `result/fsim_structure_bbch15_infection/` directories contain historical
comparisons and are not the active candidate set.

Only two structures are compared: `M0_merged_transfer` and
`M1_merged_transfer_wind_gate`. Both transfer only available, viable attached
cohorts at the beginning of each hourly step. Untransferred cohorts accrue
attached-state mortality for that hour; transferred cohorts accrue detached-state
mortality instead. Newly available cohorts are not charged an attached hour
before immediate transfer. Thus no hourly step is counted in both states.
The candidates share the
same production schedule, mortality equations, infection gate and downstream
processes. M0 has no additional meteorological transfer gate. M1 applies one
wind-speed gate and temporarily uses the historical configured wind threshold;
this value is neither newly calibrated nor a published biological constant.
Untransferred cohorts remain in the attached inventory and continue to lose
viability. Potential production and other intermediate quantities do not pass
directly through the transfer gate.

Transfer and relative leaf availability share a timestamp. No separate rain
release rule, rain deposition rule, deposition efficiency or transport delay
is included in either candidate. The existing dispersal/deposition table names
are retained for downstream compatibility; their amounts represent relative
available inoculum, not absolute airborne concentrations or 100% interception
of actual spores. Detached mortality starts at the merged transition and
continues downstream; there is no separate airborne residence period to age
or reset. Rainfall remains an input to the existing moisture-related processes.

Both alternatives now run through the core `FirstInfection` engine and
`CropSeasonFirst.simulate(use_wind_gate=False)` (M0) or
`CropSeasonFirst.simulate(use_wind_gate=True)` (M1). Omitting the argument uses
`config["dispersal"]["use_wind_gate"]`, currently True. A call override does not
mutate the configuration. The comparison script has no separate model subclass;
`run_scenario(..., config=...)` accepts caller-supplied parameters for subsequent
analysis. Missing or invalid wind data explicitly fail M1, while M0 does not
require wind for transfer. The ordinary diagnostic and the retained Morris
script also accept `--no-use-wind-gate` to select M0; the latter still screens
only the two common parameters and does not establish a wind parameter range.

The downstream leaf-moisture proxy now uses physical VPD alone in both FSIM and
FSIM-S. It is used for zoospore release, survival and infection, and remains
distinct from the nocturnal source-organ moisture rule and from the empirical
predictor in the mortality equations. Incubation completes at the shared fixed
value 1; a non-1 `incubation.spl_end` is rejected rather than silently fitted.
FSIM-S transport, spore pressure and host-gate review remain deferred. Legacy
rain-transfer parameters are explicitly retained only for that older FSIM-S
transport path and are not read by either current FSIM candidate. Historical
calibration scripts and outputs are not current analysis specifications.

The Brischetto mortality equations have been restored in both FSIM and FSIM-S before model selection. Each production event forms an independent sporangium cohort. Hourly mortality is calculated as the reciprocal of 24 times the corresponding VPD polynomial and accumulated separately for the attached and detached states of each cohort. Detached-state mortality continues without resetting from aerial transport through deposition on the host surface. In each state, viable dose equals the dose entering that state multiplied by one minus cumulative mortality and reaches zero when cumulative mortality reaches one. No fitted relative-dose cutoff is used. Parameter screening and calibration must therefore be rerun; estimates obtained under the former equation are not transferable. The revised structural audit must additionally distinguish the first biologically feasible infection from the accumulation or detection process required for that infection to produce the first field-observable lesion. Extreme transport thresholds must not be used to replace this dose-to-onset relationship.

Nocturnal production is grouped causally from 20:00 on the preceding date through 07:59 on the production date, and the resulting dose becomes available at 08:00. A moist hour is defined by RH at least 80%, rainfall greater than zero, or measured leaf wetness longer than 30 min. Because the current weather inputs contain no leaf-wetness observations, the implemented rule presently uses the RH-or-rain components. Sporulation is triggered when the nocturnal moist period is at least 3 h and its mean temperature is between 10 and 30°C. Production is calculated with the Lalancette et al. (1988) Richards response surface and standardized by the predicted number of sporangia at 20°C and 24 h, following Brischetto et al. (2021). No repeated-sporulation or lesion-age decay term is applied; post-production sporangium mortality remains a separate process.

The same two-candidate audit can also be run with:

```bash
python src/audit_fsim_dispersal_scenarios.py
```

## 6. Out-of-fold model evaluation

Random record-level or random site-year splits are not used. The primary grouped validation scheme is:

- **Leave-one-site-year-out (LOSYO)** leaves one complete site-year out for testing and is the principal reported evaluation (`n = 6` folds).

Secondary stress tests may include:

- **Leave-one-site-out (LOSO)**, which evaluates transfer to a vineyard not used for calibration;
- **Leave-one-year-out (LOYO)**, which evaluates transfer to a year not used for calibration.

Spore-pressure preprocessing, parameter estimation, and any data-dependent model selection are performed using training site-years only. Predictions from LOSYO, LOSO, and LOYO are reported separately and are not pooled as independent observations. A final fit to all site-years may summarize the model after evaluation, but must not contribute to reported predictive accuracy or replace held-out predictions.

For a bounded observed interval, signed boundary error is defined as:

```text
e = P_end - L     if P_end < L
e = P_start - U   if P_start > U
e = 0             if [P_start, P_end] overlaps (L, U]
```

A negative value denotes an early prediction and a positive value a late prediction. Evaluation reports interval overlap, signed and absolute boundary error, MAE or RMSE where appropriate, predicted-window width, and the frequency of no-onset predictions. Left-censored site-years receive a separate one-sided evaluation. Use of the first positive assessment as an exact onset date is limited to sensitivity analysis.

FSIM-S and FSIM are compared using paired out-of-fold site-year predictions. Site-year results, the direction and magnitude of paired error differences, and uncertainty across retained parameter sets are emphasized. Given the small number of independent units, inferential tests and confidence intervals are interpreted descriptively rather than as definitive evidence of general superiority.

## 7. Manuscript scope and Results plan

The active manuscript excludes regional disease-onset maps, historical regional phenology/spore-pressure reconstruction, cultivar-distribution scenarios, reference-schedule comparisons and fungicide-saving estimates. Their historical scripts, datasets and figures are retained as archived work, not as evidence in this manuscript or default Supporting Information.

The planned Results sequence is:

1. **Field-observed airborne sporangia and disease onset:** observed counts, phenology, first detection and symptom-assessment intervals.
2. **Retrospective evaluation of disease-onset windows:** six LOSYO predictions per model, timing errors, window widths and missing predictions.
3. **Incremental contribution of observed spore pressure:** paired site-year differences and focused information controls where justified.
4. **Robustness and uncertainty of onset estimates:** parameter equifinality, host gating, temporal preprocessing and weather-input sensitivity.

Management interpretation remains in Discussion. Modelled infection-to-symptom intervals are internal reconstructions, not demonstrated forecast lead times or safe spray-delay intervals.

The 2026-09-13 revision removes regional-analysis text and its dedicated figures, sets the Introduction target at about 700 English words (maximum 750, including citations), and the structured Abstract at about 200 words. Preserve the core information and logical links rather than maximizing compression. Figure 1 and its caption remain unchanged at the user’s explicit request; its regional-analysis branch is a deferred graphical update. This revision does not recalculate or reorganize the retained model-performance and sensitivity results. The Abstract temporarily emphasizes observed sporangia-to-symptom timing and the mechanistic contribution; quantitative FSIM/FSIM-S comparison results must be inserted after the revised evaluation is completed. The old three-fold calibration/validation descriptions and performance numbers remain pending replacement and must not be presented as completed LOSYO evidence.

## 8. Data adapter and diagnostic run

The optimized CK table remains the source for disease, spore, phenology, and coordinate observations. A separate input-adapter script maps `Duan`, `Guilin`, and `Mingyang` to the model identifiers `DA`, `GL`, and `NN`, maps the recorded cultivars to model varieties, validates the six site-year endpoint records, and links each case to its existing weather JSON file. It does not infer phenological start dates or modify the FSIM implementation.

The first executable analysis is a fixed-parameter diagnostic run for all six site-years. It produces one prediction record per site-year containing the FSIM predicted symptom window, the observed `(L,U]` interval, interval distance, midpoint difference, overlap status, and no-prediction status. This diagnostic precedes sensitivity analysis and parameter calibration.

After the diagnostic run is verified, the analysis proceeds to:

1. audit and selection of the dispersal/deposition structure;
2. architecture-based sensitivity analysis using the interval endpoint;
3. parsimonious training-fold calibration of FSIM;
4. estimation of a single additional spore-pressure response for FSIM-S;
5. primary LOSYO paired out-of-fold comparison; and
6. focused uncertainty analyses and secondary LOSO/LOYO stress tests.

## 9. Provisional workflow

```text
Field observations
    -> FSIM and FSIM-S construction
    -> parsimonious training-fold calibration
    -> LOSYO prediction (six held-out site-years)
    -> interval-based paired model evaluation
    -> uncertainty and secondary LOSO/LOYO stress tests
    -> bounded interpretation for preventive timing
```

## 10. Pending analytical specifications

- Final calibrated parameter subset and uncertainty ranges.
- Final interval-distance and midpoint-loss implementation.
- Agreement and quality-control evidence for airborne-sporangia counts.
- Completion of LOSYO estimates and replacement of legacy manuscript performance values.
- Robustness to host gating, temporally non-causal spore transformations and weather inputs.
- Final Abstract model-comparison numbers, supported by checked out-of-fold results.

## 11. Completed table-driven parameter response analysis (2026-09-13)

Runtime settings and candidate grids now come from
`config/fsim_runtime_parameters.csv` via `model/parameter_table.py`.
The new entries are `src/analyze_fsim_parameters.py` and
`src/report_fsim_parameter_analysis.py`. The completed descriptive run is
`result/fsim_parameter_analysis_v1/`; the earlier unsuffixed directory records
an input-validation stop and is not a scientific result.

M0 has 63 grid combinations of the two common candidates; M1 has 567
combinations including its own wind threshold. The wind range 0–8 m/s is an
exploratory weather-supported range, including zero as the M0 equivalence
control and the high tail as a stress test, not a biological confidence range.
The existing historical defaults remain unchanged and were evaluated separately.

The run produced 3,780 grid site-year evaluations plus 12 default controls.
All 63 M0 combinations and 428 of 567 M1 combinations produced predictions
for all six site-years. At complete coverage, the minimum full-data grid
interval MAEs were 28.729 and 28.035 days, respectively. These are descriptive
in-sample minima, not calibrated or out-of-fold performance. The one-day
near-optimal sets contained 7 M0 combinations and 16 M1 combinations;
I_req reached the upper analysis boundary in both. The M1 near-optimal wind
values were non-contiguous (0, 1, 4, 5, and 6 m/s), so this analysis does not
establish a unique wind threshold or superiority of M1.

The deliverables include per-site-year response surfaces, coverage and timing
metrics, coverage-prioritized grid profiles, near-optimal sets at several
descriptive tolerances, and adjacent-grid elementary effects normalized by
parameter range. These effects are not Morris or Sobol estimates. They do not
replace any subsequently required training-fold screening or calibration.

Input checks identified 24 conflicting duplicate December-31 timestamps in
each of DA_MPT_2024 and NN_JF_2024. The original JSON files are preserved.
Analysis copies retain the last record, with all combinations rerun retaining
the first record as a sensitivity check: all 1,264 paired onset outputs agree.
This establishes endpoint insensitivity for this grid, not the true weather
record. The zero-wind M1 control also matches M0 in all 378 paired grid
site-year evaluations. Source snapshots and input hashes are recorded.

Substantial onset mismatch and broad/disconnected near-optimal sets should be
reviewed before treating numerical optimization as parameter identification.
No fitted dose cutoff, additional biological parameter, or manuscript
performance replacement was introduced. Future model selection and calibration
remain confined to LOSYO training folds.

## 12. I_req upper-bound exploration to 200 (2026-09-13)

The user authorized a separate range-expansion experiment, implemented in
`config/fsim_Ireq200_exploration.csv`. All original seven I_req grid values
were retained and eleven values from 100 to 200 at increments of 10 added.
The other grids, model equations, historical defaults and original weather
files were preserved. The main runtime table still records the earlier
baseline range; this alternative table identifies the exploratory analysis.

`result/fsim_Ireq200_exploration/` contains 1,620 grid combinations (162 M0,
1,458 M1), 9,732 main site-year records including defaults, and 3,244 duplicate
weather-policy sensitivity records. The 3,792 records shared with the baseline
run reproduce exactly. At complete six-site-year prediction coverage, the
full-data grid minimum interval MAE decreases from 28.729 to 19.854 days for
M0 and from 28.035 to 19.854 days for M1. Both expanded-grid minima use
I_req=200 and VPD_w=4; the M1 minimum uses u_tr=0 and therefore coincides with
M0. These are descriptive grid minima, not calibrated biological estimates.

The expansion does not improve all site-years: comparing each model's old
and new grid-best combination, M0 improves four site-years, worsens one and
leaves one unchanged; M1 improves four and worsens two. No observed onset
interval overlaps the expanded-grid-best predicted window. Original I_req
bounds constrained the attainable grid fit, but the onset mismatch remains.

The conditional check `src/refine_fsim_Ireq_profile.py` fixes VPD_w and u_tr
at those expanded-grid-best settings and evaluates I_req=180,...,200 in unit
increments (252 main and 84 duplicate-policy records). The minimum remains
at 200. The improvement from 199 to 200 is entirely concentrated in DA_MPT_2023,
whose predicted window starts on April 23 rather than April 6, reducing its
absolute boundary error by 16.792 days and the six-site-year MAE by 2.799 days.
This endpoint jump does not establish a stable or uniquely identified parameter.
No exploration above 200 was performed.

`src/compare_fsim_Ireq_expansion.py` checks baseline reproduction, compares
site-year errors, profiles nested upper bounds and holds other parameters fixed
when isolating I_req search expansion. Expanded-range normalized sensitivity
values must not be compared naively with baseline values because the I_req
normalizing width and grid distribution changed. Results remain full-data
exploration; future calibration and model selection require training-fold
control and explicit recognition of this data-informed range choice.

## 13. Sobol error-sensitivity implementation (2026-09-13)

The new standalone entry `src/sobol_fsim.py` uses the paired SALib 1.5.2
Sobol/Saltelli sampler and estimator. The continuous independent-uniform
exploration ranges are VPD_w 1–8 hPa, I_req 45–300 °C h, and M1-only u_tr
1–8 m/s, defined in `config/fsim_sobol_parameters.csv`. This table deliberately
has no grid specification. `config/fsim_sobol_analysis.json` defines sample
sizes, seeds, second-order estimation and numerical checks.

The main response is RMSE of the six site-year onset-interval boundary errors,
with MAE and prediction availability as additional responses. Site-year signed
and absolute errors and availability are analyzed separately. The denominator
of aggregate RMSE/MAE remains six. If any prediction is missing, the aggregate
timing response for that parameter row is undefined. If a response contains
any non-finite value in a complete Saltelli design, its Sobol analysis is
explicitly unavailable; the script does not drop rows, impute dates, assign
penalties, restrict the domain or choose parameters. Constant responses and
zero A/B-base variance are also explicitly marked undefined.

The current starting design uses two independent scrambles and N=128 per model,
with nested N=32/64/128 checks, S1/ST/S2, 500 bootstrap resamples and 95%
estimator intervals. It has 3,584 parameter rows and 21,504 main site-year
evaluations, plus 7,168 duplicate-weather-policy checks (28,672 total).
The current design-only snapshot is in `result/fsim_sobol_design_n128/`;
no formal model evaluation was launched as part of this settings update.
The earlier N=512 design in `result/fsim_sobol_design/` is retained as a
historical snapshot, not the current default. Larger N is reserved for
follow-up if convergence and independent-seed diagnostics warrant it.

Implementation verification passed 46 tests, including known additive and
multiplicative-function Sobol indices, nested design reproducibility, response
row alignment and missing/constant-response handling. The N=8, one-scramble
end-to-end smoke run in `result/fsim_sobol_smoke/` completed 672 main evaluations
and 224 duplicate-policy checks. In that small design, M0 aggregate errors
were defined, but 13 of 64 M1 parameter rows lacked complete six-site-year
predictions. Consequently M1 aggregate RMSE/MAE Sobol indices were not computed.
This is a verified response-definition limitation, not a calibration result
or a formal sensitivity ranking. Larger N may expose further missing outputs.
See `config/SOBOL_README.md` for commands and interpretation requirements.

The user subsequently completed the N=128 run in `result/fsim_sobol_n128/`.
Post-run review verified 21,504 main evaluations and 7,168 duplicate-policy
checks, no runtime errors and no duplicate-policy output differences.
Aggregate RMSE/MAE are unavailable for both models and both seeds because
some design rows lack predictions. Some finite bootstrap half-widths are
extreme (approximately 1e28–1e33), so the original finite-value `ok` status
must not be interpreted as reliable estimation or convergence.
`src/plot_sobol_fsim.py` generates five PNG/PDF figures in
`fig/fsim_sobol_n128/`, covering response availability, prediction-rate
S1/ST/S2, site-year availability ST, estimable absolute-error S1/ST, and
sample-size/seed stability. Extreme intervals are explicitly annotated,
and all raw estimates are retained in the figure audit CSV. The original
analysis output is preserved; this review does not calibrate parameters
or update manuscript results. See the figure directory README for interpretation.

The user approved an exploratory 10-day accuracy tolerance. The new response
is the fraction of the fixed six site-years with a prediction and absolute
onset-interval error <=10 days; absent predictions fail that accuracy
condition without receiving an imputed error. `accuracy_tolerances_days`
in the Sobol settings enables the response for future runs. The existing
N=128 predictions were postprocessed by `src/analyze_sobol_accuracy.py`,
with no new model evaluations, into `result/fsim_sobol_n128_accuracy10/`.
Old response values and indices were checked against the parent run;
the new columns are appended to preserve legacy bootstrap seed allocation.
Three PNG/PDF figures are in `fig/fsim_sobol_n128_accuracy10/`.
Both aggregate responses are complete and estimable for both seeds, but
some site-year responses are constant or lack A/B-base variance. M0 VPD_w
ST is approximately 0.82 in both seeds; M1 u_tr ST is approximately
0.73/0.80, with substantial uncertainty and seed differences in some other
indices. Ten days is an exploratory criterion, not a validated management
standard, and these results are not calibration or validation performance.

The user next approved a 7-day tolerance and an exactly matched structural
comparison. Seven-day postprocessing is complete in
`result/fsim_sobol_n128_accuracy7/`, with figures in the matching `fig/`
directory. The future-run tolerance list is [10, 7], retaining the previous
10-day bootstrap-column order. Original runs and their snapshots are preserved.

`src/compare_fsim_paired.py` evaluated M0 at the 1,024 unique shared-coordinate
combinations in the original 2,048-row M1 design (6,144 main and 2,048
duplicate-weather checks, four workers). Exact shared parameters, model/input
hashes and fixed configuration were checked; no runtime or weather-policy
differences occurred. `src/report_fsim_paired.py` expands the M0 cache back to
every M1 design row and reports complete paired accuracy/availability contrasts,
conditional timing changes, and full-design Sobol indices for M1-minus-M0.
Outputs are in `result/fsim_paired_n128/` and `fig/fsim_paired_n128/`.

Primary descriptive summaries use only A/B base rows and separate seeds;
full Saltelli-row summaries are also retained. Across seeds, adding the wind
gate changes seven-day accuracy by -1.37 percentage points and ten-day
accuracy by -5.66 points, with 7.49% of pairs losing predictions and no
observed gains in prediction availability. These are conditional parameter-space
comparisons, not independent validation performance. DA_MPT_2022 improves,
whereas NN_JF_2024 deteriorates markedly. High wind-threshold bands have more
prediction losses, but the band summaries do not identify an optimal or
biologically validated range. Manuscript model-comparison results remain pending.

## M0 LOSYO calibration implementation

`src/calibrate_m0_losyo.py` and `config/fsim_m0_losyo.json` implement the
approved exploratory calibration workflow. Each of six folds requires
predictions for all five training site-years and minimizes their mean absolute
onset-interval boundary error. A 15x18 coarse grid is refined twice around
all sampled regions within one day of the training minimum; no feasible
training point triggers full-domain refinement. Held-out outcomes do not
enter feasibility, refinement or tie-breaking. Selection is written before
the held-out evaluation is requested. Deterministic model evaluations are
cached across folds while training objectives are recomputed separately.

The entry supports four workers, progress bars, source/input fingerprints,
weather-duplicate checks, tied and near-optimal sets, stage loss surfaces,
and OOF predictions with explicit calibration failures and missing predictions.
Full six-site OOF MAE/RMSE remain undefined if any prediction is unavailable;
conditional errors and fixed-denominator seven/ten-day accuracy are separate.
Documentation and commands are in `config/LOSYO_README.md`.

Verification: 55 tests passed, including held-out target perturbation,
selection-before-test ordering, disconnected refinement regions, infeasible
folds, candidate budgets and fixed denominators. A real six-fold smoke run
in `/private/tmp/fsim_m0_losyo_smoke_v2/` completed 150 primary and 50 duplicate
model evaluations, with zero duplicate-policy differences. The formal coarse
design is recorded in `result/fsim_m0_losyo_design/`. No formal calibration
was launched; smoke results are not scientific calibration results.

The subsequent user-requested LOSYO range update uses a dedicated runtime table,
`config/fsim_m0_losyo_parameters.csv`: I_req 40–400, with coarse/refined steps
10/5/2.5 °C h. VPD_w remains 1–8 with steps 0.5/0.25/0.125 hPa.
The starting grid is now 15x37=555 points and the fully refined lattice is
57x145=8,265 points; the per-fold candidate budget is 9,000. The model-evaluation
upper bound including duplicate-weather checks is 66,120 with cross-fold caching.
The original Sobol parameter table and past run snapshots are retained.
The current design-only snapshot is `result/fsim_m0_losyo_design_Ireq40_400/`.

## M0 LOSYO third-refinement trial

The user approved a denser-grid trial. The independent configuration
`config/fsim_m0_losyo_refine3.json` retains the 555-point initial grid,
parameter bounds and all training-selection rules, adds a third refinement
round (finest VPD_w 0.0625 hPa, I_req 1.25 °C h), and raises the candidate
budget to 35,000 (full finest lattice 32,657). The default two-round settings
are retained. The run completed in `result/fsim_m0_losyo_Ireq40_400_refine3/`
with 8,380 evaluations, including 1,961 duplicate-policy checks, and no
calibration failures, missing OOF predictions or duplicate-policy differences.

`src/compare_m0_losyo_runs.py` verified matching model/input fingerprints and
pointwise reproduction of every previous-stage training loss surface in all
six folds. The extra round did not materially lower any fold's best training
MAE; tie counts increased, changing three deterministic representative choices.
OOF MAE changed from 33.756944 to 33.743056 days and RMSE from 39.631181 to
39.610937 days. Seven/ten-day accuracy remained 0/6 and 1/6. This supports
stability of the discovered training minima under this additional refinement,
not a proof of a global continuous optimum or uniqueness of parameter estimates.
The comparison CSVs and `refinement_comparison.md` are saved in the refined
run directory. No settings were selected using held-out performance.

## Exploratory RH_sp calibration extension

The user approved adding RH_sp for one trial. `config/fsim_m0_losyo_rh_trial.json`
predeclares discrete values 75/80/85/90%; all other settings follow the verified
three-refinement M0 baseline. These are exploratory humidity thresholds, not a
literature-validated continuous parameter interval. The code uses independent
runtime configurations and process pools for each new RH slice. MP_sp and all
other fixed process values remain unchanged, and runtime config equality was
checked except for the intended RH change. RH=80 reuses the verified parent run.

`src/calibrate_m0_rh_trial.py` completed 26,443 new evaluations (including 6,386
weather-policy checks) in `result/fsim_m0_losyo_RH_trial/`, with no runtime,
calibration or held-out prediction failures and no weather-policy differences.
For each fold the joint selector reads only the four training loss surfaces;
its choice is frozen before the corresponding slice's OOF result is retrieved.
The tie rule prefers RH closest to 80, then lower RH, VPD_w and I_req. Test cases
explicitly verify selection of the training winner even when its OOF error is
worse. Sixty tests passed; all 24 slice selections, six joint selections, runtime
settings, hashes and aggregate metrics were independently rechecked.

Selected RH by held-out site-year is 75/90/80/75/90/75 in manifest order.
Five training folds improve and one is unchanged; the descriptive mean of the
six overlapping training MAEs declines from 12.8736 to 11.8750 days. OOF MAE
worsens from 33.7431 to 45.4306 days, RMSE from 39.6109 to 49.2673 days, and
10-day accuracy from 1/6 to 0/6 (7-day accuracy remains 0/6). Changes are dominated
by DA_MPT_2023 (signed error -30.33 to -46.04 days) and NN_JF_2024 (+9.08 to
+63.50 days). Other OOF errors are unchanged. This trial does not support adopting
RH as a free parameter under the tested candidates and search settings. The
original two-parameter M0 and RH=80 baseline are retained. Three PNG/PDF figures
were generated with `src/plot_m0_rh_trial.py` in `fig/fsim_m0_losyo_RH_trial/`.

## FSIM-S structural alignment (2026-09-14)

FSIM-S now inherits the FSIM process chain and adds only an observation-derived background airborne dose through an external dose-provider interface. The dose is scaled by the single parameter `spore.beta_sp`, appended as independent cohorts before dispersal, and propagated through the shared mortality, deposition, zoospore-release and infection processes. The zero-pressure case is an exact FSIM equivalence control. Legacy threshold and air/deposition multiplier paths (`alpha_sp`, `alpha_air`, `alpha_dep`, `alpha_pos`, `alpha_neg`) are retired. Calibration changes remain pending; the external pressure preprocessing is implemented separately.
