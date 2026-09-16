# Review‑comments retrospective

This retrospective has two parts. Part I summarizes the current repository, manuscript, model, data, and numerical evidence. Part II evaluates the two EJA reviewer reports and proposes revisions grounded in that understanding.

## Part I: Current Repository and Manuscript Understanding

### Evidence base

The understanding below was built from:

- `paper/manuscript_eja.pdf` (47 pages; generated 2026-05-07) and its editable source, `paper/manuscript_eja.tex`.
- `review_comments/Reviewer comments 1.docx` (14 major-revision comments).
- `review_comments/Reviewer comments 2.docx` (8 numbered comments).
- Model, calibration, sensitivity, reconstruction, plotting, data, and saved-result files in this repository.

The PDF is treated as the submitted scientific record, the TeX file as the editable manuscript, and the saved CSV/JSON files as the most direct numerical evidence. A statement in the manuscript is not treated as proven merely because it appears in the text.

### Scientific story

The paper is titled **"Spore-informed agroclimatic modelling improves the timing of grapevine downy mildew management in subtropical vineyards."** It addresses grapevine downy mildew caused by *Plasmopara viticola* in humid subtropical Guangxi, China.

The biological argument is:

1. Temperate primary-infection models, especially the Rossi model, emphasize overwintered oospores.
2. In mild-winter, short-dormancy vineyards, recurrent airborne sporangia may be important early-season inoculum.
3. Infection timing therefore depends on the coincidence of susceptible host tissue, suitable weather, and adequate inoculum pressure.
4. FSIM restructures primary- and secondary-infection processes into a cohort-based model for the first visible seasonal disease event.
5. FSIM-S extends FSIM by allowing an observation-derived spore-pressure index to modify the infection threshold.
6. Retrospective regional simulations explore possible onset patterns and spray-timing scenarios for 2000-2024.

The defensible contribution is a **promising, biologically interpretable, spore-informed modelling framework evaluated retrospectively on a small subtropical dataset**. It is not yet a field-validated operational decision-support system, and it has not demonstrated disease control under a reduced-spray programme.

### Study design and observational units

- Locations: Xing'an/Guilin (`GL`), Du'an/Hechi (`DA`), and Jiangnan/Nanning (`NN`).
- Cultivars: Wink (`WK`) at Guilin, Yeniang No. 2 (`MPT`) at Hechi, and Kyoho (`JF`) at Nanning.
- Available site-years in the reported analysis: GL-WK 2022; DA-MPT 2022, 2023, 2024; NN-JF 2023, 2024. Thus `n = 6` site-years, not nine independent site-year observations.
- Site and cultivar are completely confounded: each cultivar occurs at only one site. Cultivar and location effects cannot be separated from these data.
- The manuscript describes two commercial vineyards and one experimental orchard but does not unambiguously identify which location is the experimental orchard. This must be clarified.
- Each vineyard reportedly had traditional fungicide treatment (TT) and untreated control (CK), three plots per treatment, six vines per plot, and two permanently tagged vines per plot.
- The modelling response collapses those subsamples to one earliest confirmed oil-spot date per vineyard/site-year. Plots, vines, leaves, and microscope subsamples are not independent validation replicates. The effective model-evaluation unit is the site-year.
- Phenology was assessed weekly from BBCH 00-13 and every two days thereafter. The manuscript currently gives the first positive date but not the last disease-free date, so onset is interval-censored and should be analysed/reported that way.
- Airborne sporangia were sampled with one passive TPBZ1 trap per vineyard, beside CK, at about 1.5 m. Vaseline-coated slides were replaced every two days. Direct imaging and wash imaging were used; wash imaging used 300 uL sterile water and five 30 uL subsamples. Automated detections were checked by two technicians.

Missing protocol evidence requested by reviewers includes trap efficiency, sampled area/volume or other sampling unit, detection limit, technical replication, inter-observer agreement, agreement between direct and wash counts, and the method used to distinguish *P. viticola* from powdery mildew or other particles.

### What the models actually predict

Terminology is a central issue. Field observations record **first visible oil-spot/lesion appearance**, not the unobserved instant of infection. The mechanistic chain contains an `infection_datetime` and then an incubation calculation that produces `oil_spot_start` and `oil_spot_end`. Calibration error compares the observed symptom date with that simulated oil-spot window:

- error is zero if the observed date lies inside the simulated window;
- otherwise error is distance in days to the nearest window boundary;
- missing simulated events receive a 120-day penalty during calibration.

Therefore, manuscript text and tables should normally say "first symptom onset," "first oil-spot appearance," or "modelled symptom-expression window." If "first seasonal infection" is retained as a study-defined term, define it explicitly and never imply that infection itself was directly observed. Separate infection time from post-incubation symptom time in figures and validation.

### Model understanding in one view

The model should be understood as a mechanistic event-chain model with an observation-informed threshold extension, not as a machine-learning predictor of disease labels.

```text
Inputs
  hourly weather + BBCH gate + calibrated biological parameters
       |
       v
FSIM mechanistic chain
  sporulation -> attached pool -> dispersal -> deposition
  -> zoospore release/survival -> infection -> incubation
       |
       v
Output
  earliest simulated infection time and oil-spot appearance window

FSIM-S adds:
  observed/reconstructed relative spore pressure
       |
       v
  raises or lowers the infection threshold inside the same FSIM chain
```

FSIM-S does not replace the weather-driven mechanism and does not directly insert observed symptoms into the event chain. It changes how readily a viable deposited cohort satisfies the infection criterion. Positive relative pressure lowers the effective infection threshold; below-baseline pressure raises it.

There are two materially different FSIM-S use cases:

- **Site-year evaluation:** pressure is derived from field-observed spore counts. This tests whether measured inoculum information can improve retrospective symptom-onset timing.
- **Historical regional simulation:** direct spore observations do not exist, so pressure is reconstructed from weather using a separately calibrated latent-reservoir model. This tests a compound scenario involving uncertainty from weather, phenology reconstruction, pressure reconstruction, FSIM-S parameters, and management assumptions.

These use cases must not be presented as equivalent. Success with observed pressure does not automatically validate weather-reconstructed pressure, and neither establishes fungicide efficacy.

### FSIM and FSIM-S mechanics

The active baseline implementation is `model/first_infection.py`, controlled by `model/crop_season.py`. The spore-threshold version is `model/first_infection_spore_infthr.py`, controlled by `model/crop_season_spore_infthr.py`.

Both implement a staged chain:

1. attached sporangia production/inventory;
2. rain- or wind-triggered dispersal;
3. deposition;
4. detached-sporangia survival;
5. zoospore release and survival under moisture constraints;
6. infection when wet duration times mean wet-period temperature reaches a threshold;
7. temperature-driven incubation and an oil-spot appearance window;
8. BBCH gate and removal of equivalent infection cohorts.

Important implementation details:

- Weather is hourly. Core drivers include air temperature, relative humidity/VPD, adjusted precipitation, and wind speed.
- `CropSeasonFirst*` starts weather seven days before the supplied `BBCH_53`/legacy `HHCH_53` value, but the final gate is applied to `oil_spot_start >= bbch_53_datetime`. This is a symptom-expression gate, not clearly an infection-time susceptibility gate, and deserves biological review.
- The manuscript incorrectly describes BBCH 53 as flowering. BBCH 53 is an inflorescence-development stage (inflorescences clearly visible); flowering begins later. Reviewers ask for justification and comparison with earlier susceptible stages such as BBCH 11-15.
- FSIM has 13 calibrated parameters. FSIM-S has those 13 plus `spore__alpha_pos` and `spore__alpha_neg`, for 15 parameters against six effective observations. Sensitivity screening was performed, but the current calibration still estimates all parameters.

FSIM-S changes the infection threshold according to pressure `P_t`:

```text
P_t >= 0: IC*_t = IC * exp(-alpha_pos * P_t)
P_t <  0: IC*_t = IC * exp( alpha_neg * (-P_t))
infection if WD_t * TWD_t >= IC*_t
```

Observed spore counts are non-negative. Negative pressure values arise only after transformation: counts are interpolated to daily values, log-normalized by the site-year 95th percentile, smoothed, and centered by subtracting a data-derived baseline. Negative `S_day` therefore means below-reference **relative pressure**, not negative spores. This entire transformation, including interpolation, window selection, clipping, baseline selection, and pre-onset truncation, must be stated reproducibly.

### Calibration and evidence

The principal calibration script is `src/bayes_calibrate_first_kfold_spore_infthr.py`:

- random shuffled 3-fold cross-validation on six rows, so each validation fold contains only two site-years;
- TPE/Optuna search with approximately 20,000 baseline and 30,000 spore-model trials described in the paper (script defaults currently use 30,000);
- top-performing training candidates are compared using validation loss;
- runs exist for seeds 42, 123, and 2024;
- the final parameter set is chosen from fold-specific candidates and evaluated on all six records.

The submitted headline uses seed 42 fold summaries:

| Metric | FSIM | FSIM-S | Claimed reduction |
| --- | ---: | ---: | ---: |
| Training MAE | 6.8 d | 4.0 d | 41.5% |
| Training RMSE | 8.1 d | 4.0 d | 50.3% |
| Validation MAE | 9.3 d | 6.3 d | 32.1% |
| Validation RMSE | 10.6 d | 6.5 d | 38.9% |

Across all three stored seeds, mean validation MAE is 8.72 d for FSIM and 6.78 d for FSIM-S, but improvement is not uniform: seed 123 has equal MAE (8.0 d) and worse FSIM-S RMSE. The stored paired Wilcoxon tests over nine seed-fold pairs are not statistically significant (`p = 0.258` for validation MAE; `p = 0.164` for validation RMSE). Those nine pairs also must not be presented as nine independent biological replicates.

The selected global parameter configurations evaluated on all six records have MAE 4.67 d (FSIM) and 4.33 d (FSIM-S), only a 0.33 d difference. These are in-sample full-dataset evaluations and are not the cross-validated headline metrics.

The final FSIM-S site-year absolute errors are 5, 0, 2, 13, 4, and 2 days for DA22, DA23, DA24, GL22, NN23, and NN24. FSIM-S improves three records by one day, worsens one by one day, and ties two. This does not support wording such as "consistent improvement across site-year combinations."

#### Core validation threats

- The parameter-to-observation ratio (13-15 parameters to six site-years) creates severe identifiability/equifinality risk.
- Random row-wise folds do not directly test leave-site-out or leave-year-out transferability.
- Selecting a final configuration by evaluating fold candidates on the full dataset leaks evaluation information into model selection.
- No external location/year is held out as a final test set.
- Confidence intervals and appropriately paired site-year comparisons are absent.
- With three sites and three cultivars fully confounded, claimed cross-site or cultivar transferability is not established.

Preferred corrective sequence: reduce/fix weak parameters using literature and Morris screening; define the experimental unit; use grouped leave-one-site-out and leave-one-year-out analyses; use nested selection when feasible; report every out-of-fold site-year prediction; quantify parameter/equifinality uncertainty; then reserve genuinely new site-year data as an untouched external test.

### Regional reconstruction and management scenario

The historical workflow has three distinct components:

1. `src/calibrate_bbch53_module.py` and `src/invert_bbch53_historical.py` estimate BBCH 53 from accumulated thermal time, a seven-day heat gate, and cultivar adjustments. Stored leave-one-out BBCH results have MAE 6.5 d and RMSE 9.30 d, including a 20-day error for DA 2023.
2. `src/calibrate_spore_pressure_module.py`, `src/spore_pressure_module.py`, and `src/invert_spore_pressure_historical.py` fit a latent meteorology-driven inoculum reservoir and convert it to relative pressure. Stored LOOCV errors are on the normalized-index scale, not spore-count units, and do not constitute independent regional validation.
3. `src/invert_first_infection_historical.py` couples reconstructed phenology and pressure with FSIM-S for gridded 2000-2024 simulations. Plotting scripts create onset and spray-scenario maps.

The regional result is a scenario, not validated vineyard mapping. The manuscript reports a common onset window around DOY 180-210 and roughly 2-5 "potentially avoidable" early sprays. That estimate applies one grower-executed 2022 Du'an schedule region-wide and labels a spray more than seven days before simulated onset as potentially avoidable. It does not model fungicide identity, mode of action, residual protection, rainfall wash-off, reapplication rules, efficacy, disease severity, yield, or economic loss.

Use wording such as "scenario-estimated potentially avoidable early applications." Do not call these applications "saved" without qualification, do not claim maintained disease control, and do not recommend operational use until prospective field trials compare model-guided and standard programmes.

### Repository map

- `model/`: active FSIM and FSIM-S process implementations and default configuration.
- `observation/`: an older observational support/gating approach; it is not the main pressure-threshold implementation described in the manuscript.
- `src/bayes_calibrate_first_kfold_spore_infthr.py`: current Bayesian cross-validation engine for FSIM-S.
- `src/morris_first_infection*.py`, `src/aggregate_morris_compare_n512.py`: Morris sensitivity runs and aggregation.
- `src/select_final_calibration_params.py`, `src/summarize_calibration_results.py`, `src/export_final_fim_fims_predictions.py`, `src/build_bayes_long_table.py`: parameter selection and result-table construction.
- `src/bbch53_module.py`, `src/calibrate_bbch53_module.py`, `src/invert_bbch53_historical.py`: phenology reconstruction.
- `src/spore_pressure_module.py`, `src/calibrate_spore_pressure_module.py`, `src/invert_spore_pressure_historical.py`: meteorology-based pressure reconstruction.
- `src/invert_first_infection_historical.py`: historical gridded disease simulation.
- `src/plot_*.py`: manuscript/result graphics.
- `data/`: currently retained local weather JSONs and derived spore-pressure tables.
- `result/`: saved calibration, sensitivity, phenology, pressure-proxy, and comparison outputs.
- `fig/`: submitted and working figures; `Figure_01.png` through `Figure_13.png` are current numbered exports.
- `paper/manuscript_eja.tex`: manuscript source; `paper/manuscript_eja.pdf`: submitted/readable manuscript.
- `review_comments/`: the two EJA reports.

### Current reproducibility state

The worktree was already dirty when this guide was created. Preserve all user changes. In particular, source spreadsheets such as `data/actual_data.xlsx`, `data/Spore_Overall.xlsx`, local-station/management files, and the GL/NN weather JSONs appear deleted relative to Git, while saved result files remain. Only DA-MPT 2022-2024 weather JSONs are currently present. Consequently, most calibration and six-site-year export scripts cannot be rerun from the current working tree.

Additional hazards:

- There is no requirements/lock file or documented clean environment setup.
- Some scripts retain stale paths or imports. For example, `run/run.py` imports `model.crop_season_spore`, which is not present, rather than the current `crop_season_spore_infthr` module.
- Historical inversion defaults refer to large sibling directories such as `../input_weather_json`, `../bbch53_date`, `../spore_pressure`, and `../infection_date_2000_2024`; these inputs/outputs are not self-contained here.
- Both `BBCH_53` and legacy/misspelled `HHCH_53` keys occur. Loaders support both, but new data should use `BBCH_53` consistently.
- The manuscript references some embedded media paths and `fig/model_flow_rossi.png`; verify figure paths before rebuilding because assets have changed in the worktree.
- There is no automated test suite. Before scientific changes, add small deterministic tests for pressure transformation, threshold direction, BBCH gating, event-window error, and one fixed site-year simulation.

The deleted inputs should not be silently restored and saved results should not be overwritten. Any future rerun should first establish exact input provenance and preserve previous outputs under versioned names.

## Part II: Reviewer Evaluations and Revision Suggestions

The reviewer concerns are assessed below against the model implementation, manuscript wording, available data, and stored numerical results. A concern is treated as valid when the present evidence cannot support the corresponding manuscript claim. Where an ideal correction requires new data, a scientifically honest fallback for the current revision is also identified.

### Reviewer 1: evaluation and suggested response

| No. | Reviewer concern | Evaluation based on the repo and manuscript | Revision suggestion |
| ---: | --- | --- | --- |
| 1 | 13-15 parameters are calibrated with six site-years | Valid and critical. Morris screening was conducted, but all 13/15 parameters remain calibrated. Six outcome dates cannot identify this parameter space reliably. | Fix literature-supported and weakly sensitive parameters; calibrate a small core set; report behavioural parameter ensembles, equifinality, and prediction intervals. State that calibrated values are not uniquely identified. |
| 2 | Random three-fold CV does not test transferability | Valid. Each validation fold has only two rows, and random splitting can retain the same site in training and validation. Site and cultivar are also confounded. | Add leave-one-site-out and leave-one-year-out analyses. Report every held-out prediction. Describe these as internal stress tests and reserve transferability claims for external sites. |
| 3 | Final parameter selection leaks information | Valid. Fold candidates were evaluated on all six observations to choose the global set. The resulting global MAE is an in-sample descriptive fit. | Perform selection entirely within training data through nested CV, then test once on untouched data. If no external data exist, exclude global-fit metrics from validation claims and use the selected set only for scenario generation. |
| 4 | Evidence for FSIM-S improvement is statistically weak | Valid. Seed-42 favours FSIM-S, but the three-seed results are mixed, stored paired tests are non-significant, and seed-fold pairs are not independent biological replicates. | Publish site-year out-of-fold errors, all seed results, uncertainty intervals, and paired descriptive differences. Call the improvement preliminary unless confirmed on new independent data. |
| 5 | "First seasonal infection" is inferred from symptoms | Valid. The observed endpoint is first confirmed oil-spot appearance; infection time is a latent model output followed by incubation. | Rename the response as first symptom/oil-spot onset. Show simulated infection time and symptom-expression window separately and explain incubation explicitly. |
| 6 | Unequal assessment intervals create interval censoring | Valid. Weekly then two-day assessments mean the true symptom onset lies between the last negative and first positive inspection. | Recover both dates and score predictions against the observation interval. Report interval widths and perform sensitivity analysis if only first-positive dates remain. |
| 7 | Subsamples are not independent validation replicates | Valid. Plot, vine, leaf, slide, and wash subsamples support measurement precision, while the model uses one earliest date per site-year. | Declare the site-year as the evaluation unit. Describe lower-level replication as sampling design, and model it hierarchically only if plot-level onset data are available. |
| 8 | Spore monitoring lacks quantitative validation | Valid and unresolved in the current material. The protocol describes one trap, two counting procedures, automatic detection, and technician checks, but not analytical performance. | Add sampling units, trap efficiency, detection limit, technical replicates, recovery tests, observer agreement, and automated-versus-manual/direct-versus-wash agreement. State missing QC as a limitation. |
| 9 | Pressure-index construction and negative values are unclear | Valid for the manuscript. The code shows that non-negative counts are interpolated, log-normalized, smoothed, and centered; subtraction of the baseline creates negative relative values. | Publish the full algorithm and parameter choices. Rename the quantity a centered relative spore-pressure index and state that negative values mean below-baseline pressure, not negative spore counts. |
| 10 | Regional spore-pressure reconstruction is uncertain | Valid. A meteorology-driven latent reservoir and proxy LOOCV exist, but errors are on a normalized index and there is no independent regional pressure validation. | Report proxy target, units/scale, fold predictions, parameter uncertainty, and sensitivity. Validate at uncalibrated sites in future and propagate pressure uncertainty into onset maps. |
| 11 | Reanalysis weather should be checked against vineyard observations | Valid and unresolved by current outputs. Rainfall and derived LWD are particularly vulnerable to spatial and canopy-scale mismatch. | Compare ERA5/Open-Meteo against local stations at hourly/daily scales using bias, MAE/RMSE, correlation, event detection, and rainfall totals; propagate discrepancies through FSIM-S. |
| 12 | BBCH 53 needs correction and justification | Valid. The manuscript calls BBCH 53 flowering, and the code gates simulated oil-spot start rather than clearly gating infection susceptibility. | Correct the phenological definition. Test BBCH 11-15, BBCH 53, and alternative/no-gate formulations out of fold; verify whether the gate belongs on infection time rather than symptom time. |
| 13 | Spray savings are hypothetical | Valid. The analysis counts applications more than seven days before simulated onset under one 2022 Du'an schedule; it does not test disease control. | Replace "saved sprays" with scenario-estimated potentially avoidable early applications. Add protection-window and schedule sensitivity, specify products, and conduct prospective efficacy trials. |
| 14 | Regional extrapolation lacks uncertainty | Valid. Regional results combine reconstructed phenology, reconstructed pressure, reanalysis weather, one fitted parameter set, and simplified cultivar/schedule assumptions. | Add ensemble uncertainty or sensitivity maps and prediction intervals. Present the output as regional scenarios, not vineyard-specific operational forecasts. |

### Reviewer 2: evaluation and suggested response

| No. | Reviewer concern | Evaluation based on the repo and manuscript | Revision suggestion |
| ---: | --- | --- | --- |
| 1 | Are there three vineyards/orchards, one per location? | Valid clarification request. The manuscript names three locations and says two commercial vineyards plus one experimental orchard, but does not clearly map site to facility type. | State one study site per location and identify explicitly which two were commercial and which one was experimental. Use the same wording in Methods, Table 1, and Figure 2. |
| 2 | Grape type and downy-mildew susceptibility are missing | Valid. Cultivar names are given, but production class and resistance/susceptibility evidence are not. Because cultivar is confounded with site, this information also affects interpretation. | Add table/wine classification, relevant susceptibility rating with a source or field evidence, and a statement that site and cultivar effects cannot be separated. |
| 3 | Rainfall among sites is hard to distinguish in Figure 3 | Valid presentation problem. A climate comparison is ineffective if site-specific rainfall traces cannot be read. | Redesign using aligned site panels, distinct accessible colours/linetypes, or separate rainfall subpanels. Preserve common axes where comparison is intended. |
| 4 | Field size and number of slides/traps are unclear | Valid reproducibility issue. The manuscript states one passive trap per vineyard but not field/plot area or the exact number of simultaneously exposed slides. | Report vineyard, treatment, and plot dimensions; trap count and placement; exposed slide area; one-slide versus multiple-slide deployment; replacement interval; and missing samples. |
| 5 | Other diseases and spore discrimination are unclear | Valid assay-specificity concern. Technician verification alone does not document diagnostic criteria or exclusion of powdery mildew. | Report observed competing diseases/particles, morphological criteria, magnification, training/reference material, adjudication procedure, and molecular confirmation if available. Do not claim species-level certainty beyond the validated method. |
| 6 | Terminology for infection and "asexual sporangia" is questionable | Substantively valid. "First seasonal infection" is study-defined, while the observed response is symptom onset. Sporangia are asexual structures; there are no "sexual sporangia," but contrasting sporangia with oospores can justify one explicit use of "asexual." | Prefer "first seasonal symptom onset" for observations and "modelled initial infection" for the latent event. Define the distinction once. Use "sporangia" thereafter and "oospores" for the sexual inoculum pathway. |
| 7 | Fix the numbers at line 390 | Not yet diagnosable. Line 390 in the available extracted PDF does not reveal an obvious numerical error and may not match the reviewer's paginated/line-numbered version. | Locate the exact submitted source or annotated proof corresponding to the review. Ask the editor/reviewer for clarification only if comparison with that version still leaves the target ambiguous. |
| 8 | Model variables, practical use, validation, and spray efficacy are unclear | Valid and central. Inputs are distributed across a long mechanistic description, while the management conclusion exceeds the retrospective scenario evidence. The paper does not demonstrate that fewer applications control disease. | Add a concise input-process-output table and operational decision-flow concept. Separate retrospective timing evaluation from prospective efficacy. Describe fungicide products, modes of action, timing, residual/protection assumptions, and explicitly state that control under reduced spraying remains untested. |

Reviewer 2's overall decision condition is prospective evidence: at least one independent year showing that model-guided fewer sprays still control downy mildew. Manuscript revision alone cannot satisfy this condition.

### Integrated corrective strategy

The response should not be organized as a collection of defensive sentence edits. It should rebuild the evidence chain from observed endpoint, through model calibration and validation, to regional and management claims.

| Problem raised by reviewers | Correct interpretation | Ideal scientific solution | Honest fallback for the current revision |
| --- | --- | --- | --- |
| 13-15 parameters and six site-years | The model is structurally interpretable but statistically underidentified | Fix literature-supported parameters, calibrate only sensitive parameters, assess profile likelihood/parameter ensembles, and collect more independent site-years | Present parameter sets as non-unique behavioural solutions; show equifinality and substantially weaken parameter-level interpretation |
| Random three-fold validation | Two-row folds estimate split sensitivity more than general transferability | Use leave-one-site-out, leave-one-year-out, and a preregistered external site/year; perform all parameter selection inside training data | Add grouped validation as a stress test and call it internal/retrospective validation, never independent validation |
| Full-data global parameter selection | Choosing the best fold candidate on all observations leaks target information | Nested CV for model selection plus a completely untouched external test set | Remove the selected-global result from validation claims; use it only as a descriptive full-data fit for regional scenario generation |
| Weak statistical support for FSIM-S | Point estimates favour FSIM-S, but n is tiny, seed results are mixed, and stored tests are non-significant | Report site-year out-of-fold paired errors with uncertainty and evaluate on new independent data | Report all six errors and all seeds; describe improvement as preliminary rather than statistically established |
| Symptom date labelled as infection | Infection is latent; first positive oil spot is observed | Record last-negative/first-positive bounds, validate simulated symptom windows as interval-censored outcomes, and separately report inferred infection times | Rename the primary endpoint and acknowledge observation-interval uncertainty |
| Plot/vine/trap subsamples treated as evidence volume | Subsamples improve measurement precision but do not increase independent validation n | Use hierarchical data collection and modelling across replicated vineyards/site-years | Define site-year as the experimental unit and report within-site sampling only as protocol detail |
| Spore assay uncertainty | The pressure input is only as reliable as identification and counting | Validate trap recovery, detection limits, technical replicates, technician agreement, automated/manual agreement, and species discrimination | Add every available QC record and explicitly label missing validation as a limitation |
| Negative spore pressure | Values are centered relative-index deviations, not negative counts | Publish a complete equation/algorithm and sensitivity analysis for normalization, smoothing, and baseline choices | Rename it "centered relative spore-pressure index" and explain negative values explicitly |
| ERA5/Open-Meteo forcing | Reanalysis may not represent vineyard rainfall or canopy wetness | Compare hourly/daily forcing with local stations and propagate weather/LWD error through predictions | Provide product resolution and known limitations; avoid claims of vineyard-scale precision |
| BBCH 53 gate | The stage is misnamed and may begin later than biological susceptibility | Correct the BBCH definition and compare BBCH 11-15, BBCH 53, and ungated/alternative biologically justified gates out of fold | Correct terminology and state that BBCH 53 is a tested operational assumption, not established optimum |
| Historical pressure reconstruction | Proxy LOOCV on normalized series is not independent regional validation | Validate reconstructed pressure prospectively at multiple uncalibrated sites and propagate parameter/weather uncertainty | Present regional pressure and onset only as model scenarios; add uncertainty/sensitivity maps |
| Claimed 2-5 spray savings | Current code counts potentially avoidable schedule events; it does not simulate control efficacy | Run replicated prospective standard-versus-model-guided field trials measuring severity, yield, applications, costs, and fungicide programmes | Use "scenario-estimated potentially avoidable early applications" and remove efficacy/savings recommendations |
| Operational grower value | A date/window alone is not a complete decision rule | Define risk thresholds, lead time, update frequency, action rule, false-alarm costs, and fungicide-specific protection logic; validate prospectively | Provide a conceptual decision workflow labelled as future implementation |

#### Ideal revised study design

The strongest eventual revision would add at least one new season and preferably new vineyards that were not used for model development. Before observing those outcomes, freeze the model structure, calibrated parameters, pressure processing, BBCH rule, and management decision rule. Compare conventional and model-guided programmes in replicated field treatments, and record last-negative/first-positive symptoms, disease severity, yield, fungicide products and timing, spore counts with QC, and local weather. This would simultaneously address external validation, leakage, biological efficacy, and operational usefulness.

#### Best defensible revision without new field data

If new field evidence cannot be obtained within the revision period, the paper should be reframed as framework development and retrospective proof of concept. Add grouped sensitivity analyses, parameter ensembles, interval-aware scoring where records permit, complete methods, assay metadata, and uncertainty/schedule sensitivity. Report null and mixed results transparently. Remove or weaken claims of validation, transferability, spray savings, maintained protection, grower readiness, and field-tested decision support. This will not fully satisfy Reviewer 2's requested efficacy demonstration, but it is scientifically stronger than overstating the current evidence.

### Claim-to-evidence map

| Claim | Current evidence | Status |
| --- | --- | --- |
| Airborne sporangia occurred before symptoms | Six monitored site-years; reported lag 9-72 d | Supported descriptively, subject to assay validation |
| Spore information improves predictive accuracy | Seed-42 random 3-fold MAE 9.3 -> 6.3 d; mixed results across seeds; non-significant paired tests | Promising, not robustly established |
| Model transfers across sites/cultivars | Random folds over six confounded site-years | Not established |
| Model predicts infection | Target is confirmed lesion date; model also simulates latent infection time | Incorrect as currently phrased |
| Historical Guangxi onset clusters around DOY 180-210 | Deterministic gridded reconstruction with modelled BBCH and spore pressure | Supported only as a model scenario |
| FSIM-S saves 2-5 sprays while maintaining control | Comparison to one calendar schedule; no reduced-spray field treatment | Not supported; requires field trial |
| FSIM-S is ready for grower decision support | No prospective operational validation | Not supported |

### Revision priorities

1. Reframe the target and claims: visible symptom onset, retrospective evaluation, scenario-estimated management implications.
2. Build a response matrix tying every reviewer point to manuscript edits, new analyses, new data, or an explicit limitation.
3. Restore or document authoritative raw inputs, data dictionaries, units, missingness, last-negative/first-positive dates, field dimensions, cultivars, fungicides, and spore-QC metadata.
4. Redesign validation around grouped site/year splits and external prospective data; eliminate full-data parameter-selection leakage.
5. Reduce parameter dimensionality and quantify equifinality and prediction uncertainty.
6. Validate spore counting and meteorological forcing; propagate BBCH, pressure, weather, and parameter uncertainty to regional maps.
7. Run schedule/protection-window sensitivity as a clearly labelled scenario analysis.
8. Conduct the requested prospective field trial before making efficacy or operational recommendations.
9. Revise figures, methods, tables, abstract, discussion, conclusion, and data-availability statement to match the actual evidence.

### Manuscript revision guardrails

- Keep one scientific message per paragraph and lead with that message.
- Distinguish observation, model estimate, reconstruction, and management scenario every time.
- Keep `FSIM`/`FSIM-S` naming consistent. Some result filenames use older `FIM`/`FIM-S`; do not introduce that inconsistency into revised prose.
- Report sample size and experimental unit beside performance metrics.
- Never call random-fold evaluation "independent validation" or the selected full-data result "test performance."
- Do not hide null significance results or seed instability.
- Do not use "field-tested decision-support framework," "demonstrated spray savings," "maintaining protection," or similar language without prospective treatment evidence.
- Do not claim general subtropical transferability from Guangxi alone.
- When changing a number, trace it to a saved result and record script, parameters, seed, split, and input version.
- When adding an experiment, update the claim-evidence map before strengthening the abstract or conclusion.
- Before accepting regenerated outputs, confirm that all six site-year inputs match the manuscript, record environment and dependency versions, separate training/validation/test/full-data results, compare against saved result tables, and verify the rebuilt PDF and figures.

### Overall retrospective assessment

- **Contribution:** meaningful framework, but novelty should be framed as integration of monitored inoculum with mechanistic timing, not proven operational superiority.
- **Writing clarity:** model stages are described in detail, but the observed endpoint, pressure transformation, experimental unit, and grower decision path remain unclear.
- **Experimental strength:** field observations are valuable but only six site-years; improvements are unstable and not statistically established.
- **Evaluation completeness:** grouped/external validation, uncertainty, assay validation, local-weather comparison, gate ablation, and field management trials are missing.
- **Method soundness:** biologically interpretable, but overparameterization, leakage, reconstructed regional inputs, symptom-time terminology, and the BBCH gate threaten current conclusions.

The next submission should be considered ready only when every major abstract/introduction claim maps to explicit evidence and every reviewer item is either resolved or transparently scoped as a remaining limitation.
