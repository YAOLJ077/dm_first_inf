# Reviewer Comments and Revision Notes

## Target journal and scope

- The default target journal is **Pest Management Science (PMS)**, published by Wiley on behalf of the Society of Chemical Industry.
- Treat the manuscript as a PMS **Research Article** unless the user explicitly selects another article type.
- Keep the paper centered on original, scientifically rigorous pest-management research with clear significance to the journal's researchers and practitioners.
- The topics in this repository that support PMS fit are plant disease epidemiology, grapevine downy mildew management, airborne inoculum monitoring, agroclimatic and mechanistic disease modelling, infection-risk forecasting, decision support, fungicide-timing optimization, pesticide-use reduction potential, and climate-smart crop protection.
- Preserve the paper's core contribution: a First Seasonal Infection Model (FSIM) represents the biological process leading to the first disease event of a new subtropical grapevine growth cycle without assigning its initiating inoculum to the conventional oospore-driven primary-infection pathway; field-observed airborne sporangia are then integrated with the agroclimatic process model to improve retrospective estimation of the resulting disease-onset window and inform fungicide-timing research.
- Frame modelling advances through their biological basis, field validation, uncertainty, and practical pest-management value. Do not let the manuscript become a generic machine-learning, climate-description, or software paper detached from pest-management decisions.
- Claims about reduced fungicide use, improved timing, transferability, or operational value must remain proportional to the study design and evidence.

## Journal references

- The local source of truth for PMS requirements is `paper/PMS_Guide.docx` (author guidelines updated June 2026).
- The operational submission and formatting checklist is `paper/PMS_Submission_Requirements.md`.

## Current revision basis

- Treat `paper/manuscript.tex` as the unrevised scientific baseline. Use the former European Journal of Agronomy reviews as an internal pre-submission critique for the new PMS manuscript, not as comments already resolved.
- Do not incorporate external datasets unless the user later changes this decision. The present evidence base contains six Guangxi site-years; the site-year is the independent model-evaluation unit (`n = 6`). Plot, vine, leaf, slide, and microscopic subsamples describe sampling precision but are not additional validation units.
- Site and cultivar are fully confounded in the current design. Do not claim that their effects have been separated or that cultivar-level transferability has been demonstrated.
- Frame the study as development and retrospective evaluation of a promising spore-informed framework. Prospective operational performance and disease control under a reduced-spray programme remain untested.

## Terminology and interpretation

- Use **observed disease-onset date**, **onset of grapevine downy mildew symptoms**, or **first confirmed oil-spot appearance** for the field endpoint. Use **predicted disease-onset window** (or **predicted window of first seasonal symptom appearance**) for the model's prediction target and final output used in calibration and evaluation.
- Define **first seasonal infection** as the earliest infection event affecting susceptible grapevine tissue in a newly initiated growth or production cycle. It describes the event's temporal position within that cycle, not the date of visible symptoms and not an oospore-defined infection category.
- Explain the name **First Seasonal Infection Model (FSIM)** through the epidemiological contrast: in conventional temperate grapevine downy mildew systems, **primary infection** normally denotes infection initiated by overwintered sexual oospores; in the subtropical study system, the first infection of the new growth cycle is hypothesized to be initiated predominantly by asexual sporangial inoculum that persists or recurs across a short, weakly discontinuous off-season. `FSIM` is therefore used deliberately instead of `primary infection model` to avoid assigning the initiating event to an oospore pathway.
- Recommended definitional wording for the manuscript: "In this study, first seasonal infection denotes the earliest infection event affecting susceptible grapevine tissue in a newly initiated growth cycle. The term is used instead of primary infection because the latter conventionally implies an oospore-derived infection in temperate vineyards, whereas the initiating inoculum in the subtropical study system is hypothesized to be predominantly persistent or recurrent asexual sporangia. FSIM therefore identifies the first infection event by its position in the production cycle rather than by an assumed sexual inoculum source."
- Immediately after this definition, clarify the model scope and output: FSIM reproduces the biological process leading to symptom appearance, including infection establishment and the latent/incubation period, but it does not present the timing of those processes as the prediction endpoint. Its final output is the predicted disease-onset window, which is compared with the observed disease-onset date for calibration and error calculation.
- Use `FSIM` and `FSIM-S` consistently; avoid legacy `FIM`/`FIM-S` names in manuscript prose.
- Define the **spore-pressure field** as a continuous, site-year-specific background inoculum-pressure field constructed from in-situ measured airborne spore counts. It is an observation-derived forcing field for FSIM-S, not a second set of disease observations and not a direct estimate of absolute viable inoculum at every hour.
- Describe the implemented transformation explicitly: retain the relevant measured-count period, place observations on a daily index, interpolate between sampling dates, apply `log1p` scaling relative to the site-year 95th percentile and clip the normalized values to `[0,1]`, smooth over a window related to sampling frequency, and subtract a reference level to obtain the centered daily pressure field. State the exact reference-selection and rolling-window rules used by the finalized analysis rather than describing them generically.
- Interpret the centered field only as deviation from its reference standard: values above zero indicate higher-than-reference background spore pressure, values below zero indicate lower-than-reference background pressure, and zero denotes the reference level. Positive and negative values do **not** represent positive and negative spore counts; the original measured counts remain non-negative.
- Explain the calibration relationship accurately: FSIM-S uses the observation-derived spore-pressure field to modulate the effective infection threshold, with above-reference pressure lowering the threshold and below-reference pressure raising it through separately calibrated response coefficients. Conditional on this pressure forcing, FSIM-S parameters are calibrated by comparing predicted disease-onset windows with observed disease-onset dates; the disease model is not fitted directly to raw spore counts, infection-event timing, or latent-period timing. A separate meteorology-to-pressure module is calibrated against the normalized measured spore signal only for reconstructing pressure where direct observations are unavailable.
- Define timing error against the predicted disease-onset window: assign zero error when the observed disease-onset date falls within the predicted window; otherwise use the signed or absolute distance to the nearest window boundary, as appropriate for the reported metric. Document separately how missing predicted events are penalized.
- Audit temporal causality before finalizing the method. The current code truncates observations at first symptom onset but uses centered smoothing and bidirectional edge filling, and the crop-season controller applies an additional three-day rolling mean. Decide and document whether the study is strictly retrospective or whether a trailing, observation-available-only transformation is required for operational interpretation; remove unintended duplicate smoothing and prevent future observations from informing earlier pressure values if prospective claims are retained.
- Define BBCH 53 correctly as **inflorescences clearly visible**, not flowering. Distinguish BBCH 53 from flowering stages and treat its use as a phenological-gate hypothesis requiring biological justification and sensitivity testing.
- Distinguish throughout among direct field observations, simulated biological processes, the predicted disease-onset window, reconstructed regional inputs, and management scenarios.

## Method and evaluation revisions

- Reduce or fix parameters using biological literature and Morris sensitivity results; calibrate only a defensible influential subset where possible. Report behavioural parameter ensembles, equifinality, and prediction uncertainty rather than presenting one parameter vector as uniquely identified.
- Replace random row-wise three-fold validation as the main robustness evidence with grouped leave-one-site-out and leave-one-year-out stress tests. Keep all parameter and model selection inside each training partition; do not describe full-dataset selection or fit as independent validation.
- Report every out-of-fold site-year prediction, all analysed random seeds, failure cases, uncertainty intervals where defensible, and paired FSIM versus FSIM-S differences. Do not hide mixed or non-significant results or treat seed-fold pairs as independent biological replicates.
- Account for unequal disease-assessment intervals. Prefer last-negative/first-positive interval-censored evaluation; if those records are unavailable, state this explicitly and assess sensitivity to the known inspection intervals.
- Test biologically plausible phenological gates, including earlier susceptible stages (for example BBCH 11-15), BBCH 53, and an ungated alternative. Verify whether the gate belongs at infection establishment rather than at the final disease-onset output.
- Fully document spore sampling units, trap and slide deployment, missing samples, identification criteria, detection limits, technical replication, observer agreement, and direct-versus-wash agreement when records exist. Do not invent unavailable quality-control evidence; disclose it as a limitation.
- Compare ERA5/Open-Meteo forcing with local vineyard observations when suitable records exist, especially for rainfall and leaf wetness. Otherwise state the scale mismatch and associated uncertainty.
- Keep observed-pressure site-year evaluation separate from meteorology-reconstructed pressure used in historical simulations. Success with observed pressure does not validate the reconstructed pressure field.
- Treat regional outputs as compound scenarios and propagate, or explicitly discuss, uncertainty from weather, phenology reconstruction, spore-pressure reconstruction, disease parameters, cultivar assumptions, and management schedules.

## Claims and PMS positioning

- The defensible central claim is that observed spore information showed a **promising retrospective improvement** in symptom-onset estimation within a small Guangxi dataset. Claims of robust superiority, cross-site transferability, or operational readiness require stronger independent evidence.
- Describe management results as **scenario-estimated potentially avoidable early applications** under stated schedule and protection-window assumptions. Do not use unqualified `sprays saved`, claim maintained control, or recommend reduced fungicide use without prospective field efficacy data.
- Acknowledge that a prospective trial comparing conventional, model-guided, and untreated programmes is required to establish disease control, yield protection, pesticide reduction, and grower-level utility.
- Keep the PMS narrative centered on pest-management decisions: why inoculum-aware timing may improve early protection, what the current evidence supports, where uncertainty remains, and what must be validated before operational adoption.
- Align every major Abstract, Introduction, Discussion, and Conclusion claim with an explicit result. Unsupported claims must be weakened or removed. Present numerical effect sizes together with `n`, evaluation design, uncertainty, and limitations.
- Follow the PMS Research Article structure and limits in `paper/PMS_Submission_Requirements.md`, including the structured `BACKGROUND`, `RESULTS`, and `CONCLUSION` abstract. Complete scientific revisions and freeze results before final journal-format conversion.

## Revision workflow

1. Audit recoverable field metadata and define the endpoint, experimental unit, and evidence limitations.
2. Revise the model and validation design, then regenerate traceable results without overwriting prior outputs.
3. Rewrite Materials and Methods and Results from the finalized analyses.
4. Rebuild Discussion and Conclusion around the supported pest-management interpretation and limitations.
5. Revise the title, Introduction, and structured Abstract last, then perform a claim-evidence audit and PMS compliance check.
