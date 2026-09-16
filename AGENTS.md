# Manuscript Revision Strategy

## Target journal

The target journal is **Pest Management Science (PMS)**, and the manuscript is positioned as a **Research Article**. The scientific scope combines grapevine downy mildew epidemiology, airborne inoculum monitoring, process-based modelling, disease-onset estimation, and risk-informed fungicide timing in subtropical vineyards.

### Authoritative manuscript files

Use `paper/manuscript.tex` as the single authoritative manuscript source and regenerate `paper/manuscript.pdf` after every substantive text, equation, table, figure, or layout change. Do not directly revise `paper/manuscript.docx` unless the user explicitly requests a new Word export. The DOCX file is not expected to remain synchronized with later TeX revisions.

## Current revision scope (2026-09-13)

The paper focuses on field-observed inoculum, mechanistic disease-onset reconstruction, and the incremental information supplied by FSIM-S within six Guangxi site-years. Regional disease-onset mapping, historical phenology and spore-pressure reconstruction for regional application, and regional fungicide-saving scenarios are excluded from this manuscript. Preserve historical code and data; do not restore these analyses to the paper or its Supporting Information without a new user request.

Aim for an Introduction of **about 700 English words** (no more than **750**, including in-text citations) and a structured Abstract of **about 200 words** (no more than **250**). Preserve the principal biological information and causal links; do not pursue maximum compression at the expense of the argument. The Introduction follows management need -> empirical timing systems -> complementary Rossi/Brischetto models -> regional inoculum hypothesis -> FSIM -> FSIM-S and two retrospective research questions. Management relevance belongs in a bounded Discussion paragraph, not a quantitative spray-saving result.

This revision changes the Abstract, Introduction and regional-analysis material, with necessary workflow, conclusion and cross-reference updates. Figure 1 and its caption are explicitly retained unchanged at the user’s request, including their legacy regional-analysis branch; update them only when requested. Other model methods, sensitivity results and performance tables remain pending scientific revision. The LOSYO design below is the intended analysis, not a claim that it has already been completed. Do not relabel legacy three-fold results as LOSYO results. The current Abstract reports observational timing evidence and the mechanistic contribution; insert quantitative model-comparison results only after the revised analysis is completed and checked.

## Scientific rationale

### Management need

Downy mildew control in the study region relies largely on calendar schedules, grower experience, and weather cues. These practices can lead to prophylactic applications that are poorly synchronized with infection risk. Effective early-season management requires joint consideration of pathogen availability, host susceptibility, and environmental suitability.

### Production context

The warm, humid meteorological conditions prevalent in subtropical grape-growing regions are highly conducive to the occurrence and epidemic development of grapevine downy mildew. Cropping-system asynchrony may prolong green-tissue availability in the surrounding landscape and thereby facilitate asexual interseasonal survival of the pathogen.

### Epidemiological distinction

Rossi et al. (2008) modelled primary infection initiated by overwintered oospores. Each germinating oospore produces a **macrosporangium** that remains attached to the oosporic germ tube near the soil surface and releases zoospores that reach susceptible tissues through rain splash (Rossi and Caffi, 2012).

Sporulating lesions produce detachable asexual sporangia that may become airborne. These sporangia release zoospores on wet host surfaces and drive lesion-derived infection cycles (Brischetto et al., 2020, 2021). Although Rossi used **microsporangia** in correspondence to distinguish these propagules from oospore-derived macrosporangia, the peer-reviewed literature generally uses **sporangia**, **airborne sporangia**, or **asexual sporangia**.

### Inoculum hypothesis

Asexual sporangial inoculum is hypothesized to make an important, and potentially predominant, contribution to early-season airborne inoculum at the experimental sites because: (1) airborne sporangia were captured before the first observed lesions; (2) oospore-derived macrosporangia generally remain attached near the soil surface rather than entering the airborne spore pool; and (3) warm, humid subtropical conditions favour recurrent asexual cycles, while green tissue retained in asynchronously managed surrounding vineyards may replenish the regional airborne spore pool. Predominantly asexual reproduction or limited oospore contribution has been reported in subtropical Sao Paulo, South Africa, and most surveyed Western Australian vineyards (Camargo et al., 2019; Koopman et al., 2007; Killigrew et al., 2005).

This hypothesis does not exclude oospore-derived infections. Oospore formation has been documented in highland southern Brazil (Bitencourt et al., 2021); consequently, the relevant premise is that an exclusively oospore-driven seasonal reset cannot be assumed in the Guangxi inoculum landscape.

### Confirmed spore-observation semantics and correspondence (2026-09-16)

The researcher confirms that the CK workbook records accumulated asexual sporangium counts over collection intervals. Nonblank zero values are valid zero-count collections; intermediate NaNs are dates without independent collection results, not zero counts. Devices were located between vineyard rows in an area without fungicide applications. Each device was activated three days before its first recorded collection: represent the first interval as `(t_first - 3 days, t_first]`, with duration three days; subsequent intervals are bounded by consecutive nonblank collection dates. Include the first interval in exposure-duration summaries. Do not treat its start as unknown or request further equipment metadata as a prerequisite for the present descriptive analysis or manuscript rationale.

The researcher supplied an email excerpt from Vittorio Rossi on 2026-09-16 (the email's original date was not supplied). Rossi described experience in Brazil and India where pruning-induced production cycles and vineyards with old and new vegetation coexist without a true physiological winter stop, allowing airborne sporangia from not-yet-pruned vineyards to be present at bud break elsewhere. He explained that an exclusively oospore-based starting condition does not represent that setting and discussed the relevance of asexual-sporangium infection processes. This personal communication is an explicit conceptual basis for the regional recurrent-asexual-inoculum hypothesis, together with the pre-symptom captures and published literature. It is not a published finding of Rossi et al. (2008).

Preserve the conditional nature of the email: Rossi asked whether the Guangxi vineyards share those production conditions; the excerpt does not itself establish local winter dormancy, the composition of surrounding vineyards, or the source of each captured sporangium. Do not generalize its "biologically impossible" wording to all temperate vineyards or use it to exclude oospore-derived infection throughout subtropical viticulture. Retain `microsporangia` in the archived quotation, but use `airborne sporangia` or `asexual sporangia` in the manuscript. Distinguish the researcher-confirmed propagule identity from the still-hypothesized regional source. Maintain the excerpt and its interpretation in `SPORE_OBSERVATION_NOTES.md`; routine supplementary metadata are optional context, not a new blocking checklist.

### Model rationale

FSIM combines the cohort-based infection structure of the Rossi primary-infection model with the sporulation, dispersal, deposition, and secondary-infection processes represented by Brischetto et al. (2021). It reconstructs the biological sequence from asexual sporangium production and loss through dispersal, deposition, and zoospore release to infection, incubation, and symptom appearance under a subtropical inoculum regime. Rossi's model provides a mechanistic foundation rather than a performance comparator.

Each sporulation event forms an independent sporangium cohort. Within the attached and detached states, the hourly VPD-driven mortality rates of Brischetto et al. (2020, 2021) are accumulated for each cohort, and viable dose is calculated as the dose entering that state multiplied by one minus cumulative mortality. Detached-state mortality continues without resetting from aerial transport through deposition; a cohort ceases to be viable when cumulative mortality reaches one. Do not introduce or calibrate a separate minimum-spore cutoff.

FSIM-S extends FSIM by incorporating a background spore-pressure field derived from field measurements. The comparison between FSIM and FSIM-S evaluates the incremental contribution of observed airborne inoculum information within the same modelling framework.

## Research questions

1. How accurately can a process-based model representing airborne sporangia as early-season inoculum estimate the timing of grapevine downy mildew onset in subtropical vineyards?
2. To what extent does incorporating field-observed airborne spore pressure improve disease-onset estimation relative to the base process model without spore observations?

### Updated positioning and inferential scope

The manuscript is positioned as a **mechanistic, spore-informed reconstruction and retrospective evaluation**, not as a validated operational forecasting or spray-reduction programme. The central contribution is to link the cohort structure of Rossi et al. (2008) with the airborne-sporangium processes represented by Brischetto et al. (2020, 2021), under the hypothesis that a regional asexual inoculum reservoir is continuously present or repeatedly replenished in the subtropical production landscape. This premise concerns regional inoculum availability; it does not assert that individual sporangia survive the entire off-season, exclude oospore-derived infection, or identify the source of every captured sporangium.

The primary scientific comparison is FSIM versus FSIM-S within the same process framework. Rossi is a mechanistic foundation and biological reference, not a performance baseline. The principal evidence is the paired, out-of-fold comparison at the site-year level (`n = 6`). Claims of cultivar transferability, geographical generality, prospective forecast skill, maintained disease control, pesticide reduction, economic benefit, or operational decision-support readiness require new independent data and prospective field trials.

The two research questions are interpreted as follows: (1) whether a biologically constrained FSIM can reproduce the interval-censored timing of first visible seasonal symptoms; and (2) whether an observation-derived airborne spore-pressure field provides incremental retrospective information beyond the same weather-driven process model. Robustness and uncertainty support these questions; regional extrapolation and fungicide-saving estimation are outside the current scope.

## Model and endpoint definitions

| Concept | Standard term | Definition |
|---|---|---|
| Base model | First Seasonal Infection Model (FSIM) | Process-based model of the first seasonal infection sequence under recurrent airborne asexual inoculum |
| Spore-informed model | FSIM-S | FSIM extension incorporating an observation-derived spore-pressure field |
| Biological event | first seasonal infection | Earliest infection event affecting susceptible tissue in a new growth or production cycle, independent of an assumed sexual inoculum source |
| Observed endpoint | observed disease-onset date | Date of the first confirmed visible downy mildew symptoms or oil spots under the surveillance protocol |
| Model output | predicted disease-onset window | Period during which the first seasonal symptoms are estimated to appear |
| Management period | infection-risk period | Period during which infection is estimated to be possible or likely |
| Intervention period | protection window | Earlier period during which preventive treatment could be applied under a defined decision rule |
| Phenology | BBCH 53: inflorescences clearly visible | Phenological stage distinct from flowering |

FSIM simulates infection and incubation, but calibration and error calculation compare the **predicted disease-onset window** with the **observed disease-onset date**. The study therefore estimates disease onset retrospectively; operational forecasting requires prospectively available inputs and temporal validation.

## Abstract and Introduction wording

The Abstract and Introduction should introduce these distinctions where they support the scientific argument rather than presenting them as a standalone terminology exercise. The preferred Introduction sequence is the management need, empirical timing systems, the Rossi primary-infection pathway, the Brischetto secondary-infection pathway, the complementary scope and starting conditions of these temperate-region process models, the regional inoculum hypothesis, the FSIM reconstruction, and the study objectives. Avoid metadiscursive or signposting statements when the biological and management logic can provide the transition directly. The Introduction need not foreground the inability to observe infection; observation and calibration details belong in Materials and Methods unless they are necessary to explain a specific model output. Do not claim that the earlier models are generally unsuitable for humid subtropical viticulture; state only that neither starting condition directly represents the study situation in which airborne sporangia were present before local lesions.

Keep the opening paragraph focused on the production and management need. At first mention, use the binomial **_Plasmopara viticola_** without taxonomic authorities; this is sufficient for the epidemiological and modelling scope and keeps the opening sentence concise. Describe humid subtropical conditions as coinciding with the emergence of susceptible leaves and development of inflorescences rather than defining both organs by a single susceptible phenological stage. Use **depends heavily on repeated preventive fungicide applications** rather than unquantified wording about large fungicide inputs. When discussing conventional timing, use **may be poorly aligned with infection risk** and avoid claiming that protection was not improved unless a direct field comparison supports that outcome.

Describe the ``3--10 rule'' and EPI as empirical models that use weather and phenological indicators to guide downy mildew control timing. State that they do not explicitly resolve the complete sequence of inoculum maturation and availability, dispersal, infection, and incubation. Describe the value of mechanistic, process-based models as simulating the biological sequence from inoculum development to symptom appearance and providing a basis for aligning fungicide applications with infection risk; do not claim that process representation necessarily improves decision accuracy without direct comparative evidence.

### Rossi and Brischetto frameworks

Describe Rossi et al. (2008) as a cohort-based mechanistic foundation for oospore-driven primary infection. State that the model follows oospore maturation and germination, zoospore release, dispersal, infection, incubation, and the predicted appearance of oil spots. When the inoculum pathway matters, state that a germinating oospore produces a macrosporangium attached near the soil surface and that its zoospores reach susceptible tissue mainly through rain splash (Rossi and Caffi, 2012). Note that modelled infection is latent and that field evaluation used the first observed seasonal symptoms; do not describe infection as directly observed.

Describe Brischetto et al. (2020, 2021) as the complementary basis for secondary infection cycles: detachable asexual sporangia produced on lesions enter the air, survive and disperse according to weather, deposit on susceptible surfaces, and release zoospores under wet conditions. Brischetto et al. (2020) related the abundance of airborne sporangia, particularly estimated viable sporangia, to secondary-infection potential in a northern Italian vineyard. Brischetto et al. (2021) begins from an existing visible lesion and predicts periods favourable for secondary infection. In the Introduction, do not foreground relative-severity or disease-progress outputs because they fall outside this study's endpoint.

Present the Rossi and Brischetto frameworks as complementary process models developed for temperate viticulture: Rossi et al. represent oospore-driven primary infections, whereas Brischetto et al. represent secondary infection cycles initiated by asexual sporangia produced on existing lesions. Do not imply that primary and secondary infections form strictly separated seasonal phases; both may contribute during the growing season, with relative importance varying among regions and years. FSIM links and reorganizes these established components under the recurrent-airborne-inoculum premise; it should not be described as discovering either component.

The Rossi and Brischetto accounts may be combined into one Introduction paragraph when presented as a single review of complementary process-model development in temperate viticulture. Use **sporangia produced on downy mildew lesions** without the redundant modifier **sporulating**, and do not call these propagules **microsporangia** or **small sporangia**. State that Brischetto et al. (2020) associated the **estimated abundance of viable airborne sporangia** with infection potential, rather than claiming that viability was directly measured or that causality was demonstrated. Describe the two model outputs at the level relevant here: Rossi predicts the oil-spot appearance window, whereas Brischetto predicts periods favourable for secondary infection. Omit relative severity because this study does not evaluate disease incidence or severity.

### Regional inoculum premise

Use the following level of inference: **"Asexual sporangial inoculum is hypothesized to make a substantial and potentially predominant contribution to early-season airborne inoculum in the study system."** Pre-symptom airborne captures support this hypothesis but do not identify the source of every sporangium or infection. In the Introduction, this qualification need not be repeated as a defensive closing statement when the claim is already framed explicitly as a hypothesis; preserve the broader scope elsewhere in the manuscript that oospore-derived infections are not excluded.

In the Abstract, use **airborne sporangia** for the measured propagules and **asexual sporangial inoculum** for the inferred inoculum class. When explaining the biological source in the Introduction, prefer the explicit formulation **sporangia produced on downy mildew lesions** rather than the compressed modifier **lesion-borne**. Use **macrosporangium** only when explaining the oospore pathway. Avoid **microsporangia** as a main manuscript term; if the correspondence-based distinction is indispensable, define it locally rather than presenting it as standard published nomenclature.

### FSIM contribution and outputs

Use this core description: **"FSIM links the cohort structure of the Rossi primary-infection model with the production, dispersal, deposition, and infection processes of the Brischetto secondary-infection model under a recurrent-airborne-inoculum premise. The resulting cohort sequence extends from potential production and loss of asexual sporangia through dispersal, deposition, zoospore release, infection, and incubation to symptom appearance."** This is an epidemiologically motivated reconstruction, not a claim that either the primary- or secondary-infection component is newly discovered. Rossi's model is the mechanistic foundation, not a performance baseline.

Use **first seasonal infection** chronologically for the earliest infection event affecting susceptible tissue in a new growth cycle, without assigning the event a sexual or asexual origin in the term itself. When describing model operation, it is acceptable to state that FSIM represents airborne sporangia as the early-season inoculum under the study hypothesis. Do not define first seasonal infection as necessarily driven by asexual inoculum.

Use **predicted disease-onset window** for the model output evaluated in the paper and **observed disease-onset date** for the first confirmed visible symptoms. In the Introduction, state this output directly rather than inserting a standalone explanation about the observability of infection. In Materials and Methods, clarify that infection time and incubation duration are internal model states rather than directly observed calibration targets.

Describe FSIM-S as retaining the FSIM process chain while using an observation-derived, site-year-specific spore-pressure field to adjust the effective infection threshold. The field represents relative background airborne inoculum pressure, not an absolute viable-spore count. The FSIM versus FSIM-S comparison tests the incremental value of observed spore information within the same process framework.

### Introduction ending

Keep the final two Introduction paragraphs conceptual rather than procedural. The first should define FSIM, identify its Rossi and Brischetto foundations, state the recurrent-airborne-inoculum setting, and name the predicted disease-onset window. The second should define FSIM-S at a high level as incorporating an **observation-derived airborne spore-pressure field**, state the scientific question concerning its incremental value, identify the disease-onset comparison as retrospective, and end with the significance for risk-informed preventive management. Leave process-chain details, threshold modification, calibration and validation design to Materials and Methods.

### Management claims

Distinguish the evaluated symptom endpoint from the earlier period relevant to preventive treatment. Use **infection-risk period** for when infection is modelled as possible or likely and **protection window** only for an earlier intervention period defined by a management rule. Do not report regional spray-saving scenarios or numbers of potentially avoidable applications. Keep preventive-timing relevance conceptual and distinguish it from demonstrated intervention outcomes. Do not state that the study demonstrated pesticide reduction, maintained disease control, economic benefit, or operational decision-support readiness.

### Abstract conclusion

Use the structured Abstract conclusion to state the specific modelling gap addressed and the resulting scientific or pest-management significance. Do not repeat numerical results already given under RESULTS, and do not end the Abstract with defensive limitations. The preferred emphasis is that FSIM-S links airborne inoculum monitoring with mechanistic infection processes and translates spore and weather information into disease-onset and infection-risk windows that can support better-timed preventive programmes. Avoid broad priority claims such as **the first model for subtropical vineyards** unless a systematic literature assessment establishes both the comparison set and the geographical definition; prefer **fills a key modelling gap for humid subtropical viticulture**.

## Spore-pressure field

The spore-pressure field is a continuous, site-year-specific representation of background airborne inoculum pressure derived from in situ spore counts. Its construction comprises measured-period selection, daily indexing and interpolation, `log1p` transformation, scaling by the site-year 95th percentile, clipping to `[0,1]`, smoothing, and centering on a defined reference level.

Positive and negative field values denote deviations above and below the reference level; they are not positive or negative spore counts. FSIM-S uses this field to modify the effective infection threshold. Raw spore counts, infection-event dates, and incubation duration are not direct disease-onset calibration targets.

## Analytical scope

| Element | Study specification |
|---|---|
| Evidence base | Six Guangxi site-years; no external disease or spore datasets |
| Independent unit | Site-year (`n = 6`) |
| Design limitation | Site and cultivar are confounded; cultivar and geographical transferability remain untested |
| Timing error | Zero within the predicted window and otherwise the distance to its nearest boundary |
| Observation uncertainty | Interval between the last disease-negative and first disease-positive assessments where records permit |
| Parameterization | Biological constraints, sensitivity analysis, and explicit treatment of equifinality and uncertainty |
| Validation | **Primary:** grouped leave-one-site-year-out evaluation, leaving one complete site-year out at a time (`n = 6` folds). Secondary stress tests may include leave-one-site-out and leave-one-year-out, but they are not the primary reported evaluation. All preprocessing, parameter estimation, and model selection are confined to the training site-years. |
| Model comparison | Paired site-year differences between FSIM and FSIM-S, including missing predictions and failure cases |
| Robustness | Sensitivity to phenological gating, weather-source mismatch, and temporally non-causal data transformations |

## Pest-management significance

Symptom onset is the evaluation endpoint, whereas preventive management depends on an earlier infection-risk period. The current study links observed inoculum information with mechanistic disease-onset reconstruction. Its management significance is the biological basis for future risk-informed timing studies. It does not quantify potentially avoidable applications or establish disease control, pesticide reduction, economic benefit or operational readiness. Those outcomes require prospective comparison of conventional, model-guided and untreated programmes.

## Claim scope

| Supported interpretation | Unsupported extension |
|---|---|
| Paired retrospective evaluation tests the incremental information supplied by spore monitoring; improvement claims require completed revised results | General superiority across cultivars or production regions |
| FSIM is an epidemiologically motivated reconstruction of established mechanistic processes | Absence or irrelevance of oospores throughout subtropical viticulture |
| Inoculum-informed reconstruction motivates further preventive-timing studies | Quantified spray savings or demonstrated pesticide reduction with maintained crop protection |
| The framework has potential for risk-informed fungicide timing | Operational decision-support readiness |

## Manuscript architecture

| Section | Scientific content |
|---|---|
| Title | Spore-informed process-based modelling of grapevine downy mildew onset in subtropical production |
| Abstract | Management problem, epidemiological rationale, quantitative model comparison, and bounded management significance |
| Introduction | Management need, Rossi framework, regional inoculum distinction, spore-monitoring hypothesis, and research questions |
| Materials and Methods | Field observations, FSIM and FSIM-S processes, spore-pressure construction, calibration endpoint, validation, and uncertainty |
| Results | Inoculum observations, disease onset, model evaluation, incremental value of spore pressure, and robustness |
| Discussion | Epidemiological interpretation, relationship to previous models, management relevance, uncertainty, and transferability |
| Conclusion | Supported mechanistic contribution and conditional value for fungicide timing |

## Supporting literature

| Topic | Reference |
|---|---|
| Oospore-driven mechanistic model | Rossi et al. (2008), doi:10.1016/j.ecolmodel.2007.10.046 |
| Rain-splash dispersal of oospore-derived inoculum | Rossi and Caffi (2012), doi:10.1094/PHYTO-08-11-0223 |
| Airborne sporangia and secondary-infection potential | Brischetto et al. (2020), doi:10.3389/fpls.2020.01187 |
| Lesion-borne infection model | Brischetto et al. (2021), doi:10.3389/fpls.2021.636607 |
| Predominantly asexual population in subtropical Brazil | Camargo et al. (2019), doi:10.1111/ppa.12993 |
| Asexual persistence in South Africa | Koopman et al. (2007), doi:10.1111/j.1364-3703.2007.00429.x |
| Limited oospore occurrence in Western Australia | Killigrew et al. (2005), doi:10.1094/PD-89-0777C |
| Oospore formation in highland southern Brazil | Bitencourt et al. (2021), doi:10.1111/ppa.13431 |

## Planned Results organization

The next substantive Results revision should use: (1) field-observed airborne sporangia and disease onset; (2) retrospective evaluation of disease-onset windows; (3) incremental contribution of observed spore pressure; and (4) robustness and uncertainty of onset estimates. Report paired site-year predictions and failures, and keep full sensitivity rankings in Supporting Information where appropriate. This organization is a plan; it was not implemented in the limited 2026-09-13 text revision.

## FSIM-S implementation status (2026-09-14)

FSIM-S must remain a strict nested extension of FSIM. Its controller and model inherit the FSIM production, dispersal, deposition, mortality, leaf-moisture, BBCH15, incubation and cohort-collapse processes. The active scientific difference is injection of an observation-derived background dose through an external provider, scaled by the single parameter `spore.beta_sp`; this dose enters the cohort pathway before dispersal and then follows the same mortality, deposition and infection processes as endogenous cohorts. Legacy `alpha_sp`, `alpha_air`, `alpha_dep`, `alpha_pos` and `alpha_neg` adjustment paths are retired. A zero-pressure FSIM-S run must reproduce FSIM exactly. Do not reintroduce separate legacy FSIM-S process branches without an explicit structural rationale and matched tests.
