# Concise FSIM conceptual diagram

Format: 180 mm wide × 164 mm high; 8 pt text; PNG at 600 dpi, vector PDF and SVG. The manuscript Figure 6 has not been replaced.

Suggested caption: Conceptual structure of FSIM. Green boxes indicate drivers and derived variables, ochre boxes represent processes, blue boxes represent cohort states or event-time outputs, and rose boxes describe survival constraints. Solid arrows show the simulation sequence and dashed arrows show driving relationships. Potential sporulation is followed by transfer and merged deposition, zoospore release, infection and incubation, leading to the predicted oil spot window. The diagram omits output aggregation and evaluation details for clarity.

Variables: T, temperature; RH, relative humidity; MP, moist-night duration; TMP, mean temperature during moist-night hours; VPD, physical vapour pressure deficit used for the leaf-moisture proxy; WD, wet duration; TWD, mean temperature during the relevant wet period. Temperature and RH also drive a distinct mortality index. BBCH15 gates infection accumulation. Optional leaf-wetness measurements contribute to the night-time moisture summary. The wind gate is optional.

The current baseline FSIM is shown; proposed spore-data extensions are not included. Transfer and deposition share a timestamp. Detached mortality is not reset at deposition. Zoospore survival is represented by a time window rather than an explicit population count. The final output summarizes eligible cohort windows; omitting the aggregation box does not change the model.

Source correspondence: model/sporulation_timing.py (night drivers and production); model/first_infection.py (cohort states, mortality, release, infection, incubation); model/process_rules.py (leaf-moisture proxy); model/crop_season.py (host gate and cohort aggregation). File hashes and figure dimensions are recorded in the adjacent JSON.

Regenerate: python src/plot_fsim_conceptual_diagram.py

## Cohort interruption branches

Rightward solid branches now show failure or termination conditions: zero production (no cohort created), complete attached-state loss before transfer, complete detached-state loss before release, no release in the available viable interval, first dry condition before infection, and no eligible infection within the zoospore survival window. Mortality remains cumulative and driven by the temperature-RH index.

The green incubation branch denotes incomplete output at the end of the weather record, not biological death. Missing release or infection at a record boundary also does not establish biological absence beyond the available data. The diagram reports the current simulation outcome.

A temporarily unmet wind gate retains the attached cohort rather than terminating it. BBCH15 not yet reached does not by itself terminate a cohort; infection may still occur if the host becomes eligible before zoospore viability ends. Once infection has occurred, later dry conditions do not cancel incubation. These distinctions prevent interpreting every temporarily unmet condition as immediate cohort death.

Updated arrow interpretation: central solid arrows show progression; rightward solid arrows show failure/termination branches; dashed arrows show environmental drivers. The figure remains 180 × 164 mm, with no title or footnotes.

## Final label simplification

The terminal state is labelled `6. Oil spot`; it represents the same modelled symptom-appearance window, not an added process or a scalar date. Failure branches now show compact conditions. Mortality ≥ 1 denotes cumulative mortality, not an hourly rate. The attached condition applies before transfer; the detached condition applies before release. No new minimum-dose or Num cutoff has been added to either the model or the diagram. Infection expiry refers to the existing zoospore survival window, including host eligibility. Incubation incomplete at data end remains an output limitation rather than mortality.

## Branch label layout

Failure conditions are placed above rightward connecting arrows, with terminal boxes labelled No cohort, Terminated, or No event. The data-end branch points to Incomplete output. Branch text is 7 pt; other text remains 8 pt at 180 mm figure width. In the detached branch, mortality means cumulative mortality. No model rules were changed.

The incubation data-end branch has been removed from the figure at the user's request. Incubation now connects only to 6. Oil spot; this is a visual simplification, not a change to model handling of incomplete weather records.
