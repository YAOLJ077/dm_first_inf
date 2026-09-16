# Brischetto-to-FSIM process audit

## Central finding

FSIM is not a direct implementation of Brischetto et al. (2020, 2021). It is a new model assembled from selected biological relationships in the Brischetto and Rossi frameworks plus several locally introduced structural assumptions. That distinction is scientifically acceptable, but each new assumption must be identified and evaluated independently. The current large release and deposition thresholds should not be described as values inherited from Brischetto.

## What the two Brischetto studies established

Brischetto et al. (2020) was an observational inoculum study rather than a complete disease-onset model. Daily airborne-sporangium counts were converted to an estimated viable-sporangium dose using VPD-dependent mortality equations for attached and detached sporangia. Viable spore abundance was related to infections detected by collecting asymptomatic leaves and incubating them under favourable laboratory conditions. The reported threshold of 2.52 viable sporangia m-3 air day-1 supported negative prognosis in that vineyard; it is not a universal infection threshold and is not directly transferable to a normalized spore-pressure index.

Brischetto et al. (2021) starts from visible downy mildew lesions. A model run is initiated on each day satisfying the sporulation switch: at least 3 h of night moisture, a mean temperature of the moist period between 10 and 30 degrees C, and moist hours defined by RH at least 80%, rain greater than 0, or leaf wetness greater than 30 min. The model then calculates relative sporangial production, VPD-dependent loss, availability on healthy host surfaces, and infection during suitable wet periods. Dispersal and deposition are deliberately collapsed into `D&DR = 1`; the paper does not resolve rain release, wind release, dry deposition, or wet deposition as separate events.

## Process mapping

| Process | Brischetto basis | Current FSIM | Audit judgement |
|---|---|---|---|
| Source state | Known visible lesions capable of sporulation | An unobserved regional source pool is implicitly available from 1 January | Necessary regional extension, not inherited from Brischetto |
| Plausible regional sources | Not addressed | Surrounding green infected tissue, imported airborne inoculum, litter, and previous-season material have been discussed together | Green infected tissue and airborne import are defensible hypotheses; litter normally represents an oospore reservoir, and dead canes cannot be asserted to produce asexual sporangia without evidence |
| Sporulation switch | Night MP at least 3 h; TMP 10-30 degrees C; moist hour if RH at least 80%, rain, or measured leaf wetness | The night is now grouped from 20:00 to 07:59 by its ending date; moisture uses RH at least 80% or rain, plus leaf wetness longer than 30 min when observed | Timing and available moisture inputs now follow the published switch; fixed clock hours approximate darkness |
| Sporangial production | Lalancette response, rescaled to a relative dose and modified by repeated sporulation history | Triangular temperature response multiplied by `1-exp(-0.2 MP)` and added to one persistent pool | The formula and accumulation rule require an independent derivation; repeated-source exhaustion is absent |
| Attached mortality | `MOR' = 1/[24(9.27-1.12 VPD+0.04 VPD^2)]` each hour | The reciprocal equation is now applied every hour | Corrected to the published relationship; historical fitted parameters require re-estimation |
| Dispersal and deposition | Combined relative transfer, `D&DR = 1`, whenever viable sporangia are available | Entire pool is released when hourly rain or mean wind crosses a threshold; pool is reset to zero | New all-or-none event mechanism; thresholds are not from Brischetto |
| Wind process | Wind is discussed as a release mechanism, but no field threshold is used | Hourly mean 10-m wind is treated as a trigger | Mean wind is not a gust; explicit gust modelling requires gust observations |
| Rain process | Rain may be associated with release; no release threshold is fitted | 10.79 mm in one hour triggers full-pool release | Extreme fitted filter rather than an established biological threshold; call it rain-associated release, not necessarily splash dispersal |
| Deposition | Not separated from dispersal | Rain release deposits after 1 h; wind release waits for rainfall at least 6.45 mm in one hour | Future heavy rain is unsupported as a necessary deposition condition and duplicates later surface-moisture logic |
| Detached mortality | `MOR'' = 1/[24(5.67-0.47 VPD+0.02 VPD^2)]` each hour | The reciprocal equation is now applied every hour before and after deposition | Mortality is corrected; airborne transport delay remains an FSIM extension |
| Infection moisture | Measured leaf wetness; interruptions no longer than 1 h are allowed | Rain above 5.04 mm per hour or VPD at most 1.92 hPa; uninterrupted groups only | Not equivalent to the published wet-period definition and requires separate audit |
| Infection response | Magarey minimum wetness and Caffi relative infection severity | Rossi-style zoospore release plus `WD x TWD` threshold and Rossi incubation | A defensible hybrid only if explicitly described as reconstruction rather than retention of the Brischetto infection module |
| Endpoint | Presence/absence of viable spores and secondary infection periods or latent infection days | Earliest post-BBCH15 predicted symptom window | New endpoint translation; first feasible cohort is not necessarily first field-detectable disease |

## Why the current thresholds appear useful

The implicit regional pool can generate a new relative dose on every suitable night, and the current model lacks the lesion productivity decline used by Brischetto. Every sufficiently large cohort can then proceed toward infection, and the earliest completed cohort is selected as the onset estimate. Under this construction, early feasible cohorts are expected. The extreme release and deposition thresholds suppress many of them, while the former mortality implementation further shortened cohort survival. Their historical ability to delay onset was therefore compensatory and cannot establish biological validity.

Using the reciprocal hourly mortality equations across all hours with the current release and deposition thresholds increased mean deposited cohorts from 11.3 to 23.3 per site-year and worsened the mean absolute midpoint error from 17.18 to 34.36 d. Under broad-condition transfer, correcting mortality changed cohort survival but not the already very early first window. These results confirm that onset agreement cannot identify the dispersal, deposition, and mortality processes separately.

## Recommended regional formulation

Represent the source hypothesis as a **persistent regional inoculum reservoir**, not as confirmed continuous production from litter or dead canes. The reservoir represents the potential supply of asexual sporangia from asynchronously maintained infected green tissue and airborne immigration; local monitoring provides evidence of its presence but not the contribution of each source.

For the primary analysis, retain the Brischetto simplification that viable sporangia can become available on host surfaces without fitted hard release and deposition thresholds. Apply the published mortality equation correctly and keep deposition conceptually distinct from the free-water requirement for zoospore release and infection. If explicit transport is retained, it should be a separately labelled sensitivity model using fractional release and deposition rates, not an all-pool threshold switch. The present data cannot identify rain release, gust release, airborne survival, and rain deposition as four separate processes from six onset intervals.

Translate inoculum availability into observable onset through dose and infection efficiency. A biologically coherent structure is:

```text
regional source availability
    -> weather-dependent sporangial production
    -> viable sporangial dose on susceptible host tissue
    -> wetness- and temperature-dependent infection efficiency
    -> cohort incubation
    -> accumulated lesion expectation or detection probability
    -> predicted disease-onset window
```

FSIM can use a normalized background source level, whereas FSIM-S can scale or replace the viable-dose term with the observed spore-pressure field. Spore pressure should preferably modify available inoculum dose or infection probability rather than alter the physiological `WD x TWD` requirement itself. With only six independent site-years, use one shared dose-to-detection parameter at most, and evaluate it exclusively within grouped training folds.

## Explicit-equation check after the mortality correction

The VPD calculation is on the hPa scale required by the Brischetto mortality coefficients. The Rossi zoospore-release expression, infection threshold form, and two incubation-rate equations are transcribed consistently with Rossi et al. (2008). No second inverted or parenthesized equation was identified in these components.

The remaining differences from Brischetto et al. (2021) are structural rather than transcription errors. FSIM does not currently implement the Lalancette sporulation response surface or the Kennelly repeated-sporulation modifier; its explicit release and deposition thresholds replace `D&DR = 1`; and its downstream infection and incubation sequence is inherited from Rossi rather than the Magarey-Caffi secondary-infection component. The former calendar-day night aggregation and RH-only moisture definition have now been corrected. Current inputs contain no measured leaf-wetness duration, so the moisture switch presently uses the published RH-or-rain alternatives.

## Sporulation-rate difference and transferability

Lalancette et al. (1988) described sporangia per unit lesion area with a Richards response surface, `S = k[1-exp(B+rho)]^(1/(1-m))`, with `m = 0.5`. Its asymptote `k`, intercept `B`, and duration term `rho = -rH` are quadratic functions of temperature, while `H` is the duration of high humidity. Brischetto et al. (2021) rescaled that surface by the predicted dose at 20 degrees C and 24 h, producing a relative `SPOR` rate between zero and one. The resulting response is strongly nonlinear: production is near its lower boundary around 10 and 30 degrees C, peaks near 20 degrees C, and rises sharply only after several hours of nocturnal moisture. Brischetto then reduces the potential dose according to the number of previous sporulation events using Kennelly et al. (2007), whose empirical relationship was `ln(relative sporulation (%) + 1) = 4.757 - 0.496 X`, where `X` is the event number.

FSIM instead calculates `fT * [1-exp(-0.2 MP)]` and adds the resulting relative dose to a common persistent pool. This differs in three ways. First, the temperature response is not the empirical Richards surface. Although `fT` equals one at the configured `Topt`, it can exceed one between `Tmin` and `Tmax` whenever `Topt` is not their midpoint; with the present parameters its mathematical maximum is near 20 degrees C rather than the configured 24.3 degrees C, after which clipping conceals the excess. Second, the moisture response is already about 0.70 after 6 h and 0.80 after 8 h, whereas the Lalancette experiments showed little macroscopic production before approximately 7.5 h and about 50% of the 20 degrees C response after 8 h. Third, FSIM does not reduce the productivity of a source after repeated sporulation.

The normalized Lalancette weather response is biologically relevant to the Guangxi scenario because it describes asexual sporangia produced on infected grape leaves, the same propagule class represented by FSIM. It is therefore a stronger candidate than the current unreferenced temperature-moisture dose function. Its absolute output cannot be transferred because FSIM has no observed lesion area. Transfer of even the normalized response remains uncertain because Lalancette used infected `Vitis labrusca` 'Catawba' leaves under controlled conditions, the fitted observations covered 6-12 h of high humidity, and weather measured at the experimental vineyard may differ from conditions at unknown regional source sites.

Directly importing the complete Brischetto production compartment is not justified because its state is a known cohort of local lesions with a countable sporulation history. FSIM's regional reservoir aggregates unknown, differently aged, and continually replenished sources, so a single `SPOn` cannot be assigned from the available observations. The defensible next comparison is therefore: (i) retain the recurrent regional reservoir; (ii) replace only its weather-dependent potential production rate with the normalized Lalancette surface; and (iii) omit the Kennelly exhaustion term in the primary model or represent source renewal and exhaustion with one separately stated turnover parameter. The output should be interpreted as potential relative production or availability, not an absolute spore count. Applying the Kennelly event-number decline directly to the entire regional pool would falsely treat all sources as one ageing lesion cohort.

## Evidence status

| Claim | Evidence | Status |
|---|---|---|
| Airborne sporangia can persist under rainless field conditions | Caffi et al. (2013); Brischetto et al. (2020) | Supported |
| Viable airborne-sporangium abundance informs negative infection prognosis | Brischetto et al. (2020), single-vineyard evaluation | Supported but context-specific |
| Dispersal and deposition can be abstracted as a relative transfer | Brischetto et al. (2021) | Supported as a modelling simplification, not a measured physical rate |
| A persistent regional reservoir exists in the Guangxi study landscape | Pre-onset airborne-sporangium observations and production context | Plausible study hypothesis |
| Litter and dead canes continuously produce asexual sporangia | No direct evidence identified | Unsupported as currently phrased |
| Hourly rain and wind thresholds identify release and deposition | No supporting calibration data for these sub-processes | Unsupported |
| Earliest biologically feasible infection yields the first detectable lesion | Not established by Brischetto and not independently observed | Requires a dose/detection assumption |

## References

- Brischetto C, Bove F, Languasco L, Rossi V. Can spore sampler data be used to predict *Plasmopara viticola* infection in vineyards? *Frontiers in Plant Science*. 2020;11:1187. https://doi.org/10.3389/fpls.2020.01187
- Brischetto C, Bove F, Fedele G, Rossi V. A weather-driven model for predicting infections of grapevines by sporangia of *Plasmopara viticola*. *Frontiers in Plant Science*. 2021;12:636607. https://doi.org/10.3389/fpls.2021.636607
- Caffi T, Gilardi G, Monchiero M, Rossi V. Production and release of asexual sporangia in *Plasmopara viticola*. *Phytopathology*. 2013;103:64-73. https://doi.org/10.1094/PHYTO-04-12-0082-R

## Adversarial self-review

| Dimension | Question | Assessment |
|---|---|---|
| Contribution | Is the region-adapted element distinguished from prior models? | Yes; the persistent regional reservoir and disease-onset endpoint are identified as new assumptions |
| Clarity | Are production, release, deposition, infection, and detection kept separate? | Yes; terms are mapped process by process |
| Experimental strength | Can six onset intervals identify transport sub-processes? | No; the report explicitly limits inference and recommends no threshold fitting |
| Evaluation completeness | Have structural and equation-level discrepancies been tested? | Partly; fixed scenarios and mortality equations were tested, while source-dose and wetness formulations remain the next audit |
| Method soundness | Does the best-fitting current scenario establish biological validity? | No; the report treats its lower error as compensation rather than validation |
