# Phenology model

Regional simulations use separate Table and Wine thermal-time profiles:

- `profile_config_table.json`: Table profile.
- `profile_config_vine.json`: Wine profile (`vine` is the retained data key).
- `profile_configs.json`: profile index and selection criteria.
- `model.py`: daily temperature response and sequential stage simulation.
- `profile_calibration_core.py`: shared fitting, prediction and scoring functions.
- `susceptibility_modulation.py` and `.json`: conversion of phenological anchors
  into host susceptibility for the hourly disease model.

The calibration entry point is
[`src/calibrate_phenology_profiles.py`](../../src/calibrate_phenology_profiles.py).
It uses local observations and daily temperatures from
`data/phenology_data/calibration_data/` and writes to
`result/phenology_calibration/`. Literature-based regional calibration files
are no longer part of the active workflow. The calibration retains the
accepted site selection, assumed Nanning BBCH09 dates and candidate model
families; the base-temperature candidates remain capped at 10 C.

The runtime profiles accumulate linear growing degree days from their
configured start day. Incremental thresholds are applied in order to
BBCH09, BBCH65 and BBCH81, resetting accumulation after each stage. BBCH53
uses the fraction defined by
`susceptibility_modulation.json` under `auxiliary_stage_rules.53` when no
observed anchor is available.

`observed_anchor_fallback.json` preserves the thermal-time configuration
used by the six observed site-years. Observed stage dates take precedence;
this configuration supplies missing stages and the thermal progression
between anchors. Its numerical values were retained during cleanup to
preserve the accepted disease-model results. Regional Table/Wine runs
explicitly supply their corresponding profile configuration instead.
