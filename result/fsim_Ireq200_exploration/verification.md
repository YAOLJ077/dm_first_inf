# Verification — 2026-09-13

- Full test suite: 41 tests passed. The two range-comparison tests were also rerun after reporting updates and passed.
- `git diff --check`: passed.
- Expanded design: 1,620 grid combinations plus two unchanged historical-default controls; 9,732 unique main site-year records, with six explicit records per parameter set and zero runtime errors.
- All 3,792 records shared with the baseline run reproduce exactly, including observed intervals, prediction status, predicted windows, number of candidate windows and error metrics.
- Input JSON and evaluated model/analysis source fingerprints match the completed run metadata. Endpoint manifests match between runs. Baseline runtime configuration is byte-identical to its preserved snapshot; expanded runtime configuration matches its own snapshot.
- Complete main-run duplicate-policy check: 3,244 paired onset outputs agree. Raw weather files were not edited.
- Conditional I_req refinement: 252 main records plus 84 duplicate-policy records. All 84 duplicate-policy pairs and 36 common coarse-grid records match. All 126 M0/M1 pairs at the zero wind threshold also agree.
- Nested search minima do not increase as the allowed upper bound increases; lower-coverage combinations are not allowed to appear superior through missing-case exclusion.
- The comparison and conditional-profile figures were generated successfully; the main comparison figure was visually inspected.

This is a descriptive upper-bound and grid-resolution experiment. It neither establishes biological parameter identifiability nor replaces LOSYO calibration and evaluation.
