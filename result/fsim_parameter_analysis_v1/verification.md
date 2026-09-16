# Verification — 2026-09-13

- `dm_first_os/bin/python -m unittest discover -s tests`: 38 tests passed.
- Tests cover runtime-table loading and invalid configuration rejection, structure-specific grids, interval versus first-positive-date metrics, cached versus public-model predictions, known normalized finite-difference slopes, coverage-prioritized profiles, and non-destructive weather-duplicate handling.
- Main run: 632 parameter sets × 6 site-years = 3,792 unique records; 630 grid sets plus two historical-default controls. No runtime errors. Every set has all six site-year records, including explicit no-prediction statuses.
- Additional duplicate-weather-policy run: 1,264 records; no differences in status or predicted onset-window boundaries between keeping first and last duplicates.
- M1 zero-gate versus M0: all 378 paired grid site-year predictions agree (status, boundaries and number of windows).
- Every model/analysis source and original weather JSON matches the SHA-256 captured at run time. Parameter table snapshot retained. Original weather files were not modified.
- `git diff --check`: passed. The manuscript source was not edited by this task.
- Response surfaces and parameter profiles inspected visually; plotting completed successfully. Individual site-year panels retained in a PDF atlas.

These checks establish computational consistency for the current implementation and grid, not biological validity, unique parameter identification, or out-of-fold predictive performance.
