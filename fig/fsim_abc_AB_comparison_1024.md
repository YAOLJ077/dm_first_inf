# A versus B: saved results, seed 123, 1024 trials per fold

B was complete and the run had moved to C. B exploratory root summaries were absent in the local copy; the plot reads all twelve corresponding per-fold exports instead, validates splits, interval errors, selection roles and the one-day tolerance, and hashes the files used. No model was run and no calibration output was overwritten.

Both arms use two parameters with wind gating disabled. A ranges: VPD_w 1–4.5, I_req 120–350. B ranges: VPD_w 1–6, I_req 100–600. Other configured conditions are shared. All selected B parameter combinations in both selection strategies lie inside A's ranges.

| Selection | Evaluation | A test MAE (days) | B test MAE (days) | Completeness |
|---|---|---:|---:|---|
| Minimum training MAE | LOSYO | 41.458 | 29.986 | Both 6/6 |
| Minimum training MAE | Site | 28.799 | 27.367 conditional | A 6/6; B 5/6 |
| Minimum training MAE | Year | 35.271 | 41.729 | Both 6/6 |
| Training minimum + 1 day, then minimum test MAE | LOSYO | 36.333 | 27.840 | Both 6/6 |
| Training minimum + 1 day, then minimum test MAE | Site | 28.479 | 28.479 | Both 6/6 |
| Training minimum + 1 day, then minimum test MAE | Year | 26.167 | 26.174 | Both 6/6 |

The conditional B site MAE excludes DA 2023, which has no prediction. It is not evidence of improvement over A's complete-case coverage.

LOSYO improvement under training-minimum selection is concentrated in NN 2023 (absolute error 62.875 to 24.583 days) and NN 2024 (39.708 to 9.083 days). Under exploratory selection, NN 2023 improves from 62.875 to 11.917 days; most other results remain essentially unchanged.

For LOSYO NN 2023, A's minimum training MAE is 9.675 days and B's is 15.392 days. A's solution lies inside B's domain, demonstrating that the B search did not recover an already-known lower training objective within its domain. The exploratory candidate ceilings are consequently 10.675 and 16.392 days, respectively. The two results cannot be interpreted as the benefit of the enlarged domain alone.

For the year-2024 fold, minimum training MAE is approximately 19.698 days in both arms, but test MAE rises from 33.604 to 52.979 days. NN 2024's signed error rises from +63.583 to +102.333 days. This is a clear example of nearly identical training objectives with sharply different held-out predictions. Exploratory selection reduces the B fold MAE to 6.333 days, close to A's 6.312 days.

These data show search-path sensitivity and equifinality, rather than consistent gains from the expanded ranges. Exploratory selection uses held-out outcomes and is not independent test performance. The comparison does not establish an optimal parameter domain or a benefit from increasing the optimization budget.
