#!/usr/bin/env python3
"""Compare training-only set filters on saved FSIM candidates; no model runs.

A and B are combined within each fold, using their common known minimum.
All ties remain sets: no VPD, parameter-order, trial-count or test-score ranking.
Test summaries are descriptive audits AFTER filtering, never filter inputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'result/fsim_abc_selection_diagnosis_1024'
OUT = ROOT / 'result/fsim_saved_selection_rules_1024'
EPS = 1e-9


def nondominated(errors):
    """No other row has <= error in every training case and < in any case."""
    return np.array([not np.any(np.all(errors <= row + EPS, axis=1)
                               & np.any(errors < row - EPS, axis=1))
                     for row in errors])


def main():
    candidates = pd.read_csv(SOURCE / 'candidates.csv')
    cases = pd.read_csv(SOURCE / 'candidate_case_predictions.csv')
    OUT.mkdir(exist_ok=True)
    rows, memberships, spreads, selected_groups = [], [], [], []
    for (scheme, held), raw in candidates[candidates.arm.isin(['A', 'B'])].groupby(['scheme', 'held']):
        best = raw.training_MAE_days.min()
        g = raw[raw.training_MAE_days <= best + 1 + EPS].copy()
        g = g.drop_duplicates(['VPD_w', 'I_req', 'RH_sp', 'MP_sp', 'u_tr']).reset_index(drop=True)
        g['candidate_id'] = g.arm + ':' + g.trial.astype(str)
        train = cases[(cases.scheme == scheme) & (cases.held == held) & (cases.role == 'train')].copy()
        train['candidate_id'] = train.arm + ':' + train.trial.astype(str)
        errors = train.pivot(index='candidate_id', columns='input_key', values='error').loc[g.candidate_id].abs()
        assert errors.notna().all().all()
        assert np.allclose(errors.mean(axis=1), g.training_MAE_days, atol=EPS, rtol=0)
        assert np.allclose(errors.max(axis=1), g.training_max_error, atol=EPS, rtol=0)
        masks = {
            'minimum_MAE_ties': g.training_MAE_days <= best + EPS,
            'MAE_plus_1': np.ones(len(g), dtype=bool),
            'plus_1_min_RMSE_ties': g.training_RMSE_days <= g.training_RMSE_days.min() + EPS,
            'plus_1_min_max_error_ties': g.training_max_error <= g.training_max_error.min() + EPS,
            'plus_1_casewise_Pareto': nondominated(errors.to_numpy()),
        }
        selected_groups.append((scheme, held, g.copy(), np.asarray(masks['plus_1_casewise_Pareto'])))
        for rule, mask in masks.items():
            s = g.loc[np.asarray(mask)].copy()
            assert len(s) > 0
            complete = s[s.n_predicted == s.n_test]
            rows.append(dict(scheme=scheme, held=held, rule=rule, known_best=best,
                             n=len(s), training_day_modes=s.train_mode.nunique(),
                             training_exact_modes=s.exact_train_mode.nunique(),
                             train_MAE_min=s.training_MAE_days.min(), train_MAE_max=s.training_MAE_days.max(),
                             train_max_error_min=s.training_max_error.min(), train_max_error_max=s.training_max_error.max(),
                             test_day_modes=s.test_mode.nunique(), test_failures=len(s)-len(complete),
                             test_MAE_min=complete.test_MAE_days.min(), test_MAE_max=complete.test_MAE_days.max()))
            memberships.extend(dict(scheme=scheme, held=held, rule=rule, candidate_id=x) for x in s.candidate_id)
        near_train = train[train.candidate_id.isin(g.candidate_id)].copy()
        near_train['mid'] = pd.to_datetime(near_train.predicted_start) + (pd.to_datetime(near_train.predicted_end)-pd.to_datetime(near_train.predicted_start))/2
        for key, t in near_train.groupby('input_key'):
            spreads.append(dict(scheme=scheme, held=held, input_key=key,
                                midpoint_span_days=(t.mid.max()-t.mid.min()).total_seconds()/86400,
                                absolute_error_span_days=t.error.abs().max()-t.error.abs().min()))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'rule_sets.csv', index=False)
    pd.DataFrame(memberships).to_csv(OUT / 'retained_members.csv', index=False)
    pd.DataFrame(spreads).to_csv(OUT / 'training_case_spreads.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (_, held, g, pareto) in zip(axes.flat, [x for x in selected_groups if x[0] == 'losyo']):
        modes = sorted(g.train_mode.unique())
        for i, mode in enumerate(modes):
            t = g[g.train_mode == mode]
            ax.scatter(t.VPD_w, t.I_req, s=15, alpha=.65, color=plt.get_cmap('tab10')(i), label=f'Train pattern {i+1}')
        t = g[pareto]
        ax.scatter(t.VPD_w, t.I_req, s=36, facecolors='none', edgecolors='black', linewidths=.7, label='Casewise Pareto')
        ax.set_title(held.replace('_MPT', '').replace('_WK', '').replace('_JF', '').replace('_', ' '))
        ax.set_xlabel('VPD parameter'); ax.set_ylabel('Infection requirement')
        ax.grid(alpha=.15); ax.set_axisbelow(True)
        ax.legend(fontsize=7, loc='best')
    fig.suptitle('Saved A+B 2p candidates: training MAE within 1 day of common minimum', fontsize=12)
    fig.savefig(OUT / 'losyo_parameter_patterns.png', dpi=200)
    fig.savefig(OUT / 'losyo_parameter_patterns.pdf')
    plt.close(fig)
    (OUT / 'method.json').write_text(json.dumps(dict(
        model_runs=0, input=str(SOURCE.relative_to(ROOT)),
        pool='A+B within identical fold; same 2p model; deduplicate exact parameter combinations',
        tolerance_days=1, tie_tolerance_days=EPS,
        note='TPE sample counts are not probabilities or region volumes. Pareto uses individual training absolute interval errors. Test results are post-filter diagnostics, never used for filtering. No probabilities, confidence intervals or unique winner inferred.',
    ), indent=2))
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
