#!/usr/bin/env python3
"""Link audited collection intervals to frozen B replay cohorts; no fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def build(args):
    audit_dir = ROOT/'result/ck_spore_observation_audit'
    audit = read(audit_dir/'audit.json')
    if digest(ROOT/'data/grape_experiment_CK_site_years.xlsx') != audit['sha256']:
        raise ValueError('Workbook changed: refresh analyze_ck_spore_observations.py first')
    meta = read(args.replay/'run_metadata.json')
    if meta['status'] != 'complete' or not meta['baseline_reproduced']:
        raise ValueError('A completed, reproduced B replay is required')
    tasks = read(args.replay/'run_identity.json')['tasks']
    intervals = pd.read_csv(audit_dir/'collection_intervals.csv')
    for c in ('start_exclusive', 'end_inclusive', 'L', 'U'):
        intervals[c] = pd.to_datetime(intervals[c])
    intervals['interval_id'] = intervals.site_year.str.replace(' ', '_') + '_row_' + intervals.excel_row.astype(str)
    if intervals.interval_id.duplicated().any() or intervals['count'].lt(0).any():
        raise ValueError('Invalid interval identities/counts')
    if len(intervals) != audit['collections'] or intervals['count'].sum() != audit['total_count']:
        raise ValueError('Audit reconciliation failed')
    cohort_tables, interval_tables, summaries = [], [], []
    input_hashes = {str(audit_dir/'collection_intervals.csv'): digest(audit_dir/'collection_intervals.csv')}
    for task in tasks:
        key = task['input_key']
        site_year = key.split('_')[0] + ' ' + key.split('_')[-1]
        obs = intervals[intervals.site_year.eq(site_year)].sort_values('end_inclusive').copy()
        if obs.empty or not obs.start_exclusive.iloc[1:].reset_index(drop=True).equals(obs.end_inclusive.iloc[:-1].reset_index(drop=True)):
            raise ValueError(f'Missing or non-contiguous intervals: {key}')
        case_dir = args.replay/task['rule']/task['scheme']/task['fold']/task['case_role']/key
        marker = read(case_dir/'completed.json')['sha256']
        names = ('dispersal_cohorts.csv', 'cohort_outcomes.csv', 'primary_chain.csv', 'primary_chain_unique.csv', 'prediction.json')
        for name in names:
            path = case_dir/name
            actual = digest(path)
            if actual != marker[name]:
                raise ValueError(f'Replay artifact changed: {path}')
            input_hashes[str(path.relative_to(args.replay))] = actual
        cohorts = pd.read_csv(case_dir/'dispersal_cohorts.csv')
        outcomes = pd.read_csv(case_dir/'cohort_outcomes.csv')
        chain = pd.read_csv(case_dir/'primary_chain.csv')
        unique = pd.read_csv(case_dir/'primary_chain_unique.csv')
        pred = read(case_dir/'prediction.json')
        for table in (cohorts, outcomes, chain, unique):
            if table.Num.duplicated().any():
                raise ValueError(f'Duplicate cohort IDs: {case_dir}')
        for table in (outcomes, chain):
            extras = ['Num'] + [c for c in table if c not in cohorts]
            cohorts = cohorts.merge(table[extras], on='Num', how='left', validate='one_to_one')
        if 'outcome' not in cohorts:
            cohorts['outcome'] = 'not_exported_in_detached_survival'
        cohorts['outcome'] = cohorts.outcome.fillna('not_exported_in_detached_survival')
        for c in ('infection_datetime', 'oil_spot_start', 'oil_spot_end'):
            if c not in cohorts:
                cohorts[c] = pd.NaT
            cohorts[c] = pd.to_datetime(cohorts[c])
        dates = pd.to_datetime(cohorts.dispersal_datetime).dt.normalize()
        cohorts['interval_id'] = pd.Series(index=cohorts.index, dtype='object')
        cohorts['observation_state'] = 'unobserved'
        for _, interval in obs.iterrows():
            mask = dates.gt(interval.start_exclusive) & dates.le(interval.end_inclusive)
            if cohorts.loc[mask, 'interval_id'].notna().any():
                raise ValueError('Multiple intervals assigned to one cohort')
            cohorts.loc[mask, 'interval_id'] = interval.interval_id
            cohorts.loc[mask, 'observation_state'] = 'positive' if interval['count'] > 0 else 'zero'
        fields = ['interval_id', 'start_exclusive', 'end_inclusive', 'duration_days', 'count', 'average_count_per_day', 'excel_row']
        cohorts = cohorts.merge(obs[fields], on='interval_id', how='left', validate='many_to_one')
        cohorts = cohorts.rename(columns={'count': 'interval_count_shared_not_additive'})
        cohorts['is_retained_candidate'] = cohorts.Num.isin(unique.Num)
        if not unique.empty:
            selected = unique.sort_values(['oil_spot_start', 'infection_datetime']).iloc[0].Num
            cohorts['is_original_first_candidate'] = cohorts.Num.eq(selected)
        else:
            cohorts['is_original_first_candidate'] = False
        # Link collapsed contributors to their retained event, preserving evidence sharing.
        mapping = dict(zip(pd.to_datetime(unique.infection_datetime), unique.Num)) if not unique.empty else {}
        passed = cohorts.get('passed_host_gate', pd.Series(False, index=cohorts.index)).eq(True)
        cohorts['retained_event_Num'] = cohorts.infection_datetime.map(mapping).where(passed)
        L, U = pd.Timestamp(pred['observed_L']), pd.Timestamp(pred['observed_U'])
        if not obs.L.eq(L).all() or not obs.U.eq(U).all():
            raise ValueError(f'Observed endpoint mismatch: {key}')
        valid = cohorts.oil_spot_start.notna()
        distance = pd.Series(float('nan'), index=cohorts.index)
        distance.loc[valid] = 0.0
        early = valid & cohorts.oil_spot_end.lt(L)
        late = valid & cohorts.oil_spot_start.gt(U)
        distance.loc[early] = (L-cohorts.loc[early, 'oil_spot_end']).dt.total_seconds()/86400
        distance.loc[late] = (cohorts.loc[late, 'oil_spot_start']-U).dt.total_seconds()/86400
        cohorts['posthoc_interval_distance_days'] = distance
        cohorts['posthoc_overlaps_observed'] = valid & cohorts.oil_spot_end.gt(L) & cohorts.oil_spot_start.le(U)
        cohorts['selection_uses_test_data'] = task['rule'] != 'min_train'
        for k, v in task.items():
            cohorts[k] = v
            obs[k] = v
        obs['n_airborne_cohorts'] = obs.interval_id.map(cohorts.groupby('interval_id').size()).fillna(0).astype(int)
        for label, mask in [('completed_chain', valid), ('retained_candidates', cohorts.is_retained_candidate),
                            ('overlapping_retained_candidates', cohorts.is_retained_candidate & cohorts.posthoc_overlaps_observed)]:
            obs['n_'+label] = obs.interval_id.map(cohorts[mask].groupby('interval_id').size()).fillna(0).astype(int)
        candidates = cohorts[cohorts.is_retained_candidate]
        first = cohorts[cohorts.is_original_first_candidate]
        summary = dict(**task, selection_uses_test_data=task['rule'] != 'min_train',
            n_airborne_cohorts=len(cohorts), n_candidates=len(candidates),
            n_candidates_positive=int(candidates.observation_state.eq('positive').sum()),
            n_candidates_zero=int(candidates.observation_state.eq('zero').sum()),
            n_candidates_unobserved=int(candidates.observation_state.eq('unobserved').sum()),
            n_overlapping_candidates=int(candidates.posthoc_overlaps_observed.sum()),
            posthoc_best_candidate_distance_days=candidates.posthoc_interval_distance_days.min(),
            original_first_observation_state=first.observation_state.iloc[0] if len(first) else 'no_prediction',
            original_first_signed_error_days=pred.get('signed_interval_error_days'),
            n_positive_intervals_without_airborne_cohorts=int((obs['count'].gt(0)&obs.n_airborne_cohorts.eq(0)).sum()),
            observed_L=L, observed_U=U)
        cohort_tables.append(cohorts); interval_tables.append(obs); summaries.append(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    all_cohorts = pd.concat(cohort_tables, ignore_index=True)
    all_cohorts.to_csv(args.output/'cohort_observation_links.csv', index=False)
    all_cohorts[all_cohorts.is_retained_candidate].to_csv(args.output/'candidate_observation_links.csv', index=False)
    event_keys = ['rule', 'scheme', 'fold', 'case_role', 'input_key', 'retained_event_Num',
                  'interval_id', 'observation_state']
    event_links = all_cohorts[all_cohorts.retained_event_Num.notna()].groupby(event_keys, dropna=False).agg(
        n_contributing_cohorts=('Num', 'size'),
        interval_count_shared_not_additive=('interval_count_shared_not_additive', 'first'),
        earliest_dispersal=('dispersal_datetime', 'min'), latest_dispersal=('dispersal_datetime', 'max'))
    event_links.reset_index().to_csv(args.output/'event_interval_links.csv', index=False)
    pd.concat(interval_tables, ignore_index=True).to_csv(args.output/'interval_cohort_summary.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output/'case_diagnostics.csv', index=False)
    main = summary[summary.rule.eq('min_train') & summary.scheme.eq('losyo') & summary.case_role.eq('test')]
    main.to_csv(args.output/'losyo_primary_diagnostics.csv', index=False)
    metadata = dict(status='complete', cases=len(tasks), cohort_rows=len(all_cohorts),
        unique_observation_intervals=len(intervals), workbook_sha256=audit['sha256'], script_sha256=digest(__file__),
        input_sha256=input_hashes, replay=str(args.replay),
        interval_convention='Date-only assignment: previous collection date < normalized dispersal date <= current collection date; first interval 3 days. Collection hour unknown.',
        limitations=['Counts are shared interval evidence, not individual cohort doses; do not sum repeated counts.',
                    'Posthoc endpoint comparisons diagnose candidate coverage; never select predictions using these columns.',
                    'No support weights, penalties, probability estimates or parameter fitting applied.',
                    'Input begins with exported dispersal cohorts; unexported upstream failures are not reconstructed.'])
    (args.output/'link_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(main.to_string(index=False))
    print(f'Written {len(all_cohorts)} cohort links across {len(tasks)} runs: {args.output}')


def plot(args):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'fsim_spore_link_mpl'))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.dates as mdates
    links = pd.read_csv(args.output/'cohort_observation_links.csv', low_memory=False)
    intervals = pd.read_csv(args.output/'interval_cohort_summary.csv')
    args.fig.mkdir(parents=True, exist_ok=True)
    links = links[links.rule.eq('min_train') & links.scheme.eq('losyo') & links.case_role.eq('test')]
    intervals = intervals[intervals.rule.eq('min_train') & intervals.scheme.eq('losyo') & intervals.case_role.eq('test')]
    colors = {'positive':'#3185a5', 'zero':'#cb8135', 'unobserved':'#999999'}
    for key, group in links.groupby('input_key'):
        obs = intervals[intervals.input_key.eq(key)]
        fig = Figure(figsize=(12, 7)); FigureCanvasAgg(fig)
        top, bottom = fig.subplots(2, 1, sharex=True)
        starts = pd.to_datetime(obs.start_exclusive); ends = pd.to_datetime(obs.end_inclusive)
        top.bar(starts, obs['count'], width=(ends-starts).dt.total_seconds()/86400, align='edge', color='#3185a5', alpha=.7)
        zero = obs['count'].eq(0)
        top.scatter(ends[zero], obs.loc[zero,'count'], facecolors='none', edgecolors='#cb8135', label='Zero captures')
        top.set_ylabel('Count per collection interval'); top.legend(loc='upper right')
        candidate = group[group.is_retained_candidate].sort_values('oil_spot_start')
        for n, (_, row) in enumerate(candidate.iterrows(), 1):
            bottom.plot(pd.to_datetime([row.oil_spot_start,row.oil_spot_end]), [n,n],
                        color=colors[row.observation_state], linewidth=3)
            bottom.scatter(pd.Timestamp(row.dispersal_datetime), n, color=colors[row.observation_state], marker='|')
            bottom.plot(pd.to_datetime([row.dispersal_datetime,row.oil_spot_start]), [n,n], color=colors[row.observation_state], alpha=.25, linewidth=.7)
        for state, color in colors.items():
            bottom.plot([],[],color=color,label=f'Air-input interval: {state}')
        L,U = pd.Timestamp(obs.L.iloc[0]),pd.Timestamp(obs.U.iloc[0])
        for ax in (top,bottom):
            ax.axvspan(L,U,color='#c4493e',alpha=.2)
            ax.grid(axis='x',alpha=.15)
        bottom.set_ylabel('Retained candidate (chronological)')
        bottom.set_xlabel('Date; shaded red: observed onset interval')
        bottom.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
        bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        fig.suptitle(f'{key} | B minimum-training LOSYO | retrospective linkage')
        fig.text(.06,.015,'Bars show interval totals, not daily counts. Ticks: dispersal; segments: candidate symptom windows.\nColors label shared observation evidence; no weighting or new prediction has been applied.',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.95))
        for ext in ('png','pdf'):
            fig.savefig(args.fig/f'{key}_spore_cohort_links.{ext}',dpi=160)
        fig.clear()
    print(f'Figures: {args.fig}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,default=ROOT/'result/spore_observation/plan1_b_fixed')
    parser.add_argument('--output',type=Path,default=ROOT/'result/spore_observation/plan1_b_links')
    parser.add_argument('--fig',type=Path,default=ROOT/'fig/spore_observation/plan1_b_links')
    parser.add_argument('--plot',action='store_true',help='Build tables and plot six primary LOSYO cases')
    parser.add_argument('--plot-only',action='store_true',help='Plot existing association tables')
    args = parser.parse_args()
    if not args.plot_only:
        build(args)
    if args.plot or args.plot_only:
        plot(args)
