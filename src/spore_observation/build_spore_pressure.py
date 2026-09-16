#!/usr/bin/env python3
"""Build causal daily capture-pressure inputs, with training-only fold scaling."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from replay_b_fixed import ROOT, sha, save_frame, save_json


def fit_scale(intervals, training_keys):
    train = intervals[intervals.input_key.isin(training_keys)].copy()
    if set(train.input_key) != set(training_keys):
        raise ValueError('Missing training observations')
    weights = train.duration_days / train.groupby('input_key').duration_days.transform('sum') / len(training_keys)
    mean = float((weights*train.log_rate).sum())
    sd = float(np.sqrt((weights*(train.log_rate-mean)**2).sum()))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError('Training pressure has no usable variation')
    return mean, sd


def daily_history(intervals, manifest, delay, max_age):
    tables = []
    for _, case in manifest.iterrows():
        obs = intervals[intervals.input_key.eq(case.input_key)].sort_values('available_date')
        days = pd.DataFrame({'date': pd.date_range(case.weather_start, case.weather_end, freq='D')})
        fields = ['interval_id', 'end_inclusive', 'available_date', 'capture_rate', 'log_rate', 'count', 'duration_days']
        daily = pd.merge_asof(days, obs[fields], left_on='date', right_on='available_date', direction='backward')
        daily['input_key'] = case.input_key
        daily['days_since_collection'] = (daily.date-daily.end_inclusive).dt.days
        daily['days_since_available'] = (daily.date-daily.available_date).dt.days
        fresh = daily.interval_id.notna() & daily.days_since_collection.le(max_age)
        daily['pressure_available'] = fresh
        daily['observation_state'] = np.select(
            [daily.interval_id.isna(), ~fresh, daily['count'].eq(0)],
            ['unobserved', 'stale', 'zero'], default='positive')
        daily['current_capture_rate_estimate'] = daily.capture_rate.where(fresh)
        daily['current_log_rate'] = daily.log_rate.where(fresh)
        daily = daily.rename(columns={'capture_rate':'last_interval_capture_rate', 'log_rate':'last_interval_log_rate',
                                      'count':'last_interval_count', 'duration_days':'last_interval_duration_days'})
        tables.append(daily)
    return pd.concat(tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'result/spore_observation/plan1_pressure')
    parser.add_argument('--report-delay-days', type=int, default=1, help='Assumed delay from collection date to availability (default 1)')
    parser.add_argument('--max-age-days', type=int, default=7, help='Max age since collection, inclusive; not a viability parameter')
    parser.add_argument('--reference-rate', type=float, default=1.0, help='Fixed reference rate in counts/day for log1p(rate/reference)')
    args = parser.parse_args()
    if args.report_delay_days < 1 or args.max_age_days < args.report_delay_days or not np.isfinite(args.reference_rate) or args.reference_rate <= 0:
        parser.error('Use delay >=1, max-age >= delay, finite reference-rate >0')
    baseline = ROOT/'result/fsim_abc_tpe_seed123'
    audit_dir = ROOT/'result/ck_spore_observation_audit'
    source = audit_dir/'collection_intervals.csv'
    audit = json.loads((audit_dir/'audit.json').read_text())
    workbook = ROOT/'data/grape_experiment_CK_site_years.xlsx'
    if sha(workbook) != audit['sha256']:
        raise ValueError('Workbook changed; regenerate the observation audit first')
    manifest_path = ROOT/'result/fsim_inputs_seed123_exploration/site_year_manifest.csv'
    manifest = pd.read_csv(manifest_path)
    manifest['site_year'] = manifest.place_code + ' ' + manifest.Year.astype(str)
    raw = pd.read_csv(source)
    # Never use observed disease dates or outcome-based cutoffs to construct pressure.
    fields = ['site_year','excel_row','start_exclusive','end_inclusive','duration_days','count']
    intervals = raw[fields].merge(manifest[['site_year','input_key']], on='site_year', validate='many_to_one')
    for name in ('start_exclusive','end_inclusive'):
        intervals[name] = pd.to_datetime(intervals[name])
    durations = (intervals.end_inclusive-intervals.start_exclusive).dt.days
    if not durations.eq(intervals.duration_days).all() or durations.le(0).any() or intervals['count'].lt(0).any() or intervals['count'].isna().any():
        raise ValueError('Invalid interval durations/counts')
    if len(intervals) != audit['collections'] or intervals['count'].sum() != audit['total_count']:
        raise ValueError('Observation audit totals mismatch')
    for _, group in intervals.groupby('input_key'):
        group = group.sort_values('end_inclusive')
        if group.duration_days.iloc[0] != 3 or not np.array_equal(group.start_exclusive.iloc[1:].values, group.end_inclusive.iloc[:-1].values):
            raise ValueError('Expected contiguous collection intervals and first exposure of 3 days')
    intervals['interval_id'] = intervals.input_key + '_row_' + intervals.excel_row.astype(str)
    if intervals.interval_id.duplicated().any():
        raise ValueError('Duplicate interval IDs')
    intervals['capture_rate'] = intervals['count']/intervals.duration_days
    intervals['log_rate'] = np.log1p(intervals.capture_rate/args.reference_rate)
    intervals['available_date'] = intervals.end_inclusive + pd.to_timedelta(args.report_delay_days, unit='D')
    daily = daily_history(intervals, manifest, args.report_delay_days, args.max_age_days)
    scalers, all_daily, all_intervals = [], [], []
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in (source,audit_dir/'audit.json',manifest_path,workbook)}
    for scheme in ('losyo','site','year'):
        for file in sorted((baseline/f'B/{scheme}/seed_123/folds').glob('*/budget_1024/selection.json')):
            if file.parent.parent.name.startswith('.'):
                continue
            selection = json.loads(file.read_text())
            if selection['selection_uses_test_data'] or selection['selection_rule'] != 'minimum_training_MAE':
                raise ValueError('Expected training-selected B split')
            train, test = selection['training_keys'], selection['test_keys']
            if set(train)&set(test) or sorted(train+test) != sorted(manifest.input_key):
                raise ValueError('Invalid fold partition')
            mean, sd = fit_scale(intervals, train)
            fold = file.parent.parent.name
            scalers.append(dict(scheme=scheme,fold=fold,training_keys=';'.join(train),test_keys=';'.join(test),
                                mean_log_rate=mean,sd_log_rate=sd,n_training_site_years=len(train)))
            inputs[str(file.relative_to(ROOT))] = sha(file)
            for frame, log_col, target in ((daily,'current_log_rate',all_daily),(intervals,'log_rate',all_intervals)):
                out = frame.copy()
                out['scheme'],out['fold'] = scheme,fold
                out['case_role'] = np.where(out.input_key.isin(train),'train','test')
                out['relative_pressure_z'] = (out[log_col]-mean)/sd
                if log_col == 'current_log_rate':
                    out['fixed_mapping_factor'] = np.exp(out.relative_pressure_z.fillna(0))
                target.append(out)
    if len(scalers) != 12:
        raise ValueError('Expected 12 B folds')
    daily_folds = pd.concat(all_daily, ignore_index=True)
    known = daily_folds.available_date.notna()
    assert daily_folds.loc[known,'available_date'].le(daily_folds.loc[known,'date']).all()
    assert daily_folds.loc[~daily_folds.pressure_available,'relative_pressure_z'].isna().all()
    assert daily_folds.loc[~daily_folds.pressure_available,'fixed_mapping_factor'].eq(1).all()
    assert np.isfinite(daily_folds.fixed_mapping_factor).all()
    output = args.output_dir.resolve()
    for protected in (baseline,audit_dir,ROOT/'data'):
        if output==protected or protected in output.parents or output in protected.parents:
            parser.error('Use a separate output directory')
    identity = dict(script_sha256=sha(__file__),input_sha256=inputs,
                    report_delay_days=args.report_delay_days,max_age_days=args.max_age_days,reference_rate=args.reference_rate)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idfile = output/'run_identity.json'
        if idfile.exists() and json.loads(idfile.read_text()) != identity:
            raise ValueError('Inputs/settings changed; use a new --output-dir')
        save_json(idfile,identity)
        save_json(output/'run_metadata.json',{'status':'running'})
        save_frame(output/'capture_intervals.csv',intervals)
        save_frame(output/'daily_capture_background.csv',daily)
        save_frame(output/'fold_scalers.csv',pd.DataFrame(scalers))
        save_frame(output/'interval_pressure_by_fold.csv',pd.concat(all_intervals,ignore_index=True))
        save_frame(output/'daily_pressure_by_fold.csv',daily_folds)
        save_frame(output/'losyo_oof_daily_pressure.csv',daily_folds[daily_folds.scheme.eq('losyo')&daily_folds.case_role.eq('test')])
        summary = daily_folds.groupby(['scheme','fold','case_role','input_key','observation_state']).agg(
            n_days=('date','size'),min_z=('relative_pressure_z','min'),max_z=('relative_pressure_z','max')).reset_index()
        save_frame(output/'pressure_summary.csv',summary)
        save_json(output/'run_metadata.json',dict(status='complete',n_intervals=len(intervals),total_count=float(intervals['count'].sum()),
            n_daily_rows=len(daily),n_folds=len(scalers),n_fold_daily_rows=len(daily_folds),
            scaling='Equal site-year weight; within each training site-year duration-weighted log-rate population mean/SD',
            availability='Assumed collection date + report delay, then last observation carried forward until max age since collection',
            limitations=['Availability time is assumed, not recorded; max age is an input freshness rule, not pathogen survival.',
                         'Training seasons assumed available before deployment; grouped CV is not chronological validation.',
                         'No daily within-interval counts inferred; interval z is diagnostic, not usable before available_date.',
                         'Missing/stale z remains NaN; fixed factor 1 is explicit baseline fallback.',
                         'exp(z) is an uncalibrated candidate mapping; model not modified; no beta fitted.']))
    print(f'Complete: {len(intervals)} intervals; {len(scalers)} folds; {len(daily_folds)} daily fold rows.\n{output}')


if __name__ == '__main__':
    main()
