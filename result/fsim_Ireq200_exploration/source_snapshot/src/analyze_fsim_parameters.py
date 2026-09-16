#!/usr/bin/env python3
"""Run a table-defined FSIM grid: retrospective response and identifiability study.

No calibration, model selection, LOSYO or FSIM-S results are produced here.
Production/transfer/survival are cached only within a case and wind setting;
all downstream evaluations use the ordinary FirstInfection implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model.parameter_table import DEFAULT_TABLE, candidate_rows, load_config
from model.crop_season import CropSeasonFirst
from model.first_infection import FirstInfection


PARAM_KEYS = {'VPD_w': ('zoospore_alive', 'vpd'),
              'I_req': ('infection', 'infection_condition'),
              'u_tr': ('dispersal', 'wind_min')}


def configure(base, sample):
    cfg = copy.deepcopy(base)
    cfg['dispersal']['use_wind_gate'] = sample['model'] == 'M1'
    for name, (group, key) in PARAM_KEYS.items():
        value = sample.get(name)
        if value is not None and pd.notna(value):
            cfg[group][key] = float(value)
    return cfg


def score_window(l, u, start, end):
    """Metric infimum is zero at L, but (L,U] excludes exact contact at L."""
    if not l < u or pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError('Invalid observed interval or predicted window')
    day = pd.Timedelta(days=1)
    signed = (end-l)/day if end < l else ((start-u)/day if start > u else 0.)
    point_signed = (end-u)/day if end < u else ((start-u)/day if start > u else 0.)
    return {
        'predicted_start': start.isoformat(), 'predicted_end': end.isoformat(),
        'predicted_midpoint_doy': ((start + (end-start)/2) - pd.Timestamp(start.year, 1, 1))/day + 1,
        'window_width_days': (end-start)/day,
        'overlap': bool(end > l and start <= u),
        'signed_interval_error_days': float(signed),
        'interval_distance_days': abs(float(signed)),
        'abs_midpoint_error_days': abs(((start+(end-start)/2)-(l+(u-l)/2))/day),
        'first_positive_distance_days': abs(float(point_signed)),
    }


def make_design(table, levels_override=None):
    samples = []
    for variant in ('M0', 'M1'):
        rows = candidate_rows(variant, table)
        axes = [np.asarray(r['grid_values'], dtype=float)
                if not levels_override and 'grid_values' in r
                else np.linspace(float(r['lower']), float(r['upper']),
                                 levels_override or int(r['grid_levels'])) for r in rows]
        for values in itertools.product(*axes):
            samples.append({'sample_id': f'{variant}_g{len(samples):05d}', 'model': variant,
                            'design': 'grid', 'u_tr': np.nan,
                            **dict(zip([r['parameter'] for r in rows], values))})
        samples.append({'sample_id': f'{variant}_default', 'model': variant,
                        'design': 'default', 'u_tr': np.nan,
                        **{r['parameter']: r['value'] for r in rows}})
    return pd.DataFrame(samples)


def load_cases(manifest):
    frame = pd.read_csv(manifest, encoding='utf-8-sig')
    if len(frame) != 6 or frame.input_key.duplicated().any():
        raise ValueError('This analysis requires six unique site-years')
    cases, errors, context, duplicate_cases, repairs = [], [], [], [], []
    for row in frame.itertuples(index=False):
        try:
            path = ROOT / row.weather_json
            rb = json.loads(path.read_text())
            w = pd.DataFrame(rb['weather_hourly'])
            alternate = None
            duplicate_count = int(w.datetime.duplicated().sum())
            if duplicate_count:
                # Preserve both possible resolutions; do not assert either is ground truth.
                alternate = copy.deepcopy(rb)
                alternate['weather_hourly'] = w.drop_duplicates('datetime', keep='first').to_dict('records')
                duplicate_rows = w[w.datetime.duplicated(keep=False)]
                repairs.append({'input_key': row.input_key, 'extra_rows': duplicate_count,
                                'first_duplicate': duplicate_rows.datetime.min(),
                                'last_duplicate': duplicate_rows.datetime.max(),
                                'primary_policy': 'keep_last', 'sensitivity_policy': 'keep_first'})
                w = w.drop_duplicates('datetime', keep='last').reset_index(drop=True)
                rb['weather_hourly'] = w.to_dict('records')
            times = pd.DatetimeIndex(pd.to_datetime(w['datetime']))
            year = int(row.Year)
            expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h', inclusive='left')
            if not times.equals(expected):
                raise ValueError('Weather must be a complete ordered hourly calendar year')
            for col in ('air_temp', 'relative_humidity', 'precip_acc_period_adjusted', 'wind_speed'):
                values = pd.to_numeric(w[col], errors='coerce')
                if not np.isfinite(values).all():
                    raise ValueError(f'Non-finite weather input: {col}')
            if (w.wind_speed < 0).any() or not w.relative_humidity.between(0, 100).all():
                raise ValueError('Invalid wind or relative humidity')
            gate = pd.to_datetime(rb.get('BBCH_15'), errors='coerce')
            l, u = pd.Timestamp(row.L), pd.Timestamp(row.U)
            if pd.isna(gate) or gate.year != year or not l < u or l.year != year or u.year != year:
                raise ValueError('Missing/invalid gate or endpoint')
            if hasattr(row, 'rb_bbch15') and gate != pd.Timestamp(row.rb_bbch15):
                raise ValueError('Manifest and weather JSON disagree on BBCH15')
            cases.append({'input_key': row.input_key, 'L': str(l), 'U': str(u), 'rb': rb,
                          'weather_policy': 'keep_last' if duplicate_count else 'original'})
            if alternate is not None:
                duplicate_cases.append({**cases[-1], 'rb': alternate, 'weather_policy': 'keep_first'})
            # Fixed Jan-June period, independent of onset dates; full year also reported.
            for period, wind in [('full_year', w.wind_speed), ('Jan_Jun', w.loc[times.month <= 6, 'wind_speed'])]:
                context.append({'input_key': row.input_key, 'period': period, 'n_hours': len(wind),
                                **{f'p{q:g}': wind.quantile(q/100) for q in (50,90,95,99,99.9)},
                                'maximum': wind.max(),
                                **{f'fraction_ge_{x:g}': float((wind >= x).mean()) for x in range(9)}})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'input_key': row.input_key, 'status': 'input_error',
                           'error_type': type(exc).__name__, 'message': str(exc)})
    return cases, errors, pd.DataFrame(context), duplicate_cases, pd.DataFrame(repairs)


def evaluate_group(task):
    case, samples, base = task
    records = []
    try:
        cfg = configure(base, samples[0])
        cs = CropSeasonFirst(case['rb'], cfg)
        engine = FirstInfection(cs, cfg)
        engine.stage_attached_and_dispersal()
        engine.stage_deposition()
        engine.stage_detached_survival()
    except Exception as exc:
        return [{**s, 'input_key': case['input_key'], 'weather_policy': case.get('weather_policy','original'), 'status': 'runtime_error',
                 'error_type': type(exc).__name__, 'error_message': str(exc),
                 'traceback': traceback.format_exc()} for s in samples]
    for sample in samples:
        record = {**sample, 'input_key': case['input_key'], 'observed_L': case['L'],
                  'observed_U': case['U'], 'status': 'no_prediction', 'overlap': False,
                  'weather_policy': case.get('weather_policy','original')}
        try:
            cfg = configure(base, sample)
            cs.config = cfg
            engine.cfg = cfg
            engine.stage_primary_chain()
            cs.apply_bbch_15_gate()
            cs.collapse_equivalent_cohorts()
            windows = cs.primary_chain_unique
            record['n_windows'] = 0 if windows is None else len(windows)
            if windows is not None and not windows.empty:
                first = windows.sort_values(['oil_spot_start', 'infection_datetime']).iloc[0]
                record.update(score_window(pd.Timestamp(case['L']), pd.Timestamp(case['U']),
                                           pd.Timestamp(first.oil_spot_start), pd.Timestamp(first.oil_spot_end)))
                record['status'] = 'predicted'
        except Exception as exc:
            record.update(status='runtime_error', error_type=type(exc).__name__,
                          error_message=str(exc), traceback=traceback.format_exc())
        records.append(record)
    return records


def summarize(predictions):
    records = []
    for sid, block in predictions.groupby('sample_id', sort=False):
        first = block.iloc[0]
        valid = block[block.status == 'predicted']
        n = len(valid)
        records.append({**{k: first[k] for k in ('sample_id','model','design','VPD_w','I_req','u_tr')},
                        'n_site_years': len(block), 'n_predictions': n,
                        'n_no_prediction': int((block.status == 'no_prediction').sum()),
                        'n_errors': int((block.status == 'runtime_error').sum()),
                        'n_overlaps': int(valid.overlap.sum()),
                        'conditional_mae_days': valid.interval_distance_days.mean(),
                        'all_site_year_mae_days': valid.interval_distance_days.mean() if n == len(block) else np.nan,
                        'mean_signed_error_days': valid.signed_interval_error_days.mean(),
                        'mean_window_width_days': valid.window_width_days.mean(),
                        'mean_first_positive_distance_days': valid.first_positive_distance_days.mean()})
    return pd.DataFrame(records)


def profile_and_near_optimal(summary, table):
    profiles, retained, ranges = [], [], []
    for model, block in summary[summary.design == 'grid'].groupby('model'):
        max_coverage = block.n_predictions.max()
        eligible = block[(block.n_predictions == max_coverage) & (block.n_predictions > 0)]
        if eligible.empty:
            continue
        best = eligible.conditional_mae_days.min()
        for row in candidate_rows(model, table):
            name = row['parameter']
            for value, group in block.groupby(name):
                nmax = group.n_predictions.max()
                options = group[group.n_predictions == nmax]
                loss = options.conditional_mae_days.min()
                profiles.append({'model': model, 'parameter': name, 'value': value,
                                 'max_n_predictions': nmax, 'profile_mae_days': loss,
                                 'comparable_to_global_best': nmax == max_coverage,
                                 'delta_mae_days': loss-best if nmax == max_coverage else np.nan})
        for tolerance in (0., .5, 1., 2.):
            near = eligible[eligible.conditional_mae_days <= best+tolerance+1e-12].copy()
            near['tolerance_days'] = tolerance
            near['delta_mae_days'] = near.conditional_mae_days-best
            retained.append(near)
            for row in candidate_rows(model, table):
                name = row['parameter']
                ranges.append({'model': model, 'parameter': name, 'tolerance_days': tolerance,
                               'best_mae_days': best, 'max_n_predictions': max_coverage,
                               'n_retained': len(near), 'n_grid': len(block),
                               'min_retained': near[name].min(), 'max_retained': near[name].max(),
                               'n_distinct_values': near[name].nunique(),
                               'touches_lower': bool(np.isclose(near[name].min(), float(row['lower']))),
                               'touches_upper': bool(np.isclose(near[name].max(), float(row['upper'])))})
    return pd.DataFrame(profiles), pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(), pd.DataFrame(ranges)


def elementary_effects(predictions, summary, table):
    """Adjacent full-grid effects per unit normalized input; NOT Morris/Sobol.

    No-prediction transitions are counted separately. Missing timing endpoints
    are never assigned a numerical penalty or interpolated.
    """
    output = []
    for model in ('M0','M1'):
        rows = candidate_rows(model, table)
        names = [r['parameter'] for r in rows]
        scopes = [(key, block, ['interval_distance_days', 'predicted_midpoint_doy'])
                  for key, block in predictions[(predictions.model == model) & (predictions.design == 'grid')].groupby('input_key')]
        scopes.append(('ALL_SIX', summary[(summary.model == model) & (summary.design == 'grid')],
                       ['all_site_year_mae_days', 'n_predictions']))
        for scope, block, responses in scopes:
            for row in rows:
                name = row['parameter']
                others = [n for n in names if n != name]
                groups = list(block.groupby(others, dropna=False))
                for response in responses:
                    effects, edge_count, transitions = [], 0, 0
                    for _, group in groups:
                        group = group.sort_values(name)
                        x = group[name].to_numpy(float)
                        y = group[response].to_numpy(float)
                        valid = np.isfinite(y[:-1]) & np.isfinite(y[1:])
                        dx = np.diff(x)/(float(row['upper'])-float(row['lower']))
                        effects.extend((np.diff(y)[valid]/dx[valid]).tolist())
                        edge_count += len(dx)
                        if 'status' in group:
                            status = (group.status == 'predicted').to_numpy()
                            transitions += int(np.sum(status[:-1] != status[1:]))
                    e = np.asarray(effects)
                    output.append({'model': model, 'scope': scope, 'parameter': name,
                                   'response': response, 'n_edges': edge_count, 'n_valid_edges': len(e),
                                   'prediction_status_transitions': transitions,
                                   'mean_effect': e.mean() if len(e) else np.nan,
                                   'mean_abs_effect': np.abs(e).mean() if len(e) else np.nan,
                                   'sd_effect': e.std(ddof=1) if len(e)>1 else np.nan,
                                   'fraction_zero_effect': np.isclose(e,0).mean() if len(e) else np.nan})
    return pd.DataFrame(output)


def check_zero_gate(predictions):
    grid = predictions[predictions.design == 'grid']
    m0 = grid[grid.model == 'M0']
    m1 = grid[(grid.model == 'M1') & (grid.u_tr == 0)]
    if m1.empty:
        return {'status': 'not_in_design'}
    merged = m0.merge(m1, on=['input_key','VPD_w','I_req'], suffixes=('_M0','_M1'), validate='one_to_one')
    for col in ('status','predicted_start','predicted_end','n_windows'):
        a, b = merged[col+'_M0'].fillna('MISSING'), merged[col+'_M1'].fillna('MISSING')
        if not (a == b).all():
            raise AssertionError(f'M1 zero wind gate differs from M0: {col}')
    if len(merged) != len(m0):
        raise AssertionError('Incomplete zero-gate comparison')
    return {'status': 'passed', 'n_paired_site_year_predictions': len(merged)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-table', type=Path, default=DEFAULT_TABLE)
    parser.add_argument('--manifest', type=Path, default=ROOT/'result/fsim_inputs/site_year_manifest.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'result/fsim_parameter_analysis')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--grid-levels', type=int, help='Override all axes for a smoke test; >=3')
    args = parser.parse_args()
    if args.workers < 1 or (args.grid_levels is not None and args.grid_levels < 3):
        parser.error('workers must be positive and grid-levels >=3')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be empty; preserve previous results')
    base = load_config(args.config_table)
    design = make_design(args.config_table, args.grid_levels)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config_table, out/'parameter_table.csv')
    shutil.copy2(args.manifest, out/'site_year_manifest.csv')
    design.to_csv(out/'design.csv', index=False)
    cases, errors, context, duplicate_cases, repairs = load_cases(args.manifest)
    repairs.to_csv(out/'weather_duplicate_audit.csv', index=False)
    context.to_csv(out/'wind_context.csv', index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out/'input_errors.csv', index=False)
        raise SystemExit('Input validation failed; see input_errors.csv')
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    source_files = list((ROOT/'model').glob('*.py')) + [Path(__file__)]
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'purpose': 'descriptive full-data grid; not calibration or held-out validation',
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'arguments': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'n_design_points': len(design), 'n_site_year_evaluations': len(design)*len(cases),
                'n_duplicate_sensitivity_evaluations': len(design)*len(duplicate_cases),
                'objective': 'prediction coverage first; observed-interval distance second',
                'near_optimal_tolerances_days': [0,.5,1,2],
                'tolerance_interpretation': 'descriptive thresholds, not confidence intervals',
                'source_sha256': {}, 'input_sha256': {}}
    for path in source_files:
        rel = path.resolve().relative_to(ROOT)
        target = snapshot/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target)
        metadata['source_sha256'][str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = pd.read_csv(args.manifest, encoding='utf-8-sig')
    for name in manifest.weather_json:
        metadata['input_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    tasks = [(case, group.to_dict('records'), base) for case in cases + duplicate_cases
             for _, group in design.groupby(['model','u_tr'], dropna=False, sort=False)]
    records = []
    print(f'{len(design)} parameter sets, {len(cases)} site-years, {len(tasks)} cached groups', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_group, task) for task in tasks]
        for i, future in enumerate(as_completed(futures),1):
            batch = future.result()
            records.extend(batch)
            # A readable checkpoint survives interruption; it is not a completed run.
            with (out/'predictions_checkpoint.jsonl').open('a') as checkpoint:
                checkpoint.write(pd.DataFrame(batch).to_json(orient='records', lines=True))
            print(f'Completed {i}/{len(tasks)} groups ({len(records)} site-year evaluations)', flush=True)
    predictions = pd.DataFrame(records).sort_values(['model','sample_id','input_key'])
    for col in ('predicted_start','predicted_end','predicted_midpoint_doy','window_width_days',
                'signed_interval_error_days','interval_distance_days','abs_midpoint_error_days',
                'first_positive_distance_days','n_windows'):
        if col not in predictions:
            predictions[col] = np.nan
    duplicate_predictions = predictions[predictions.weather_policy == 'keep_first'].copy()
    predictions = predictions[predictions.weather_policy != 'keep_first'].copy()
    predictions.to_csv(out/'predictions.csv', index=False)
    duplicate_predictions.to_csv(out/'duplicate_policy_predictions.csv', index=False)
    all_errors = pd.DataFrame(records)
    failures = all_errors[all_errors.status == 'runtime_error']
    if not failures.empty:
        failures.to_csv(out/'runtime_errors.csv', index=False)
        metadata['status'] = 'failed_runtime_errors'
        (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
        raise SystemExit('Runtime errors occurred; no scientific summaries produced')
    summary = summarize(predictions)
    summary.to_csv(out/'sample_metrics.csv', index=False)
    profiles, near, ranges = profile_and_near_optimal(summary, args.config_table)
    profiles.to_csv(out/'parameter_profiles.csv', index=False)
    near.to_csv(out/'near_optimal_sets.csv', index=False)
    ranges.to_csv(out/'near_optimal_ranges.csv', index=False)
    elementary_effects(predictions, summary, args.config_table).to_csv(out/'grid_elementary_effects.csv',index=False)
    metadata['zero_gate_equivalence'] = check_zero_gate(predictions)
    if not duplicate_predictions.empty:
        comparison = predictions.merge(duplicate_predictions, on=['sample_id','input_key'],
                                       suffixes=('_last','_first'), validate='one_to_one')
        for col in ('status','predicted_start','predicted_end'):
            comparison[col+'_equal'] = comparison[col+'_last'].fillna('MISSING') == comparison[col+'_first'].fillna('MISSING')
        equal_columns = ['status_equal','predicted_start_equal','predicted_end_equal']
        comparison[['sample_id','input_key']+equal_columns].to_csv(out/'duplicate_policy_comparison.csv',index=False)
        metadata['duplicate_policy_check'] = {'n_pairs': len(comparison),
            'n_different_onset_outputs': int((~comparison[equal_columns].all(axis=1)).sum())}
    metadata['status'] = 'complete'
    metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    if not ranges.empty:
        print(ranges[ranges.tolerance_days == 1].to_string(index=False),flush=True)
    print(f'Saved {out}',flush=True)


if __name__ == '__main__':
    main()
