#!/usr/bin/env python3
"""Six-fold LOSYO: 2p Grid/TPE, then 3p Grid/TPE, then 4p Grid/TPE.

Reuse the frozen engines and training-only local Grid/TPE implementations.
Each dimension has an isolated cache; only one shared joblib pool is open at a
time. Six folds share that pool while each TPE study proposes sequentially.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import signal

import pandas as pd

import calibrate_fsim_joblib_local as base_runner
import calibrate_fsim_joblib_3p as wind_runner

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('2p_grid', '2p_tpe', '3p_grid', '3p_tpe', '4p_grid', '4p_tpe')


def dimension_settings(settings, dimension):
    s = copy.deepcopy(settings)
    s['arms'] = [f'{dimension}_grid', f'{dimension}_tpe']
    s['schedule'] = 'staged'
    if dimension != '3p':
        s['search_space'].pop('u_tr')
        s['initial_point'].pop('u_tr')
        s['local_grid']['coarse_values'].pop('u_tr')
    return s


def make_jobs(s, manifest):
    keys = manifest.input_key.tolist()
    if len(keys) != 6 or len(set(keys)) != 6 or manifest.input_key.isna().any():
        raise ValueError('LOSYO requires exactly six unique, nonmissing site-year keys')
    jobs = []
    for arm in s['arms']:
        seeds = s['seeds'] if arm.endswith('tpe') else [s['grid_order_seed']]
        for seed in seeds:
            for key in keys:
                jobs.append(dict(id=f'losyo/{arm}/seed_{seed}/folds/{key}',
                                 scheme='losyo', arm=arm, seed=seed, held_out=key,
                                 train=[k for k in keys if k != key], test=[key],
                                 n_trials=base_runner.trial_budget(s, arm)))
    return jobs


def smoke_settings(settings):
    s = copy.deepcopy(settings)
    s['search_space']['VPD_w'].update(grid_step=7.0)
    s['search_space']['I_req'].update(grid_step=410.0)
    s['search_space']['RH_sp'].update(low=80.0, high=85.0, grid_step=5.0)
    s['search_space']['MP_sp'].update(low=3, high=4, grid_step=1)
    s['initial_point'] = dict(VPD_w=1.0, I_req=40.0, RH_sp=80.0, MP_sp=3, u_tr=0.0)
    s['local_grid'].update(coarse_values=dict(VPD_w=[1.0, 8.0], I_req=[40.0, 450.0],
                                            u_tr=[0.0, 8.0]),
                           RH_values=[80.0, 85.0], MP_values=[3, 4],
                           max_candidates_per_fold=20, refinement_rounds=1)
    s.update(common_budget=4, n_startup_trials=2, tpe_trials=4,
             training_checkpoints=[4, 20], smoke=True)
    return s


def prepare(settings):
    if settings['schemes'] != ['losyo'] or settings['arms'] != list(ORDER):
        raise ValueError('Use six LOSYO folds and the fixed 2p/3p/4p Grid/TPE order')
    if settings['schedule'] != 'dimension_then_optimizer':
        raise ValueError('Use dimension_then_optimizer scheduling')
    manifest = pd.read_csv(ROOT / settings['manifest'])
    dimensions = []
    for dimension in ('2p', '3p', '4p'):
        runner = wind_runner if dimension == '3p' else base_runner
        s = dimension_settings(settings, dimension)
        # Existing validators also check legacy CLI schemes. All scientific
        # checks are reused; actual LOSYO jobs are constructed below, never by
        # the legacy site/year grouping function.
        validation = copy.deepcopy(s)
        validation['schemes'] = ['site']
        runner.validate_settings(validation)
        jobs = make_jobs(s, manifest)
        versions, frozen = runner.verify_engine(s)
        identity = runner.run_identity(s, jobs, versions, frozen)
        identity['source_sha256'][str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        dimensions.append((dimension, runner, s, jobs, identity))
    return manifest, dimensions


def run_dimension(item, cases, alternates, output, metadata, metadata_path, progress):
    dimension, runner, s, jobs, _ = item
    directory = output / dimension
    directory.mkdir(parents=True, exist_ok=True)
    store = runner.Store(directory / 'checkpoints.sqlite')
    evaluator = None
    try:
        base = runner.core.load_config(ROOT / s['parameter_table'])
        base['dispersal']['use_wind_gate'] = dimension == '3p'
        evaluator = runner.SharedEvaluator(cases, alternates, base, s['workers'], store)

        def complete(job, result):
            metadata['jobs'][job['id']] = result
            metadata.setdefault('unique_model_evaluations', {})[dimension] = store.evaluation_count()
            runner.core.write_json(metadata_path, metadata)
            print(f'Completed {job["id"]}: {result["n_trials"]} trials', flush=True)

        def stage_started(stage):
            metadata['active_stage'] = f'{dimension}_{stage}'
            runner.core.write_json(metadata_path, metadata)
            print(f'Starting {dimension} {stage.upper()}; shared pool limit={s["workers"]}', flush=True)

        runner.run_stages(s, jobs, {}, evaluator, store, directory, progress, complete, stage_started)
    finally:
        if evaluator is not None:
            evaluator.close()
        store.close()


def matched_budget(dimensions, output):
    """Choose the common cutoff from training trial counts, never test errors."""
    counts = []
    for dimension, runner, _, jobs, _ in dimensions:
        store = runner.Store(output / dimension / 'checkpoints.sqlite')
        try:
            for job in jobs:
                if store.heldout(job['id'], -1) is None:
                    raise ValueError(f'Incomplete final selection: {job["id"]}')
                counts.append(len(store.trials(job['id'])))
        finally:
            store.close()
    if not counts or min(counts) < 1:
        raise ValueError('No complete training trials')
    return min(counts)


def write_reports(settings, dimensions, cases, alternates, output):
    cutoff = matched_budget(dimensions, output)
    rows, folds, methods = [], [], []
    for dimension, runner, s, jobs, _ in dimensions:
        store = runner.Store(output / dimension / 'checkpoints.sqlite')
        evaluator = None
        try:
            # A matched cutoff may select a different training optimum and
            # require a new held-out model call. It never updates an optimizer.
            base = runner.core.load_config(ROOT / s['parameter_table'])
            base['dispersal']['use_wind_gate'] = dimension == '3p'
            evaluator = runner.SharedEvaluator(cases, alternates, base, s['workers'], store)
            for job in jobs:
                runner.select_budget(s, job, store.trials(job['id']), cutoff, evaluator, store,
                                     output / dimension / job['id'])
            for arm in s['arms']:
                seeds = s['seeds'] if arm.endswith('tpe') else [s['grid_order_seed']]
                for seed in seeds:
                    selected = [j for j in jobs if j['arm'] == arm and j['seed'] == seed]
                    for view, budget in [('matched', cutoff), ('final', -1)]:
                        subset, trial_counts = [], []
                        for job in selected:
                            held = store.heldout(job['id'], budget)
                            selection = held['selection']
                            tags = dict(scheme='losyo', arm=arm, seed=seed, view=view,
                                        held_out=job['held_out'], n_trials=selection['n_trials'])
                            trial_counts.append(selection['n_trials'])
                            folds.append(dict(**tags, **{k: v for k, v in selection.items() if k != 'n_trials'}))
                            for prediction in held['predictions']:
                                subset.append({**prediction, **tags,
                                               'calibration_status': selection['calibration_status'],
                                               'signed_interval_error_days': prediction.get('signed_interval_error_days')})
                        frame = pd.DataFrame(subset)
                        summary = runner.core.group_summary(frame, s, {j['held_out']: j['test'] for j in selected})
                        methods.append(dict(scheme='losyo', arm=arm, seed=seed, view=view,
                                            min_trials_per_fold=min(trial_counts), max_trials_per_fold=max(trial_counts),
                                            **summary))
                        rows.extend(subset)
        finally:
            if evaluator is not None:
                evaluator.close()
            store.close()
    frame = pd.DataFrame(rows)
    paired, comparisons = base_runner.compare_predictions(frame)
    for name, table in [('oof_predictions', frame), ('fold_summary', pd.DataFrame(folds)),
                        ('method_summary', pd.DataFrame(methods)), ('paired_predictions', paired),
                        ('comparison_summary', comparisons)]:
        table.to_csv(output / f'{name}.csv', index=False)
    base_runner.core.write_json(output / 'comparison_design.json', dict(
        matched_trials_per_fold=cutoff, cutoff_rule='minimum completed training trial count across all arms/folds/seeds',
        matched_tpe_within_startup=cutoff <= settings['n_startup_trials'],
        independent_units=6, execution_order=list(ORDER)))
    (output / 'comparison_report.md').write_text(
        '# LOSYO FSIM parameter and optimizer comparison\n\n'
        'Six site-years; each held-out site-year is excluded from its five-case training set.\n'
        f'Execution order: {", ".join(ORDER)}.\n\n'
        f'view=matched uses {cutoff} training trials per fold for every arm. The cutoff depends only on '
        'completed training trial counts, not held-out errors. '
        f'TPE startup trials: {settings["n_startup_trials"]}; a cutoff within startup does not assess adaptive TPE.\n'
        'view=final uses all completed trials: Grid is adaptive coarse-to-fine refinement with a cap; '
        'it can stop early. TPE uses its configured full budget. Check same_candidate_budget.\n'
        '2p: VPD/I, RH=80 and MP=3, wind gate off. '
        '3p: VPD/I/wind threshold, RH=80 and MP=3, wind gate on. '
        '4p: VPD/I/RH/MP, wind gate off. These are not a sequentially nested 2p/3p/4p chain.\n'
        'Grid RH and wind layers are discrete; TPE RH and wind thresholds are continuous. '
        'Equal budgets do not establish equal convergence.\n'
        'TPE starts from the configured data-independent point, not the preceding Grid optimum. '
        'No historical LOSYO result or held-out error is used to initialize or select parameters.\n'
        'Errors are signed distances between predicted and observed onset intervals; overlap has zero error. '
        'Positive paired delta means the right arm has larger absolute error. '
        'Failures remain in the paired tables; complete MAE requires six predictions.\n'
        'Seeds and optimizer arms are not independent biological replicates. This evaluates frozen FSIM, not FSIM-S.\n'
        'The old LOSYO 4p run used different RH/MP bounds and anchors; do not merge it into these results.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=ROOT / 'config/fsim_losyo_joblib.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, choices=base_runner.WORKERS)
    parser.add_argument('--design-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true', help='Rebuild reports; may evaluate a new matched held-out selection')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    if args.workers is not None:
        settings['workers'] = args.workers
    if args.smoke:
        settings = smoke_settings(settings)
    manifest, dimensions = prepare(settings)
    cases, errors, _, alternates, repairs = base_runner.core.load_cases(ROOT / settings['manifest'])
    if errors:
        raise ValueError(errors)
    if not settings['check_duplicate_policy']:
        alternates = []
    if {c['input_key'] for c in cases} != set(manifest.input_key):
        raise ValueError('Loaded cases differ from the six-fold manifest')
    identity = dict(format_version=1, order=list(ORDER), dimensions={d: i for d, _, _, _, i in dimensions})
    budget = dict(workers=settings['workers'], backend='joblib-loky', order=list(ORDER),
                  n_fold_jobs=sum(len(j) for _, _, _, j, _ in dimensions),
                  grid_candidate_cap=settings['local_grid']['max_candidates_per_fold'],
                  tpe_trials=settings['tpe_trials'], startup_trials=settings['n_startup_trials'],
                  coarse_candidates={d: len(r.fsim_local_grid.coarse_design(s, f'{d}_grid')[0]) for d, r, s, _, _ in dimensions},
                  maximum_training_trials=sum(j['n_trials'] for _, _, _, jobs, _ in dimensions for j in jobs))
    print(json.dumps(budget, indent=2), flush=True)
    if args.design_only:
        print('Design verified; no calibration started.', flush=True)
        return
    output = (args.output_dir or ROOT / ('result/fsim_losyo_joblib_smoke' if args.smoke else 'result/fsim_losyo_joblib')).resolve()
    resume = args.resume or args.report_only
    with base_runner.output_lock(output):
        identity_path = output / 'run_identity.json'
        if resume:
            if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
                raise ValueError('Resume requires identical inputs, code, versions and scientific settings; workers may change')
            for dimension, _, _, _, _ in dimensions:
                if not (output / dimension / 'checkpoints.sqlite').is_file():
                    # A future dimension may not yet have started.
                    if (output / dimension).exists():
                        raise ValueError(f'Missing checkpoint database: {dimension}')
        else:
            if any(p.name != '.run.lock' for p in output.iterdir()):
                raise ValueError('Use an empty output directory or --resume; historical results are protected')
            base_runner.core.write_json(identity_path, identity)
            base_runner.core.write_json(output / 'effective_settings.json', settings)
            base_runner.core.write_json(output / 'split_design.json', [j for _, _, _, jobs, _ in dimensions for j in jobs])
            manifest.to_csv(output / 'site_year_manifest.csv', index=False)
            repairs.to_csv(output / 'weather_duplicate_audit.csv', index=False)
            for _, _, _, _, item_identity in dimensions:
                for name in item_identity['source_sha256']:
                    target = output / 'source_snapshot' / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / name, target)
        metadata_path = output / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else dict(jobs={}, sessions=[])
        metadata.update(status='running', smoke=bool(args.smoke), budget=budget)
        metadata.pop('error', None)
        metadata['sessions'].append(dict(started_utc=base_runner.utc_now(), workers=settings['workers'], resumed=resume))
        base_runner.core.write_json(metadata_path, metadata)
        signal.signal(signal.SIGINT, base_runner.stop_on_signal)
        signal.signal(signal.SIGTERM, base_runner.stop_on_signal)
        base_runner.optuna.logging.set_verbosity(base_runner.optuna.logging.WARNING)
        try:
            if not args.report_only:
                for item in dimensions:
                    run_dimension(item, cases, alternates, output, metadata, metadata_path, not args.no_progress)
            metadata['active_stage'] = 'matched_budget_reporting'
            base_runner.core.write_json(metadata_path, metadata)
            write_reports(settings, dimensions, cases, alternates, output)
            metadata.update(status='complete', active_stage=None, completed_utc=base_runner.utc_now())
        except BaseException as exc:
            metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            raise
        finally:
            base_runner.core.write_json(metadata_path, metadata)
    print(output, flush=True)


if __name__ == '__main__':
    main()
