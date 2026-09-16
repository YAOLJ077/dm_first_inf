#!/usr/bin/env python3
"""Four-parameter constrained TPE for M0, with six training-only LOSYO folds.

Sequential ask/tell; four processes evaluate site-years within each trial.
Completed trials are atomically checkpointed and replayed on --resume, including
sampler calls, so the proposal sequence is reproducible with unchanged versions.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model.parameter_table import load_config
from src.analyze_fsim_parameters import configure, evaluate_group, load_cases
from src.calibrate_m0_losyo import summarize_oof

NAMES = ('VPD_w', 'I_req', 'RH_sp', 'MP_sp')


def write_json(path, value):
    """Atomic checkpoints: an interrupted trial can be recomputed safely."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate_settings(s):
    if list(s['search_space']) != list(NAMES):
        raise ValueError('Exactly VPD_w, I_req, RH_sp and MP_sp must be configured in order')
    for name, spec in s['search_space'].items():
        lo, hi = spec['low'], spec['high']
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            raise ValueError(f'Invalid bounds: {name}')
        expected = 'int' if name == 'MP_sp' else 'float'
        if spec['type'] != expected:
            raise ValueError(f'{name} must use {expected} sampling')
        if expected == 'int' and (type(lo) is not int or type(hi) is not int or spec.get('step') != 1):
            raise ValueError('MP requires integer bounds and step=1')
    if s['search_space']['RH_sp']['high'] > 100 or s['search_space']['MP_sp']['high'] > 12:
        raise ValueError('RH or night duration outside its physical domain')
    if any(type(s[k]) is not int or s[k] < 1 for k in ('n_trials', 'n_startup_trials')):
        raise ValueError('Trial counts must be positive integers')
    if s['n_startup_trials'] > s['n_trials'] or s['workers'] not in (2, 4):
        raise ValueError('Invalid startup budget or workers')
    if not s['seeds'] or len(set(s['seeds'])) != len(s['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in s['seeds']):
        raise ValueError('Invalid seed')
    if s['objective'] != 'training_interval_MAE_days' or s['require_all_training_predictions'] is not True:
        raise ValueError('Requires training interval MAE and complete training predictions')
    if type(s['check_duplicate_policy']) is not bool:
        raise ValueError('Duplicate policy check must be boolean')
    if not np.isfinite(s['tie_atol_days']) or s['tie_atol_days'] < 0:
        raise ValueError('Invalid tie tolerance')
    for key in ('accuracy_tolerances_days', 'near_optimal_tolerances_days'):
        if not s[key] or any(not np.isfinite(v) or v < 0 for v in s[key]):
            raise ValueError(f'Invalid {key}')
    if any(type(v) is not int or not 0 < v <= s['n_trials'] for v in s['checkpoints']):
        raise ValueError('Invalid checkpoints')


def runtime_config(base, params):
    cfg = configure(base, {'model': 'M0', **params})
    cfg['mp_tmp']['rh_wet_night'] = params['RH_sp']
    cfg['fixed']['min_moist_hours'] = params['MP_sp']
    return cfg


def worker_init(cases, base):
    global _CASES, _BASE
    _CASES, _BASE = cases, base


def worker_evaluate(task):
    cid, params = task
    # One sample per fresh engine: never reuse upstream states across RH/MP.
    sample = {'model': 'M0', 'sample_id': 'tpe', **params}
    return evaluate_group((_CASES[cid], [sample], runtime_config(_BASE, params)))[0]


def training_metrics(records, train_keys):
    if (len(train_keys) != 5 or len(set(train_keys)) != 5
            or len(records) != len(train_keys) or {r['input_key'] for r in records} != set(train_keys)):
        raise ValueError('Training records must cover exactly the five training site-years')
    if any(r['status'] not in ('predicted', 'no_prediction') for r in records):
        raise RuntimeError('Model runtime error; stop rather than classify as no prediction')
    missing = sum(r['status'] == 'no_prediction' for r in records)
    errors = [float(r['signed_interval_error_days']) for r in records if r['status'] == 'predicted']
    if not np.isfinite(errors).all():
        raise ValueError('Non-finite error for a predicted window')
    mae = None if missing else float(np.mean(np.abs(errors)))
    return dict(feasible=missing == 0, n_no_prediction=missing,
                training_MAE_days=mae,
                training_RMSE_days=None if missing else float(np.sqrt(np.mean(np.square(errors)))),
                # This placeholder is exclusively optimizer-internal. Positive
                # constraints exclude it from feasible TPE ranking and selection.
                optimizer_value=0.0 if missing else mae)


def constraints(trial):
    return (trial.user_attrs['n_no_prediction'],)


def make_study(seed, s, anchor):
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                         n_startup_trials=s['n_startup_trials'],
                                         constraints_func=constraints)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.enqueue_trial(anchor)
    return study


def ask_params(study, space):
    trial = study.ask()
    params = {}
    for name, spec in space.items():
        if spec['type'] == 'int':
            params[name] = trial.suggest_int(name, spec['low'], spec['high'], step=spec['step'])
        else:
            params[name] = trial.suggest_float(name, spec['low'], spec['high'])
    return trial, params


def tell(study, trial, metrics):
    trial.set_user_attr('n_no_prediction', metrics['n_no_prediction'])
    trial.set_user_attr('training_MAE_days', metrics['training_MAE_days'])
    study.tell(trial, metrics['optimizer_value'])


def choose_training(trials, atol):
    """No held-out predictions enter this selector, including across seeds."""
    valid = [t for t in trials if t['metrics']['feasible']]
    if not valid:
        return None, []
    best = min(t['metrics']['training_MAE_days'] for t in valid)
    ties = [t for t in valid if t['metrics']['training_MAE_days'] <= best + atol]
    def order(t):
        p = t['params']
        return (abs(p['MP_sp'] - 3), abs(p['RH_sp'] - 80), p['RH_sp'],
                p['VPD_w'], p['I_req'], t['seed'], t['number'])
    ties.sort(key=order)
    return ties[0], ties


class Evaluator:
    def __init__(self, cases, alternates, base, workers):
        self.base = base
        self.cases = cases + alternates
        self.primary = {c['input_key']: i for i, c in enumerate(cases)}
        self.alternate = {c['input_key']: i + len(cases) for i, c in enumerate(alternates)}
        self.pool = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init, initargs=(self.cases, base))

    def evaluate(self, params, keys):
        ids = [self.primary[k] for k in keys] + [self.alternate[k] for k in keys if k in self.alternate]
        # map returns in submitted order regardless of process completion order.
        rows = list(self.pool.map(worker_evaluate, [(cid, params) for cid in ids]))
        for r in rows:
            if r['status'] not in ('predicted', 'no_prediction'):
                raise RuntimeError(json.dumps(r))
        primary = rows[:len(keys)]
        alt = {r['input_key']: r for r in rows[len(keys):]}
        for r in primary:
            if r['input_key'] in alt:
                if any(r.get(col) != alt[r['input_key']].get(col)
                       for col in ('status', 'predicted_start', 'predicted_end')):
                    raise ValueError(f'Duplicate weather policies disagree: {r["input_key"]}')
        return primary, rows[len(keys):]

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def run_seed(s, seed, anchor, keys, directory, evaluator, progress=True):
    directory.mkdir(parents=True, exist_ok=True)
    study = make_study(seed, s, anchor)
    trials = []
    saved = sorted(directory.glob('trial_*.json'))
    if [p.name for p in saved] != [f'trial_{i:04d}.json' for i in range(len(saved))]:
        raise ValueError('Non-contiguous trial checkpoints')
    if len(saved) > s['n_trials']:
        raise ValueError('Checkpoint exceeds configured budget')
    with tqdm(total=s['n_trials'], desc=f'{directory.parent.name} seed={seed}',
              unit='trial', disable=not progress, dynamic_ncols=True) as bar:
        for i in range(s['n_trials']):
            trial, params = ask_params(study, s['search_space'])
            path = directory / f'trial_{i:04d}.json'
            if path.exists():
                record = json.loads(path.read_text())
                if (record['number'] != trial.number or record['seed'] != seed
                        or record['params'] != params or record['training_keys'] != keys):
                    raise ValueError('Sampler replay differs; keep software, settings and inputs unchanged')
                if training_metrics(record['predictions'], keys) != record['metrics']:
                    raise ValueError('Invalid saved training metrics')
            else:
                predictions, duplicate_predictions = evaluator.evaluate(params, keys)
                record = dict(number=trial.number, seed=seed, params=params,
                              training_keys=keys, predictions=predictions,
                              duplicate_predictions=duplicate_predictions,
                              metrics=training_metrics(predictions, keys))
                write_json(path, record)
            tell(study, trial, record['metrics'])
            trials.append(record)
            choice, _ = choose_training(trials, s['tie_atol_days'])
            bar.update(1)
            bar.set_postfix(best_MAE='NA' if choice is None else f'{choice["metrics"]["training_MAE_days"]:.3f}',
                            feasible=f'{sum(t["metrics"]["feasible"] for t in trials)}/{len(trials)}', refresh=False)
    return trials


def flat_trial(t):
    return {'seed': t['seed'], 'trial': t['number'], **t['params'], **t['metrics']}


def save_training_reports(directory, trials, s):
    frame = pd.DataFrame([flat_trial(t) for t in trials])
    frame.to_csv(directory / 'trials.csv', index=False)
    curve = []
    for seed, group in frame.groupby('seed', sort=False):
        best = group.training_MAE_days.cummin().ffill()
        for i, (_, row) in enumerate(group.iterrows()):
            curve.append(dict(seed=seed, completed_trials=i+1,
                              best_training_MAE_days=best.iloc[i],
                              n_feasible=int(group.feasible.iloc[:i+1].sum())))
    convergence = pd.DataFrame(curve)
    convergence.to_csv(directory / 'convergence.csv', index=False)
    convergence[convergence.completed_trials.isin(s['checkpoints'])].to_csv(directory / 'checkpoints.csv', index=False)
    choice, ties = choose_training(trials, s['tie_atol_days'])
    pd.DataFrame([flat_trial(t) for t in ties]).to_csv(directory / 'tied_best_parameters.csv', index=False)
    if choice is not None:
        best = choice['metrics']['training_MAE_days']
        near = []
        for tol in s['near_optimal_tolerances_days']:
            q = frame[frame.feasible & frame.training_MAE_days.le(best + tol + s['tie_atol_days'])].copy()
            q['tolerance_days'] = tol
            near.append(q)
        pd.concat(near, ignore_index=True).to_csv(directory / 'near_optimal_parameters.csv', index=False)
    return choice


def read_anchors(s, keys):
    parent = ROOT / s['anchor_run']
    meta = json.loads((parent / 'run_metadata.json').read_text())
    if meta['status'] != 'complete' or meta.get('smoke') or meta.get('model') != 'M0':
        raise ValueError('Anchor must be a completed formal M0 run')
    for name, digest in meta['input_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Anchor input has changed: {name}')
    if (ROOT / s['manifest']).read_bytes() != (parent / 'site_year_manifest.csv').read_bytes():
        raise ValueError('Anchor manifest differs')
    anchors = {}
    for key in keys:
        selection = json.loads((parent / 'folds' / key / 'selection.json').read_text())
        if selection['held_out'] != key or set(selection['training_keys']) != set(keys) - {key}:
            raise ValueError('Anchor training fold mismatch')
        params = {name: selection[name] for name in NAMES[:2]}
        params.update(RH_sp=80, MP_sp=3)
        for name, value in params.items():
            spec = s['search_space'][name]
            if not spec['low'] <= value <= spec['high']:
                raise ValueError(f'Anchor outside search space: {key} {name}')
        anchors[key] = params
    return anchors


def fingerprint(s, settings_path, anchors):
    sources = list((ROOT / 'model').glob('*.py')) + [Path(__file__),
        ROOT / 'src/analyze_fsim_parameters.py', ROOT / 'src/calibrate_m0_losyo.py',
        ROOT / s['parameter_table'], ROOT / 'config/fsim_runtime_parameters.csv', settings_path]
    manifest = ROOT / s['manifest']
    inputs = [manifest] + [ROOT / name for name in pd.read_csv(manifest).weather_json]
    hashes = lambda paths: {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(settings=s, anchors=anchors, source_sha256=hashes(sources), input_sha256=hashes(inputs),
                versions={'python': platform.python_version(), **{n: importlib.metadata.version(n)
                          for n in ('optuna', 'numpy', 'pandas', 'tqdm', 'scipy')}})


def run_folds(s, cases, anchors, out, evaluator, progress):
    keys = [c['input_key'] for c in cases]
    oof, selections, count, duplicate_count = [], [], 0, 0
    for key in keys:
        directory = out / 'folds' / key
        directory.mkdir(parents=True, exist_ok=True)
        train_keys = [k for k in keys if k != key]
        trials = []
        for seed in s['seeds']:
            trials.extend(run_seed(s, seed, anchors[key], train_keys, directory / f'seed_{seed}', evaluator, progress))
        count += sum(len(t['predictions']) + len(t['duplicate_predictions']) for t in trials)
        duplicate_count += sum(len(t['duplicate_predictions']) for t in trials)
        choice = save_training_reports(directory, trials, s)
        selection = dict(held_out=key, training_keys=train_keys,
                         calibration_status='selected' if choice else 'no_feasible_training_combination')
        if choice is not None:
            selection.update(**choice['params'], seed=choice['seed'], trial=choice['number'],
                             selected_training_MAE_days=choice['metrics']['training_MAE_days'])
        # Freeze selection before any access to this fold's held-out evaluation.
        write_json(directory / 'selection.json', selection)
        selections.append(selection)
        if choice is None:
            record = dict(input_key=key, status='calibration_failed', signed_interval_error_days=None)
        else:
            pd.DataFrame(choice['predictions']).to_csv(directory / 'training_predictions.csv', index=False)
            write_json(directory / 'effective_runtime_config.json', runtime_config(evaluator.base, choice['params']))
            checkpoint = directory / 'held_out_checkpoint.json'
            if checkpoint.exists():
                held = json.loads(checkpoint.read_text())
                if held['params'] != choice['params'] or held['held_out'] != key:
                    raise ValueError('Held-out checkpoint does not match frozen selection')
            else:
                predictions, duplicates = evaluator.evaluate(choice['params'], [key])
                held = dict(params=choice['params'], held_out=key, predictions=predictions, duplicate_predictions=duplicates)
                write_json(checkpoint, held)
            count += len(held['predictions']) + len(held['duplicate_predictions'])
            duplicate_count += len(held['duplicate_predictions'])
            record = held['predictions'][0]
            record.setdefault('signed_interval_error_days', None)
        record = {**record, 'calibration_status': selection['calibration_status']}
        pd.DataFrame([record]).to_csv(directory / 'held_out_prediction.csv', index=False)
        oof.append(record)
    frame = pd.DataFrame(oof)
    frame.to_csv(out / 'oof_predictions.csv', index=False)
    pd.DataFrame(selections).to_csv(out / 'fold_summary.csv', index=False)
    summary = summarize_oof(frame, s)
    write_json(out / 'oof_summary.json', summary)
    return summary, count, duplicate_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=ROOT / 'config/fsim_m0_losyo_tpe.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/fsim_m0_losyo_tpe_4p_n512')
    parser.add_argument('--workers', type=int, choices=(2, 4))
    parser.add_argument('--resume', action='store_true', help='Replay completed trials; continue an unchanged run')
    parser.add_argument('--design-only', action='store_true', help='Validate inputs/settings and print budget; no calibration or output directory')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    s = json.loads(args.settings.read_text())
    if args.workers is not None:
        s['workers'] = args.workers
    validate_settings(s)
    base = load_config(ROOT / s['parameter_table'])
    base['dispersal']['use_wind_gate'] = False
    cases, errors, _, alternates, repairs = load_cases(ROOT / s['manifest'])
    if errors:
        raise ValueError(errors)
    if not s['check_duplicate_policy']:
        alternates = []
    keys = [c['input_key'] for c in cases]
    anchors = read_anchors(s, keys)
    identity = fingerprint(s, args.settings.resolve(), anchors)
    budget = dict(parameter_trials=6 * len(s['seeds']) * s['n_trials'],
                  main_training_evaluations=30 * len(s['seeds']) * s['n_trials'],
                  duplicate_training_evaluations=5 * len(alternates) * len(s['seeds']) * s['n_trials'],
                  maximum_held_out_evaluations=6 + len(alternates), workers=s['workers'])
    print(json.dumps(budget, indent=2), flush=True)
    if args.design_only:
        print('Design validated; no calibration performed and no result directory created.')
        return
    out = args.output_dir.resolve()
    if args.resume:
        if not (out / 'run_identity.json').is_file():
            parser.error('--resume requires an existing checkpoint directory')
    elif out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory, or --resume for the same run')
    out.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers, including a second --resume command.
    with (out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another process is already using this output directory')
        if args.resume:
            if json.loads((out / 'run_identity.json').read_text()) != identity:
                raise ValueError('Resume requires identical settings, sources, inputs, anchors and package versions')
        else:
            for name in identity['source_sha256']:
                path = Path(name)
                rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('external') / path.name
                dest = out / 'source_snapshot' / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            shutil.copy2(ROOT / s['manifest'], out / 'site_year_manifest.csv')
            write_json(out / 'effective_settings.json', s)
            write_json(out / 'run_identity.json', identity)
            repairs.to_csv(out / 'weather_duplicate_audit.csv', index=False)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        metadata = dict(status='running', model='M0', started_utc=datetime.now(timezone.utc).isoformat(),
                        resumed=args.resume, budget=budget, settings=s)
        write_json(out / 'run_metadata.json', metadata)
        evaluator = Evaluator(cases, alternates, base, s['workers'])
        try:
            summary, count, duplicate_count = run_folds(s, cases, anchors, out, evaluator, not args.no_progress)
            metadata.update(status='complete', n_model_evaluations_in_completed_records=count,
                            n_duplicate_evaluations=duplicate_count, duplicate_policy_differences=0,
                            completed_utc=datetime.now(timezone.utc).isoformat())
            print(json.dumps(summary, indent=2), flush=True)
        except BaseException as exc:
            metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                            error_type=type(exc).__name__, error_message=str(exc))
            raise
        finally:
            evaluator.close()
            write_json(out / 'run_metadata.json', metadata)


if __name__ == '__main__':
    main()
