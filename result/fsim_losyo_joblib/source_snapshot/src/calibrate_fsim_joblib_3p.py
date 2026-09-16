#!/usr/bin/env python3
"""Independent 3p FSIM Grid/TPE: VPD, infection threshold and wind threshold.

All arms use the same frozen engine. SQLite holds exact model evaluations and
atomic trial checkpoints. Only training keys are exposed to an optimizer.
The process pool is global; each fold/study uses sequential proposals.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import calibrate_m0_grouped_tpe as core
import fsim_3p_grid as fsim_local_grid
import fsim_grid_global
import fsim_joblib_pool
from fsim_joblib_pool import JoblibPool

NAMES = (*core.NAMES, 'u_tr')
ARMS = ('3p_grid', '3p_tpe')
WORKERS = (1, 2, 4, 8, 16, 32, 64)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stop_on_signal(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}; retaining checkpoints')


def json_text(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def active_names(arm):
    return ('VPD_w', 'I_req', 'u_tr')


def grid_axis(spec):
    count = (spec['high'] - spec['low']) / spec['grid_step']
    if not np.isclose(count, round(count), atol=1e-9, rtol=0):
        raise ValueError('Grid step must divide its range exactly')
    values = np.linspace(spec['low'], spec['high'], round(count) + 1)
    return [int(v) if spec['type'] == 'int' else float(v) for v in values]


def grid_candidates(s, arm):
    names = active_names(arm)
    candidates = [dict(s['fixed_2p'], **dict(zip(names, point)))
                  for point in itertools.product(*(grid_axis(s['search_space'][n]) for n in names))]
    # All methods pay for the same data-independent first trial. Shuffle the
    # remaining grid without using outcomes, so early budgets do not scan one edge.
    initial = s['initial_point']
    index = next(i for i, p in enumerate(candidates) if p == initial)
    first = candidates.pop(index)
    order = np.random.default_rng(s['grid_order_seed']).permutation(len(candidates))
    return [first] + [candidates[int(i)] for i in order]


def trial_budget(s, arm):
    return s['local_grid']['max_candidates_per_fold'] if arm.endswith('grid') else s['tpe_trials']


def validate_settings(s):
    local = s['local_grid']
    wind = s['search_space']['u_tr']
    if wind != dict(low=0, high=8, type='float', grid_step=1):
        raise ValueError('Wind requires continuous bounds 0-8 and Grid step 1')
    expected_wind = [0.0, 8.0] if s.get('smoke') else grid_axis(wind)
    if local['coarse_values']['u_tr'] != expected_wind:
        raise ValueError('Grid wind axis must cover 0-8 at step 1 (endpoints only in smoke mode)')
    if type(local['max_candidates_per_fold']) is not int or local['max_candidates_per_fold'] < 1:
        raise ValueError('Invalid Grid candidate cap')
    if type(local['refinement_rounds']) is not int or not 1 <= local['refinement_rounds'] <= 8:
        raise ValueError('Invalid refinement rounds')
    if not np.isfinite(local['margin_days']) or local['margin_days'] < 0:
        raise ValueError('Invalid refinement margin')
    for n in NAMES[:2]:
        axis = local['coarse_values'][n]
        if len(axis) < 2 or axis != sorted(set(axis)) or not all(np.isfinite(v) for v in axis):
            raise ValueError('Invalid coarse axis')
        if axis[0] != s['search_space'][n]['low'] or axis[-1] != s['search_space'][n]['high']:
            raise ValueError('Coarse grid must cover both endpoints')
        if s['initial_point'][n] not in axis:
            raise ValueError('Initial point must be in coarse grid')
    for n, field in [('RH_sp', 'RH_values'), ('MP_sp', 'MP_values')]:
        if local[field] != grid_axis(s['search_space'][n]):
            raise ValueError('Discrete Grid axes must match the configured steps')
    for arm in s['arms']:
        if arm.endswith('grid') and len(fsim_local_grid.coarse_design(s, arm)[0]) > trial_budget(s, arm):
            raise ValueError('Coarse design exceeds Grid budget')
    if s['schedule'] != 'staged':
        raise ValueError('Unknown schedule')
    if s['workers'] not in WORKERS:
        raise ValueError(f'workers must be one of {WORKERS}')
    for field, allowed in [('schemes', {'site', 'year'}), ('arms', set(ARMS))]:
        if not s[field] or len(set(s[field])) != len(s[field]) or not set(s[field]) <= allowed:
            raise ValueError(f'Invalid {field}')
    if list(s['search_space']) != list(NAMES):
        raise ValueError('Expected VPD_w, I_req, RH_sp, MP_sp, u_tr')
    if set(s['fixed_2p']) != set(core.NAMES[2:]) or set(s['initial_point']) != set(NAMES):
        raise ValueError('Invalid fixed parameters or initial point')
    for name, spec in s['search_space'].items():
        if spec['type'] != ('int' if name == 'MP_sp' else 'float'):
            raise ValueError(f'Wrong parameter type: {name}')
        if not all(np.isfinite(spec[k]) for k in ('low', 'high', 'grid_step')) or not (0 <= spec['low'] < spec['high'] if name == 'u_tr' else 0 < spec['low'] < spec['high']) or spec['grid_step'] <= 0:
            raise ValueError(f'Invalid range/step: {name}')
        if name == 'MP_sp' and any(type(spec[k]) is not int for k in ('low', 'high', 'grid_step')):
            raise ValueError('MP_sp requires integer bounds and step')
        if s['initial_point'][name] not in grid_axis(spec):
            raise ValueError(f'Initial point must lie on the grid: {name}')
    if s['fixed_2p'] != {n: s['initial_point'][n] for n in core.NAMES[2:]}:
        raise ValueError('Two-parameter slice must contain the common initial point')
    if s['fixed_2p'] != {'RH_sp': 80.0, 'MP_sp': 3}:
        raise ValueError('Two-parameter control fixes RH_sp=80 and MP_sp=3')
    if s['tpe_trials'] != 'match_grid' and (type(s['tpe_trials']) is not int or s['tpe_trials'] < 1):
        raise ValueError('tpe_trials must be match_grid or a positive integer')
    if type(s['n_startup_trials']) is not int or s['n_startup_trials'] < 1:
        raise ValueError('Invalid startup budget')
    if type(s['common_budget']) is not int or not 1 <= s['common_budget'] <= min(trial_budget(s, a) for a in s['arms']):
        raise ValueError('Common budget must be achievable by every selected arm')
    if any(s['n_startup_trials'] > trial_budget(s, a) for a in s['arms'] if a.endswith('tpe')):
        raise ValueError('Startup exceeds TPE trial budget')
    if not s['seeds'] or len(set(s['seeds'])) != len(s['seeds']):
        raise ValueError('Use one or more unique TPE seeds')
    if any(type(v) is not int or not 0 <= v < 2**32 for v in s['seeds'] + [s['grid_order_seed']]):
        raise ValueError('Invalid seed')
    if type(s['check_duplicate_policy']) is not bool:
        raise ValueError('Invalid duplicate policy')
    for key in ('near_optimal_tolerances_days', 'accuracy_tolerances_days'):
        if not s[key] or any(not np.isfinite(v) or v < 0 for v in s[key]):
            raise ValueError(f'Invalid {key}')
    if not np.isfinite(s['tie_atol_days']) or s['tie_atol_days'] < 0:
        raise ValueError('Invalid tie tolerance')
    if any(type(v) is not int or v < 1 for v in s['training_checkpoints']):
        raise ValueError('Invalid training checkpoints')


def make_jobs(s, manifest):
    keys = manifest.input_key.tolist()
    jobs = []
    for scheme in s['schemes']:
        for arm in s['arms']:
            seeds = s['seeds'] if arm.endswith('tpe') else [s['grid_order_seed']]
            for seed in seeds:
                for group, test in core.make_groups(manifest, scheme).items():
                    jobs.append(dict(id=f'{scheme}/{arm}/seed_{seed}/folds/{group}',
                                     scheme=scheme, arm=arm, seed=seed, held_out=group,
                                     train=[k for k in keys if k not in test], test=test,
                                     n_trials=trial_budget(s, arm)))
    return jobs


def verify_engine(s):
    frozen = json.loads((core.REFERENCE / 'snapshot_manifest.json').read_text())
    for name, digest in frozen['sha256'].items():
        if hashlib.sha256((core.REFERENCE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen model changed: {name}')
    if (ROOT / s['parameter_table']).resolve() != core.REFERENCE / 'config/fsim_m0_losyo_parameters.csv':
        raise ValueError('Use the shared frozen parameter table for all arms')
    versions = {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'optuna', 'tqdm', 'joblib')}}
    for name, version in s['required_versions'].items():
        if versions.get(name) != version:
            raise ValueError(f'Required {name}={version}; found {versions.get(name)}')
    return versions, frozen


def run_identity(s, jobs, versions, frozen):
    sources = [Path(__file__), Path(fsim_joblib_pool.__file__), Path(fsim_grid_global.__file__), Path(fsim_local_grid.__file__), Path(core.__file__), core.REFERENCE / 'snapshot_manifest.json']
    sources += [core.REFERENCE / p for p in frozen['sha256']]
    manifest = ROOT / s['manifest']
    inputs = [manifest] + [ROOT / p for p in pd.read_csv(manifest).weather_json]
    def hashes(paths):
        return {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    scientific = copy.deepcopy(s)
    # Worker count changes scheduling only; each study still has sequential
    # ask/tell, fixed seeds and an outcome-independent grid order.
    scientific.pop('workers')
    return dict(format_version=1, backend='joblib-loky', settings=scientific, jobs=jobs, versions=versions,
                source_sha256=hashes(sources), input_sha256=hashes(inputs))


class Store:
    """One writer connection per run, serialized across coordinator threads."""
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        # A single owner process and DELETE journaling also allow ordinary copies
        # after shutdown; never copy this database while the run is active.
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS evaluations (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS trials (job TEXT, number INTEGER, payload TEXT NOT NULL, PRIMARY KEY(job, number))')
        self.db.execute('CREATE TABLE IF NOT EXISTS heldout (job TEXT, budget INTEGER, payload TEXT NOT NULL, PRIMARY KEY(job, budget))')
        self.db.commit()

    def evaluation(self, key):
        with self.lock:
            row = self.db.execute('SELECT payload FROM evaluations WHERE key=?', (key,)).fetchone()
            return None if row is None else json.loads(row[0])

    def save_evaluation(self, key, payload):
        with self.lock:
            self.db.execute('INSERT OR IGNORE INTO evaluations VALUES (?, ?)', (key, json_text(payload)))
            self.db.commit()

    def trials(self, job):
        with self.lock:
            rows = self.db.execute('SELECT number, payload FROM trials WHERE job=? ORDER BY number', (job,)).fetchall()
        if [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError(f'Non-contiguous checkpoints: {job}')
        return [json.loads(row[1]) for row in rows]

    def save_trial(self, job, record):
        with self.lock:
            self.db.execute('INSERT INTO trials VALUES (?, ?, ?)', (job, record['number'], json_text(record)))
            self.db.commit()

    def heldout(self, job, budget):
        with self.lock:
            row = self.db.execute('SELECT payload FROM heldout WHERE job=? AND budget=?', (job, budget)).fetchone()
            return None if row is None else json.loads(row[0])

    def save_heldout(self, job, budget, payload):
        with self.lock:
            self.db.execute('INSERT INTO heldout VALUES (?, ?, ?)', (job, budget, json_text(payload)))
            self.db.commit()

    def evaluation_count(self):
        with self.lock:
            return self.db.execute('SELECT count(*) FROM evaluations').fetchone()[0]

    def close(self):
        self.db.close()


def runtime_config(base, params):
    cfg = core.configure(base, {'model': 'M1', **params})
    cfg['mp_tmp']['rh_wet_night'] = params['RH_sp']
    cfg['fixed']['min_moist_hours'] = params['MP_sp']
    return cfg


def timed_worker(task):
    started = time.perf_counter()
    cid, params = task
    sample = {'model': 'M1', 'sample_id': '3p', **params}
    record = core.evaluate_group((core._CASES[cid], [sample], runtime_config(core._BASE, params)))[0]
    return dict(record=record, model_seconds=time.perf_counter() - started, worker_pid=os.getpid())


class SharedEvaluator:
    def __init__(self, cases, alternates, base, workers, store):
        self.store, self.base = store, base
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.inflight = {}
        self.cases = cases + alternates
        self.primary = {c['input_key']: i for i, c in enumerate(cases)}
        self.alternate = {c['input_key']: i + len(cases) for i, c in enumerate(alternates)}
        self.pool = JoblibPool(max_workers=workers,
                                        initializer=core.worker_init, initargs=(self.cases, base))

    def cache_key(self, cid, params):
        value = [self.cases[cid]['input_key'], 'alternate' if cid >= len(self.primary) else 'primary',
                 [float(params[n]) for n in NAMES]]
        return hashlib.sha256(json_text(value).encode()).hexdigest()

    def evaluate(self, params, keys, cached_only=False):
        if self.stop.is_set():
            raise RuntimeError('Run stopping; completed checkpoints retained')
        ids = [self.primary[k] for k in keys] + [self.alternate[k] for k in keys if k in self.alternate]
        requested = []
        for cid in ids:
            key = self.cache_key(cid, params)
            with self.lock:
                cached = self.store.evaluation(key)
                if cached is not None:
                    requested.append((cid, key, cached, None))
                    continue
                if cached_only:
                    raise ValueError(f'Missing cached evaluation for a saved trial: {key}')
                if key not in self.inflight:
                    self.inflight[key] = self.pool.submit(timed_worker, (cid, params))
                requested.append((cid, key, None, self.inflight[key]))
        rows = []
        for cid, key, cached, future in requested:
            payload = cached if cached is not None else future.result()
            record = payload['record']
            if record['input_key'] != self.cases[cid]['input_key'] or any(record[n] != params[n] for n in NAMES):
                raise ValueError(f'Cached model identity mismatch: {key}')
            if record['status'] not in ('predicted', 'no_prediction'):
                raise RuntimeError(json_text(record))
            if cached is None:
                with self.lock:
                    self.store.save_evaluation(key, payload)
                    self.inflight.pop(key, None)
            rows.append(record)
        primary, alternate = rows[:len(keys)], rows[len(keys):]
        lookup = {r['input_key']: r for r in alternate}
        for record in primary:
            other = lookup.get(record['input_key'])
            if other is not None and any(record.get(n) != other.get(n) for n in ('status', 'predicted_start', 'predicted_end')):
                raise ValueError(f'Duplicate weather policies disagree: {record["input_key"]}')
        return primary, alternate

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


def make_study(s, job):
    sampler = optuna.samplers.TPESampler(seed=job['seed'], multivariate=True,
                                        n_startup_trials=s['n_startup_trials'], constraints_func=core.constraints)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.enqueue_trial({n: s['initial_point'][n] for n in active_names(job['arm'])})
    return study


def propose(s, job, study):
    trial = study.ask()
    params = dict(s['fixed_2p'])
    for name in active_names(job['arm']):
        spec = s['search_space'][name]
        params[name] = (trial.suggest_int(name, spec['low'], spec['high']) if spec['type'] == 'int'
                        else trial.suggest_float(name, spec['low'], spec['high']))
    return trial, params


def select_budget(s, job, records, budget, evaluator, store, directory):
    choice, ties = core.choose_training(records[:budget], s['tie_atol_days'])
    selection = dict(training_keys=job['train'], test_keys=job['test'], n_trials=budget,
                     calibration_status='selected' if choice else 'no_feasible_training_combination')
    if choice is not None:
        selection.update(**choice['params'], selected_trial=choice['number'],
                         training_MAE_days=choice['metrics']['training_MAE_days'])
    target = directory / f'budget_{budget}'
    target.mkdir(parents=True, exist_ok=True)
    # Persist training-only selection before any held-out call, including replay.
    core.write_json(target / 'selection.json', selection)
    pd.DataFrame([core.flat_trial(t) for t in ties]).to_csv(target / 'tied_best_parameters.csv', index=False)
    if choice is not None:
        near = [dict(tolerance_days=tol, **core.flat_trial(t)) for tol in s['near_optimal_tolerances_days']
                for t in records[:budget] if t['metrics']['feasible'] and
                t['metrics']['training_MAE_days'] <= choice['metrics']['training_MAE_days'] + tol + s['tie_atol_days']]
        pd.DataFrame(near).to_csv(target / 'near_optimal_parameters.csv', index=False)
        core.write_json(target / 'effective_runtime_config.json', runtime_config(evaluator.base, choice['params']))
    held = store.heldout(job['id'], budget)
    if held is not None:
        if held['selection'] != selection:
            raise ValueError('Saved held-out selection differs from replay')
        pd.DataFrame(held['predictions']).to_csv(target / 'held_out_predictions.csv', index=False)
        return held
    if choice is None:
        predictions = [dict(input_key=k, status='calibration_failed', signed_interval_error_days=None) for k in job['test']]
        duplicates = []
    else:
        predictions, duplicates = evaluator.evaluate(choice['params'], job['test'])
    held = dict(selection=selection, predictions=predictions, duplicate_predictions=duplicates)
    store.save_heldout(job['id'], budget, held)
    pd.DataFrame(predictions).to_csv(target / 'held_out_predictions.csv', index=False)
    return held


def run_job(s, job, candidates, evaluator, store, output, progress=True, position=0):
    directory = output / job['id']
    directory.mkdir(parents=True, exist_ok=True)
    core.write_json(directory / 'job.json', job)
    saved = store.trials(job['id'])
    if len(saved) > job['n_trials']:
        raise ValueError('Saved trials exceed configured budget')
    study = make_study(s, job) if job['arm'].endswith('tpe') else None
    records, curve = [], []
    best, feasible, logical, elapsed = None, 0, 0, 0.0
    label = f'{job["scheme"]}/{job["arm"]}/{job["held_out"]}/{job["seed"]}'
    with tqdm(total=job['n_trials'], desc=label, unit='trial', position=position,
              disable=not progress, dynamic_ncols=True) as bar:
        for i in range(job['n_trials']):
            if evaluator.stop.is_set():
                raise RuntimeError('Run stopping')
            trial, params = propose(s, job, study) if study is not None else (None, candidates[i])
            if i < len(saved):
                record = saved[i]
                if record['number'] != i or record['params'] != params or record['training_keys'] != job['train'] or record['seed'] != job['seed']:
                    raise ValueError(f'Proposal/checkpoint mismatch: {job["id"]}, trial {i}')
                predictions, duplicates = evaluator.evaluate(params, job['train'], cached_only=True)
                if core.training_metrics(predictions, job['train']) != record['metrics']:
                    raise ValueError('Saved training metrics do not match model cache')
            else:
                started = time.perf_counter()
                predictions, duplicates = evaluator.evaluate(params, job['train'])
                duration = time.perf_counter() - started
                record = dict(number=i, seed=job['seed'], params=params, training_keys=job['train'],
                              metrics=core.training_metrics(predictions, job['train']),
                              requested_model_evaluations=len(predictions) + len(duplicates),
                              elapsed_seconds=duration)
                store.save_trial(job['id'], record)
            if study is not None:
                core.tell(study, trial, record['metrics'])
            records.append(record)
            if record['metrics']['feasible']:
                feasible += 1
                value = record['metrics']['training_MAE_days']
                best = value if best is None else min(best, value)
            logical += record['requested_model_evaluations']
            elapsed += record['elapsed_seconds']
            curve.append(dict(completed_trials=i + 1, best_training_MAE_days=best, n_feasible=feasible,
                              logical_model_evaluations=logical, shared_pool_wait_and_compute_seconds=elapsed))
            bar.set_postfix(best_MAE='NA' if best is None else f'{best:.3f}', feasible=f'{feasible}/{i+1}', refresh=False)
            bar.update(1)
            if not progress and (i + 1) % 128 == 0 and i >= len(saved):
                print(f'{label}: {i+1}/{job["n_trials"]}, best_MAE={best}, feasible={feasible}/{i+1}', flush=True)
    pd.DataFrame([core.flat_trial(t) for t in records]).to_csv(directory / 'trials.csv', index=False)
    convergence = pd.DataFrame(curve)
    convergence.to_csv(directory / 'convergence.csv', index=False)
    convergence[convergence.completed_trials.isin(s['training_checkpoints'] + [job['n_trials']])].to_csv(directory / 'training_checkpoints.csv', index=False)
    for budget in sorted({s['common_budget'], job['n_trials']}):
        select_budget(s, job, records, budget, evaluator, store, directory)
    held = store.heldout(job['id'], job['n_trials'])
    if store.heldout(job['id'], -1) is None:
        store.save_heldout(job['id'], -1, held)
    elif store.heldout(job['id'], -1) != held:
        raise ValueError('Final TPE selection differs from saved result')
    return dict(status='complete', n_trials=len(records), feasible=feasible,
                logical_model_evaluations=logical, shared_pool_wait_and_compute_seconds=elapsed)


def compare_predictions(frame):
    paired, summaries = [], []
    for (scheme, view), block in frame.groupby(['scheme', 'view'], sort=True):
        options = list(block.groupby(['arm', 'seed'], sort=True))
        for ((left_arm, left_seed), left), ((right_arm, right_seed), right) in itertools.combinations(options, 2):
            if left_arm == right_arm or (left_arm.endswith('tpe') and right_arm.endswith('tpe') and left_seed != right_seed):
                continue
            joined = left.merge(right, on='input_key', suffixes=('_left', '_right'), validate='one_to_one')
            if len(joined) != 6:
                raise ValueError('Pairing requires the same six site-years')
            tags = dict(scheme=scheme, view=view, left_arm=left_arm, right_arm=right_arm,
                        left_seed=int(left_seed), right_seed=int(right_seed))
            diffs, left_ok_count, right_ok_count = [], 0, 0
            for row in joined.to_dict('records'):
                a, b = row['status_left'] == 'predicted', row['status_right'] == 'predicted'
                left_ok_count += a; right_ok_count += b
                delta = abs(row['signed_interval_error_days_right']) - abs(row['signed_interval_error_days_left']) if a and b else None
                if delta is not None:
                    diffs.append(delta)
                paired.append(dict(**tags, input_key=row['input_key'], held_out=row['held_out_left'],
                                   status_left=row['status_left'], status_right=row['status_right'],
                                   n_trials_left=row['n_trials_left'], n_trials_right=row['n_trials_right'],
                                   error_left=row['signed_interval_error_days_left'], error_right=row['signed_interval_error_days_right'],
                                   delta_absolute_error_days=delta))
            summaries.append(dict(**tags, n_site_years=6, n_paired_predictions=len(diffs),
                                  n_predictions_left=left_ok_count, n_predictions_right=right_ok_count,
                                  n_trials_left=int(left.n_trials.iloc[0]) if left.n_trials.nunique()==1 else None,
                                  n_trials_right=int(right.n_trials.iloc[0]) if right.n_trials.nunique()==1 else None,
                                  min_trials_left=int(left.n_trials.min()), max_trials_left=int(left.n_trials.max()),
                                  min_trials_right=int(right.n_trials.min()), max_trials_right=int(right.n_trials.max()),
                                  same_candidate_budget=bool(joined.n_trials_left.eq(joined.n_trials_right).all()),
                                  mean_delta_days=float(np.mean(diffs)) if len(diffs) == 6 else None,
                                  conditional_mean_delta_days=float(np.mean(diffs)) if diffs else None,
                                  n_improved=sum(d < 0 for d in diffs), n_worsened=sum(d > 0 for d in diffs)))
    return pd.DataFrame(paired), pd.DataFrame(summaries)


def write_comparison(s, jobs, store, output):
    rows, folds, methods = [], [], []
    groups = {}
    for job in jobs:
        groups.setdefault((job['scheme'], job['arm'], job['seed']), []).append(job)
    for (scheme, arm, seed), group_jobs in groups.items():
        for view, budget in [('common', s['common_budget']), ('final', -1)]:
            held = [store.heldout(j['id'], budget) for j in group_jobs]
            if any(h is None for h in held):
                continue
            subset = []
            for job, result in zip(group_jobs, held):
                tags = dict(scheme=scheme, arm=arm, seed=seed, view=view, n_trials=result['selection']['n_trials'], held_out=job['held_out'])
                folds.append(dict(**tags, **{k:v for k,v in result['selection'].items() if k != 'n_trials'}))
                for prediction in result['predictions']:
                    subset.append({**prediction, **tags,
                                   'calibration_status': result['selection']['calibration_status'],
                                   'signed_interval_error_days': prediction.get('signed_interval_error_days')})
            frame = pd.DataFrame(subset)
            summary = core.group_summary(frame, s, {j['held_out']:j['test'] for j in group_jobs})
            counts = [h['selection']['n_trials'] for h in held]
            methods.append(dict(scheme=scheme, arm=arm, seed=seed, view=view,
                                n_trials=counts[0] if len(set(counts)) == 1 else None,
                                min_trials_per_fold=min(counts), max_trials_per_fold=max(counts), **summary))
            rows.extend(subset)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'oof_predictions.csv', index=False)
    pd.DataFrame(folds).to_csv(output / 'fold_summary.csv', index=False)
    pd.DataFrame(methods).to_csv(output / 'method_summary.csv', index=False)
    paired, summary = compare_predictions(frame)
    paired.to_csv(output / 'paired_predictions.csv', index=False)
    summary.to_csv(output / 'comparison_summary.csv', index=False)
    text = ['# Unified FSIM calibration comparison', '',
            'Primary cross-arm comparison: view=common (same number of training trials).',
            'view=final uses each arm\'s full budget; check same_candidate_budget before interpreting differences.',
            'Grid refines VPD/I separately at fixed wind levels 0,1,...,8; it does not refine wind. Final budgets may differ across folds.',
            'Positive delta_absolute_error_days means the right method has larger error.',
            'Missing predictions remain in the paired table. Complete MAE requires all six predictions.',
            'Site/year analyses reuse the same six site-years; TPE seeds are not independent field replicates.',
            'This is a frozen FSIM wind-gate parameter comparison, not FSIM-S evaluation.',
            'Runtime is shared-pool, cache-dependent timing, not a standalone speed benchmark.', '',
            '## Completed method summaries', '', '```csv', pd.DataFrame(methods).to_csv(index=False).strip(), '```', '']
    (output / 'comparison_report.md').write_text('\n'.join(text))


def run_jobs(s, jobs, candidates, evaluator, store, output, progress, on_complete):
    coordinators = ThreadPoolExecutor(max_workers=len(jobs))
    futures = {}
    try:
        for position, job in enumerate(jobs):
            futures[coordinators.submit(run_job, s, job, candidates.get(job['arm']), evaluator,
                                       store, output, progress, position)] = job
        for future in as_completed(futures):
            on_complete(futures[future], future.result())
    except BaseException:
        evaluator.stop.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        coordinators.shutdown(wait=True, cancel_futures=True)


def run_stages(s, jobs, candidates, evaluator, store, output, progress, on_complete, on_stage):
    for stage in ('grid', 'tpe'):
        selected = [j for j in jobs if j['arm'].endswith(stage)]
        if selected:
            on_stage(stage)
            if stage == 'grid':
                fsim_local_grid.run_local_grid(s, selected, evaluator, store, output, progress, sys.modules[__name__], on_complete)
            else:
                run_jobs(s, selected, candidates, evaluator, store, output, progress, on_complete)


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Output directory is already in use') from None
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=ROOT / 'config/fsim_m0_3p_joblib.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, choices=WORKERS)
    parser.add_argument('--schedule', choices=('staged',))
    parser.add_argument('--scheme', choices=('both', 'site', 'year'), default='both')
    parser.add_argument('--arms', nargs='+', choices=ARMS)
    parser.add_argument('--design-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Tiny real-data implementation check; not scientific calibration')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    s = json.loads(args.settings.read_text())
    if args.schedule is not None:
        s['schedule'] = args.schedule
    if args.workers is not None:
        s['workers'] = args.workers
    if args.scheme != 'both':
        s['schemes'] = [args.scheme]
    if args.arms:
        s['arms'] = args.arms
    if args.smoke:
        s['search_space']['VPD_w'].update(low=1.0, high=8.0, grid_step=7.0)
        s['search_space']['I_req'].update(low=40.0, high=450.0, grid_step=410.0)
        s['search_space']['RH_sp'].update(low=80.0, high=85.0, grid_step=5.0)
        s['search_space']['MP_sp'].update(low=3, high=4, grid_step=1)
        s['initial_point'] = dict(VPD_w=1.0, I_req=40.0, RH_sp=80.0, MP_sp=3, u_tr=0.0)
        s['local_grid'].update(coarse_values=dict(VPD_w=[1.0,8.0], I_req=[40.0,450.0], u_tr=[0.0,8.0]),
                              RH_values=[80.0,85.0], MP_values=[3,4],
                              max_candidates_per_fold=12, refinement_rounds=1)
        s.update(common_budget=4, n_startup_trials=2, tpe_trials=4, training_checkpoints=[4,20], smoke=True)
    validate_settings(s)
    versions, frozen = verify_engine(s)
    manifest = pd.read_csv(ROOT / s['manifest'])
    jobs = make_jobs(s, manifest)
    cases, errors, _, alternates, repairs = core.load_cases(ROOT / s['manifest'])
    if errors:
        raise ValueError(errors)
    if not s['check_duplicate_policy']:
        alternates = []
    altkeys = {c['input_key'] for c in alternates}
    budget = dict(workers=s['workers'], backend='joblib-loky', stage_order=(['grid_and_tpe'] if s['schedule'] == 'concurrent' else ['grid', 'tpe']), grid_dispatch='training_only_local_refinement_global_queue', grid_budget_is_cap=True,
                  coarse_candidates={a: len(fsim_local_grid.coarse_design(s,a)[0]) for a in s['arms'] if a.endswith('grid')},
                  n_fold_jobs=len(jobs),
                  trials_per_fold={a: trial_budget(s,a) for a in s['arms']}, common_budget=s['common_budget'],
                  total_training_trials=sum(j['n_trials'] for j in jobs),
                  logical_training_model_evaluations=sum(j['n_trials'] * (len(j['train']) + len(set(j['train']) & altkeys)) for j in jobs),
                  note='Actual model calls are reduced by exact shared caching; held-out calls are additional.')
    print(json.dumps(budget, indent=2), flush=True)
    identity = run_identity(s, jobs, versions, frozen)
    if args.design_only:
        print('Design verified; no calibration or output directory created.', flush=True)
        return
    output = (args.output_dir or ROOT / ('result/fsim_m0_3p_joblib_smoke' if args.smoke else 'result/fsim_m0_3p_joblib')).resolve()
    protected = [(ROOT / name).resolve() for name in ('result/fsim_unified_calibration', 'result/fsim_unified_joblib', 'result/fsim_unified_joblib_global', 'result/fsim_unified_joblib_local')]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError('Previous result directories are protected; use a separate global-queue output directory')
    with output_lock(output):
        resume = args.resume or args.report_only
        identity_path = output / 'run_identity.json'
        if resume:
            if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
                raise ValueError('Resume/report requires identical scientific settings, sources, inputs and versions; worker count may change')
            if not (output / 'checkpoints.sqlite').is_file():
                raise ValueError('Checkpoint database is missing')
        else:
            if any(p.name != '.run.lock' for p in output.iterdir()):
                raise ValueError('Use an empty output directory or --resume; old outputs are never overwritten')
            core.write_json(identity_path, identity)
            core.write_json(output / 'effective_settings.json', s)
            core.write_json(output / 'split_design.json', jobs)
            manifest.to_csv(output / 'site_year_manifest.csv', index=False)
            repairs.to_csv(output / 'weather_duplicate_audit.csv', index=False)
            for name in identity['source_sha256']:
                target = output / 'source_snapshot' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
        store = Store(output / 'checkpoints.sqlite')
        evaluator = None
        metadata_path = output / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text()) if resume and metadata_path.exists() else dict(started_utc=utc_now(), jobs={}, sessions=[])
        try:
            if args.report_only:
                write_comparison(s, jobs, store, output)
                return
            # nohup/background shells may inherit SIGINT ignored. Explicitly
            # restore graceful interruption in the owner process, not workers.
            signal.signal(signal.SIGINT, stop_on_signal)
            signal.signal(signal.SIGTERM, stop_on_signal)
            metadata.update(status='running', backend='joblib-loky', model='frozen_FSIM_wind_gate_3p', smoke=bool(args.smoke), budget=budget)
            metadata.pop('error', None)
            metadata['sessions'].append(dict(started_utc=utc_now(), workers=s['workers'], resumed=bool(args.resume)))
            core.write_json(metadata_path, metadata)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            base = core.load_config(ROOT / s['parameter_table'])
            base['dispersal']['use_wind_gate'] = True
            evaluator = SharedEvaluator(cases, alternates, base, s['workers'], store)
            candidates = {}  # Local plans are generated from training results each round.
            def complete(job, result):
                metadata['jobs'][job['id']] = result
                metadata['unique_model_evaluations'] = store.evaluation_count()
                core.write_json(metadata_path, metadata)
                write_comparison(s, jobs, store, output)
                print(f'Completed {job["id"]}: {result["n_trials"]} trials', flush=True)
            def stage_started(stage):
                metadata['active_stage'] = stage
                core.write_json(metadata_path, metadata)
                print(f'Starting {stage.upper()} stage; shared model process limit={s["workers"]}', flush=True)
            run_stages(s, jobs, candidates, evaluator, store, output, not args.no_progress, complete, stage_started)
            metadata.update(status='complete', completed_utc=utc_now())
            metadata['active_stage'] = None
        except BaseException as exc:
            metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            raise
        finally:
            if evaluator is not None:
                evaluator.close()
            if not args.report_only:
                metadata['unique_model_evaluations'] = store.evaluation_count()
                core.write_json(metadata_path, metadata)
            store.close()
    print(output, flush=True)


if __name__ == '__main__':
    main()
