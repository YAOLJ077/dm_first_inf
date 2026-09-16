#!/usr/bin/env python3
"""Four-parameter constrained TPE for grouped site/year holdouts.

Concurrent folds share one process pool; each fold keeps sequential ask/tell.
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'src/tpe_m0_reference'
# A private package root prevents importing the subsequently modified live model.
sys.path.insert(0, str(REFERENCE))
from model.parameter_table import load_config
import src.analyze_fsim_parameters as reference_analysis
from src.analyze_fsim_parameters import configure, evaluate_group, load_cases
from src.calibrate_m0_losyo import summarize_oof
reference_analysis.ROOT = ROOT  # Weather paths remain relative to the project.


NAMES = ('VPD_w', 'I_req', 'RH_sp', 'MP_sp')


def write_json(path, value):
    """Atomic checkpoints: an interrupted trial can be recomputed safely."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate_settings(s):
    if list(s['search_space']) != list(NAMES):
        raise ValueError('Four parameters VPD_w, I_req, RH_sp, MP_sp required')
    for name, spec in s['search_space'].items():
        lo, hi = spec['low'], spec['high']
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            raise ValueError(f'Invalid bounds: {name}')
        expected = 'int' if name == 'MP_sp' else 'float'
        if spec['type'] != expected:
            raise ValueError(f'{name} must use {expected} sampling')
        if expected == 'int' and (type(lo) is not int or type(hi) is not int or spec.get('step') != 1):
            raise ValueError('MP requires integer bounds and step=1')
    if s['n_trials'] != s['n_startup_trials'] + s['n_tpe_trials']:
        raise ValueError('Total trials must equal startup plus TPE trials')
    if len(s['seeds']) != 1:
        raise ValueError('Use one seed')
    if any(type(s[k]) is not int or s[k] < 1 for k in ('n_trials', 'n_startup_trials')):
        raise ValueError('Trial counts must be positive integers')
    if s['n_startup_trials'] > s['n_trials'] or s['workers'] not in (2, 4, 8, 16, 32):
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
    if (not train_keys or len(set(train_keys)) != len(train_keys)
            or len(records) != len(train_keys) or {r['input_key'] for r in records} != set(train_keys)):
        raise ValueError('Training records must cover exactly the current training site-years')
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
    study.enqueue_trial({k: anchor[k] for k in NAMES})
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
        self.stop = threading.Event()
        self.cases = cases + alternates
        self.primary = {c['input_key']: i for i, c in enumerate(cases)}
        self.alternate = {c['input_key']: i + len(cases) for i, c in enumerate(alternates)}
        self.pool = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init, initargs=(self.cases, base))

    def evaluate(self, params, keys):
        if self.stop.is_set():
            raise RuntimeError('Run interrupted; completed checkpoints retained')
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


def run_seed(s, seed, anchor, keys, directory, evaluator, progress=True, position=None):
    directory.mkdir(parents=True, exist_ok=True)
    study = make_study(seed, s, anchor)
    trials = []
    saved = sorted(directory.glob('trial_*.json'))
    if [p.name for p in saved] != [f'trial_{i:04d}.json' for i in range(len(saved))]:
        raise ValueError('Non-contiguous trial checkpoints')
    if len(saved) > s['n_trials']:
        raise ValueError('Checkpoint exceeds configured budget')
    with tqdm(total=s['n_trials'], desc=f'{directory.parents[2].name}/{directory.parent.name} seed={seed}',
              unit='trial', disable=not progress, dynamic_ncols=True, position=position) as bar:
        for i in range(s['n_trials']):
            if getattr(evaluator, 'stop', None) is not None and evaluator.stop.is_set():
                raise RuntimeError('Run interrupted; completed checkpoints retained')
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


def fingerprint(s, settings_path, anchors):
    sources = [Path(__file__), settings_path, REFERENCE / 'snapshot_manifest.json']
    sources += [REFERENCE / name for name in json.loads((REFERENCE / 'snapshot_manifest.json').read_text())['sha256']]
    manifest = ROOT / s['manifest']
    inputs = [manifest] + [ROOT / name for name in pd.read_csv(manifest).weather_json]
    def hashes(paths):
        result = {}
        for path in paths:
            resolved = path.resolve()
            key = resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved
            result[str(key)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result
    return dict(settings=s, anchors=anchors, source_sha256=hashes(sources), input_sha256=hashes(inputs),
                versions={'python': platform.python_version(), **{n: importlib.metadata.version(n)
                          for n in ('optuna', 'numpy', 'pandas', 'tqdm', 'scipy')}})


def verify_recorded_inputs(recorded, manifest):
    paths = [manifest] + [ROOT / name for name in pd.read_csv(manifest).weather_json]
    expected = {path.relative_to(ROOT): path for path in paths}
    if len(expected) != len(paths):
        raise ValueError('Duplicate input in manifest')
    checked = set()
    for name, digest in recorded.items():
        old = Path(name)
        matches = [rel for rel in expected if old == rel or
                   (old.is_absolute() and old.parts[-len(rel.parts):] == rel.parts)]
        if len(matches) != 1 or matches[0] in checked:
            raise ValueError(f'Unexpected or duplicate reference input: {name}')
        rel = matches[0]
        if hashlib.sha256(expected[rel].read_bytes()).hexdigest() != digest:
            raise ValueError(f'Reference input changed: {rel}')
        checked.add(rel)
    if checked != set(expected):
        raise ValueError(f'Reference inputs missing: {sorted(map(str, set(expected) - checked))}')


def verify_reference(s):
    manifest = json.loads((REFERENCE / 'snapshot_manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Reference snapshot changed: {name}')
    parent = ROOT / s['reference_run']
    identity = json.loads((parent / 'run_identity.json').read_text())
    old = identity['settings']
    for key in ('manifest', 'seeds',
                'objective', 'require_all_training_predictions', 'check_duplicate_policy',
                'accuracy_tolerances_days', 'tie_atol_days'):
        if s[key] != old[key]:
            raise ValueError(f'Control setting differs from four-parameter run: {key}')
    for name in NAMES:
        if s['search_space'][name] != old['search_space'][name]:
            raise ValueError(f'Control bounds differ: {name}')
    verify_recorded_inputs(identity['input_sha256'], ROOT / s['manifest'])
    for name, version in identity['versions'].items():
        actual = platform.python_version() if name == 'python' else importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f'Use reference dependency version: {name}={version}')
    print('Verified frozen four-parameter model, inputs, versions and four-parameter search ranges.', flush=True)


def make_groups(manifest, scheme):
    column={'site':'place_code','year':'Year'}[scheme]
    if len(manifest)!=6 or manifest.input_key.duplicated().any() or manifest[column].isna().any():
        raise ValueError('Requires six unique site-years and nonmissing group labels')
    groups={str(value):frame.input_key.tolist() for value,frame in manifest.groupby(column,sort=True)}
    if len(groups)!=3 or any(len(keys)==6 for keys in groups.values()):
        raise ValueError('Expected three nonempty held-out groups')
    return groups


def group_summary(frame,s,groups):
    expected=[k for keys in groups.values() for k in keys]
    if len(frame)!=6 or frame.input_key.duplicated().any() or set(frame.input_key)!=set(expected):
        raise ValueError('Each site-year must receive exactly one out-of-fold result')
    selected=frame.calibration_status.eq('selected');available=selected&frame.status.eq('predicted')
    e=pd.to_numeric(frame.signed_interval_error_days,errors='coerce')
    if not np.isfinite(e[available]).all():raise ValueError('Nonfinite predicted error')
    metrics={'n_folds':len(groups),'n_site_years':6,
             'n_calibration_failed_folds':int(frame.loc[~selected,'held_out'].nunique()),
             'n_calibration_failed_site_years':int((~selected).sum()),
             'n_predictions':int(available.sum()),'n_no_predictions':int((selected&~available).sum()),
             'prediction_rate_fixed_six':float(available.sum()/6),
             'MAE_days':float(e.abs().mean()) if available.all() else None,
             'RMSE_days':float(np.sqrt(np.mean(e**2))) if available.all() else None,
             'conditional_MAE_days':float(e[available].abs().mean()) if available.any() else None,
             'conditional_RMSE_days':float(np.sqrt(np.mean(e[available]**2))) if available.any() else None}
    group_maes=[]
    for group in groups:
        mask=frame.held_out.eq(group)
        group_maes.append(float(e[mask].abs().mean()) if available[mask].all() else None)
    metrics['equal_group_MAE_days']=float(np.mean(group_maes)) if all(v is not None for v in group_maes) else None
    for tol in s['accuracy_tolerances_days']:
        metrics[f'accurate_prediction_rate_{tol:g}d_fixed_six']=float((available&e.abs().le(tol)).sum()/6)
    return metrics



def read_group_anchors(s, designs, keys):
    parent=ROOT/s['anchor_run']
    meta=json.loads((parent/'run_metadata.json').read_text())
    if meta['status']!='complete':raise ValueError('Grouped grid anchor must be complete')
    verify_recorded_inputs(meta['input_sha256'], ROOT / s['manifest'])
    anchors={}
    for scheme,groups in designs.items():
        anchors[scheme]={}
        for group,test in groups.items():
            selection=json.loads((parent/scheme/'folds'/group/'selection.json').read_text())
            if set(selection['test_keys'])!=set(test) or set(selection['training_keys'])!=set(keys)-set(test):
                raise ValueError('Anchor must come from exactly the same training group')
            params={k:selection[k] for k in ('VPD_w','I_req')};params.update(RH_sp=80.,MP_sp=3)
            if any(not s['search_space'][k]['low']<=v<=s['search_space'][k]['high'] for k,v in params.items()):
                raise ValueError('Anchor outside search space')
            anchors[scheme][group]=params
    return anchors


def run_one_group(s,keys,group,test_keys,anchor,out,evaluator,progress,position=None):
    count=0;duplicate_count=0
    directory=out/'folds'/group;directory.mkdir(parents=True,exist_ok=True)
    train_keys=[k for k in keys if k not in test_keys]
    trials=run_seed(s,s['seeds'][0],anchor,train_keys,
                    directory/f'seed_{s["seeds"][0]}',evaluator,progress,position)
    count+=sum(len(t['predictions'])+len(t['duplicate_predictions']) for t in trials)
    duplicate_count+=sum(len(t['duplicate_predictions']) for t in trials)
    choice=save_training_reports(directory,trials,s)
    selection=dict(held_out=group,training_keys=train_keys,test_keys=test_keys,
                   calibration_status='selected' if choice else 'no_feasible_training_combination')
    if choice:
        selection.update(**choice['params'],seed=choice['seed'],trial=choice['number'],
                         selected_training_MAE_days=choice['metrics']['training_MAE_days'])
    write_json(directory/'selection.json',selection)
    if choice is None:
        records=[dict(input_key=k,status='calibration_failed',signed_interval_error_days=None) for k in test_keys]
    else:
        pd.DataFrame(choice['predictions']).to_csv(directory/'training_predictions.csv',index=False)
        write_json(directory/'effective_runtime_config.json',runtime_config(evaluator.base,choice['params']))
        checkpoint=directory/'held_out_checkpoint.json'
        if checkpoint.exists():
            held=json.loads(checkpoint.read_text())
            if held['params']!=choice['params'] or held['test_keys']!=test_keys:
                raise ValueError('Saved held-out result does not match frozen selection')
        else:
            records,duplicates=evaluator.evaluate(choice['params'],test_keys)
            held=dict(params=choice['params'],test_keys=test_keys,predictions=records,duplicate_predictions=duplicates)
            write_json(checkpoint,held)
        records=held['predictions'];count+=len(records)+len(held['duplicate_predictions'])
        duplicate_count+=len(held['duplicate_predictions'])
    records=[{**r,'held_out':group,'calibration_status':selection['calibration_status'],
              'signed_interval_error_days':r.get('signed_interval_error_days')} for r in records]
    pd.DataFrame(records).to_csv(directory/'held_out_predictions.csv',index=False)
    return selection,records,count,duplicate_count


def run_groups(s,keys,groups,anchors,out,evaluator,progress,completed=None):
    """Summarize in fixed group order; completed may arrive in any worker order."""
    out.mkdir(parents=True,exist_ok=True)
    oof=[];selections=[];count=0;duplicate_count=0
    for group,test_keys in groups.items():
        result=(completed[group] if completed is not None else
                run_one_group(s,keys,group,test_keys,anchors[group],out,evaluator,progress))
        selection,records,n,duplicates=result
        selections.append(selection);oof.extend(records);count+=n;duplicate_count+=duplicates
    frame=pd.DataFrame(oof);frame.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'fold_summary.csv',index=False)
    summary=group_summary(frame,s,groups);write_json(out/'oof_summary.json',summary)
    metrics=[]
    for group,g in frame.groupby('held_out'):
        available=g.status.eq('predicted');e=pd.to_numeric(g.signed_interval_error_days,errors='coerce')
        metrics.append(dict(held_out=group,n_test_site_years=len(g),n_predictions=int(available.sum()),
                            MAE_days=float(e.abs().mean()) if available.all() else None,
                            RMSE_days=float(np.sqrt(np.mean(e**2))) if available.all() else None))
    pd.DataFrame(metrics).to_csv(out/'test_group_metrics.csv',index=False)
    return summary,count,duplicate_count


def run_parallel_folds(s,keys,designs,anchors,out,evaluator,progress,on_scheme_complete=None):
    """One coordinator per fold, one shared model pool, no cross-fold TPE history."""
    tasks=[(scheme,group,test) for scheme,groups in designs.items() for group,test in groups.items()]
    completed={scheme:{} for scheme in designs};summaries={}
    coordinators=ThreadPoolExecutor(max_workers=len(tasks),thread_name_prefix='tpe_fold')
    futures={}
    try:
        for position,(scheme,group,test) in enumerate(tasks):
            future=coordinators.submit(run_one_group,s,keys,group,test,anchors[scheme][group],
                out/scheme,evaluator,progress,position)
            futures[future]=(scheme,group)
        for future in as_completed(futures):
            scheme,group=futures[future]
            completed[scheme][group]=future.result()
            if len(completed[scheme])==len(designs[scheme]):
                result=run_groups(s,keys,designs[scheme],anchors[scheme],out/scheme,evaluator,False,completed[scheme])
                summaries[scheme]=result
                if on_scheme_complete is not None:on_scheme_complete(scheme,result)
    except BaseException:
        if getattr(evaluator,'stop',None) is not None:evaluator.stop.set()
        for future in futures:future.cancel()
        raise
    finally:
        # Active trials finish at most their current evaluation before noticing stop.
        coordinators.shutdown(wait=True,cancel_futures=True)
    return summaries


def compare_grid(out,reference):
    old=pd.read_csv(reference/'oof_predictions.csv')
    new=pd.read_csv(out/'oof_predictions.csv')
    columns=['input_key','status','signed_interval_error_days']
    paired=old[columns].merge(new,on='input_key',validate='one_to_one',suffixes=('_grid','_tpe'))
    paired['delta_absolute_error_days']=paired.signed_interval_error_days_tpe.abs()-paired.signed_interval_error_days_grid.abs()
    paired.to_csv(out/'comparison_with_grouped_grid.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,default=ROOT/'config/fsim_m0_grouped_tpe_4p.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_m0_grouped_tpe_4p_n1024_start128_parallel32')
    parser.add_argument('--scheme',choices=['both','site','year'],default='both')
    parser.add_argument('--workers',type=int,choices=[2,4,8,16,32])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--design-only',action='store_true')
    parser.add_argument('--no-progress',action='store_true')
    args=parser.parse_args();s=json.loads(args.settings.read_text())
    if args.workers:s['workers']=args.workers
    if args.scheme!='both':s['validation_schemes']=[args.scheme]
    if not s['validation_schemes'] or len(set(s['validation_schemes']))!=len(s['validation_schemes']) or not set(s['validation_schemes'])<={'site','year'}:
        raise ValueError('Choose site and/or year')
    validate_settings(s);verify_reference(s)
    manifest=pd.read_csv(ROOT/s['manifest']);keys=manifest.input_key.tolist()
    designs={scheme:make_groups(manifest,scheme) for scheme in s['validation_schemes']}
    anchors=read_group_anchors(s,designs,keys)
    base=load_config(ROOT/s['parameter_table']);base['dispersal']['use_wind_gate']=False
    cases,errors,_,alternates,repairs=load_cases(ROOT/s['manifest'])
    if errors:raise ValueError(errors)
    if not s['check_duplicate_policy']:alternates=[]
    altkeys={c['input_key'] for c in alternates}
    budget={}
    for scheme,groups in designs.items():
        for group,test in groups.items():
            print(f'{scheme} {group}: training={[k for k in keys if k not in test]}; test={test}',flush=True)
        budget[scheme]=dict(parameter_trials=len(groups)*s['n_trials'],
            main_training_evaluations=sum(len(keys)-len(test) for test in groups.values())*s['n_trials'],
            duplicate_training_evaluations=sum(len(altkeys-set(test)) for test in groups.values())*s['n_trials'],
            maximum_held_out_evaluations=len(keys)+len(altkeys))
    print(json.dumps(budget,indent=2),flush=True)
    identity=fingerprint(s,args.settings.resolve(),anchors);identity['split_design']=designs
    if args.design_only:
        print('Design verified; no model calibration or output creation.');return
    out=args.output_dir.resolve()
    if args.resume:
        if not (out/'run_identity.json').exists():parser.error('No checkpoint to resume')
    elif out.exists() and any(out.iterdir()):parser.error('Use a new empty directory or --resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Output directory already in use')
        if args.resume:
            if json.loads((out/'run_identity.json').read_text())!=identity:
                raise ValueError('Resume requires unchanged settings, sources, inputs and versions')
        else:
            write_json(out/'run_identity.json',identity);write_json(out/'effective_settings.json',s)
            write_json(out/'split_design.json',designs)
            manifest.to_csv(out/'site_year_manifest.csv',index=False);repairs.to_csv(out/'weather_duplicate_audit.csv',index=False)
            for name in identity['source_sha256']:
                key=Path(name);path=key if key.is_absolute() else ROOT/key
                rel=Path('external')/key.name if key.is_absolute() else key
                dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        metadata=dict(status='running',model='M0',parallel_strategy='concurrent_folds_shared_process_pool',settings=s,budget=budget,schemes={},resumed=args.resume,
                      started_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'run_metadata.json',metadata)
        evaluator=Evaluator(cases,alternates,base,s['workers'])
        try:
            def completed_scheme(scheme,result):
                summary,count,duplicates=result
                compare_grid(out/scheme,ROOT/s['anchor_run']/scheme)
                metadata['schemes'][scheme]=dict(status='complete',summary=summary,n_model_evaluations_in_completed_records=count,
                    n_duplicate_evaluations=duplicates,duplicate_policy_differences=0)
                write_json(out/'run_metadata.json',metadata)
            run_parallel_folds(s,keys,designs,anchors,out,evaluator,not args.no_progress,completed_scheme)
            metadata.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            metadata.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
        finally:
            evaluator.close();write_json(out/'run_metadata.json',metadata)
    print(out,flush=True)


if __name__=='__main__':main()
