"""Shared frozen-B downstream evaluation for fixed pressure and beta-only TPE."""
import ast
import inspect
import json
import textwrap
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import replay_b_fixed as base
import numpy as np
import pandas as pd

ROOT = base.ROOT
CONTEXT = OrderedDict()
JOBS = {}
PRESSURE = None
METHOD = None
REPLAY_DIR = None


def prepare(args):
    """Verify frozen source, original inputs, train-only splits and pressure provenance."""
    options = SimpleNamespace(baseline=ROOT/'result/fsim_abc_tpe_seed123', budget=1024,
        schemes=args.schemes, rules=['min_train'], folds=args.folds, case_role='all',
        output_dir=args.output_root/'unused', fig_dir=args.output_root/'unused')
    tasks, selections, identity, snapshot, manifest = base.prepare(options)
    meta = base.read_json(args.replay_dir/'run_metadata.json')
    if meta['status'] != 'complete' or not meta['baseline_reproduced']:
        raise ValueError('A completed, reproduced B replay is required')
    pressure_meta = base.read_json(args.pressure_dir/'run_metadata.json')
    pressure_identity = base.read_json(args.pressure_dir/'run_identity.json')
    if pressure_meta['status'] != 'complete':
        raise ValueError('Pressure input incomplete')
    for path, digest in pressure_identity['input_sha256'].items():
        if base.sha(ROOT/path) != digest:
            raise ValueError(f'Pressure source changed: {path}')
    pressure_file = args.pressure_dir/'daily_pressure_by_fold.csv'
    pressure = pd.read_csv(pressure_file)
    scalers = pd.read_csv(args.pressure_dir/'fold_scalers.csv')
    if pressure.duplicated(['scheme','fold','input_key','date']).any():
        raise ValueError('Duplicate pressure day')
    jobs = {}
    source_hashes = {}
    for task in tasks:
        jobid = f'{task["scheme"]}/{task["fold"]}'
        if jobid not in jobs:
            s = task['selection']
            jobs[jobid] = dict(id=jobid, scheme=task['scheme'], fold=task['fold'],
                train=s['training_keys'], test=s['test_keys'], config=task['config'], selection=s)
            scaler = scalers[scalers.scheme.eq(task['scheme'])&scalers.fold.astype(str).eq(task['fold'])]
            if len(scaler)!=1 or set(scaler.iloc[0].training_keys.split(';')) != set(s['training_keys']):
                raise ValueError('Pressure scaler training split mismatch')
            rows = pressure[pressure.scheme.eq(task['scheme'])&pressure.fold.astype(str).eq(task['fold'])]
            available = rows.pressure_available.eq(True)
            z = (rows.current_log_rate-scaler.iloc[0].mean_log_rate)/scaler.iloc[0].sd_log_rate
            if not np.allclose(rows.relative_pressure_z, z, atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError('Pressure values do not match saved training scaler')
            if rows.loc[~available,'relative_pressure_z'].notna().any():
                raise ValueError('Missing pressure must be neutral, not an observed zero')
            known = rows.available_date.notna()
            if (pd.to_datetime(rows.loc[known,'available_date'])>pd.to_datetime(rows.loc[known,'date'])).any():
                raise ValueError('Future observation used in pressure input')
            expected_role = np.where(rows.input_key.isin(s['training_keys']),'train','test')
            if not rows.case_role.eq(expected_role).all():
                raise ValueError('Pressure train/test roles mismatch')
        directory = args.replay_dir/'min_train'/jobid/task['case_role']/task['input_key']
        marker = base.read_json(directory/'completed.json')['sha256']
        for name in ('detached_survival.csv','primary_chain.csv','primary_chain_gated.csv',
                     'primary_chain_unique.csv','prediction.json','effective_runtime_config.json'):
            path = directory/name
            if base.sha(path)!=marker[name]:
                raise ValueError(f'Cached B artifact changed: {path}')
            source_hashes[str(path.relative_to(args.replay_dir))] = marker[name]
        if base.read_json(directory/'effective_runtime_config.json') != task['config']:
            raise ValueError('Cached B configuration mismatch')
    identity.update(pressure_identity=pressure_identity, pressure_sha256=base.sha(pressure_file),
        pressure_scalers_sha256=base.sha(args.pressure_dir/'fold_scalers.csv'), cached_B_hashes=source_hashes,
        experiment_code_sha256={name:base.sha(Path(__file__).parent/name) for name in
            ('pressure_model_core.py','pressure_experiment_runner.py','evaluate_fixed_pressure.py','calibrate_pressure_tpe.py')},
        mapping='H * exp(beta * z_at_dispersal_date) >= original I; unavailable z -> 0',
        beta_range=[0.0,1.0], beta_fixed=1.0)
    return jobs, identity, snapshot, manifest


def compile_method(original):
    """Alter exactly one infection comparison in memory; frozen source stays intact."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and ast.unparse(node) == "WD * TWD >= inf['infection_condition']":
            node.left = ast.BinOp(left=node.left, op=ast.Mult(),
                right=ast.parse('self._pressure_factors[int(c.Num)]', mode='eval').body)
            count += 1
    if count != 1:
        raise ValueError('Expected exactly one frozen infection threshold comparison')
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<pressure_infection_comparison>', 'exec'), namespace)
    return namespace[original.__name__]


def initialize(snapshot, manifest, jobs, pressure_dir, replay_dir):
    global JOBS, PRESSURE, METHOD, REPLAY_DIR
    base.initialize(snapshot, manifest)
    JOBS = jobs
    PRESSURE = pd.read_csv(Path(pressure_dir)/'daily_pressure_by_fold.csv', parse_dates=['date'])
    REPLAY_DIR = Path(replay_dir)
    METHOD = compile_method(base.CORE.FirstInfection.stage_primary_chain)


def context(jobid, key):
    token = (jobid,key)
    if token in CONTEXT:
        CONTEXT.move_to_end(token)
        return CONTEXT[token]
    job = JOBS[jobid]
    role = 'train' if key in job['train'] else 'test'
    directory = REPLAY_DIR/'min_train'/jobid/role/key
    case = base.CASES[key]
    cs = base.CORE.CropSeasonFirst(case['rb'], job['config'])
    detached = pd.read_csv(directory/'detached_survival.csv', float_precision='round_trip')
    for col in detached:
        if col.endswith('_datetime') or col=='sporangia_alive_end':
            detached[col] = pd.to_datetime(detached[col])
    cs.detached_survival = detached
    engine = base.CORE.FirstInfection(cs, job['config'], use_wind_gate=False)
    rows = PRESSURE[PRESSURE.scheme.eq(job['scheme']) & PRESSURE.fold.astype(str).eq(job['fold']) & PRESSURE.input_key.eq(key)].set_index('date')
    days = pd.to_datetime(detached.dispersal_datetime).dt.normalize()
    if not days.isin(rows.index).all():
        raise ValueError(f'Missing pressure calendar dates: {jobid}/{key}')
    evidence = rows.loc[days].reset_index(drop=True)
    z = dict(zip(detached.Num.astype(int),evidence.relative_pressure_z.fillna(0).astype(float)))
    result = (case, cs, engine, z, directory, evidence)
    CONTEXT[token] = result
    while len(CONTEXT)>8:
        CONTEXT.popitem(last=False)
    return result


def check_table(actual, reference):
    expected = pd.read_csv(reference, float_precision='round_trip')
    if actual is None or (actual.empty and not len(actual.columns)):
        actual = pd.DataFrame(columns=['Num'])
    # Round-trip only the in-memory table's CSV representation, to compare date formatting.
    from io import StringIO
    observed = pd.read_csv(StringIO(actual.to_csv(index=False)), float_precision='round_trip')
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)


def evaluate(jobid, key, beta, export=None, validate=False):
    if not 0 <= beta <= 1:
        raise ValueError('Only beta in [0,1] is supported')
    case, cs, engine, z, directory, evidence = context(jobid,key)
    job = JOBS[jobid]
    engine._pressure_factors = {n:float(np.exp(beta*v)) for n,v in z.items()}
    METHOD(engine)
    cs.apply_bbch_15_gate()
    cs.collapse_equivalent_cohorts()
    windows = cs.primary_chain_unique
    record = dict(scheme=job['scheme'],fold=job['fold'],input_key=key,
        case_role='train' if key in job['train'] else 'test',beta=float(beta),
        VPD_w=job['selection']['VPD_w'],I_req=job['selection']['I_req'],
        status='no_prediction',n_windows=0 if windows is None else len(windows),
        weather_policy=case['weather_policy'],observed_L=case['L'],observed_U=case['U'],overlap=False,
        selection_uses_test_data=False)
    if windows is not None and not windows.empty:
        first = windows.sort_values(['oil_spot_start','infection_datetime']).iloc[0]
        record.update(status='predicted',**base.CORE.score_window(pd.Timestamp(case['L']),pd.Timestamp(case['U']),
            pd.Timestamp(first.oil_spot_start),pd.Timestamp(first.oil_spot_end)))
    if validate:
        if beta != 0 or base.check_expected(record,base.read_json(directory/'prediction.json')):
            raise ValueError(f'Beta zero did not reproduce B: {jobid}/{key}')
        for name in ('primary_chain','primary_chain_gated','primary_chain_unique'):
            check_table(getattr(cs,name),directory/(name+'.csv'))
        record['baseline_verified'] = True
    if export:
        out = Path(export)
        for name in ('primary_chain','primary_chain_gated','primary_chain_unique'):
            table = getattr(cs,name)
            if table is None or (table.empty and not len(table.columns)):
                table=pd.DataFrame(columns=['Num'])
            base.save_frame(out/(name+'.csv'),table)
        old = pd.read_csv(directory/'primary_chain.csv')
        state = cs.detached_survival[['Num','dispersal_datetime']].copy()
        state['relative_pressure_z'] = evidence.relative_pressure_z.to_numpy()
        state['observation_state'] = evidence.observation_state.to_numpy()
        state['factor'] = state.Num.map(engine._pressure_factors)
        for prefix,chain in [('baseline',old),('new',cs.primary_chain)]:
            mapping = dict(zip(chain.Num,chain.infection_datetime)) if chain is not None and not chain.empty else {}
            state[prefix+'_infection_datetime'] = pd.to_datetime(state.Num.map(mapping))
        before,after=state.baseline_infection_datetime.notna(),state.new_infection_datetime.notna()
        state['change'] = np.select([~before&after,before&~after,before&after],['gained','lost','retained'],default='neither')
        state['infection_shift_hours'] = (state.new_infection_datetime-state.baseline_infection_datetime).dt.total_seconds()/3600
        base.save_frame(out/'cohort_changes.csv',state)
        base.save_json(out/'prediction.json',record)
        base.save_json(out/'effective_runtime_config.json',job['config'])
    return record


def train_metrics(jobid, beta):
    records = [evaluate(jobid,key,beta) for key in JOBS[jobid]['train']]
    missing = sum(r['status']=='no_prediction' for r in records)
    mae = None if missing else float(np.mean([r['interval_distance_days'] for r in records]))
    return dict(feasible=missing==0,n_no_prediction=missing,training_MAE_days=mae,
                optimizer_value=0.0 if missing else mae)


def constraints(trial):
    return (trial.user_attrs['n_no_prediction'],)
