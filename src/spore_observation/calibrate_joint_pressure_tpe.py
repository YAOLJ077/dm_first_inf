#!/usr/bin/env python3
"""Joint VPD/I/beta TPE for the historical pressure-threshold experiment.

Full simulation per candidate; train-only selection; separate output from frozen-B runs.
"""
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import pressure_model_core as core
import pressure_experiment_runner as runner


def evaluate(jobid, key, params, export=None, validate=False, exploratory=False):
    job = core.JOBS[jobid]
    case = core.base.CASES[key]
    cfg = copy.deepcopy(job['config'])
    cfg['zoospore_alive']['vpd'] = params['VPD_w']
    cfg['infection']['infection_condition'] = params['I_req']
    cs = core.base.CORE.CropSeasonFirst(case['rb'], cfg)
    engine = core.base.CORE.FirstInfection(cs, cfg, use_wind_gate=False)
    engine.stage_attached_and_dispersal()
    engine.stage_deposition()
    engine.stage_detached_survival()
    rows = core.PRESSURE[core.PRESSURE.scheme.eq(job['scheme']) &
        core.PRESSURE.fold.astype(str).eq(job['fold']) & core.PRESSURE.input_key.eq(key)].set_index('date')
    detached = cs.detached_survival
    days = pd.to_datetime(detached.dispersal_datetime).dt.normalize()
    if not days.isin(rows.index).all():
        raise ValueError('Missing pressure calendar dates')
    evidence = rows.loc[days].reset_index(drop=True)
    factors = np.exp(params['beta'] * evidence.relative_pressure_z.fillna(0).to_numpy())
    engine._pressure_factors = dict(zip(detached.Num.astype(int), factors))
    core.METHOD(engine)
    cs.apply_bbch_15_gate()
    cs.collapse_equivalent_cohorts()
    windows = cs.primary_chain_unique
    role = 'train' if key in job['train'] else 'test'
    record = dict(scheme=job['scheme'], fold=job['fold'], input_key=key, case_role=role,
        **params, status='no_prediction', n_windows=0 if windows is None else len(windows),
        observed_L=case['L'], observed_U=case['U'], weather_policy=case['weather_policy'],
        overlap=False, selection_uses_test_data=exploratory,
        evaluation_role='exploratory_test_selected' if exploratory else 'training_selected_baseline')
    if windows is not None and not windows.empty:
        first = windows.sort_values(['oil_spot_start', 'infection_datetime']).iloc[0]
        record.update(status='predicted', **core.base.CORE.score_window(
            pd.Timestamp(case['L']), pd.Timestamp(case['U']),
            pd.Timestamp(first.oil_spot_start), pd.Timestamp(first.oil_spot_end)))
    if validate:
        directory = core.REPLAY_DIR/'min_train'/jobid/role/key
        if params['beta'] != 0 or core.base.check_expected(record, core.base.read_json(directory/'prediction.json')):
            raise ValueError(f'Full-simulation B reproduction failed: {jobid}/{key}')
        for name in ('primary_chain', 'primary_chain_gated', 'primary_chain_unique'):
            core.check_table(getattr(cs, name), directory/(name+'.csv'))
    if export:
        directory = Path(export)
        for name in core.base.TABLE_NAMES:
            table = getattr(cs, name)
            if table is None or (table.empty and not len(table.columns)):
                table = pd.DataFrame(columns=['Num'])
            core.base.save_frame(directory/(name+'.csv'), table)
        evidence = evidence.copy()
        evidence['Num'] = detached.Num.to_numpy()
        evidence['factor'] = factors
        core.base.save_frame(directory/'cohort_pressure.csv', evidence)
        core.base.save_json(directory/'effective_runtime_config.json', cfg)
        core.base.save_json(directory/'prediction.json', record)
    return record


def metrics(records):
    missing = sum(r['status'] != 'predicted' for r in records)
    mae = None if missing else float(np.mean([r['interval_distance_days'] for r in records]))
    return dict(feasible=missing == 0, n_no_prediction=missing,
                training_MAE_days=mae, optimizer_value=0.0 if missing else mae)


def exploratory_output(pool, jobs, histories, output, workers):
    """Evaluate unique +1-day candidates; choose ONE vector per complete test fold."""
    records, selections = [], []
    for jobid, history in histories.items():
        job = jobs[jobid]
        valid = [r for r in history if r['metrics']['feasible']]
        best = min(r['metrics']['training_MAE_days'] for r in valid)
        unique = {}
        for r in valid:
            if r['metrics']['training_MAE_days'] <= best + 1.0 + 1e-9:
                unique.setdefault(tuple(r['params'][k] for k in ('VPD_w','I_req','beta')), r)
        candidates = list(unique.values())
        summaries, predictions, eligible = [], [], []
        # Bound pending tasks rather than enqueueing every candidate of every fold.
        batch = max(1, workers // len(job['test']))
        for start in range(0,len(candidates),batch):
            pending = [(r,[pool.submit(evaluate,jobid,key,r['params'],None,False,True)
                           for key in job['test']]) for r in candidates[start:start+batch]]
            for r, futures in pending:
                block = [f.result() for f in futures]
                m = metrics(block)
                summaries.append(dict(number=r['number'],**r['params'],
                    training_MAE_days=r['metrics']['training_MAE_days'],
                    test_MAE_days=m['training_MAE_days'],test_prediction_complete=m['feasible'],
                    n_test=len(block),n_test_predicted=len(block)-m['n_no_prediction']))
                predictions.extend(dict(p,candidate_trial=r['number']) for p in block)
                if m['feasible']:
                    eligible.append((r,m))
            print(f'Exploratory {jobid}: {min(start+batch,len(candidates))}/{len(candidates)} candidates',flush=True)
        directory = output/'exploratory/folds'/jobid
        core.base.save_frame(directory/'selection_candidates.csv',pd.DataFrame(summaries))
        core.base.save_frame(directory/'candidate_test_predictions.csv',pd.DataFrame(predictions))
        selection = dict(scheme=job['scheme'],fold=job['fold'],selection_uses_test_data=True,
            evaluation_role='exploratory_test_selected',training_keys=job['train'],test_keys=job['test'],
            selection_rule='training_minimum_plus_1_day_then_minimum_complete_test_MAE',
            selection_tolerance_days=1.0,minimum_training_MAE_days=best,
            n_unique_candidates=len(candidates),n_complete_test_candidates=len(eligible),
            calibration_status='selected' if eligible else 'no_complete_test_prediction')
        if eligible:
            test_min = min(m['training_MAE_days'] for _,m in eligible)
            r,m = min(((r,m) for r,m in eligible if m['training_MAE_days']<=test_min+1e-9),
                key=lambda pair:(pair[0]['metrics']['training_MAE_days'],pair[0]['params']['beta'],pair[0]['number']))
            selection.update(**r['params'],selected_trial=r['number'],
                training_MAE_days=r['metrics']['training_MAE_days'],test_MAE_days=m['training_MAE_days'])
            futures = [pool.submit(evaluate,jobid,key,r['params'],
                str(directory/('train' if key in job['train'] else 'test')/key),False,True)
                for key in job['train']+job['test']]
            block = [dict(f.result(),arm='joint_tpe') for f in futures]
            for role,expected in [('train',r['metrics']['training_MAE_days']),('test',m['training_MAE_days'])]:
                replay = metrics([p for p in block if p['case_role']==role])
                if not replay['feasible'] or not np.isclose(replay['training_MAE_days'],expected,atol=1e-8,rtol=0):
                    raise ValueError('Exploratory selected replay mismatch')
            records.extend(block)
        else:
            records.extend(dict(scheme=job['scheme'],fold=job['fold'],input_key=key,case_role='test',
                arm='joint_tpe',status='selection_failed',n_windows=0,overlap=False,
                selection_uses_test_data=True,evaluation_role='exploratory_test_selected') for key in job['test'])
        core.base.save_json(directory/'selection.json',selection)
        selections.append(selection)
    core.base.save_frame(output/'exploratory/selected_parameters.csv',pd.DataFrame(selections))
    return records


def main():
    parser = runner.parser()
    parser.description = __doc__
    parser.set_defaults(output_root=core.ROOT/'result/spore_observation/plan1_joint_pressure_tpe')
    args = parser.parse_args()
    args = runner.normalize(args)
    if not 2 <= args.startup_trials < args.trials:
        parser.error('Require 2 <= startup-trials < trials; trials includes startup')
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    jobs, identity, snapshot, manifest = core.prepare(args)
    original = core.base.read_json(core.ROOT/'result/fsim_abc_tpe_seed123/run_identity.json')
    space = copy.deepcopy(original['settings']['plans']['B']['search_space'])
    space['beta'] = [0.0, 1.0]
    identity.update(mode='joint_VPD_I_beta', search_space=space, beta_range=space['beta'],
        beta_fixed=None, seed=args.seed, startup_trials=args.startup_trials,
        adaptive_trials=args.trials-args.startup_trials, total_trials=args.trials,
        adaptive_batch_per_fold=2, joint_script_sha256=core.base.sha(__file__),
        optuna_version=optuna.__version__, full_simulation=True,
        output_rules=['minimum_training_MAE','training_minimum_plus_1_day_then_minimum_complete_test_MAE'],
        selection_tolerance_days=1.0)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output/'.joint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idpath = output/'joint_run_identity.json'
        if idpath.exists():
            if not args.resume or core.base.read_json(idpath) != identity:
                raise ValueError('Use a fresh output directory or --resume with identical design/code/inputs')
        elif any(p.name != '.joint.lock' and not p.name.startswith('._') for p in output.iterdir()):
            raise ValueError('Output directory is not empty')
        core.base.save_json(idpath, identity)
        status = dict(status='running', stage='baseline_checks', workers=args.workers,
            n_folds=len(jobs), total_trials=len(jobs)*args.trials,
            started_utc=datetime.now(timezone.utc).isoformat())
        core.base.save_json(output/'run_metadata.json', status)
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=core.initialize,
                initargs=(str(snapshot), str(manifest), jobs, str(args.pressure_dir), str(args.replay_dir))) as pool:
                anchors = []
                for jobid, job in jobs.items():
                    params = {k:job['selection'][k] for k in ('VPD_w','I_req')}
                    params['beta'] = 0.0
                    anchors.extend(pool.submit(evaluate, jobid, key, params, None, True)
                                   for key in job['train']+job['test'])
                baseline = [dict(f.result(), arm='B') for f in anchors]
                core.base.save_frame(output/'baseline_predictions.csv', pd.DataFrame(baseline))
                for jobid, job in jobs.items():
                    r = [r for r in baseline if r['scheme']==job['scheme'] and r['fold']==job['fold'] and r['case_role']=='train']
                    if not np.isclose(metrics(r)['training_MAE_days'], job['selection']['training_MAE_days'], atol=1e-8, rtol=0):
                        raise ValueError('Baseline training objective mismatch')
                studies, histories = {}, {}
                distributions = {k:optuna.distributions.FloatDistribution(*bounds) for k,bounds in space.items()}
                for jobid, job in jobs.items():
                    path = output/'folds'/jobid/'trials.json'
                    history = core.base.read_json(path) if path.exists() else []
                    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(
                        seed=args.seed, multivariate=True, constant_liar=True,
                        n_startup_trials=args.startup_trials, constraints_func=core.constraints))
                    for i,row in enumerate(history):
                        if row['number'] != i or len(history)>args.trials:
                            raise ValueError('Invalid checkpoint history')
                        m = row['metrics']
                        study.add_trial(optuna.trial.create_trial(params=row['params'], distributions=distributions,
                            value=m['optimizer_value'], user_attrs=m, system_attrs={'constraints':(m['n_no_prediction'],)}))
                    # Two training-independent initial anchors; counted in startup.
                    for index,beta in enumerate((0.0,1.0)):
                        if len(history)<=index:
                            study.enqueue_trial(dict(VPD_w=job['selection']['VPD_w'], I_req=job['selection']['I_req'], beta=beta))
                    studies[jobid], histories[jobid] = study, history
                if args.resume:
                    print('Warm resume: completed trials retained; sampler trajectory may differ from uninterrupted execution.', flush=True)
                while any(len(h)<args.trials for h in histories.values()):
                    pending = []
                    for jobid, job in jobs.items():
                        for _ in range(min(2,args.trials-len(histories[jobid]))):
                            trial = studies[jobid].ask()
                            params = {k:trial.suggest_float(k,*bounds) for k,bounds in space.items()}
                            futures = [pool.submit(evaluate,jobid,key,params) for key in job['train']]
                            pending.append((jobid,trial,params,futures))
                    # Fixed ask/tell order makes worker scheduling irrelevant to proposals.
                    for jobid,trial,params,futures in pending:
                        m = metrics([f.result() for f in futures])
                        for k,v in m.items(): trial.set_user_attr(k,v)
                        studies[jobid].tell(trial,m['optimizer_value'])
                        histories[jobid].append(dict(number=trial.number,params=params,metrics=m))
                        core.base.save_json(output/'folds'/jobid/'trials.json',histories[jobid])
                    status.update(stage='TPE', completed_trials=sum(map(len,histories.values())), baseline_checks_passed=True)
                    core.base.save_json(output/'run_metadata.json',status)
                    print(f"TPE {status['completed_trials']}/{status['total_trials']}",flush=True)
                records, selections = [], []
                for jobid, history in histories.items():
                    job = jobs[jobid]
                    core.base.save_frame(output/'folds'/jobid/'trials.csv',pd.DataFrame([
                        dict(number=r['number'],**r['params'],**r['metrics']) for r in history]))
                    valid = [r for r in history if r['metrics']['feasible']]
                    if not valid:
                        raise ValueError(f'No feasible joint candidate: {jobid}; inspect saved trials')
                    best = min(r['metrics']['training_MAE_days'] for r in valid)
                    choice = min((r for r in valid if r['metrics']['training_MAE_days']<=best+1e-10),
                                 key=lambda r:(r['params']['beta'],r['number']))
                    core.base.save_json(output/'folds'/jobid/'selection.json',dict(choice,
                        selection_uses_test_data=False,training_keys=job['train'],test_keys=job['test'],
                        selection_rule='minimum_training_MAE_then_smaller_beta_then_trial_number'))
                    futures = [pool.submit(evaluate,jobid,key,choice['params'],
                        str(output/'folds'/jobid/('train' if key in job['train'] else 'test')/key))
                        for key in job['train']+job['test']]
                    block = [dict(f.result(),arm='joint_tpe') for f in futures]
                    replay = metrics([r for r in block if r['case_role']=='train'])
                    if not replay['feasible'] or not np.isclose(replay['training_MAE_days'],best,atol=1e-8,rtol=0):
                        raise ValueError('Selected training replay mismatch')
                    records.extend(block)
                    selections.append(dict(scheme=job['scheme'],fold=job['fold'],**choice['params'],
                        selected_trial=choice['number'],training_MAE_days=best))
                runner.summarize(baseline+records,output)
                core.base.save_frame(output/'selected_parameters.csv',pd.DataFrame(selections))
                status.update(stage='exploratory_selection')
                core.base.save_json(output/'run_metadata.json',status)
                exploratory = exploratory_output(pool,jobs,histories,output,args.workers)
                runner.summarize(baseline+exploratory,output/'exploratory')
                # Keep one identical B row set in each version so that every
                # site-year/fold comparison has an explicit paired baseline.
                minimum = pd.DataFrame(baseline+records).assign(selection_version='minimum')
                tolerance = pd.DataFrame(baseline+exploratory).assign(selection_version='tolerance_1day')
                combined = pd.concat([minimum, tolerance], ignore_index=True)
                core.base.save_frame(output/'both_versions_predictions.csv',combined)
                comparison = []
                for version, frame in [('minimum', minimum), ('tolerance_1day', tolerance)]:
                    test = frame[frame.case_role.eq('test')].copy()
                    b = test[test.arm.eq('B')][['scheme','fold','input_key','status','interval_distance_days']].rename(
                        columns={'status':'baseline_status','interval_distance_days':'baseline_MAE_days'})
                    j = test[test.arm.eq('joint_tpe')][['scheme','fold','input_key','status','interval_distance_days']].rename(
                        columns={'status':'joint_status','interval_distance_days':'joint_MAE_days'})
                    paired = j.merge(b, on=['scheme','fold','input_key'], how='left', validate='one_to_one')
                    paired['selection_version'] = version
                    paired['MAE_change_vs_B_days'] = paired['joint_MAE_days'] - paired['baseline_MAE_days']
                    paired['both_predicted'] = paired[['joint_MAE_days','baseline_MAE_days']].notna().all(axis=1)
                    comparison.append(paired)
                core.base.save_frame(output/'versions_vs_B_predictions.csv', pd.concat(comparison, ignore_index=True))
                summaries = []
                for version, frame in [('minimum', minimum), ('tolerance_1day', tolerance)]:
                    (output/version).mkdir(parents=True, exist_ok=True)
                    s = runner.summarize(frame.to_dict('records'), output/version)
                    summaries.append(pd.read_csv(output/version/'method_summary.csv').assign(selection_version=version))
                core.base.save_frame(output/'versions_method_summary.csv', pd.concat(summaries, ignore_index=True))
            status.update(status='complete',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat())
            core.base.save_json(output/'run_metadata.json',status)
        except BaseException as exc:
            status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            core.base.save_json(output/'run_metadata.json',status)
            raise
    print(f'Joint calibration complete: {output}',flush=True)


if __name__ == '__main__':
    main()
