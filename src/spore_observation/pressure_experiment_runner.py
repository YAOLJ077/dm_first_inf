"""Common orchestration; parent owns Optuna studies, workers only evaluate models."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import multiprocessing as mp
from pathlib import Path

import pressure_model_core as core
import pandas as pd


def parser():
    p=argparse.ArgumentParser(description='Frozen B + pressure experiments; no B re-calibration')
    p.add_argument('--workers',type=int,default=64)
    p.add_argument('--trials',type=int,default=1024,help='Total beta trials per fold, including beta=0 and beta=1')
    p.add_argument('--startup-trials',type=int,default=128)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--schemes',nargs='+',choices=['losyo','site','year'],default=['losyo','site','year'])
    p.add_argument('--folds',nargs='+',help='Optional exact fold names for smoke checks')
    p.add_argument('--output-root',type=Path,default=core.ROOT/'result/spore_observation/plan1_pressure_experiments')
    p.add_argument('--pressure-dir',type=Path,default=core.ROOT/'result/spore_observation/plan1_pressure')
    p.add_argument('--replay-dir',type=Path,default=core.ROOT/'result/spore_observation/plan1_b_fixed')
    p.add_argument('--resume',action='store_true')
    return p


def normalize(args):
    if args.workers<1 or args.trials<2 or args.startup_trials<1:
        raise ValueError('workers/startup must be positive; trials >=2')
    if len(set(args.schemes))!=len(args.schemes):
        raise ValueError('Duplicate schemes')
    for name in ('output_root','pressure_dir','replay_dir'):
        setattr(args,name,getattr(args,name).resolve())
    for protected in (args.pressure_dir,args.replay_dir,core.ROOT/'result/fsim_abc_tpe_seed123'):
        if args.output_root==protected or protected in args.output_root.parents or args.output_root in protected.parents:
            raise ValueError('Use a separate experiment output root')
    return args


def anchors(jobid,key,destination):
    zero=core.evaluate(jobid,key,0.0,validate=True)
    one=core.evaluate(jobid,key,1.0,export=destination)
    return zero,one


def summarize(records,directory):
    frame=pd.DataFrame(records)
    for column in ('interval_distance_days','signed_interval_error_days','window_width_days'):
        if column not in frame:
            frame[column]=float('nan')
    frame=frame.sort_values(['arm','scheme','fold','case_role','input_key'])
    core.base.save_frame(directory/'predictions.csv',frame)
    summaries=[]
    for (arm,scheme),g in frame[frame.case_role.eq('test')].groupby(['arm','scheme']):
        ok=g[g.status.eq('predicted')]
        summaries.append(dict(arm=arm,scheme=scheme,n_test=len(g),n_predictions=len(ok),
            n_no_predictions=len(g)-len(ok),MAE_days=ok.interval_distance_days.mean() if len(ok)==len(g) else float('nan'),
            conditional_MAE_days=ok.interval_distance_days.mean(),mean_signed_error_days=ok.signed_interval_error_days.mean(),
            mean_window_width_days=ok.window_width_days.mean(),n_overlap=int(ok.overlap.sum())))
    core.base.save_frame(directory/'method_summary.csv',pd.DataFrame(summaries))
    return frame


def run(mode,args):
    args=normalize(args)
    jobs,identity,snapshot,manifest=core.prepare(args)
    directory=args.output_root/mode
    identity.update(mode=mode,seed=args.seed,
        trials=args.trials if mode=='beta_tpe' else None,startup_trials=args.startup_trials if mode=='beta_tpe' else None)
    if mode=='beta_tpe':
        identity['optuna_version']=importlib.metadata.version('optuna')
    directory.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (directory/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idpath=directory/'run_identity.json'
        if idpath.exists():
            if not args.resume or core.base.read_json(idpath)!=identity:
                raise ValueError('Existing experiment: use --resume with identical inputs/code/design, or a new output root')
        elif any(x.name!='.run.lock' and not x.name.startswith('._') for x in directory.iterdir()):
            raise ValueError('Nonempty experiment directory lacks identity')
        core.base.save_json(idpath,identity)
        status=dict(status='running',mode=mode,workers=args.workers,started_utc=datetime.now(timezone.utc).isoformat(),
                    n_folds=len(jobs),stage='anchors')
        core.base.save_json(directory/'run_metadata.json',status)
        try:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                initializer=core.initialize,initargs=(str(snapshot),str(manifest),jobs,str(args.pressure_dir),str(args.replay_dir))) as pool:
                futures=[]
                for jobid,job in jobs.items():
                    for key in job['train']+job['test']:
                        role='train' if key in job['train'] else 'test'
                        export=str(directory/'folds'/jobid/role/key) if mode=='fixed' else None
                        futures.append(pool.submit(anchors,jobid,key,export))
                baseline,ones=[],[]
                for future in as_completed(futures):
                    zero,one=future.result()
                    baseline.append(dict(zero,arm='B'));ones.append(dict(one,arm='fixed'))
                    if len(baseline)%6==0:
                        print(f'{mode}: anchor checks {len(baseline)}/{len(futures)}',flush=True)
                core.base.save_frame(directory/'anchor_predictions.csv',pd.DataFrame(baseline+ones))
                # Training beta=0 must also reproduce saved B selection objective.
                for jobid,job in jobs.items():
                    errors=[r['interval_distance_days'] for r in baseline if r['scheme']==job['scheme'] and r['fold']==job['fold'] and r['case_role']=='train']
                    if abs(sum(errors)/len(errors)-job['selection']['training_MAE_days'])>1e-8:
                        raise ValueError(f'Training baseline mismatch: {jobid}')
                if mode=='fixed':
                    for jobid,job in jobs.items():
                        core.base.save_json(directory/'folds'/jobid/'selection.json',dict(beta=1.0,
                            selection_rule='fixed_mapping_no_fitting',selection_uses_test_data=False,
                            training_keys=job['train'],test_keys=job['test'],B_selection=job['selection']))
                    summarize(baseline+ones,directory)
                else:
                    selected=calibrate(args,jobs,pool,directory,status,ones)
                    outputs=[]
                    for jobid,choice in selected.items():
                        job=jobs[jobid]
                        for key in job['train']+job['test']:
                            role='train' if key in job['train'] else 'test'
                            outputs.append(pool.submit(core.evaluate,jobid,key,choice['beta'],
                                str(directory/'folds'/jobid/role/key)))
                    records=[dict(f.result(),arm='beta_tpe') for f in as_completed(outputs)]
                    for jobid,choice in selected.items():
                        job=jobs[jobid]
                        train=[r for r in records if r['scheme']==job['scheme'] and r['fold']==job['fold'] and r['case_role']=='train']
                        if any(r['status']!='predicted' for r in train) or abs(sum(r['interval_distance_days'] for r in train)/len(train)-choice['metrics']['training_MAE_days'])>1e-8:
                            raise ValueError('Selected training objective failed replay')
                    summarize(baseline+records,directory)
            status.update(status='complete',stage='complete',baseline_checks_passed=True,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            core.base.save_json(directory/'run_metadata.json',status)
        except BaseException as exc:
            status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            core.base.save_json(directory/'run_metadata.json',status)
            raise
    print(f'{mode} complete: {directory}',flush=True)


def calibrate(args,jobs,pool,directory,status,ones):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    histories={};studies={}
    for jobid in jobs:
        path=directory/'folds'/jobid/'trials.json'
        history=core.base.read_json(path) if path.exists() else []
        if len(history)>args.trials:
            raise ValueError('Existing trial count exceeds requested budget')
        histories[jobid]=history
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(
            seed=args.seed,multivariate=True,constant_liar=True,n_startup_trials=args.startup_trials,
            constraints_func=core.constraints))
        for i,row in enumerate(history):
            if row['number']!=i or not 0<=row['beta']<=1:
                raise ValueError('Malformed saved trial history')
            m=row['metrics']
            study.add_trial(optuna.trial.create_trial(params={'beta':row['beta']},
                distributions={'beta':optuna.distributions.FloatDistribution(0.0,1.0)},value=m['optimizer_value'],
                user_attrs=m,system_attrs={'constraints':(m['n_no_prediction'],)}))
        if len(history)==0:study.enqueue_trial({'beta':0.0})
        if len(history)<2:study.enqueue_trial({'beta':1.0})
        studies[jobid]=study
    if args.resume:
        print('Resume: completed trials retained; TPE warm-starts from history. Sampler trajectory may differ from an uninterrupted run.',flush=True)
    target=len(jobs)*args.trials
    while sum(len(h) for h in histories.values())<target:
        pending=[];reserved={j:0 for j in jobs}
        # Round-robin across folds; a single global pool caps concurrency.
        while len(pending)<args.workers:
            progress=False
            for jobid in jobs:
                if len(histories[jobid])+reserved[jobid]>=args.trials or len(pending)>=args.workers:continue
                trial=studies[jobid].ask();beta=trial.suggest_float('beta',0.0,1.0)
                pending.append((jobid,trial,beta,pool.submit(core.train_metrics,jobid,beta)))
                reserved[jobid]+=1;progress=True
            if not progress:break
        # Tell in ask order, independent of worker completion order.
        for jobid,trial,beta,future in pending:
            metrics=future.result()
            for k,v in metrics.items():trial.set_user_attr(k,v)
            studies[jobid].tell(trial,metrics['optimizer_value'])
            history=histories[jobid]
            row=dict(number=len(history),beta=beta,metrics=metrics)
            if row['number'] in (0,1):
                expected_beta=float(row['number'])
                if beta!=expected_beta:raise ValueError('Missing endpoint anchor')
                if beta==1:
                    job=jobs[jobid]
                    train=[r for r in ones if r['scheme']==job['scheme'] and r['fold']==job['fold'] and r['case_role']=='train']
                    missing=sum(r['status']=='no_prediction' for r in train)
                    if missing!=metrics['n_no_prediction']:raise ValueError('Beta one / fixed missingness mismatch')
                    if not missing and abs(sum(r['interval_distance_days'] for r in train)/len(train)-metrics['training_MAE_days'])>1e-8:
                        raise ValueError('Beta one / fixed training mismatch')
            history.append(row)
            core.base.save_json(directory/'folds'/jobid/'trials.json',history)
        completed=sum(len(h) for h in histories.values())
        status.update(stage='TPE',completed_trials=completed,total_trials=target)
        core.base.save_json(directory/'run_metadata.json',status)
        print(f'TPE: {completed}/{target} trials complete',flush=True)
    selections={}
    for jobid,history in histories.items():
        valid=[r for r in history if r['metrics']['feasible']]
        if not valid:raise ValueError(f'No feasible beta, despite baseline anchor: {jobid}')
        best=min(r['metrics']['training_MAE_days'] for r in valid)
        choice=min((r for r in valid if r['metrics']['training_MAE_days']<=best+1e-10),key=lambda r:(r['beta'],r['number']))
        job=jobs[jobid];selections[jobid]=choice
        core.base.save_json(directory/'folds'/jobid/'selection.json',dict(choice,
            selection_rule='minimum_training_MAE_then_smaller_beta_then_trial_number',selection_uses_test_data=False,
            training_keys=job['train'],test_keys=job['test'],B_selection=job['selection']))
        core.base.save_frame(directory/'folds'/jobid/'trials.csv',pd.DataFrame([
            dict(number=r['number'],beta=r['beta'],**r['metrics']) for r in history]))
    return selections


def entry(mode):
    run(mode,parser().parse_args())
