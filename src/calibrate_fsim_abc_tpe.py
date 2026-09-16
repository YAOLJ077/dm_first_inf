#!/usr/bin/env python3
"""Seed-123 A/B/C TPE: 12 concurrent studies, startup prefetch, fixed pairs.

One continuously fed 64-process pool serves all plans in A/B/C order. All
startup proposals are made before feedback; adaptive pairs are asked before
either is told. Each study commits/feeds back in trial order, regardless of
process completion order. This is batch TPE, not legacy sequential TPE.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, wait
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import optuna
import pandas as pd

import calibrate_fsim_joblib_local as legacy
from fsim_joblib_pool import JoblibPool

ROOT = Path(__file__).resolve().parents[1]
core = legacy.core
NAMES = ('VPD_w', 'I_req', 'RH_sp', 'MP_sp', 'u_tr')


def model_worker(task):
    cid, params, wind_gate = task
    started = time.perf_counter()
    sample = dict(model='M1' if wind_gate else 'M0', sample_id='abc_tpe', **params)
    cfg = core.configure(core._BASE, sample)
    cfg['mp_tmp']['rh_wet_night'] = params['RH_sp']
    cfg['fixed']['min_moist_hours'] = params['MP_sp']
    cfg['dispersal']['use_wind_gate'] = wind_gate
    record = core.evaluate_group((core._CASES[cid], [sample], cfg))[0]
    return dict(record=record, model_seconds=time.perf_counter()-started,
                worker_started=started, worker_pid=os.getpid())


class Store(legacy.Store):
    def save_evaluations(self, entries):
        with self.lock:
            with self.db:
                self.db.executemany('INSERT OR IGNORE INTO evaluations VALUES (?, ?)',
                    [(key,legacy.json_text(payload)) for key,payload in entries])

    def save_batch(self, job, records):
        # Atomic startup/pair checkpoints; evaluations are committed as they
        # finish and survive interruption even before a whole batch completes.
        with self.lock:
            with self.db:
                self.db.executemany('INSERT INTO trials VALUES (?, ?, ?)',
                    [(job, r['number'], legacy.json_text(r)) for r in records])


def validate(s):
    if s['order'] != ['A', 'B', 'C'] or s['schemes'] != ['losyo', 'site', 'year']:
        raise ValueError('Requires A/B/C order and LOSYO/site/year studies')
    if s['seed'] != 123 or s['batch_size'] != 2 or s['fixed_parameters'] != {'RH_sp':80, 'MP_sp':3}:
        raise ValueError('Requires seed=123, adaptive batch=2, RH=80, MP=3')
    if s['workers'] not in legacy.WORKERS or s['queue_capacity'] < s['workers']:
        raise ValueError('Unsupported workers or queue capacity smaller than workers')
    for name in ('n_trials', 'n_startup_trials', 'queue_capacity'):
        if type(s[name]) is not int or s[name] < 1:
            raise ValueError(f'Invalid {name}')
    if s['n_startup_trials'] >= s['n_trials'] or (s['n_trials']-s['n_startup_trials']) % 2:
        raise ValueError('Budget must include startup plus complete adaptive pairs')
    if not s.get('smoke') and (s['n_startup_trials'] != 128 or s['n_trials'] not in (1024,2048)):
        raise ValueError('Formal budget is 1024/2048 with 128 startup trials')
    if not np.isfinite(s['heartbeat_seconds']) or s['heartbeat_seconds'] <= 0:
        raise ValueError('Invalid heartbeat interval')
    for plan in s['order']:
        p=s['plans'][plan]
        names=['VPD_w','I_req'] + (['u_tr'] if plan=='C' else [])
        if list(p['search_space']) != names or p['wind_gate'] != (plan=='C'):
            raise ValueError(f'Incorrect dimensions or wind gate for {plan}')
        for name,(lo,hi) in p['search_space'].items():
            if not np.isfinite([lo,hi]).all() or not lo < hi or lo < 0 or (name!='u_tr' and lo==0):
                raise ValueError(f'Invalid {plan}/{name} bounds')
            if not lo <= s['initial_point'][name] <= hi:
                raise ValueError('Initial point outside search space')
    if any(s['plans']['B']['search_space'][n] != s['plans']['C']['search_space'][n] for n in ('VPD_w','I_req')):
        raise ValueError('B and C must have identical VPD/I ranges')
    if type(s['check_duplicate_policy']) is not bool:
        raise ValueError('Duplicate policy must be boolean')
    if s['selection_tolerance_days'] != 1.0:
        raise ValueError('This exploration requires a fixed 1-day training MAE tolerance')
    if s['selection_rule'] != 'training_tolerance_then_test_MAE':
        raise ValueError('Requires exploratory test-MAE selection within the training tolerance')
    for name in ('near_optimal_tolerances_days','accuracy_tolerances_days'):
        if not s[name] or any(not np.isfinite(v) or v < 0 for v in s[name]):
            raise ValueError(f'Invalid {name}')


def make_jobs(s, manifest, plan):
    keys=manifest.input_key.tolist()
    if len(keys)!=6 or len(set(keys))!=6 or manifest.input_key.isna().any():
        raise ValueError('Exactly six unique site-years required')
    jobs=[]
    for scheme in s['schemes']:
        groups={k:[k] for k in keys} if scheme=='losyo' else core.make_groups(manifest,scheme)
        for group,test in groups.items():
            jobs.append(dict(id=f'{plan}/{scheme}/seed_{s["seed"]}/folds/{group}',
                plan=plan, arm=plan, scheme=scheme, held_out=group, seed=s['seed'],
                train=[k for k in keys if k not in test], test=test))
    return jobs


def make_study(s, job):
    sampler=optuna.samplers.TPESampler(seed=s['seed'], multivariate=True,
        constant_liar=True, n_startup_trials=s['n_startup_trials'], constraints_func=core.constraints)
    study=optuna.create_study(direction='minimize',sampler=sampler)
    study.enqueue_trial({n:s['initial_point'][n] for n in s['plans'][job['plan']]['search_space']})
    return study


def ask(s, job, study):
    trial=study.ask()
    params=dict(s['fixed_parameters'],u_tr=0.0)
    for name,(lo,hi) in s['plans'][job['plan']]['search_space'].items():
        params[name]=trial.suggest_float(name,lo,hi)
    return trial,params


class Batch:
    def __init__(self, candidates, keys, wind_gate, scheduler):
        self.candidates=candidates
        self.requests=[]
        self.todo=deque()
        self.pending=set()
        self.results={}
        ids=[scheduler.primary[k] for k in keys]
        ids += [scheduler.alternate[k] for k in keys if k in scheduler.alternate]
        seen=set()
        for _,params in candidates:
            requests=[]
            for cid in ids:
                key=scheduler.cache_key(cid,params,wind_gate)
                requests.append(key)
                if key not in seen:
                    seen.add(key)
                    self.todo.append((key,cid,params,wind_gate))
            self.requests.append(requests)
        self.keys=keys

    @property
    def ready(self):
        return not self.todo and not self.pending

    def predictions(self,index):
        rows=[self.results[k] for k in self.requests[index]]
        primary=rows[:len(self.keys)]; alternates=rows[len(self.keys):]
        lookup={r['input_key']:r for r in alternates}
        for r in primary:
            other=lookup.get(r['input_key'])
            if other and any(r.get(n)!=other.get(n) for n in ('status','predicted_start','predicted_end')):
                raise ValueError(f'Duplicate weather policies disagree: {r["input_key"]}')
        return primary,alternates


class StudyState:
    def __init__(self,s,job,store,scheduler,on_complete):
        self.s,self.job,self.store,self.scheduler=s,job,store,scheduler
        self.on_complete=on_complete
        self.study=make_study(s,job)
        self.saved=store.trials(job['id'])
        if len(self.saved)>s['n_trials']:
            raise ValueError('Cannot shrink a completed/checkpointed budget')
        self.records=[]
        self.batch=None
        self.done=False

    def advance(self):
        if len(self.records)==self.s['n_trials']:
            self.done=True
            self.on_complete(self)
            return
        count=self.s['n_startup_trials'] if not self.records else self.s['batch_size']
        candidates=[ask(self.s,self.job,self.study) for _ in range(count)]
        for trial,params in candidates:
            if trial.number<len(self.saved):
                r=self.saved[trial.number]
                if (r['number']!=trial.number or r['params']!=params or r['seed']!=self.job['seed']
                    or r['training_keys']!=self.job['train']):
                    raise ValueError(f'Checkpoint proposal mismatch: {self.job["id"]}/{trial.number}')
        self.batch=Batch(candidates,self.job['train'],self.s['plans'][self.job['plan']]['wind_gate'],self.scheduler)

    def consume(self):
        records=[]
        for i,(trial,params) in enumerate(self.batch.candidates):
            predictions,duplicates=self.batch.predictions(i)
            metrics=core.training_metrics(predictions,self.job['train'])
            r=dict(number=trial.number,seed=self.job['seed'],params=params,training_keys=self.job['train'],
                   metrics=metrics,requested_model_evaluations=len(predictions)+len(duplicates))
            if trial.number<len(self.saved):
                if self.saved[trial.number]!=r:
                    raise ValueError('Saved metrics differ from exact evaluation replay')
            records.append(r)
        self.store.save_batch(self.job['id'],[r for r in records if r['number']>=len(self.saved)])
        # Never feed back the first candidate while the second is still pending.
        for (trial,_),r in zip(self.batch.candidates,records):
            core.tell(self.study,trial,r['metrics'])
        self.records.extend(records)
        self.batch=None


class EvaluationState:
    """A frozen held-out selection uses the same cache and bounded scheduler."""
    def __init__(self,params,keys,wind_gate,scheduler,on_complete):
        self.batch=Batch([(None,params)],keys,wind_gate,scheduler)
        self.done=False
        self.on_complete=on_complete

    def advance(self):
        raise RuntimeError('Evaluation batch already initialized')

    def consume(self):
        self.on_complete(*self.batch.predictions(0))
        self.done=True


class CandidateEvaluationState(EvaluationState):
    """All training-qualified candidates for one fold share the global queue."""
    def __init__(self,candidates,keys,wind_gate,scheduler,on_complete):
        self.candidates=candidates
        self.batch=Batch([(None,r['params']) for r in candidates],keys,wind_gate,scheduler)
        self.done=False
        self.on_complete=on_complete

    def consume(self):
        self.on_complete([self.batch.predictions(i) for i in range(len(self.candidates))])
        self.done=True


class Scheduler:
    """One owner thread; fair request dispatch; at most queue_capacity futures."""
    def __init__(self,s,cases,alternates,base,store,output,pool=None,worker=model_worker):
        self.s,self.store,self.output,self.base=s,store,output,base
        self.cases=cases+alternates
        self.primary={c['input_key']:i for i,c in enumerate(cases)}
        self.alternate={c['input_key']:i+len(cases) for i,c in enumerate(alternates)}
        self.pool=pool or JoblibPool(s['workers'],initializer=core.worker_init,initargs=(self.cases,base))
        self.worker=worker
        self.inflight={}
        self.future_keys={}
        self.lru=OrderedDict()
        self.phase='setup'
        self.started=time.perf_counter()
        self.last_heartbeat=0
        self.stats=dict(requests=0,cache_hits=0,inflight_shares=0,submitted=0,completed=0,
                        model_seconds=0.0,queue_wait_seconds=0.0,peak_inflight=0)
        self.pids=set()
        self.waiting_by_study={}

    def cache_key(self,cid,params,wind_gate):
        value=[self.cases[cid]['input_key'],'alternate' if cid>=len(self.primary) else 'primary',
               bool(wind_gate),[float(params[n]) for n in NAMES]]
        return hashlib.sha256(legacy.json_text(value).encode()).hexdigest()

    def cached(self,key):
        if key in self.lru:
            self.lru.move_to_end(key)
            return self.lru[key]
        payload=self.store.evaluation(key)
        if payload is not None:
            self.remember(key,payload)
        return payload

    def remember(self,key,payload):
        self.lru[key]=payload
        self.lru.move_to_end(key)
        if len(self.lru)>4096:self.lru.popitem(last=False)

    def dispatch_one(self,batch):
        if not batch.todo:return False
        key,cid,params,gate=batch.todo[0]
        payload=self.cached(key)
        if payload is not None:
            batch.results[key]=payload['record'];self.stats['cache_hits']+=1
        elif key in self.inflight:
            self.inflight[key]['waiters'].append(batch)
            batch.pending.add(key);self.stats['inflight_shares']+=1
        else:
            if len(self.inflight)>=self.s['queue_capacity']:return False
            future=self.pool.submit(self.worker,(cid,params,gate))
            self.inflight[key]=dict(future=future,cid=cid,params=params,waiters=[batch],submitted=time.perf_counter())
            self.future_keys[future]=key
            batch.pending.add(key);self.stats['submitted']+=1
            self.stats['peak_inflight']=max(self.stats['peak_inflight'],len(self.inflight))
        batch.todo.popleft();self.stats['requests']+=1
        return True

    def collect(self):
        if not self.future_keys:return
        done,_=wait(list(self.future_keys),timeout=1,return_when=FIRST_COMPLETED)
        completed=[]
        for future in done:
            key=self.future_keys.pop(future);task=self.inflight.pop(key)
            payload=future.result();r=payload['record']
            if r['input_key']!=self.cases[task['cid']]['input_key'] or any(r[n]!=task['params'][n] for n in NAMES):
                raise ValueError('Worker returned mismatched model parameters/case')
            if r['status'] not in ('predicted','no_prediction'):
                raise RuntimeError(legacy.json_text(r))
            completed.append((key,payload,task))
        # Commit completed results together, not one filesystem sync per case.
        self.store.save_evaluations([(key,payload) for key,payload,_ in completed])
        for key,payload,task in completed:
            r=payload['record'];self.remember(key,payload)
            for batch in task['waiters']:
                batch.results[key]=r;batch.pending.remove(key)
            self.stats['completed']+=1;self.stats['model_seconds']+=payload['model_seconds']
            self.stats['queue_wait_seconds']+=max(0,payload['worker_started']-task['submitted'])
            self.pids.add(payload['worker_pid'])

    def heartbeat(self,states,force=False):
        now=time.perf_counter()
        if not force and now-self.last_heartbeat<self.s['heartbeat_seconds']:return
        self.last_heartbeat=now
        elapsed=now-self.started
        row=dict(utc=legacy.utc_now(),phase=self.phase,elapsed_seconds=elapsed,
            in_flight_tasks=len(self.inflight),ready_requests=sum(len(x.batch.todo) for x in states if x.batch),
            unfinished_studies=sum(not x.done for x in states),worker_pids_seen=len(self.pids),
            completed_model_tasks_per_second=self.stats['completed']/max(elapsed,1e-9),
            completed_model_time_utilization=self.stats['model_seconds']/max(elapsed*self.s['workers'],1e-9),**self.stats)
        path=self.output/'scheduler_metrics.csv'
        exists=path.exists()
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not exists:writer.writeheader()
            writer.writerow(row)
        progress=[]
        for state in states:
            if isinstance(state,StudyState):
                progress.append(dict(job=state.job['id'],completed_trials=len(state.records),
                    queued_requests=len(state.batch.todo) if state.batch else 0,
                    pending_unique_results=len(state.batch.pending) if state.batch else 0,done=state.done))
        core.write_json(self.output/'live_progress.json',dict(**row,studies=progress))
        print(f'{self.phase}: unfinished={row["unfinished_studies"]}, in_flight={len(self.inflight)}, '
              f'queued={row["ready_requests"]}, computed={self.stats["completed"]}, '
              f'cache_hits={self.stats["cache_hits"]}',flush=True)

    def run(self,states,phase):
        self.phase=phase
        for state in states:
            if state.batch is None:state.advance()
        while any(not x.done for x in states):
            # Round-robin across all 12 studies rather than exhausting a large
            # startup plan from the first study before serving the others.
            while True:
                progressed=False
                for state in states:
                    if not state.done and state.batch:
                        progressed=self.dispatch_one(state.batch) or progressed
                if not progressed:break
            for state in states:
                if not state.done and state.batch and state.batch.ready:
                    state.consume()
                    if not state.done:state.advance()
            self.heartbeat(states)
            self.collect()
        self.heartbeat(states,force=True)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)


def save_training(s,state,output):
    directory=output/state.job['id'];directory.mkdir(parents=True,exist_ok=True)
    core.write_json(directory/'job.json',state.job)
    frame=pd.DataFrame([core.flat_trial(r) for r in state.records])
    frame.to_csv(directory/'trials.csv',index=False)
    losses=frame.training_MAE_days.where(frame.feasible).cummin().ffill()
    curve=pd.DataFrame(dict(completed_trials=np.arange(1,len(frame)+1),best_training_MAE_days=losses,
                            n_feasible=frame.feasible.cumsum()))
    curve.to_csv(directory/'convergence.csv',index=False)
    curve[curve.completed_trials.isin(s['training_checkpoints']+[s['n_trials']])].to_csv(directory/'training_checkpoints.csv',index=False)


def selection_candidates(records,s):
    """Fixed 1-day tolerance, not an estimated one-standard-error interval.

    Only complete training predictions qualify. Deduplicate exact parameter
    combinations so repeated TPE proposals do not receive extra weight.
    """
    best,_=core.choose_training(records,s['tie_atol_days'])
    if best is None:
        return []
    limit=best['metrics']['training_MAE_days']+s['selection_tolerance_days']
    candidates={}
    for r in records:
        if r['metrics']['feasible'] and r['metrics']['training_MAE_days']<=limit+s['tie_atol_days']:
            key=tuple(float(r['params'][n]) for n in NAMES)
            if key not in candidates or r['number']<candidates[key]['number']:
                candidates[key]=r
    return sorted(candidates.values(),key=lambda r:r['number'])


def exploratory_selection(s,job,budget,records,candidates,results):
    """Select one joint parameter vector for a whole test fold, never per case.

    Incomplete test predictions cannot compete via a conditional MAE. An empty
    feasible test set yields explicit selection failure, retaining the baseline.
    """
    best,_=core.choose_training(records[:budget],s['tie_atol_days'])
    if len(candidates)!=len(results):
        raise ValueError('Missing candidate test results')
    summaries=[]; predictions=[]; duplicate_rows=[]
    for r,(primary,duplicates) in zip(candidates,results):
        metrics=core.training_metrics(primary,job['test'])
        errors=[abs(p['signed_interval_error_days']) for p in primary if p['status']=='predicted']
        summaries.append(dict(**core.flat_trial(r),test_prediction_complete=metrics['feasible'],
            n_test=len(job['test']),n_test_predicted=len(errors),
            test_MAE_days=metrics['training_MAE_days'],test_RMSE_days=metrics['training_RMSE_days'],
            conditional_test_MAE_days=float(np.mean(errors)) if errors else None))
        predictions.extend(dict(p,candidate_trial=r['number']) for p in primary)
        duplicate_rows.extend(dict(p,candidate_trial=r['number']) for p in duplicates)
    eligible=[i for i,m in enumerate(summaries) if m['test_prediction_complete']]
    selected=None
    if eligible:
        test_min=min(summaries[i]['test_MAE_days'] for i in eligible)
        ties=[i for i in eligible if summaries[i]['test_MAE_days']<=test_min+s['tie_atol_days']]
        selected=min(ties,key=lambda i:(candidates[i]['metrics']['training_MAE_days'],
            candidates[i]['metrics']['training_RMSE_days'],
            *(float(candidates[i]['params'][n]) for n in NAMES),candidates[i]['number']))
    selection=dict(training_keys=job['train'],test_keys=job['test'],n_trials=budget,
        selection_rule=s['selection_rule'],selection_tolerance_days=s['selection_tolerance_days'],
        selection_uses_test_data=True,evaluation_role='exploratory_test_selected',
        minimum_training_MAE_days=best['metrics']['training_MAE_days'] if best else None,
        n_unique_candidates=len(candidates),n_complete_test_candidates=len(eligible),
        calibration_status='selected' if selected is not None else
            ('no_complete_test_prediction' if candidates else 'no_feasible_training_combination'))
    if selected is not None:
        r=candidates[selected]
        selection.update(**r['params'],selected_trial=r['number'],
            training_MAE_days=r['metrics']['training_MAE_days'],
            test_MAE_days=summaries[selected]['test_MAE_days'])
        chosen,duplicates=results[selected]
    else:
        chosen=[dict(input_key=k,status='selection_failed',signed_interval_error_days=None) for k in job['test']]
        duplicates=[]
    return dict(selection=selection,predictions=chosen,duplicate_predictions=duplicates,
        candidate_metrics=summaries,candidate_predictions=predictions,
        candidate_duplicate_predictions=duplicate_rows)


def write_exploratory_files(directory,payload,base,s,job):
    directory.mkdir(parents=True,exist_ok=True)
    selection=payload['selection']
    core.write_json(directory/'selection.json',selection)
    for name,field in [('selected_predictions','predictions'),('candidate_metrics','candidate_metrics'),
                       ('candidate_predictions','candidate_predictions')]:
        frame=pd.DataFrame(payload[field])
        frame['selection_uses_test_data']=True
        frame['evaluation_role']='exploratory_test_selected'
        frame.to_csv(directory/f'{name}.csv',index=False)
    if selection['calibration_status']=='selected':
        sample=dict(model='M1' if s['plans'][job['plan']]['wind_gate'] else 'M0',
                    **{n:selection[n] for n in NAMES})
        cfg=core.configure(base,sample)
        cfg['mp_tmp']['rh_wet_night']=selection['RH_sp']
        cfg['fixed']['min_moist_hours']=selection['MP_sp']
        core.write_json(directory/'effective_runtime_config.json',cfg)


def report_exploratory(s,jobs,store,scheduler,output):
    states=[]
    budgets=[v for v in (1024,2048) if v<=s['n_trials']] if not s.get('smoke') else [s['n_trials']]
    for job in jobs:
        records=store.trials(job['id'])
        for budget in budgets:
            candidates=selection_candidates(records[:budget],s)
            directory=output/job['id']/f'budget_{budget}'/'exploratory'
            existing=store.heldout(job['id']+'/exploratory',budget)
            if existing is not None:
                write_exploratory_files(directory,existing,scheduler.base,s,job)
                continue
            def finished(results,job=job,budget=budget,records=records,candidates=candidates,directory=directory):
                payload=exploratory_selection(s,job,budget,records,candidates,results)
                store.save_heldout(job['id']+'/exploratory',budget,payload)
                write_exploratory_files(directory,payload,scheduler.base,s,job)
            if candidates:
                states.append(CandidateEvaluationState(candidates,job['test'],
                    s['plans'][job['plan']]['wind_gate'],scheduler,finished))
            else:
                finished([])
    if states:
        scheduler.run(states,jobs[0]['plan']+'_exploratory_selection')


def report_plan(s,jobs,store,scheduler,output):
    states=[]
    for job in jobs:
        records=store.trials(job['id'])
        budgets=[v for v in (1024,2048) if v<=s['n_trials']] if not s.get('smoke') else [s['n_trials']]
        for budget in budgets:
            if len(records)<budget:raise ValueError('Incomplete training records')
            choice,ties=core.choose_training(records[:budget],s['tie_atol_days'])
            selection=dict(training_keys=job['train'],test_keys=job['test'],n_trials=budget,
                selection_rule='minimum_training_MAE',selection_uses_test_data=False,
                evaluation_role='training_selected_baseline',
                calibration_status='selected' if choice else 'no_feasible_training_combination')
            if choice:
                selection.update(**choice['params'],selected_trial=choice['number'],training_MAE_days=choice['metrics']['training_MAE_days'])
            directory=output/job['id']/f'budget_{budget}';directory.mkdir(parents=True,exist_ok=True)
            core.write_json(directory/'selection.json',selection)
            candidates=selection_candidates(records[:budget],s)
            pd.DataFrame([core.flat_trial(r) for r in candidates]).to_csv(
                directory/'selection_candidates.csv',index=False)
            core.write_json(directory/'candidate_rule.json',dict(
                rule='fixed_training_MAE_tolerance',tolerance_days=s['selection_tolerance_days'],
                training_predictions_must_be_complete=True,
                minimum_training_MAE_days=choice['metrics']['training_MAE_days'] if choice else None,
                n_unique_candidates=len(candidates)))
            pd.DataFrame([core.flat_trial(t) for t in ties]).to_csv(directory/'tied_best_parameters.csv',index=False)
            if choice:
                sample=dict(model='M1' if s['plans'][job['plan']]['wind_gate'] else 'M0',**choice['params'])
                cfg=core.configure(scheduler.base,sample)
                cfg['mp_tmp']['rh_wet_night']=choice['params']['RH_sp']
                cfg['fixed']['min_moist_hours']=choice['params']['MP_sp']
                core.write_json(directory/'effective_runtime_config.json',cfg)
                near=[dict(tolerance_days=tol,**core.flat_trial(r)) for tol in s['near_optimal_tolerances_days']
                    for r in records[:budget] if r['metrics']['feasible'] and
                    r['metrics']['training_MAE_days']<=choice['metrics']['training_MAE_days']+tol+s['tie_atol_days']]
                pd.DataFrame(near).to_csv(directory/'near_optimal_parameters.csv',index=False)
            existing=store.heldout(job['id'],budget)
            if existing:
                if existing['selection']!=selection:raise ValueError('Frozen held-out selection changed')
                pd.DataFrame(existing['predictions']).to_csv(directory/'held_out_predictions.csv',index=False)
                continue
            def finished(predictions,duplicates,job=job,budget=budget,selection=selection,directory=directory):
                store.save_heldout(job['id'],budget,dict(selection=selection,predictions=predictions,duplicate_predictions=duplicates))
                pd.DataFrame(predictions).to_csv(directory/'held_out_predictions.csv',index=False)
            if choice:
                states.append(EvaluationState(choice['params'],job['test'],s['plans'][job['plan']]['wind_gate'],scheduler,finished))
            else:
                finished([dict(input_key=k,status='calibration_failed',signed_interval_error_days=None) for k in job['test']],[])
    if states:scheduler.run(states,jobs[0]['plan']+'_held_out')


def write_summary(s,all_jobs,store,output,exploratory=False):
    rows,folds,methods=[],[],[]
    budgets=[s['n_trials']] if s.get('smoke') else [n for n in (1024,2048) if n<=s['n_trials']]
    for plan in s['order']:
        for scheme in s['schemes']:
            jobs=[j for j in all_jobs if j['plan']==plan and j['scheme']==scheme]
            for budget in budgets:
                held=[store.heldout(j['id']+('/exploratory' if exploratory else ''),budget) for j in jobs]
                if any(h is None for h in held):continue
                subset=[]
                for job,h in zip(jobs,held):
                    tag=dict(arm=plan,scheme=scheme,held_out=job['held_out'],seed=s['seed'],
                             view=f'budget_{budget}',n_trials=budget)
                    folds.append(dict(**tag,**{k:v for k,v in h['selection'].items() if k!='n_trials'}))
                    subset.extend({**r,**tag,'calibration_status':h['selection']['calibration_status'],
                        'signed_interval_error_days':r.get('signed_interval_error_days')} for r in h['predictions'])
                frame=pd.DataFrame(subset)
                methods.append(dict(arm=plan,scheme=scheme,seed=s['seed'],n_trials=budget,
                    **core.group_summary(frame,s,{j['held_out']:j['test'] for j in jobs})))
                rows.extend(subset)
    if not rows:return
    frame=pd.DataFrame(rows)
    paired,comparisons=legacy.compare_predictions(frame)
    for name,table in [('oof_predictions',frame),('fold_summary',pd.DataFrame(folds)),
        ('method_summary',pd.DataFrame(methods)),('paired_predictions',paired),('comparison_summary',comparisons)]:
        table['selection_uses_test_data']=exploratory
        table['evaluation_role']='exploratory_test_selected' if exploratory else 'training_selected_baseline'
        if exploratory:
            table.rename(columns={'n_calibration_failed_folds':'n_selection_failed_folds',
                'n_calibration_failed_site_years':'n_selection_failed_site_years'},inplace=True)
        filename=('exploratory_'+('predictions' if name=='oof_predictions' else name)) if exploratory else name
        table.to_csv(output/f'{filename}.csv',index=False)


def identity(s,all_jobs):
    versions,frozen=legacy.verify_engine(s)
    source={Path(__file__).resolve(),Path(legacy.__file__),Path(core.__file__),ROOT/'src/fsim_joblib_pool.py',
            core.REFERENCE/'snapshot_manifest.json'}
    source.update(core.REFERENCE/p for p in frozen['sha256'])
    manifest=pd.read_csv(ROOT/s['manifest'])
    inputs={ROOT/s['manifest']}
    inputs.update(ROOT/p for p in manifest.weather_json)
    inputs.update(ROOT/p for p in manifest.observations_csv)
    def hashes(paths):
        return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    scientific=copy.deepcopy(s)
    for key in ('workers','queue_capacity','heartbeat_seconds','n_trials'):scientific.pop(key)
    return dict(format_version=1,algorithm='prefetched-startup-fixed-pair-TPE-constant-liar',settings=scientific,
        jobs=all_jobs,versions=versions,source_sha256=hashes(source),input_sha256=hashes(inputs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,default=ROOT/'config/fsim_abc_tpe_seed123.json')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--workers',type=int,choices=legacy.WORKERS)
    parser.add_argument('--queue-capacity',type=int)
    parser.add_argument('--trials',type=int,choices=(1024,2048))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--design-only',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Tiny real-data run; never use as scientific results')
    args=parser.parse_args()
    s=json.loads(args.settings.read_text())
    for name in ('workers','queue_capacity'):
        if getattr(args,name) is not None:s[name]=getattr(args,name)
    if args.trials is not None:s['n_trials']=args.trials
    if args.smoke:s.update(smoke=True,n_trials=8,n_startup_trials=4,training_checkpoints=[4,6,8])
    validate(s)
    manifest=pd.read_csv(ROOT/s['manifest'])
    all_jobs=[j for plan in s['order'] for j in make_jobs(s,manifest,plan)]
    run_identity=identity(s,all_jobs)
    cases,errors,_,alternates,repairs=core.load_cases(ROOT/s['manifest'])
    if errors:raise ValueError(errors)
    if not s['check_duplicate_policy']:alternates=[]
    print(json.dumps(dict(order=s['order'],workers=s['workers'],queue_capacity=s['queue_capacity'],
        concurrent_studies_per_plan=12,studies_total=len(all_jobs),seed=s['seed'],trials_per_study=s['n_trials'],
        startup_trials=s['n_startup_trials'],adaptive_batch=2,constant_liar=True,
        total_trials=len(all_jobs)*s['n_trials'],plans=s['plans'],
        selection_rule=s['selection_rule'],selection_tolerance_days=s['selection_tolerance_days'],
        selection_uses_test_data=True),indent=2),flush=True)
    if args.design_only:
        print('Design verified; no calibration started.');return
    output=(args.output_dir or ROOT/('result/fsim_abc_tpe_seed123_smoke' if args.smoke else 'result/fsim_abc_tpe_seed123')).resolve()
    resume=args.resume or args.report_only
    with legacy.output_lock(output):
        ipath=output/'run_identity.json'
        metadata_path=output/'run_metadata.json'
        if resume:
            if not ipath.is_file() or json.loads(ipath.read_text())!=run_identity:
                raise ValueError('Resume requires identical scientific settings, code, inputs and versions')
            if not (output/'checkpoints.sqlite').is_file():raise ValueError('Missing checkpoint database')
            metadata=json.loads(metadata_path.read_text())
            if s['n_trials']<metadata['target_trials']:raise ValueError('Cannot reduce the target budget')
        else:
            if any(p.name!='.run.lock' for p in output.iterdir()):raise ValueError('Use empty output directory or --resume')
            core.write_json(ipath,run_identity)
            core.write_json(output/'split_design.json',all_jobs)
            manifest.to_csv(output/'site_year_manifest.csv',index=False)
            repairs.to_csv(output/'weather_duplicate_audit.csv',index=False)
            for name in run_identity['source_sha256']:
                target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/name,target)
            metadata=dict(jobs={},sessions=[],stage_history=[])
        core.write_json(output/'effective_settings.json',s)
        metadata.update(status='running',smoke=bool(s.get('smoke')),target_trials=s['n_trials'],
            selection_rule=s['selection_rule'],selection_tolerance_days=s['selection_tolerance_days'])
        metadata.pop('error',None)
        metadata['sessions'].append(dict(started_utc=legacy.utc_now(),workers=s['workers'],queue_capacity=s['queue_capacity'],target_trials=s['n_trials']))
        core.write_json(metadata_path,metadata)
        signal.signal(signal.SIGINT,legacy.stop_on_signal);signal.signal(signal.SIGTERM,legacy.stop_on_signal)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        store=Store(output/'checkpoints.sqlite');scheduler=None
        try:
            base=core.load_config(ROOT/s['parameter_table'])
            scheduler=Scheduler(s,cases,alternates,base,store,output)
            for plan in s['order']:
                jobs=[j for j in all_jobs if j['plan']==plan]
                metadata['active_plan']=plan
                metadata['stage_history'].append(dict(plan=plan,started_utc=legacy.utc_now(),target_trials=s['n_trials']))
                core.write_json(metadata_path,metadata)
                if not args.report_only:
                    def completed(state):
                        save_training(s,state,output)
                        choice,_=core.choose_training(state.records,s['tie_atol_days'])
                        metadata['jobs'][state.job['id']]=dict(status='complete',n_trials=len(state.records),
                            training_MAE_days=choice['metrics']['training_MAE_days'] if choice else None)
                        core.write_json(metadata_path,metadata)
                        print(f'Completed {state.job["id"]}: {len(state.records)} trials',flush=True)
                    states=[StudyState(s,j,store,scheduler,completed) for j in jobs]
                    scheduler.run(states,plan+'_training')
                report_plan(s,jobs,store,scheduler,output)
                write_summary(s,all_jobs,store,output)
                report_exploratory(s,jobs,store,scheduler,output)
                write_summary(s,all_jobs,store,output,exploratory=True)
            metadata.update(status='complete',active_plan=None,completed_utc=legacy.utc_now())
        except BaseException as exc:
            metadata.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
            raise
        finally:
            if scheduler is not None:
                scheduler.close()
                metadata['last_session_scheduler']=scheduler.stats
            metadata['unique_model_evaluations']=store.evaluation_count()
            core.write_json(metadata_path,metadata);store.close()
    print(output,flush=True)


if __name__=='__main__':
    main()
