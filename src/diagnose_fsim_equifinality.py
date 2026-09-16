#!/usr/bin/env python3
"""Retrospective held-out stability audit of saved candidates, without fitting.

Inspect saved sampled points within one day of the fold minimum and successive
training-selected incumbents. Never run the model or select using test outcomes.
Original runs are opened immutable/read-only; absent predictions remain unknown.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import calibrate_fsim_joblib_local as old

ROOT=Path(__file__).resolve().parents[1]
core=old.core
TOLS=(0,.05,.5,1)


def key(dim,params,case):
    names=core.NAMES+(('u_tr',) if dim=='3p' else ())
    return old.json_text([dim,case,[float(params[n]) for n in names]])


def signature(rows):
    return old.json_text([[r['input_key'],r['status'],r.get('predicted_start'),r.get('predicted_end')] for r in rows])


def load(output):
    jobs=[]; cache={}; audits=[]
    for directory,dim in [('fsim_unified_joblib_local','2p'),('fsim_m0_3p_joblib','3p')]:
        root=ROOT/'result'/directory
        identity=json.loads((root/'run_identity.json').read_text())
        settings=json.loads((root/'effective_settings.json').read_text())
        old.verify_engine(settings)
        checked={**identity['input_sha256'],**{p:h for p,h in identity['source_sha256'].items()
                   if p.startswith('src/tpe_m0_reference/') or p=='src/calibrate_m0_grouped_tpe.py'}}
        for p,h in checked.items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
        assert json.loads((root/'run_metadata.json').read_text())['status']=='complete'
        db=sqlite3.connect(f'file:{root}/checkpoints.sqlite?mode=ro&immutable=1',uri=True)
        for payload, in db.execute('SELECT payload FROM evaluations'):
            r=json.loads(payload)['record']
            if r.get('weather_policy')=='keep_first':continue
            if dim=='2p' and (r['RH_sp']!=80 or r['MP_sp']!=3):continue
            cache[key(dim,r,r['input_key'])]=r
        split=json.loads((root/'split_design.json').read_text())
        for job in split:
            if not job['arm'].startswith(dim):continue
            records=[json.loads(p) for p, in db.execute('SELECT payload FROM trials WHERE job=? ORDER BY number',(job['id'],))]
            jobs.append(dict(job,dim=dim,source=directory,records=records,partial_run=False))
        db.close()
        audits.append(dict(source=directory,verified_files=checked))
    # The old LOSYO TPE has only three completed folds. Keep explicitly partial.
    root=ROOT/'result/fsim_m0_losyo_tpe_2p_n1024'
    ident=json.loads((root/'run_identity.json').read_text())
    for p,h in ident['input_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for folder in sorted((root/'folds').iterdir()):
        if not (folder/'selection.json').exists():continue
        rows=list(csv.DictReader((folder/'trials.csv').open()))
        if len(rows)!=1024:continue
        selection=json.loads((folder/'selection.json').read_text())
        saved_holdout=json.loads((folder/'held_out_checkpoint.json').read_text())
        for r in saved_holdout['predictions']:
            cache[key('2p',saved_holdout['params'],r['input_key'])]=r
        records=[]
        for r in rows:
            records.append(dict(number=int(r['trial']),seed=int(r['seed']),
                params={n:float(r[n]) for n in core.NAMES},training_keys=selection['training_keys'],
                metrics=dict(feasible=r['feasible']=='True',training_MAE_days=float(r['training_MAE_days']) if r['training_MAE_days'] else None)))
        jobs.append(dict(id='losyo/2p_tpe/'+folder.name,dim='2p',scheme='losyo',arm='2p_tpe',
            held_out=folder.name,seed=records[0]['seed'],train=selection['training_keys'],test=[folder.name],
            source=str(root.relative_to(ROOT/'result')),records=records,partial_run=True,legacy_folder=str(folder)))
    core.write_json(output/'source_audit.json',dict(sources=audits,
        scope='Completed site/year runs plus three completed folds of the incomplete legacy 2p LOSYO run',
        original_runs_read_only=True,near_tolerances_days=TOLS))
    return jobs,cache,settings


def plan(jobs,cache):
    needed={}
    for j in jobs:
        records=j['records'];best,ties=core.choose_training(records,1e-9)
        j['best']=best;j['minimum']=best['metrics']['training_MAE_days']
        incumbents=[]; last=None
        for i in range(len(records)):
            current,_=core.choose_training(records[:i+1],1e-9)
            if current and current['number']!=last:
                incumbents.append(current['number']);last=current['number']
        j['incumbents']=incumbents
        selected={r['number']:r for r in records if r['metrics']['feasible'] and r['metrics']['training_MAE_days']<=j['minimum']+1+1e-9}
        selected.update({r['number']:r for r in records if r['number'] in incumbents})
        # Recover exact JSON floats and saved training records for partial LOSYO.
        if j.get('legacy_folder'):
            for n in selected:
                path=Path(j['legacy_folder'])/f'seed_{j["seed"]}'/f'trial_{n:04d}.json'
                r=json.loads(path.read_text());selected[n]=r
                for row in r['predictions']:cache[key('2p',r['params'],row['input_key'])]=row
        j['candidates']=list(selected.values())
        for r in j['candidates']:
            for case in j['train']+j['test']:
                k=key(j['dim'],r['params'],case)
                if k not in cache:needed[k]=(j['dim'],r['params'],case)
    return needed


def report(jobs,cache,output):
    points=[]; predictions=[]; summary=[]; jumps=[]
    for j in jobs:
        tag={n:j[n] for n in ('source','dim','arm','scheme','held_out','partial_run')}
        metrics={}
        for r in j['candidates']:
            p=r['params'];train=[cache[key(j['dim'],p,k)] for k in j['train']]
            test=[cache.get(key(j['dim'],p,k),dict(input_key=k,status='not_saved')) for k in j['test']]
            computed=core.training_metrics(train,j['train'])
            assert computed['feasible'] and np.isclose(computed['training_MAE_days'],r['metrics']['training_MAE_days'],atol=1e-9,rtol=0)
            valid=[abs(t['signed_interval_error_days']) for t in test if t['status']=='predicted']
            row=dict(**tag,trial=r['number'],**p,train_MAE=r['metrics']['training_MAE_days'],
                delta_train_MAE=r['metrics']['training_MAE_days']-j['minimum'],selected=r['number']==j['best']['number'],
                n_test=len(test),n_evaluated=sum(t['status']!='not_saved' for t in test),n_predicted=len(valid),test_MAE=sum(valid)/len(valid) if len(valid)==len(test) else None,
                conditional_test_MAE=sum(valid)/len(valid) if valid else None,train_signature=signature(train),
                test_signature=signature(test),parameter_key=key(j['dim'],p,''))
            metrics[r['number']]=row;points.append(row)
            predictions.extend(dict(**tag,trial=r['number'],train_MAE=row['train_MAE'],delta_train_MAE=row['delta_train_MAE'],**t) for t in test)
        for tol in TOLS:
            candidates=[r for r in metrics.values() if r['delta_train_MAE']<=tol+1e-9]
            candidates=list({r['parameter_key']:r for r in candidates}.values())
            for case in j['test']:
                records=[cache.get(key(j['dim'],r,case),dict(input_key=case,status='not_saved')) for r in candidates]
                good=[r for r in records if r['status']=='predicted']
                errors=[r['signed_interval_error_days'] for r in good]
                starts=[pd.Timestamp(r['predicted_start']) for r in good]
                summary.append(dict(**tag,tolerance_days=tol,input_key=case,n_unique_parameters=len(candidates),
                    n_training_patterns=len({r['train_signature'] for r in candidates}),
                    n_predicted=len(good),n_failed=sum(r['status']=='no_prediction' for r in records),
                    n_not_saved=sum(r['status']=='not_saved' for r in records),
                    n_distinct_windows=len({(r.get('predicted_start'),r.get('predicted_end')) for r in good}),
                    error_min=min(errors) if errors else None,error_max=max(errors) if errors else None,
                    error_span=max(errors)-min(errors) if errors else None,
                    start_span_days=(max(starts)-min(starts)).total_seconds()/86400 if starts else None))
        for before,after in zip(j['incumbents'],j['incumbents'][1:]):
            a,b=metrics[before],metrics[after]
            for case in j['test']:
                x,y=[cache.get(key(j['dim'],r,case),dict(input_key=case,status='not_saved')) for r in (a,b)]
                ex,ey=x.get('signed_interval_error_days'),y.get('signed_interval_error_days')
                jumps.append(dict(**tag,before_trial=before,after_trial=after,
                    train_before=a['train_MAE'],train_after=b['train_MAE'],train_improvement=a['train_MAE']-b['train_MAE'],
                    input_key=case,before_status=x['status'],after_status=y['status'],
                    error_before=ex,error_after=ey,error_jump=ey-ex if ex is not None and ey is not None else None,
                    test_MAE_before=a['test_MAE'],test_MAE_after=b['test_MAE']))
    for name,rows in [('candidate_metrics',points),('candidate_test_predictions',predictions),('stability_summary',summary),('incumbent_changes',jumps)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    print(pd.DataFrame(summary).query('tolerance_days == 0 and arm.str.endswith("tpe")')[['dim','scheme','held_out','input_key','n_unique_parameters','n_failed','error_span']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_equifinality_saved_20260915')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    jobs,cache,s=load(args.output_dir)
    needed=plan(jobs,cache)
    print(f'Folds: {len(jobs)}; candidate trial references: {sum(len(j["candidates"]) for j in jobs)}; missing model evaluations: {len(needed)}',flush=True)
    print('Saved results only: missing predictions will NOT be evaluated.',flush=True)
    report(jobs,cache,args.output_dir)
    print(args.output_dir,flush=True)


if __name__=='__main__':main()
