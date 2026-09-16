#!/usr/bin/env python3
"""Diagnose and select robust parameters from scheme-B TPE results.

The selector is deliberately test-blind for the reported robust choice.  Test
metrics are retained only as an explicitly labelled oracle diagnostic.
It handles LOSYO, leave-site-out and leave-year-out result trees.
"""
from __future__ import annotations
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def one_fold(args):
    scheme, fold, tolerances, dv, di = args
    trials = pd.read_csv(fold/'trials.csv')
    trials = trials[trials.feasible == True].copy()
    if trials.empty: return []
    best = float(trials.training_MAE_days.min())
    out=[]
    # Existing exploratory candidate metrics contain held-out predictions for
    # the +1-day set. They are used only to quantify oracle leakage.
    cm = fold/'budget_1024'/'exploratory'/'candidate_metrics.csv'
    test = pd.read_csv(cm) if cm.exists() else pd.DataFrame()
    for tol in tolerances:
        x=trials[trials.training_MAE_days <= best+tol+1e-9].copy()
        x['tolerance_days']=tol; x['fold']=fold.name; x['scheme']=scheme; x['best_training_MAE_days']=best
        # A parameter-neighbourhood stability proxy: local trial density and
        # distance to the nearest other feasible trial. This is test-blind.
        p=x[['VPD_w','I_req']].to_numpy(float)
        allp=trials[['VPD_w','I_req']].to_numpy(float)
        if len(allp)>1:
            scale=np.array([dv,di],float)
            z=(p[:,None,:]-allp[None,:,:])/scale
            dist=np.sqrt((z*z).sum(2)); dist[dist==0]=np.inf
            x['nearest_parameter_distance']=dist.min(1)
            x['local_density_1']= (dist<=1).sum(1)
        else:
            x['nearest_parameter_distance']=np.nan; x['local_density_1']=0
        if not test.empty and tol==1.0:
            cols=['trial','test_MAE_days','test_prediction_complete','n_test_predicted']
            keep=[c for c in cols if c in test.columns]
            x=x.merge(test[keep],on='trial',how='left')
        else:
            x['test_MAE_days']=np.nan; x['test_prediction_complete']=np.nan
        out.extend(x.to_dict('records'))
    return out

def select(df):
    rows=[]
    for (scheme,tol),g in df.groupby(['scheme','tolerance_days']):
        # robust score is based only on training data and local density.
        g=g.copy(); g['robust_score']=g.training_MAE_days + 0.05*g.nearest_parameter_distance - 0.002*g.local_density_1
        # candidate medoid: closest to the weighted centre of the dense near-optimal cloud
        w=1/(1+g.training_MAE_days-g.best_training_MAE_days)
        centre=np.average(g[['VPD_w','I_req']],axis=0,weights=w)
        g['centre_distance']=np.sqrt((((g[['VPD_w','I_req']].to_numpy()-centre)/[.1,10])**2).sum(1))
        g=g.sort_values(['robust_score','centre_distance','trial'])
        r=g.iloc[0].to_dict(); r['selection']='robust_training_blind'; rows.append(r)
        c=g.dropna(subset=['test_MAE_days'])
        if not c.empty:
            q=c.loc[c.test_MAE_days.idxmin()].to_dict(); q['selection']='oracle_test_within_tolerance'; rows.append(q)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('result/fsim_abc_tpe_seed123/B'))
    ap.add_argument('--out',type=Path,default=Path('result/fsim_abc_tpe_seed123/B/robust_selection'))
    ap.add_argument('--workers',type=int,default=64)
    ap.add_argument('--tolerances',nargs='+',type=float,default=[.5,1.,2.])
    ap.add_argument('--vpd-scale',type=float,default=.1); ap.add_argument('--ireq-scale',type=float,default=10.)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for scheme in ('losyo','site','year'):
        for fold in sorted((a.root/scheme/'seed_123'/'folds').glob('*')):
            if (fold/'trials.csv').exists(): jobs.append((scheme,fold,a.tolerances,a.vpd_scale,a.ireq_scale))
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rec in tqdm(ex.map(one_fold,jobs),total=len(jobs),desc='scheme-B folds'):
            records.extend(rec)
    df=pd.DataFrame(records)
    if df.empty: raise SystemExit('No scheme-B trials found')
    df.to_csv(a.out/'all_near_optimal_candidates.csv',index=False)
    sel=select(df); sel.to_csv(a.out/'selection_summary.csv',index=False)
    for scheme in ('losyo','site','year'):
        z=sel[sel.scheme==scheme]; z.to_csv(a.out/f'{scheme}_selection.csv',index=False)
    json.dump({'root':str(a.root),'workers':a.workers,'tolerances':a.tolerances,'test_metrics_role':'oracle diagnostic only','selection':'training blind robust score'},open(a.out/'run_metadata.json','w'),indent=2)
    print(sel[['scheme','fold','tolerance_days','selection','VPD_w','I_req','training_MAE_days','test_MAE_days']].to_string(index=False))
if __name__=='__main__': main()
