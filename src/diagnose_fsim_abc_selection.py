#!/usr/bin/env python3
"""Audit saved same-fold regions and prediction modes; never execute FSIM."""
from pathlib import Path
import hashlib,json,sqlite3
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'result/fsim_abc_tpe_seed123'
O=ROOT/'result/fsim_abc_selection_diagnosis_1024'
NAMES=('VPD_w','I_req','RH_sp','MP_sp','u_tr')

def dumps(x): return json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':'))

def main():
    metadata=json.loads((R/'run_metadata.json').read_text())
    if metadata['status']!='complete': raise ValueError('Requires completed run for immutable read')
    db=sqlite3.connect((R/'checkpoints.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
    manifest=pd.read_csv(R/'site_year_manifest.csv'); keys=sorted(manifest.input_key)
    summaries=[]; allc=[]; cases=[]; alltr=[]; source_hashes={}
    for arm in 'ABC':
      for path in sorted((R/arm).glob('*/seed_123/folds/*/trials.csv')):
        job=str(path.parent.relative_to(R));scheme=path.parts[-5];held=path.parent.name
        records=[json.loads(x[0]) for x in db.execute('select payload from trials where job=? order by number',(job,))]
        if len(records)!=1024: raise ValueError((job,len(records)))
        selected={s:json.loads((path.parent/'budget_1024'/p/'selection.json').read_text())['selected_trial'] for s,p in [('minimum',''),('tolerance','exploratory')]}
        valid=[r for r in records if r['metrics']['feasible']]
        best=min(r['metrics']['training_MAE_days'] for r in valid)
        train=sorted(records[0]['training_keys']);test=sorted(set(keys)-set(train))
        source_hashes[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for rec in records:
            alltr.append(dict(arm=arm,scheme=scheme,held=held,trial=rec['number'],**rec['params'],**rec['metrics']))
        cand=[]
        for rec in valid:
          m=rec['metrics'];par=rec['params']
          if m['training_MAE_days']>best+1+1e-9:continue
          day_sigs={};exact_sigs={};e={};statuses={}
          for key in keys:
            cache_key=hashlib.sha256(dumps([key,'primary',arm=='C',[float(par[n]) for n in NAMES]]).encode()).hexdigest()
            saved=db.execute('select payload from evaluations where key=?',(cache_key,)).fetchone()
            if saved is None: raise ValueError(('Missing cached case',job,rec['number'],key))
            p=json.loads(saved[0])['record'];status=p['status'];statuses[key]=status
            if status=='predicted':
                st,en=pd.Timestamp(p['predicted_start']),pd.Timestamp(p['predicted_end'])
                lo,hi=pd.Timestamp(p['observed_L']),pd.Timestamp(p['observed_U'])
                err=(en-lo).total_seconds()/86400 if en<lo else (st-hi).total_seconds()/86400 if st>hi else 0.
                if not np.isclose(err,p['signed_interval_error_days'],atol=1e-9,rtol=0):raise ValueError('Error mismatch')
                exact_sigs[key]=f'{st.isoformat()}/{en.isoformat()}'
                day_sigs[key]=f'{st.date()}/{en.date()}'
                e[key]=err
            else: day_sigs[key]=exact_sigs[key]=status;e[key]=None
            cases.append(dict(arm=arm,scheme=scheme,held=held,trial=rec['number'],input_key=key,role='train' if key in train else 'test',status=status,predicted_start=p.get('predicted_start'),predicted_end=p.get('predicted_end'),error=e[key]))
          if any(e[k] is None for k in train):raise ValueError('Incomplete training candidate')
          mae=np.mean([abs(e[k]) for k in train]);rmse=np.sqrt(np.mean([e[k]**2 for k in train]))
          if not np.isclose(mae,m['training_MAE_days'],atol=1e-9,rtol=0):raise ValueError('Training MAE mismatch')
          if not np.isclose(rmse,m['training_RMSE_days'],atol=1e-9,rtol=0):raise ValueError('Training RMSE mismatch')
          train_sig=';'.join(k+':'+day_sigs[k] for k in train)
          test_sig=';'.join(k+':'+day_sigs[k] for k in test)
          exact_train=';'.join(k+':'+exact_sigs[k] for k in train)
          good=[e[k] for k in test if e[k] is not None]
          row=dict(arm=arm,scheme=scheme,held=held,trial=rec['number'],**par,
                   training_MAE_days=mae,training_RMSE_days=rmse,training_bias=np.mean([e[k] for k in train]),training_max_error=max(abs(e[k]) for k in train),
                   exact_tie=abs(mae-best)<=1e-9,selected_minimum=rec['number']==selected['minimum'],selected_tolerance=rec['number']==selected['tolerance'],
                   n_test=len(test),n_predicted=len(good),test_MAE_days=np.mean(np.abs(good)) if len(good)==len(test) else np.nan,
                   train_mode=hashlib.sha256(train_sig.encode()).hexdigest()[:12],exact_train_mode=hashlib.sha256(exact_train.encode()).hexdigest()[:12],
                   test_mode=hashlib.sha256(test_sig.encode()).hexdigest()[:12],train_pattern=train_sig,test_pattern=test_sig)
          cand.append(row);allc.append(row)
        df=pd.DataFrame(cand);ties=df[df.exact_tie];complete=ties[ties.n_predicted==ties.n_test]
        split_modes=df.groupby('train_mode').test_mode.nunique()
        split_exact=df.groupby('exact_train_mode').test_mode.nunique()
        summaries.append(dict(arm=arm,scheme=scheme,held=held,best=best,near_n=len(df),tie_n=len(ties),train_modes=df.train_mode.nunique(),test_modes=df.test_mode.nunique(),
                              ambiguous_train_day_modes=int((split_modes>1).sum()),ambiguous_train_exact_modes=int((split_exact>1).sum()),
                              tie_test_min=complete.test_MAE_days.min(),tie_test_max=complete.test_MAE_days.max(),tie_test_failures=int((ties.n_predicted<ties.n_test).sum()),
                              near_test_failures=int((df.n_predicted<df.n_test).sum())))
    O.mkdir(exist_ok=True)
    candidates=pd.DataFrame(allc);trials=pd.DataFrame(alltr);summary=pd.DataFrame(summaries)
    for name,df in [('candidates',candidates),('candidate_case_predictions',pd.DataFrame(cases)),('fold_diagnostics',summary)]:df.to_csv(O/(name+'.csv'),index=False)
    mode_rows=[]
    for (arm,scheme,held,mode),g in candidates.groupby(['arm','scheme','held','train_mode']):
      row=dict(arm=arm,scheme=scheme,held=held,train_mode=mode,n=len(g),train_min=g.training_MAE_days.min(),train_max=g.training_MAE_days.max(),test_modes=g.test_mode.nunique(),test_min=g.test_MAE_days.min(),test_max=g.test_MAE_days.max(),failures=int((g.n_predicted<g.n_test).sum()),selected_minimum=bool(g.selected_minimum.any()),selected_tolerance=bool(g.selected_tolerance.any()),train_pattern=g.iloc[0].train_pattern)
      for p in ['VPD_w','I_req','u_tr']:row[p+'_min']=g[p].min();row[p+'_max']=g[p].max()
      mode_rows.append(row)
    pd.DataFrame(mode_rows).to_csv(O/'training_prediction_regions.csv',index=False)
    known=[]
    for (scheme,held),g in trials.groupby(['scheme','held']):
      f=g[g.feasible.eq(True)]
      best_ab=f[f.arm.isin(['A','B'])].training_MAE_days.min();best_abc=f.training_MAE_days.min()
      row=dict(scheme=scheme,held=held,best_AB=best_ab,best_ABC=best_abc)
      for a in 'ABC':row[a+'_best']=f[f.arm==a].training_MAE_days.min()
      row['B_gap_known_AB']=row['B_best']-best_ab
      row['C_gap_known_ABC']=row['C_best']-best_abc
      known.append(row)
    pd.DataFrame(known).to_csv(O/'same_fold_known_minima.csv',index=False)
    (O/'audit.json').write_text(json.dumps(dict(models_executed=0,cache_read_only=True,training_mode_definition='ordered training case predicted start/end calendar dates; no held-out outcomes used to define training mode',mode_counts_not_region_volumes=True,n_cached_case_records_checked=len(cases),source_trial_sha256=source_hashes),indent=2))
    print(summary.round(3).to_string(index=False));print('\nKNOWN MINIMA');print(pd.DataFrame(known).round(3).to_string(index=False))
if __name__=='__main__':main()
