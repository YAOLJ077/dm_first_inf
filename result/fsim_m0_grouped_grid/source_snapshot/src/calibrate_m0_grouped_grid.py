#!/usr/bin/env python3
"""Two-parameter grid LOSO/LOYO with a frozen model matching the legacy LOSYO run."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'src/grid_m0_reference'
sys.path.insert(0,str(REFERENCE))
from model.parameter_table import load_config, candidate_rows
import src.analyze_fsim_parameters as analysis
from src.analyze_fsim_parameters import load_cases
from src.calibrate_m0_losyo import (EvaluationCache, make_lattice, sample_for,
    best_and_ties, refine_cells, validate_settings, NAMES)
analysis.ROOT=ROOT

def training_scores(records,train_keys):
    """Explicit key filter prevents held-out records from affecting any objective."""
    if not train_keys or len(set(train_keys))!=len(train_keys):raise ValueError('Unique nonempty training keys required')
    frame=pd.DataFrame(records)
    frame=frame[frame.input_key.isin(train_keys)]
    if frame.empty:raise ValueError('Empty training predictions')
    rows=[]
    for sid,g in frame.groupby('sample_id',sort=False):
        if len(g)!=len(train_keys) or set(g.input_key)!=set(train_keys):raise ValueError('Incomplete/duplicate training evaluations')
        if not g.status.isin(['predicted','no_prediction']).all():raise ValueError('Runtime failure cannot be a calibration loss')
        a=g.status.eq('predicted');e=pd.to_numeric(g.signed_interval_error_days,errors='coerce')
        if not np.isfinite(e[a]).all():raise ValueError('Nonfinite error for prediction')
        first=g.iloc[0]
        rows.append({k:first[k] for k in ['sample_id','grid_x','grid_y']+NAMES}|
                    {'n_training_predictions':int(a.sum()),'feasible':bool(a.all()),
                     'training_MAE_days':float(e.abs().mean()) if a.all() else np.nan,
                     'training_RMSE_days':float(np.sqrt(np.mean(e**2))) if a.all() else np.nan})
    return pd.DataFrame(rows)


def run_grouped(cases,cache,bounds,s,out,groups,progress=True):
    keys=[c['input_key'] for c in cases];extent,initial,initial_cells=make_lattice(bounds,s)
    folds=[];oof=[]
    for holdout, test_keys in tqdm(groups.items(),desc='Grouped folds',unit='fold',disable=not progress):
        foldout=out/'folds'/holdout;foldout.mkdir(parents=True)
        train=[k for k in keys if k not in test_keys]
        points=set(initial);cells=list(initial_cells);stages=[];birth={p:0 for p in points}
        stopping='round_limit';last_best=None;scores=None
        for stage in range(s['refinement_rounds']+1):
            samples=[sample_for(p,extent,bounds) for p in sorted(points)]
            records=cache.evaluate(samples,train,f'{holdout} training {stage}')
            scores=training_scores(records,train)
            scores['first_stage']=[birth[(int(r.grid_x),int(r.grid_y))] for r in scores.itertuples()]
            best,ties=best_and_ties(scores,s['tie_atol_days'])
            scores.to_csv(foldout/f'loss_surface_stage_{stage}.csv',index=False)
            stages.append({'stage':stage,'n_candidates':len(points),'n_feasible':int(scores.feasible.sum()),
                           'best_training_MAE_days':best,'improvement_days':None if best is None or last_best is None else last_best-best,
                           'n_tied_best':len(ties)})
            last_best=best
            if stage==s['refinement_rounds']:break
            proposed,new_cells,count=refine_cells(cells,scores,s);new=proposed-points
            if not new:stopping='no_more_refinable_cells';break
            if len(points|new)>s['max_candidates_per_fold']:
                stopping='candidate_budget';stages[-1]['proposed_candidate_count']=len(points|new);break
            cells=new_cells;points.update(new);birth.update({p:stage+1 for p in new})
        pd.DataFrame(records).to_csv(foldout/'training_predictions.csv',index=False)
        pd.DataFrame(stages).to_csv(foldout/'refinement_history.csv',index=False)
        scores.to_csv(foldout/'loss_surface.csv',index=False)
        best,ties=best_and_ties(scores,s['tie_atol_days'])
        fold={'held_out':holdout,'training_keys':train,'test_keys':test_keys,'stopping_reason':stopping,'n_candidates':len(scores),
              'best_training_MAE_days':best,'n_tied_best':len(ties),'calibration_status':'selected' if best is not None else 'no_feasible_training_combination'}
        if best is None:
            (foldout/'selection.json').write_text(json.dumps(fold,indent=2))
            oof.extend({'input_key':key,'held_out':holdout,'calibration_status':fold['calibration_status'],'status':'calibration_failed','signed_interval_error_days':np.nan} for key in test_keys)
        else:
            chosen=ties.iloc[0]
            ties.to_csv(foldout/'tied_best_parameters.csv',index=False)
            near=[]
            for tol in s['near_optimal_tolerances_days']:
                q=scores[scores.feasible & scores.training_MAE_days.le(best+tol+s['tie_atol_days'])].copy()
                q['tolerance_days']=tol;q['delta_MAE_days']=q.training_MAE_days-best;near.append(q)
            pd.concat(near,ignore_index=True).to_csv(foldout/'near_optimal_parameters.csv',index=False)
            chosen_sample=sample_for((int(chosen.grid_x),int(chosen.grid_y)),extent,bounds)
            fold.update({k:float(chosen[k]) for k in NAMES})
            fold['selected_sample_id']=chosen.sample_id;fold['selected_training_MAE_days']=float(chosen.training_MAE_days)
            fold['near_optimal_touches_boundary']={name:bool(any(
                np.isclose(pd.concat(near)[name],v,atol=1e-10,rtol=0).any() for v in bounds[k])) for k,name in enumerate(NAMES)}
            fold['selected_touches_boundary']={name:bool(any(np.isclose(chosen[name],v,atol=1e-10,rtol=0) for v in bounds[k])) for k,name in enumerate(NAMES)}
            # Freeze selection before requesting any held-out evaluation.
            (foldout/'selection.json').write_text(json.dumps(fold,indent=2))
            tests=cache.evaluate([chosen_sample],test_keys,f'{holdout} held-out')
            records_test=[{**test,'calibration_status':'selected','held_out':holdout} for test in tests]
            pd.DataFrame(records_test).to_csv(foldout/'held_out_predictions.csv',index=False)
            oof.extend(records_test)
        folds.append(fold)
    frame=pd.DataFrame(oof);frame.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame([{k:v for k,v in f.items() if not isinstance(v,(list,dict))} for f in folds]).to_csv(out/'fold_summary.csv',index=False)
    summary=group_summary(frame,s,groups)
    (out/'oof_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return folds,summary



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


def verify_reference(s):
    reference=json.loads((REFERENCE/'snapshot_manifest.json').read_text())
    for name,digest in reference['sha256'].items():
        if hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen model or grid helper changed: {name}')
    parent=ROOT/s['reference_run'];meta=json.loads((parent/'run_metadata.json').read_text())
    if meta['status']!='complete' or meta.get('smoke'):raise ValueError('Requires formal completed reference')
    for name,digest in meta['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Reference input changed: {name}')
    old=meta['settings']
    for name in ('manifest','coarse_levels','refinement_rounds','refinement_factor','refinement_margin_days',
                 'tie_rule','tie_atol_days','objective','require_all_training_predictions',
                 'check_duplicate_policy','max_candidates_per_fold','accuracy_tolerances_days'):
        if s[name]!=old[name]:raise ValueError(f'Grid must match prior LOSYO: {name}')
    if not s['validation_schemes'] or not set(s['validation_schemes'])<= {'site','year'}:
        raise ValueError('Select site and/or year validation')


def write_comparison(out,reference):
    old=pd.read_csv(reference/'oof_predictions.csv')
    new=pd.read_csv(out/'oof_predictions.csv')
    columns=['input_key','status','signed_interval_error_days','VPD_w','I_req']
    for name in ('VPD_w','I_req'):
        if name not in new:new[name]=np.nan
    comparison=old[columns].merge(new,on='input_key',validate='one_to_one',suffixes=('_LOSYO','_grouped'))
    comparison['delta_absolute_error_days']=comparison.signed_interval_error_days_grouped.abs()-comparison.signed_interval_error_days_LOSYO.abs()
    comparison.to_csv(out/'comparison_with_LOSYO.csv',index=False)
    reports=[]
    for name,g in new.groupby('held_out'):
        available=g.status.eq('predicted');e=pd.to_numeric(g.signed_interval_error_days,errors='coerce')
        reports.append(dict(held_out=name,n_test_site_years=len(g),n_predictions=int(available.sum()),
            MAE_days=float(e.abs().mean()) if available.all() else None,
            RMSE_days=float(np.sqrt(np.mean(e**2))) if available.all() else None))
    pd.DataFrame(reports).to_csv(out/'test_group_metrics.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,default=ROOT/'config/fsim_m0_grouped_grid.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_m0_grouped_grid')
    p.add_argument('--scheme',choices=['both','site','year'],default='both')
    p.add_argument('--workers',type=int,choices=[2,4])
    p.add_argument('--design-only',action='store_true',help='Validate and print folds, without model evaluation or output creation')
    p.add_argument('--no-progress',action='store_true')
    args=p.parse_args();s=json.loads(args.settings.read_text())
    if args.workers:s['workers']=args.workers
    if args.scheme!='both':s['validation_schemes']=[args.scheme]
    validate_settings(s);verify_reference(s)
    base=load_config(ROOT/s['parameter_table']);base['dispersal']['use_wind_gate']=False
    if base['mp_tmp']['rh_wet_night']!=80 or base['fixed']['min_moist_hours']!=3:
        raise ValueError('Requires RH=80 and MP=3')
    candidates=candidate_rows('M0',ROOT/s['parameter_table'])
    if [r['parameter'] for r in candidates]!=NAMES:raise ValueError('Only VPD_w and I_req may vary')
    bounds=[[float(r['lower']),float(r['upper'])] for r in candidates]
    if bounds!=[[1.,8.],[40.,400.]]:raise ValueError('Bounds must match legacy two-parameter grid')
    manifest=pd.read_csv(ROOT/s['manifest'])
    cases,errors,_,alternates,repairs=load_cases(ROOT/s['manifest'])
    if errors:raise ValueError(errors)
    if not s['check_duplicate_policy']:alternates=[]
    designs={scheme:make_groups(manifest,scheme) for scheme in s['validation_schemes']}
    design=[];keys=manifest.input_key.tolist()
    for scheme,groups in designs.items():
        for group,test in groups.items():
            train=[k for k in keys if k not in test]
            design.append(dict(scheme=scheme,held_out=group,training_keys=train,test_keys=test))
            print(f'{scheme} {group}: train={train}; test={test}',flush=True)
    extent,points,_=make_lattice(bounds,s)
    print(f'Grid: bounds={bounds}; coarse={len(points)}; final spacing='+str([(hi-lo)/n for (lo,hi),n in zip(bounds,extent)]),flush=True)
    if args.design_only:
        print('Design verified; no calibration started.');return
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a new empty output directory; no resume support')
    out.mkdir(parents=True,exist_ok=True)
    (out/'effective_settings.json').write_text(json.dumps(s,indent=2))
    (out/'split_design.json').write_text(json.dumps(design,indent=2))
    manifest.to_csv(out/'site_year_manifest.csv',index=False);repairs.to_csv(out/'weather_duplicate_audit.csv',index=False)
    (out/'effective_runtime_config.json').write_text(json.dumps(base,indent=2))
    sources=[Path(__file__),args.settings.resolve(),REFERENCE/'snapshot_manifest.json']
    sources += [REFERENCE/name for name in json.loads((REFERENCE/'snapshot_manifest.json').read_text())['sha256']]
    metadata=dict(status='running',model='M0',settings=s,started_utc=datetime.now(timezone.utc).isoformat(),source_sha256={},input_sha256={},schemes={})
    metadata['versions']={'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['numpy','pandas','tqdm']}}
    for source in sources:
        relative=source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('external')/source.name
        dest=out/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        metadata['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    for source in [ROOT/s['manifest']]+[ROOT/name for name in manifest.weather_json]:
        metadata['input_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    def save(): (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    save()
    try:
        for scheme,groups in designs.items():
            dest=out/scheme;dest.mkdir()
            cache=EvaluationCache(cases,alternates,base,s,dest,not args.no_progress)
            try:
                folds,summary=run_grouped(cases,cache,bounds,s,dest,groups,not args.no_progress)
                write_comparison(dest,ROOT/s['reference_run'])
                metadata['schemes'][scheme]=dict(status='complete',summary=summary,n_model_evaluations=cache.count,
                    n_duplicate_evaluations=sum(cid>=len(cases) for cid,_ in cache.cache),duplicate_policy_differences=0)
            finally:
                cache.close();save()
        metadata.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
    finally:save()
    print(out,flush=True)


if __name__=='__main__':main()
