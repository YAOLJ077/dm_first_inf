#!/usr/bin/env python3
"""Discrete RH_sp expansion of M0 LOSYO; independent training-only RH selection."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.parameter_table import load_config
from src.analyze_fsim_parameters import load_cases
from src.calibrate_m0_losyo import EvaluationCache,run_losyo,summarize_oof,write_report,validate_settings


def validate_trial(s):
    values=s['RH_sp_values']
    if not isinstance(values,list) or values!=sorted(set(values)) or not values:
        raise ValueError('RH candidates must be unique and ordered')
    if any(type(v) not in [int,float] or not np.isfinite(v) or not 0<v<=100 for v in values):
        raise ValueError('Invalid RH candidates')
    if s['reference_RH_sp']!=80 or 80 not in values:raise ValueError('Retain the RH=80 baseline')
    if s['workers'] not in [2,4]:raise ValueError('Workers must be 2 or 4')
    if s['tie_rule']!='closest_RH_to_80_then_lowest_RH_then_lowest_VPD_w_then_lowest_I_req':
        raise ValueError('Unsupported tie rule')


def choose_rh_from_training(surfaces,settings,reference=80):
    """Pure selector: accepts training loss surfaces, never test predictions."""
    tables=[]
    for rh,frame in surfaces.items():
        g=frame.copy();g['RH_sp']=rh;tables.append(g)
    all_scores=pd.concat(tables,ignore_index=True)
    eligible=all_scores[all_scores.feasible].copy()
    if eligible.empty:return None,eligible,all_scores
    best=eligible.training_MAE_days.min()
    ties=eligible[eligible.training_MAE_days.le(best+settings['tie_atol_days'])].copy()
    ties['distance_to_reference_RH']=(ties.RH_sp-reference).abs()
    ties=ties.sort_values(['distance_to_reference_RH','RH_sp','VPD_w','I_req']).reset_index(drop=True)
    return ties.iloc[0],ties,all_scores


def check_baseline(path):
    m=json.loads((path/'run_metadata.json').read_text())
    if m['status']!='complete' or m.get('model')!='M0' or m.get('smoke'):
        raise ValueError('Requires completed non-smoke M0 baseline')
    if m.get('duplicate_policy_differences')!=0:raise ValueError('Unverified baseline weather policies')
    for name,h in {**m['source_sha256'],**m['input_sha256']}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError(f'Baseline source/input changed: {name}')
    cfg=load_config(path/'parameter_table.csv')
    if cfg['mp_tmp']['rh_wet_night']!=80:raise ValueError('Baseline RH must equal 80')
    return m,cfg


def combine_runs(runs,settings,trial,out):
    """Freeze final training-only choices, then retrieve the corresponding held-out results."""
    baseline=runs[80];manifest=pd.read_csv(baseline/'site_year_manifest.csv');keys=manifest.input_key.tolist()
    folds=[];predictions=[];profiles=[]
    for key in keys:
        foldout=out/'folds'/key;foldout.mkdir(parents=True,exist_ok=True)
        surfaces={rh:pd.read_csv(path/'folds'/key/'loss_surface.csv') for rh,path in runs.items()}
        choice,ties,scores=choose_rh_from_training(surfaces,settings,trial['reference_RH_sp'])
        scores.to_csv(foldout/'training_loss_surface.csv',index=False)
        for rh,g in scores.groupby('RH_sp'):
            valid=g[g.feasible]
            profiles.append(dict(held_out=key,RH_sp=rh,n_candidates=len(g),n_feasible=len(valid),
                                 best_training_MAE_days=valid.training_MAE_days.min()))
        if choice is None:
            selection={'held_out':key,'calibration_status':'no_feasible_training_combination'}
            (foldout/'selection.json').write_text(json.dumps(selection,indent=2))
            predictions.append(dict(input_key=key,status='calibration_failed',calibration_status=selection['calibration_status'],signed_interval_error_days=np.nan))
        else:
            selection={'held_out':key,'training_keys':[k for k in keys if k!=key],'calibration_status':'selected',
                       'RH_sp':float(choice.RH_sp),'VPD_w':float(choice.VPD_w),'I_req':float(choice.I_req),
                       'selected_sample_id':choice.sample_id,'selected_training_MAE_days':float(choice.training_MAE_days),
                       'n_tied_best':len(ties),'tie_rule':trial['tie_rule'],'n_candidates':len(scores)}
            # This write precedes reading any candidate's OOF result for this fold.
            (foldout/'selection.json').write_text(json.dumps(selection,indent=2))
            ties.to_csv(foldout/'tied_best_parameters.csv',index=False)
            near=[]
            for tol in settings['near_optimal_tolerances_days']:
                q=scores[scores.feasible & scores.training_MAE_days.le(choice.training_MAE_days+tol+settings['tie_atol_days'])].copy()
                q['tolerance_days']=tol;q['delta_MAE_days']=q.training_MAE_days-choice.training_MAE_days;near.append(q)
            pd.concat(near,ignore_index=True).to_csv(foldout/'near_optimal_parameters.csv',index=False)
            source=runs[choice.RH_sp]
            prediction=pd.read_csv(source/'folds'/key/'held_out_prediction.csv').iloc[0].to_dict()
            if prediction['sample_id']!=choice.sample_id:raise ValueError('Slice and joint tie rules disagree')
            for name in ['VPD_w','I_req']:
                if prediction[name]!=selection[name]:raise ValueError('Selected coordinates differ')
            prediction.update(RH_sp=selection['RH_sp']);predictions.append(prediction)
            pd.DataFrame([prediction]).to_csv(foldout/'held_out_prediction.csv',index=False)
        folds.append(selection)
    oof=pd.DataFrame(predictions);oof.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(folds).drop(columns=['training_keys'],errors='ignore').to_csv(out/'fold_summary.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'RH_training_profiles.csv',index=False)
    summary=summarize_oof(oof,settings);(out/'oof_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    baseline_oof=pd.read_csv(baseline/'oof_predictions.csv')
    comparison=baseline_oof[['input_key','VPD_w','I_req','signed_interval_error_days','status']].merge(
        oof[['input_key','RH_sp','VPD_w','I_req','signed_interval_error_days','status']],on='input_key',validate='one_to_one',suffixes=('_baseline','_RH_trial'))
    base_folds=pd.read_csv(baseline/'fold_summary.csv').set_index('held_out')
    for name in ['baseline_training_MAE_days','RH_training_MAE_days']:comparison[name]=np.nan
    for f in folds:
        idx=comparison.input_key.eq(f['held_out'])
        comparison.loc[idx,'baseline_training_MAE_days']=base_folds.loc[f['held_out'],'best_training_MAE_days']
        comparison.loc[idx,'RH_training_MAE_days']=f.get('selected_training_MAE_days',np.nan)
    comparison.to_csv(out/'baseline_comparison.csv',index=False)
    base_summary=json.loads((baseline/'oof_summary.json').read_text())
    pd.DataFrame([{'metric':k,'baseline':base_summary[k],'RH_trial':v} for k,v in summary.items()]).to_csv(out/'oof_comparison.csv',index=False)
    lines=['# M0 新增 RH_sp 的 LOSYO 试验','',
           'RH候选：'+str(trial['RH_sp_values'])+'%。这些是离散探索值，不是已验证的连续生物学范围。',
           '每个RH切片分别按相同范围和三轮二维加密校准VPD_w、I_req，再依据五个训练站点的MAE选取RH。MP_sp和其他参数不变。',
           'RH=80复用已完成基线。其他RH通过每个独立进程池的运行配置传入模型，缓存不跨RH切片复用；每个切片的 effective_runtime_config.json 记录真实配置。',
           '三参数选参仅接收训练损失面。RH并列时优先离80最近者，再按RH、VPD_w、I_req升序；冻结选择后才读取相应留出结果。',
           '切片结果中计算的其他留出预测仅作审计，不参与选参。对照差异仍是同一组六站点的探索结果。','',
           '| 留出站点 | 选定RH | VPD_w | I_req | 原训练MAE | 新训练MAE | 原留出误差 | 新留出误差 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison.itertuples():lines.append(f'| {r.input_key} | {r.RH_sp:g} | {r.VPD_w_RH_trial:g} | {r.I_req_RH_trial:g} | {r.baseline_training_MAE_days:.3f} | {r.RH_training_MAE_days:.3f} | {r.signed_interval_error_days_baseline:.3f} | {r.signed_interval_error_days_RH_trial:.3f} |')
    lines+=['','负误差为预测偏早，正误差为偏晚；均为窗口间的边界距离。','',
            '```json',json.dumps(summary,indent=2),'```','',
            '若选定RH触及75或90，记录为候选端点，不自动扩界。改善训练误差不保证改善留出误差。',
            'RH控制湿润小时的判定，降雨/叶湿替代条件保留；它不是直接产孢量倍率，也没有引入新的感染剂量截断。',
            '结果文件：RH_training_profiles.csv（每折各RH训练最低损失）、fold_summary.csv、oof_predictions.csv、baseline_comparison.csv、oof_comparison.csv；folds下保存三参数损失面、并列和近优集合。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,default=ROOT/'config/fsim_m0_losyo_rh_trial.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_m0_losyo_RH_trial')
    p.add_argument('--workers',type=int,choices=[2,4])
    p.add_argument('--no-progress',action='store_true')
    args=p.parse_args();trial=json.loads(args.settings.read_text())
    if args.workers:trial['workers']=args.workers
    validate_trial(trial);baseline=(ROOT/trial['baseline_run']).resolve();parent,base=check_baseline(baseline)
    settings=copy.deepcopy(parent['settings']);settings['workers']=trial['workers'];validate_settings(settings)
    cases,errors,context,alternates,repairs=load_cases(baseline/'site_year_manifest.csv')
    if errors:raise ValueError(errors)
    if not settings['check_duplicate_policy']:alternates=[]
    bounds=[parent['bounds'][name] for name in ['VPD_w','I_req']]
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'effective_trial_settings.json').write_text(json.dumps(trial,indent=2))
    metadata={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),'parent_run':str(baseline),
              'trial':trial,'slice_settings':settings,'n_new_model_evaluations':0,
              'baseline_metadata_sha256':hashlib.sha256((baseline/'run_metadata.json').read_bytes()).hexdigest(),
              'input_sha256':parent['input_sha256'],'source_sha256':{},'slices':{'80':{'status':'reused_verified_baseline','path':str(baseline)}}}
    for path in [Path(__file__),ROOT/'src/calibrate_m0_losyo.py',ROOT/'src/analyze_fsim_parameters.py',args.settings.resolve()]+list((ROOT/'model').glob('*.py')):
        rel=path.relative_to(ROOT);dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        metadata['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    runs={80:baseline}
    try:
        for rh in trial['RH_sp_values']:
            if rh==80:continue
            dest=out/f'RH_{rh:g}';dest.mkdir();runs[rh]=dest
            cfg=copy.deepcopy(base);cfg['dispersal']['use_wind_gate']=False;cfg['mp_tmp']['rh_wet_night']=float(rh)
            (dest/'effective_runtime_config.json').write_text(json.dumps(cfg,indent=2))
            (dest/'effective_settings.json').write_text(json.dumps({**settings,'RH_sp':rh},indent=2))
            shutil.copy2(baseline/'site_year_manifest.csv',dest/'site_year_manifest.csv')
            print(f'RH_sp={rh:g}%: six training-only folds, three refinement rounds, workers={settings["workers"]}',flush=True)
            cache=EvaluationCache(cases,alternates,cfg,settings,dest,not args.no_progress)
            try:
                folds,summary=run_losyo(cases,cache,bounds,settings,dest,not args.no_progress)
                write_report(dest,settings,folds,summary,False)
                metadata['slices'][f'{rh:g}']={'status':'complete','path':str(dest),'n_model_evaluations':cache.count,
                    'n_duplicate_evaluations':sum(cid>=len(cases) for cid,_ in cache.cache),'duplicate_policy_differences':0}
            finally:
                cache.close();metadata['n_new_model_evaluations']+=cache.count
                (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
        summary=combine_runs(runs,settings,trial,out)
        metadata.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        metadata.update(status='failed',error_type=type(exc).__name__,error_message=str(exc));raise
    finally:(out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out/'README.md',flush=True)


if __name__=='__main__':main()
