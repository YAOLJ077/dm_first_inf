#!/usr/bin/env python3
"""M0 leave-one-site-year-out grid calibration with training-only refinement."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.parameter_table import candidate_rows,load_config
from src.analyze_fsim_parameters import load_cases,evaluate_group

NAMES=['VPD_w','I_req']
_WORKER_CASES=None
_WORKER_BASE=None


def validate_settings(s):
    for key in ['refinement_factor','workers','chunk_size','max_candidates_per_fold']:
        if type(s[key]) is not int or s[key]<1:raise ValueError(f'Invalid {key}')
    if s['refinement_factor']<2:raise ValueError('Refinement factor must be >=2')
    if type(s['refinement_rounds']) is not int or not 0<=s['refinement_rounds']<=6:
        raise ValueError('Refinement rounds must be 0..6')
    if len(s['coarse_levels'])!=2 or any(type(n) is not int or n<3 for n in s['coarse_levels']):
        raise ValueError('Two coarse levels >=3 required')
    if np.prod(s['coarse_levels'])>s['max_candidates_per_fold']:raise ValueError('Budget smaller than coarse grid')
    for key in ['refinement_margin_days','tie_atol_days']:
        if not np.isfinite(s[key]) or s[key]<0:raise ValueError(f'Invalid {key}')
    for key in ['near_optimal_tolerances_days','accuracy_tolerances_days']:
        if not s[key] or any(not np.isfinite(v) or v<=0 for v in s[key]):raise ValueError(f'Invalid {key}')
    if s['refinement_margin_days']<max(s['near_optimal_tolerances_days']):
        raise ValueError('Refinement margin must cover reported near-optimal tolerances')
    if s['objective']!='training_interval_MAE_days' or s['require_all_training_predictions'] is not True:
        raise ValueError('Requires complete-training-prediction MAE objective')
    if s['tie_rule']!='lowest_VPD_w_then_lowest_I_req':raise ValueError('Unsupported tie rule')
    if type(s['check_duplicate_policy']) is not bool:raise ValueError('Invalid duplicate-policy setting')


def make_lattice(bounds,s):
    scale=s['refinement_factor']**s['refinement_rounds']
    extent=tuple((n-1)*scale for n in s['coarse_levels'])
    points=set(product(range(0,extent[0]+1,scale),range(0,extent[1]+1,scale)))
    cells=[(i,j,scale) for i in range(0,extent[0],scale) for j in range(0,extent[1],scale)]
    return extent,points,cells


def sample_for(point,extent,bounds):
    sample={'sample_id':f'M0_x{point[0]}_y{point[1]}','model':'M0','design':'losyo_grid','u_tr':np.nan}
    for k,name in enumerate(NAMES):sample[name]=bounds[k][0]+point[k]/extent[k]*(bounds[k][1]-bounds[k][0])
    sample['grid_x'],sample['grid_y']=point
    return sample


def training_scores(records,train_keys):
    """Explicit key filter prevents held-out records from affecting any objective."""
    if len(train_keys)!=5 or len(set(train_keys))!=5:raise ValueError('Exactly five training site-years required')
    frame=pd.DataFrame(records)
    frame=frame[frame.input_key.isin(train_keys)]
    if frame.empty:raise ValueError('Empty training predictions')
    rows=[]
    for sid,g in frame.groupby('sample_id',sort=False):
        if len(g)!=5 or set(g.input_key)!=set(train_keys):raise ValueError('Incomplete/duplicate training evaluations')
        if not g.status.isin(['predicted','no_prediction']).all():raise ValueError('Runtime failure cannot be a calibration loss')
        a=g.status.eq('predicted');e=pd.to_numeric(g.signed_interval_error_days,errors='coerce')
        if not np.isfinite(e[a]).all():raise ValueError('Nonfinite error for prediction')
        first=g.iloc[0]
        rows.append({k:first[k] for k in ['sample_id','grid_x','grid_y']+NAMES}|
                    {'n_training_predictions':int(a.sum()),'feasible':bool(a.all()),
                     'training_MAE_days':float(e.abs().mean()) if a.all() else np.nan,
                     'training_RMSE_days':float(np.sqrt(np.mean(e**2))) if a.all() else np.nan})
    return pd.DataFrame(rows)


def best_and_ties(scores,atol):
    eligible=scores[scores.feasible]
    if eligible.empty:return None,eligible.copy()
    best=float(eligible.training_MAE_days.min())
    ties=eligible[eligible.training_MAE_days<=best+atol].sort_values(NAMES)
    return best,ties


def refine_cells(cells,scores,s):
    """Split every leaf touching a training-near-optimal vertex, including disconnected regions.

    With no feasible point, split the whole current domain. Narrow basins between
    sampled vertices can still be missed; this is a finite-resolution search.
    """
    best,_=best_and_ties(scores,s['tie_atol_days'])
    mask=scores.feasible.copy()
    if best is not None:
        mask &= scores.training_MAE_days.le(best+s['refinement_margin_days']+s['tie_atol_days'])
    competitive=set(map(tuple,scores.loc[mask,['grid_x','grid_y']].to_numpy(dtype=int)))
    new=set();leaves=[];count=0;factor=s['refinement_factor']
    for x,y,width in cells:
        vertices={(x,y),(x+width,y),(x,y+width),(x+width,y+width)}
        if width>=factor and (best is None or vertices&competitive):
            step=width//factor;count+=1
            new.update(product(range(x,x+width+1,step),range(y,y+width+1,step)))
            leaves.extend((i,j,step) for i in range(x,x+width,step) for j in range(y,y+width,step))
        else:leaves.append((x,y,width))
    return new,leaves,count


def worker_init(cases,base):
    global _WORKER_CASES,_WORKER_BASE
    _WORKER_CASES=cases;_WORKER_BASE=base


def worker_run(task):
    case_id,samples=task
    return evaluate_group((_WORKER_CASES[case_id],samples,_WORKER_BASE))


class EvaluationCache:
    def __init__(self,cases,alternates,base,settings,out,progress):
        self.cases=cases+alternates;self.settings=settings;self.out=out;self.progress=progress
        self.cache={};self.count=0;self.duplicate_checks=0
        self.primary={c['input_key']:i for i,c in enumerate(cases)}
        self.alt={c['input_key']:i+len(cases) for i,c in enumerate(alternates)}
        self.pool=ProcessPoolExecutor(max_workers=settings['workers'],initializer=worker_init,initargs=(self.cases,base))

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)

    def evaluate(self,samples,keys,label):
        tasks=[]
        for key in keys:
            ids=[self.primary[key]]+([self.alt[key]] if key in self.alt else [])
            for cid in ids:
                missing=[s for s in samples if (cid,s['sample_id']) not in self.cache]
                for start in range(0,len(missing),self.settings['chunk_size']):
                    tasks.append((cid,missing[start:start+self.settings['chunk_size']]))
        futures={self.pool.submit(worker_run,t):t[0] for t in tasks}
        with tqdm(total=sum(len(t[1]) for t in tasks),desc=label,unit='eval',disable=not self.progress,dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                cid=futures[future];batch=future.result()
                with (self.out/'evaluation_checkpoint.jsonl').open('a') as f:
                    f.write(pd.DataFrame(batch).to_json(orient='records',lines=True))
                for r in batch:
                    if r['status'] not in ['predicted','no_prediction']:raise RuntimeError(f'Model error: {r}')
                    r.setdefault('signed_interval_error_days',np.nan)
                    self.cache[cid,r['sample_id']]=r
                self.count+=len(batch);bar.update(len(batch))
        records=[]
        for key in keys:
            for s in samples:
                r=self.cache[self.primary[key],s['sample_id']]
                if key in self.alt:
                    alt=self.cache[self.alt[key],s['sample_id']]
                    for col in ['status','predicted_start','predicted_end']:
                        if r.get(col)!=alt.get(col):raise ValueError(f'Duplicate-weather output differs: {key} {s["sample_id"]}')
                records.append(r)
        return records


def summarize_oof(oof,settings):
    selected=oof.calibration_status.eq('selected')
    available=selected & oof.status.eq('predicted')
    e=pd.to_numeric(oof.signed_interval_error_days,errors='coerce')
    result={'n_folds':len(oof),'n_calibration_failures':int((~selected).sum()),
            'n_predictions':int(available.sum()),'n_no_predictions':int((selected & ~available).sum()),
            'prediction_rate_fixed_six':float(available.sum()/6),
            'MAE_days':float(e.abs().mean()) if available.all() else None,
            'RMSE_days':float(np.sqrt(np.mean(e**2))) if available.all() else None,
            'conditional_MAE_days':float(e[available].abs().mean()) if available.any() else None,
            'conditional_RMSE_days':float(np.sqrt(np.mean(e[available]**2))) if available.any() else None}
    for t in settings['accuracy_tolerances_days']:
        result[f'accurate_prediction_rate_{t:g}d_fixed_six']=float((available&e.abs().le(t)).sum()/6)
    return result


def run_losyo(cases,cache,bounds,s,out,progress=True):
    keys=[c['input_key'] for c in cases];extent,initial,initial_cells=make_lattice(bounds,s)
    folds=[];oof=[]
    for holdout in tqdm(keys,desc='LOSYO folds',unit='fold',disable=not progress):
        foldout=out/'folds'/holdout;foldout.mkdir(parents=True)
        train=[k for k in keys if k!=holdout]
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
        fold={'held_out':holdout,'training_keys':train,'stopping_reason':stopping,'n_candidates':len(scores),
              'best_training_MAE_days':best,'n_tied_best':len(ties),'calibration_status':'selected' if best is not None else 'no_feasible_training_combination'}
        if best is None:
            (foldout/'selection.json').write_text(json.dumps(fold,indent=2))
            oof.append({'input_key':holdout,'calibration_status':fold['calibration_status'],'status':'calibration_failed','signed_interval_error_days':np.nan})
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
            test=cache.evaluate([chosen_sample],[holdout],f'{holdout} held-out')[0]
            record={**test,'calibration_status':'selected','held_out':holdout}
            pd.DataFrame([record]).to_csv(foldout/'held_out_prediction.csv',index=False)
            oof.append(record)
        folds.append(fold)
    frame=pd.DataFrame(oof);frame.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame([{k:v for k,v in f.items() if not isinstance(v,(list,dict))} for f in folds]).to_csv(out/'fold_summary.csv',index=False)
    summary=summarize_oof(frame,s)
    (out/'oof_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return folds,summary


def write_report(out,s,folds,summary,smoke):
    lines=['# M0 LOSYO 校准','',f'运行类型：{"小规模实现验证，不用于科学解释" if smoke else "正式校准"}。',
           '每折只使用五个训练站点年份；要求五个均产生预测，在可行组合中最小化绝对区间距离的平均值（MAE）。',
           '所有达到训练最佳MAE +1天的已采样顶点所接触的网格单元均参与加密；无可行点时全域加密。具体阈值以 effective_settings.json 为准。',
           '并列最优在数值容差内按 VPD_w、I_req 升序选取已计算的实际组合；不按留出结果选择。该规则仅使结果可复现，不表示低参数值更可信。',
           '网格结果依赖分辨率；局部加密可能遗漏粗网格顶点之间狭窄的低误差区域，不能称为连续空间的精确全局最优。',
           '近优集合是训练损失集合，不是后验分布或置信区间。触边只记录，不自动扩展参数范围。','',
           '| 留出站点年份 | 状态 | 最佳训练MAE（天） | 候选数 |', '|---|---|---:|---:|']
    for f in folds:lines.append(f'| {f["held_out"]} | {f["calibration_status"]} | {f["best_training_MAE_days"]} | {f["n_candidates"]} |')
    lines+=['','OOF汇总：','', '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','',
            '任何留出预测缺失时，完整六站点MAE/RMSE为null；conditional指标仅针对成功预测，必须与失败数同时查看。',
            '7/10天准确比例分母固定六；校准失败也不计为准确预测，但单独列出失败数。没有给缺失预测赋予日期或误差。',
            '文件：folds/*/loss_surface*.csv 为训练损失面；refinement_history.csv 为加密变化；selection.json 为冻结选择；',
            'tied_best_parameters.csv / near_optimal_parameters.csv 为并列及近优集合；held_out_prediction.csv 为单个留出评价。',
            'oof_predictions.csv 为六折结果，fold_summary.csv 为参数与训练目标汇总。模型计算缓存仅复用固定参数和站点的确定性结果，不复用其他折的选参决策。','']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,default=ROOT/'config/fsim_m0_losyo.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_m0_losyo')
    p.add_argument('--workers',type=int,choices=[2,4])
    p.add_argument('--smoke',action='store_true',help='Tiny six-fold implementation check only')
    p.add_argument('--design-only',action='store_true',help='Validate inputs and write design without evaluating model')
    p.add_argument('--no-progress',action='store_true')
    args=p.parse_args();s=json.loads(args.settings.read_text())
    if args.workers is not None:s['workers']=args.workers
    if args.smoke:s.update(coarse_levels=[3,3],refinement_rounds=1,max_candidates_per_fold=25)
    validate_settings(s)
    table=ROOT/s['parameter_table'];manifest=ROOT/s['manifest'];base=load_config(table)
    base['dispersal']['use_wind_gate']=False
    candidates=candidate_rows('M0',table)
    if [r['parameter'] for r in candidates]!=NAMES:raise ValueError('M0 must open only VPD_w and I_req')
    bounds=[[float(r['lower']),float(r['upper'])] for r in candidates]
    cases,errors,context,alternates,repairs=load_cases(manifest)
    if errors:raise ValueError(errors)
    if not s['check_duplicate_policy']:alternates=[]
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('Use an empty output directory; existing results are preserved')
    out.mkdir(parents=True,exist_ok=True)
    extent,points,_=make_lattice(bounds,s)
    pd.DataFrame([sample_for(pt,extent,bounds) for pt in sorted(points)]).to_csv(out/'coarse_design.csv',index=False)
    shutil.copy2(table,out/'parameter_table.csv');shutil.copy2(manifest,out/'site_year_manifest.csv')
    repairs.to_csv(out/'weather_duplicate_audit.csv',index=False)
    (out/'effective_settings.json').write_text(json.dumps(s,indent=2))
    metadata={'status':'design_only' if args.design_only else 'running','model':'M0','smoke':args.smoke,
              'created_utc':datetime.now(timezone.utc).isoformat(),'settings':s,'bounds':dict(zip(NAMES,bounds)),
              'final_lattice_spacing':dict(zip(NAMES,[(hi-lo)/n for (lo,hi),n in zip(bounds,extent)])),
              'n_coarse_candidates':len(points),'full_fine_grid_candidates':int(np.prod(np.array(extent)+1)),
              'source_sha256':{},'input_sha256':{}}
    for path in list((ROOT/'model').glob('*.py'))+[Path(__file__),ROOT/'src/analyze_fsim_parameters.py',args.settings.resolve(),table]:
        dest=out/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        metadata['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [manifest]+[ROOT/f for f in pd.read_csv(manifest).weather_json]:
        metadata['input_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Six M0 folds; {len(points)} coarse candidates; up to {metadata["full_fine_grid_candidates"]} points on final lattice; workers={s["workers"]}',flush=True)
    if args.design_only:return
    cache=EvaluationCache(cases,alternates,base,s,out,not args.no_progress)
    try:
        folds,summary=run_losyo(cases,cache,bounds,s,out,not args.no_progress)
        write_report(out,s,folds,summary,args.smoke)
        metadata.update(status='complete_smoke' if args.smoke else 'complete',
                        n_model_evaluations=cache.count,n_calibration_failures=summary['n_calibration_failures'],
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        metadata.update(status='failed',error_type=type(exc).__name__,error_message=str(exc));raise
    finally:
        cache.close();metadata['n_model_evaluations']=cache.count
        (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out/'README.md',flush=True)


if __name__=='__main__':main()
