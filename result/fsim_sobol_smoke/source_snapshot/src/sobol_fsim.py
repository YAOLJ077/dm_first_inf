#!/usr/bin/env python3
"""Sobol sensitivity of FSIM timing errors and prediction availability.

No grid search, optimization, parameter selection, or calibration. Timing
responses remain undefined whenever any required prediction is absent.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.sample import sobol as sampler
from SALib.analyze import sobol as estimator

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.parameter_table import candidate_rows, load_config
from src.analyze_fsim_parameters import load_cases, evaluate_group

_CASES=None
_CONFIG=None


def validate_settings(s):
    if s['distribution']!='independent_uniform':
        raise ValueError('Only explicitly independent uniform parameter distributions are supported')
    if not s['models'] or len(set(s['models']))!=len(s['models']) or not set(s['models'])<={'M0','M1'}:
        raise ValueError('Models must be a unique subset of M0/M1')
    points=s['checkpoints']
    if (points!=sorted(set(points)) or not points or points[-1]!=s['base_n']
            or any(not isinstance(n,int) or n<8 or n&(n-1) for n in points)):
        raise ValueError('Checkpoints must be increasing powers of two >=8, ending at base_n')
    seeds=s['sampling_seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(not isinstance(n,int) or n<=0 for n in seeds):
        raise ValueError('Sampling seeds must be unique positive integers')
    if (not isinstance(s['second_order'],bool) or not isinstance(s['check_duplicate_policy'],bool)
            or s['workers']<1 or s['chunk_size']<1 or s['bootstrap_resamples']<2
            or not 0<s['confidence_level']<1 or s['stability_absolute_tolerance']<=0):
        raise ValueError('Invalid execution or estimator settings')


def make_design(settings, table):
    designs=[];problems={}
    for model in settings['models']:
        rows=candidate_rows(model,table)
        problem={'num_vars':len(rows),'names':[r['parameter'] for r in rows],
                 'bounds':[[float(r['lower']),float(r['upper'])] for r in rows]}
        problems[model]=problem
        for seed in settings['sampling_seeds']:
            x=sampler.sample(problem,settings['base_n'],calc_second_order=settings['second_order'],
                             scramble=True,seed=seed)
            d=pd.DataFrame(x,columns=problem['names'])
            d['model']=model;d['seed']=seed;d['design_index']=np.arange(len(d))
            d['sample_id']=[f'{model}_s{seed}_r{i:07d}' for i in range(len(d))]
            d['design']='sobol'
            if 'u_tr' not in d: d['u_tr']=np.nan
            designs.append(d)
    return pd.concat(designs,ignore_index=True),problems


def build_responses(predictions,design,site_keys):
    """Align every output to the original Saltelli row order; never filter rows."""
    if len(site_keys)!=6 or len(set(site_keys))!=6:
        raise ValueError('The fixed evaluation unit is six unique site-years')
    if predictions.duplicated(['sample_id','input_key']).any():
        raise ValueError('Duplicate main prediction records')
    if set(predictions.sample_id)!=set(design.sample_id) or set(predictions.input_key)!=set(site_keys):
        raise ValueError('Unexpected or absent sample/site identifiers')
    status=predictions.pivot(index='sample_id',columns='input_key',values='status').reindex(
        index=design.sample_id,columns=site_keys)
    if status.isna().any().any() or not status.isin(['predicted','no_prediction']).all().all():
        raise ValueError('Missing records or runtime errors cannot be treated as no prediction')
    errors=predictions.pivot(index='sample_id',columns='input_key',values='signed_interval_error_days').reindex(
        index=design.sample_id,columns=site_keys).astype(float)
    available=status.eq('predicted').to_numpy()
    e=errors.to_numpy()
    if not np.isfinite(e[available]).all():
        raise ValueError('A reported prediction has a non-finite timing error')
    e=np.where(available,e,np.nan)
    complete=available.all(axis=1)
    output=design[['sample_id','model','seed','design_index']].copy().reset_index(drop=True)
    output['ALL_SIX|prediction_rate']=available.mean(axis=1)
    # NumPy mean intentionally propagates NaN; no varying denominators.
    output['ALL_SIX|RMSE_days']=np.where(complete,np.sqrt(np.mean(e**2,axis=1)),np.nan)
    output['ALL_SIX|MAE_days']=np.where(complete,np.mean(np.abs(e),axis=1),np.nan)
    for i,key in enumerate(site_keys):
        output[f'{key}|has_prediction']=available[:,i].astype(float)
        output[f'{key}|absolute_error_days']=np.abs(e[:,i])
        output[f'{key}|signed_error_days']=e[:,i]
    return output


def analyze_response(problem,y,settings,seed):
    """A complete design with a finite, variable response is mandatory."""
    y=np.asarray(y,dtype=float)
    diagnostics={'n_outputs':len(y),'n_missing':int((~np.isfinite(y)).sum()),
                 'status':'ok','warnings':''}
    if diagnostics['n_missing']:
        diagnostics['status']='unavailable_missing_predictions'
        return diagnostics,[]
    if np.ptp(y)==0:
        diagnostics.update(status='undefined_constant_response',constant_value=float(y[0]))
        return diagnostics,[]
    d=problem['num_vars'];step=2*d+2 if settings['second_order'] else d+2
    if len(y)%step: raise ValueError('Response length does not match Saltelli design')
    if np.ptp(np.r_[y[0::step],y[step-1::step]])==0:
        diagnostics['status']='undefined_base_sample_variance'
        return diagnostics,[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        si=estimator.analyze(problem,y,calc_second_order=settings['second_order'],
                             num_resamples=settings['bootstrap_resamples'],
                             conf_level=settings['confidence_level'],seed=seed,
                             print_to_console=False)
    diagnostics['warnings']=' | '.join(sorted({str(w.message) for w in caught}))
    rows=[]
    for i,name in enumerate(problem['names']):
        for kind in ('S1','ST'):
            rows.append({'index_type':kind,'parameter':name,'parameter_2':'',
                         'estimate':float(si[kind][i]),'confidence_half_width':float(si[kind+'_conf'][i])})
        if settings['second_order']:
            for j in range(i+1,d):
                rows.append({'index_type':'S2','parameter':name,'parameter_2':problem['names'][j],
                             'estimate':float(si['S2'][i,j]),'confidence_half_width':float(si['S2_conf'][i,j])})
    if not all(np.isfinite(r['estimate']) and np.isfinite(r['confidence_half_width']) for r in rows):
        diagnostics['status']='unstable_numerical_estimate'
    # Finite sample indices can be negative or exceed one; preserve estimates.
    return diagnostics,rows


def analyze_all(responses,problems,settings):
    diagnostics=[];indices=[]
    cols=[c for c in responses if '|' in c]
    for (model,seed),group in responses.groupby(['model','seed']):
        group=group.sort_values('design_index')
        if not np.array_equal(group.design_index,np.arange(len(group))):
            raise ValueError('Sobol row order or row completeness was lost')
        p=problems[model];step=2*p['num_vars']+2 if settings['second_order'] else p['num_vars']+2
        if len(group)!=settings['base_n']*step: raise ValueError('Unexpected design size')
        for n in settings['checkpoints']:
            prefix=group.iloc[:n*step]
            for k,col in enumerate(cols):
                scope,response=col.split('|')
                tags={'model':model,'seed':int(seed),'base_n':n,'scope':scope,'response':response}
                diag,rows=analyze_response(p,prefix[col].to_numpy(),settings,int(seed)+k+1)
                diagnostics.append({**tags,**diag})
                indices.extend({**tags,'status':diag['status'],**row} for row in rows)
    return pd.DataFrame(diagnostics),pd.DataFrame(indices,columns=[
        'model','seed','base_n','scope','response','status','index_type','parameter','parameter_2',
        'estimate','confidence_half_width'])


def convergence_tables(indices,settings):
    identity=['model','seed','scope','response','index_type','parameter','parameter_2']
    changes=[];spreads=[]
    for key,g in indices.groupby(identity,dropna=False):
        g=g.sort_values('base_n')
        for n0,n1 in zip(settings['checkpoints'],settings['checkpoints'][1:]):
            a=g[g.base_n==n0];b=g[g.base_n==n1]
            if a.empty or b.empty: continue
            a,b=a.iloc[0],b.iloc[0]
            delta=abs(b.estimate-a.estimate)
            changes.append({**dict(zip(identity,key)),'n_from':n0,'n_to':n1,
                            'absolute_change':delta,
                            'within_configured_tolerance':bool(a.status==b.status=='ok' and
                                delta<=settings['stability_absolute_tolerance'])})
    identity2=[k for k in identity if k!='seed']
    top=indices[(indices.base_n==settings['base_n'])&(indices.status=='ok')]
    for key,g in top.groupby(identity2,dropna=False):
        spreads.append({**dict(zip(identity2,key)),'n_seeds':len(g),
                        'estimate_range_across_seeds':g.estimate.max()-g.estimate.min(),
                        'all_seeds_available':len(g)==len(settings['sampling_seeds'])})
    return pd.DataFrame(changes,columns=identity+['n_from','n_to','absolute_change','within_configured_tolerance']),pd.DataFrame(
        spreads,columns=identity2+['n_seeds','estimate_range_across_seeds','all_seeds_available'])


def init_worker(cases,config):
    global _CASES,_CONFIG
    _CASES=cases;_CONFIG=config


def run_task(task):
    case_id,samples=task
    return evaluate_group((_CASES[case_id],samples,_CONFIG))


def report(diag,metadata,out):
    final=diag[(diag.base_n==metadata['settings']['base_n'])&(diag.scope=='ALL_SIX')]
    lines=['# FSIM Sobol 分析运行说明','',
           f'运行类型：{"小样本端到端检查，不能解释敏感性排名" if metadata["smoke"] else "正式抽样；指数能否解释仍需查看收敛检查"}。','',
           '主响应为固定六站点年份RMSE，辅以MAE和预测产生率；逐站点年份另算有符号误差、绝对误差和是否产生预测。误差沿用预测窗口到观测显症区间的距离，未使用任何最优值选择。','',
           '| 模型 | 抽样种子 | 响应 | 状态 | 缺失响应数 |',
           '|---|---:|---|---|---:|']
    for r in final.itertuples():
        lines.append(f'| {r.model} | {r.seed} | {r.response} | {r.status} | {r.n_missing} |')
    lines+=['','状态含义：','',
            '- `ok`：本次抽样具备估计条件；不自动意味着已经收敛。',
            '- `unavailable_missing_predictions`：响应存在无预测造成的缺失；不删除样本、不填惩罚值，整项Sobol估计跳过。',
            '- `undefined_constant_response`：响应没有变异，指数没有定义，不等同于所有参数效应为零。',
            '- `undefined_base_sample_variance`：Saltelli的A/B基样本无变异。',
            '- `unstable_numerical_estimate`：存在非有限估计或置信区间，不能据此给出稳定排名。','',
            '## 解释与限制','',
            '独立均匀取样是本次参数空间探索假设，不是生物学参数分布。Sobol指数描述所选范围内响应变异的贡献；它不表示参数增大必然改善误差。',
            '各站点年份分别计算，不能把模型运行次数当作独立田间样本。RMSE强调大误差；MAE作为补充。重叠率不作为本轮响应。',
            '逐样本六站点年份中只要一个无预测，汇总RMSE和MAE即记为缺失。不能以较少站点的误差替代。对于含缺失的完整设计，脚本只报告响应可用性及预测产生率的合法Sobol估计，不自动缩小范围或构造条件Sobol。',
            '置信区间是指数估计的bootstrap区间，不是六个站点年份泛化性能区间。有限样本指数可能小于0或大于1，原值保留。',
            '收敛文件比较递增N下的指数变化，以及独立scramble种子的差异；阈值只是预设诊断标准，不能代替数值和科学判断。',
            '此前发现的重复年末天气时间戳在分析副本中保留末次记录，按配置复算保留首次记录；原始天气不修改。若两种策略结果不同，须在正式解释前解决输入不确定性。','',
            '## 输出','',
            '- `sobol_design.csv`：按原始顺序保存的Saltelli设计。',
            '- `predictions.csv`：逐抽样、逐站点年份预测，包括无预测记录。',
            '- `response_vectors.csv`：与抽样行严格对齐的响应。',
            '- `response_diagnostics.csv`：每个响应在各N和种子下是否能估计。',
            '- `sobol_indices.csv`：S1、ST、S2及置信区间半宽；不输出最佳参数。',
            '- `convergence.csv`、`seed_stability.csv`：样本量和种子稳定性检查。',
            '- `run_metadata.json`、`source_snapshot/`：配置、版本、源代码及输入指纹。','']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,default=ROOT/'config/fsim_sobol_analysis.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_sobol')
    p.add_argument('--smoke',action='store_true',help='N=8, one seed; implementation check only')
    p.add_argument('--design-only',action='store_true',help='Write configuration/design without running the model')
    args=p.parse_args();s=json.loads(args.settings.read_text())
    if args.smoke:
        s.update(base_n=8,checkpoints=[8],sampling_seeds=[s['sampling_seeds'][0]],bootstrap_resamples=100)
    validate_settings(s)
    table=ROOT/s['parameter_table'];manifest=ROOT/s['manifest']
    base=load_config(table);design,problems=make_design(s,table)
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): p.error('Use an empty output directory; existing results are preserved')
    out.mkdir(parents=True,exist_ok=True)
    design.to_csv(out/'sobol_design.csv',index=False)
    shutil.copy2(table,out/'parameter_table.csv');shutil.copy2(manifest,out/'site_year_manifest.csv')
    (out/'effective_settings.json').write_text(json.dumps(s,indent=2))
    cases,errors,context,alternates,repairs=load_cases(manifest)
    repairs.to_csv(out/'weather_duplicate_audit.csv',index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out/'input_errors.csv',index=False)
        raise SystemExit('Input validation failed')
    all_cases=cases+alternates if s['check_duplicate_policy'] else cases
    metadata={'status':'design_only' if args.design_only else 'running',
              'created_utc':datetime.now(timezone.utc).isoformat(),'smoke':args.smoke,'settings':s,'problems':problems,
              'n_parameter_rows':len(design),'n_main_evaluations':len(design)*6,
              'n_duplicate_evaluations':len(design)*(len(all_cases)-6),
              'python':platform.python_version(),'SALib':importlib.metadata.version('SALib'),
              'numpy':np.__version__,'pandas':pd.__version__,'source_sha256':{},'input_sha256':{}}
    files=list((ROOT/'model').glob('*.py'))+[Path(__file__),ROOT/'src/analyze_fsim_parameters.py',table,args.settings.resolve()]
    for path in files:
        rel=path.resolve().relative_to(ROOT);dest=out/'source_snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        metadata['source_sha256'][str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [manifest]+[ROOT/n for n in pd.read_csv(manifest).weather_json]:
        metadata['input_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'{len(design)} parameter rows; {metadata["n_main_evaluations"]} main evaluations; {metadata["n_duplicate_evaluations"]} duplicate checks',flush=True)
    if args.design_only: return
    tasks=[]
    for c in range(len(all_cases)):
        for _,block in design.groupby(['model','seed','u_tr'],sort=False,dropna=False):
            for start in range(0,len(block),s['chunk_size']):
                tasks.append((c,block.iloc[start:start+s['chunk_size']].to_dict('records')))
    records=[]
    try:
        with ProcessPoolExecutor(max_workers=s['workers'],initializer=init_worker,initargs=(all_cases,base)) as pool:
            futures=[pool.submit(run_task,task) for task in tasks]
            for i,future in enumerate(as_completed(futures),1):
                batch=future.result();records.extend(batch)
                with (out/'checkpoint.jsonl').open('a') as f:
                    f.write(pd.DataFrame(batch).to_json(orient='records',lines=True))
                if i%max(1,len(tasks)//20)==0 or i==len(tasks):
                    print(f'{i}/{len(tasks)} groups complete',flush=True)
        all_predictions=pd.DataFrame(records)
        errors=all_predictions[all_predictions.status=='runtime_error']
        if not errors.empty:
            errors.to_csv(out/'runtime_errors.csv',index=False)
            raise RuntimeError('Model runtime errors; no sensitivity estimates produced')
        for col in ['signed_interval_error_days','predicted_start','predicted_end']:
            if col not in all_predictions: all_predictions[col]=np.nan
        predictions=all_predictions[all_predictions.weather_policy!='keep_first'].copy()
        alternate=all_predictions[all_predictions.weather_policy=='keep_first']
        predictions.sort_values(['model','seed','design_index','input_key']).to_csv(out/'predictions.csv',index=False)
        if not alternate.empty:
            alternate.to_csv(out/'duplicate_policy_predictions.csv',index=False)
            paired=predictions.merge(alternate,on=['sample_id','input_key'],suffixes=('_last','_first'),validate='one_to_one')
            same=np.ones(len(paired),dtype=bool)
            for col in ['status','predicted_start','predicted_end']:
                same &= paired[col+'_last'].fillna('MISSING').eq(paired[col+'_first'].fillna('MISSING')).to_numpy()
            metadata['duplicate_policy_check']={'n_pairs':len(paired),'n_differences':int((~same).sum())}
            paired.loc[~same].to_csv(out/'duplicate_policy_differences.csv',index=False)
            if not same.all(): raise ValueError('Duplicate weather policies change outputs; resolve inputs before Sobol inference')
        responses=build_responses(predictions,design,[c['input_key'] for c in cases])
        responses.to_csv(out/'response_vectors.csv',index=False)
        diag,indices=analyze_all(responses,problems,s)
        diag.to_csv(out/'response_diagnostics.csv',index=False);indices.to_csv(out/'sobol_indices.csv',index=False)
        conv,seeds=convergence_tables(indices,s)
        conv.to_csv(out/'convergence.csv',index=False);seeds.to_csv(out/'seed_stability.csv',index=False)
        metadata['status']='complete_smoke' if args.smoke else 'complete'
        metadata['completed_utc']=datetime.now(timezone.utc).isoformat()
        report(diag,metadata,out)
    except Exception as exc:
        metadata.update(status='failed',error_type=type(exc).__name__,error_message=str(exc))
        raise
    finally:
        (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out/'README.md',flush=True)


if __name__=='__main__': main()
