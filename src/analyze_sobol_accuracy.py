#!/usr/bin/env python3
"""Add accuracy-tolerance responses to completed Sobol predictions, without model runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.sobol_fsim import (build_responses, analyze_all, convergence_tables,
                           validate_settings, accuracy_response_name, report)
from src.plot_sobol_fsim import plt, COLORS, LABELS, EXTREME_CI, forest, index_legend, save


def plot_results(responses,indices,settings,sites,out,days):
    name=accuracy_response_name(days);seeds=settings['sampling_seeds'];n=settings['base_n']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
    idx=indices[indices.response==name].copy().fillna({'parameter_2':''})
    idx['extreme_interval']=(~np.isfinite(idx.confidence_half_width))|(idx.confidence_half_width>EXTREME_CI)
    idx.to_csv(out/'plot_index_audit.csv',index=False)
    summary=[]
    for (model,seed),g in responses.groupby(['model','seed']):
        for scope in ['ALL_SIX']+sites:
            accurate=g[f'{scope}|{name}'].mean()
            available=g[f'{scope}|'+('prediction_rate' if scope=='ALL_SIX' else 'has_prediction')].mean()
            summary.append(dict(model=model,seed=seed,scope=scope,accurate_rate=accurate,
                                inaccurate_rate=available-accurate,no_prediction_rate=1-available))
    summary=pd.DataFrame(summary);summary.to_csv(out/'accuracy_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.17,wspace=.38)
    fig.suptitle(f'Prediction outcomes at a {days:g}-day error tolerance | N = {n}',y=.97,weight='bold')
    labels=['Six-site average']+sites
    for ax,model in zip(axes,settings['models']):
        for si,seed in enumerate(seeds):
            g=summary[(summary.model==model)&(summary.seed==seed)].set_index('scope').loc[['ALL_SIX']+sites]
            y=np.arange(len(labels))+(si-.5)*.34
            left=np.zeros(len(labels))
            for col,color,label in [('accurate_rate','#009E73','Prediction within 10 d'),
                                    ('inaccurate_rate','#E6A65D','Prediction outside 10 d'),
                                    ('no_prediction_rate','#BC6571','No prediction')]:
                width=g[col].to_numpy()*100
                ax.barh(y,width,left=left,height=.30,color=color,label=label if si==0 else None,
                        edgecolor='white',linewidth=.4)
                left+=width
            for yy,v in zip(y,g.accurate_rate):
                inside=v>=.08
                ax.text(v*50 if inside else max(1,v*100+1),yy,f'{v*100:.1f}',
                        ha='center' if inside else 'left',va='center',fontsize=7,
                        color='white' if inside else '#333333')
        ax.set(yticks=range(len(labels)),yticklabels=labels,xlim=(0,100),xlabel='Design outcome (%)')
        ax.invert_yaxis();ax.set_title(model,loc='left');ax.spines[['top','right']].set_visible(False)
    handles,labels_=axes[0].get_legend_handles_labels()
    labels_=[x.replace('10 d',f'{days:g} d') for x in labels_]
    fig.legend(handles,labels_,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False)
    save(fig,out,'01_accuracy_outcomes',
         f'Within each site-year: upper bar = seed {seeds[0]}, lower bar = seed {seeds[1]}. Numbers label the within-tolerance percentage.\n'
         'Accuracy requires a prediction and absolute onset-interval error ≤ tolerance; no prediction never counts as accurate.\n'
         'Six-site response uses a fixed denominator of six. Design percentages are descriptive, not independent field success probabilities.')

    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    fig.subplots_adjust(left=.1,right=.97,top=.81,bottom=.17,hspace=.5,wspace=.4)
    fig.suptitle(f'Sobol sensitivity of the within-{days:g}-day prediction fraction | N = {n}',y=.98,weight='bold')
    index_legend(fig,seeds)
    final=idx[(idx.base_n==n)&(idx.scope=='ALL_SIX')]
    for col,model in enumerate(settings['models']):
        g=final[final.model==model];pars=list(COLORS)[:2] if model=='M0' else list(COLORS)
        forest(axes[0,col],g[g.index_type!='S2'],pars,seeds)
        axes[0,col].set_title(f'{model} | S1 and ST',loc='left')
        pairs=[(a,b) for j,a in enumerate(pars) for b in pars[j+1:]]
        forest(axes[1,col],g[g.index_type=='S2'],pairs,seeds,('S2',))
        axes[1,col].set_title(f'{model} | S2',loc='left')
    save(fig,out,'02_accuracy_sobol',
         'Response: fraction of six site-years with a prediction and absolute onset-interval error within the tolerance.\n'
         'Bars: estimate ± 95% bootstrap half-width. Raw values outside [0,1] are retained; crossed points mark extreme intervals.\n'
         f'This is a new accuracy response, not RMSE. {days:g} days is an exploratory tolerance, not a validated management standard.')

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fig.subplots_adjust(left=.08,right=.97,top=.82,bottom=.15,hspace=.4,wspace=.3)
    fig.suptitle(f'Within-{days:g}-day accuracy: sample-size and seed stability',y=.98,weight='bold')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=c,label=LABELS[p]) for p,c in COLORS.items()]
    handles += [Line2D([],[],color='#333333',ls='-' if i==0 else '--',label=f'Seed {i+1}: {s}') for i,s in enumerate(seeds)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False,fontsize=9)
    for col,model in enumerate(settings['models']):
        for row,kind in enumerate(['S1','ST']):
            ax=axes[row,col]
            q=idx[(idx.model==model)&(idx.scope=='ALL_SIX')&(idx.index_type==kind)]
            for (par,seed),g in q.groupby(['parameter','seed']):
                g=g.sort_values('base_n');si=seeds.index(seed)
                ax.plot(g.base_n,g.estimate,color=COLORS[par],marker='o',ms=4,
                        ls='-' if si==0 else '--',mfc=COLORS[par] if si==0 else 'white')
            ax.set(xticks=settings['checkpoints'],xlabel='Base sample size N',ylabel=kind)
            ax.set_title(f'{model} | {kind}',loc='left');ax.grid(alpha=.15)
            ax.axhline(0,color='#999999',lw=.6);ax.axhline(1,color='#bbbbbb',ls=':',lw=.6)
            ax.spines[['top','right']].set_visible(False)
    save(fig,out,'03_accuracy_convergence',
         'Nested prefixes reuse the same model evaluations. Independent seeds are not averaged. Panels have independent y scales.\n'
         'A complete response permits estimation but does not establish convergence; inspect intervals and seed differences together.')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'result/fsim_sobol_n128')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_sobol_n128_accuracy10')
    parser.add_argument('--figure-dir',type=Path,default=ROOT/'fig/fsim_sobol_n128_accuracy10')
    parser.add_argument('--tolerance-days',type=float,default=10)
    args=parser.parse_args();source=args.input_dir.resolve();out=args.output_dir.resolve();figdir=args.figure_dir.resolve()
    meta=json.loads((source/'run_metadata.json').read_text())
    if meta['status']!='complete': raise ValueError('Requires completed model evaluation')
    settings=dict(meta['settings']);settings['accuracy_tolerances_days']=[args.tolerance_days]
    validate_settings(settings)
    for path in [out,figdir]:
        if path.exists() and any(path.iterdir()): raise ValueError(f'Use an empty output directory: {path}')
    design=pd.read_csv(source/'sobol_design.csv');pred=pd.read_csv(source/'predictions.csv')
    sites=pd.read_csv(source/'site_year_manifest.csv').input_key.tolist()
    if len(pred)!=meta['n_main_evaluations']: raise ValueError('Prediction count mismatch')
    if len(pd.read_csv(source/'duplicate_policy_differences.csv')): raise ValueError('Unresolved duplicate-policy differences')
    responses=build_responses(pred,design,sites,[args.tolerance_days])
    old=pd.read_csv(source/'response_vectors.csv')
    pd.testing.assert_frame_equal(responses[old.columns],old,check_exact=False,rtol=1e-12,atol=1e-12)
    for path in [out,figdir]: path.mkdir(parents=True,exist_ok=True)
    responses.to_csv(out/'response_vectors.csv',index=False)
    diag,indices=analyze_all(responses,meta['problems'],settings,show_progress=True)
    legacy=pd.read_csv(source/'sobol_indices.csv')
    retained=indices[indices.response.isin(legacy.response.unique())].reset_index(drop=True).fillna({'parameter_2':''})
    pd.testing.assert_frame_equal(retained,legacy.fillna({'parameter_2':''}),check_exact=False,rtol=1e-10,atol=1e-12)
    for filename,frame in [('response_diagnostics',diag),('sobol_indices',indices)]:
        frame.to_csv(out/f'{filename}.csv',index=False)
    conv,stability=convergence_tables(indices,settings)
    conv.to_csv(out/'convergence.csv',index=False);stability.to_csv(out/'seed_stability.csv',index=False)
    (out/'effective_settings.json').write_text(json.dumps(settings,indent=2))
    newmeta=dict(meta)
    newmeta.update(status='complete',settings=settings,analysis_kind='accuracy_response_postprocessing',
                   parent_run=str(source),new_model_evaluations=0,
                   postprocessed_utc=datetime.now(timezone.utc).isoformat(),
                   parent_results_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.csv')})
    newmeta['postprocessing_source_sha256']={}
    for path in [Path(__file__),ROOT/'src/sobol_fsim.py',ROOT/'src/plot_sobol_fsim.py']:
        dest=out/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        newmeta['postprocessing_source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'run_metadata.json').write_text(json.dumps(newmeta,indent=2))
    report(diag,newmeta,out)
    summary=plot_results(responses,indices,settings,sites,figdir,args.tolerance_days)
    name=accuracy_response_name(args.tolerance_days)
    final=indices[(indices.base_n==settings['base_n'])&(indices.scope=='ALL_SIX')&(indices.response==name)]
    lines=[f'# 容许误差 {args.tolerance_days:g} 天的准确预测比例：探索分析','',
           'A_j=1 当且仅当模型产生显症窗口，且窗口与观测区间的绝对边界距离 ≤ 容许误差；否则 A_j=0。汇总 A=sum(A_j)/6。',
           f'程序错误不能当作无预测。原始 RMSE/MAE 定义和缺失规则未变；{args.tolerance_days:g} 天不是已经验证的管理标准。','',
           f'原运行：`{source}`；新增分析：`{out}`。新增模型评估次数：0。',
           '原有响应与指数已作一致性核对；新指标追加至响应列末尾，保留旧指标 bootstrap 种子。','',
           '| 模型 | 种子 | 准确预测比例（设计均值） |', '|---|---:|---:|']
    for r in summary[summary.scope=='ALL_SIX'].itertuples():
        lines.append(f'| {r.model} | {r.seed} | {r.accurate_rate:.2%} |')
    lines+=['','该比例是指定参数分布下模型设计输出的描述，不是独立验证准确率。','',
            '| 模型 | 种子 | 参数 | ST | 95% bootstrap 半宽 |','|---|---:|---|---:|---:|']
    for r in final[final.index_type=='ST'].itertuples():
        lines.append(f'| {r.model} | {r.seed} | {r.parameter} | {r.estimate:.3f} | {r.confidence_half_width:.3f} |')
    lines+=['','01_accuracy_outcomes：准确预测、超出容许误差和无预测的组成。',
            '02_accuracy_sobol：准确预测比例的 S1/ST/S2。03_accuracy_convergence：样本量与种子稳定性。',
            '完整响应不保证指数收敛；不可将 ST 解释为提高参数可改善多少准确率，也不可用 M0/M1 的指数大小判断模型优劣。',
            '每图均含 300 dpi PNG 和矢量 PDF。plot_index_audit.csv 保留所有新指标指数及极端区间显示标记。','']
    (figdir/'README.md').write_text('\n'.join(lines))
    print(final.to_string(index=False))
    print(figdir/'README.md')


if __name__=='__main__': main()
