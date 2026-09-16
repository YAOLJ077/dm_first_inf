#!/usr/bin/env python3
"""Report exact-coordinate M0/M1 structural contrasts, without parameter selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.sobol_fsim import analyze_all, convergence_tables
from src.plot_sobol_fsim import plt, COLORS, LABELS, EXTREME_CI, forest, index_legend, save

TOLERANCES=[7,10]


def build_pairs(mapping,m0,m1):
    """Keep one pair per original M1 row/site; preserve repeated Saltelli coordinates."""
    if m0.duplicated(['sample_id','input_key']).any() or m1.duplicated(['sample_id','input_key']).any():
        raise ValueError('Duplicate prediction identity')
    fields=['sample_id','input_key','status','signed_interval_error_days','predicted_start','predicted_end']
    base=m0[fields+['VPD_w','I_req']].rename(columns={'sample_id':'common_id'})
    base=base.rename(columns={c:c+'_M0' for c in base if c not in ['common_id','input_key']})
    gated=m1[fields+['VPD_w','I_req','u_tr']].rename(columns={c:c+'_M1' for c in fields[2:]+['VPD_w','I_req','u_tr']})
    pairs=mapping.merge(gated,on='sample_id',validate='one_to_many').merge(base,on=['common_id','input_key'],validate='many_to_one')
    if len(pairs)!=len(mapping)*6 or pairs.duplicated(['sample_id','input_key']).any():
        raise ValueError('Every mapped row must have six unique pairs')
    for p in ['VPD_w','I_req']:
        if not (pairs[p].eq(pairs[p+'_M0']).all() and pairs[p].eq(pairs[p+'_M1']).all()):
            raise ValueError('Common parameter coordinates differ')
    if not pairs.u_tr.eq(pairs.u_tr_M1).all():raise ValueError('M1 wind coordinate mismatch')
    for model in ['M0','M1']:
        if not pairs['status_'+model].isin(['predicted','no_prediction']).all():raise ValueError('Runtime or missing prediction records')
        avail=pairs['status_'+model].eq('predicted')
        e=pairs['signed_interval_error_days_'+model]
        if not np.isfinite(e[avail]).all():raise ValueError('Predicted error must be finite')
        pairs['available_'+model]=avail.astype(int)
        pairs['absolute_error_'+model]=e.abs().where(avail)
        for t in TOLERANCES:
            pairs[f'accurate_{t}d_{model}']=(avail&e.abs().le(t)).astype(int)
    a=pairs.available_M0.eq(1);b=pairs.available_M1.eq(1)
    pairs['availability_transition']=np.select([a&b,a&~b,~a&b],['both','lost','gained'],default='neither')
    both=a&b
    pairs['absolute_error_delta_days']=(pairs.absolute_error_M1-pairs.absolute_error_M0).where(both)
    for model in ['M0','M1']:
        start=pd.to_datetime(pairs['predicted_start_'+model],format='mixed');end=pd.to_datetime(pairs['predicted_end_'+model],format='mixed')
        pairs['midpoint_'+model]=start+(end-start)/2
    pairs['onset_shift_days']=((pairs.midpoint_M1-pairs.midpoint_M0)/pd.Timedelta(days=1)).where(both)
    pairs['availability_delta']=pairs.available_M1-pairs.available_M0
    for t in TOLERANCES:
        a=pairs[f'accurate_{t}d_M0'];b=pairs[f'accurate_{t}d_M1']
        pairs[f'accuracy_delta_{t}d']=b-a
        pairs[f'accuracy_transition_{t}d']=np.select([a.eq(1)&b.eq(1),a.eq(1)&b.eq(0),a.eq(0)&b.eq(1)],
                                                     ['both_accurate','lost_accuracy','gained_accuracy'],default='neither_accurate')
    return pairs.sort_values(['seed','design_index','input_key']).reset_index(drop=True)


def summarize_pairs(pairs,sites,stride):
    records=[];wind=[]
    for sample_set,subset in [('base_AB',pairs[pairs.design_index.mod(stride).isin([0,stride-1])]),('saltelli_all',pairs)]:
        for seed,g in subset.groupby('seed'):
            for scope in ['ALL_SIX']+sites:
                q=g if scope=='ALL_SIX' else g[g.input_key.eq(scope)]
                both=q[q.availability_transition.eq('both')]
                r=dict(sample_set=sample_set,seed=seed,scope=scope,n_pairs=len(q),n_both=len(both),
                       n_parameter_rows=q.sample_id.nunique())
                for k in ['both','lost','gained','neither']:
                    r['availability_'+k+'_rate']=q.availability_transition.eq(k).mean()
                for model in ['M0','M1']:
                    r['prediction_rate_'+model]=q['available_'+model].mean()
                for t in TOLERANCES:
                    for model in ['M0','M1']:r[f'accuracy_{t}d_{model}']=q[f'accurate_{t}d_{model}'].mean()
                    for k in ['gained_accuracy','lost_accuracy','both_accurate','neither_accurate']:
                        r[f'{k}_{t}d_rate']=q[f'accuracy_transition_{t}d'].eq(k).mean()
                    r[f'accuracy_delta_{t}d']=q[f'accuracy_delta_{t}d'].mean()
                for col in ['onset_shift_days','absolute_error_delta_days']:
                    for quant,label in [(0.25,'q25'),(.5,'median'),(.75,'q75')]:
                        r[col+'_'+label]=both[col].quantile(quant)
                r['both_shift_earlier_rate']=both.onset_shift_days.lt(0).mean()
                r['both_shift_later_rate']=both.onset_shift_days.gt(0).mean()
                records.append(r)
            if sample_set=='base_AB':
                b=g.copy();b['wind_bin']=pd.cut(b.u_tr,bins=np.arange(1,9),right=False,include_lowest=True)
                for label,q in b.groupby('wind_bin',observed=True):
                    row=dict(seed=seed,wind_lower=label.left,wind_upper=label.right,n_pairs=len(q),
                             n_parameter_rows=q.sample_id.nunique(),
                             lost_prediction_rate=q.availability_transition.eq('lost').mean(),
                             gained_prediction_rate=q.availability_transition.eq('gained').mean())
                    for t in TOLERANCES:row[f'accuracy_delta_{t}d']=q[f'accuracy_delta_{t}d'].mean()
                    wind.append(row)
    return pd.DataFrame(records),pd.DataFrame(wind)


def delta_responses(pairs,mapping,sites):
    out=mapping[['sample_id','model','seed','design_index']].copy()
    out['model']='M1_minus_M0'
    for col in ['availability_delta']+[f'accuracy_delta_{t}d' for t in TOLERANCES]:
        table=pairs.pivot(index='sample_id',columns='input_key',values=col).reindex(index=out.sample_id,columns=sites)
        if table.isna().any().any():raise ValueError('Incomplete paired response')
        out['ALL_SIX|'+col]=table.mean(axis=1).to_numpy()
        for site in sites:out[site+'|'+col]=table[site].to_numpy()
    return out


def plots(summary,wind,indices,settings,sites,out):
    seeds=settings['sampling_seeds'];n=settings['base_n'];q=summary[summary.sample_set.eq('base_AB')]
    scopes=['ALL_SIX']+sites;labels=['Six-site average']+sites
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'xtick.labelsize':9,'ytick.labelsize':9})
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    fig.subplots_adjust(left=.15,right=.98,top=.85,bottom=.19,wspace=.42)
    fig.suptitle('Matched wind-gate contrast: gains and losses in accurate predictions',y=.97,weight='bold')
    for ax,t in zip(axes,TOLERANCES):
        for si,seed in enumerate(seeds):
            g=q[q.seed.eq(seed)].set_index('scope').loc[scopes];y=np.arange(len(scopes))+(si-.5)*.32
            ax.barh(y,g[f'gained_accuracy_{t}d_rate']*100,height=.28,color='#009E73',label='Gained accuracy' if si==0 else None)
            ax.barh(y,-g[f'lost_accuracy_{t}d_rate']*100,height=.28,color='#BC6571',label='Lost accuracy' if si==0 else None)
            ax.scatter(g[f'accuracy_delta_{t}d']*100,y,color='black',s=15,zorder=3,label='Net change' if si==0 else None)
        ax.axvline(0,color='#444444',lw=.7);ax.set(yticks=range(len(scopes)),yticklabels=labels,xlabel='Percentage points of matched pairs')
        ax.invert_yaxis();ax.set_title(f'Absolute error ≤ {t} days',loc='left');ax.spines[['top','right']].set_visible(False)
    handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='upper center',bbox_to_anchor=(.5,.935),ncol=3,frameon=False)
    lim=max(max(abs(v) for v in ax.get_xlim()) for ax in axes)
    for ax in axes:ax.set_xlim(-lim,lim)
    save(fig,out,'01_paired_accuracy_changes',
         f'Common VPD_w and I_req are exactly matched. Upper/lower bars: seeds {seeds[0]}/{seeds[1]}.\n'
         'Summaries use A/B base rows only; the six-site average weights each site-year equally. Positive net change favours M1.\n'
         'Gained/lost accuracy includes prediction availability changes. These are parameter-space contrasts, not out-of-fold validation.')

    fig,axes=plt.subplots(1,2,figsize=(12,6))
    fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.19,wspace=.4)
    fig.suptitle('Matched prediction availability: effect of adding the wind gate',y=.97,weight='bold')
    categories=[('both','#B6C4C9','Both predict'),('lost','#BC6571','M0 only: prediction lost'),('gained','#009E73','M1 only: prediction gained'),('neither','#DEDEDE','Neither predicts')]
    for ax,seed in zip(axes,seeds):
        g=q[q.seed.eq(seed)].set_index('scope').loc[scopes];left=np.zeros(len(scopes))
        for cat,c,label in categories:
            values=g[f'availability_{cat}_rate'].to_numpy()*100
            ax.barh(np.arange(len(scopes)),values,left=left,color=c,label=label,height=.7);left+=values
        ax.set(yticks=range(len(scopes)),yticklabels=labels,xlim=(0,100),xlabel='Matched pairs (%)')
        ax.invert_yaxis();ax.set_title(f'Seed {seed}',loc='left');ax.spines[['top','right']].set_visible(False)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.935),ncol=2,frameon=False,fontsize=9)
    save(fig,out,'02_paired_availability',
         'A/B base rows; identical shared parameters, weather, phenology and fixed configuration within each pair.\n'
         'Missing predictions remain explicit outcomes. The number of simulation pairs is not the number of independent field observations.')

    fig,axes=plt.subplots(1,2,figsize=(12,6))
    fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.2,wspace=.4)
    fig.suptitle('Timing changes among matched pairs with predictions from both models',y=.97,weight='bold')
    for ax,col,title in zip(axes,['onset_shift_days','absolute_error_delta_days'],['Onset-window midpoint: M1 − M0','Absolute interval error: M1 − M0']):
        for si,seed in enumerate(seeds):
            g=q[q.seed.eq(seed)].set_index('scope').loc[sites];y=np.arange(len(sites))+(si-.5)*.25
            v=g[col+'_median'];lo=g[col+'_q25'];hi=g[col+'_q75']
            ax.errorbar(v,y,xerr=np.vstack([v-lo,hi-v]),fmt='o',color='#0072B2' if si==0 else '#D55E00',capsize=3,label=f'Seed {seed}')
        ax.axvline(0,color='#555555',lw=.7);ax.set(yticks=range(len(sites)),yticklabels=sites,xlabel='Days')
        ax.invert_yaxis();ax.set_title(title,loc='left');ax.spines[['top','right']].set_visible(False)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
    save(fig,out,'03_paired_timing_changes',
         'Points: medians; bars: 25th–75th percentiles across matched A/B rows (not confidence intervals).\n'
         'Left: positive means later onset. Right: positive means larger error. Conditional summaries exclude pairs without both predictions.\n'
         'Read with the availability figure; these conditional timing differences do not replace a complete six-site error comparison.')

    idx=indices.copy().fillna({'parameter_2':''});idx['extreme_interval']=(~np.isfinite(idx.confidence_half_width))|(idx.confidence_half_width>EXTREME_CI)
    idx.to_csv(out/'paired_index_plot_audit.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    fig.subplots_adjust(left=.12,right=.97,top=.82,bottom=.17,hspace=.5,wspace=.38)
    fig.suptitle(f'What drives the matched accuracy difference? Sobol indices | N = {n}',y=.98,weight='bold')
    index_legend(fig,seeds)
    for col,t in enumerate(TOLERANCES):
        g=idx[(idx.base_n==n)&idx.scope.eq('ALL_SIX')&idx.response.eq(f'accuracy_delta_{t}d')]
        forest(axes[0,col],g[g.index_type.ne('S2')],list(COLORS),seeds)
        axes[0,col].set_title(f'{t}-day tolerance | S1 and ST',loc='left')
        pairs=[('VPD_w','I_req'),('VPD_w','u_tr'),('I_req','u_tr')]
        forest(axes[1,col],g[g.index_type.eq('S2')],pairs,seeds,('S2',))
        axes[1,col].set_title(f'{t}-day tolerance | S2',loc='left')
    save(fig,out,'04_paired_delta_sobol',
         'Response: within-tolerance fraction from M1 minus that from its exactly matched M0. Full Saltelli row order is retained.\n'
         'Bars: estimate ± 95% bootstrap half-width. Indices quantify variance in the structural contrast, not whether the gate is beneficial.\n'
         'No parameters are selected; index uncertainty and seed stability must be assessed before claiming a stable importance ordering.')

    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    fig.subplots_adjust(left=.08,right=.98,top=.8,bottom=.22,wspace=.3)
    fig.suptitle('Matched structural differences across predeclared wind-threshold bands',y=.97,weight='bold')
    for si,seed in enumerate(seeds):
        g=wind[wind.seed.eq(seed)].sort_values('wind_lower');x=(g.wind_lower+g.wind_upper)/2
        for t,c in [(7,'#0072B2'),(10,'#009E73')]:
            axes[0].plot(x,g[f'accuracy_delta_{t}d']*100,color=c,marker='o',ls='-' if si==0 else '--',label=f'{t} d / seed {si+1}')
        axes[1].plot(x,g.lost_prediction_rate*100,color='#BC6571',marker='o',ls='-' if si==0 else '--',label=f'Seed {si+1}')
    for ax in axes:
        ax.axhline(0,color='#555555',lw=.7);ax.set(xticks=np.arange(1.5,8),xticklabels=[f'{i}–{i+1}' for i in range(1,8)],xlabel='Wind threshold band (m/s)')
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('Accuracy change: M1 − M0 (percentage points)');axes[1].set_ylabel('Prediction lost after adding gate (%)')
    save(fig,out,'05_paired_wind_bands',
         'A/B rows only; fixed bands [1,2), …, [7,8). Comparisons within each pair are exact; common-parameter samples differ between bands.\n'
         'These descriptive band summaries are not an optimized threshold profile. They do not establish a biologically justified wind cutoff.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'result/fsim_paired_n128')
    parser.add_argument('--figure-dir',type=Path,default=ROOT/'fig/fsim_paired_n128')
    args=parser.parse_args();out=args.input_dir.resolve();figdir=args.figure_dir.resolve()
    metadata=json.loads((out/'run_metadata.json').read_text());source=Path(metadata['parent_run'])
    if metadata['status']!='complete':raise ValueError('Matched evaluations must finish first')
    for f,h in metadata['parent_results_sha256'].items():
        if hashlib.sha256((source/f).read_bytes()).hexdigest()!=h:raise ValueError(f'Parent result changed: {f}')
    mapping=pd.read_csv(out/'paired_design.csv',float_precision='round_trip')
    m0=pd.read_csv(out/'m0_predictions.csv',float_precision='round_trip')
    m1=pd.read_csv(source/'predictions.csv',float_precision='round_trip').query('model=="M1"')
    sites=pd.read_csv(source/'site_year_manifest.csv').input_key.tolist()
    parent=json.loads((source/'run_metadata.json').read_text());settings=dict(parent['settings'])
    problem=parent['problems']['M1'];stride=2*problem['num_vars']+2 if settings['second_order'] else problem['num_vars']+2
    pairs=build_pairs(mapping,m0,m1)
    pairs.to_csv(out/'paired_predictions.csv',index=False)
    summary,wind=summarize_pairs(pairs,sites,stride)
    summary.to_csv(out/'paired_summary.csv',index=False);wind.to_csv(out/'wind_band_summary.csv',index=False)
    responses=delta_responses(pairs,mapping,sites)
    responses.to_csv(out/'paired_response_vectors.csv',index=False)
    settings['models']=['M1_minus_M0']
    diag,indices=analyze_all(responses,{'M1_minus_M0':problem},settings,show_progress=True)
    diag.to_csv(out/'paired_response_diagnostics.csv',index=False);indices.to_csv(out/'paired_sobol_indices.csv',index=False)
    conv,stability=convergence_tables(indices,settings)
    conv.to_csv(out/'paired_convergence.csv',index=False);stability.to_csv(out/'paired_seed_stability.csv',index=False)
    figdir.mkdir(parents=True,exist_ok=True)
    plots(summary,wind,indices,settings,sites,figdir)
    lines=['# M0–M1 完全匹配的结构比较','',
           '每对模拟使用相同 VPD_w、I_req、站点年份、天气、物候和固定系数；M1 按已有 u_tr 开启门控，M0 关闭门控。',
           f'保留 M1 的 {len(mapping)} 行完整 Sobol 设计。只计算 {metadata["n_unique_common_rows"]} 组不重复的 M0 共同参数，随后按原始行号展开。',
           f'新增主评估 {metadata["n_new_main_evaluations"]} 次、重复天气策略复算 {metadata["n_new_duplicate_evaluations"]} 次；无程序错误，复算无差异。',
           '描述性主汇总仅使用 A/B 基础行，两个种子分别报告；完整 Saltelli 行的汇总另存于 paired_summary.csv。所有 Sobol 指数使用完整设计。',
           '模拟次数不是独立田间样本量；本比较隔离代码中的门控结构差异，不构成 LOSYO 验证或风速作用的田间因果证据。','',
           '| 种子 | 容许天数 | 匹配M0准确比例 | M1准确比例 | 净变化（百分点） | 改善比例 | 恶化比例 |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary[summary.sample_set.eq('base_AB')&summary.scope.eq('ALL_SIX')].to_dict('records'):
        for t in TOLERANCES:
            lines.append(f'| {r["seed"]} | {t} | {r[f"accuracy_{t}d_M0"]:.2%} | {r[f"accuracy_{t}d_M1"]:.2%} | {100*r[f"accuracy_delta_{t}d"]:+.2f} | {r[f"gained_accuracy_{t}d_rate"]:.2%} | {r[f"lost_accuracy_{t}d_rate"]:.2%} |')
    lines+=['','## 站点差异（两个种子的等权均值）','',
            '| 站点年份 | 7天净变化（百分点） | 10天净变化（百分点） | 丢失预测比例 |',
            '|---|---:|---:|---:|']
    base_summary=summary[summary.sample_set.eq('base_AB')]
    for site,g in base_summary[base_summary.scope.ne('ALL_SIX')].groupby('scope'):
        lines.append(f'| {site} | {100*g.accuracy_delta_7d.mean():+.2f} | {100*g.accuracy_delta_10d.mean():+.2f} | {g.availability_lost_rate.mean():.2%} |')
    ab=pairs[pairs.design_index.mod(stride).isin([0,stride-1])]
    lines+=['','## 准确性损失来自哪里','',
            '以下为两个种子合计的模拟配对记录数，不是独立田间样本数。']
    for t in TOLERANCES:
        both=ab.availability_transition.eq('both')
        gain=int((both&ab[f'accuracy_delta_{t}d'].eq(1)).sum())
        loss=int((both&ab[f'accuracy_delta_{t}d'].eq(-1)).sum())
        lost_prediction=int((ab.availability_transition.eq('lost')&ab[f'accuracy_delta_{t}d'].eq(-1)).sum())
        lines.append(f'- {t}天标准：双方均有预测时，新增准确记录 {gain}，丢失准确记录 {loss}；另有 {lost_prediction} 条原本准确的记录因 M1 无预测而丢失。')
    lines+=['','当前范围下，M0 的平均准确预测比例更高；7天与10天的差距大小不同。风速门控在部分站点改善结果，在其他站点恶化或抑制预测，不能被概括为普遍有益的附加结构。',
            '未来应优先解释这种站点差异，并核对高风速阈值的生物学与数据依据。不能据本次分带曲线选择最优阈值后再把同一批结果当作独立验证。']
    lines+=['','## 解释','',
            '准确性按绝对区间边界误差 ≤7/10 天定义；无预测不满足条件。原始 RMSE 不作补值。',
            'onset_shift_days>0 表示 M1 显症窗口中点更晚；absolute_error_delta_days>0 表示 M1 绝对误差更大。两者只在双方都有预测时定义，必须与预测丢失率一起解释。',
            '风速分带为预先固定的1 m/s宽度，仅描述结构差异随门控阈值的变化，不能据此选最优阈值或宣称范围已获生物学验证。',
            '准确比例差的 Sobol 指数量化结构差异的变异来源，不表示门控改善或恶化的方向。用配对差值判断方向，用指数分析变异贡献。',
            '新增7天分析另见 result/fsim_sobol_n128_accuracy7/ 和 fig/fsim_sobol_n128_accuracy7/。','',
            '图件：01 配对准确性增益/损失；02 预测产生/丢失；03 双方可预测时的时间变化；04 配对准确性差的 Sobol；05 固定风速分带。',
            '每图保存 PNG 和矢量 PDF。配对明细、摘要、完整响应、指数和收敛诊断保存在本结果目录。','']
    (out/'README.md').write_text('\n'.join(lines));(figdir/'README.md').write_text('\n'.join(lines))
    path=Path(__file__);dest=out/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    metadata['report_source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata['report_input_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [out/'m0_predictions.csv',out/'paired_design.csv']}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(summary[summary.sample_set.eq('base_AB')&summary.scope.eq('ALL_SIX')].to_string(index=False))
    print(out/'README.md')


if __name__=='__main__':main()
