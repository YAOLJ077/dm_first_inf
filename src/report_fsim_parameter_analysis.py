#!/usr/bin/env python3
"""Plot and report the completed FSIM response-grid analysis (no fitting)."""
from __future__ import annotations
import argparse
import json
import os
import tempfile
import shutil
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'fsim-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'fsim-cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def panel(ax, frame, column, vmax, title, cmap='viridis', text_column=None, vmin=0):
    matrix=frame.pivot(index='I_req',columns='VPD_w',values=column).sort_index()
    x,y=matrix.columns.to_numpy(),matrix.index.to_numpy()
    # Cells show evaluated grid nodes; no fitted interpolation is implied.
    image=ax.pcolormesh(x,y,np.ma.masked_invalid(matrix.to_numpy(float)),
                        shading='nearest',vmin=vmin,vmax=max(vmax,vmin+1e-9),cmap=cmap)
    ax.set_facecolor('#dedede')
    ax.set(title=title,xlabel='VPD_w (hPa)',ylabel='I_req (°C h)')
    if text_column:
        n=frame.pivot(index='I_req',columns='VPD_w',values=text_column).reindex(index=y,columns=x)
        for j, yy in enumerate(y):
            for i, xx in enumerate(x):
                if n.iloc[j,i] < 6:
                    ax.text(xx,yy,str(int(n.iloc[j,i])),ha='center',va='center',fontsize=6,color='red')
    return image


def plot_atlas(predictions, summary, out):
    grid=predictions[predictions.design=='grid']
    s=summary[summary.design=='grid']
    winds=sorted(s.loc[s.model=='M1','u_tr'].unique())
    choices=[('M0',None)]+[('M1',wind) for wind in winds]
    vmax=float(grid.interval_distance_days.max())
    with PdfPages(out/'site_year_response_atlas.pdf') as pdf:
        for model,wind in choices:
            subset=grid[grid.model==model]
            if wind is not None: subset=subset[np.isclose(subset.u_tr,wind)]
            fig,axes=plt.subplots(2,3,figsize=(11.7,7.5),layout='constrained')
            for ax,(key,block) in zip(axes.flat,subset.groupby('input_key',sort=True)):
                im=panel(ax,block,'interval_distance_days',vmax,key)
                missing=block[block.status!='predicted']
                ax.scatter(missing.VPD_w,missing.I_req,marker='x',color='black',s=15)
            title=model if wind is None else f'{model}: u_tr = {wind:g} m/s'
            fig.suptitle(f'{title} | observed-interval distance (days)\nGrey cells with ×: no prediction; descriptive full-data analysis')
            fig.colorbar(im,ax=list(axes.flat),label='Absolute interval error (days)',shrink=.8)
            pdf.savefig(fig);plt.close(fig)
    ncols=min(5,len(choices));nrows=int(np.ceil(len(choices)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(3.2*ncols,3.25*nrows),layout='constrained',squeeze=False)
    vmax=float(s.all_site_year_mae_days.max())
    for ax,(model,wind) in zip(axes.flat,choices):
        block=s[s.model==model]
        if wind is not None: block=block[np.isclose(block.u_tr,wind)]
        title=model if wind is None else f'M1: u_tr={wind:g}'
        im=panel(ax,block,'all_site_year_mae_days',vmax,title,text_column='n_predictions',vmin=float(s.all_site_year_mae_days.min()))
    for ax in list(axes.flat)[len(choices):]: ax.set_visible(False)
    fig.suptitle('Response surfaces: MAE only where all six site-years have predictions\nGrey cells: incomplete coverage; red labels: number of predictions (out of six)')
    fig.colorbar(im,ax=list(axes.flat),label='All-six MAE (days)',shrink=.8)
    fig.savefig(out/'response_surfaces.png',dpi=180)
    fig.savefig(out/'response_surfaces.pdf');plt.close(fig)


def plot_profiles(profiles, effects, near, out):
    names=['VPD_w','I_req','u_tr']
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.5),layout='constrained')
    for i,model in enumerate(['M0','M1']):
        for j,name in enumerate(names):
            ax=axes[i,j];block=profiles[(profiles.model==model)&(profiles.parameter==name)].sort_values('value')
            if block.empty: ax.set_visible(False);continue
            comparable=block.comparable_to_global_best.astype(bool)
            ax.plot(block.value,block.profile_mae_days.where(comparable),'o-',color='#245e91')
            ax.scatter(block.loc[~comparable,'value'],block.loc[~comparable,'profile_mae_days'],marker='x',color='red',label='Lower prediction coverage')
            ax.set(title=f'{model}: {name}',xlabel=name,ylabel='Profile interval MAE (days)')
            if (~comparable).any(): ax.legend(fontsize=7,loc='upper left')
            twin=ax.twinx();twin.plot(block.value,block.max_n_predictions,':',color='#b46a37',alpha=.7)
            twin.set(ylim=(-.2,6.3),yticks=[0,3,6],ylabel='Predictions / 6')
    fig.suptitle('Grid profiles: best error at each parameter value, prioritizing prediction coverage\nDescriptive profiles; no likelihood or confidence interval interpretation')
    fig.savefig(out/'parameter_profiles.png',dpi=180);fig.savefig(out/'parameter_profiles.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    chosen=effects[(effects.response=='interval_distance_days')&(effects.scope!='ALL_SIX')]
    limit=chosen.mean_abs_effect.max()
    for ax,model in zip(axes,['M0','M1']):
        matrix=chosen[chosen.model==model].pivot(index='scope',columns='parameter',values='mean_abs_effect')
        matrix=matrix.reindex(columns=[n for n in names if n in matrix.columns])
        im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=limit,cmap='YlGnBu',aspect='auto')
        ax.set(xticks=range(len(matrix.columns)),xticklabels=matrix.columns,
               yticks=range(len(matrix.index)),yticklabels=matrix.index,title=model)
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                v=matrix.iloc[i,j];ax.text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v>limit*.55 else 'black')
    fig.suptitle('Mean absolute adjacent-grid effect on interval error\nDays per unit normalized parameter; valid prediction pairs only (not Morris/Sobol)')
    fig.colorbar(im,ax=list(axes),label='Mean absolute effect (days)',shrink=.8)
    fig.savefig(out/'grid_sensitivity.png',dpi=180);fig.savefig(out/'grid_sensitivity.pdf');plt.close(fig)
    if not near.empty:
        fig,axes=plt.subplots(1,3,figsize=(11.5,3.8),layout='constrained')
        m1=near[(near.model=='M1')&(near.tolerance_days==1)]
        for ax,(x,y) in zip(axes,[('VPD_w','I_req'),('VPD_w','u_tr'),('I_req','u_tr')]):
            im=ax.scatter(m1[x],m1[y],c=m1.delta_mae_days,vmin=0,vmax=1,cmap='viridis',s=40,edgecolors='black',linewidths=.3)
            ax.set(xlabel=x,ylabel=y)
        fig.suptitle('M1 near-optimal grid sets: within 1 day of best MAE at maximum coverage\nPairwise projections; overlapping points are not probability densities')
        fig.colorbar(im,ax=list(axes),label='MAE above grid minimum (days)')
        fig.savefig(out/'near_optimal_M1.png',dpi=180);fig.savefig(out/'near_optimal_M1.pdf');plt.close(fig)


def report(predictions, summary, profiles, ranges, effects, metadata, out):
    lines=['# FSIM 参数响应与可辨识性诊断','',
           '本报告为六个站点年份的全数据描述性网格分析，不是校准验收、结构选择或 LOSYO 验证。', '',
           '## 配置与设计','',
           f'- 本次配置表：`{metadata["arguments"]["config_table"]}`；本次快照：`parameter_table.csv`。',
           '- u_tr 范围为结合已有风速数据设置的探索区间，不是生物学置信区间；0 为与 M0 的一致性对照，高端为稀少转移压力测试。',
           '- 原始风速请求元数据仍不完整；沿用项目既有 m/s 数值核查结论。气象输入与冠层风速的对应关系仍属输入不确定性。',
           '- 首先比较产生预测的站点年份数量，再比较观测显症区间到预测窗口的平均距离；只在六个站点年份均有预测时提供 all-six MAE。',
           '- 首次阳性日期距离另列，不用任意惩罚值替代无预测。观测区间采用 (L,U]，仅接触 L 时距离为零但不计区间重叠。','',
           '## 网格结果','',
           '| 版本 | 网格组合数 | 六站点年份均有预测的组合数 | 最优组合的预测数 | 最小区间 MAE（天） |',
           '|---|---:|---:|---:|---:|']
    design_lines=[]
    for model, block in summary[summary.design=='grid'].groupby('model'):
        dimensions=[]
        for parameter in ('VPD_w','I_req','u_tr'):
            values=block[parameter].dropna()
            if not values.empty:
                dimensions.append(f'{parameter}: {values.min():g}–{values.max():g}（{values.nunique()}点）')
        design_lines.append(f'- {model}：'+ '；'.join(dimensions)+f'；共 {len(block)} 个网格组合。')
    pos=lines.index('## 配置与设计')+2
    lines[pos:pos]=design_lines
    for model,block in summary[summary.design=='grid'].groupby('model'):
        maxn=block.n_predictions.max();best=block[block.n_predictions==maxn].conditional_mae_days.min()
        lines.append(f'| {model} | {len(block)} | {int((block.n_predictions==6).sum())} | {maxn} | {best:.3f} |')
    lines+=['','上表是全数据网格最小误差，不能作为泛化性能，也不能据此认定风速版本优于无风速版本。','',
            '## 近优集合与参数约束','',
            '以下采用“与本网格最小误差相差不超过 1 天，且预测数量达到本版本最大值”的描述性集合。另输出 0、0.5 和 2 天阈值；均不是统计置信区间。','',
            '| 版本 | 参数 | 近优值范围 | 不同网格取值数 | 接触分析下界 | 接触分析上界 |',
            '|---|---|---|---:|---|---|']
    for r in ranges[ranges.tolerance_days==1].itertuples(index=False):
        lines.append(f'| {r.model} | {r.parameter} | {r.min_retained:g}–{r.max_retained:g} | {r.n_distinct_values} | {r.touches_lower} | {r.touches_upper} |')
    lines+=['','应同时检查网格最小误差及逐站点年份的偏早/偏晚情况，复核输入、观察终点与模型输出之间的关系，再决定训练折内校准方案。近优集合含多个风速值时，不能据此认定唯一阈值。', '', '这些范围是离散保留值的最小值与最大值，内部不一定连续。范围宽、多个组合输出相同、或接触边界，提示现有数据约束有限；不能据此声称结构性不可辨识，也不能直接扩大参数范围。','',
            '## 网格敏感性','',
            '对每个站点年份沿各参数方向计算相邻网格差分，并除以归一化参数步长；同时输出平均绝对效应、效应标准差、零效应比例、有效边数和预测状态转换数。该指标依赖本轮范围及网格分布，不是 Morris 抽样结果，也不是 Sobol 方差贡献率。缺失预测的时间差分保持缺失。','']
    for model in ['M0','M1']:
        e=effects[(effects.model==model)&(effects.scope=='ALL_SIX')&(effects.response=='all_site_year_mae_days')].sort_values('mean_abs_effect',ascending=False)
        lines.append(f'- {model} 的六站点年份 MAE 网格平均绝对效应排序：'+ ' > '.join(f'{r.parameter}（{r.mean_abs_effect:.2f}）' for r in e.itertuples())+'。')
    lines+=['','## 输入与实现检查','',
            f'- 完成 {len(predictions)} 次主分析站点年份评估（包含 12 次默认值对照）；程序错误 {int((predictions.status=="runtime_error").sum())} 次。',
            f'- M1 在 u_tr=0 时与 M0 的一致性：{metadata["zero_gate_equivalence"]}。',
            '- DA_MPT_2024、NN_JF_2024 各有 24 个重复的 12 月 31 日时间戳，记录值不同。原始文件未修改；主分析保留末次记录，并对全部参数组合使用保留首次记录做复算。',
            f'- 重复记录策略对照：{metadata.get("duplicate_policy_check",{})}。该检查仅说明本次首次显症输出是否受影响，不证明任一重复记录是真值。',
            '- 模型固定关系的数值未重新拟合；缓存实现已与公开模型入口核对。完整配置、源代码快照和输入 SHA-256 保存在结果目录。','',
            '## 后续使用','',
            '本轮结果用于定位响应平坦区、参数补偿及边界问题。正式参数筛选、M0/M1 选择和校准必须限制在 LOSYO 训练折内部；不得将本网格最优值直接带入留出站点年份而称为独立验证。FSIM-S 尚未同步的过程不属于本次分析。',
            '', '网格分辨率有限，剖面并非连续最优解；后续局部细化也应在训练折内进行。当前固定参数采用既定默认关系，其不确定性尚未在此传播。','',
            '## 输出文件','',
            '- `predictions.csv`：逐组合、逐站点年份的预测和终点评价。',
            '- `sample_metrics.csv`：预测完整性、误差、窗口宽度。',
            '- `parameter_profiles.csv`、`near_optimal_sets.csv`、`near_optimal_ranges.csv`：剖面与等效参数集合。',
            '- `grid_elementary_effects.csv`：逐站点年份及汇总敏感性。',
            '- `response_surfaces.png`、`parameter_profiles.png`、`grid_sensitivity.png`：主要图。',
            '- `site_year_response_atlas.pdf`：各风速切片下六站点年份的全部响应面。',
            '- `duplicate_policy_predictions.csv`、`duplicate_policy_comparison.csv`：重复天气输入策略复算。','']
    (out/'analysis_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir',type=Path,default=ROOT/'result/fsim_parameter_analysis_v1')
    args=p.parse_args();out=args.result_dir
    metadata=json.loads((out/'run_metadata.json').read_text())
    if metadata['status']!='complete': raise SystemExit('Analysis must complete before reporting')
    predictions=pd.read_csv(out/'predictions.csv')
    summary=pd.read_csv(out/'sample_metrics.csv')
    profiles=pd.read_csv(out/'parameter_profiles.csv')
    ranges=pd.read_csv(out/'near_optimal_ranges.csv')
    near=pd.read_csv(out/'near_optimal_sets.csv')
    effects=pd.read_csv(out/'grid_elementary_effects.csv')
    plot_atlas(predictions,summary,out)
    plot_profiles(profiles,effects,near,out)
    report(predictions,summary,profiles,ranges,effects,metadata,out)
    shutil.copy2(__file__,out/'source_snapshot/src'/Path(__file__).name)
    print(out/'analysis_report.md')


if __name__=='__main__': main()
