"""Plot completed FSIM Sobol results; never rerun, filter or calibrate the model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/fsim_sobol_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'VPD_w': '#0072B2', 'I_req': '#D55E00', 'u_tr': '#009E73'}
LABELS = {'VPD_w': r'$VPD_w$', 'I_req': r'$I_{req}$', 'u_tr': r'$u_{tr}$'}
# Display safeguard only, not a statistical convergence criterion. Preserve all
# raw estimates and widths in plot_index_audit.csv and the original results.
EXTREME_CI = 1e6


def save(fig, out, name, caption):
    fig.text(.02, .015, caption, fontsize=9, va='bottom', linespacing=1.5)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{suffix}', dpi=300, facecolor='white')
    plt.close(fig)


def style_axis(ax):
    ax.axvline(0, color='#777777', lw=.7, zorder=0)
    ax.axvline(1, color='#bbbbbb', lw=.7, ls=':', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)


def forest(ax, frame, items, seeds, kinds=('S1', 'ST')):
    """Draw each seed separately; extreme/nonfinite intervals are labelled, not clipped."""
    offsets = np.linspace(-.25, .25, len(seeds) * len(kinds))
    any_extreme = False
    for j, item in enumerate(items):
        par, par2 = item if isinstance(item, tuple) else (item, '')
        for si, seed in enumerate(seeds):
            for ki, kind in enumerate(kinds):
                row = frame[(frame.seed == seed) & (frame.parameter == par)
                            & (frame.parameter_2 == par2) & (frame.index_type == kind)]
                if row.empty:
                    continue
                if len(row) != 1:
                    raise ValueError('Duplicate index identity in forest plot')
                r = row.iloc[0]
                y = j + offsets[si * len(kinds) + ki]
                color = COLORS[par]
                bad = bool(r.extreme_interval)
                any_extreme |= bad
                marker = 'o' if kind in ('S1', 'S2') else 's'
                if not np.isfinite(r.estimate):
                    continue
                kwargs = dict(color=color, marker=marker, markersize=5, linestyle='none',
                              markerfacecolor=color if si == 0 else 'white', markeredgewidth=1.1)
                if bad:
                    ax.plot(r.estimate, y, **kwargs)
                    ax.plot(r.estimate, y, marker='x', color='#9C3D10',
                            markersize=3, linestyle='none', markeredgewidth=.8)
                else:
                    ax.errorbar(r.estimate, y, xerr=r.confidence_half_width,
                                capsize=2, elinewidth=.9, **kwargs)
    ax.set_yticks(range(len(items)), [
        f'{LABELS[x[0]]} × {LABELS[x[1]]}' if isinstance(x, tuple) else LABELS[x]
        for x in items])
    ax.set_ylim(len(items) - .5, -.5)
    ax.set_xlabel('Sobol index')
    style_axis(ax)
    if any_extreme:
        widths = frame.loc[frame.extreme_interval, 'confidence_half_width']
        finite = widths[np.isfinite(widths)]
        label = ('Extreme bootstrap intervals omitted\n'
                 + (f'half-widths: {finite.min():.1e}–{finite.max():.1e}' if len(finite)
                    else 'non-finite half-widths'))
        ax.text(.02, .98, label, transform=ax.transAxes, va='top', fontsize=8,
                color='#9C3D10', bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
        ax.set_ylim(len(items) - .5, -1.0)


def index_legend(fig, seeds, timing=True):
    handles = [Line2D([], [], color='#333333', marker='o', ls='none', label='S1' if timing else 'S2')]
    if timing:
        handles.append(Line2D([], [], color='#333333', marker='s', ls='none', label='ST'))
    handles += [Line2D([], [], color='#333333', marker='o', ls='none',
                       markerfacecolor='#333333' if i == 0 else 'white',
                       label=f'Seed {i+1}: {seed}') for i, seed in enumerate(seeds)]
    handles.append(Line2D([], [], color='#9C3D10', marker='x', ls='none', label='Extreme interval: raw point only'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94),
               ncol=len(handles), frameon=False, fontsize=8)


def availability(responses, diagnostics, sites, settings, out):
    seeds = settings['sampling_seeds']; n = settings['base_n']
    groups = [(m, s) for m in settings['models'] for s in seeds]
    labels = [f'{m} / seed {seeds.index(s)+1}' for m, s in groups]
    records = []
    for m, s in groups:
        g = responses[(responses.model == m) & (responses.seed == s)]
        for scope in sites + ['ALL_SIX']:
            count = (int(g[f'{scope}|has_prediction'].eq(0).sum()) if scope != 'ALL_SIX'
                     else int(g['ALL_SIX|RMSE_days'].isna().sum()))
            records.append(dict(model=m, seed=s, scope=scope, n_parameter_rows=len(g),
                                n_missing=count, missing_percent=100*count/len(g)))
    table = pd.DataFrame(records)
    table.to_csv(out/'prediction_availability_summary.csv', index=False)
    a = np.array([[table[(table.model == m)&(table.seed == s)&(table.scope == scope)]
                   .missing_percent.iloc[0] for m,s in groups] for scope in sites+['ALL_SIX']])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), gridspec_kw={'width_ratios':[1.15,1]})
    fig.subplots_adjust(left=.14, right=.98, bottom=.23, top=.84, wspace=.55)
    fig.suptitle(f'Prediction availability and timing-response coverage | N = {n}', y=.97, weight='bold')
    im=axes[0].imshow(a, cmap='YlOrRd', vmin=0, vmax=max(30, a.max()), aspect='auto')
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            axes[0].text(x,y,f'{a[y,x]:.1f}%',ha='center',va='center',fontsize=9,
                         color='white' if a[y,x]>20 else '#222222')
    axes[0].set(xticks=range(len(groups)), xticklabels=labels,
                yticks=range(len(sites)+1), yticklabels=sites+['Any of six missing'])
    axes[0].tick_params(axis='x',rotation=30)
    axes[0].set_title('(a) Missing predictions (% of design rows)',loc='left',fontsize=10)
    cb=fig.colorbar(im,ax=axes[0],fraction=.045,pad=.03);cb.set_label('Missing (%)')
    response_keys=[('ALL_SIX','RMSE_days'),('ALL_SIX','MAE_days')]+[(s,'absolute_error_days') for s in sites]
    statuses=[]
    for scope,response in response_keys:
        row=[]
        for m,s in groups:
            d=diagnostics[(diagnostics.model==m)&(diagnostics.seed==s)&(diagnostics.base_n==n)
                          &(diagnostics.scope==scope)&(diagnostics.response==response)].iloc[0]
            row.append(0 if d.status=='ok' else 1 if d.status=='unavailable_missing_predictions' else 2)
        statuses.append(row)
    b=np.array(statuses)
    axes[1].imshow(b,cmap=ListedColormap(['#A4D7CC','#EDD2C7','#D9D9D9']),vmin=0,vmax=2,aspect='auto')
    for y in range(b.shape[0]):
        for x in range(b.shape[1]):
            axes[1].text(x,y,['E','M','U'][b[y,x]],ha='center',va='center')
    axes[1].set(xticks=range(len(groups)),xticklabels=labels,yticks=range(len(response_keys)),
                yticklabels=['Six-site RMSE','Six-site MAE']+sites)
    axes[1].tick_params(axis='x',rotation=30)
    axes[1].set_title('(b) Timing Sobol estimation status',loc='left',fontsize=10)
    save(fig,out,'01_response_availability',
         'E: estimate available (not proof of convergence); M: missing response, no Sobol decomposition; U: undefined/unstable.\n'
         'Site-year rows in (b) refer to absolute timing error. One missing site-year makes aggregate RMSE and MAE undefined.\n'
         'Percentages describe the Saltelli design; rows are dependent model evaluations, not independent field observations.')
    return table


def aggregate(indices, settings, out):
    n=settings['base_n']; seeds=settings['sampling_seeds']
    q=indices[(indices.base_n==n)&(indices.scope=='ALL_SIX')&(indices.response=='prediction_rate')]
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    fig.subplots_adjust(left=.10,right=.97,top=.82,bottom=.17,hspace=.55,wspace=.40)
    fig.suptitle(f'Parameter contributions to prediction-rate variance | N = {n}',y=.98,weight='bold')
    index_legend(fig,seeds)
    for col,model in enumerate(settings['models']):
        g=q[q.model==model];pars=list(COLORS)[:2] if model=='M0' else list(COLORS)
        forest(axes[0,col],g[g.index_type!='S2'],pars,seeds)
        axes[0,col].set_title(f'({chr(97+col)}) {model}: first and total effects',loc='left',fontsize=11)
        pairs=[(a,b) for j,a in enumerate(pars) for b in pars[j+1:]]
        forest(axes[1,col],g[g.index_type=='S2'],pairs,seeds,('S2',))
        axes[1,col].set_title(f'({chr(99+col)}) {model}: pairwise interactions (S2)',loc='left',fontsize=11)
    save(fig,out,'02_prediction_rate_sobol',
         'Response: fraction of the six site-years with a prediction. Bars: estimate ± 95% bootstrap half-width.\n'
         'Crosses retain raw estimates with extreme intervals; they do not support a reliable sensitivity ranking. No indices are clipped.\n'
         'Each model has its own variance denominator. These indices do not quantify error reduction or model superiority.')


def site_availability(indices, diagnostics, sites, settings, out):
    n=settings['base_n'];seeds=settings['sampling_seeds']
    q=indices[(indices.base_n==n)&(indices.response=='has_prediction')&(indices.index_type=='ST')]
    fig,axes=plt.subplots(len(sites),2,figsize=(12,17))
    fig.subplots_adjust(left=.10,right=.97,top=.92,bottom=.07,hspace=.95,wspace=.35)
    fig.suptitle(f'Site-year prediction availability: total effects (ST) | N = {n}',y=.98,weight='bold')
    handles=[Line2D([],[],color='#333333',marker='s',ls='none',
                    markerfacecolor='#333333' if i==0 else 'white',label=f'Seed {i+1}: {s}')
             for i,s in enumerate(seeds)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.963),ncol=2,frameon=False)
    for row,site in enumerate(sites):
        for col,model in enumerate(settings['models']):
            ax=axes[row,col];g=q[(q.model==model)&(q.scope==site)]
            pars=list(COLORS)[:2] if model=='M0' else list(COLORS)
            forest(ax,g,pars,seeds,('ST',))
            ax.set_title(f'{model} | {site}',loc='left',fontsize=10)
            absent=[str(i+1) for i,s in enumerate(seeds) if g[g.seed==s].empty]
            if absent:
                ax.text(.98,.05,'No ST: seed '+', '.join(absent)+'\nSee response diagnostics',
                        transform=ax.transAxes,ha='right',fontsize=8,color='#666666')
    save(fig,out,'03_site_prediction_sobol',
         'Binary response: prediction present (1) or absent (0). Bars: estimate ± 95% bootstrap half-width; panels use independent x scales.\n'
         'Crosses: extreme intervals omitted and explicitly annotated. Constant responses have undefined ST, not zero sensitivity.\n'
         'ST includes interactions; estimates above one are retained. Uncertainty and rare missing outcomes preclude some rankings.')


def timing(indices, diagnostics, settings, out):
    n=settings['base_n'];seeds=settings['sampling_seeds']
    q=indices[(indices.base_n==n)&(indices.response=='absolute_error_days')&(indices.index_type!='S2')]
    panels=list(q[['model','scope']].drop_duplicates().itertuples(index=False,name=None))
    if not panels:
        return
    fig,axes=plt.subplots(1,len(panels),figsize=(6*len(panels),5.5),squeeze=False)
    fig.subplots_adjust(left=.09,right=.97,top=.77,bottom=.25,wspace=.38)
    fig.suptitle(f'Absolute timing-error sensitivity: estimable site-years only | N = {n}',y=.98,weight='bold')
    index_legend(fig,seeds)
    for ax,(model,scope) in zip(axes[0],panels):
        g=q[(q.model==model)&(q.scope==scope)]
        forest(ax,g,list(COLORS)[:2] if model=='M0' else list(COLORS),seeds)
        ax.set_title(f'{model} | {scope}',loc='left',fontsize=11)
        missing=[str(i+1) for i,s in enumerate(seeds) if g[g.seed==s].empty]
        if missing:
            ax.text(.98,.97,'Seed '+', '.join(missing)+': unavailable\nMissing timing responses',
                    transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#9C3D10')
    save(fig,out,'04_timing_error_sobol',
         'Response: absolute distance between predicted and observed onset intervals (days). Bars: estimate ± 95% bootstrap half-width.\n'
         'Only complete designs are plotted; no incomplete rows are removed. Missing panels are not evidence of weak parameter effects.\n'
         'Six-site RMSE/MAE indices are unavailable for both models. These site-specific estimates cannot substitute for aggregate error sensitivity.')


def convergence(indices, settings, out):
    q=indices[(indices.scope=='ALL_SIX')&(indices.response=='prediction_rate')&(indices.index_type.isin(['S1','ST']))]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fig.subplots_adjust(left=.08,right=.97,top=.82,bottom=.17,hspace=.36,wspace=.28)
    fig.suptitle('Prediction-rate Sobol estimates: sample-size and seed stability',y=.98,weight='bold')
    handles=[Line2D([],[],color=c,label=LABELS[p]) for p,c in COLORS.items()]
    handles += [Line2D([],[],color='#333333',ls='-' if i==0 else '--',marker='o',
                        label=f'Seed {i+1}: {s}') for i,s in enumerate(settings['sampling_seeds'])]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False,fontsize=9)
    for col,model in enumerate(settings['models']):
        for row,kind in enumerate(['S1','ST']):
            ax=axes[row,col]; g=q[(q.model==model)&(q.index_type==kind)]
            for (par,seed),block in g.groupby(['parameter','seed']):
                b=block.sort_values('base_n');si=settings['sampling_seeds'].index(seed)
                ax.plot(b.base_n,b.estimate,color=COLORS[par],marker='o',ms=4,
                        ls='-' if si==0 else '--',mfc=COLORS[par] if si==0 else 'white')
            ax.axhline(0,color='#777777',lw=.7);ax.axhline(1,color='#bbbbbb',ls=':',lw=.7)
            ax.set(xticks=settings['checkpoints'],xlabel='Base sample size N',ylabel=kind)
            ax.set_title(f'{model} | {kind}',loc='left',fontsize=11)
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    save(fig,out,'05_convergence',
         'Points are raw estimates from nested prefixes; seeds are shown separately. Axis ranges vary; no estimates are clipped.\n'
         'Point-estimate stability alone is insufficient: compare bootstrap uncertainty and seed differences as well.\n'
         'M0 has extreme bootstrap intervals (see Figure 02); these curves do not establish convergence or a reliable ranking.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'result/fsim_sobol_n128')
    p.add_argument('--output-dir',type=Path,default=ROOT/'fig/fsim_sobol_n128')
    args=p.parse_args();source=args.input_dir.resolve();out=args.output_dir.resolve()
    metadata=json.loads((source/'run_metadata.json').read_text());settings=metadata['settings']
    if metadata['status']!='complete':
        raise ValueError('Only a completed analysis can be plotted')
    if settings['models']!=['M0','M1'] or len(settings['sampling_seeds'])!=2:
        raise ValueError('This figure layout requires M0/M1 and two sampling seeds')
    frames={name:pd.read_csv(source/f'{name}.csv') for name in [
        'sobol_design','predictions','duplicate_policy_predictions','duplicate_policy_differences',
        'response_vectors','response_diagnostics','sobol_indices','site_year_manifest','convergence','seed_stability']}
    design=frames['sobol_design'];pred=frames['predictions'];responses=frames['response_vectors']
    if len(pred)!=metadata['n_main_evaluations'] or len(frames['duplicate_policy_predictions'])!=metadata['n_duplicate_evaluations']:
        raise ValueError('Evaluation counts do not match metadata')
    if len(frames['duplicate_policy_differences']):
        raise ValueError('Resolve duplicate-weather-policy differences before interpreting figures')
    if not pred.status.isin(['predicted','no_prediction']).all():
        raise ValueError('Runtime errors present in predictions')
    if pred.duplicated(['sample_id','input_key']).any() or not pred.groupby('sample_id').size().eq(6).all():
        raise ValueError('Each parameter row must have exactly six unique site-year evaluations')
    if not responses.sample_id.equals(design.sample_id) or not set(pred.sample_id)==set(design.sample_id):
        raise ValueError('Design and response identifiers do not agree')
    indices=frames['sobol_indices'].fillna({'parameter_2':''})
    indices['extreme_interval']=(~np.isfinite(indices.confidence_half_width)) | (indices.confidence_half_width>EXTREME_CI)
    out.mkdir(parents=True,exist_ok=True)
    indices.to_csv(out/'plot_index_audit.csv',index=False)
    sites=frames['site_year_manifest'].input_key.tolist()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                         'pdf.fonttype':42,'ps.fonttype':42})
    summary=availability(responses,frames['response_diagnostics'],sites,settings,out)
    aggregate(indices,settings,out)
    site_availability(indices,frames['response_diagnostics'],sites,settings,out)
    timing(indices,frames['response_diagnostics'],settings,out)
    convergence(indices,settings,out)
    audit={'input_directory':str(source),'settings':settings,
           'n_main_evaluations':len(pred),'n_duplicate_evaluations':len(frames['duplicate_policy_predictions']),
           'runtime_errors':0,'duplicate_policy_differences':0,
           'extreme_interval_display_threshold':EXTREME_CI,
           'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.csv')},
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plot_metadata.json').write_text(json.dumps(audit,indent=2))
    lines=['# Sobol N='+str(settings['base_n'])+' 结果图与解读','',
           '输入：`'+str(source)+'`。本脚本只读取已完成结果，不重新运行模型或寻优。',
           f'完成 {len(pred):,} 次主评估和 {len(frames["duplicate_policy_predictions"]):,} 次重复天气策略复算；无程序错误，天气策略比较无差异。','',
           '## 图件','',
           '- `01_response_availability`：无预测比例及误差响应能否进行 Sobol 分解。',
           '- `02_prediction_rate_sobol`：六站点预测产生率的 S1、ST、S2，分种子显示。',
           '- `03_site_prediction_sobol`：逐站点年份预测产生与否的 ST。',
           '- `04_timing_error_sobol`：有完整误差响应的站点年份 S1、ST；不替代汇总 RMSE 分析。',
           '- `05_convergence`：N 前缀与独立种子的 S1、ST 变化。',
           '每图同时保存 300 dpi PNG 和矢量 PDF；参数顺序固定，不按估计值排序。','',
           '## 完整预测覆盖','',
           '| 模型 | 种子 | 设计行数 | 至少一个站点无预测 | 完整六站点比例 |',
           '|---|---:|---:|---:|---:|']
    for r in summary[summary.scope=='ALL_SIX'].itertuples():
        lines.append(f'| {r.model} | {r.seed} | {r.n_parameter_rows} | {r.n_missing} | {100-r.missing_percent:.2f}% |')
    lines += ['', '## 解释限制','',
              '两个模型、两个种子的汇总 RMSE 和 MAE 都存在缺失，不能得到本参数域的标准 Sobol 误差分解。没有删除缺失组合、缩小参数域或赋予惩罚误差。',
              '原分析的 status=ok 仅通过有限值检查。复核发现有些 bootstrap 半宽虽有限却达到约 1e28–1e33，不能当作可靠区间；尤其影响 M0 预测产生率，以及 M1 的 NN_JF_2023 预测产生与否。',
              '绘图以半宽 >1e6 或非有限值作为显示保护标记（不是新的统计收敛阈值）：保留原始点并画叉号，明确注明未绘制的区间量级。完整值和标记保存在 plot_index_audit.csv；原始结果文件未修改。',
              '普通误差棒为估计值 ± SALib 输出的 95% bootstrap 半宽，表示指数估计不确定性，不表示田间泛化误差。指数不裁剪至 [0,1]，ST 不要求相加为 1。',
              'M1 汇总预测产生率中 u_tr 的 ST 在两个种子下均最高，但置信区间较宽；不能将其表述为误差最重要参数或稳定的完整排序。S2 的区间跨零，不能据点估计确认具体交互。',
              'M0 仅 GL_WK_2022 的绝对误差在两个种子均可估计；NN_JF_2024 仅一个种子可估计。N=128 的误差敏感性证据不完整，预测产生率的稳定性也不足以支持确定排名。',
              '本分析条件于其他参数固定和六个站点年份；M0/M1 的 Sobol 方差分母不同，不能用指数大小判断模型优劣。图中比例是 Saltelli 设计的描述性比例，不是独立重复试验的成功率。','',
              '复现：`dm_first_os/bin/python src/plot_sobol_fsim.py --input-dir result/fsim_sobol_n128 --output-dir fig/fsim_sobol_n128`']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Validated {len(pred):,} main evaluations; wrote figures to {out}')


if __name__=='__main__':
    main()
