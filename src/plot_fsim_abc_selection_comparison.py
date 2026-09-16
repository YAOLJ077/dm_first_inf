#!/usr/bin/env python3
"""Plot saved A/B/C results: training minimum versus test-selected tolerance.

Three rows: LOSYO, leave-site-out, leave-year-out. No model execution.
Each arm must have both complete exported selection tables, even if other arms
of the overall run are still in progress.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMES = ('losyo', 'site', 'year')
LABELS = {'losyo': 'LOSYO', 'site': 'Leave-one-site-out', 'year': 'Leave-one-year-out'}
STRATEGIES = (
    ('minimum', 'Minimum training MAE', 'oof_predictions.csv', 'fold_summary.csv', False),
    ('tolerance', 'Training minimum + 1 day; minimum test MAE',
     'exploratory_predictions.csv', 'exploratory_fold_summary.csv', True),
)


def load(results, arm, seed, budget):
    hashes = {}
    def read(name):
        content = (results / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        return pd.read_csv(io.BytesIO(content), dtype={'held_out': str}, float_precision='round_trip')
    manifest = read('site_year_manifest.csv')
    keys = manifest.input_key.tolist()
    if len(keys) != 6 or len(set(keys)) != 6:
        raise ValueError('Expected six distinct site-years')
    expected = {
        'losyo': {k: [k] for k in keys},
        'site': {str(k): g.input_key.tolist() for k, g in manifest.groupby('place_code')},
        'year': {str(k): g.input_key.tolist() for k, g in manifest.groupby('Year')},
    }
    output, case_rows = [], []
    for strategy, _, predictions_file, folds_file, uses_test in STRATEGIES:
        p, f = read(predictions_file), read(folds_file)
        filters = lambda x: x.arm.eq(arm) & x.seed.eq(seed) & x.n_trials.eq(budget)
        p, f = p[filters(p)].copy(), f[filters(f)].copy()
        if len(p) != 18 or len(f) != 12:
            # Per-fold exports may be available before the root summary is
            # refreshed or synchronized. Read all folds from one source level;
            # never mix incomplete root summaries with individual fold rows.
            predictions, summaries = [], []
            for scheme, fold_keys in expected.items():
                for held in fold_keys:
                    directory = Path(arm)/scheme/f'seed_{seed}'/'folds'/held/f'budget_{budget}'
                    if uses_test:
                        directory /= 'exploratory'
                    selection_name = str(directory/'selection.json')
                    content = (results/selection_name).read_bytes()
                    hashes[selection_name] = hashlib.sha256(content).hexdigest()
                    selection = json.loads(content)
                    if selection['n_trials'] != budget or selection['selection_uses_test_data'] != uses_test:
                        raise ValueError(f'Incorrect per-fold selection metadata: {directory}')
                    tag = dict(arm=arm,scheme=scheme,held_out=held,seed=seed,n_trials=budget)
                    summary = {**selection, **tag}
                    for key in ('training_keys','test_keys'):
                        summary[key] = repr(summary[key])
                    summaries.append(summary)
                    block = read(str(directory/('selected_predictions.csv' if uses_test else 'held_out_predictions.csv')))
                    for key, value in tag.items():
                        block[key] = value
                    block['calibration_status'] = selection['calibration_status']
                    predictions.append(block)
            p, f = pd.concat(predictions,ignore_index=True), pd.DataFrame(summaries)
            if len(p) != 18 or len(f) != 12:
                raise ValueError(f'Incomplete per-fold exports: {arm}/{strategy}/{budget}')
        if not p.selection_uses_test_data.eq(uses_test).all() or not f.selection_uses_test_data.eq(uses_test).all():
            raise ValueError('Selection-role labels do not match the selected files')
        for scheme in SCHEMES:
            pf, ff = p[p.scheme.eq(scheme)], f[f.scheme.eq(scheme)]
            if set(pf.input_key) != set(keys) or pf.input_key.duplicated().any():
                raise ValueError(f'Duplicate/missing predictions: {strategy}/{scheme}')
            if set(ff.held_out) != set(expected[scheme]) or ff.held_out.duplicated().any():
                raise ValueError(f'Duplicate/missing folds: {strategy}/{scheme}')
            for held_out, test_keys in expected[scheme].items():
                row = ff[ff.held_out.eq(held_out)].iloc[0]
                train_keys = ast.literal_eval(row.training_keys)
                if set(ast.literal_eval(row.test_keys)) != set(test_keys) or set(train_keys) != set(keys)-set(test_keys):
                    raise ValueError(f'Incorrect split: {scheme}/{held_out}')
                cases = pf[pf.held_out.eq(held_out)]
                if set(cases.input_key) != set(test_keys):
                    raise ValueError('Predictions do not match the test fold')
                good = cases[cases.status.eq('predicted')]
                errors = pd.to_numeric(good.signed_interval_error_days, errors='raise')
                if not np.isfinite(errors).all():
                    raise ValueError('Nonfinite predicted error')
                for r in good.itertuples():
                    start, end, lo, hi = map(pd.Timestamp, (r.predicted_start, r.predicted_end, r.observed_L, r.observed_U))
                    recomputed = (end-lo).total_seconds()/86400 if end < lo else (start-hi).total_seconds()/86400 if start > hi else 0
                    if start > end or not np.isclose(recomputed, r.signed_interval_error_days, atol=1e-9, rtol=0):
                        raise ValueError('Saved interval error does not match dates')
                if uses_test and row.calibration_status == 'selected':
                    if not np.isclose(row.selection_tolerance_days, 1) or row.training_MAE_days > row.minimum_training_MAE_days+1+1e-9:
                        raise ValueError('Selected point violates the 1-day training tolerance')
                    if len(good) != len(cases) or not np.isclose(errors.abs().mean(), row.test_MAE_days):
                        raise ValueError('Exploratory selection has incomplete or inconsistent test metrics')
                output.append(dict(strategy=strategy,arm=arm,scheme=scheme,held_out=held_out,
                    seed=seed,n_trials=budget,n_train=len(train_keys),n_test=len(test_keys),n_predicted=len(good),
                    training_MAE_days=row.get('training_MAE_days',np.nan),
                    test_MAE_days=float(errors.abs().mean()) if len(good)==len(cases) else np.nan,
                    conditional_test_MAE_days=float(errors.abs().mean()) if len(good) else np.nan,
                    bias_days=float(errors.mean()) if len(good) else np.nan,
                    **{n:row.get(n,np.nan) for n in ('VPD_w','I_req','u_tr','selected_trial','n_unique_candidates')},
                    calibration_status=row.calibration_status,selection_uses_test_data=uses_test,
                    train_cases='; '.join(train_keys),test_cases='; '.join(test_keys),
                    case_errors='; '.join(f'{r.input_key}: {r.signed_interval_error_days:+.2f}'
                        if r.status=='predicted' else f'{r.input_key}: NA ({r.status})' for r in cases.itertuples())))
                case_rows.extend(dict(r,strategy=strategy) for r in cases.to_dict('records'))
    return pd.DataFrame(output), pd.DataFrame(case_rows), expected, hashes


def plot(folds, cases, expected, output, arm, seed, budget):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fsim_plot_mpl')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig = Figure(figsize=(13.5,12))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3,2,sharey=True)
    values=folds[['training_MAE_days','test_MAE_days']].to_numpy(dtype=float).ravel()
    finite=values[np.isfinite(values)]
    high = max(10,float(finite.max()) if len(finite) else 10)
    upper=np.ceil((high+max(12,high*.25))/10)*10
    colors = ('#397A9E','#DB8938')
    for i, scheme in enumerate(SCHEMES):
        order = list(expected[scheme])
        for j,(strategy,title,_,_,_) in enumerate(STRATEGIES):
            ax=axes[i,j]
            sub=folds[(folds.strategy==strategy)&(folds.scheme==scheme)].set_index('held_out').loc[order]
            points=cases[(cases.strategy==strategy)&(cases.scheme==scheme)]
            ax.axhline(0,color='#4B5563',lw=.9,zorder=2)
            for x,(held,r) in enumerate(sub.iterrows()):
                for offset,value,color,label in [(-.18,r.training_MAE_days,'#AAB5C4','Train MAE'),
                                                  (.18,r.test_MAE_days,colors[j],'Test MAE')]:
                    if np.isfinite(value):
                        ax.bar(x+offset,value,width=.33,color=color,zorder=3,label=label if x==0 else None)
                        ax.annotate(f'{value:.2f}',(x+offset,value),xytext=(0,4),
                                    textcoords='offset points',ha='center',va='bottom',fontsize=8)
                    else:
                        ax.text(x+offset,1,'NA',ha='center',va='bottom',color='#A12C32',fontsize=9)
            labels=[]
            for h,r in sub.iterrows():
                name=h.replace('_MPT_', ' ').replace('_WK_', ' ').replace('_JF_', ' ')
                labels.append(f'{name}\n({int(r.n_predicted)}/{int(r.n_test)})')
            ax.set_xticks(range(len(order)),labels,fontsize=9 if scheme=='losyo' else 10)
            good=points[points.status.eq('predicted')]
            mae=good.signed_interval_error_days.abs().mean()
            metric='Test MAE (pooled)' if len(good)==len(points) else 'Conditional test MAE'
            mae_label=f'{mae:.2f}' if len(good) else 'NA'
            ax.set_title(f'({chr(97+i*2+j)}) {LABELS[scheme]}',loc='left',pad=26,fontweight='bold')
            ax.text(0,1.02,f'{metric}: {mae_label} d  |  Predictions: {len(good)}/{len(points)}',
                    transform=ax.transAxes,fontsize=10,color='#45505D')
            if j==0:ax.set_ylabel('Mean absolute interval error (days)')
            ax.set_ylim(0,upper)
            ax.set_xlabel('Held-out fold',labelpad=8,fontsize=10)
            ax.legend(handles=[Patch(facecolor='#AAB5C4',label='Train MAE'),
                               Patch(facecolor=colors[j],label='Test MAE')],
                      loc='upper right',ncol=2,frameon=False,fontsize=9)
            ax.yaxis.set_major_locator(MultipleLocator(20 if upper>=80 else 10))
            ax.grid(axis='y',color='#E3E7EC',lw=.8,zorder=0)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Plan {arm}: {"3p" if arm=="C" else "2p"} TPE selection comparison',fontsize=19,y=.981)
    fig.text(.5,.951,f'seed = {seed}  |  {budget} training trials per fold  |  six site-years',ha='center',fontsize=11)
    fig.text(.292,.916,'Minimum training MAE',ha='center',fontsize=14,color=colors[0],fontweight='bold')
    fig.text(.756,.922,'Training MAE ≤ minimum + 1 day',ha='center',fontsize=13,color=colors[1],fontweight='bold')
    fig.text(.756,.903,'Then minimum test MAE (exploratory)',ha='center',fontsize=11,color=colors[1])
    fig.text(.075,.032,'Paired bars: training MAE (gray) and test MAE (blue/orange), using the parameters selected in each column.\n'
             'Parentheses: predicted / test cases. Missing predictions are not assigned zero; panel test MAE pools site-years.\n'
             'Right column uses test outcomes for selection and does not represent independent test performance.',fontsize=9,linespacing=1.5)
    fig.subplots_adjust(left=.075,right=.98,bottom=.14,top=.855,hspace=.65,wspace=.18)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(output.with_suffix('.pdf'),facecolor='white')
    fig.clear()


def terminal_report(folds,cases,arm,budget,status):
    lines=[f'方案 {arm}，每折 {budget} 次；运行状态：{status}。本报告仅使用该方案已导出的完整结果。',
           '左列：最低训练 MAE；右列：训练 MAE ≤ 最低值+1天后，按测试 MAE 选优（探索性）。',
           '偏差为有符号区间误差：正值偏晚，负值偏早；NA 不按零计算。']
    for scheme in SCHEMES:
        lines.extend(['',f'=== {LABELS[scheme]} ==='])
        for strategy,label,_,_,_ in STRATEGIES:
            sub=folds[(folds.strategy==strategy)&(folds.scheme==scheme)].copy()
            sub['预测/测试']=sub.n_predicted.astype(str)+'/'+sub.n_test.astype(str)
            lines.append(label)
            columns=['held_out','n_train','training_MAE_days','test_MAE_days','bias_days','预测/测试','VPD_w','I_req','u_tr','selected_trial']
            lines.append(sub[columns].rename(columns={'held_out':'留出折','n_train':'训练数',
                'training_MAE_days':'训练MAE','test_MAE_days':'测试MAE','bias_days':'偏差',
                'selected_trial':'trial编号(从0开始)'}).to_string(index=False,float_format=lambda v:f'{v:.3f}',na_rep='NA'))
            for r in sub.itertuples():lines.append(f'  {r.held_out}: {r.case_errors}')
            p=cases[(cases.strategy==strategy)&(cases.scheme==scheme)]
            v=p[p.status.eq('predicted')].signed_interval_error_days
            label_mae='MAE' if len(v)==len(p) else '条件MAE'
            lines.append(f'  总体：预测 {len(v)}/{len(p)}；{label_mae}={v.abs().mean():.3f} 天；偏差={v.mean():+.3f} 天')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'result/fsim_abc_tpe_seed123')
    parser.add_argument('--arm',choices=['A','B','C'],default='A')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--budget',type=int,default=1024)
    parser.add_argument('--output',type=Path,help='Output filename stem; PNG/PDF/CSV/TXT are written alongside it')
    args=parser.parse_args()
    output=args.output or ROOT/f'fig/fsim_abc_{args.arm}_selection_comparison_{args.budget}'
    folds,cases,expected,hashes=load(args.results,args.arm,args.seed,args.budget)
    plot(folds,cases,expected,output,args.arm,args.seed,args.budget)
    meta=json.loads((args.results/'run_metadata.json').read_text())
    report=terminal_report(folds,cases,args.arm,args.budget,meta.get('status','unknown'))
    folds.to_csv(output.with_name(output.name+'_folds.csv'),index=False)
    cases.to_csv(output.with_name(output.name+'_cases.csv'),index=False)
    output.with_suffix('.txt').write_text(report)
    output.with_name(output.name+'_audit.json').write_text(json.dumps(dict(
        arm=args.arm,budget=args.budget,seed=args.seed,run_status=meta.get('status'),active_plan=meta.get('active_plan'),
        input_sha256=hashes,checked_splits=24,checked_prediction_rows=36,models_executed=0),indent=2))
    print(report,flush=True)
    print(f'已绘图：{output}.png\n        {output}.pdf\n分折报告：{output}.txt',flush=True)


if __name__=='__main__':main()
