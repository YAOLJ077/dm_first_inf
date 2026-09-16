#!/usr/bin/env python3
"""Plot saved joint calibration and compare B under matched selection rules."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {'minimum': 'min_train', 'tolerance_1day': 'train_plus1_test_selected'}
SCHEMES = {'losyo': 'LOSYO', 'site': 'Leave-one-site-out', 'year': 'Leave-one-year-out'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'result/spore_observation/plan1_joint_pressure_tpe')
    parser.add_argument('--baseline', type=Path, default=ROOT/'result/spore_observation/plan1_b_fixed')
    parser.add_argument('--fig-dir', type=Path, default=ROOT/'fig')
    args = parser.parse_args()
    meta = json.loads((args.results/'run_metadata.json').read_text())
    assert meta['status'] == 'complete' and meta['baseline_checks_passed']
    assert json.loads((args.baseline/'run_metadata.json').read_text())['baseline_reproduced']
    joint = pd.read_csv(args.results/'both_versions_predictions.csv')
    joint = joint[joint.arm.eq('joint_tpe')].copy()
    baseline = pd.read_csv(args.baseline/'predictions.csv')
    baseline['selection_version'] = baseline.rule.map({v:k for k,v in VERSIONS.items()})
    baseline['arm'] = 'B'
    frame = pd.concat([joint, baseline[baseline.selection_version.notna()]], ignore_index=True)
    keys = ['selection_version','scheme','fold','input_key','case_role']
    assert not frame.duplicated(keys+['arm']).any()
    b = frame[frame.arm.eq('B')]
    j = frame[frame.arm.eq('joint_tpe')]
    paired = j.merge(b, on=keys, suffixes=('_joint','_B'), validate='one_to_one', how='outer', indicator=True)
    assert paired['_merge'].eq('both').all()
    paired['both_predicted'] = paired.status_joint.eq('predicted') & paired.status_B.eq('predicted')
    paired['error_change_days'] = (paired.interval_distance_days_joint-paired.interval_distance_days_B).where(paired.both_predicted)
    paired.to_csv(args.results/'matched_B_paired_predictions.csv', index=False)
    folds, summary = [], []
    for (version,scheme,arm), g in frame.groupby(['selection_version','scheme','arm']):
        test = g[g.case_role.eq('test')]
        ok = test[test.status.eq('predicted')]
        summary.append(dict(selection_version=version,scheme=scheme,arm=arm,n=len(test),n_predicted=len(ok),
            MAE_days=ok.interval_distance_days.mean() if len(ok)==len(test) else np.nan,
            conditional_MAE_days=ok.interval_distance_days.mean(),n_overlap=int(ok.overlap.sum())))
        for fold, f in g.groupby('fold'):
            tr, te = f[f.case_role.eq('train')], f[f.case_role.eq('test')]
            good = te[te.status.eq('predicted')]
            assert tr.status.eq('predicted').all()
            folds.append(dict(selection_version=version,scheme=scheme,arm=arm,fold=fold,
                train_MAE=tr.interval_distance_days.mean(),test_MAE=good.interval_distance_days.mean(),
                n=len(te),n_predicted=len(good)))
    summary, folds = pd.DataFrame(summary), pd.DataFrame(folds)
    summary.to_csv(args.results/'matched_B_method_summary.csv',index=False)
    folds.to_csv(args.results/'joint_plot_fold_summary.csv',index=False)
    args.fig_dir.mkdir(parents=True,exist_ok=True)
    colors = ['#387fa3','#df8733']
    for comparison in [False, True]:
        fig, axes = plt.subplots(3,2,figsize=(15,13))
        maximum = folds[['train_MAE','test_MAE']].max().max()
        for row,(scheme,label) in enumerate(SCHEMES.items()):
            for col,version in enumerate(VERSIONS):
                ax=axes[row,col]
                data=folds[(folds.scheme==scheme)&(folds.selection_version==version)]
                jd=data[data.arm.eq('joint_tpe')].sort_values('fold').set_index('fold')
                bd=data[data.arm.eq('B')].set_index('fold').loc[jd.index]
                x=np.arange(len(jd)); left=bd.test_MAE if comparison else jd.train_MAE
                for vals,offset,color,name in [(left,-.19,'#a9b5c5','B test MAE' if comparison else 'Train MAE'),(jd.test_MAE,.19,colors[col],'Joint test MAE' if comparison else 'Test MAE')]:
                    bars=ax.bar(x+offset,vals,.37,color=color,label=name)
                    for i,bar in enumerate(bars):
                        incomplete=comparison and offset<0 and bd.iloc[i].n_predicted<bd.iloc[i].n
                        ax.annotate(f'{bar.get_height():.2f}'+('*' if incomplete else ''),(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
                names=[' '.join(f.split('_')[::2]) if '_' in f else f for f in jd.index]
                counts=[f'B {int(bd.iloc[i].n_predicted)}/{int(bd.iloc[i].n)}; J {int(r.n_predicted)}/{int(r.n)}' if comparison else f'{int(r.n_predicted)}/{int(r.n)}' for i,(_,r) in enumerate(jd.iterrows())]
                ax.set_xticks(x,[f'{n}\n({c})' for n,c in zip(names,counts)],fontsize=9)
                s=summary[(summary.scheme==scheme)&(summary.selection_version==version)&summary.arm.eq('joint_tpe')].iloc[0]
                ax.set_title(f'({chr(97+row*2+col)}) {label}',loc='left',fontweight='bold',pad=32)
                ax.text(0,1.02,f'Joint test MAE (pooled): {s.conditional_MAE_days:.2f} d | Predictions: {s.n_predicted}/{s.n}',transform=ax.transAxes,color='#536173',fontsize=10)
                ax.set_ylim(0,np.ceil(maximum/20)*20+30)
                ax.set_xlabel('Held-out fold')
                if col==0: ax.set_ylabel('Mean absolute interval error (days)')
                ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
                ax.legend(loc='upper right',fontsize=9)
        fig.suptitle('Joint VPD / I / beta TPE'+(' vs matched-rule B' if comparison else ': selection comparison'),fontsize=22,y=.98)
        fig.text(.5,.947,'seed = 123 | 1024 training trials per fold | six site-years',ha='center',fontsize=12)
        fig.text(.28,.91,'Minimum training MAE',ha='center',color=colors[0],fontsize=16,weight='bold')
        fig.text(.76,.902,'Training MAE ≤ minimum + 1 day\nThen minimum test MAE (exploratory)',ha='center',color=colors[1],fontsize=14)
        fig.text(.075,.025,('B and joint use the same selection rule in each column; * denotes conditional MAE when B misses a prediction.\n' if comparison else 'Paired bars: training MAE (gray) and test MAE (blue/orange) for the selected parameters.\n')+'Parentheses: predicted / held-out cases; pooled MAE weights site-years equally.\nRight column uses test outcomes for selection and does not represent independent test performance.',fontsize=10)
        fig.subplots_adjust(left=.075,right=.98,top=.84,bottom=.13,hspace=.65,wspace=.2)
        name='joint_pressure_vs_B_matched' if comparison else 'joint_pressure_selection_comparison_1024'
        for ext in ('png','pdf'):fig.savefig(args.fig_dir/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    diagnostics=[]
    for file in sorted((args.results/'folds').glob('*/*/trials.json')):
        history=json.loads(file.read_text());valid=[r for r in history if r['metrics']['feasible']]
        best=min(r['metrics']['training_MAE_days'] for r in valid)
        near=[r for r in valid if r['metrics']['training_MAE_days']<=best+1+1e-9]
        candidates=pd.read_csv(args.results/'exploratory/folds'/file.parent.relative_to(args.results/'folds')/'selection_candidates.csv')
        diagnostics.append(dict(scheme=file.parent.parent.name,fold=file.parent.name,n_trials=len(history),n_feasible=len(valid),n_near=len(near),beta_near_min=min(r['params']['beta'] for r in near),beta_near_max=max(r['params']['beta'] for r in near),test_near_min=candidates.test_MAE_days.min(),test_near_max=candidates.test_MAE_days.max()))
    pd.DataFrame(diagnostics).to_csv(args.results/'joint_near_optimal_diagnostics.csv',index=False)
    print(summary.to_string(index=False))
    print('\nLOSYO minimum paired changes:')
    print(paired[(paired.selection_version=='minimum')&(paired.scheme=='losyo')&(paired.case_role=='test')][['input_key','interval_distance_days_B','interval_distance_days_joint','error_change_days']].to_string(index=False))
    print('\nNear-optimal diagnostics:');print(pd.DataFrame(diagnostics).to_string(index=False))

if __name__=='__main__': main()
