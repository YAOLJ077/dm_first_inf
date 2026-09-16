#!/usr/bin/env python3
"""Run fixed and beta-only experiments sequentially, sharing one worker budget."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pressure_experiment_runner as runner
import pressure_model_core as core
import pandas as pd
import numpy as np


def compare(args):
    fixed_dir=args.output_root/'fixed'; tpe_dir=args.output_root/'beta_tpe'
    for directory in (fixed_dir,tpe_dir):
        if core.base.read_json(directory/'run_metadata.json')['status']!='complete':
            raise ValueError('Both experiments must complete before comparison')
    left=pd.read_csv(fixed_dir/'anchor_predictions.csv')
    right=pd.read_csv(tpe_dir/'anchor_predictions.csv')
    keys=['arm','scheme','fold','case_role','input_key']
    columns=keys+['status','n_windows','predicted_start','predicted_end','signed_interval_error_days']
    pd.testing.assert_frame_equal(left[columns].sort_values(keys).reset_index(drop=True),
        right[columns].sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=False,atol=1e-8,rtol=0)
    frames=[pd.read_csv(fixed_dir/'predictions.csv'),pd.read_csv(tpe_dir/'predictions.csv')]
    records=pd.concat([frames[0],frames[1][frames[1].arm.eq('beta_tpe')]],ignore_index=True)
    destination=args.output_root/'comparison'
    destination.mkdir(parents=True,exist_ok=True)
    runner.summarize(records.to_dict('records'),destination)
    tests=records[records.case_role.eq('test')]
    key=['scheme','fold','input_key']
    baseline=tests[tests.arm.eq('B')][key+['status','interval_distance_days']].rename(
        columns={'status':'B_status','interval_distance_days':'B_error_days'})
    paired=tests[~tests.arm.eq('B')].merge(baseline,on=key,validate='many_to_one')
    paired['error_change_days']=paired.interval_distance_days-paired.B_error_days
    core.base.save_frame(destination/'paired_test_changes.csv',paired)
    selections=[]
    for file in sorted((tpe_dir/'folds').glob('*/*/selection.json')):
        s=core.base.read_json(file)
        selections.append(dict(scheme=file.parent.parent.name,fold=file.parent.name,beta=s['beta'],
            selected_trial=s['number'],training_MAE_days=s['metrics']['training_MAE_days']))
    core.base.save_frame(destination/'selected_beta.csv',pd.DataFrame(selections))
    plot(records,args.output_root)
    core.base.save_json(destination/'comparison_checks.json',dict(status='complete',
        beta_zero_matches_B=True,beta_one_matches_fixed=True,selection_uses_test_data=False,
        note='Error changes only defined when both runs predict. Conditional MAE must be read with coverage.'))
    print(f'Comparison: {destination}',flush=True)


def plot(records,output_root):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'fsim_pressure_compare_mpl'))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(16,12));FigureCanvasAgg(fig)
    axes=fig.subplots(3,2)
    arms=['B','fixed','beta_tpe'];colors=['#7c8b9c','#dd8834','#35849e']
    for i,scheme in enumerate(['losyo','site','year']):
        for j,role in enumerate(['train','test']):
            ax=axes[i,j]
            data=records[records.scheme.eq(scheme)&records.case_role.eq(role)]
            if data.empty:ax.set_visible(False);continue
            folds=sorted(data.fold.astype(str).unique())
            maxima=[]
            for n,(arm,color) in enumerate(zip(arms,colors)):
                values=[];counts=[]
                for fold in folds:
                    g=data[data.arm.eq(arm)&data.fold.astype(str).eq(fold)]
                    ok=g[g.status.eq('predicted')]
                    values.append(ok.interval_distance_days.mean());counts.append((len(ok),len(g)))
                bars=ax.bar(np.arange(len(folds))+(n-1)*.25,values,.24,label=arm,color=color)
                for bar,value,(pred,total) in zip(bars,values,counts):
                    if np.isfinite(value):
                        maxima.append(value)
                        ax.annotate(f'{value:.1f}'+('*' if pred<total else ''),(bar.get_x()+bar.get_width()/2,value),
                            xytext=(0,4),textcoords='offset points',ha='center',fontsize=7)
                    else:
                        ax.text(bar.get_x()+bar.get_width()/2,1,'NA',ha='center',fontsize=7)
            ax.set_xticks(np.arange(len(folds)),[f.replace('_MPT','').replace('_WK','').replace('_JF','').replace('_',' ') for f in folds],rotation=15)
            ax.set_ylim(0,max(maxima+[1])*1.25)
            coverage='; '.join(f'{arm} {int(data[data.arm.eq(arm)].status.eq("predicted").sum())}/{int(data.arm.eq(arm).sum())}' for arm in arms)
            ax.set_title(f'{scheme.upper()} | {role}\nPredictions: {coverage}',loc='left',fontsize=10)
            ax.set_ylabel('Interval MAE (days)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.legend(fontsize=8)
    fig.suptitle('Frozen B vs fixed pressure mapping vs beta-only TPE',fontsize=17)
    fig.text(.06,.02,'* Conditional MAE when predictions are missing; NA is not zero.\nB parameters fixed per fold. Beta selected on training cases only; no B re-calibration.',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.95))
    # Keep experiment-specific figures beside its comparison, with a unique main fig folder for the default run.
    default=core.ROOT/'result/spore_observation/plan1_pressure_experiments'
    fig_dir=(core.ROOT/'fig/spore_observation/plan1_pressure_experiments') if output_root==default else output_root/'fig'
    fig_dir.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(fig_dir/f'B_fixed_beta_comparison.{ext}',dpi=160)
    fig.clear()
    print(f'Figures: {fig_dir}',flush=True)


def main():
    args=runner.normalize(runner.parser().parse_args())
    args.output_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output_root/'.scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('evaluate_fixed_pressure.py','calibrate_pressure_tpe.py'):
            command=[sys.executable,str(Path(__file__).parent/name),*sys.argv[1:]]
            print(f'Starting {name}; global worker limit {args.workers}',flush=True)
            subprocess.run(command,check=True)
        compare(args)


if __name__=='__main__':
    main()
