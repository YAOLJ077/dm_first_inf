#!/usr/bin/env python3
"""Plot completed M0 RH_sp LOSYO trial without selecting parameters from OOF data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.plot_sobol_fsim import plt,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'result/fsim_m0_losyo_RH_trial')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'fig/fsim_m0_losyo_RH_trial')
    args=parser.parse_args();source=args.input_dir.resolve();out=args.output_dir.resolve()
    meta=json.loads((source/'run_metadata.json').read_text())
    if meta['status']!='complete':raise ValueError('Complete trial required')
    q=pd.read_csv(source/'baseline_comparison.csv');profiles=pd.read_csv(source/'RH_training_profiles.csv')
    if len(q)!=6 or q.input_key.duplicated().any():raise ValueError('Expected six OOF pairs')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'xtick.labelsize':9,'ytick.labelsize':9})
    fig,axes=plt.subplots(2,3,figsize=(12,7.5))
    fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.17,hspace=.45,wspace=.3)
    fig.suptitle('RH candidate selection: five-site training MAE in each LOSYO fold',y=.97,weight='bold')
    for ax,row in zip(axes.flat,q.itertuples()):
        g=profiles[profiles.held_out.eq(row.input_key)].sort_values('RH_sp')
        ax.plot(g.RH_sp,g.best_training_MAE_days,color='#0072B2',marker='o')
        chosen=g[g.RH_sp.eq(row.RH_sp)].iloc[0]
        ax.scatter(chosen.RH_sp,chosen.best_training_MAE_days,s=75,facecolors='none',edgecolors='#D55E00',linewidths=2,zorder=3)
        ax.set_title(f'Leave out {row.input_key}',loc='left',fontsize=10)
        ax.set(xticks=meta['trial']['RH_sp_values'],xlabel='RH threshold (%)',ylabel='Training MAE (days)')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    save(fig,out,'01_RH_training_selection',
         'Each RH value has its own VPD_w / I_req calibration with identical bounds and three refinement rounds. Orange ring: selected RH.\n'
         'Selection uses only five training site-years; no held-out errors enter the decision. RH values are discrete exploratory candidates.\n'
         'Lines connect sampled thresholds for readability; they are not interpolated training loss or evidence of a continuous optimum.')

    fig,axes=plt.subplots(1,2,figsize=(12,6))
    fig.subplots_adjust(left=.16,right=.98,top=.82,bottom=.2,wspace=.32)
    fig.suptitle('Does adding RH_sp improve M0? Training and held-out errors',y=.97,weight='bold')
    y=np.arange(len(q))
    for ax,cols,title in zip(axes,[('baseline_training_MAE_days','RH_training_MAE_days'),
                                 ('signed_interval_error_days_baseline','signed_interval_error_days_RH_trial')],
                            ['Five-site training MAE','Held-out signed interval error']):
        for j,(v0,v1) in enumerate(q[list(cols)].itertuples(index=False,name=None)):
            ax.plot([v0,v1],[j,j],color='#bbbbbb',lw=1.2)
        ax.scatter(q[cols[0]],y,s=35,color='#0072B2',label='M0: RH fixed at 80%')
        ax.scatter(q[cols[1]],y,s=40,color='#D55E00',marker='s',label='M0: RH selected from training')
        ax.set(yticks=y,yticklabels=q.input_key,xlabel='Days');ax.set_title(title,loc='left',fontsize=11)
        ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.axvline(0,color='#555555',lw=.7)
    axes[1].axvspan(-7,7,color='#009E73',alpha=.10,zorder=0)
    axes[1].axvline(-10,color='#009E73',ls=':',lw=1);axes[1].axvline(10,color='#009E73',ls=':',lw=1)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False)
    save(fig,out,'02_training_and_OOF_comparison',
         'Right: negative = predicted onset too early; positive = too late. Shading marks ±7 days and dotted lines mark ±10 days.\n'
         'Training improvement is not held-out improvement. Each OOF prediction uses the RH and other parameters chosen without that site-year.\n'
         'The connected points are model comparisons, not confidence intervals. There are six independent site-years.')

    cols=['RH_sp','VPD_w_RH_trial','I_req_RH_trial']
    fig,axes=plt.subplots(1,3,figsize=(12,5.6));fig.subplots_adjust(left=.16,right=.98,top=.85,bottom=.2,wspace=.45)
    fig.suptitle('Training-selected parameters across LOSYO folds',y=.97,weight='bold')
    for ax,col,label in zip(axes,cols,['RH threshold (%)',r'$VPD_w$ (hPa)',r'$I_{req}$ (°C h)']):
        ax.scatter(q[col],y,color='#D55E00',marker='s',label='RH trial')
        if col!='RH_sp':
            old=col.replace('_RH_trial','_baseline');ax.scatter(q[old],y,color='#0072B2',label='Baseline')
        else:ax.axvline(80,color='#0072B2',ls=':',label='Baseline RH')
        ax.set(yticks=y,yticklabels=q.input_key,xlabel=label);ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axes[0].set_xticks(meta['trial']['RH_sp_values'])
    save(fig,out,'03_selected_parameters',
         'Orange squares: parameters selected with RH open. Blue circles / dotted line: original two-parameter M0 with RH fixed at 80%.\n'
         'Parameters are training-selected representatives of sampled minima, not unique estimates or posterior means.\n'
         'Endpoint selection does not by itself justify extending the RH range.')
    audit={'input_directory':str(source),'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.csv')},
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plot_metadata.json').write_text(json.dumps(audit,indent=2))
    (out/'README.md').write_text('图件读取已完成RH试验，未重新选参。\n\n01：各RH的训练MAE与选定值；02：训练与留出误差对照；03：跨折参数变化。\n\n每图保存PNG和矢量PDF。详细结果见 '+str(source/'README.md')+'。\n')
    print(out)


if __name__=='__main__':main()
