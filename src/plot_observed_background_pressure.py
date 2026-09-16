#!/usr/bin/env python3
"""Plot interval-level observed background airborne-spore pressure.

Creates one 2x3 composite figure for the six site-years. Pressure is shown as
an interval step series; the secondary axis shows interval-normalized spore
intensity (spores per collection day). This is diagnostic visualization only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix'})
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_INPUT=ROOT/'result'/'observed_background_pressure'/'background_pressure_intervals.csv'
DEFAULT_OUTPUT=ROOT/'fig'/'observed_background_pressure_composite.png'
GROUP=['Place','Year','Plot','Variety']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=DEFAULT_INPUT);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--dpi',type=int,default=300);a=p.parse_args()
 d=pd.read_csv(a.input,encoding='utf-8-sig',parse_dates=['sampling_start','sampling_end'])
 required=set(GROUP+['sampling_start','sampling_end','spore_total','interval_intensity','pressure'])
 if required-set(d.columns): raise ValueError(f'Missing columns: {sorted(required-set(d.columns))}')
 d=d.sort_values(GROUP+['sampling_end']); groups=list(d.groupby(GROUP,sort=True));
 fig,axes=plt.subplots(2,3,figsize=(15,8.2),sharey=True,constrained_layout=True); axes=axes.ravel()
 for ax,(key,g) in zip(axes,groups):
  g=g.sort_values('sampling_end').copy(); x=g.sampling_end; y=g.pressure
  # Draw interval pressure as a post-step, with markers at completed collection dates.
  ax.step(x,y,where='post',color='#146c94',linewidth=2,label='Relative pressure')
  ax.scatter(x,y,s=22,color='#146c94',edgecolor='white',linewidth=.4,zorder=3)
  ax.fill_between(x,0,y,step='post',color='#76c7c8',alpha=.18)
  ax.set_ylim(-0.03,1.08); ax.grid(axis='y',alpha=.25,linewidth=.7); ax.set_title(f'{key[0]} {key[1]} ({key[3]})',loc='left',fontsize=11,fontweight='bold')
  ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')); ax.tick_params(axis='x',labelrotation=35,labelsize=8)
  ax2=ax.twinx(); ax2.plot(x,g.interval_intensity,color='#d95f02',marker='o',markersize=3.5,linewidth=1.1,alpha=.82,label='Interval intensity')
  ymax=max(float(g.interval_intensity.max()),1.0); ax2.set_ylim(0,ymax*1.22); ax2.tick_params(axis='y',labelsize=8,colors='#9c4a00'); ax2.spines['right'].set_color('#d95f02')
  if ax is axes[0]: ax.set_ylabel('Relative pressure (0–1)')
  if ax2 is axes[2]: ax2.set_ylabel('Spore count / collection day',color='#9c4a00')
 axes[0].legend(loc='upper left',frameon=False,fontsize=8)
 # Shared figure note clarifies that dates are interval end dates.
 fig.suptitle('Observed background airborne-spore pressure by site-year',fontsize=15,fontweight='bold')
 a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=a.dpi,bbox_inches='tight'); plt.close(fig); print(f'Saved {a.output}')
if __name__=='__main__': main()
