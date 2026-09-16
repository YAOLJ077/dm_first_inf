#!/usr/bin/env python3
"""Build interval-level observed background airborne-spore pressure.

The workbook records one Spore value per completed slide-collection interval;
blank dates between values are not independent missing observations.  This
script keeps the interval as the observational unit and creates a daily view
with the interval pressure held constant over the interval.  It does not fit
FSIM parameters or inject pressure into a model.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "grape_experiment_CK_site_years.xlsx"
DEFAULT_OUT = ROOT / "result" / "observed_background_pressure"
GROUP = ["Place", "Year", "Plot", "Variety"]

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--p95",type=float,default=None,help="Optional fixed training-fold Q95; otherwise descriptive Q95 is written.")
    p.add_argument("--max-gap-days",type=int,default=4,help="Flag intervals longer than this; do not discard them.")
    return p.parse_args()

def load(path):
    df=pd.read_excel(path,sheet_name="CK_Data")
    required=set(GROUP+['Date','Spore'])
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    df=df.copy(); df['Date']=pd.to_datetime(df['Date'],errors='coerce').dt.normalize()
    df['Spore']=pd.to_numeric(df['Spore'],errors='coerce')
    if df.Date.isna().any(): raise ValueError('Invalid Date values')
    if (df.Spore<0).any(): raise ValueError('Spore counts must be non-negative')
    return df

def build_intervals(df,max_gap):
    rows=[]
    for key,g in df.groupby(GROUP,sort=True):
        obs=g[g.Spore.notna()].groupby('Date',as_index=False).agg(spore_total=('Spore','sum'), n_records=('Spore','size')).sort_values('Date')
        if obs.empty: continue
        # A value belongs to the collection period since the previous
        # completed slide change. The first interval starts at t-2 days,
        # matching the nominal three-day protocol.
        for j,r in obs.iterrows():
            end=pd.Timestamp(r.Date)
            if j==obs.index[0]:
                start=end-pd.Timedelta(days=2)
                source='nominal_first_interval'
            else:
                prev=pd.Timestamp(obs.loc[j-1,'Date'])
                start=prev+pd.Timedelta(days=1)
                source='between_completed_collections'
            duration=(end-start).days+1
            total=float(r.spore_total)
            intensity=total/duration
            rows.append(dict(zip(GROUP,key),sampling_start=start,sampling_end=end,
                duration_days=duration,spore_total=total,source_records=int(r.n_records),
                interval_intensity=intensity,interval_flag=('long_interval' if duration>max_gap else 'nominal_or_short')))
    out=pd.DataFrame(rows)
    if out.empty: raise ValueError('No non-empty Spore observations found')
    return out.sort_values(GROUP+['sampling_end']).reset_index(drop=True)

def add_pressure(intervals,p95):
    x=np.log1p(intervals.interval_intensity.astype(float))
    q=float(np.quantile(x,0.95)) if p95 is None else float(p95)
    if not np.isfinite(q) or q<=0: raise ValueError('Q95 must be positive; no positive interval intensity found')
    out=intervals.copy(); out['log_interval_intensity']=x; out['q95_log_scale']=q
    out['pressure']=(x/q).clip(0,1)
    return out,q

def expand_daily(intervals):
    rows=[]
    for r in intervals.itertuples(index=False):
        for d in pd.date_range(r.sampling_start,r.sampling_end,freq='D'):
            rows.append({**{k:getattr(r,k) for k in GROUP},'date':d,
              'sampling_start':r.sampling_start,'sampling_end':r.sampling_end,
              'duration_days':r.duration_days,'spore_total':r.spore_total,
              'interval_intensity':r.interval_intensity,'pressure':r.pressure,
              'pressure_allocation_weight':1.0/r.duration_days,
              'pressure_status':'observed_interval'})
    return pd.DataFrame(rows).sort_values(GROUP+['date']).reset_index(drop=True)

def main():
    a=parse_args(); df=load(a.input); intervals=build_intervals(df,a.max_gap_days); intervals,q=add_pressure(intervals,a.p95); daily=expand_daily(intervals)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    intervals.to_csv(a.output_dir/'background_pressure_intervals.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(a.output_dir/'background_pressure_daily.csv',index=False,encoding='utf-8-sig')
    meta={'input':str(a.input.relative_to(ROOT) if a.input.is_relative_to(ROOT) else a.input),'observational_unit':'completed slide-collection interval','nominal_interval_days':3,'first_interval_rule':'end_date minus 2 days through end_date','subsequent_interval_rule':'day after previous completed collection through current end date','pressure_definition':'clip(log1p(spore_total/duration_days)/Q95, 0, 1)','q95_log_scale':q,'q95_source':'command-line fixed value' if a.p95 is not None else 'descriptive all-interval value; refit within each training fold for evaluation','daily_allocation':'constant interval pressure; weight=1/duration_days for external-dose allocation','long_interval_flag_days':a.max_gap_days,'n_intervals':int(len(intervals)),'n_daily_rows':int(len(daily))}
    (a.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n',encoding='utf-8')
    print(f'Saved {len(intervals)} intervals and {len(daily)} daily rows to {a.output_dir}')
    print(intervals.groupby(GROUP).agg(intervals=('sampling_end','size'),total_spore=('spore_total','sum'),positive_intervals=('spore_total',lambda x:int((x>0).sum())),max_pressure=('pressure','max')).to_string())
if __name__=='__main__': main()
