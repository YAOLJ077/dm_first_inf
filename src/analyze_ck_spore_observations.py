#!/usr/bin/env python3
"""Audit CK interval counts and symptom timing without altering source data.

No daily interpolation, model calibration, viability inference or source attribution.
The first collection covers three days, as confirmed by the researcher.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITES = {'Duan': 'DA', 'Guilin': 'GL', 'Mingyang': 'NN'}


def markdown(frame):
    def fmt(v):
        if pd.isna(v):
            return '—'
        if isinstance(v, float):
            return f'{v:.2f}'.rstrip('0').rstrip('.')
        return str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/grape_experiment_CK_site_years.xlsx')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/ck_spore_observation_audit')
    args = parser.parse_args()
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    df = pd.read_excel(args.input, sheet_name='CK_Data')
    df['excel_row'] = range(2, len(df) + 2)
    assert df.Plot.eq('CK').all(), 'Unexpected treatment'
    df['Date'] = pd.to_datetime(df.Date, errors='raise').dt.normalize()
    assert df.Date.notna().all()
    for col in ['Spore', 'disease', 'GS']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    assert not df.duplicated(['Place', 'Year', 'Date']).any()
    assert df.Date.dt.year.eq(df.Year).all()
    assert (df.Spore.dropna() >= 0).all()
    assert (df.Spore.dropna() % 1 == 0).all()
    assert set(df.disease.dropna().unique()) <= {0, 1}
    intervals, summaries, windows, runs, phenology = [], [], [], [], []
    for (place, year), group in df.groupby(['Place', 'Year'], sort=True):
        g = group.sort_values('Date')
        assert g.Variety.nunique() == 1
        key = f'{SITES[place]} {year}'
        positive_disease = g.loc[g.disease.gt(0), 'Date']
        assert len(positive_disease)
        upper = positive_disease.min()
        lower = g.loc[g.disease.eq(0) & g.Date.lt(upper), 'Date'].max()
        assert pd.notna(lower)
        obs = g[g.Spore.notna()].copy().reset_index(drop=True)
        obs['start'] = obs.Date.shift()
        obs.loc[0, 'start'] = obs.loc[0, 'Date'] - pd.Timedelta(days=3)
        obs['days'] = (obs.Date - obs.start).dt.days
        obs['rate'] = obs.Spore / obs.days
        for r in obs.itertuples():
            intervals.append(dict(site_year=key, place=place, year=int(year), variety=g.Variety.iloc[0],
                excel_row=r.excel_row, start_exclusive=r.start, end_inclusive=r.Date,
                duration_days=r.days, count=r.Spore, average_count_per_day=r.rate,
                exposure_start_status='researcher_confirmed_first_3_days' if r.Index == 0 else 'previous_collection',
                relation_to_onset='completed_by_L' if r.Date <= lower else
                    ('ends_in_L_U' if r.Date <= upper else 'after_U'), L=lower, U=upper))
        pre = obs[obs.Date.le(lower)]
        nz = obs[obs.Spore.gt(0)]
        pnz = pre[pre.Spore.gt(0)]
        known = obs[obs.days.notna()]
        peak = obs.loc[obs.Spore.idxmax()]
        first = nz.iloc[0]
        lastpre = pnz.iloc[-1]
        gap = obs.days.dropna().astype(int).value_counts().sort_index()
        summaries.append(dict(site_year=key, rows=len(g), nan_spore=int(g.Spore.isna().sum()),
            first_collection=obs.Date.min().date(), last_collection=obs.Date.max().date(),
            n_collections=len(obs), n_positive=len(nz), n_zero=int(obs.Spore.eq(0).sum()),
            positive_percent=100*len(nz)/len(obs), total=int(obs.Spore.sum()),
            median_count=obs.Spore.median(), mean_count=obs.Spore.mean(),
            gap_distribution='; '.join(f'{d}d:{n}' for d,n in gap.items()),
            known_exposure_days=int(known.days.sum()), known_total=int(known.Spore.sum()),
            known_average_per_day=known.Spore.sum()/known.days.sum(),
            peak_count=int(peak.Spore), peak_date=peak.Date.date(),
            peak_share_percent=100*peak.Spore/obs.Spore.sum(),
            peak_rate=known.rate.max(), peak_rate_date=known.loc[known.rate.idxmax(),'Date'].date(),
            L=lower.date(), U=upper.date(), onset_width_days=(upper-lower).days,
            first_nonzero=first.Date.date(), first_nonzero_start=first.start.date(),
            first_to_L_days=(lower-first.Date).days, first_to_U_days=(upper-first.Date).days,
            last_nonzero_by_L=lastpre.Date.date(), last_nonzero_count=int(lastpre.Spore),
            last_nonzero_to_L_days=(lower-lastpre.Date).days,
            last_nonzero_to_U_days=(upper-lastpre.Date).days,
            n_completed_by_L=len(pre), n_positive_by_L=len(pnz), total_by_L=int(pre.Spore.sum()),
            total_in_L_U=int(obs.loc[obs.Date.gt(lower)&obs.Date.le(upper),'Spore'].sum()),
            positive_sequence='; '.join(f'{r.Date:%m-%d}:{r.Spore:g}' for r in nz.itertuples())))
        for anchor_name, anchor in [('L', lower), ('U', upper)]:
            for width in (7, 14, 21):
                left = anchor - pd.Timedelta(days=width)
                completed = obs[obs.Date.le(anchor) & obs.Date.gt(left)]
                full = completed[completed.start.ge(left)]
                boundary = completed[completed.start.lt(left) | completed.start.isna()]
                coverage = sum((r.Date-max(left,r.start)).days for r in completed.itertuples() if pd.notna(r.start))
                windows.append(dict(site_year=key, anchor=anchor_name, anchor_date=anchor.date(),
                    lookback_days=width, window_start_exclusive=left.date(),
                    n_completed_collections=len(completed), n_positive=int(completed.Spore.gt(0).sum()),
                    completed_count=int(completed.Spore.sum()), fully_contained_count=int(full.Spore.sum()),
                    boundary_count=int(boundary.Spore.sum()), covered_days_known=coverage,
                    count_bounds_for_covered_part=f'{int(full.Spore.sum())}–{int(completed.Spore.sum())}'))
        # Runs refer to consecutive completed collections, not daily zero imputations.
        labels = obs.Spore.gt(0).ne(obs.Spore.gt(0).shift()).cumsum()
        for _, z in obs.groupby(labels):
            if not z.Spore.eq(0).all():
                continue
            start = z.start.iloc[0]
            end = z.Date.iloc[-1]
            runs.append(dict(site_year=key, first_zero_collection=z.Date.iloc[0].date(),
                last_zero_collection=end.date(), n_zero_collections=len(z),
                exposure_start_exclusive=start, known_duration_days=(end-start).days if pd.notna(start) else None,
                start_unknown=pd.isna(start), ends_by_L=end <= lower))
        for stage in [15, 53]:
            reached = g[g.GS.ge(stage)]
            r = reached.iloc[0] if len(reached) else None
            phenology.append(dict(site_year=key, target_BBCH=stage,
                first_recorded_at_or_above=r.Date.date() if r is not None else None,
                recorded_GS=r.GS if r is not None else None,
                excel_row=int(r.excel_row) if r is not None else None))
    summary = pd.DataFrame(summaries)
    assert len(summary) == 6
    assert summary.total.sum() == df.Spore.sum()
    endpoint_path = ROOT / 'data/site_year_onset_endpoints.csv'
    ep = pd.read_csv(endpoint_path)
    for r in summary.itertuples():
        code, year = r.site_year.split()
        e = ep[ep.place_code.eq(code) & ep.Year.eq(int(year))]
        assert len(e) == 1
        assert pd.Timestamp(e.iloc[0].L).date() == r.L and pd.Timestamp(e.iloc[0].U).date() == r.U
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frames = dict(site_year_summary=summary, collection_intervals=pd.DataFrame(intervals),
                  onset_lookback_windows=pd.DataFrame(windows), zero_runs=pd.DataFrame(runs),
                  workbook_phenology=pd.DataFrame(phenology))
    for name, frame in frames.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    audit = dict(input=str(args.input), sheet='CK_Data', sha256=digest,
        rows=len(df), site_years=len(summary), collections=int(df.Spore.notna().sum()),
        positive_collections=int(df.Spore.gt(0).sum()), zero_collections=int(df.Spore.eq(0).sum()),
        blank_spore_rows=int(df.Spore.isna().sum()), total_count=int(df.Spore.sum()),
        first_exposure_start='first collection date minus 3 days; researcher confirmed 2026-09-16',
        interval_rule='first: (t_first - 3 days, t_first]; subsequent: (previous nonblank collection date, current collection date]',
        endpoint_csv_matches=True,
        checks=['unique site-year dates','nonnegative integer counts','six groups','endpoint agreement','total reconciliation'])
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    tables = []
    for name, cols in [('overview',['site_year','n_collections','n_zero','n_positive','total','median_count','gap_distribution']),
                       ('timing',['site_year','L','U','first_nonzero','first_to_L_days','first_to_U_days','last_nonzero_by_L','total_by_L','total_in_L_U']),
                       ('peaks',['site_year','peak_date','peak_count','peak_share_percent','known_average_per_day','peak_rate','peak_rate_date']),
                       ('sequence',['site_year','positive_sequence'])]:
        tables.append(f'## {name}\n\n'+markdown(summary[cols]))
    tables.append('## windows\n\n'+markdown(frames['onset_lookback_windows'].query("anchor == 'L'")[
        ['site_year','lookback_days','n_completed_collections','n_positive','completed_count','count_bounds_for_covered_part','covered_days_known']]))
    tables.append('## zeros\n\n'+markdown(frames['zero_runs']))
    tables.append('## phenology\n\n'+markdown(frames['workbook_phenology']))
    (out/'tables.md').write_text('\n\n'.join(tables)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
