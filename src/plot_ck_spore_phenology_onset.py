#!/usr/bin/env python3
"""Plot raw CK spore counts, phenological dates and observed onset intervals.

Spore bars use recorded collection dates, without interpolation. Missing GS
dates may use the explicitly labelled A-run manifest reference dates.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITES = {'Duan': 'DA', 'Guilin': 'GL', 'Mingyang': 'NN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/grape_experiment_CK_site_years.xlsx')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'result/fsim_abc_tpe_seed123/site_year_manifest.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'fig/ck_spore_phenology_onset')
    parser.add_argument('--workbook-only', action='store_true', help='Do not supplement missing phenology from the A-run manifest.')
    args = parser.parse_args()
    df = pd.read_excel(args.input, sheet_name='CK_Data')
    df = df.loc[df.Plot.eq('CK')].copy()
    df['Date'] = pd.to_datetime(df.Date, errors='raise').dt.normalize()
    for col in ['Spore', 'GS', 'disease']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if df.Date.isna().any() or (df.Spore.dropna() < 0).any():
        raise ValueError('Invalid dates or negative spore counts')
    manifest = None if args.workbook_only else pd.read_csv(args.manifest)
    groups = list(df.groupby(['Place', 'Year'], sort=True))
    if len(groups) != 6:
        raise ValueError(f'Expected six site-years, found {len(groups)}')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    blue, green, purple, red = '#4386A5', '#258351', '#8555A1', '#C14B3F'
    audit = []
    for i, ((place, year), group) in enumerate(groups):
        ax = axes.flat[i]
        group = group.sort_values('Date')
        if group.Date.duplicated().any() or group.Variety.nunique() != 1:
            raise ValueError(f'Ambiguous observations: {place} {year}')
        obs = group.loc[group.Spore.notna()]
        positive = group.loc[group.disease.gt(0), 'Date']
        if positive.empty:
            raise ValueError(f'No first disease-positive date: {place} {year}')
        upper = positive.min()
        negative = group.loc[group.disease.eq(0) & group.Date.lt(upper), 'Date']
        if negative.empty:
            raise ValueError(f'No preceding disease-negative date: {place} {year}')
        lower = negative.max()
        stages = {}
        for stage in [15, 53]:
            reached = group.loc[group.GS.ge(stage)]
            if not reached.empty:
                row = reached.iloc[0]
                stages[stage] = (row.Date, 'workbook', float(row.GS))
            elif manifest is not None:
                ref = manifest.loc[manifest.Place.eq(place) & manifest.Year.eq(year)]
                if len(ref) != 1:
                    raise ValueError(f'Missing or ambiguous manifest reference: {place} {year}')
                stages[stage] = (pd.Timestamp(ref.iloc[0][f'rb_bbch{stage}']), 'A input reference', None)
            else:
                stages[stage] = (pd.NaT, 'missing', None)
        ax.bar(obs.Date, obs.Spore, width=1.6, color=blue, zorder=3)
        zeros = obs.loc[obs.Spore.eq(0)]
        ax.scatter(zeros.Date, zeros.Spore, s=15, facecolors='white', edgecolors=blue,
                   zorder=4, clip_on=False)
        peak = max(1, float(obs.Spore.max()))
        ax.set_ylim(0, peak * 1.6)
        all_dates = list(group.Date) + [v[0] for v in stages.values() if pd.notna(v[0])]
        ax.set_xlim(min(all_dates) - pd.Timedelta(days=4), max(all_dates) + pd.Timedelta(days=6))
        same_date = stages[15][0] == stages[53][0]
        for stage, color, label_y in [(15, green, .98), (53, purple, .89)]:
            date, source, recorded_gs = stages[stage]
            if pd.isna(date):
                ax.text(.02, label_y, f'BBCH {stage}: not recorded', transform=ax.transAxes,
                        color=color, va='top', fontsize=9)
                continue
            lo, hi = ((0, .48) if stage == 15 else (.48, .84)) if same_date else (0, .84)
            ax.axvline(date, ymin=lo, ymax=hi, color=color,
                       linestyle=':' if source == 'A input reference' else '--', linewidth=1.6, zorder=2)
            suffix = '*' if source == 'A input reference' else ''
            qualifier = f' [GS {recorded_gs:g}]' if recorded_gs is not None and recorded_gs > stage else ''
            label = f'BBCH {stage}{suffix}: {date:%m-%d}{qualifier}'
            ax.text(date, label_y, label, transform=ax.get_xaxis_transform(), color=color,
                    ha='left', va='top', fontsize=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1})
        trans = ax.get_xaxis_transform()
        ax.plot([lower, upper], [.76, .76], color=red, lw=3.5, transform=trans, zorder=5)
        ax.plot(lower, .76, 'o', mfc='white', mec=red, ms=6, transform=trans, zorder=6)
        ax.plot(upper, .76, 'o', color=red, ms=6, transform=trans, zorder=6)
        ax.text(upper, .80, f'{lower:%m-%d} to {upper:%m-%d}', ha='right', va='bottom',
                color=red, fontsize=9, transform=trans)
        ax.set_title(f'({chr(97+i)}) {SITES[place]} {year}  ({place})', loc='left', fontweight='bold', pad=12)
        ax.set_ylabel('Spore count per collection')
        ax.set_xlabel('Date')
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.grid(axis='y', color='#E4E8EB', zorder=0)
        row = dict(site=SITES[place], year=year, n_spore_observations=len(obs),
                   total_spores=obs.Spore.sum(), last_negative=lower.date(), first_positive=upper.date())
        for stage, (date, source, gs) in stages.items():
            row.update({f'bbch{stage}_date': date.date() if pd.notna(date) else None,
                        f'bbch{stage}_source': source, f'bbch{stage}_recorded_GS': gs})
        audit.append(row)
    fig.suptitle('Observed spores, phenology and disease onset', fontsize=21, y=.983)
    handles = [Patch(color=blue, label='Recorded spore count'),
               Line2D([], [], color=green, ls='--', label='BBCH 15'),
               Line2D([], [], color=purple, ls='--', label='BBCH 53'),
               Line2D([], [], color=red, lw=3, label='Last negative to first positive')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95), ncol=4, frameon=False)
    fig.subplots_adjust(left=.075, right=.975, bottom=.13, top=.885, hspace=.48, wspace=.20)
    notes = ('Bars: raw counts at recorded collection dates; no daily interpolation. Open blue circles: observed zeros. Panel scales differ.\n'
             'Red segment: observed disease-onset interval (open endpoint = last negative; filled endpoint = first positive); height is illustrative.\n'
             'Phenology: first recorded GS reaching or exceeding each stage; brackets show a higher recorded GS. Coincident dates share a split vertical line.\n'
             + ('* Dotted lines: 2024 phenology from A-run inputs because GS is absent in the workbook.' if not args.workbook_only else 'Missing phenology is labelled as not recorded.'))
    fig.text(.075, .028, notes, ha='left', va='bottom', fontsize=9, linespacing=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(args.output.with_suffix('.' + ext), dpi=220, facecolor='white')
    plt.close(fig)
    report = pd.DataFrame(audit)
    report.to_csv(args.output.with_name(args.output.name + '_dates.csv'), index=False)
    print(report.to_string(index=False))
    print(f'\nSaved: {args.output}.png and .pdf')


if __name__ == '__main__':
    main()
