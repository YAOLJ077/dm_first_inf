#!/usr/bin/env python3
"""Audit completed grouped 2p/3p/4p results without recalibration or model edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {'2p': ['VPD_w', 'I_req'], '3p': ['VPD_w', 'I_req', 'u_tr'],
          '4p': ['VPD_w', 'I_req', 'RH_sp', 'MP_sp']}


def analyze(output):
    roots = [ROOT / 'result/fsim_unified_joblib_local', ROOT / 'result/fsim_m0_3p_joblib']
    identities = [json.loads((p / 'run_identity.json').read_text()) for p in roots]
    assert identities[0]['input_sha256'] == identities[1]['input_sha256']
    for name, digest in identities[0]['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    shared = set(identities[0]['source_sha256']) & set(identities[1]['source_sha256'])
    assert all(identities[0]['source_sha256'][k] == identities[1]['source_sha256'][k] for k in shared)
    folds, oof, trials, audit = [], [], [], {}
    for root in roots:
        meta = json.loads((root / 'run_metadata.json').read_text())
        assert meta['status'] == 'complete'
        settings = json.loads((root / 'effective_settings.json').read_text())
        f = pd.read_csv(root / 'fold_summary.csv', dtype={'held_out': str})
        p = pd.read_csv(root / 'oof_predictions.csv', dtype={'held_out': str})
        folds.append(f)
        oof.append(p)
        audit[str(root.relative_to(ROOT))] = {'status': meta['status'], 'seeds': settings['seeds'],
            'settings': settings, 'files_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ('fold_summary.csv', 'oof_predictions.csv', 'run_identity.json')}}
        for row in f[f.view.eq('final')].itertuples():
            path = root / row.scheme / row.arm / f'seed_{row.seed}' / 'folds' / row.held_out / 'trials.csv'
            t = pd.read_csv(path)
            assert len(t) == row.n_trials
            assert np.isclose(t.loc[t.feasible, 'training_MAE_days'].min(), row.training_MAE_days)
            t = t.rename(columns={'trial': 'trial_index'})
            t['scheme'], t['arm'], t['held_out'] = row.scheme, row.arm, row.held_out
            t['source'] = str(path.relative_to(ROOT))
            trials.append(t)
    folds, oof, trials = pd.concat(folds, ignore_index=True), pd.concat(oof, ignore_index=True), pd.concat(trials, ignore_index=True)
    assert len(folds[folds.view.eq('final')]) == 36
    for _, p in oof.groupby(['scheme', 'arm', 'view', 'seed']):
        assert len(p) == p.input_key.nunique() == 6
    for r in oof[oof.status.eq('predicted')].itertuples():
        start, end, lo, hi = map(pd.Timestamp, [r.predicted_start, r.predicted_end, r.observed_L, r.observed_U])
        error = (end - lo).total_seconds()/86400 if end < lo else (start - hi).total_seconds()/86400 if start > hi else 0
        assert np.isclose(error, r.signed_interval_error_days)

    summary, fold_metrics = [], []
    for (scheme, arm, view), p in oof.groupby(['scheme', 'arm', 'view']):
        valid = p[p.status.eq('predicted')]
        e = valid.signed_interval_error_days
        summary.append(dict(scheme=scheme, arm=arm, view=view, n_predictions=len(valid), n_test=len(p),
            MAE_days=e.abs().mean() if len(valid)==6 else np.nan, conditional_MAE_days=e.abs().mean(),
            bias_days=e.mean(), max_absolute_error_days=e.abs().max(), n_within_10d=int(e.abs().le(10).sum()),
            n_trials_min=p.n_trials.min(), n_trials_max=p.n_trials.max()))
        for held, q in p.groupby('held_out'):
            v = q[q.status.eq('predicted')].signed_interval_error_days
            train = folds[folds.scheme.eq(scheme)&folds.arm.eq(arm)&folds.view.eq(view)&folds.held_out.eq(held)].iloc[0]
            fold_metrics.append(dict(scheme=scheme, arm=arm, view=view, held_out=held,
                train_MAE=train.training_MAE_days, test_MAE=v.abs().mean(), test_bias=v.mean(),
                gap_days=v.abs().mean()-train.training_MAE_days, n_predictions=len(v), n_test=len(q)))

    geometry, near_points, search, convergence = [], [], [], []
    groupcols = ['scheme', 'arm', 'held_out']
    for (scheme, arm, held), t in trials.groupby(groupcols, sort=True):
        feasible = t[t.feasible].copy()
        best = feasible.training_MAE_days.min()
        dim = arm[:2]
        tags = dict(scheme=scheme, arm=arm, held_out=held)
        search.append(dict(**tags, n_trials=len(t), n_feasible=len(feasible), feasible_fraction=len(feasible)/len(t),
            best_MAE=best, exact_ties=int(np.isclose(feasible.training_MAE_days,best,atol=1e-9,rtol=0).sum()),
            best_at_128=t.iloc[:128].loc[lambda x:x.feasible,'training_MAE_days'].min(),
            best_at_630=t.iloc[:630].loc[lambda x:x.feasible,'training_MAE_days'].min(),
            improvement_after_630=t.iloc[:630].loc[lambda x:x.feasible,'training_MAE_days'].min()-best))
        curve=t.training_MAE_days.where(t.feasible).cummin().ffill()
        convergence.extend(dict(**tags, completed_trials=i+1, training_MAE=value) for i,value in enumerate(curve))
        for tol in (0.5, 1.0):
            near=feasible[feasible.training_MAE_days.le(best+tol+1e-9)].drop_duplicates(PARAMS[dim])
            pts=near.assign(tolerance_days=tol)
            near_points.append(pts)
            for param in PARAMS[dim]:
                spec=audit[str(roots[1 if dim=='3p' else 0].relative_to(ROOT))]['settings']['search_space'][param]
                width=spec['high']-spec['low']
                geometry.append(dict(**tags, tolerance_days=tol, parameter=param, n_unique=len(near),
                    minimum=near[param].min(), q05=near[param].quantile(.05), median=near[param].median(),
                    q95=near[param].quantile(.95), maximum=near[param].max(),
                    sampled_span_fraction=(near[param].max()-near[param].min())/width,
                    lower_edge_fraction=near[param].le(spec['low']+.05*width).mean(),
                    upper_edge_fraction=near[param].ge(spec['high']-.05*width).mean(),
                    VPD_I_spearman=near.VPD_w.corr(near.I_req,method='spearman')
                    if len(near)>2 and near.VPD_w.nunique()>1 and near.I_req.nunique()>1 else np.nan))
    pooled=[]
    trials['dimension']=trials.arm.str[:2]
    for (scheme, dim, held), t in trials.groupby(['scheme','dimension','held_out']):
        f=t[t.feasible]; best=f.training_MAE_days.min()
        for tol in (.5,1.0):
            near=f[f.training_MAE_days.le(best+tol+1e-9)]
            for param in PARAMS[dim]:
                pooled.append(dict(scheme=scheme, dimension=dim, held_out=held, tolerance_days=tol,
                    best_training_MAE=best, parameter=param, minimum=near[param].min(), maximum=near[param].max(),
                    n_grid=int(near.arm.str.endswith('grid').sum()), n_tpe=int(near.arm.str.endswith('tpe').sum())))
    final=oof[oof.view.eq('final')]
    pairs=[]
    for scheme in ('site','year'):
        p=final[final.scheme.eq(scheme)&final.arm.str.endswith('tpe')]
        for left,right in [('2p_tpe','3p_tpe'),('2p_tpe','4p_tpe'),('3p_tpe','4p_tpe')]:
            a=p[p.arm.eq(left)].set_index('input_key'); b=p[p.arm.eq(right)].set_index('input_key')
            for key in a.index:
                ea,eb=a.loc[key,'signed_interval_error_days'],b.loc[key,'signed_interval_error_days']
                pairs.append(dict(scheme=scheme,left=left,right=right,input_key=key,
                    left_status=a.loc[key,'status'],right_status=b.loc[key,'status'],left_error=ea,right_error=eb,
                    delta_absolute_error=abs(eb)-abs(ea)))
    context=[]
    for row in pd.read_csv(ROOT/'result/fsim_inputs/site_year_manifest.csv').itertuples():
        d=json.loads((ROOT/row.weather_json).read_text())
        w=pd.DataFrame(d['weather_hourly']);w['datetime']=pd.to_datetime(w.datetime)
        w=w.drop_duplicates('datetime',keep='last')
        q=w[(w.datetime>=pd.Timestamp(row.rb_bbch15))&(w.datetime<pd.Timestamp(row.U)+pd.Timedelta(days=1))]
        obs=pd.read_csv(ROOT/row.observations_csv)
        coordinates=';'.join(obs.coordinates.dropna().astype(str).unique()) if 'coordinates' in obs else ''
        context.append(dict(input_key=row.input_key,latitude=d.get('latitude'),longitude=d.get('longitude'),
            observation_coordinates=coordinates,BBCH15=row.rb_bbch15,BBCH53=row.rb_bbch53,
            gate_to_first_positive_days=(pd.Timestamp(row.U)-pd.Timestamp(row.rb_bbch15)).days,
            wind_q50=q.wind_speed.quantile(.5),wind_q95=q.wind_speed.quantile(.95),wind_max=q.wind_speed.max(),
            fraction_wind_ge5=q.wind_speed.ge(5).mean()))
    tables={'input_context':pd.DataFrame(context),'method_diagnostics':pd.DataFrame(summary),'fold_diagnostics':pd.DataFrame(fold_metrics),
        'search_diagnostics':pd.DataFrame(search),'near_optimal_geometry':pd.DataFrame(geometry),
        'near_optimal_points':pd.concat(near_points,ignore_index=True),'pooled_near_optimal_geometry':pd.DataFrame(pooled),
        'training_convergence':pd.DataFrame(convergence),'paired_tpe_errors':pd.DataFrame(pairs),
        'selected_parameters':folds,'held_out_predictions':oof}
    output.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items(): table.to_csv(output/f'{name}.csv',index=False)
    audit['checks']={'same_inputs':True,'current_inputs_match_recorded_hashes':True,'same_shared_sources':True,'completed_final_folds':36,
        'errors_recomputed':True,'training_minima_checked':True,'near_optimal_note':
        'Sampled training loss <= method/fold minimum + tolerance; sample ranges are not confidence intervals or posterior probabilities.'}
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    plot(output,tables)
    print(tables['method_diagnostics'].query('view == "final"').to_string(index=False))
    print('\nSearch diagnostics\n'+tables['search_diagnostics'].to_string(index=False))


def plot(output,tables):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/fsim_diagnostic_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    colors=['#2878A0','#DC8F32','#4F995D']
    fig,axes=plt.subplots(3,2,figsize=(12,12),sharex=True,sharey=True)
    near=tables['near_optimal_points']; selected=tables['selected_parameters']
    for i,dim in enumerate(('2p','3p','4p')):
        for j,scheme in enumerate(('site','year')):
            ax=axes[i,j]
            for color,fold in zip(colors,('DA','GL','NN') if scheme=='site' else ('2022','2023','2024')):
                q=near[near.arm.eq(dim+'_tpe')&near.scheme.eq(scheme)&near.held_out.eq(fold)&near.tolerance_days.eq(1)]
                ax.scatter(q.VPD_w,q.I_req,s=13,alpha=.45,color=color,label=fold)
                p=selected[selected.arm.eq(dim+'_tpe')&selected.scheme.eq(scheme)&selected.held_out.eq(fold)&selected.view.eq('final')].iloc[0]
                ax.scatter(p.VPD_w,p.I_req,marker='*',s=180,color=color,edgecolors='black',linewidths=.6,zorder=5)
            ax.set(title=f'{dim} TPE: leave-one-{scheme}-out',xlim=(.9,8.1),ylim=(30,465))
            ax.axhline(450,color='gray',linestyle=':',linewidth=.8)
            ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8)
            if j==0:ax.set_ylabel('I_req')
            if i==2:ax.set_xlabel('VPD_w')
    fig.suptitle('Sampled near-optimal training space',fontsize=16)
    fig.text(.08,.02,'Points: training MAE within 1 day of each fold minimum; stars: selected parameters.\n'
             'Projection onto VPD/I; additional parameters vary. Sampling density is not probability or confidence.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(output/'near_optimal_space.png',dpi=180);fig.savefig(output/'near_optimal_space.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    p=tables['held_out_predictions'];p=p[p.view.eq('final')]
    arms=['2p_grid','2p_tpe','3p_grid','3p_tpe','4p_grid','4p_tpe']
    keys=sorted(p.input_key.unique())
    for ax,scheme in zip(axes,('site','year')):
        q=p[p.scheme.eq(scheme)].pivot(index='input_key',columns='arm',values='signed_interval_error_days').reindex(index=keys,columns=arms)
        im=ax.imshow(q.to_numpy(),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=220),aspect='auto')
        for row in range(6):
            for col in range(6):
                value=q.iloc[row,col]
                ax.text(col,row,'NA' if pd.isna(value) else f'{value:+.1f}',ha='center',va='center',fontsize=9,
                        color='white' if np.isfinite(value) and (value < -65 or value>150) else 'black')
        ax.set(xticks=range(6),xticklabels=arms,yticks=range(6),yticklabels=keys,title=f'Leave-one-{scheme}-out')
        ax.tick_params(axis='x',rotation=45)
    fig.colorbar(im,ax=axes,label='Signed interval error (days); positive = later',shrink=.8)
    fig.suptitle('Final held-out errors: failures retained; Grid budgets differ')
    fig.savefig(output/'held_out_error_heatmap.png',dpi=180);fig.savefig(output/'held_out_error_heatmap.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
    c=tables['training_convergence']
    for i,dim in enumerate(('2p','3p','4p')):
        for j,scheme in enumerate(('site','year')):
            ax=axes[i,j]
            for color,fold in zip(colors,('DA','GL','NN') if scheme=='site' else ('2022','2023','2024')):
                for method,style in [('grid','--'),('tpe','-')]:
                    q=c[c.arm.eq(dim+'_'+method)&c.scheme.eq(scheme)&c.held_out.eq(fold)]
                    ax.plot(q.completed_trials,q.training_MAE,color=color,linestyle=style,label=f'{fold} {method}')
            ax.set(title=f'{dim}: leave-one-{scheme}-out',ylim=(0,45));ax.grid(alpha=.2)
            ax.axvline(128,color='gray',alpha=.4,lw=.8);ax.axvline(630,color='gray',alpha=.4,lw=.8)
            ax.legend(fontsize=7,ncol=2)
            if j==0:ax.set_ylabel('Best training MAE (days)')
            if i==2:ax.set_xlabel('Completed training candidates')
    fig.suptitle('Training convergence: Grid versus TPE (early errors above 45 days clipped)')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output/'training_convergence.png',dpi=180);fig.savefig(output/'training_convergence.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'result/fsim_grouped_diagnosis_20260915')
    analyze(parser.parse_args().output)
