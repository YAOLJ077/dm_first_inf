"""Independent 3p grid: refine VPD/I within each fixed integer wind layer."""
import itertools
import json
import math
import time

import numpy as np
import pandas as pd

import fsim_grid_global

NAMES = ('VPD_w', 'I_req', 'RH_sp', 'MP_sp', 'u_tr')


def key(p):
    return tuple(float(p[n]) for n in NAMES)


def parameter(x, y, rh, mp, wind):
    return dict(VPD_w=float(x), I_req=float(y), RH_sp=float(rh), MP_sp=int(mp), u_tr=float(wind))


def coarse_design(s, arm):
    x, y = (s['local_grid']['coarse_values'][n] for n in NAMES[:2])
    rh = [s['fixed_2p']['RH_sp']]
    mp = [s['fixed_2p']['MP_sp']]
    wind = s['local_grid']['coarse_values']['u_tr']
    params = [parameter(a,b,c,d,w) for a,b,c,d,w in itertools.product(x,y,rh,mp,wind)]
    first = next(p for p in params if p == s['initial_point'])
    params.remove(first)
    rng = np.random.default_rng(s['grid_order_seed'])
    params = [first] + [params[int(i)] for i in rng.permutation(len(params))]
    cells = [(a,b,c,d,r,m,w) for (a,b),(c,d),r,m,w in itertools.product(
        zip(x[:-1],x[1:]), zip(y[:-1],y[1:]), rh,mp,wind)]
    return params, cells


def stencil(cell):
    a,b,c,d,r,m,w = cell
    x,y = (a+b)/2, (c+d)/2
    # Give all competitive cells their centre before their edge points.
    coords = [(x,y),(a,y),(b,y),(x,c),(x,d),(a,c),(a,d),(b,c),(b,d)]
    return [parameter(u,v,r,m,w) for u,v in coords]


def proposals(s, cells, records, limit, round_number):
    seen = {key(t['params']) for t in records}
    feasible = [t for t in records if t['metrics']['feasible']]
    if feasible:
        best = min(t['metrics']['training_MAE_days'] for t in feasible)
        near = [t['params'] for t in feasible if t['metrics']['training_MAE_days'] <=
                best + s['local_grid']['margin_days'] + s['tie_atol_days']]
    else:
        near = None  # Without a feasible seed, refine broadly within the budget.
    eligible = []
    for cell in cells:
        a,b,c,d,r,m,w = cell
        if near is None or any(p['RH_sp']==r and p['MP_sp']==m and p['u_tr']==w and
                               a<=p['VPD_w']<=b and c<=p['I_req']<=d for p in near):
            eligible.append([p for p in stencil(cell) if key(p) not in seen])
    rng = np.random.default_rng(s['grid_order_seed'] + round_number)
    eligible = [eligible[int(i)] for i in rng.permutation(len(eligible))]
    selected = []
    for level in itertools.zip_longest(*eligible):
        for p in level:
            if p is not None and key(p) not in seen:
                seen.add(key(p)); selected.append(p)
                if len(selected) >= limit:
                    return selected
    return selected


def split_completed_cells(cells, records):
    seen = {key(t['params']) for t in records}
    leaves = []
    for cell in cells:
        a,b,c,d,r,m,w = cell
        if all(key(p) in seen for p in stencil(cell)):
            x,y=(a+b)/2,(c+d)/2
            leaves.extend((u,v,q,z,r,m,w) for (u,v),(q,z) in itertools.product([(a,x),(x,b)],[(c,y),(y,d)]))
        else:
            leaves.append(cell)
    return leaves


def run_local_grid(s, jobs, evaluator, store, output, progress, runner, on_complete):
    core = runner.core
    saved = {j['id']: store.trials(j['id']) for j in jobs}
    records = {j['id']: [] for j in jobs}
    cells = {j['id']: coarse_design(s,j['arm'])[1] for j in jobs}
    rounds = s['local_grid']['refinement_rounds']
    for round_number in range(rounds + 1):
        plans, synthetic = {}, []
        for job in jobs:
            jid=job['id']; current=records[jid]
            if round_number == 0:
                plan=coarse_design(s,job['arm'])[0]
            else:
                remaining=job['n_trials']-len(current)
                if remaining<=0:
                    plan=[]
                else:
                    limit=math.ceil(remaining/(rounds-round_number+1))
                    plan=proposals(s,cells[jid],current,limit,round_number)
            directory=output/jid
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/f'grid_round_{round_number}.json'
            audit=dict(round=round_number,training_keys=job['train'],params=plan,
                       previous_trials=len(current),candidate_cap=job['n_trials'])
            if path.exists() and json.loads(path.read_text()) != audit:
                raise ValueError('Local Grid plan differs from saved training-only replay')
            core.write_json(path,audit)
            # A synthetic arm routes this fold's plan through global cache deduplication.
            arm=jid+'_grid'; plans[arm]=plan
            synthetic.append(dict(job,arm=arm))
        round_output=output/'grid_rounds'/f'round_{round_number}'
        round_output.mkdir(parents=True,exist_ok=True)
        print(f'LOCAL GRID round {round_number}: {sum(map(len,plans.values()))} fold candidates before deduplication',flush=True)
        fsim_grid_global.run_grid_models(s,synthetic,plans,evaluator,store,round_output,progress,core,runner.timed_worker)
        for job in jobs:
            jid=job['id']
            for p in plans[jid+'_grid']:
                started=time.perf_counter()
                predictions,duplicates=evaluator.evaluate(p,job['train'],cached_only=True)
                metrics=core.training_metrics(predictions,job['train'])
                i=len(records[jid])
                if i<len(saved[jid]):
                    record=saved[jid][i]
                    if (record['params']!=p or record['number']!=i or record['round']!=round_number or
                        record['training_keys']!=job['train'] or record['metrics']!=metrics or record['seed']!=job['seed']):
                        raise ValueError('Local Grid checkpoint mismatch')
                else:
                    record=dict(number=i,round=round_number,seed=job['seed'],params=p,training_keys=job['train'],metrics=metrics,
                                requested_model_evaluations=len(predictions)+len(duplicates),
                                elapsed_seconds=time.perf_counter()-started)
                    store.save_trial(jid,record)
                records[jid].append(record)
            cells[jid]=split_completed_cells(cells[jid],records[jid])
        print(f'LOCAL GRID round {round_number} training records committed',flush=True)
    # No held-out results have influenced any refinement round.
    for job in jobs:
        trials=records[job['id']]; directory=output/job['id']
        if len(trials)<len(saved[job['id']]):
            raise ValueError('Saved trials exceed deterministic local design')
        core.write_json(directory/'job.json',job)
        pd.DataFrame([dict(round=t['round'],**core.flat_trial(t)) for t in trials]).to_csv(directory/'trials.csv',index=False)
        curve=[]; best=None
        for i,t in enumerate(trials):
            if t['metrics']['feasible']:
                value=t['metrics']['training_MAE_days']; best=value if best is None else min(best,value)
            curve.append(dict(completed_trials=i+1,round=t['round'],best_training_MAE_days=best))
        pd.DataFrame(curve).to_csv(directory/'convergence.csv',index=False)
        budgets={len(trials)}
        if len(trials)>=s['common_budget']:
            budgets.add(s['common_budget'])
        for budget in sorted(budgets):
            held=runner.select_budget(s,job,trials,budget,evaluator,store,directory)
            if budget==len(trials):
                final=store.heldout(job['id'],-1)
                if final is None:
                    store.save_heldout(job['id'],-1,held)
                elif final!=held:
                    raise ValueError('Final local Grid selection differs from checkpoint')
        on_complete(job,dict(status='complete',n_trials=len(trials),candidate_cap=job['n_trials'],
                             stopping_reason='budget_reached' if len(trials)==job['n_trials'] else 'refinement_round_limit_or_no_new_candidates'))
