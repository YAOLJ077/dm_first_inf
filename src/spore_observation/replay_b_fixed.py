#!/usr/bin/env python3
"""Replay saved scheme-B selections and export cohorts; never calibrate/select.

The +1-day arm replays the already saved, test-selected exploratory choice.
Only the original run's frozen source snapshot is imported by this program.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

ROOT = Path(__file__).resolve().parents[2]
RULES = ('min_train', 'train_plus1_test_selected')
SCHEMES = ('losyo', 'site', 'year')
CORE = None
CASES = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def initialize(snapshot, manifest):
    global CORE, CASES
    sys.path.insert(0, snapshot)
    CORE = importlib.import_module('src.analyze_fsim_parameters')
    if Path(CORE.__file__).resolve().parent.parent != Path(snapshot).resolve():
        raise RuntimeError('Wrong analysis module imported; frozen snapshot isolation failed')
    if Path(sys.modules['model.first_infection'].__file__).resolve().parent.parent != Path(snapshot).resolve():
        raise RuntimeError('Wrong model imported; frozen snapshot isolation failed')
    CORE.ROOT = ROOT
    cases, errors, _, _, _ = CORE.load_cases(Path(manifest))
    if errors:
        raise ValueError(f'Baseline case loading failed: {errors}')
    CASES = {case['input_key']: case for case in cases}


def check_expected(record, expected):
    """Missing predictions are outcomes, not zero errors or replay failures."""
    import pandas as pd
    fields = ['status', 'n_windows', 'weather_policy', 'predicted_start', 'predicted_end',
              'signed_interval_error_days', 'interval_distance_days', 'window_width_days', 'overlap']
    differences = []
    for name in fields:
        want = expected.get(name)
        got = record.get(name)
        if want is None or (not isinstance(want, str) and pd.isna(want)):
            if got is not None:
                differences.append(name)
        elif name in ('predicted_start', 'predicted_end'):
            if got is None or pd.Timestamp(got) != pd.Timestamp(want):
                differences.append(name)
        elif name.endswith('_days'):
            if got is None or not math.isclose(float(got), float(want), rel_tol=0, abs_tol=1e-8):
                differences.append(name)
        elif got != want:
            differences.append(name)
    return differences


TABLE_NAMES = ('attached_inventory', 'dispersal_cohorts', 'deposition',
               'detached_survival', 'primary_chain', 'primary_chain_gated', 'primary_chain_unique')


def replay(task):
    import pandas as pd
    started = time.monotonic()
    case = CASES[task['input_key']]
    cfg = task['config']
    cs = CORE.CropSeasonFirst(case['rb'], cfg)
    engine = CORE.FirstInfection(cs, cfg, use_wind_gate=False)
    engine.stage_attached_and_dispersal()
    engine.stage_deposition()
    engine.stage_detached_survival()
    engine.stage_primary_chain()
    cs.apply_bbch_15_gate()
    cs.collapse_equivalent_cohorts()
    windows = cs.primary_chain_unique
    record = {k: task[k] for k in ('rule', 'scheme', 'fold', 'input_key', 'case_role',
                                  'selection_uses_test_data', 'evaluation_role')}
    record.update({k: task['selection'][k] for k in ('VPD_w', 'I_req', 'RH_sp', 'MP_sp', 'u_tr', 'selected_trial')})
    record.update(status='no_prediction', n_windows=0 if windows is None else len(windows),
                  observed_L=case['L'], observed_U=case['U'], overlap=False,
                  weather_policy=case['weather_policy'], saved_training_MAE_days=task['selection']['training_MAE_days'])
    if windows is not None and not windows.empty:
        first = windows.sort_values(['oil_spot_start', 'infection_datetime']).iloc[0]
        record.update(status='predicted', **CORE.score_window(
            pd.Timestamp(case['L']), pd.Timestamp(case['U']),
            pd.Timestamp(first.oil_spot_start), pd.Timestamp(first.oil_spot_end)))
    differences = check_expected(record, task['expected']) if task['case_role'] == 'test' else []
    record['baseline_match'] = not differences if task['case_role'] == 'test' else None
    record['baseline_mismatch_fields'] = ','.join(differences)
    out = Path(task['output'])
    artifacts = []
    for name in TABLE_NAMES:
        frame = getattr(cs, name)
        if frame is None or (frame.empty and len(frame.columns) == 0):
            frame = pd.DataFrame(columns=['Num'])
        save_frame(out / f'{name}.csv', frame)
        artifacts.append(f'{name}.csv')
    # Preserve all cohort contributions before infection-time deduplication.
    fate = cs.detached_survival.copy()
    raw_ids = set(cs.primary_chain.Num) if cs.primary_chain is not None and not cs.primary_chain.empty else set()
    gated_ids = set(cs.primary_chain_gated.Num) if cs.primary_chain_gated is not None and not cs.primary_chain_gated.empty else set()
    selected_ids = set(windows.Num) if windows is not None and not windows.empty else set()
    fate['completed_primary_chain'] = fate.Num.isin(raw_ids)
    fate['passed_host_gate'] = fate.Num.isin(gated_ids)
    fate['retained_after_collapse'] = fate.Num.isin(selected_ids)
    fate['outcome'] = ['retained_window' if n in selected_ids else
                       'merged_same_infection_time' if n in gated_ids else
                       'excluded_host_gate' if n in raw_ids else 'primary_chain_not_completed'
                       for n in fate.Num]
    # The frozen engine does not expose individual early-exit reasons.
    save_frame(out/'cohort_outcomes.csv', fate)
    artifacts.append('cohort_outcomes.csv')
    record['model_seconds'] = time.monotonic() - started
    save_json(out/'prediction.json', record)
    save_json(out/'selection.json', task['selection'])
    save_json(out/'effective_runtime_config.json', cfg)
    artifacts += ['prediction.json', 'selection.json', 'effective_runtime_config.json']
    save_json(out/'completed.json', {'sha256': {name: sha(out/name) for name in artifacts}})
    return record


def resumed_record(task):
    out = Path(task['output'])
    marker = out/'completed.json'
    if not marker.exists():
        return None
    try:
        saved = read_json(marker)['sha256']
        if all((out/name).is_file() and sha(out/name) == digest for name, digest in saved.items()):
            return read_json(out/'prediction.json')
    except (OSError, ValueError, KeyError):
        pass
    return None


def prepare(args):
    import pandas as pd
    baseline = args.baseline.resolve()
    identity = read_json(baseline/'run_identity.json')
    snapshot = baseline/'source_snapshot/src/tpe_m0_reference'
    settings = identity['settings']
    if settings['plans']['B'] != {'wind_gate': False, 'search_space': {'VPD_w': [1, 6], 'I_req': [100, 600]}}:
        raise ValueError('Baseline is not the agreed wide-space two-parameter B design')
    if settings['seed'] != 123 or settings['fixed_parameters'] != {'RH_sp': 80, 'MP_sp': 3}:
        raise ValueError('Unexpected B baseline seed/fixed parameters')
    verified = {}
    for name, digest in identity['source_sha256'].items():
        if name.startswith('src/tpe_m0_reference/'):
            file = baseline/'source_snapshot'/name
            if sha(file) != digest:
                raise ValueError(f'Frozen source hash mismatch: {file}')
            verified[str(file.relative_to(baseline))] = digest
    for name, digest in identity['input_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Original input changed: {name}')
    manifest = ROOT/settings['manifest']
    manifest_frame = pd.read_csv(manifest)
    keys = manifest_frame.input_key.tolist()
    versions = {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'matplotlib')}}
    for name in ('python', 'numpy', 'pandas'):
        if versions[name] != identity['versions'][name]:
            raise ValueError(f'Replay requires {name}={identity["versions"][name]}; found {versions[name]}')
    tasks, selections = [], {}
    source_files = {'run_identity.json': sha(baseline/'run_identity.json')}
    matched_folds = set()
    for scheme in args.schemes:
        folds = sorted((baseline/f'B/{scheme}/seed_123/folds').iterdir())
        for fold in folds:
            if not fold.is_dir() or fold.name.startswith('.') or (args.folds and fold.name not in args.folds):
                continue
            matched_folds.add(fold.name)
            for rule in args.rules:
                directory = fold/f'budget_{args.budget}'
                if rule != 'min_train':
                    directory /= 'exploratory'
                selection = read_json(directory/'selection.json')
                cfg = read_json(directory/'effective_runtime_config.json')
                exploratory = rule != 'min_train'
                expected_rule = 'training_tolerance_then_test_MAE' if exploratory else 'minimum_training_MAE'
                if (selection['calibration_status'] != 'selected' or selection['n_trials'] != args.budget
                        or selection['selection_uses_test_data'] != exploratory
                        or selection['selection_rule'] != expected_rule):
                    raise ValueError(f'Invalid saved selection: {directory}')
                if exploratory and selection['selection_tolerance_days'] != 1.0:
                    raise ValueError('Expected the saved +1-day exploratory selection')
                train, test = selection['training_keys'], selection['test_keys']
                if set(train) & set(test) or sorted(train+test) != sorted(keys):
                    raise ValueError('Invalid train/test partition')
                expected_test = ([fold.name] if scheme == 'losyo' else
                    manifest_frame.loc[manifest_frame['place_code' if scheme == 'site' else 'Year'].astype(str).eq(fold.name), 'input_key'].tolist())
                if sorted(test) != sorted(expected_test):
                    raise ValueError(f'Fold membership mismatch: {fold}')
                if not (1 <= selection['VPD_w'] <= 6 and 100 <= selection['I_req'] <= 600
                        and selection['RH_sp'] == 80 and selection['MP_sp'] == 3 and selection['u_tr'] == 0):
                    raise ValueError('Saved parameters do not match B design')
                if (cfg['zoospore_alive']['vpd'] != selection['VPD_w']
                        or cfg['infection']['infection_condition'] != selection['I_req']
                        or cfg['mp_tmp']['rh_wet_night'] != 80 or cfg['fixed']['min_moist_hours'] != 3
                        or cfg['dispersal']['use_wind_gate'] is not False):
                    raise ValueError('Saved runtime configuration disagrees with selected parameters')
                predfile = directory/('selected_predictions.csv' if exploratory else 'held_out_predictions.csv')
                frame = pd.read_csv(predfile)
                if frame.input_key.duplicated().any() or sorted(frame.input_key) != sorted(test):
                    raise ValueError(f'Invalid reference predictions: {predfile}')
                expected = {r['input_key']: {k: (None if pd.isna(v) else v) for k, v in r.items()}
                            for r in frame.to_dict('records')}
                selection_id = f'{rule}/{scheme}/{fold.name}'
                selections[selection_id] = selection
                for file in (directory/'selection.json', directory/'effective_runtime_config.json', predfile):
                    source_files[str(file.relative_to(baseline))] = sha(file)
                for role, members in [('train', train), ('test', test)]:
                    if args.case_role != 'all' and args.case_role != role:
                        continue
                    for key in members:
                        tasks.append(dict(rule=rule, scheme=scheme, fold=fold.name, input_key=key,
                            case_role=role, selection=selection, config=cfg,
                            selection_uses_test_data=exploratory, evaluation_role=selection['evaluation_role'],
                            expected=expected.get(key), output=str(args.output_dir/selection_id/role/key)))
    if not tasks or (args.folds and set(args.folds) != matched_folds):
        raise ValueError('No matching tasks or unknown --folds name')
    run_identity = dict(script_sha256=sha(__file__), baseline=str(baseline), budget=args.budget,
        schemes=args.schemes, rules=args.rules, folds=args.folds, case_role=args.case_role,
        source_hashes=verified, input_hashes=identity['input_sha256'], selection_files=source_files,
        versions=versions, fig_dir=str(args.fig_dir.resolve()),
        tasks=[{k: t[k] for k in ('rule', 'scheme', 'fold', 'case_role', 'input_key')} for t in tasks])
    return tasks, selections, run_identity, snapshot, manifest


def summarize(records, selections, output):
    import pandas as pd
    frame = pd.DataFrame(records)
    for col in ('interval_distance_days', 'signed_interval_error_days', 'window_width_days'):
        if col not in frame:
            frame[col] = float('nan')
    frame = frame.sort_values(['rule', 'scheme', 'fold', 'case_role', 'input_key'])
    save_frame(output/'predictions.csv', frame)
    folds = []
    for (rule, scheme, fold), group in frame.groupby(['rule', 'scheme', 'fold'], sort=True):
        s = selections[f'{rule}/{scheme}/{fold}']
        r = dict(rule=rule, scheme=scheme, fold=fold, selection_uses_test_data=s['selection_uses_test_data'],
                 saved_training_MAE_days=s['training_MAE_days'])
        for role in ('train', 'test'):
            sub = group[group.case_role.eq(role)]
            predicted = sub[sub.status.eq('predicted')]
            r.update({f'{role}_n':len(sub), f'{role}_predicted':len(predicted),
                f'{role}_conditional_MAE_days':predicted.interval_distance_days.mean(),
                f'{role}_MAE_days':predicted.interval_distance_days.mean() if len(sub) and len(predicted)==len(sub) else float('nan')})
        r['training_match'] = (math.isclose(r['train_MAE_days'], s['training_MAE_days'], abs_tol=1e-8, rel_tol=0)
                               if r['train_n'] else None)
        folds.append(r)
    folds = pd.DataFrame(folds)
    save_frame(output/'fold_summary.csv', folds)
    methods = []
    for (rule, scheme), g in frame[frame.case_role.eq('test')].groupby(['rule', 'scheme']):
        ok = g[g.status.eq('predicted')]
        methods.append(dict(rule=rule, scheme=scheme, n_test=len(g), n_predictions=len(ok),
            n_no_predictions=len(g)-len(ok), MAE_days=ok.interval_distance_days.mean() if len(ok)==len(g) else float('nan'),
            conditional_MAE_days=ok.interval_distance_days.mean(), mean_signed_error_days=ok.signed_interval_error_days.mean(),
            mean_window_width_days=ok.window_width_days.mean(), selection_uses_test_data=rule!=RULES[0]))
    save_frame(output/'method_summary.csv', pd.DataFrame(methods))
    failures = frame[frame.case_role.eq('test') & frame.baseline_match.eq(False)]
    save_frame(output/'reproduction_checks.csv', frame[frame.case_role.eq('test')][
        ['rule','scheme','fold','input_key','status','baseline_match','baseline_mismatch_fields']])
    return frame, folds, len(failures)==0 and not folds.training_match.eq(False).any()


def plot(frame, folds, fig_dir, budget):
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'fsim_spore_replay_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(15, 12))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2, squeeze=False)
    labels = {'losyo':'LOSYO', 'site':'Leave-one-site-out', 'year':'Leave-one-year-out'}
    for row, scheme in enumerate(SCHEMES):
        for col, rule in enumerate(RULES):
            ax = axes[row, col]
            data = folds[folds.scheme.eq(scheme) & folds.rule.eq(rule)].sort_values('fold')
            if data.empty:
                ax.set_visible(False)
                continue
            x = np.arange(len(data)); test = data.test_conditional_MAE_days.to_numpy(float)
            train = data.saved_training_MAE_days.to_numpy(float)
            bars = ax.bar(x-.19, train, .38, color='#aab6c6', label='Saved training MAE')
            bars2 = ax.bar(x+.19, test, .38, color='#3b80a1' if col==0 else '#df8936', label='Replayed test MAE*')
            for batch in (bars,bars2):
                for b in batch:
                    if np.isfinite(b.get_height()):
                        ax.annotate(f'{b.get_height():.2f}', (b.get_x()+b.get_width()/2,b.get_height()),
                                    xytext=(0,4), textcoords='offset points', ha='center',fontsize=8)
            names = [' '.join(f.split('_')[::2]) if '_' in f else f for f in data.fold]
            ax.set_xticks(x, [f'{n}\n({int(p)}/{int(t)})' for n,p,t in zip(names,data.test_predicted,data.test_n)])
            ax.set_ylabel('Absolute interval error (days)')
            ax.set_ylim(0, max(1, np.nanmax(np.r_[train,test]))*1.27)
            subset = frame[frame.rule.eq(rule)&frame.scheme.eq(scheme)&frame.case_role.eq('test')]
            ok = subset[subset.status.eq('predicted')]
            value = f'{ok.interval_distance_days.mean():.2f}' if len(ok) else 'NA'
            qualifier = 'conditional ' if len(ok)!=len(subset) else ''
            ax.set_title(f'{labels[scheme]}\n{qualifier}test MAE: {value} d; predictions {len(ok)}/{len(subset)}',loc='left',fontsize=11)
            ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            ax.legend(fontsize=8,loc='upper right')
    fig.suptitle(f'B fixed-parameter replay | {budget} original training trials per fold',fontsize=17,y=.99)
    fig.text(.27,.947,'Minimum training MAE',ha='center',color='#3b80a1',fontsize=13)
    fig.text(.75,.947,'Training minimum + 1 day; then minimum test MAE\nExploratory: test outcomes used for selection',ha='center',color='#df8936',fontsize=11)
    fig.text(.07,.025,'* Missing predictions are retained; incomplete test MAE is conditional on predicted cases.\n'
             'Parentheses: predicted / held-out cases. Training bars use saved selection metrics; replay checks are in fold_summary.csv.\n'
             'No new calibration or parameter selection. The right column is not independent test performance.',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.925),h_pad=3)
    for ext in ('png','pdf'):
        fig.savefig(fig_dir/f'b_fixed_selection_comparison.{ext}',dpi=160)
    fig.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'result/fsim_abc_tpe_seed123')
    parser.add_argument('--budget',type=int,default=1024,choices=(1024,2048))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'result/spore_observation/plan1_b_fixed')
    parser.add_argument('--fig-dir',type=Path,default=ROOT/'fig/spore_observation/plan1_b_fixed')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--schemes',nargs='+',choices=SCHEMES,default=list(SCHEMES))
    parser.add_argument('--rules',nargs='+',choices=RULES,default=list(RULES))
    parser.add_argument('--folds',nargs='+',help='Optional exact fold names, e.g. DA_MPT_2022 or DA')
    parser.add_argument('--case-role',choices=('all','test','train'),default='all')
    parser.add_argument('--design-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.workers<1 or len(set(args.rules))!=len(args.rules) or len(set(args.schemes))!=len(args.schemes):
        parser.error('Use positive workers and unique schemes/rules')
    args.output_dir=args.output_dir.resolve(); args.fig_dir=args.fig_dir.resolve()
    baseline=args.baseline.resolve()
    for destination in (args.output_dir, args.fig_dir):
        if destination == baseline or baseline in destination.parents or destination in baseline.parents:
            parser.error('Replay output must be separate from the original calibration directory')
    tasks,selections,identity,snapshot,manifest=prepare(args)
    print(f'Fixed replay: {len(selections)} selections; {len(tasks)} case runs; '
          f'{sum(t["case_role"]=="test" for t in tasks)} held-out predictions. No optimization.',flush=True)
    if args.design_only:
        print('Frozen source, input hashes, partitions, selections and numerical versions verified.')
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output_dir/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another replay is writing this output directory')
        ipath=args.output_dir/'run_identity.json'
        if ipath.exists():
            if not args.resume or read_json(ipath)!=identity:
                raise SystemExit('Output already exists. Use --resume with identical design/code or choose new result and fig directories.')
        elif any(p.name not in ('.run.lock', '.DS_Store') and not p.name.startswith('._')
                 for p in args.output_dir.iterdir()):
            raise SystemExit('Refusing a nonempty output directory without matching run identity')
        save_json(ipath,identity)
        status={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'total':len(tasks),'completed':0}
        save_json(args.output_dir/'run_metadata.json',status)
        records=[]; pending=[]
        for task in tasks:
            cached=resumed_record(task) if args.resume else None
            if cached is not None: records.append(cached)
            else: pending.append(task)
        print(f'Restored {len(records)} completed runs; computing {len(pending)}.',flush=True)
        try:
            if pending:
                with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                    initializer=initialize,initargs=(str(snapshot),str(manifest))) as pool:
                    futures={pool.submit(replay,t):t for t in pending}
                    for future in as_completed(futures):
                        record=future.result();records.append(record)
                        status.update(completed=len(records),updated_utc=datetime.now(timezone.utc).isoformat())
                        save_json(args.output_dir/'run_metadata.json',status)
                        print(f'[{len(records)}/{len(tasks)}] {record["rule"]}/{record["scheme"]}/{record["fold"]}/'
                              f'{record["case_role"]}/{record["input_key"]}: {record["status"]}',flush=True)
            frame,folds,passed=summarize(records,selections,args.output_dir)
            plot(frame,folds,args.fig_dir,args.budget)
            status.update(status='complete' if passed else 'reproduction_mismatch',completed=len(records),
                          finished_utc=datetime.now(timezone.utc).isoformat(),baseline_reproduced=passed)
            save_json(args.output_dir/'run_metadata.json',status)
            if not passed:
                raise SystemExit('Replay mismatch; inspect reproduction_checks.csv and fold_summary.csv. Do not use as matched baseline.')
            print(f'Completed and verified. Results: {args.output_dir}\nFigures: {args.fig_dir}',flush=True)
        except Exception as exc:
            status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            save_json(args.output_dir/'run_metadata.json',status)
            raise


if __name__=='__main__':
    main()
