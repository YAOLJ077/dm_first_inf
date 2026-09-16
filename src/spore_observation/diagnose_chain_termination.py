#!/usr/bin/env python3
"""Read-only execution tracing of the frozen B primary chain; no calibration."""
import argparse
import ast
import inspect
import sys
import textwrap
from pathlib import Path

import pandas as pd
import replay_b_fixed as replay

ROOT = replay.ROOT
KEYS = ['rule', 'scheme', 'fold', 'case_role', 'input_key']
REASONS = [
    ('start >= end', 'no_sporangial_survival_window'),
    ('not wet.any()', 'no_wet_hours_before_sporangial_death'),
    ('release_time is None', 'release_condition_not_met'),
    ('rz >= np.searchsorted(self.time_index, np.datetime64(suz_end))', 'no_post_release_survival_window'),
    ('not wet2.any()', 'no_host_eligible_wet_hours'),
    ('infection_time is None', 'infection_condition_not_met'),
    ('len(idx1) == 0 or len(idx2) == 0', 'incubation_not_completed'),
]


def compare_tables(source_dir, output_dir):
    """Dates/IDs/states exact; allow only machine precision in float columns."""
    max_difference = 0.0
    for name in replay.TABLE_NAMES:
        left = pd.read_csv(source_dir/(name+'.csv'), float_precision='round_trip')
        right = pd.read_csv(output_dir/(name+'.csv'), float_precision='round_trip')
        if list(left.columns) != list(right.columns) or len(left) != len(right):
            raise ValueError(f'Table shape/columns changed: {name}')
        for col in left:
            if pd.api.types.is_float_dtype(left[col]) and pd.api.types.is_float_dtype(right[col]) and col != 'Num':
                if not left[col].isna().equals(right[col].isna()):
                    raise ValueError(f'Missingness changed: {name}/{col}')
                difference = (left[col]-right[col]).abs().max()
                if pd.notna(difference):
                    max_difference = max(max_difference, float(difference))
                if pd.notna(difference) and difference > 1e-12:
                    raise ValueError(f'Numerical state changed: {name}/{col}: {difference}')
            else:
                pd.testing.assert_series_equal(left[col], right[col], check_dtype=False, check_exact=True)
    return max_difference


def traced_call(engine, original):
    """Observe only the original function frame; never replace its calculations."""
    lines, first_line = inspect.getsourcelines(original)
    tree = ast.parse(textwrap.dedent(''.join(lines)))
    stops = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and len(node.body) == 1 and isinstance(node.body[0], ast.Continue):
            condition = ast.unparse(node.test)
            matching = [(i, label) for i, (test, label) in enumerate(REASONS) if test == condition]
            if len(matching) != 1:
                raise ValueError(f'Unrecognised frozen branch: {condition}')
            rank, label = matching[0]
            stops[first_line+node.body[0].lineno-1] = (rank, label, condition)
    successes = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'results.append']
    if len(stops) != len(REASONS) or len(successes) != 1:
        raise ValueError('Unexpected frozen primary-chain structure')
    stops[first_line+successes[0].lineno-1] = (7, 'completed_primary_chain', 'results.append')
    records = {}

    def tracer(frame, event, arg):
        if frame.f_code is not original.__code__:
            return None
        if event == 'line' and frame.f_lineno in stops:
            rank, reason, condition = stops[frame.f_lineno]
            values = frame.f_locals
            num = int(values['c'].Num)
            if num in records:
                return tracer
            row = dict(Num=num, terminal_reason=reason, source_line=frame.f_lineno,
                       branch_condition=condition, stage_index=rank,
                       sporangial_window_steps=int(values['end']-values['start']),
                       host_ready_datetime=getattr(engine.cs, 'bbch_15_datetime', pd.NaT))
            # Locals persist across loop iterations: only read fields reached by this cohort.
            if rank >= 1:
                row['wet_hours_before_sporangial_death'] = int(values['wet'].sum())
            if rank >= 3:
                row['release_datetime'] = values['release_time']
                row['zoospore_survival_end'] = values['suz_end']
            if rank >= 4:
                row['host_eligible_window_steps'] = max(0, int(values['end2']-values['start2']))
                row['host_eligible_wet_hours'] = int(values['wet2'].sum())
                host = row['host_ready_datetime']
                row['host_ready_at_or_after_suz_end'] = bool(pd.notna(host) and host >= values['suz_end'])
            if rank >= 6:
                row['infection_datetime_trace'] = values['infection_time']
                row['incubation_start_boundary_reached'] = bool(len(values['idx1']))
                row['incubation_end_boundary_reached'] = bool(len(values['idx2']))
            records[num] = row
        return tracer

    previous = sys.gettrace()
    if previous is not None:
        raise RuntimeError('An existing Python tracer is active; run without debugger/coverage')
    sys.settrace(tracer)
    try:
        original(engine)
    finally:
        sys.settrace(previous)
    expected = set(engine.cs.detached_survival.Num.astype(int))
    if set(records) != expected:
        raise ValueError('Not every detached cohort has exactly one traced terminal state')
    completed = set(engine.cs.primary_chain.Num.astype(int)) if not engine.cs.primary_chain.empty else set()
    if completed != {n for n, r in records.items() if r['terminal_reason'] == 'completed_primary_chain'}:
        raise ValueError('Trace completion disagrees with actual primary chain')
    return pd.DataFrame(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-dir', type=Path, default=ROOT/'result/spore_observation/plan1_b_fixed')
    parser.add_argument('--links-dir', type=Path, default=ROOT/'result/spore_observation/plan1_b_links')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'result/spore_observation/plan1_chain_diagnostics')
    parser.add_argument('--schemes', nargs='+', choices=replay.SCHEMES, default=['losyo'])
    parser.add_argument('--rules', nargs='+', choices=replay.RULES, default=['min_train'])
    parser.add_argument('--case-role', choices=['all', 'train', 'test'], default='test')
    parser.add_argument('--folds', nargs='+')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.replay_dir = args.replay_dir.resolve()
    args.links_dir = args.links_dir.resolve()
    for original_dir in (args.replay_dir, args.links_dir):
        if args.output_dir == original_dir or original_dir in args.output_dir.parents or args.output_dir in original_dir.parents:
            parser.error('Choose a separate diagnostic output directory')
    meta = replay.read_json(args.replay_dir/'run_metadata.json')
    if meta['status'] != 'complete' or not meta['baseline_reproduced']:
        raise ValueError('Completed, verified replay required')
    link_meta = replay.read_json(args.links_dir/'link_metadata.json')
    if link_meta['status'] != 'complete':
        raise ValueError('Completed association tables required')
    args.baseline = ROOT/'result/fsim_abc_tpe_seed123'
    args.budget = 1024
    args.fig_dir = args.output_dir/'unused_fig'
    tasks, _, identity, snapshot, manifest = replay.prepare(args)
    links = pd.read_csv(args.links_dir/'cohort_observation_links.csv', low_memory=False)
    identity.update(diagnostic_script_sha256=replay.sha(__file__),
                    replay_script_sha256=replay.sha(Path(replay.__file__)),
                    association_sha256=replay.sha(args.links_dir/'cohort_observation_links.csv'),
                    replay_identity_sha256=replay.sha(args.replay_dir/'run_identity.json'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output_dir/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        idpath = args.output_dir/'run_identity.json'
        if idpath.exists() and replay.read_json(idpath) != identity:
            raise ValueError('Diagnostic identity changed; use another --output-dir')
        replay.save_json(idpath, identity)
        status = dict(status='running', total=len(tasks), completed=0)
        replay.save_json(args.output_dir/'run_metadata.json', status)
        replay.initialize(str(snapshot), str(manifest))
        original = replay.CORE.FirstInfection.stage_primary_chain
        captured = []

        def wrapper(engine):
            captured.append(traced_call(engine, original))

        replay.CORE.FirstInfection.stage_primary_chain = wrapper
        tables, checks = [], []
        try:
            for index, task in enumerate(tasks, 1):
                captured.clear()
                source_dir = args.replay_dir/task['rule']/task['scheme']/task['fold']/task['case_role']/task['input_key']
                saved_hashes = replay.read_json(source_dir/'completed.json')['sha256']
                # Establish identity against the actual previous replay, including train cases.
                for name in ['effective_runtime_config.json', 'prediction.json'] + [n+'.csv' for n in replay.TABLE_NAMES]:
                    if replay.sha(source_dir/name) != saved_hashes[name]:
                        raise ValueError(f'Previous replay artifact changed: {source_dir/name}')
                if replay.read_json(source_dir/'effective_runtime_config.json') != task['config']:
                    raise ValueError('Replay config differs from saved B selection')
                prediction = replay.replay(task)
                check = {k: task[k] for k in KEYS}
                check['prediction_matches'] = not replay.check_expected(prediction, replay.read_json(source_dir/'prediction.json'))
                check['all_seven_tables_identical'] = all(
                    replay.sha(Path(task['output'])/(name+'.csv')) == saved_hashes[name+'.csv'] for name in replay.TABLE_NAMES)
                check['max_float_difference'] = compare_tables(source_dir, Path(task['output']))
                check['all_seven_tables_equivalent'] = True
                if not check['prediction_matches']:
                    raise ValueError(f'Diagnostic instrumentation changed results: {check}')
                selected = links
                for k in KEYS:
                    selected = selected[selected[k].astype(str).eq(str(task[k]))]
                trace = captured[0]
                if set(trace.Num) != set(selected.Num):
                    raise ValueError('Association cohort IDs differ from traced detached cohorts')
                merged = selected.merge(trace, on='Num', validate='one_to_one')
                if not merged.terminal_reason.eq('completed_primary_chain').equals(merged.completed_primary_chain.eq(True)):
                    raise ValueError('Association completion labels differ from trace')
                replay.save_frame(Path(task['output'])/'termination_trace.csv', merged)
                tables.append(merged); checks.append(check)
                status['completed'] = index
                replay.save_json(args.output_dir/'run_metadata.json', status)
                print(f'[{index}/{len(tasks)}] {task["scheme"]}/{task["fold"]}/{task["input_key"]}: {len(merged)} cohorts; reproduction passed', flush=True)
            combined = pd.concat(tables, ignore_index=True)
            replay.save_frame(args.output_dir/'cohort_termination_diagnostics.csv', combined)
            replay.save_frame(args.output_dir/'reproduction_checks.csv', pd.DataFrame(checks))
            grouping = KEYS+['observation_state', 'terminal_reason']
            counts = combined.groupby(grouping, dropna=False).size().rename('n_cohorts').reset_index()
            replay.save_frame(args.output_dir/'termination_summary.csv', counts)
            positive = combined[combined.observation_state.eq('positive')]
            replay.save_frame(args.output_dir/'positive_interval_termination.csv', positive.groupby(
                KEYS+['interval_id', 'start_exclusive', 'end_inclusive', 'interval_count_shared_not_additive', 'terminal_reason'],
                dropna=False).size().rename('n_cohorts').reset_index())
            status.update(status='complete', all_reproduction_checks_passed=True,
                          traced_cohorts=len(combined), trace_method='Python line events, original frozen code unchanged')
            replay.save_json(args.output_dir/'run_metadata.json', status)
            print(counts[counts.observation_state.eq('positive')].to_string(index=False))
            print(f'Results: {args.output_dir}')
        except Exception as exc:
            status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            replay.save_json(args.output_dir/'run_metadata.json', status)
            raise
        finally:
            replay.CORE.FirstInfection.stage_primary_chain = original


if __name__ == '__main__':
    main()
