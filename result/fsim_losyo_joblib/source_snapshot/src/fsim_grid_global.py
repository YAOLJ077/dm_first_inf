"""One finite joblib stream for unique Grid model evaluations, independent of folds."""
import time
from datetime import datetime, timezone

from joblib import Parallel, delayed, parallel_config
from tqdm.auto import tqdm


def unique_grid_tasks(jobs, candidates, evaluator):
    """Deduplicate only identical model inputs; never combine training objectives."""
    arms = {}
    for job in jobs:
        if job['arm'].endswith('grid'):
            arms.setdefault(job['arm'], set()).update(job['train'])
    seen = set()
    for arm, keys in arms.items():
        ids = [i for k, i in evaluator.primary.items() if k in keys]
        ids += [i for k, i in evaluator.alternate.items() if k in keys]
        for params in candidates[arm]:
            for cid in ids:
                key = evaluator.cache_key(cid, params)
                if key not in seen:
                    seen.add(key)
                    yield key, cid, params


def calculate_task(key, cid, params, worker):
    return key, cid, params, worker((cid, params))


def run_grid_models(s, jobs, candidates, evaluator, store, output, progress, core, worker):
    """Continuously refill the process pool; no per-fold candidate window or batch barrier."""
    started = time.perf_counter()
    tasks = list(unique_grid_tasks(jobs, candidates, evaluator))
    with store.lock:
        existing = {row[0] for row in store.db.execute('SELECT key FROM evaluations')}
    missing = [task for task in tasks if task[0] not in existing]
    metadata = dict(status='running', backend='joblib-loky', workers=s['workers'],
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    unique_grid_model_tasks=len(tasks), cached_at_start=len(tasks) - len(missing),
                    completed_new_model_tasks=0, model_seconds_sum=0.0,
                    note='One global task stream; no per-fold candidate limit. Fold selection follows cached model computation.')
    report = output / 'grid_execution.json'
    core.write_json(report, metadata)
    print(f'GRID global queue: {len(tasks)} unique model tasks; {len(missing)} uncached; '
          f'process limit={s["workers"]}', flush=True)
    pids = set()
    try:
        if missing:
            if s['workers'] == 1:
                core.worker_init(evaluator.cases, evaluator.base)
            # Parallel dispatches another item whenever a worker completes. The
            # pre_dispatch buffer bounds queued work, not the number of folds or trials.
            with parallel_config(backend='loky', inner_max_num_threads=1):
                with Parallel(n_jobs=s['workers'], return_as='generator_unordered', batch_size=1,
                              pre_dispatch=2 * s['workers'], initializer=core.worker_init,
                              initargs=(evaluator.cases, evaluator.base)) as parallel:
                    results = parallel(delayed(calculate_task)(key, cid, params, worker)
                                       for key, cid, params in missing)
                    with tqdm(total=len(missing), desc='GRID unique model computations', unit='model',
                              disable=not progress, dynamic_ncols=True) as bar:
                        for key, cid, params, payload in results:
                            if evaluator.stop.is_set():
                                raise RuntimeError('Run stopping')
                            record = payload['record']
                            if record['input_key'] != evaluator.cases[cid]['input_key'] or any(
                                    record[n] != params[n] for n in core.NAMES):
                                raise ValueError('Grid model identity mismatch')
                            if record['status'] not in ('predicted', 'no_prediction'):
                                raise RuntimeError(str(record))
                            store.save_evaluation(key, payload)
                            metadata['completed_new_model_tasks'] += 1
                            metadata['model_seconds_sum'] += payload['model_seconds']
                            pids.add(payload['worker_pid'])
                            bar.update(1)
                            if metadata['completed_new_model_tasks'] % 256 == 0:
                                metadata['elapsed_seconds'] = time.perf_counter() - started
                                core.write_json(report, metadata)
                                if not progress:
                                    print(f'GRID models: {metadata["completed_new_model_tasks"]}/{len(missing)}', flush=True)
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        evaluator.stop.set()
        raise
    finally:
        metadata['elapsed_seconds'] = time.perf_counter() - started
        metadata['worker_pids_observed'] = sorted(pids)
        core.write_json(report, metadata)
    print('GRID model cache complete; assembling training-only fold selections and held-out reports.', flush=True)
    return metadata
