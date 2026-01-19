# -*- coding: utf-8 -*-
"""
Baseline calibration (Bayes + TPE + LOO) and spore-process continuation
starting from the baseline global-best parameters.

Part A: Baseline LOO calibration
  - Per fold: optimize train_mae; select best by val_mae.
  - Output per-fold best params and metrics (mae/std/rmse).
  - Summary across folds.
  - Global-best: evaluate each fold-best params on all samples.

Part B: Two-stage spore-process continuation (Bayes + TPE + LOO)
  - Start from baseline global-best params.
  - Stage 1: optimize spore-consistency loss within a local absolute range.
  - Stage 2: optimize train_mae within local absolute ranges; spore params
    can vary only in an absolute elastic range around stage1 best.
  - Output per-fold best params and metrics (same format as baseline).
"""

import os
import sys
import heapq
import json
import copy
import argparse
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import numpy as np
import pandas as pd
import ray
import optuna
from optuna.samplers import TPESampler
import time

# ======================================================
# PATHS
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_DATA_PATH = os.path.join(DATA_DIR, "Spore_overall.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season import CropSeasonFirst
from model.config import config

# ======================================================
# SETTINGS
# ======================================================
PENALTY_DAYS = 120.0
RANDOM_SEED = 42
FOLD_SEED = 42
N_JOBS = 128
RB_CACHE_MAX = 128
SLOW_TRIALS_KEEP = 50

SPORE_CORE_DAYS = 14

# Base ranges (global)
BASE_RANGES = {
    "mp_tmp__rh_wet_night": (85, 95),
    "sporulation__MP_min": (3, 6),
    "sporulation__Tmin": (8, 15),
    "sporulation__Topt": (18, 24),
    "sporulation__Tmax": (25, 35),
    "dispersal__rain_min": (6.0, 12.0),
    "dispersal__wind_min": (4.0, 12.0),
    "deposition__rain_min": (4.0, 12.0),
    "zoospore_alive__rain": (3.0, 6.0),
    "zoospore_alive__vpd": (1.5, 3.5),
    "infection__infection_condition": (50, 200),
    "incubation__spl_end": (1.0, 1.5),
    "thresholds__min_spores": (0.02, 0.06)
}


# Local absolute ranges for baseline local refinement (absolute deltas).
LOCAL_ABS_BASELINE = {
    "mp_tmp__rh_wet_night": 2.0,
    "sporulation__MP_min": 2.0,
    "sporulation__Tmin": 0.5,
    "sporulation__Topt": 0.5,
    "sporulation__Tmax": 1.0,
    "dispersal__rain_min": 2.0,
    "dispersal__wind_min": 2.0,
    "deposition__rain_min": 2.0,
    "zoospore_alive__rain": 0.5,
    "zoospore_alive__vpd": 1.0,
    "infection__infection_condition": 30.0,
    "incubation__spl_end": 0.1,
    "thresholds__min_spores": 0.02,
}

# Local absolute ranges for spore two-stage continuation (absolute deltas).
LOCAL_ABS_SPORE = {
    "mp_tmp__rh_wet_night": 5.0,
    "sporulation__MP_min": 3.0,
    "sporulation__Tmin": 1.0,
    "sporulation__Topt": 1.0,
    "sporulation__Tmax": 2.0,
    "dispersal__rain_min": 3.0,
    "dispersal__wind_min": 3.0,
    "deposition__rain_min": 3.0,
    "zoospore_alive__rain": 1.0,
    "zoospore_alive__vpd": 1.5,
    "infection__infection_condition": 60.0,
    "incubation__spl_end": 0.2,
    "thresholds__min_spores": 0.03,
}

# Stage 1 params (spore-process)
STAGE1_PARAMS = {
    "mp_tmp__rh_wet_night",
    "sporulation__MP_min",
    "sporulation__Tmin",
    "sporulation__Topt",
    "sporulation__Tmax",
    "thresholds__min_spores",
    "dispersal__wind_min",
    "deposition__rain_min",
}

# Stage 2 params (non-spore process)
STAGE2_PARAMS = {
    "zoospore_alive__rain",
    "zoospore_alive__vpd",
    "infection__infection_condition",
    "incubation__spl_end",
}

# Elastic range for stage1 params in stage2 (absolute deltas).
STAGE1_RELAX_ABS = {
    "mp_tmp__rh_wet_night": 3.0,
    "sporulation__MP_min": 2.0,
    "sporulation__Tmin": 1.0,
    "sporulation__Topt": 1.0,
    "sporulation__Tmax": 1.0,
    "thresholds__min_spores": 0.02,
    "dispersal__wind_min": 2.0,
    "deposition__rain_min": 2.0,
}


def expand_range_1p5(lo: float, hi: float) -> Tuple[float, float]:
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    new_half = 1.5 * half
    return center - new_half, center + new_half


def build_param_ranges() -> Dict[str, Tuple[float, float]]:
    return {k: expand_range_1p5(v[0], v[1]) for k, v in BASE_RANGES.items()}


def load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)


def load_spores() -> pd.DataFrame:
    df = pd.read_excel(SPORE_DATA_PATH)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df["spore"] = pd.to_numeric(df["spore"], errors="coerce").fillna(0)
    df["present"] = df["spore"] > 0
    return df


def build_spore_windows(df: pd.DataFrame) -> Dict[Tuple[str, int], List[Tuple[object, object, bool]]]:
    windows = {}
    for (plot, year), g in df.groupby(["plot", "year"]):
        g = g.sort_values("date")
        prev_date = None
        entries = []
        for _, row in g.iterrows():
            cur_date = row["date"].date()
            start = prev_date or cur_date
            end = cur_date
            entries.append((start, end, bool(row["present"])))
            prev_date = cur_date
        windows[(str(plot), int(year))] = entries
    return windows


def apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in values.items():
        group, sub = k.split("__")
        cfg[group][sub] = float(v)
    return cfg


_RB_CACHE: Dict[str, dict] = {}


def load_rb_cached(path: str) -> dict:
    if path not in _RB_CACHE:
        with open(path, "r") as f:
            _RB_CACHE[path] = json.load(f)
    return _RB_CACHE[path]


def _load_rb_cached(path: str, cache: Dict[str, dict]) -> dict:
    if path not in cache:
        with open(path, "r") as f:
            cache[path] = json.load(f)
    return cache[path]


def _calc_error(actual_date, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return np.nan
    first = event_df.iloc[0]
    doy_actual = pd.to_datetime(actual_date).dayofyear
    doy_start = pd.to_datetime(first["oil_spot_start"]).dayofyear
    doy_end = pd.to_datetime(first["oil_spot_end"]).dayofyear
    if doy_end < doy_actual:
        return float(doy_end - doy_actual)
    if doy_start > doy_actual:
        return float(doy_start - doy_actual)
    return 0.0


def compute_error_stats(cfg: dict, rows: List[pd.Series]) -> Tuple[float, float, float]:
    errors = []
    for row in rows:
        key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue
        try:
            rb = load_rb_cached(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            err = _calc_error(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            err = np.nan
        if np.isfinite(err):
            errors.append(float(err))
        else:
            errors.append(float(PENALTY_DAYS))
    if not errors:
        return PENALTY_DAYS, 0.0, PENALTY_DAYS
    abs_err = np.abs(np.array(errors, dtype=float))
    mae = float(abs_err.mean())
    std = float(abs_err.std(ddof=0))
    rmse = float(np.sqrt(np.mean(np.square(abs_err))))
    return mae, std, rmse


def build_folds(n: int, k: int, seed: int) -> List[List[int]]:
    if k <= 1 or k > n:
        k = n
    folds = [[] for _ in range(k)]
    rng = np.random.RandomState(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    for i, idx in enumerate(idxs):
        folds[i % k].append(idx)
    return folds


def _progress_print(
    done: int,
    total: int,
    running: int,
    running_max: int,
    prefix: str,
    last_ts: float,
    interval_s: float,
    durations: Optional[List[float]] = None,
    io_durations: Optional[List[float]] = None,
    sim_durations: Optional[List[float]] = None
) -> float:
    now = time.time()
    if now - last_ts >= interval_s:
        msg = f"Progress: {done}/{total} completed | running: {running}/{running_max}"
        if durations:
            window = durations[-200:]
            p50, p90, p99 = np.percentile(window, [50, 90, 99])
            msg += f" | t50={p50:.2f}s t90={p90:.2f}s t99={p99:.2f}s"
        if io_durations:
            window = io_durations[-200:]
            io50, io90 = np.percentile(window, [50, 90])
            msg += f" | io50={io50:.2f}s io90={io90:.2f}s"
        if sim_durations:
            window = sim_durations[-200:]
            sim50, sim90 = np.percentile(window, [50, 90])
            msg += f" | sim50={sim50:.2f}s sim90={sim90:.2f}s"
        if prefix:
            msg = f"{prefix} {msg}"
        print(msg, flush=True)
        return now
    return last_ts


def _stage_start(prefix: str, total: int, running: int, running_max: int) -> None:
    msg = f"{prefix} Start | total trials: {total} | running: {running}/{running_max}"
    print(msg, flush=True)


def _record_slow_trial(heap: List[Tuple[float, Dict[str, float]]], entry: Dict[str, float], max_keep: int) -> None:
    dur = float(entry.get("duration_total", 0.0))
    if len(heap) < max_keep:
        heapq.heappush(heap, (dur, entry))
        return
    if heap and dur > heap[0][0]:
        heapq.heapreplace(heap, (dur, entry))


def define_search_space(trial: optuna.Trial, ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    params = {}
    for param_name, (lo, hi) in ranges.items():
        params[param_name] = trial.suggest_float(param_name, lo, hi)
    return params


def build_local_ranges_from_center(
    center: Dict[str, float],
    global_ranges: Dict[str, Tuple[float, float]],
    abs_ranges: Dict[str, float]
) -> Dict[str, Tuple[float, float]]:
    local = {}
    for key, (g_lo, g_hi) in global_ranges.items():
        if key not in center:
            local[key] = (g_lo, g_hi)
            continue
        delta = abs_ranges.get(key, 0.0)
        if delta <= 0:
            local[key] = (g_lo, g_hi)
            continue
        lo = max(g_lo, center[key] - delta)
        hi = min(g_hi, center[key] + delta)
        if lo >= hi:
            lo, hi = g_lo, g_hi
        local[key] = (lo, hi)
    return local


@ray.remote(num_cpus=1)
class ModelWorker:
    def __init__(self, rows_ref, spore_ref):
        if isinstance(rows_ref, ray.ObjectRef):
            self.rows = ray.get(rows_ref)
        else:
            self.rows = rows_ref
        if isinstance(spore_ref, ray.ObjectRef):
            self.spore_windows = ray.get(spore_ref)
        else:
            self.spore_windows = spore_ref
        self.rb_cache: "OrderedDict[str, dict]" = OrderedDict()
        self.rb_cache_max = RB_CACHE_MAX

    def _load_rb_cached_lru(self, path: str) -> dict:
        if path in self.rb_cache:
            self.rb_cache.move_to_end(path)
            return self.rb_cache[path]
        with open(path, "r") as f:
            data = json.load(f)
        self.rb_cache[path] = data
        self.rb_cache.move_to_end(path)
        if self.rb_cache_max > 0 and len(self.rb_cache) > self.rb_cache_max:
            self.rb_cache.popitem(last=False)
        return data

    def _compute_error_stats_idx(self, cfg: dict, idxs: List[int]) -> Tuple[float, float, float, float, float]:
        errors = []
        io_s = 0.0
        sim_s = 0.0
        for i in idxs:
            row = self.rows[i]
            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
            if not os.path.exists(rb_path):
                continue
            try:
                t0 = time.time()
                rb = self._load_rb_cached_lru(rb_path)
                io_s += time.time() - t0
                t1 = time.time()
                cs = CropSeasonFirst(csrb=rb, config=cfg)
                cs.simulate()
                sim_s += time.time() - t1
                err = _calc_error(row.first_infection_date, cs.primary_chain_unique)
            except Exception:
                err = np.nan
            if np.isfinite(err):
                errors.append(float(err))
            else:
                errors.append(float(PENALTY_DAYS))
        if not errors:
            return PENALTY_DAYS, 0.0, PENALTY_DAYS
        abs_err = np.abs(np.array(errors, dtype=float))
        mae = float(abs_err.mean())
        std = float(abs_err.std(ddof=0))
        rmse = float(np.sqrt(np.mean(np.square(abs_err))))
        return mae, std, rmse, io_s, sim_s

    def eval_mae(self, params: Dict[str, float], train_idx: List[int], val_idx: List[int]) -> Tuple[float, float, Dict[str, float]]:
        t_all = time.time()
        cfg = apply_params(config, params)
        train_mae, _, _, train_io, train_sim = self._compute_error_stats_idx(cfg, train_idx)
        val_mae, _, _, val_io, val_sim = self._compute_error_stats_idx(cfg, val_idx)
        timing = {
            "total": time.time() - t_all,
            "io": train_io + val_io,
            "sim": train_sim + val_sim,
        }
        return train_mae, val_mae, timing

    def _spore_loss_idx(self, cfg: dict, idxs: List[int], core_days: int) -> Tuple[float, float, float]:
        losses = []
        io_s = 0.0
        sim_s = 0.0
        for i in idxs:
            row = self.rows[i]
            key = (str(row.plot), int(row.first_infection_date.year))
            obs = self.spore_windows.get(key, [])
            if not obs:
                losses.append(1.0)
                continue
            rb_key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{rb_key}_grape_rb.json")
            if not os.path.exists(rb_path):
                losses.append(1.0)
                continue
            try:
                t0 = time.time()
                rb = self._load_rb_cached_lru(rb_path)
                io_s += time.time() - t0
                t1 = time.time()
                cs = CropSeasonFirst(csrb=rb, config=cfg)
                cs.simulate()
                sim_s += time.time() - t1
                if cs.primary_chain_unique is None or cs.primary_chain_unique.empty:
                    losses.append(1.0)
                    continue
                oil_start_dt = pd.to_datetime(cs.primary_chain_unique.iloc[0]["oil_spot_start"]).normalize()
            except Exception:
                losses.append(1.0)
                continue

            win_start = (oil_start_dt - pd.Timedelta(days=core_days)).date()
            win_end = oil_start_dt.date()
            total = 0
            hit = 0
            for s, e, present in obs:
                if e < win_start or s > win_end:
                    continue
                total += 1
                if present:
                    hit += 1
            if total == 0:
                losses.append(1.0)
            else:
                consistency = hit / total
                losses.append(1.0 - consistency)
        loss = float(np.mean(losses)) if losses else 1.0
        return loss, io_s, sim_s

    def eval_spore(self, params: Dict[str, float], train_idx: List[int], val_idx: List[int], core_days: int) -> Tuple[float, float, Dict[str, float]]:
        t_all = time.time()
        cfg = apply_params(config, params)
        train_loss, train_io, train_sim = self._spore_loss_idx(cfg, train_idx, core_days)
        val_loss, val_io, val_sim = self._spore_loss_idx(cfg, val_idx, core_days)
        timing = {
            "total": time.time() - t_all,
            "io": train_io + val_io,
            "sim": train_sim + val_sim,
        }
        return train_loss, val_loss, timing


def _make_workers(n_jobs: int, rows_ref, spore_ref) -> List[ray.actor.ActorHandle]:
    return [ModelWorker.remote(rows_ref, spore_ref) for _ in range(n_jobs)]


def _replace_worker(workers: List[ray.actor.ActorHandle], old_worker, rows_ref, spore_ref):
    try:
        idx = workers.index(old_worker)
    except ValueError:
        idx = None
    ray.kill(old_worker, no_restart=True)
    new_worker = ModelWorker.remote(rows_ref, spore_ref)
    if idx is not None:
        workers[idx] = new_worker
    else:
        workers.append(new_worker)
    return new_worker


def run_baseline_loo(
    rows: List[pd.Series],
    k: int,
    global_ranges: Dict[str, Tuple[float, float]],
    workers: List[ray.actor.ActorHandle],
    rows_ref,
    spore_ref,
    n_jobs: int,
    restart_every: int,
    n_trials_stage1: int,
    n_trials_stage2: int,
    n_startup_stage1: int,
    n_startup_stage2: int,
    progress_interval: float,
    seed: int
) -> Tuple[List[Dict[str, float]], pd.DataFrame, Dict[str, float]]:
    folds = build_folds(len(rows), k, FOLD_SEED)
    fold_best_rows = []
    fold_params = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = [i for i in range(len(rows)) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        # Stage 1 (global)
        sampler1 = TPESampler(seed=seed + fold_id, multivariate=True, n_startup_trials=n_startup_stage1)
        study1 = optuna.create_study(sampler=sampler1, direction="minimize")
        trial_rows = []
        durations = []
        io_durations = []
        sim_durations = []
        slow_trials = []
        io_durations = []
        sim_durations = []
        slow_trials = []
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        _stage_start(f"[Baseline][Fold {fold_id}/{k}][Stage1]", n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(
            0,
            n_trials_stage1,
            0,
            max_conc,
            f"[Baseline][Fold {fold_id}/{k}][Stage1]",
            last_ts,
            progress_interval,
            durations,
            io_durations,
            sim_durations,
        )

        while completed < n_trials_stage1:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage1:
                trial = study1.ask()
                params = define_search_space(trial, global_ranges)
                worker = available.pop(0)
                fut = worker.eval_mae.remote(params, train_idx, val_idx)
                pending[fut] = (trial, params, worker, time.time())
            if not pending:
                time.sleep(0.01)
                continue
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params, worker, start_ts = pending.pop(fut)
                train_mae, val_mae, timing = ray.get(fut)
                durations.append(float(timing["total"]))
                io_durations.append(float(timing["io"]))
                sim_durations.append(float(timing["sim"]))
                study1.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    **params
                })
                _record_slow_trial(
                    slow_trials,
                    {
                        "duration_total": float(timing["total"]),
                        "duration_io": float(timing["io"]),
                        "duration_sim": float(timing["sim"]),
                        "train_mae": float(train_mae),
                        "val_mae": float(val_mae),
                        **params,
                    },
                    SLOW_TRIALS_KEEP,
                )
                completed += 1
                if next_restart is not None and completed >= next_restart:
                    worker = _replace_worker(workers, worker, rows_ref, spore_ref)
                    next_restart += restart_every
                available.append(worker)
            last_ts = _progress_print(
                completed,
                n_trials_stage1,
                len(pending),
                max_conc,
                f"[Baseline][Fold {fold_id}/{k}][Stage1]",
                last_ts,
                progress_interval,
                durations,
                io_durations,
                sim_durations,
            )

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage1_out = os.path.join(RESULT_DIR, f"baseline_loo_fold_{fold_id}_stage1_trials.csv")
        df_trials.to_csv(stage1_out, index=False)
        if slow_trials:
            slow_sorted = [row for _, row in sorted(slow_trials, key=lambda x: x[0], reverse=True)]
            slow_out = os.path.join(RESULT_DIR, f"baseline_loo_fold_{fold_id}_stage1_slow_trials.csv")
            pd.DataFrame(slow_sorted).to_csv(slow_out, index=False)

        best_row = df_trials.sort_values("train_mae").iloc[0]
        best_params_stage1 = {
            param_name: best_row[param_name]
            for param_name in best_row.index
            if param_name not in ("train_mae", "val_mae")
        }
        local_ranges = build_local_ranges_from_center(best_params_stage1, global_ranges, LOCAL_ABS_BASELINE)

        # Stage 2 (local)
        sampler2 = TPESampler(seed=seed + fold_id + 1000, multivariate=True, n_startup_trials=n_startup_stage2)
        study2 = optuna.create_study(sampler=sampler2, direction="minimize")
        trial_rows = []
        durations = []
        io_durations = []
        sim_durations = []
        slow_trials = []
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        _stage_start(f"[Baseline][Fold {fold_id}/{k}][Stage2]", n_trials_stage2, 0, max_conc)
        last_ts = _progress_print(
            0,
            n_trials_stage2,
            0,
            max_conc,
            f"[Baseline][Fold {fold_id}/{k}][Stage2]",
            last_ts,
            progress_interval,
            durations,
            io_durations,
            sim_durations,
        )
        while completed < n_trials_stage2:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage2:
                trial = study2.ask()
                params = define_search_space(trial, local_ranges)
                worker = available.pop(0)
                fut = worker.eval_mae.remote(params, train_idx, val_idx)
                pending[fut] = (trial, params, worker, time.time())
            if not pending:
                time.sleep(0.01)
                continue
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params, worker, start_ts = pending.pop(fut)
                train_mae, val_mae, timing = ray.get(fut)
                durations.append(float(timing["total"]))
                io_durations.append(float(timing["io"]))
                sim_durations.append(float(timing["sim"]))
                study2.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    **params
                })
                _record_slow_trial(
                    slow_trials,
                    {
                        "duration_total": float(timing["total"]),
                        "duration_io": float(timing["io"]),
                        "duration_sim": float(timing["sim"]),
                        "train_mae": float(train_mae),
                        "val_mae": float(val_mae),
                        **params,
                    },
                    SLOW_TRIALS_KEEP,
                )
                completed += 1
                if next_restart is not None and completed >= next_restart:
                    worker = _replace_worker(workers, worker, rows_ref, spore_ref)
                    next_restart += restart_every
                available.append(worker)
            last_ts = _progress_print(
                completed,
                n_trials_stage2,
                len(pending),
                max_conc,
                f"[Baseline][Fold {fold_id}/{k}][Stage2]",
                last_ts,
                progress_interval,
                durations,
                io_durations,
                sim_durations,
            )

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage2_out = os.path.join(RESULT_DIR, f"baseline_loo_fold_{fold_id}_trials.csv")
        df_trials.to_csv(stage2_out, index=False)
        if slow_trials:
            slow_sorted = [row for _, row in sorted(slow_trials, key=lambda x: x[0], reverse=True)]
            slow_out = os.path.join(RESULT_DIR, f"baseline_loo_fold_{fold_id}_stage2_slow_trials.csv")
            pd.DataFrame(slow_sorted).to_csv(slow_out, index=False)

        best_train_row = df_trials.sort_values("train_mae").iloc[0]
        best_params = {
            param_name: best_train_row[param_name]
            for param_name in best_train_row.index
            if param_name not in ("train_mae", "val_mae")
        }
        fold_params.append(best_params)

        cfg = apply_params(config, best_params)
        train_mae, train_std, train_rmse = compute_error_stats(cfg, train_rows)
        val_mae, val_std, val_rmse = compute_error_stats(cfg, val_rows)

        fold_best_rows.append({
            "fold": fold_id,
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "train_std": float(train_std),
            "val_std": float(val_std),
            "train_var": float(train_std ** 2),
            "val_var": float(val_std ** 2),
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
        })

        best_out = os.path.join(RESULT_DIR, f"baseline_loo_best_params_fold{fold_id}.json")
        with open(best_out, "w") as f:
            json.dump({
                "fold": fold_id,
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "train_std": float(train_std),
                "val_std": float(val_std),
                "train_var": float(train_std ** 2),
                "val_var": float(val_std ** 2),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "params": best_params,
            }, f, indent=2)

    summary = pd.DataFrame(fold_best_rows)
    summary_out = os.path.join(RESULT_DIR, "baseline_loo_summary.csv")
    summary.to_csv(summary_out, index=False)
    summary_stats = {
        "train_mae_mean": float(summary["train_mae"].mean()),
        "train_mae_std": float(summary["train_mae"].std(ddof=0)),
        "train_mae_var": float(summary["train_mae"].var(ddof=0)),
        "val_mae_mean": float(summary["val_mae"].mean()),
        "val_mae_std": float(summary["val_mae"].std(ddof=0)),
        "val_mae_var": float(summary["val_mae"].var(ddof=0)),
        "train_rmse": float(np.sqrt(np.mean(np.square(summary["train_mae"])))),
        "val_rmse": float(np.sqrt(np.mean(np.square(summary["val_mae"])))),
        "n_folds": int(len(summary)),
    }
    summary_json = os.path.join(RESULT_DIR, "baseline_loo_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Global-best selection
    global_rows = []
    for i, params in enumerate(fold_params, start=1):
        cfg = apply_params(config, params)
        global_mae, global_std, global_rmse = compute_error_stats(cfg, rows)
        global_rows.append({
            "fold": i,
            "global_mae": float(global_mae),
            "global_std": float(global_std),
            "global_rmse": float(global_rmse),
        })
    df_global = pd.DataFrame(global_rows)
    best_idx = int(df_global["global_mae"].idxmin())
    best_fold = int(df_global.loc[best_idx, "fold"])
    best_params = fold_params[best_idx]
    best_row = summary[summary["fold"] == best_fold].iloc[0]
    global_out = os.path.join(RESULT_DIR, "baseline_loo_best_global.json")
    with open(global_out, "w") as f:
        json.dump({
            "best_from_fold": best_fold,
            "train_mae": float(best_row["train_mae"]),
            "val_mae": float(best_row["val_mae"]),
            "train_std": float(best_row["train_std"]),
            "val_std": float(best_row["val_std"]),
            "train_var": float(best_row["train_var"]),
            "val_var": float(best_row["val_var"]),
            "global_mae": float(df_global.loc[best_idx, "global_mae"]),
            "global_std": float(df_global.loc[best_idx, "global_std"]),
            "global_var": float(df_global.loc[best_idx, "global_std"] ** 2),
            "global_rmse": float(df_global.loc[best_idx, "global_rmse"]),
            "params": best_params,
        }, f, indent=2)

    return fold_params, summary, {
        "best_params": best_params,
        "best_fold": best_fold,
        "global_mae": float(df_global.loc[best_idx, "global_mae"]),
        "global_std": float(df_global.loc[best_idx, "global_std"]),
    }


def run_spore_two_stage_loo(
    rows: List[pd.Series],
    k: int,
    global_ranges: Dict[str, Tuple[float, float]],
    global_best: Dict[str, float],
    workers: List[ray.actor.ActorHandle],
    rows_ref,
    spore_ref,
    n_jobs: int,
    restart_every: int,
    n_trials_stage1: int,
    n_trials_stage2: int,
    n_startup_stage1: int,
    n_startup_stage2: int,
    progress_interval: float,
    seed: int
) -> None:
    folds = build_folds(len(rows), k, FOLD_SEED)
    stage2_best_rows = []
    stage2_params = []

    stage1_local = build_local_ranges_from_center(global_best, global_ranges, LOCAL_ABS_SPORE)

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = [i for i in range(len(rows)) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        # Stage 1: spore consistency
        stage1_ranges = {k: v for k, v in stage1_local.items() if k in STAGE1_PARAMS}
        fixed_base = {k: global_best[k] for k in global_best if k not in STAGE1_PARAMS}

        sampler1 = TPESampler(seed=seed + fold_id + 2000, multivariate=True, n_startup_trials=n_startup_stage1)
        study1 = optuna.create_study(sampler=sampler1, direction="minimize")
        trial_rows = []
        durations = []
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        _stage_start(f"[Spore][Fold {fold_id}/{k}][Stage1]", n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(
            0,
            n_trials_stage1,
            0,
            max_conc,
            f"[Spore][Fold {fold_id}/{k}][Stage1]",
            last_ts,
            progress_interval,
            durations,
            io_durations,
            sim_durations,
        )

        while completed < n_trials_stage1:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage1:
                trial = study1.ask()
                params = define_search_space(trial, stage1_ranges)
                params = {**fixed_base, **params}
                worker = available.pop(0)
                fut = worker.eval_spore.remote(params, train_idx, val_idx, SPORE_CORE_DAYS)
                pending[fut] = (trial, params, worker, time.time())
            if not pending:
                time.sleep(0.01)
                continue
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params, worker, start_ts = pending.pop(fut)
                train_loss, val_loss, timing = ray.get(fut)
                durations.append(float(timing["total"]))
                io_durations.append(float(timing["io"]))
                sim_durations.append(float(timing["sim"]))
                study1.tell(trial, train_loss)
                trial_rows.append({
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    **params
                })
                _record_slow_trial(
                    slow_trials,
                    {
                        "duration_total": float(timing["total"]),
                        "duration_io": float(timing["io"]),
                        "duration_sim": float(timing["sim"]),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        **params,
                    },
                    SLOW_TRIALS_KEEP,
                )
                completed += 1
                if next_restart is not None and completed >= next_restart:
                    worker = _replace_worker(workers, worker, rows_ref, spore_ref)
                    next_restart += restart_every
                available.append(worker)
            last_ts = _progress_print(
                completed,
                n_trials_stage1,
                len(pending),
                max_conc,
                f"[Spore][Fold {fold_id}/{k}][Stage1]",
                last_ts,
                progress_interval,
                durations,
                io_durations,
                sim_durations,
            )

        df_stage1 = pd.DataFrame(trial_rows).sort_values("train_loss")
        stage1_out = os.path.join(RESULT_DIR, f"spore_loo_stage1_fold{fold_id}.csv")
        df_stage1.to_csv(stage1_out, index=False)
        if slow_trials:
            slow_sorted = [row for _, row in sorted(slow_trials, key=lambda x: x[0], reverse=True)]
            slow_out = os.path.join(RESULT_DIR, f"spore_loo_stage1_fold{fold_id}_slow_trials.csv")
            pd.DataFrame(slow_sorted).to_csv(slow_out, index=False)
        best1 = df_stage1.sort_values("train_loss").iloc[0]
        stage1_best = {
            param_name: best1[param_name]
            for param_name in best1.index
            if param_name not in ("train_loss", "val_loss")
        }
        stage1_best_out = os.path.join(RESULT_DIR, f"spore_loo_stage1_best_fold{fold_id}.json")
        with open(stage1_best_out, "w") as f:
            json.dump({
                "fold": fold_id,
                "train_loss": float(best1["train_loss"]),
                "val_loss": float(best1["val_loss"]),
                "params": stage1_best,
            }, f, indent=2)

        # Stage 2: MAE calibration
        stage2_ranges = {k: v for k, v in stage1_local.items() if k in STAGE2_PARAMS}
        for param_name, delta in STAGE1_RELAX_ABS.items():
            if param_name not in stage1_best:
                continue
            g_lo, g_hi = global_ranges[param_name]
            lo = max(g_lo, stage1_best[param_name] - delta)
            hi = min(g_hi, stage1_best[param_name] + delta)
            stage2_ranges[param_name] = (lo, hi)

        fixed_stage2 = {k: stage1_best[k] for k in stage1_best if k not in STAGE2_PARAMS}

        sampler2 = TPESampler(seed=seed + fold_id + 3000, multivariate=True, n_startup_trials=n_startup_stage2)
        study2 = optuna.create_study(sampler=sampler2, direction="minimize")
        trial_rows = []
        durations = []
        io_durations = []
        sim_durations = []
        slow_trials = []
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        _stage_start(f"[Spore][Fold {fold_id}/{k}][Stage2]", n_trials_stage2, 0, max_conc)
        last_ts = _progress_print(
            0,
            n_trials_stage2,
            0,
            max_conc,
            f"[Spore][Fold {fold_id}/{k}][Stage2]",
            last_ts,
            progress_interval,
            durations,
            io_durations,
            sim_durations,
        )
        while completed < n_trials_stage2:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage2:
                trial = study2.ask()
                params = define_search_space(trial, stage2_ranges)
                params = {**fixed_stage2, **params}
                worker = available.pop(0)
                fut = worker.eval_mae.remote(params, train_idx, val_idx)
                pending[fut] = (trial, params, worker, time.time())
            if not pending:
                time.sleep(0.01)
                continue
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params, worker, start_ts = pending.pop(fut)
                train_mae, val_mae, timing = ray.get(fut)
                durations.append(float(timing["total"]))
                io_durations.append(float(timing["io"]))
                sim_durations.append(float(timing["sim"]))
                study2.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    **params
                })
                _record_slow_trial(
                    slow_trials,
                    {
                        "duration_total": float(timing["total"]),
                        "duration_io": float(timing["io"]),
                        "duration_sim": float(timing["sim"]),
                        "train_mae": float(train_mae),
                        "val_mae": float(val_mae),
                        **params,
                    },
                    SLOW_TRIALS_KEEP,
                )
                completed += 1
                if next_restart is not None and completed >= next_restart:
                    worker = _replace_worker(workers, worker, rows_ref, spore_ref)
                    next_restart += restart_every
                available.append(worker)
            last_ts = _progress_print(
                completed,
                n_trials_stage2,
                len(pending),
                max_conc,
                f"[Spore][Fold {fold_id}/{k}][Stage2]",
                last_ts,
                progress_interval,
                durations,
                io_durations,
                sim_durations,
            )

        df_stage2 = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage2_out = os.path.join(RESULT_DIR, f"spore_loo_stage2_fold{fold_id}.csv")
        df_stage2.to_csv(stage2_out, index=False)
        if slow_trials:
            slow_sorted = [row for _, row in sorted(slow_trials, key=lambda x: x[0], reverse=True)]
            slow_out = os.path.join(RESULT_DIR, f"spore_loo_stage2_fold{fold_id}_slow_trials.csv")
            pd.DataFrame(slow_sorted).to_csv(slow_out, index=False)
        best2 = df_stage2.sort_values("train_mae").iloc[0]
        best_params = {
            param_name: best2[param_name]
            for param_name in best2.index
            if param_name not in ("train_mae", "val_mae")
        }
        stage2_params.append(best_params)

        cfg = apply_params(config, best_params)
        train_mae, train_std, train_rmse = compute_error_stats(cfg, train_rows)
        val_mae, val_std, val_rmse = compute_error_stats(cfg, val_rows)

        stage2_best_rows.append({
            "fold": fold_id,
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "train_std": float(train_std),
            "val_std": float(val_std),
            "train_var": float(train_std ** 2),
            "val_var": float(val_std ** 2),
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
        })

        best_out = os.path.join(RESULT_DIR, f"spore_loo_best_params_fold{fold_id}.json")
        with open(best_out, "w") as f:
            json.dump({
                "fold": fold_id,
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "train_std": float(train_std),
                "val_std": float(val_std),
                "train_var": float(train_std ** 2),
                "val_var": float(val_std ** 2),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "params": best_params,
            }, f, indent=2)

    summary = pd.DataFrame(stage2_best_rows)
    summary_out = os.path.join(RESULT_DIR, "spore_loo_summary.csv")
    summary.to_csv(summary_out, index=False)
    summary_stats = {
        "train_mae_mean": float(summary["train_mae"].mean()),
        "train_mae_std": float(summary["train_mae"].std(ddof=0)),
        "train_mae_var": float(summary["train_mae"].var(ddof=0)),
        "val_mae_mean": float(summary["val_mae"].mean()),
        "val_mae_std": float(summary["val_mae"].std(ddof=0)),
        "val_mae_var": float(summary["val_mae"].var(ddof=0)),
        "train_rmse": float(np.sqrt(np.mean(np.square(summary["train_mae"])))),
        "val_rmse": float(np.sqrt(np.mean(np.square(summary["val_mae"])))),
        "n_folds": int(len(summary)),
        "spore_core_days": int(SPORE_CORE_DAYS),
    }
    summary_json = os.path.join(RESULT_DIR, "spore_loo_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Global-best selection
    global_rows = []
    for i, params in enumerate(stage2_params, start=1):
        cfg = apply_params(config, params)
        global_mae, global_std, global_rmse = compute_error_stats(cfg, rows)
        global_rows.append({
            "fold": i,
            "global_mae": float(global_mae),
            "global_std": float(global_std),
            "global_rmse": float(global_rmse),
        })
    df_global = pd.DataFrame(global_rows)
    best_idx = int(df_global["global_mae"].idxmin())
    best_fold = int(df_global.loc[best_idx, "fold"])
    best_params = stage2_params[best_idx]
    best_row = summary[summary["fold"] == best_fold].iloc[0]
    global_out = os.path.join(RESULT_DIR, "spore_loo_best_global.json")
    with open(global_out, "w") as f:
        json.dump({
            "best_from_fold": best_fold,
            "train_mae": float(best_row["train_mae"]),
            "val_mae": float(best_row["val_mae"]),
            "train_std": float(best_row["train_std"]),
            "val_std": float(best_row["val_std"]),
            "train_var": float(best_row["train_var"]),
            "val_var": float(best_row["val_var"]),
            "global_mae": float(df_global.loc[best_idx, "global_mae"]),
            "global_std": float(df_global.loc[best_idx, "global_std"]),
            "global_var": float(df_global.loc[best_idx, "global_std"] ** 2),
            "global_rmse": float(df_global.loc[best_idx, "global_rmse"]),
            "params": best_params,
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-trials-base-stage1", type=int, default=10000)
    parser.add_argument("--n-trials-base-stage2", type=int, default=4000)
    parser.add_argument("--n-startup-base-stage1", type=int, default=1000)
    parser.add_argument("--n-startup-base-stage2", type=int, default=400)
    parser.add_argument("--n-trials-spore-stage1", type=int, default=10000)
    parser.add_argument("--n-trials-spore-stage2", type=int, default=4000)
    parser.add_argument("--n-startup-spore-stage1", type=int, default=1000)
    parser.add_argument("--n-startup-spore-stage2", type=int, default=400)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--restart-every", type=int, default=200)
    args = parser.parse_args()

    ray.init(num_cpus=args.n_jobs, include_dashboard=False, logging_level="ERROR")

    rows = list(load_actual().itertuples(index=False))
    global_ranges = build_param_ranges()
    spore_df = load_spores()
    spore_windows = build_spore_windows(spore_df)

    rows_ref = ray.put(rows)
    spore_ref = ray.put(spore_windows)
    workers = _make_workers(args.n_jobs, rows_ref, spore_ref)

    # Part A: baseline
    fold_params, summary, global_best_info = run_baseline_loo(
        rows=rows,
        k=args.kfold,
        global_ranges=global_ranges,
        workers=workers,
        rows_ref=rows_ref,
        spore_ref=spore_ref,
        n_jobs=args.n_jobs,
        restart_every=args.restart_every,
        n_trials_stage1=args.n_trials_base_stage1,
        n_trials_stage2=args.n_trials_base_stage2,
        n_startup_stage1=args.n_startup_base_stage1,
        n_startup_stage2=args.n_startup_base_stage2,
        progress_interval=args.progress_interval,
        seed=args.seed,
    )

    # Part B: spore-process continuation from baseline global best
    run_spore_two_stage_loo(
        rows=rows,
        k=args.kfold,
        global_ranges=global_ranges,
        global_best=global_best_info["best_params"],
        workers=workers,
        rows_ref=rows_ref,
        spore_ref=spore_ref,
        n_jobs=args.n_jobs,
        restart_every=args.restart_every,
        n_trials_stage1=args.n_trials_spore_stage1,
        n_trials_stage2=args.n_trials_spore_stage2,
        n_startup_stage1=args.n_startup_spore_stage1,
        n_startup_stage2=args.n_startup_spore_stage2,
        progress_interval=args.progress_interval,
        seed=args.seed,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
