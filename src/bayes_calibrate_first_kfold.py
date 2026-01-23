# -*- coding: utf-8 -*-
"""
Baseline calibration (Bayes + TPE + k-fold).

Per fold:
  - Stage1: optimize train_mae; select best by val_mae within train topK
            under the constraint val_mae >= train_mae (or closest if none).
  - Stage2: local refinement around stage1 best with the same selection rule.
Outputs:
  - Per-fold best params and metrics (mae/std/rmse/coverage/precision/recall).
  - Summary across folds.
  - Global-best: evaluate each fold-best params on all samples.
"""

import os
import sys
import json
import copy
import argparse
import math
from typing import Dict, List, Tuple
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
TOPK_PCT = 1.0

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
    "mp_tmp__rh_wet_night": 1.0,
    "sporulation__MP_min": 1.0,
    "sporulation__Tmin": 0.5,
    "sporulation__Topt": 0.5,
    "sporulation__Tmax": 1.0,
    "dispersal__rain_min": 2.0,
    "dispersal__wind_min": 2.0,
    "deposition__rain_min": 2.0,
    "zoospore_alive__rain": 1.0,
    "zoospore_alive__vpd": 1.0,
    "infection__infection_condition": 25.0,
    "incubation__spl_end": 0.1,
    "thresholds__min_spores": 0.02,
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


def apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in values.items():
        if "__" not in k:
            continue
        group, sub = k.split("__", 1)
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


def _calc_coverage(actual_date, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return 0.0
    first = event_df.iloc[0]
    start = pd.to_datetime(first["oil_spot_start"]).normalize()
    end = pd.to_datetime(first["oil_spot_end"]).normalize()
    act = pd.to_datetime(actual_date).normalize()
    return 1.0 if start <= act <= end else 0.0


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


def compute_error_stats_with_coverage(cfg: dict, rows: List[pd.Series]) -> Tuple[float, float, float, float, float, float]:
    errors = []
    covers = []
    for row in rows:
        key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            errors.append(float(PENALTY_DAYS))
            covers.append(0.0)
            continue
        try:
            rb = load_rb_cached(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            err = _calc_error(row.first_infection_date, cs.primary_chain_unique)
            cov = _calc_coverage(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            err = np.nan
            cov = 0.0
        if np.isfinite(err):
            errors.append(float(err))
        else:
            errors.append(float(PENALTY_DAYS))
        covers.append(float(cov))
    if not errors:
        return PENALTY_DAYS, 0.0, PENALTY_DAYS, 0.0, 0.0, 0.0
    abs_err = np.abs(np.array(errors, dtype=float))
    mae = float(abs_err.mean())
    std = float(abs_err.std(ddof=0))
    rmse = float(np.sqrt(np.mean(np.square(abs_err))))
    cov_mean = float(np.mean(covers)) if covers else 0.0
    return mae, std, rmse, cov_mean, cov_mean, cov_mean


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
    interval_s: float
) -> float:
    now = time.time()
    if now - last_ts >= interval_s:
        msg = f"Progress: {done}/{total} completed | running: {running}/{running_max}"
        if prefix:
            msg = f"{prefix} {msg}"
        print(msg, flush=True)
        return now
    return last_ts


def _stage_start(prefix: str, total: int, running: int, running_max: int) -> None:
    msg = f"{prefix} Start | total trials: {total} | running: {running}/{running_max}"
    print(msg, flush=True)



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


def _topk_by_train(df: pd.DataFrame, pct: float, loss_col: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty trials dataframe.")
    df_sorted = df.sort_values(loss_col).reset_index(drop=False)
    n_top = max(1, int(math.ceil(len(df_sorted) * pct / 100.0)))
    return df_sorted.iloc[:n_top]


def select_topk_val_best(df: pd.DataFrame, pct: float) -> pd.Series:
    if df.empty:
        raise ValueError("Empty trials dataframe.")
    df_sorted = df.sort_values("train_mae").reset_index(drop=True)
    n_top = max(1, int(math.ceil(len(df_sorted) * pct / 100.0)))
    top = df_sorted.iloc[:n_top]
    return top.sort_values("val_mae").iloc[0]


def select_topk_val_best_loss(df: pd.DataFrame, pct: float) -> pd.Series:
    if df.empty:
        raise ValueError("Empty trials dataframe.")
    df_sorted = df.sort_values("train_loss").reset_index(drop=True)
    n_top = max(1, int(math.ceil(len(df_sorted) * pct / 100.0)))
    top = df_sorted.iloc[:n_top]
    return top.sort_values("val_loss").iloc[0]


def select_topk_val_best_constrained(df: pd.DataFrame, pct: float) -> pd.Series:
    if df.empty:
        raise ValueError("Empty trials dataframe.")
    df_sorted = df.sort_values("train_mae").reset_index(drop=True)
    n_top = max(1, int(math.ceil(len(df_sorted) * pct / 100.0)))
    top = df_sorted.iloc[:n_top].copy()
    valid = top[top["val_mae"] > top["train_mae"]]
    if not valid.empty:
        best = valid.sort_values("val_mae").iloc[0]
    else:
        equal = top[top["val_mae"] == top["train_mae"]]
        if not equal.empty:
            best = equal.sort_values("val_mae").iloc[0]
        else:
            diff = (top["val_mae"] - top["train_mae"]).abs()
            best = top.iloc[diff.values.argmin()]
    if "train_std" in top.columns:
        same = top[top["val_mae"] == best["val_mae"]]
        if len(same) > 1:
            best = same.sort_values("train_std").iloc[0]
    return best


@ray.remote(num_cpus=1)
class ModelWorker:
    def __init__(self, rows_ref):
        if isinstance(rows_ref, ray.ObjectRef):
            self.rows = ray.get(rows_ref)
        else:
            self.rows = rows_ref
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

    def _compute_error_stats_idx(self, cfg: dict, idxs: List[int]) -> Tuple[float, float, float]:
        errors = []
        for i in idxs:
            row = self.rows[i]
            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
            if not os.path.exists(rb_path):
                continue
            try:
                rb = self._load_rb_cached_lru(rb_path)
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

    def eval_mae(
        self, params: Dict[str, float], train_idx: List[int], val_idx: List[int]
    ) -> Tuple[float, float, float, float, float, float]:
        cfg = apply_params(config, params)
        train_mae, train_std, train_rmse = self._compute_error_stats_idx(cfg, train_idx)
        val_mae, val_std, val_rmse = self._compute_error_stats_idx(cfg, val_idx)
        return train_mae, val_mae, train_std, val_std, train_rmse, val_rmse


def _make_workers(n_jobs: int, rows_ref) -> List[ray.actor.ActorHandle]:
    return [ModelWorker.remote(rows_ref) for _ in range(n_jobs)]


def _replace_worker(workers: List[ray.actor.ActorHandle], old_worker, rows_ref):
    try:
        idx = workers.index(old_worker)
    except ValueError:
        idx = None
    ray.kill(old_worker, no_restart=True)
    new_worker = ModelWorker.remote(rows_ref)
    if idx is not None:
        workers[idx] = new_worker
    else:
        workers.append(new_worker)
    return new_worker


def run_baseline_kfold(
    rows: List[pd.Series],
    k: int,
    global_ranges: Dict[str, Tuple[float, float]],
    workers: List[ray.actor.ActorHandle],
    rows_ref,
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
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        _stage_start(f"[Baseline][Fold {fold_id}/{k}][Stage1]", n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage1, 0, max_conc, f"[Baseline][Fold {fold_id}/{k}][Stage1]", last_ts, progress_interval)

        while completed < n_trials_stage1:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage1:
                trial = study1.ask()
                params = define_search_space(trial, global_ranges)
                worker = available.pop(0)
                fut = worker.eval_mae.remote(params, train_idx, val_idx)
                pending[fut] = (trial, params, worker)
            if not pending:
                time.sleep(0.01)
                continue
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params, worker = pending.pop(fut)
                train_mae, val_mae, train_std, val_std, train_rmse, val_rmse = ray.get(fut)
                study1.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    "train_std": float(train_std),
                    "val_std": float(val_std),
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse),
                    **params
                })
                completed += 1
                if next_restart is not None and completed >= next_restart:
                    worker = _replace_worker(workers, worker, rows_ref)
                    next_restart += restart_every
                available.append(worker)
            last_ts = _progress_print(completed, n_trials_stage1, len(pending), max_conc, f"[Baseline][Fold {fold_id}/{k}][Stage1]", last_ts, progress_interval)

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage1_out = os.path.join(RESULT_DIR, f"baseline_kfold_fold_{fold_id}_stage1_trials.csv")
        df_trials.to_csv(stage1_out, index=False)
        best_row = select_topk_val_best_constrained(df_trials, TOPK_PCT)
        best_params_stage1 = {
            param_name: best_row[param_name]
            for param_name in best_row.index
            if "__" in param_name
        }
        local_ranges = build_local_ranges_from_center(best_params_stage1, global_ranges, LOCAL_ABS_BASELINE)

        # Stage 2 (local)
        sampler2 = TPESampler(seed=seed + fold_id + 1000, multivariate=True, n_startup_trials=n_startup_stage2)
        study2 = optuna.create_study(sampler=sampler2, direction="minimize")
        trial_rows = []
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        _stage_start(f"[Baseline][Fold {fold_id}/{k}][Stage2]", n_trials_stage2, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage2, 0, max_conc, f"[Baseline][Fold {fold_id}/{k}][Stage2]", last_ts, progress_interval)
        while completed < n_trials_stage2:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage2:
                trial = study2.ask()
                params = define_search_space(trial, local_ranges)
                worker = available.pop(0)
                fut = worker.eval_mae.remote(params, train_idx, val_idx)
                pending[fut] = (trial, params, worker)
            if not pending:
                time.sleep(0.01)
                continue
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params, worker = pending.pop(fut)
                train_mae, val_mae, train_std, val_std, train_rmse, val_rmse = ray.get(fut)
                study2.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    "train_std": float(train_std),
                    "val_std": float(val_std),
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse),
                    **params
                })
                completed += 1
                if next_restart is not None and completed >= next_restart:
                    worker = _replace_worker(workers, worker, rows_ref)
                    next_restart += restart_every
                available.append(worker)
            last_ts = _progress_print(completed, n_trials_stage2, len(pending), max_conc, f"[Baseline][Fold {fold_id}/{k}][Stage2]", last_ts, progress_interval)

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage2_out = os.path.join(RESULT_DIR, f"baseline_kfold_fold_{fold_id}_trials.csv")
        df_trials.to_csv(stage2_out, index=False)
        best_row = select_topk_val_best_constrained(df_trials, TOPK_PCT)
        best_params = {
            param_name: best_row[param_name]
            for param_name in best_row.index
            if "__" in param_name
        }
        fold_params.append(best_params)

        cfg = apply_params(config, best_params)
        train_mae, train_std, train_rmse, train_cov, train_prec, train_rec = compute_error_stats_with_coverage(cfg, train_rows)
        val_mae, val_std, val_rmse, val_cov, val_prec, val_rec = compute_error_stats_with_coverage(cfg, val_rows)

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
            "train_coverage": float(train_cov),
            "val_coverage": float(val_cov),
            "train_precision": float(train_prec),
            "val_precision": float(val_prec),
            "train_recall": float(train_rec),
            "val_recall": float(val_rec),
        })

        best_out = os.path.join(RESULT_DIR, f"baseline_kfold_best_params_fold{fold_id}.json")
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
                "train_coverage": float(train_cov),
                "val_coverage": float(val_cov),
                "train_precision": float(train_prec),
                "val_precision": float(val_prec),
                "train_recall": float(train_rec),
                "val_recall": float(val_rec),
                "params": best_params,
            }, f, indent=2)

    summary = pd.DataFrame(fold_best_rows)
    summary_out = os.path.join(RESULT_DIR, "baseline_kfold_summary.csv")
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
        "train_coverage_mean": float(summary["train_coverage"].mean()),
        "train_coverage_std": float(summary["train_coverage"].std(ddof=0)),
        "val_coverage_mean": float(summary["val_coverage"].mean()),
        "val_coverage_std": float(summary["val_coverage"].std(ddof=0)),
        "train_precision_mean": float(summary["train_precision"].mean()),
        "train_precision_std": float(summary["train_precision"].std(ddof=0)),
        "val_precision_mean": float(summary["val_precision"].mean()),
        "val_precision_std": float(summary["val_precision"].std(ddof=0)),
        "train_recall_mean": float(summary["train_recall"].mean()),
        "train_recall_std": float(summary["train_recall"].std(ddof=0)),
        "val_recall_mean": float(summary["val_recall"].mean()),
        "val_recall_std": float(summary["val_recall"].std(ddof=0)),
        "n_folds": int(len(summary)),
    }
    summary_json = os.path.join(RESULT_DIR, "baseline_kfold_summary.json")
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
    global_out = os.path.join(RESULT_DIR, "baseline_kfold_best_global.json")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-trials-base-stage1", type=int, default=18000)
    parser.add_argument("--n-trials-base-stage2", type=int, default=6000)
    parser.add_argument("--n-startup-base-stage1", type=int, default=3600)
    parser.add_argument("--n-startup-base-stage2", type=int, default=6000)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--restart-every", type=int, default=200)
    args = parser.parse_args()

    ray.init(num_cpus=args.n_jobs, include_dashboard=False, logging_level="ERROR")

    rows = list(load_actual().itertuples(index=False))
    global_ranges = build_param_ranges()
    rows_ref = ray.put(rows)
    workers = _make_workers(args.n_jobs, rows_ref)

    # Part A: baseline
    if args.skip_baseline:
        baseline_best_path = os.path.join(RESULT_DIR, "baseline_kfold_best_global.json")
        if not os.path.exists(baseline_best_path):
            raise FileNotFoundError(f"Missing baseline best file: {baseline_best_path}")
        with open(baseline_best_path, "r") as f:
            global_best_info = json.load(f)
        if "best_params" not in global_best_info:
            if "params" in global_best_info:
                global_best_info["best_params"] = global_best_info["params"]
            else:
                raise KeyError(f"Missing 'best_params' in {baseline_best_path}. Keys: {list(global_best_info.keys())}")
    else:
        fold_params, summary, global_best_info = run_baseline_kfold(
            rows=rows,
            k=args.kfold,
            global_ranges=global_ranges,
            workers=workers,
            rows_ref=rows_ref,
            n_jobs=args.n_jobs,
            restart_every=args.restart_every,
            n_trials_stage1=args.n_trials_base_stage1,
            n_trials_stage2=args.n_trials_base_stage2,
            n_startup_stage1=args.n_startup_base_stage1,
            n_startup_stage2=args.n_startup_base_stage2,
            progress_interval=args.progress_interval,
            seed=args.seed,
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
