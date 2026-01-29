# -*- coding: utf-8 -*-
"""
Baseline calibration (Bayes + TPE + k-fold, single-stage).

Per fold:
  - Single-stage global search (1.5x expanded ranges) to optimize train_mae;
    select best by val_mae within train topK under the constraint
    val_mae > train_mae (or equal/closest if none).
Outputs:
  - Per-fold best params and metrics (mae/std/rmse).
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

from model.crop_season_spore import CropSeasonFirstSpore
from model.config import config

# ======================================================
# SETTINGS
# ======================================================
PENALTY_DAYS = 120.0
RANDOM_SEED = 42
FOLD_SEED = 42
N_JOBS = 128
RB_CACHE_MAX = 64
TOPK_PCT = 5.0

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
    "zoospore_alive__rain": (3.0, 8.0),
    "zoospore_alive__vpd": (1.5, 4.0),
    "infection__infection_condition": (50, 150),
    "incubation__spl_end": (1.0, 1.5),
    "thresholds__min_spores": (0.05, 0.25),
    "spore__alpha": (-2.0, 2.0)
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
        if group not in cfg:
            cfg[group] = {}
        cfg[group][sub] = float(v)
    return cfg


def _normalize_spore_series(spore_vals: pd.Series) -> pd.Series:
    vals = spore_vals.dropna().astype(float)
    if vals.empty:
        return pd.Series(dtype=float)
    p95 = float(vals.quantile(0.95))
    if p95 <= 0:
        return spore_vals.fillna(0.0).astype(float) * 0.0
    norm = (np.log1p(spore_vals.astype(float)) / np.log1p(p95)).clip(lower=0.0, upper=1.0)
    return norm


def build_spore_day_map(actual_df: pd.DataFrame) -> Dict[Tuple[str, int], pd.Series]:
    spore_path = os.path.join(DATA_DIR, "Spore_Overall.xlsx")
    spore = pd.read_excel(spore_path)
    spore["date"] = pd.to_datetime(spore["date"])

    actual_df = actual_df.copy()
    actual_df["first_infection_date"] = pd.to_datetime(actual_df["first_infection_date"], errors="coerce")
    actual_inf = {
        (str(r.plot), int(r.year)): r.first_infection_date
        for r in actual_df.itertuples(index=False)
    }

    spore_day_map: Dict[Tuple[str, int], pd.Series] = {}
    for (plot, year), g in spore.groupby(["plot", "year"]):
        key = (str(plot), int(year))
        if key not in actual_inf:
            continue
        g = g.sort_values("date")
        obs = g[g["spore"].notna()].copy()
        if obs.empty:
            continue
        start = obs["date"].min().normalize()
        end = obs["date"].max().normalize()
        obs_inf = actual_inf.get(key)
        if pd.notna(obs_inf):
            end = min(end, obs_inf.normalize())
        if end < start:
            continue

        daily_index = pd.date_range(start=start, end=end, freq="D")
        series = pd.Series(index=daily_index, dtype=float)
        obs = obs[(obs["date"] >= start) & (obs["date"] <= end)]
        if obs.empty:
            continue
        series.loc[obs["date"].dt.normalize()] = obs["spore"].values
        series = series.interpolate(method="time", limit_direction="both")
        series = _normalize_spore_series(series)
        spore_day_map[key] = series

    return spore_day_map


_RB_CACHE: Dict[str, dict] = {}
SPORE_DAY_MAP: Dict[Tuple[str, int], pd.Series] = {}


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
            spore_day = SPORE_DAY_MAP.get((str(row.plot), int(row.first_infection_date.year)))
            cs = CropSeasonFirstSpore(csrb=rb, config=cfg, spore_day=spore_day)
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
    def __init__(self, rows_ref, spore_day_map):
        if isinstance(rows_ref, ray.ObjectRef):
            self.rows = ray.get(rows_ref)
        else:
            self.rows = rows_ref
        if isinstance(spore_day_map, ray.ObjectRef):
            self.spore_day_map = ray.get(spore_day_map)
        else:
            self.spore_day_map = spore_day_map
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
                spore_day = self.spore_day_map.get((str(row.plot), int(row.first_infection_date.year)))
                cs = CropSeasonFirstSpore(csrb=rb, config=cfg, spore_day=spore_day)
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


def _make_workers(n_jobs: int, rows_ref, spore_day_map_ref) -> List[ray.actor.ActorHandle]:
    return [ModelWorker.remote(rows_ref, spore_day_map_ref) for _ in range(n_jobs)]


def _replace_worker(workers: List[ray.actor.ActorHandle], old_worker, rows_ref, spore_day_map_ref):
    try:
        idx = workers.index(old_worker)
    except ValueError:
        idx = None
    ray.kill(old_worker, no_restart=True)
    new_worker = ModelWorker.remote(rows_ref, spore_day_map_ref)
    if idx is not None:
        workers[idx] = new_worker
    else:
        workers.append(new_worker)
    return new_worker


def run_baseline_kfold(
    rows: List[pd.Series],
    k: int,
    stage1_ranges: Dict[str, Tuple[float, float]],
    workers: List[ray.actor.ActorHandle],
    rows_ref,
    spore_day_map_ref,
    n_jobs: int,
    restart_every: int,
    n_trials_stage1: int,
    n_startup_stage1: int,
    progress_interval: float,
    seed: int,
    fold_ids: List[int],
    fold_seed: int,
    seed_tag: str = ""
) -> Tuple[List[Dict[str, float]], pd.DataFrame, Dict[str, float]]:
    folds = build_folds(len(rows), k, fold_seed)
    fold_best_rows = []
    fold_params = []

    for fold_id, val_idx in enumerate(folds, start=1):
        if fold_ids and fold_id not in fold_ids:
            continue
        train_idx = [i for i in range(len(rows)) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        sampler = TPESampler(seed=seed + fold_id, multivariate=True, n_startup_trials=n_startup_stage1)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        trial_rows = []
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        prefix = f"{seed_tag}[Baseline][Fold {fold_id}/{k}]"
        _stage_start(prefix, n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage1, 0, max_conc, prefix, last_ts, progress_interval)

        while completed < n_trials_stage1:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage1:
                trial = study.ask()
                params = define_search_space(trial, stage1_ranges)
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
                study.tell(trial, train_mae)
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
                    worker = _replace_worker(workers, worker, rows_ref, spore_day_map_ref)
                    next_restart += restart_every
                available.append(worker)
            last_ts = _progress_print(completed, n_trials_stage1, len(pending), max_conc, prefix, last_ts, progress_interval)

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        fold_dir = os.path.join(RESULT_DIR, f"baseline_fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        trials_out = os.path.join(fold_dir, "baseline_kfold_trials.csv")
        df_trials.to_csv(trials_out, index=False)
        best_row = select_topk_val_best_constrained(df_trials, TOPK_PCT)
        best_params = {
            param_name: best_row[param_name]
            for param_name in best_row.index
            if "__" in param_name
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

        best_out = os.path.join(fold_dir, "baseline_kfold_best_params.json")
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
    global RESULT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--seed-list", type=str, default="42,123,2024")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--folds", type=str, default="")
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    parser.add_argument("--n-trials-base", type=int, default=20000)
    parser.add_argument("--n-startup-base", type=int, default=8000)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--restart-every", type=int, default=200)
    args = parser.parse_args()

    seed_list = []
    if args.seed_list.strip():
        seed_list = [int(x) for x in args.seed_list.split(",") if x.strip()]
    if not seed_list:
        seed_list = [args.seed]

    fold_ids: List[int] = []
    if args.fold is not None:
        fold_ids = [int(args.fold)]
    elif args.folds.strip():
        fold_ids = [int(x) for x in args.folds.split(",") if x.strip()]

    ray.init(num_cpus=args.n_jobs, include_dashboard=False, logging_level="ERROR")

    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))
    spore_day_map = build_spore_day_map(actual_df)
    global SPORE_DAY_MAP
    SPORE_DAY_MAP = spore_day_map
    stage1_ranges = build_param_ranges()
    rows_ref = ray.put(rows)
    spore_day_map_ref = ray.put(spore_day_map)
    workers = _make_workers(args.n_jobs, rows_ref, spore_day_map_ref)

    base_result_dir = RESULT_DIR
    seed_summaries = []
    seed_globals = []

    for seed in seed_list:
        RESULT_DIR = os.path.join(base_result_dir, f"seed_{seed}")
        os.makedirs(RESULT_DIR, exist_ok=True)
        print(f"[Seed {seed}] Output -> {RESULT_DIR}", flush=True)

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
            _, _, _ = run_baseline_kfold(
                rows=rows,
                k=args.kfold,
                stage1_ranges=stage1_ranges,
                workers=workers,
                rows_ref=rows_ref,
                spore_day_map_ref=spore_day_map_ref,
                n_jobs=args.n_jobs,
                restart_every=args.restart_every,
                n_trials_stage1=args.n_trials_base,
                n_startup_stage1=args.n_startup_base,
                progress_interval=args.progress_interval,
                seed=seed,
                fold_ids=fold_ids,
                fold_seed=args.fold_seed,
                seed_tag=f"[Seed {seed}] ",
            )

        summary_path = os.path.join(RESULT_DIR, "baseline_kfold_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                s = json.load(f)
            s["seed"] = seed
            seed_summaries.append(s)
        global_path = os.path.join(RESULT_DIR, "baseline_kfold_best_global.json")
        if os.path.exists(global_path):
            with open(global_path, "r") as f:
                g = json.load(f)
            g["seed"] = seed
            seed_globals.append(g)

    if len(seed_list) > 1 and seed_summaries:
        df = pd.DataFrame(seed_summaries)
        seed_summary_dir = os.path.join(base_result_dir, "seed_summary")
        os.makedirs(seed_summary_dir, exist_ok=True)
        seed_csv = os.path.join(seed_summary_dir, "baseline_kfold_summary_seeds.csv")
        df.to_csv(seed_csv, index=False)
        agg = {}
        for col in df.columns:
            if col == "seed":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                agg[f"{col}_mean"] = float(df[col].mean())
                agg[f"{col}_std"] = float(df[col].std(ddof=0))
        agg["n_seeds"] = int(len(seed_summaries))
        summary_path = os.path.join(seed_summary_dir, "baseline_kfold_summary_seeds.json")
        with open(summary_path, "w") as f:
            json.dump(agg, f, indent=2)
        if seed_globals:
            df_g = pd.DataFrame(seed_globals)
            globals_csv = os.path.join(seed_summary_dir, "baseline_kfold_best_global_seeds.csv")
            df_g.to_csv(globals_csv, index=False)
            if "global_mae" in df_g.columns:
                agg_g = {
                    "global_mae_mean": float(df_g["global_mae"].mean()),
                    "global_mae_std": float(df_g["global_mae"].std(ddof=0)),
                    "n_seeds": int(len(seed_globals)),
                }
                gpath = os.path.join(seed_summary_dir, "baseline_kfold_best_global_seeds.json")
                with open(gpath, "w") as f:
                    json.dump(agg_g, f, indent=2)

    ray.shutdown()


if __name__ == "__main__":
    main()
