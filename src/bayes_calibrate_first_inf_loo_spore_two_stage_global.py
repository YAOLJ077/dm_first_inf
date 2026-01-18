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
import json
import copy
import argparse
from typing import Dict, List, Tuple

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")

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

SPORE_CORE_DAYS = 14

# Base ranges (global)
BASE_RANGES = {
    "mp_tmp__rh_wet_night": (80, 95),
    "sporulation__MP_min": (2, 6),
    "sporulation__Tmin": (8, 15),
    "sporulation__Topt": (18, 24),
    "sporulation__Tmax": (25, 35),
    "dispersal__rain_min": (2.0, 12.0),
    "dispersal__wind_min": (4.0, 12.0),
    "deposition__rain_min": (4.0, 12.0),
    "zoospore_alive__rain": (3.0, 6.0),
    "zoospore_alive__vpd": (1.0, 4.0),
    "infection__infection_condition": (30, 300),
    "incubation__spl_end": (1.0, 1.5),
    "thresholds__min_spores": (0.01, 0.08),
}

# Local absolute ranges around baseline global best (absolute deltas).
LOCAL_ABS = {
    "mp_tmp__rh_wet_night": 5.0,
    "sporulation__MP_min": 2.0,
    "sporulation__Tmin": 0.5,
    "sporulation__Topt": 0.5,
    "sporulation__Tmax": 1.0,
    "dispersal__rain_min": 2.0,
    "dispersal__wind_min": 2.0,
    "deposition__rain_min": 2.0,
    "zoospore_alive__rain": 1.0,
    "zoospore_alive__vpd": 1.0,
    "infection__infection_condition": 50.0,
    "incubation__spl_end": 0.1,
    "thresholds__min_spores": 0.02,
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
    "mp_tmp__rh_wet_night": 5.0,
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


def build_local_ranges_from_pool(
    df_pool: pd.DataFrame,
    global_ranges: Dict[str, Tuple[float, float]],
    pad_ratio: float = 0.1
) -> Dict[str, Tuple[float, float]]:
    local = {}
    for key, (g_lo, g_hi) in global_ranges.items():
        if key not in df_pool.columns:
            local[key] = (g_lo, g_hi)
            continue
        vmin = float(df_pool[key].min())
        vmax = float(df_pool[key].max())
        span = max(1e-6, vmax - vmin)
        pad = pad_ratio * span
        lo = max(g_lo, vmin - pad)
        hi = min(g_hi, vmax + pad)
        if lo >= hi:
            lo, hi = g_lo, g_hi
        local[key] = (lo, hi)
    return local


def spore_consistency_loss(
    cfg: dict,
    rows: List[pd.Series],
    spore_windows: Dict[Tuple[str, int], List[Tuple[object, object, bool]]],
    core_days: int
) -> float:
    losses = []
    for row in rows:
        key = (str(row.plot), int(row.first_infection_date.year))
        obs = spore_windows.get(key, [])
        if not obs:
            losses.append(1.0)
            continue
        rb_key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(DATA_DIR, f"{rb_key}_grape_rb.json")
        if not os.path.exists(rb_path):
            losses.append(1.0)
            continue
        try:
            rb = load_rb_cached(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
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
    return float(np.mean(losses)) if losses else 1.0


@ray.remote
def eval_mae(params: Dict[str, float], train_rows: List[pd.Series], val_rows: List[pd.Series]) -> Tuple[float, float]:
    cfg = apply_params(config, params)
    train_mae, _, _ = compute_error_stats(cfg, train_rows)
    val_mae, _, _ = compute_error_stats(cfg, val_rows)
    return train_mae, val_mae


@ray.remote
def eval_spore(params: Dict[str, float], train_rows: List[pd.Series], val_rows: List[pd.Series],
               spore_windows: Dict[Tuple[str, int], List[Tuple[object, object, bool]]], core_days: int) -> Tuple[float, float]:
    cfg = apply_params(config, params)
    train_loss = spore_consistency_loss(cfg, train_rows, spore_windows, core_days)
    val_loss = spore_consistency_loss(cfg, val_rows, spore_windows, core_days)
    return train_loss, val_loss


def run_baseline_loo(
    rows: List[pd.Series],
    k: int,
    global_ranges: Dict[str, Tuple[float, float]],
    n_trials_stage1: int,
    n_trials_stage2: int,
    n_startup_stage1: int,
    n_startup_stage2: int,
    train_top_pct: float,
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
        max_conc = max(1, int(ray.available_resources().get("CPU", 1)))
        _stage_start(f"[Baseline][Fold {fold_id}/{k}][Stage1]", n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage1, 0, max_conc, f"[Baseline][Fold {fold_id}/{k}][Stage1]", last_ts, progress_interval)

        while completed < n_trials_stage1:
            while len(pending) < max_conc and completed + len(pending) < n_trials_stage1:
                trial = study1.ask()
                params = define_search_space(trial, global_ranges)
                fut = eval_mae.remote(params, train_rows, val_rows)
                pending[fut] = (trial, params)
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params = pending.pop(fut)
                train_mae, val_mae = ray.get(fut)
                study1.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    **params
                })
                completed += 1
            last_ts = _progress_print(
                completed,
                n_trials_stage1,
                len(pending),
                max_conc,
                f"[Baseline][Fold {fold_id}/{k}][Stage1]",
                last_ts,
                progress_interval
            )

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage1_out = os.path.join(RESULT_DIR, f"baseline_loo_fold_{fold_id}_stage1_trials.csv")
        df_trials.to_csv(stage1_out, index=False)

        if train_top_pct > 0:
            n_keep = max(1, int(len(df_trials) * train_top_pct / 100.0))
            pool = df_trials.head(n_keep)
        else:
            pool = df_trials
        local_ranges = build_local_ranges_from_pool(pool, global_ranges, pad_ratio=0.1)

        # Stage 2 (local)
        sampler2 = TPESampler(seed=seed + fold_id + 1000, multivariate=True, n_startup_trials=n_startup_stage2)
        study2 = optuna.create_study(sampler=sampler2, direction="minimize")
        trial_rows = []
        last_ts = 0.0
        pending = {}
        completed = 0
        _stage_start(f"[Baseline][Fold {fold_id}/{k}][Stage2]", n_trials_stage2, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage2, 0, max_conc, f"[Baseline][Fold {fold_id}/{k}][Stage2]", last_ts, progress_interval)
        while completed < n_trials_stage2:
            while len(pending) < max_conc and completed + len(pending) < n_trials_stage2:
                trial = study2.ask()
                params = define_search_space(trial, local_ranges)
                fut = eval_mae.remote(params, train_rows, val_rows)
                pending[fut] = (trial, params)
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params = pending.pop(fut)
                train_mae, val_mae = ray.get(fut)
                study2.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    **params
                })
                completed += 1
            last_ts = _progress_print(
                completed,
                n_trials_stage2,
                len(pending),
                max_conc,
                f"[Baseline][Fold {fold_id}/{k}][Stage2]",
                last_ts,
                progress_interval
            )

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage2_out = os.path.join(RESULT_DIR, f"baseline_loo_fold_{fold_id}_trials.csv")
        df_trials.to_csv(stage2_out, index=False)

        if train_top_pct > 0:
            n_keep = max(1, int(len(df_trials) * train_top_pct / 100.0))
            pool = df_trials.head(n_keep)
        else:
            pool = df_trials
        best_val_row = pool.sort_values("val_mae").iloc[0]
        best_params = {
            param_name: best_val_row[param_name]
            for param_name in best_val_row.index
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
    spore_windows: Dict[Tuple[str, int], List[Tuple[object, object, bool]]],
    k: int,
    global_ranges: Dict[str, Tuple[float, float]],
    global_best: Dict[str, float],
    n_trials_stage1: int,
    n_trials_stage2: int,
    n_startup_stage1: int,
    n_startup_stage2: int,
    train_top_pct: float,
    progress_interval: float,
    seed: int
) -> None:
    folds = build_folds(len(rows), k, FOLD_SEED)
    stage2_best_rows = []
    stage2_params = []

    stage1_local = build_local_ranges_from_center(global_best, global_ranges, LOCAL_ABS)

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
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, int(ray.available_resources().get("CPU", 1)))
        _stage_start(f"[Spore][Fold {fold_id}/{k}][Stage1]", n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage1, 0, max_conc, f"[Spore][Fold {fold_id}/{k}][Stage1]", last_ts, progress_interval)

        while completed < n_trials_stage1:
            while len(pending) < max_conc and completed + len(pending) < n_trials_stage1:
                trial = study1.ask()
                params = define_search_space(trial, stage1_ranges)
                params = {**fixed_base, **params}
                fut = eval_spore.remote(params, train_rows, val_rows, spore_windows, SPORE_CORE_DAYS)
                pending[fut] = (trial, params)
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params = pending.pop(fut)
                train_loss, val_loss = ray.get(fut)
                study1.tell(trial, train_loss)
                trial_rows.append({
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    **params
                })
                completed += 1
            last_ts = _progress_print(
                completed,
                n_trials_stage1,
                len(pending),
                max_conc,
                f"[Spore][Fold {fold_id}/{k}][Stage1]",
                last_ts,
                progress_interval
            )

        df_stage1 = pd.DataFrame(trial_rows).sort_values("train_loss")
        stage1_out = os.path.join(RESULT_DIR, f"spore_loo_stage1_fold{fold_id}.csv")
        df_stage1.to_csv(stage1_out, index=False)
        if train_top_pct > 0:
            n_keep1 = max(1, int(len(df_stage1) * train_top_pct / 100.0))
            pool1 = df_stage1.head(n_keep1)
        else:
            pool1 = df_stage1
        best1 = pool1.sort_values("val_loss").iloc[0]
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
        last_ts = 0.0
        pending = {}
        completed = 0
        _stage_start(f"[Spore][Fold {fold_id}/{k}][Stage2]", n_trials_stage2, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage2, 0, max_conc, f"[Spore][Fold {fold_id}/{k}][Stage2]", last_ts, progress_interval)
        while completed < n_trials_stage2:
            while len(pending) < max_conc and completed + len(pending) < n_trials_stage2:
                trial = study2.ask()
                params = define_search_space(trial, stage2_ranges)
                params = {**fixed_stage2, **params}
                fut = eval_mae.remote(params, train_rows, val_rows)
                pending[fut] = (trial, params)
            done_ids, _ = ray.wait(list(pending.keys()), num_returns=1)
            for fut in done_ids:
                trial, params = pending.pop(fut)
                train_mae, val_mae = ray.get(fut)
                study2.tell(trial, train_mae)
                trial_rows.append({
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    **params
                })
                completed += 1
            last_ts = _progress_print(
                completed,
                n_trials_stage2,
                len(pending),
                max_conc,
                f"[Spore][Fold {fold_id}/{k}][Stage2]",
                last_ts,
                progress_interval
            )

        df_stage2 = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage2_out = os.path.join(RESULT_DIR, f"spore_loo_stage2_fold{fold_id}.csv")
        df_stage2.to_csv(stage2_out, index=False)
        if train_top_pct > 0:
            n_keep2 = max(1, int(len(df_stage2) * train_top_pct / 100.0))
            pool2 = df_stage2.head(n_keep2)
        else:
            pool2 = df_stage2
        best2 = pool2.sort_values("val_mae").iloc[0]
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
    parser.add_argument("--train-top-pct", type=float, default=0.0)
    parser.add_argument("--n-trials-base-stage1", type=int, default=10000)
    parser.add_argument("--n-trials-base-stage2", type=int, default=6000)
    parser.add_argument("--n-startup-base-stage1", type=int, default=1000)
    parser.add_argument("--n-startup-base-stage2", type=int, default=600)
    parser.add_argument("--n-trials-spore-stage1", type=int, default=10000)
    parser.add_argument("--n-trials-spore-stage2", type=int, default=6000)
    parser.add_argument("--n-startup-spore-stage1", type=int, default=1000)
    parser.add_argument("--n-startup-spore-stage2", type=int, default=600)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    args = parser.parse_args()

    ray.init(num_cpus=args.n_jobs, include_dashboard=False)

    rows = list(load_actual().itertuples(index=False))
    global_ranges = build_param_ranges()

    # Part A: baseline
    fold_params, summary, global_best_info = run_baseline_loo(
        rows=rows,
        k=args.kfold,
        global_ranges=global_ranges,
        n_trials_stage1=args.n_trials_base_stage1,
        n_trials_stage2=args.n_trials_base_stage2,
        n_startup_stage1=args.n_startup_base_stage1,
        n_startup_stage2=args.n_startup_base_stage2,
        train_top_pct=args.train_top_pct,
        progress_interval=args.progress_interval,
        seed=args.seed,
    )

    # Part B: spore-process continuation from baseline global best
    spore_df = load_spores()
    spore_windows = build_spore_windows(spore_df)
    run_spore_two_stage_loo(
        rows=rows,
        spore_windows=spore_windows,
        k=args.kfold,
        global_ranges=global_ranges,
        global_best=global_best_info["best_params"],
        n_trials_stage1=args.n_trials_spore_stage1,
        n_trials_stage2=args.n_trials_spore_stage2,
        n_startup_stage1=args.n_startup_spore_stage1,
        n_startup_stage2=args.n_startup_spore_stage2,
        train_top_pct=args.train_top_pct,
        progress_interval=args.progress_interval,
        seed=args.seed,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
