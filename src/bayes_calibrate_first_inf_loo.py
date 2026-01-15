# -*- coding: utf-8 -*-
"""
Bayesian calibration (Optuna) with LOO.

Per fold:
  - Optimize train_mae on the fold's training set (n-1 samples).
  - Keep top% by train_mae.
  - Select the best by val_mae on the single held-out sample.

Outputs per-fold best params and overall mean train/val MAE.
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
# Make project root importable
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# PATH SETUP
# ======================================================
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
N_JOBS = 64
RANDOM_SEED = 42
FOLD_SEED = 42

# Base ranges from Morris settings
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
    "thresholds__min_spores": (0.02, 0.08),
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
        group, sub = k.split("__")
        cfg[group][sub] = float(v)
    return cfg


_GLOBAL_RANGES: Dict[str, Tuple[float, float]] = {}


def _progress_print(done: int, total: int, running: int, prefix: str, last_ts: float, interval_s: float) -> float:
    now = time.time()
    if now - last_ts >= interval_s:
        msg = f"Progress: {done}/{total} completed | running: {running}"
        if prefix:
            msg = f"{prefix} {msg}"
        print(msg, flush=True)
        return now
    return last_ts


def define_search_space(trial: optuna.Trial) -> Dict[str, float]:
    params = {}
    tmin_lo, tmin_hi = _GLOBAL_RANGES["sporulation__Tmin"]
    topt_lo, topt_hi = _GLOBAL_RANGES["sporulation__Topt"]
    tmax_lo, tmax_hi = _GLOBAL_RANGES["sporulation__Tmax"]

    Tmin = trial.suggest_float("sporulation__Tmin", tmin_lo, tmin_hi)
    Topt = trial.suggest_float("sporulation__Topt", max(topt_lo, Tmin + 0.1), topt_hi)
    Tmax = trial.suggest_float("sporulation__Tmax", max(tmax_lo, Topt + 0.1), tmax_hi)

    params["sporulation__Tmin"] = Tmin
    params["sporulation__Topt"] = Topt
    params["sporulation__Tmax"] = Tmax

    for k, (lo, hi) in _GLOBAL_RANGES.items():
        if k in ("sporulation__Tmin", "sporulation__Topt", "sporulation__Tmax"):
            continue
        params[k] = trial.suggest_float(k, lo, hi)

    return params


def calc_error(actual_date, event_df: pd.DataFrame) -> float:
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


_RB_CACHE: Dict[str, dict] = {}


def load_rb_cached(path: str) -> dict:
    if path not in _RB_CACHE:
        with open(path, "r") as f:
            _RB_CACHE[path] = json.load(f)
    return _RB_CACHE[path]


def compute_mae(cfg: dict, rows: List[pd.Series]) -> float:
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
            err = calc_error(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            err = np.nan
        if np.isfinite(err):
            errors.append(abs(err))
        else:
            errors.append(PENALTY_DAYS)
    if not errors:
        return PENALTY_DAYS
    return float(np.mean(errors))


def build_folds(n: int, k: int, seed: int) -> List[List[int]]:
    if k <= 1 or k > n:
        k = n
    folds = [[] for _ in range(k)]
    rng = np.random.RandomState(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    return folds


@ray.remote
def eval_params(params: Dict[str, float], train_rows: List[pd.Series], val_rows: List[pd.Series]) -> Tuple[float, float]:
    model_cfg = apply_params(config, params)
    train_mae = compute_mae(model_cfg, train_rows)
    val_mae = compute_mae(model_cfg, val_rows)
    return train_mae, val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=8000)
    parser.add_argument("--n-startup", type=int, default=800)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--cpus-per-trial", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--train-top-k", type=int, default=100)
    parser.add_argument("--train-top-pct", type=float, default=5.0)
    parser.add_argument("--kfold", type=int, default=6)
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--no-quiet", dest="quiet", action="store_false")
    args = parser.parse_args()

    ranges = build_param_ranges()
    _GLOBAL_RANGES.update(ranges)
    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))

    ray.init(num_cpus=args.n_jobs, include_dashboard=False)

    n_samples = len(rows)
    k = args.kfold if args.kfold > 0 else n_samples
    folds = build_folds(n_samples, k, FOLD_SEED)

    fold_best_rows = []
    fold_params = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = [i for i in range(n_samples) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        sampler = TPESampler(
            seed=args.seed + fold_id,
            multivariate=True,
            n_startup_trials=args.n_startup,
            warn_independent_sampling=False
        )
        study = optuna.create_study(sampler=sampler, direction="minimize")
        max_conc = args.max_concurrent
        if max_conc <= 0:
            max_conc = max(1, args.n_jobs // max(1, args.cpus_per_trial))

        pending = []
        completed = 0
        last_ts = 0.0
        prefix = f"[Fold {fold_id}/{k}]"
        trial_rows = []

        while completed < args.n_trials:
            while len(pending) < max_conc and completed + len(pending) < args.n_trials:
                trial = study.ask()
                params = define_search_space(trial)
                fut = eval_params.remote(params, train_rows, val_rows)
                pending.append((trial, params, fut))

            done_ids, _ = ray.wait([f for _, _, f in pending], num_returns=1)
            for trial, params, fut in list(pending):
                if fut in done_ids:
                    train_mae, val_mae = ray.get(fut)
                    study.tell(trial, train_mae)
                    trial_rows.append({
                        "trial_number": trial.number,
                        "train_mae": float(train_mae),
                        "val_mae": float(val_mae),
                        **params
                    })
                    pending.remove((trial, params, fut))
                    completed += 1
                    break

            running = len(pending)
            last_ts = _progress_print(
                completed,
                args.n_trials,
                running,
                prefix,
                last_ts,
                args.progress_interval
            )

        df_trials = pd.DataFrame(trial_rows)
        fold_trials_path = os.path.join(RESULT_DIR, f"loo_fold_{fold_id}_trials.csv")
        df_trials.to_csv(fold_trials_path, index=False)

        pool = df_trials.dropna(subset=["train_mae", "val_mae"]).copy()
        pool = pool.sort_values("train_mae", ascending=True)
        if args.train_top_pct > 0:
            n_keep = max(1, int(len(pool) * args.train_top_pct / 100.0))
        else:
            n_keep = min(args.train_top_k, len(pool))
        pool = pool.head(n_keep)

        best_val_row = pool.sort_values("val_mae", ascending=True).iloc[0]
        best_params = {
            k: best_val_row[k]
            for k in best_val_row.index
            if k not in ("trial_number", "train_mae", "val_mae")
        }
        fold_params.append(best_params)
        fold_best_rows.append({
            "fold": fold_id,
            "avg_train_mae": float(best_val_row["train_mae"]),
            "val_mae": float(best_val_row["val_mae"]),
        })

        best_out = os.path.join(RESULT_DIR, f"loo_best_params_fold{fold_id}.json")
        with open(best_out, "w") as f:
            json.dump({
                "fold": fold_id,
                "avg_train_mae": float(best_val_row["train_mae"]),
                "val_mae": float(best_val_row["val_mae"]),
                "params": best_params,
            }, f, indent=2)

    summary = pd.DataFrame(fold_best_rows)
    summary_out = os.path.join(RESULT_DIR, "loo_summary.csv")
    summary.to_csv(summary_out, index=False)

    summary_stats = {
        "train_mae_mean": float(summary["avg_train_mae"].mean()),
        "train_mae_std": float(summary["avg_train_mae"].std()),
        "val_mae_mean": float(summary["val_mae"].mean()),
        "val_mae_std": float(summary["val_mae"].std()),
        "train_rmse": float(np.sqrt(np.mean(np.square(summary["avg_train_mae"])))), 
        "val_rmse": float(np.sqrt(np.mean(np.square(summary["val_mae"])))), 
        "n_folds": int(len(summary)),
    }
    summary_json = os.path.join(RESULT_DIR, "loo_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Global selection: evaluate each fold-best params on all samples.
    global_rows = []
    for i, params in enumerate(fold_params, start=1):
        cfg = apply_params(config, params)
        global_mae = compute_mae(cfg, rows)
        global_rows.append({
            "fold": i,
            "global_mae": float(global_mae),
        })
    df_global = pd.DataFrame(global_rows)
    best_idx = int(df_global["global_mae"].idxmin())
    best_global = {
        "best_from_fold": int(df_global.loc[best_idx, "fold"]),
        "global_mae": float(df_global.loc[best_idx, "global_mae"]),
        "params": fold_params[best_idx],
    }
    global_out = os.path.join(RESULT_DIR, "loo_best_global.json")
    with open(global_out, "w") as f:
        json.dump(best_global, f, indent=2)

    if not args.quiet:
        print(f"Saved LOO summary: {summary_out}")
        print(f"Saved LOO summary json: {summary_json}")
        print(f"Saved LOO global best: {global_out}")

    ray.shutdown()


if __name__ == "__main__":
    main()
