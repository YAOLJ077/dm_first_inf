# -*- coding: utf-8 -*-
"""
Bayesian calibration (Optuna) for FIRST infection model.
Objective: minimize LOOCV MAE with penalty for missing events.
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
from ray import tune
from ray.tune import CLIReporter, Callback
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
import optuna
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage
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
PENALTY_DAYS = 90.0
N_JOBS = min(64, os.cpu_count())
RANDOM_SEED = 42

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
    return df


def to_doy(dt):
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear


def calc_error(actual_date, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return np.nan

    first = event_df.iloc[0]
    doy_actual = to_doy(actual_date)
    doy_start = to_doy(first["oil_spot_start"])
    doy_end = to_doy(first["oil_spot_end"])

    if doy_actual is None or doy_start is None or doy_end is None:
        return np.nan

    if doy_end < doy_actual:
        return doy_end - doy_actual
    if doy_start > doy_actual:
        return doy_start - doy_actual
    return 0.0


_RB_CACHE: Dict[str, dict] = {}


def load_rb_cached(rb_path: str) -> dict:
    if rb_path not in _RB_CACHE:
        with open(rb_path, "r") as f:
            _RB_CACHE[rb_path] = json.load(f)
    return _RB_CACHE[rb_path]


def apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in values.items():
        group, sub = k.split("__")
        cfg[group][sub] = float(v)
    return cfg


_GLOBAL_RANGES: Dict[str, Tuple[float, float]] = {}

def short_trial_dirname(trial):
    return f"trial_{trial.trial_id}"


class ProgressOnlyCallback(Callback):
    def __init__(self, total: int, interval_s: float = 5.0, prefix: str = ""):
        self.total = total
        self.interval_s = interval_s
        self._last_ts = 0.0
        self.prefix = prefix

    def on_trial_result(self, iteration, trials, trial, result, **info):
        done = sum(t.status == "TERMINATED" for t in trials)
        now = time.time()
        if now - self._last_ts >= self.interval_s:
            self._last_ts = now
            msg = f"Progress: {done}/{self.total} trials completed"
            if self.prefix:
                msg = f"{self.prefix} {msg}"
            print(msg, flush=True)

def define_search_space(trial: optuna.Trial) -> Dict[str, float]:
    """
    Optuna define-by-run search space with Tmin < Topt < Tmax.
    """
    params = {}

    tmin_lo, tmin_hi = _GLOBAL_RANGES["sporulation__Tmin"]
    topt_lo, topt_hi = _GLOBAL_RANGES["sporulation__Topt"]
    tmax_lo, tmax_hi = _GLOBAL_RANGES["sporulation__Tmax"]

    Tmin = trial.suggest_float("sporulation__Tmin", tmin_lo, tmin_hi)
    Topt = trial.suggest_float(
        "sporulation__Topt",
        max(topt_lo, Tmin + 0.1),
        topt_hi
    )
    Tmax = trial.suggest_float(
        "sporulation__Tmax",
        max(tmax_lo, Topt + 0.1),
        tmax_hi
    )

    params["sporulation__Tmin"] = Tmin
    params["sporulation__Topt"] = Topt
    params["sporulation__Tmax"] = Tmax

    for k, (lo, hi) in _GLOBAL_RANGES.items():
        if k in ("sporulation__Tmin", "sporulation__Topt", "sporulation__Tmax"):
            continue
        params[k] = trial.suggest_float(k, lo, hi)

    return params


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


def build_folds(n: int, k: int) -> List[List[int]]:
    if k <= 1 or k > n:
        k = n
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(range(n)):
        folds[i % k].append(idx)
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=6000)
    parser.add_argument("--n-startup", type=int, default=600)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--cpus-per-trial", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "result", "ray_results")
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--train-top-k", type=int, default=100)
    parser.add_argument("--train-top-pct", type=float, default=1.0)
    parser.add_argument("--kfold", type=int, default=3)
    args = parser.parse_args()

    ranges = build_param_ranges()
    _GLOBAL_RANGES.update(ranges)
    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))

    reporter = CLIReporter(metric_columns=["train_mae"])
    verbose = 1
    log_to_file = False
    if args.quiet:
        os.environ.pop("AIR_VERBOSITY", None)
        reporter = None
        verbose = 0
        log_to_file = True

    ray.init(num_cpus=args.n_jobs, include_dashboard=False)

    n_samples = len(rows)
    k = args.kfold if args.kfold > 0 else n_samples
    folds = build_folds(n_samples, k)

    fold_rows = []
    overall_best = {
        "best_train_params": None,
        "best_train_mae": None,
        "best_train_val_mae": None,
        "best_val_params": None,
        "best_val_mae": None,
        "best_val_train_mae": None,
    }

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

        optuna_storage = InMemoryStorage()
        optuna_search = OptunaSearch(
            space=define_search_space,
            sampler=sampler,
            metric="train_mae",
            mode="min",
            storage=optuna_storage,
            study_name=f"first_inf_mae_fold_{fold_id}"
        )
        if args.max_concurrent > 0:
            optuna_search = ConcurrencyLimiter(
                optuna_search,
                max_concurrent=args.max_concurrent
            )

        def trainable(cfg):
            model_cfg = apply_params(config, cfg)
            train_mae = compute_mae(model_cfg, train_rows)
            val_mae = compute_mae(model_cfg, val_rows)
            tune.report({"train_mae": train_mae, "val_mae": val_mae})

        callbacks = [
            ProgressOnlyCallback(
                args.n_trials,
                args.progress_interval,
                prefix=f"[Fold {fold_id}/{k}]"
            )
        ]

        analysis = tune.run(
            trainable,
            name=f"first_inf_mae_fold_{fold_id}",
            num_samples=args.n_trials,
            search_alg=optuna_search,
            config={},
            resources_per_trial={"cpu": args.cpus_per_trial},
            progress_reporter=reporter,
            storage_path=args.storage_path,
            trial_dirname_creator=short_trial_dirname,
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
            callbacks=callbacks,
            log_to_file=log_to_file,
            verbose=verbose
        )

        df_trials = analysis.results_df
        fold_trials_path = os.path.join(RESULT_DIR, f"fold_{fold_id}_trials.csv")
        df_trials.to_csv(fold_trials_path, index=False)

        best_trial = analysis.get_best_trial(metric="train_mae", mode="min", scope="all")
        best_cfg = best_trial.config
        best_train = best_trial.last_result.get("train_mae")
        best_val = best_trial.last_result.get("val_mae", best_train)

        pool = df_trials.dropna(subset=["train_mae", "val_mae"]).copy()
        pool = pool.sort_values("train_mae", ascending=True)
        if args.train_top_pct > 0:
            n_keep = max(1, int(len(pool) * args.train_top_pct / 100.0))
        else:
            n_keep = min(args.train_top_k, len(pool))
        pool = pool.head(n_keep)

        pool_out = os.path.join(RESULT_DIR, f"fold_{fold_id}_train_top_pool.csv")
        pool.to_csv(pool_out, index=False)

        best_val_row = None
        if not pool.empty:
            best_val_row = pool.sort_values("val_mae", ascending=True).iloc[0]

        fold_rows.append({
            "fold": fold_id,
            "train_mae_best_train": float(best_train),
            "val_mae_at_best_train": float(best_val),
            "train_mae_best_val": float(best_val_row["train_mae"]) if best_val_row is not None else np.nan,
            "val_mae_best_val": float(best_val_row["val_mae"]) if best_val_row is not None else np.nan,
        })

        if overall_best["best_train_mae"] is None or best_train < overall_best["best_train_mae"]:
            overall_best["best_train_mae"] = float(best_train)
            overall_best["best_train_val_mae"] = float(best_val)
            overall_best["best_train_params"] = best_cfg

        if best_val_row is not None:
            if overall_best["best_val_mae"] is None or best_val_row["val_mae"] < overall_best["best_val_mae"]:
                overall_best["best_val_mae"] = float(best_val_row["val_mae"])
                overall_best["best_val_train_mae"] = float(best_val_row["train_mae"])
                overall_best["best_val_params"] = {
                    k.replace("config/", ""): best_val_row[k]
                    for k in best_val_row.index
                    if k.startswith("config/")
                }

    summary_df = pd.DataFrame(fold_rows)
    summary_out = os.path.join(RESULT_DIR, "nested_loocv_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    best_out = {
        "nested_loocv_avg_train_mae": float(summary_df["train_mae_best_train"].mean()),
        "nested_loocv_avg_val_mae": float(summary_df["val_mae_at_best_train"].mean()),
        "nested_loocv_avg_train_mae_best_val": float(summary_df["train_mae_best_val"].mean()),
        "nested_loocv_avg_val_mae_best_val": float(summary_df["val_mae_best_val"].mean()),
        "best_train_params": overall_best["best_train_params"],
        "best_train_mae": overall_best["best_train_mae"],
        "best_train_val_mae": overall_best["best_train_val_mae"],
        "best_val_params": overall_best["best_val_params"],
        "best_val_mae": overall_best["best_val_mae"],
        "best_val_train_mae": overall_best["best_val_train_mae"],
    }

    with open(os.path.join(RESULT_DIR, "best_params.json"), "w") as f:
        json.dump(best_out, f, indent=2)

    print(f"Saved nested summary: {summary_out}")
    print(f"Saved best params to: {os.path.join(RESULT_DIR, 'best_params.json')}")

    ray.shutdown()


if __name__ == "__main__":
    main()
