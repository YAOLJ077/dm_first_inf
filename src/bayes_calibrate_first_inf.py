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

import numpy as np
import pandas as pd

import ray
from ray import tune
from ray.tune import CLIReporter, Callback
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
    def __init__(self, total: int, interval_s: float = 5.0):
        self.total = total
        self.interval_s = interval_s
        self._last_ts = 0.0

    def on_trial_result(self, iteration, trials, trial, result, **info):
        done = sum(t.status == "TERMINATED" for t in trials)
        now = time.time()
        if now - self._last_ts >= self.interval_s:
            self._last_ts = now
            print(f"Progress: {done}/{self.total} trials completed", flush=True)

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


def loocv_mae(cfg: dict, rows: List[pd.Series]) -> float:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=6000)
    parser.add_argument("--n-startup", type=int, default=600)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "result", "ray_results")
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    args = parser.parse_args()

    ranges = build_param_ranges()
    _GLOBAL_RANGES.update(ranges)
    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))

    sampler = TPESampler(
        seed=args.seed,
        multivariate=True,
        n_startup_trials=args.n_startup
    )

    def trainable(cfg):
        model_cfg = apply_params(config, cfg)
        mae = loocv_mae(model_cfg, rows)
        tune.report({"mae": mae})

    optuna_storage = InMemoryStorage()
    optuna_search = OptunaSearch(
        space=define_search_space,
        sampler=sampler,
        metric="mae",
        mode="min",
        storage=optuna_storage,
        study_name="first_inf_mae_loocv"
    )

    reporter = CLIReporter(metric_columns=["mae"])
    verbose = 1
    log_to_file = False
    callbacks = [ProgressOnlyCallback(args.n_trials, args.progress_interval)]
    if args.quiet:
        os.environ.pop("AIR_VERBOSITY", None)
        reporter = None
        verbose = 0
        log_to_file = True

    ray.init(num_cpus=args.n_jobs, include_dashboard=False)

    analysis = tune.run(
        trainable,
        name="first_inf_mae_loocv",
        num_samples=args.n_trials,
        search_alg=optuna_search,
        config={},
        resources_per_trial={"cpu": 1},
        progress_reporter=reporter,
        storage_path=args.storage_path,
        trial_dirname_creator=short_trial_dirname,
        checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
        callbacks=callbacks,
        log_to_file=log_to_file,
        verbose=verbose
    )

    best_trial = analysis.get_best_trial(metric="mae", mode="min", scope="all")
    best_cfg = best_trial.config
    best_value = best_trial.last_result["mae"]

    best_out = {
        "value": float(best_value),
        "params": best_cfg
    }

    with open(os.path.join(RESULT_DIR, "best_params.json"), "w") as f:
        json.dump(best_out, f, indent=2)

    df_trials = analysis.results_df
    df_trials.to_csv(os.path.join(RESULT_DIR, "trials.csv"), index=False)

    print(f"Best MAE (LOOCV): {best_out['value']:.2f}")
    print(f"Saved best params to: {os.path.join(RESULT_DIR, 'best_params.json')}")

    ray.shutdown()


if __name__ == "__main__":
    main()
