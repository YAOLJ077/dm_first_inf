# -*- coding: utf-8 -*-
"""
Ray + KFold Grid calibration for FIRST infection model (ONLY architecture)
=========================================================================
Design principles:
- DOY-level event-based error (same as Sobol)
- Optimization target: MAE (absolute error)
- RMSE reported for stability assessment
- Ray parallelism (primary mature pattern)
- One Ray task = one parameter combination (train + val together)
- KFold cross-validation
- Worker-local JSON cache for fast IO
"""

import os

# 完全关闭 Ray dashboard 和 metrics
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


import sys
import json
import copy
import itertools
import collections
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
from sklearn.model_selection import KFold

# ======================================================
# PATH SETUP
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
OUT_DIR = os.path.join(PROJECT_ROOT, "result", "grid")
os.makedirs(OUT_DIR, exist_ok=True)

from model.crop_season_only import CropSeasonFirst
from model.config import config as BASE_CONFIG

# ======================================================
# GLOBAL SETTINGS
# ======================================================
RANDOM_SEED = 42
PENALTY_DAYS = 30.0

N_CPUS = min(128, os.cpu_count() or 64)
BATCH_SIZE = 2000

SELECTION_MODE = "topk_then_val"
TOP_PERCENT = 0.01

N_FOLDS = 3

# ======================================================
# INIT RAY
# ======================================================
if not ray.is_initialized():
    ray.init(
    num_cpus=N_CPUS,
    include_dashboard=False,
    log_to_driver=False,
    ignore_reinit_error=True,
)
# ======================================================
# PARAMETER GRID
# Single value  -> fixed
# Multiple values -> tuned
# ======================================================
PARAM_GRID = {
    "mp_tmp__rh_wet_night": np.linspace(80, 95, 4),
    "sporulation__MP_min": np.linspace(3, 6, 4),
    "sporulation__Tmin": np.linspace(8, 14, 4),
    "sporulation__Topt": [20],
    "sporulation__Tmax": [30],
    "dispersal__rain_min": np.linspace(2.0, 6.0, 3),
    "dispersal__wind_min": np.linspace(2.0, 6.0, 3),
    "deposition__rain_min": np.linspace(2.0, 10.0, 5),
    "zoospore_alive__rain": [2.0],
    "zoospore_alive__vpd": np.linspace(2.0, 4.0, 2),
    "infection__infection_condition": np.linspace(40, 80, 4),
    "incubation__spl_end": np.linspace(0.8, 1.2, 4)
}

# PARAM_GRID = {
#     "mp_tmp__rh_wet_night": [90],
#     "sporulation__MP_min": np.linspace(2, 8, 2),
#     "sporulation__Tmin": np.linspace(5, 15, 2),
#     "sporulation__Topt": np.linspace(15, 25, 2),
#     "sporulation__Tmax": np.linspace(25, 35, 2),
#     "dispersal__rain_min": np.linspace(1.0, 6.0, 2),
#     "dispersal__wind_min": np.linspace(1.0, 6.0, 2),
#     "deposition__rain_min": np.linspace(1.0, 6.0, 2),
#     "zoospore_alive__rain": np.linspace(1.0, 6.0, 2),
#     "zoospore_alive__vpd": np.linspace(1.0, 6.0, 2),
#     "infection__infection_condition": np.linspace(40, 120, 2),
#     "incubation__spl_end": [1.0]
# }

# ======================================================
# LOAD ACTUAL DATA
# ======================================================
def load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)

# ======================================================
# DOY UTILITY
# ======================================================
def to_doy(dt):
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear

# ======================================================
# ERROR FUNCTION (DOY, event-based)
# ======================================================
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
        return float(doy_end - doy_actual)
    elif doy_start > doy_actual:
        return float(doy_start - doy_actual)
    else:
        return 0.0

# ======================================================
# APPLY PARAMS
# ======================================================
def apply_params(base_cfg: dict, params: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg

# ======================================================
# WORKER-LOCAL JSON CACHE
# ======================================================
_RB_CACHE: Dict[str, dict] = {}

def load_rb_cached(path: str) -> dict:
    if path not in _RB_CACHE:
        with open(path, "r") as f:
            _RB_CACHE[path] = json.load(f)
    return _RB_CACHE[path]

# ======================================================
# EVALUATE DATASET
# ======================================================
def eval_dataset(rows: List[Dict[str, Any]], cfg: dict) -> Tuple[float, float, List[float]]:
    signed_errs = []

    for row in rows:
        key = f"{row['plot']}_{row['variety']}_{row['first_infection_date'].year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue

        try:
            rb = load_rb_cached(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()

            e = calc_error(row["first_infection_date"], cs.primary_chain_unique)
            if np.isfinite(e):
                signed_errs.append(float(e))

        except Exception:
            continue

    if not signed_errs:
        return PENALTY_DAYS, PENALTY_DAYS, []

    arr = np.asarray(signed_errs)
    mae = float(np.mean(np.abs(arr)))
    rmse = float(np.sqrt(np.mean(arr ** 2)))
    return mae, rmse, signed_errs

# ======================================================
# RAY TASK: ONE PARAM COMBINATION
# ======================================================
@ray.remote
def simulate_one_combination_ray(
    values: Tuple[float, ...],
    keys: List[str],
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
):

    params = dict(zip(keys, values))
    cfg = apply_params(BASE_CONFIG, params)

    train_mae, train_rmse, train_errs = eval_dataset(train_rows, cfg)
    val_mae, val_rmse, val_errs = eval_dataset(val_rows, cfg)

    return {
        "params": params,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "train_errors": train_errs,
        "val_errors": val_errs,
    }

# ======================================================
# SELECTION LOGIC (PRIMARY STYLE)
# ======================================================
def pick_best_row(df: pd.DataFrame) -> pd.Series:
    if SELECTION_MODE == "train_only":
        return df.nsmallest(1, "train_mae").iloc[0]
    elif SELECTION_MODE == "val_only":
        return df.nsmallest(1, "val_mae").iloc[0]
    elif SELECTION_MODE == "topk_then_val":
        k = max(int(len(df) * TOP_PERCENT), 1)
        top = df.nsmallest(k, "train_mae")
        return top.nsmallest(1, "val_mae").iloc[0]
    else:
        raise ValueError("Invalid SELECTION_MODE")

# ======================================================
# GRID SEARCH (ONE FOLD)
# ======================================================
def grid_search_one_fold(train_df, val_df, run_id):

    keys = list(PARAM_GRID.keys())
    grid = list(itertools.product(*PARAM_GRID.values()))
    print(f"🔍 {run_id} | combinations = {len(grid)}")

    train_rows = [r._asdict() for r in train_df.itertuples(index=False)]
    val_rows = [r._asdict() for r in val_df.itertuples(index=False)]

    all_results = []

    for i in range(0, len(grid), BATCH_SIZE):
        batch = grid[i:i + BATCH_SIZE]

        futures = [
            simulate_one_combination_ray.remote(vals, keys, train_rows, val_rows)
            for vals in batch
        ]

        with tqdm(total=len(futures), desc=f"{run_id} batch {i//BATCH_SIZE+1}") as pbar:
            while futures:
                done, futures = ray.wait(futures, num_returns=1)
                all_results.append(ray.get(done[0]))
                pbar.update(1)

    df = pd.DataFrame([
        {
            **res["params"],
            "train_mae": res["train_mae"],
            "train_rmse": res["train_rmse"],
            "val_mae": res["val_mae"],
            "val_rmse": res["val_rmse"],
        }
        for res in all_results
    ])

    df.to_csv(os.path.join(OUT_DIR, f"grid_results_{run_id}.csv"), index=False)

    best = pick_best_row(df)

    out_json = os.path.join(OUT_DIR, f"best_params_{run_id}.json")
    with open(out_json, "w") as f:
        json.dump({
            "best_params": {k: float(best[k]) for k in keys},
            "train_mae": float(best["train_mae"]),
            "train_rmse": float(best["train_rmse"]),
            "val_mae": float(best["val_mae"]),
            "val_rmse": float(best["val_rmse"]),
        }, f, indent=2)

    print(f"\n✅ Best ({run_id})")
    print(best)

    return best.to_dict()

# ======================================================
# KFold SUMMARY
# ======================================================
def summarize_kfold(run_id, n_folds):

    rows, params_list = [], []

    for i in range(1, n_folds + 1):
        path = os.path.join(OUT_DIR, f"best_params_{run_id}_fold{i}.json")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            obj = json.load(f)

        rows.append({
            "fold": i,
            "train_mae": obj["train_mae"],
            "val_mae": obj["val_mae"],
            "train_rmse": obj["train_rmse"],
            "val_rmse": obj["val_rmse"],
        })
        params_list.append(obj["best_params"])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"kfold_detailed_{run_id}.csv"), index=False)

    summary = {
        "train_mae_mean": df["train_mae"].mean(),
        "train_mae_std": df["train_mae"].std(),
        "val_mae_mean": df["val_mae"].mean(),
        "val_mae_std": df["val_mae"].std(),
        "train_rmse_mean": df["train_rmse"].mean(),
        "train_rmse_std": df["train_rmse"].std(),
        "val_rmse_mean": df["val_rmse"].mean(),
        "val_rmse_std": df["val_rmse"].std(),
        "n_folds": len(df),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(OUT_DIR, f"kfold_summary_{run_id}.csv"),
        index=False
    )

    agg = {}
    for k in PARAM_GRID.keys():
        vals = [p[k] for p in params_list]
        agg[k] = float(np.median(vals))

    with open(os.path.join(OUT_DIR, f"recommended_params_{run_id}.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print("\n📊 FINAL K-FOLD RESULT")
    print(summary)
    print("Recommended params:")
    print(agg)

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)

    df = load_actual()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    run_id = "FIRST_GRID"

    for i, (tr, va) in enumerate(kf.split(df), 1):
        grid_search_one_fold(
            df.iloc[tr],
            df.iloc[va],
            run_id=f"{run_id}_fold{i}"
        )

    summarize_kfold(run_id, n_folds=N_FOLDS)

    ray.shutdown()
