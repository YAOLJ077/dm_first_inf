# -*- coding: utf-8 -*-
"""
Two-stage Ray + KFold Grid calibration for FIRST infection model
===============================================================
Stage-1: Sporangia window consistency filter (process constraint)
Stage-2: MAE minimization (final objective)

Author: you + ChatGPT
"""

import os

# ---------- Disable Ray dashboard ----------
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import sys
import json
import copy
import itertools
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
OUT_DIR = os.path.join(PROJECT_ROOT, "result", "grid_two_stage")
os.makedirs(OUT_DIR, exist_ok=True)

ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_WINDOW_PATH = os.path.join(DATA_DIR, "spore_observation_windows_14d.csv")

from model.crop_season_only import CropSeasonFirst
from model.config import config as BASE_CONFIG

# ======================================================
# GLOBAL SETTINGS
# ======================================================
RANDOM_SEED = 42
N_FOLDS = 3
N_CPUS = min(128, os.cpu_count() or 64)
BATCH_SIZE = 2000

TOP_PERCENT = 0.01              # stage-2 selection
STAGE1_THRESHOLD = 1.0          # ⭐ 核心调参点
FP_WEIGHT = 0.2                 # false positive penalty
PENALTY_DAYS = 30.0

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
    "incubation__spl_end": np.linspace(0.8, 1.2, 4),
}

# ======================================================
# LOAD DATA
# ======================================================
def load_actual():
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)

def load_spore_windows():
    df = pd.read_csv(
        SPORE_WINDOW_PATH,
        parse_dates=["window_start", "window_end"]
    )
    return df

# ======================================================
# UTILITIES
# ======================================================
def to_doy(dt):
    return pd.to_datetime(dt).dayofyear if pd.notna(dt) else None

def calc_error(actual_date, event_df):
    if event_df is None or event_df.empty:
        return np.nan

    first = event_df.iloc[0]
    a = to_doy(actual_date)
    s = to_doy(first["oil_spot_start"])
    e = to_doy(first["oil_spot_end"])

    if a is None or s is None or e is None:
        return np.nan

    if e < a:
        return e - a
    elif s > a:
        return s - a
    else:
        return 0.0

def apply_params(base_cfg, params):
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg

# ======================================================
# WORKER-LOCAL JSON CACHE
# ======================================================
_RB_CACHE = {}

def load_rb_cached(path):
    if path not in _RB_CACHE:
        with open(path, "r") as f:
            _RB_CACHE[path] = json.load(f)
    return _RB_CACHE[path]

# ======================================================
# STAGE-1: SPORANGIA WINDOW MATCH
# ======================================================
def stage1_spore_loss(cs, spore_df, plot, year):
    obs = spore_df[
        (spore_df["plot"] == plot) &
        (spore_df["year"] == year)
    ]
    if obs.empty or cs.detached_survival is None:
        return np.inf

    model_windows = [
        (r["dispersal_datetime"], r["sporangia_alive_end"])
        for _, r in cs.detached_survival.iterrows()
    ]

    fn, fp = 0, 0
    for _, o in obs.iterrows():
        covered = any(
            not (o["window_end"] < ms or o["window_start"] > me)
            for ms, me in model_windows
        )
        if o["label"] == 1 and not covered:
            fn += 1
        elif o["label"] == 0 and covered:
            fp += 1

    return fn + FP_WEIGHT * fp

# ======================================================
# EVALUATE DATASET (MAE)
# ======================================================
def eval_dataset(rows, cfg):
    errs = []
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
                errs.append(e)
        except Exception:
            continue

    if not errs:
        return PENALTY_DAYS, PENALTY_DAYS

    arr = np.asarray(errs)
    return float(np.mean(np.abs(arr))), float(np.sqrt(np.mean(arr**2)))

# ======================================================
# RAY TASK
# ======================================================
@ray.remote
def simulate_one_combination_ray(
    values,
    keys,
    train_rows,
    val_rows,
    spore_df,
):
    params = dict(zip(keys, values))
    cfg = apply_params(BASE_CONFIG, params)

    # ---------- Stage-1 ----------
    stage1_losses = []
    for row in train_rows:
        rb_path = os.path.join(
            DATA_DIR,
            f"{row['plot']}_{row['variety']}_{row['first_infection_date'].year}_grape_rb.json"
        )
        if not os.path.exists(rb_path):
            continue
        rb = load_rb_cached(rb_path)
        cs = CropSeasonFirst(csrb=rb, config=cfg)
        cs.simulate()
        l1 = stage1_spore_loss(
            cs,
            spore_df,
            row["plot"],
            row["first_infection_date"].year
        )
        stage1_losses.append(l1)

    if not stage1_losses or np.mean(stage1_losses) > STAGE1_THRESHOLD:
        return None

    # ---------- Stage-2 ----------
    train_mae, train_rmse = eval_dataset(train_rows, cfg)
    val_mae, val_rmse = eval_dataset(val_rows, cfg)

    return {
        "params": params,
        "stage1_loss": float(np.mean(stage1_losses)),
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
    }

# ======================================================
# GRID SEARCH ONE FOLD
# ======================================================
def grid_search_one_fold(train_df, val_df, spore_df, run_id):

    keys = list(PARAM_GRID.keys())
    grid = list(itertools.product(*PARAM_GRID.values()))
    print(f"🔍 {run_id} | combinations = {len(grid)}")

    train_rows = [r._asdict() for r in train_df.itertuples(index=False)]
    val_rows = [r._asdict() for r in val_df.itertuples(index=False)]

    results = []

    for i in range(0, len(grid), BATCH_SIZE):
        batch = grid[i:i + BATCH_SIZE]
        futures = [
            simulate_one_combination_ray.remote(
                vals, keys, train_rows, val_rows, spore_df
            )
            for vals in batch
        ]
        for f in tqdm(ray.get(futures), desc=f"{run_id} batch {i//BATCH_SIZE+1}"):
            if f is not None:
                results.append(f)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, f"grid_results_{run_id}.csv"), index=False)

    k = max(int(len(df) * TOP_PERCENT), 1)
    best = df.nsmallest(k, "train_mae").nsmallest(1, "val_mae").iloc[0]

    with open(os.path.join(OUT_DIR, f"best_params_{run_id}.json"), "w") as f:
        json.dump(best.to_dict(), f, indent=2)

    print(f"\n✅ Best ({run_id})")
    print(best)

    return best

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)

    df = load_actual()
    spore_df = load_spore_windows()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    run_id = "FIRST_GRID_TWO_STAGE"

    for i, (tr, va) in enumerate(kf.split(df), 1):
        grid_search_one_fold(
            df.iloc[tr],
            df.iloc[va],
            spore_df,
            run_id=f"{run_id}_fold{i}"
        )

    ray.shutdown()
