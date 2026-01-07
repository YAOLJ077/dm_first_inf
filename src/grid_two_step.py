# -*- coding: utf-8 -*-
"""
Two-stage calibration for FIRST infection model
================================================
Stage-1: Spore presence likelihood (biological plausibility)
Stage-2: FIRST phenology accuracy (MAE / RMSE)

Author: you + ChatGPT
"""

# ======================================================
# ENV & IMPORTS
# ======================================================
import os
import sys
import json
import copy
import itertools
import numpy as np
import pandas as pd
import ray
from sklearn.model_selection import KFold
from tqdm import tqdm

# ======================================================
# PATH SETUP
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "result", "grid_two_stage_presence")
os.makedirs(OUT_DIR, exist_ok=True)

ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_PATH = os.path.join(DATA_DIR, "Spore_Overall.xlsx")

# ======================================================
# MODEL IMPORT
# ======================================================
from model.crop_season import CropSeasonFirst
from model.config import config as BASE_CONFIG

# ======================================================
# GLOBAL SETTINGS
# ======================================================
RANDOM_SEED = 42
N_FOLDS = 3
N_CPUS = min(128, os.cpu_count() or 32)
BATCH_SIZE = 1000

STAGE1_KEEP_RATIO = 0.30
PENALTY_DAYS = 60.0

PRESENCE_WINDOW_DAYS = 14
ALPHA = 0.25
EPS = 1e-8

# ======================================================
# RAY INIT
# ======================================================
if not ray.is_initialized():
    ray.init(
        num_cpus=N_CPUS,
        include_dashboard=False,
        log_to_driver=False,
        ignore_reinit_error=True,
    )

# ======================================================
# PARAM GRIDS
# ======================================================
STAGE1_PARAM_GRID = {
    "mp_tmp__rh_wet_night": np.linspace(90, 98, 5),
    "sporulation__MP_min": np.linspace(3, 5, 3),
    "sporulation__Tmax": np.linspace(28, 34, 3),
    "dispersal__rain_min": np.linspace(8, 16, 5),
    "dispersal__wind_min": np.linspace(4, 12, 5),
    "deposition__rain_min": np.linspace(4, 10, 4),
}

STAGE2_PARAM_GRID = {
    "zoospore_alive__rain": [2.0],
    "zoospore_alive__vpd": np.linspace(1, 3, 5),
    "infection__infection_condition": np.linspace(120, 200, 5),
    "incubation__spl_end": [1.5],
}

# ======================================================
# DATA LOADERS
# ======================================================
def load_actual():
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)

def load_spore_presence():
    df = pd.read_excel(SPORE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["spore"] = pd.to_numeric(df["spore"], errors="coerce")
    return df[df["spore"] > 0]

# ======================================================
# UTILS
# ======================================================
_RB_CACHE = {}

def load_rb(path):
    if path not in _RB_CACHE:
        with open(path, "r") as f:
            _RB_CACHE[path] = json.load(f)
    return _RB_CACHE[path]

def apply_params(base_cfg, params):
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg

# ======================================================
# PRESENCE LIKELIHOOD
# ======================================================
def presence_likelihood(t_pred, pres_dates):
    if t_pred is None:
        return EPS

    w0 = t_pred - pd.Timedelta(days=PRESENCE_WINDOW_DAYS)
    pres = [d for d in pres_dates if w0 <= d < t_pred]

    if not pres:
        return EPS

    dmin = min((t_pred - d).days for d in pres)
    return float(np.exp(-ALPHA * dmin))

# ======================================================
# STAGE-1 EVALUATION
# ======================================================
def eval_stage1_likelihood(rows, cfg, spore_df):

    Ls = []

    for row in rows:
        key = f"{row['plot']}_{row['variety']}_{row['first_infection_date'].year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue

        try:
            rb = load_rb(rb_path)
            cs = CropSeasonFirst(rb, cfg)
            cs.simulate()

            pc = cs.primary_chain_unique
            if pc is None or pc.empty:
                continue

            t_pred = pc.iloc[0]["oil_spot_start"]

            pres_dates = spore_df[
                (spore_df["plot"] == row["plot"]) &
                (spore_df["year"] == row["first_infection_date"].year)
            ]["date"].tolist()

            Ls.append(presence_likelihood(t_pred, pres_dates))

        except Exception:
            continue

    if not Ls:
        return np.inf

    return -float(np.mean(np.log(np.asarray(Ls) + EPS)))

# ======================================================
# DATASET MAE / RMSE
# ======================================================
def eval_dataset(rows, cfg):

    errs = []

    for row in rows:
        key = f"{row['plot']}_{row['variety']}_{row['first_infection_date'].year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue

        try:
            rb = load_rb(rb_path)
            cs = CropSeasonFirst(rb, cfg)
            cs.simulate()

            pc = cs.primary_chain_unique
            if pc is None or pc.empty:
                continue

            pred = pc.iloc[0]["oil_spot_start"]
            err = (pred - row["first_infection_date"]).days
            errs.append(err)

        except Exception:
            continue

    if not errs:
        return PENALTY_DAYS, PENALTY_DAYS

    arr = np.asarray(errs)
    return float(np.mean(np.abs(arr))), float(np.sqrt(np.mean(arr**2)))

# ======================================================
# RAY TASKS
# ======================================================
@ray.remote
def stage1_task(values, keys, train_rows, spore_df):
    params = dict(zip(keys, values))
    cfg = apply_params(BASE_CONFIG, params)
    loss = eval_stage1_likelihood(train_rows, cfg, spore_df)
    return {"params": params, "stage1_loss": loss}

@ray.remote
def stage2_task(stage1_params, values, train_rows, val_rows):
    params = stage1_params.copy()
    params.update(dict(zip(STAGE2_PARAM_GRID.keys(), values)))
    cfg = apply_params(BASE_CONFIG, params)

    train_mae, train_rmse = eval_dataset(train_rows, cfg)
    val_mae, val_rmse = eval_dataset(val_rows, cfg)

    return {
        "params": params,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
    }

# ======================================================
# GRID SEARCH FUNCTIONS
# ======================================================
def stage1_grid_search(train_df, spore_df):

    keys = list(STAGE1_PARAM_GRID.keys())
    grid = list(itertools.product(*STAGE1_PARAM_GRID.values()))
    train_rows = [r._asdict() for r in train_df.itertuples(index=False)]

    results = []

    for i in range(0, len(grid), BATCH_SIZE):
        futures = [
            stage1_task.remote(vals, keys, train_rows, spore_df)
            for vals in grid[i:i+BATCH_SIZE]
        ]
        results.extend(ray.get(futures))

    df = pd.DataFrame(results)
    n_keep = max(int(len(df) * STAGE1_KEEP_RATIO), 1)
    return df.nsmallest(n_keep, "stage1_loss")

def stage2_grid_search(stage1_df, train_df, val_df):

    train_rows = [r._asdict() for r in train_df.itertuples(index=False)]
    val_rows = [r._asdict() for r in val_df.itertuples(index=False)]
    stage2_grid = list(itertools.product(*STAGE2_PARAM_GRID.values()))

    futures = []

    for _, row in stage1_df.iterrows():
        for vals in stage2_grid:
            futures.append(
                stage2_task.remote(row["params"], vals, train_rows, val_rows)
            )

    results = ray.get(futures)
    return min(results, key=lambda x: x["val_mae"])

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)

    df = load_actual()
    spore_df = load_spore_presence()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    summary = []

    for i, (tr, va) in enumerate(kf.split(df), 1):

        train_df, val_df = df.iloc[tr], df.iloc[va]

        stage1_best = stage1_grid_search(train_df, spore_df)
        best = stage2_grid_search(stage1_best, train_df, val_df)

        summary.append({
            "fold": i,
            "train_mae": best["train_mae"],
            "val_mae": best["val_mae"],
            "train_rmse": best["train_rmse"],
            "val_rmse": best["val_rmse"],
        })

    pd.DataFrame(summary).to_csv(
        os.path.join(OUT_DIR, "kfold_summary.csv"),
        index=False
    )

    ray.shutdown()
