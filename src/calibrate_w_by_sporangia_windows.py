# -*- coding: utf-8 -*-
"""
Calibration of observational weight w
based on sporangia window consistency
=====================================
- tqdm progress monitoring
- optional parallel execution
- no infection simulation
"""

import os
import sys
import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# ======================================================
# PATH SETUP
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORANGIA_OBS_PATH = os.path.join(DATA_DIR, "sporangia_obs.xlsx")
GRID_PARAM_PATH = os.path.join(
    PROJECT_ROOT, "result", "grid", "recommended_params_FIRST_GRID.json"
)

# ======================================================
# IMPORT MODEL
# ======================================================
from model.crop_season import CropSeasonFirst
from model.config import config as BASE_CONFIG

# ======================================================
# SETTINGS
# ======================================================
W_CANDIDATES = np.linspace(0.05, 0.6, 12)
ALPHA = 0.7

USE_PARALLEL = True
N_JOBS = min(8, os.cpu_count() or 1)

# ======================================================
# LOADERS
# ======================================================
def load_actual():
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df

def load_sporangia_obs():
    df = pd.read_excel(SPORANGIA_OBS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def load_best_params():
    with open(GRID_PARAM_PATH) as f:
        return json.load(f)

def apply_params(base_cfg, params):
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg

# ======================================================
# BUILD OBS WINDOW
# ======================================================
def build_obs_window(obs_df, place, year, start, end):
    df = obs_df[
        (obs_df["Place"] == place)
        & (obs_df["Year"] == year)
        & (obs_df["Date"] >= start)
        & (obs_df["Date"] <= end)
        & (obs_df["Spore"] > 0)
    ]
    return set(df["Date"].dt.normalize())

# ======================================================
# PROCESS ONE SITE-YEAR
# ======================================================
def process_one_case(row, cfg_fixed, sp_obs_df, w):

    key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
    rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
    if not os.path.exists(rb_path):
        return 0, 0, 0

    with open(rb_path) as f:
        rb = json.load(f)

    cs = CropSeasonFirst(csrb=rb, config=cfg_fixed)
    cs.simulate(use_obs_gate=False)

    df = cs.dispersal_cohorts
    if df is None or df.empty:
        return 0, 0, 0

    obs_window = build_obs_window(
        sp_obs_df,
        cs.place,
        cs.year,
        cs.bbch_53_datetime,
        row.first_infection_date,
    )

    hits, false_pos, total = 0, 0, 0

    for _, c in df.iterrows():
        total += 1
        d = c.dispersal_datetime.normalize()

        if d in obs_window:
            hits += 1
        else:
            false_pos += w

    return hits, false_pos, total

# ======================================================
# MAIN
# ======================================================
def main():

    actual_df = load_actual()
    sp_obs_df = load_sporangia_obs()
    best_params = load_best_params()
    cfg_fixed = apply_params(BASE_CONFIG, best_params)

    records = []

    for w in tqdm(W_CANDIDATES, desc="Calibrating w"):

        if USE_PARALLEL:
            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_case)(
                    row, cfg_fixed, sp_obs_df, w
                )
                for row in actual_df.itertuples(index=False)
            )
        else:
            results = [
                process_one_case(row, cfg_fixed, sp_obs_df, w)
                for row in actual_df.itertuples(index=False)
            ]

        hits = sum(r[0] for r in results)
        false_pos = sum(r[1] for r in results)
        total = sum(r[2] for r in results)

        if total == 0:
            continue

        hit_rate = hits / total
        false_rate = false_pos / total
        score = hit_rate - ALPHA * false_rate

        records.append({
            "w": w,
            "hit_rate": hit_rate,
            "false_rate": false_rate,
            "score": score,
        })

    df = pd.DataFrame(records).sort_values("score", ascending=False)

    out = os.path.join(PROJECT_ROOT, "result", "w_calibration_by_windows.csv")
    df.to_csv(out, index=False)

    print("\n=== w calibration result (top) ===")
    print(df.head(10))
    print(f"\nSaved to: {out}")

    print("\nRecommended w (robust choice):")
    print(df.iloc[0]["w"])


if __name__ == "__main__":
    main()
