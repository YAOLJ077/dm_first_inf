# -*- coding: utf-8 -*-
"""
Sobol sensitivity analysis for FIRST infection model (subtropical DM)
====================================================================
- Memory-only simulation (no Excel output)
- Error defined on FIRST effective oil-spot window
- joblib parallel
- SALib Sobol (full structure)
"""


import json
import copy
import warnings
from typing import Dict, List
import openpyxl

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

import os
import sys
# ======================================================
# Make project root importable
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))      # .../dm_first_inf/src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))  # .../dm_first_inf

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# PATH SETUP (relative)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "sobol")

os.makedirs(RESULT_DIR, exist_ok=True)


from model.crop_season import CropSeasonFirst
from model.config import config



# ======================================================
# GLOBAL SETTINGS
# ======================================================
N_JOBS = 128
RANDOM_SEED = 42
PENALTY_HOURS = 720.0     # 30 days
SAMPLE_SIZES = [256, 512]

# ======================================================
# PARAMETER SPACEpip
# ======================================================
PARAM_RANGES = {
    "mp_tmp__rh_wet_night": (75, 95),
    "sporulation__MP_min": (2, 5),
    "sporulation__Tmin": (8, 12),
    "sporulation__Topt": (18, 22),
    "sporulation__Tmax": (28, 32),
    "dispersal__rain_min": (1.0, 6.0),
    "dispersal__wind_min": (1.0, 6.0),
    "deposition__rain_min": (1.0, 6.0),
    "zoospore_alive__rain": (1.0, 6.0),
    "zoospore_alive__vpd": (1.0, 4.0),
    "infection__infection_condition": (40, 100),
    "incubation__spl_end": (0.8, 1.2),
}

# ======================================================
# LOAD ACTUAL DATA
# ======================================================
def load_actual():
    if not os.path.exists(ACTUAL_DATA_PATH):
        raise FileNotFoundError(f"Actual data not found: {ACTUAL_DATA_PATH}")

    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df

# ======================================================
# APPLY PARAMS
# ======================================================
def apply_params(base_cfg: dict, values: np.ndarray, names: List[str]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for v, k in zip(values, names):
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg

# ======================================================
# ERROR FUNCTION
# ======================================================
def calc_error(actual_date, cohort_df):
    df = cohort_df[
        (cohort_df["infection_effective"] == True) &
        (~cohort_df["oil_spot_start"].isna()) &
        (~cohort_df["oil_spot_end"].isna())
    ]

    if df.empty:
        return np.nan

    first = df.iloc[0]
    ws = pd.to_datetime(first["oil_spot_start"])
    we = pd.to_datetime(first["oil_spot_end"])

    if we < actual_date:
        return (we - actual_date).total_seconds() / 3600.0
    elif ws > actual_date:
        return (ws - actual_date).total_seconds() / 3600.0
    else:
        return 0.0

# ======================================================
# SINGLE SIMULATION
# ======================================================
def simulate_one(row, cfg):
    key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
    rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")

    if not os.path.exists(rb_path):
        return np.nan

    try:
        with open(rb_path, "r") as f:
            rb = json.load(f)

        cs = CropSeasonFirst(csrb=rb, config=cfg)
        cs.simulate(base_output_dir=None)
        return calc_error(row.first_infection_date, cs.cohort_results)

    except Exception:
        return np.nan

# ======================================================
# SOBOL EVALUATION
# ======================================================
def evaluate(X, param_names, actual_df):
    rows = list(actual_df.itertuples(index=False))

    def run_one(i):
        cfg = apply_params(config, X[i], param_names)
        errs = [simulate_one(r, cfg) for r in rows]
        mae = np.nanmean(np.abs(errs))
        if np.isnan(mae) or np.isinf(mae):
            mae = PENALTY_HOURS
        return mae

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Y = Parallel(n_jobs=N_JOBS)(
            delayed(run_one)(i)
            for i in tqdm(range(len(X)), desc="Sobol simulations")
        )

    return np.array(Y)

# ======================================================
# MAIN
# ======================================================
def main():
    np.random.seed(RANDOM_SEED)
    actual_df = load_actual()

    param_names = list(PARAM_RANGES.keys())
    bounds = [PARAM_RANGES[k] for k in param_names]

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds
    }

    for N in SAMPLE_SIZES:
        print(f"\n=== Sobol N={N} ===")
        X = sobol_sample.sample(problem, N, calc_second_order=False)
        Y = evaluate(X, param_names, actual_df)

        Y_std = (Y - np.mean(Y)) / np.std(Y)

        Si = sobol_analyze.analyze(
            problem, Y_std, calc_second_order=False, print_to_console=True
        )

        df = pd.DataFrame({
            "Parameter": param_names,
            "S1": Si["S1"],
            "ST": Si["ST"],
        }).sort_values("ST", ascending=False)

        out = os.path.join(RESULT_DIR, f"sobol_first_inf_N{N}.xlsx")
        df.to_excel(out, index=False)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
