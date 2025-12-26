# -*- coding: utf-8 -*-
"""
Sobol sensitivity analysis for FIRST infection model (subtropical DM)
====================================================================
- Parallel over Sobol samples
- JSON loaded once per process (process-local cache)
- Memory-only simulation
- Error defined on FIRST effective oil-spot window (DOY-based)
"""

import os
import sys
import json
import copy
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

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
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "sobol")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season import CropSeasonFirst
from model.config import config

# ======================================================
# GLOBAL SETTINGS
# ======================================================
N_JOBS = min(60, os.cpu_count())
RANDOM_SEED = 42
PENALTY_DAYS = 30.0      # 30 days penalty (DOY scale)
SAMPLE_SIZES = [64]

# ======================================================
# PARAMETER SPACE
# ======================================================
PARAM_RANGES = {
    "mp_tmp__rh_wet_night": (75, 95),
    "sporulation__MP_min": (2, 8),
    "sporulation__Tmin": (5, 15),
    "sporulation__Topt": (15, 25),
    "sporulation__Tmax": (25, 35),
    "dispersal__rain_min": (1.0, 6.0),
    "dispersal__wind_min": (1.0, 6.0),
    "deposition__rain_min": (1.0, 6.0),
    "zoospore_alive__rain": (1.0, 6.0),
    "zoospore_alive__vpd": (1.0, 6.0),
    "infection__infection_condition": (40, 120),
    "incubation__spl_end": (0.8, 1.2),
}

# ======================================================
# LOAD ACTUAL DATA
# ======================================================
def load_actual() -> pd.DataFrame:
    if not os.path.exists(ACTUAL_DATA_PATH):
        raise FileNotFoundError(ACTUAL_DATA_PATH)

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
# DOY UTILITY
# ======================================================
def to_doy(dt) -> int:
    """
    Convert datetime-like to day-of-year (DOY).
    """
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear

# ======================================================
# ERROR FUNCTION (DOY-BASED)
# ======================================================
def calc_error(actual_date, cohort_df) -> float:
    df = cohort_df[
        (cohort_df["infection_effective"] == True) &
        (~cohort_df["oil_spot_start"].isna()) &
        (~cohort_df["oil_spot_end"].isna())
    ]

    if df.empty:
        return np.nan

    first = df.iloc[0]

    doy_actual = to_doy(actual_date)
    doy_start = to_doy(first["oil_spot_start"])
    doy_end = to_doy(first["oil_spot_end"])

    if doy_actual is None or doy_start is None or doy_end is None:
        return np.nan

    # simulated too early
    if doy_end < doy_actual:
        return doy_end - doy_actual
    # simulated too late
    elif doy_start > doy_actual:
        return doy_start - doy_actual
    # observed date inside window
    else:
        return 0.0

# ======================================================
# ---- PROCESS-LOCAL JSON CACHE ----
# ======================================================
_RB_CACHE: Dict[str, dict] = {}

def load_rb_cached(rb_path: str) -> dict:
    if rb_path not in _RB_CACHE:
        with open(rb_path, "r") as f:
            _RB_CACHE[rb_path] = json.load(f)
    return _RB_CACHE[rb_path]

# ======================================================
# SOBOL EVALUATION (PARALLEL OVER SAMPLES)
# ======================================================
def evaluate(X: np.ndarray, param_names: List[str], actual_df: pd.DataFrame) -> np.ndarray:
    rows = list(actual_df.itertuples(index=False))
    n_samples = len(X)

    def run_one_sample(i):
        cfg = apply_params(config, X[i], param_names)
        errs = []

        for row in rows:
            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")

            if not os.path.exists(rb_path):
                continue

            try:
                rb = load_rb_cached(rb_path)
                cs = CropSeasonFirst(csrb=rb, config=cfg)
                cs.simulate(base_output_dir=None)
                e = calc_error(row.first_infection_date, cs.cohort_results)
                if np.isfinite(e):
                    errs.append(abs(e))
            except Exception:
                continue

        if len(errs) == 0:
            return PENALTY_DAYS

        return float(np.mean(errs))

    Y = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        batch_size=1,
        pre_dispatch="2*n_jobs"
    )(
        delayed(run_one_sample)(i)
        for i in tqdm(range(n_samples), desc="Sobol samples")
    )

    return np.asarray(Y, dtype=float)

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

        print(
            "Y summary:",
            "NaN =", np.isnan(Y).sum(),
            "Inf =", np.isinf(Y).sum(),
            "Std =", np.std(Y)
        )

        Si = sobol_analyze.analyze(
            problem,
            Y,
            calc_second_order=False,
            print_to_console=True
        )

        df = pd.DataFrame({
            "Parameter": param_names,
            "S1": Si["S1"],
            "ST": Si["ST"],
        }).sort_values("ST", ascending=False)

        out = os.path.join(RESULT_DIR, f"sobol_first_inf_DOY_N{N}.xlsx")
        df.to_excel(out, index=False)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
