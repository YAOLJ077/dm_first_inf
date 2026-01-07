# -*- coding: utf-8 -*-
"""
Sobol sensitivity analysis for FIRST infection model (ONLY architecture)
========================================================================
Enhancements:
- MAE (absolute error)
- Signed error (early / late bias)
- Failure rate (no FIRST infection event)
- Sobol analysis on all three metrics
- Memory-only simulation
"""

import os
import sys
import json
import copy
from typing import List, Dict, Tuple

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
N_JOBS = min(64, os.cpu_count())
RANDOM_SEED = 42
PENALTY_DAYS = 60.0
SAMPLE_SIZES = [2048]

# ======================================================
# PARAMETER SPACE (UNCHANGED)
# ======================================================
PARAM_RANGES = {
    "mp_tmp__rh_wet_night": (85, 98),
    "sporulation__MP_min": (2, 5),
    "sporulation__Tmin": (8, 15),
    "sporulation__Topt": (18, 24),
    "sporulation__Tmax": (25, 35),
    "dispersal__rain_min": (6.0, 15.0),
    "dispersal__wind_min": (3.0, 12.0),
    "deposition__rain_min": (2.0, 10.0),
    "zoospore_alive__rain": (1.0, 3.0),
    "zoospore_alive__vpd": (1.0, 3.0),
    "infection__infection_condition": (50, 200),
    "incubation__spl_end": (1.0, 1.5)
}

# ======================================================
# LOAD ACTUAL DATA
# ======================================================
def load_actual() -> pd.DataFrame:
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
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear

# ======================================================
# ERROR FUNCTION (EVENT-BASED)
# ======================================================
def calc_error(actual_date, event_df: pd.DataFrame) -> float:
    """
    Signed error in DOY:
    negative → simulated earlier
    positive → simulated later
    """
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
    elif doy_start > doy_actual:
        return doy_start - doy_actual
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
# SOBOL EVALUATION
# ======================================================
def evaluate(
    X: np.ndarray,
    param_names: List[str],
    actual_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    rows = list(actual_df.itertuples(index=False))
    n_samples = len(X)

    def run_one_sample(i):

        cfg = apply_params(config, X[i], param_names)

        signed_errs = []
        abs_errs = []
        n_total = 0
        n_valid = 0

        for row in rows:
            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
            if not os.path.exists(rb_path):
                continue

            n_total += 1

            try:
                rb = load_rb_cached(rb_path)
                cs = CropSeasonFirst(csrb=rb, config=cfg)
                cs.simulate()

                e = calc_error(
                    row.first_infection_date,
                    cs.primary_chain_unique
                )

                if np.isfinite(e):
                    signed_errs.append(e)
                    abs_errs.append(abs(e))
                    n_valid += 1

            except Exception:
                continue

        if n_valid == 0:
            return PENALTY_DAYS, PENALTY_DAYS, 1.0

        return (
            float(np.mean(abs_errs)),                # MAE
            float(np.mean(signed_errs)),             # signed bias
            1.0 - n_valid / max(1, n_total)          # failure rate
        )

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        batch_size=1,
        pre_dispatch="2*n_jobs"
    )(
        delayed(run_one_sample)(i)
        for i in tqdm(range(n_samples), desc="Sobol samples")
    )

    results = np.asarray(results)
    return results[:, 0], results[:, 1], results[:, 2]

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
        Y_mae, Y_signed, Y_fail = evaluate(X, param_names, actual_df)

        for label, Y in zip(
            ["MAE", "Signed", "Failure"],
            [Y_mae, Y_signed, Y_fail]
        ):
            print(f"{label}: mean={np.mean(Y):.2f}, std={np.std(Y):.2f}")

            Si = sobol_analyze.analyze(
                problem,
                Y,
                calc_second_order=False,
                print_to_console=False
            )

            df = pd.DataFrame({
                "Parameter": param_names,
                "S1": Si["S1"],
                "ST": Si["ST"],
            }).sort_values("ST", ascending=False)

            out = os.path.join(
                RESULT_DIR,
                f"sobol_first_inf_ONLY_{label}_N{N}_process_change.xlsx"
            )

            df.to_excel(out, index=False)
            print(f"Saved: {out}")

if __name__ == "__main__":
    main()
