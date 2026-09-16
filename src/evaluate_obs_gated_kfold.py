# -*- coding: utf-8 -*-
# LEGACY / 旧版：不属于当前 FSIM 分析入口。
# Status: legacy_analysis; see config/project_registry.json and config/README.md.
# Retained for historical traceability; current results require separate validation.
"""Legacy fixed-parameter FSIM evaluation.

The former observation-gated branch is not part of the current model API.
Use ``run_fsim_diagnostic.py`` for the current site-year workflow.
"""

import os
import sys
import json
import copy
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

# ======================================================
# PATH SETUP
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
GRID_PARAM_PATH = os.path.join(
    PROJECT_ROOT, "result", "grid", "recommended_params_FIRST_GRID.json"
)

# ======================================================
# IMPORT MODEL
# ======================================================
from model.crop_season import CropSeasonFirst
from model.config import config as BASE_CONFIG

# ======================================================
# GLOBAL SETTINGS
# ======================================================
N_FOLDS = 3
RANDOM_SEED = 42
PENALTY_DAYS = 60.0

# ======================================================
# LOAD DATA
# ======================================================
def load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)

def load_best_params() -> Dict[str, float]:
    with open(GRID_PARAM_PATH) as f:
        return json.load(f)

# ======================================================
# APPLY FIXED PARAMS
# ======================================================
def apply_params(base_cfg: dict, params: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg

# ======================================================
# DOY UTILITY
# ======================================================
def to_doy(dt):
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear

# ======================================================
# ERROR FUNCTION (UNCHANGED)
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
# LOAD RB JSON (LOCAL CACHE)
# ======================================================
_RB_CACHE: Dict[str, dict] = {}

def load_rb_cached(path: str) -> dict:
    if path not in _RB_CACHE:
        with open(path, "r") as f:
            _RB_CACHE[path] = json.load(f)
    return _RB_CACHE[path]

# ======================================================
# MAIN EVALUATION
# ======================================================
def main():

    actual_df = load_actual()
    best_params = load_best_params()

    cfg_fixed = apply_params(BASE_CONFIG, best_params)

    kf = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    fold_results = []

    for fold_i, (_, val_idx) in enumerate(kf.split(actual_df), 1):

        val_df = actual_df.iloc[val_idx].reset_index(drop=True)

        abs_err_baseline = []

        print(f"\n=== Fold {fold_i} ===")

        for row in tqdm(val_df.itertuples(index=False), total=len(val_df)):

            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
            if not os.path.exists(rb_path):
                continue

            rb = load_rb_cached(rb_path)

            # ---------- BASELINE ----------
            cs0 = CropSeasonFirst(csrb=rb, config=cfg_fixed)
            cs0.simulate()

            e0 = calc_error(row.first_infection_date, cs0.primary_chain_unique)
            if np.isfinite(e0):
                abs_err_baseline.append(abs(e0))

        fold_results.append({
            "fold": fold_i,
            "baseline_mae": float(np.mean(abs_err_baseline)) if abs_err_baseline else PENALTY_DAYS,
            "n_baseline": len(abs_err_baseline),
        })

        print(f"Fold {fold_i} | FSIM MAE = {fold_results[-1]['baseline_mae']:.2f}")

    # ==================================================
    # SUMMARY
    # ==================================================
    df = pd.DataFrame(fold_results)
    print("\n=== KFold SUMMARY ===")
    print(df)

    print("\nMean FSIM MAE:", df["baseline_mae"].mean())

    out_path = os.path.join(
        PROJECT_ROOT, "result", "fsim_legacy_kfold_summary.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
