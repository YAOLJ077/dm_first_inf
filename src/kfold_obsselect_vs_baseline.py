# -*- coding: utf-8 -*-
"""
KFold evaluation:
Original baseline (early-break)
vs
Multi-candidate baseline (earliest selection)
================================================
Purpose:
- Verify behavioral equivalence
- Ensure MAE comparability before further model enhancement
"""

import os
import sys
import json
import copy
import numpy as np
import pandas as pd
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
BEST_PARAM_PATH = os.path.join(
    PROJECT_ROOT, "result", "grid", "recommended_params_FIRST_GRID.json"
)

# ======================================================
# IMPORT MODELS
# ======================================================
from model.crop_season import CropSeasonFirst                  # original baseline
from model.crop_season_first_multicand import CropSeasonFirstMultiCand
from model.config import config as BASE_CONFIG

# ======================================================
# UTILS
# ======================================================
def load_actual():
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)

def load_best_params():
    with open(BEST_PARAM_PATH) as f:
        return json.load(f)

def apply_params(base_cfg, params):
    cfg = copy.deepcopy(base_cfg)
    for k, v in params.items():
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg

def to_doy(dt):
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear

def calc_error(actual_date, event_df):
    """
    Event-based DOY error:
    negative -> simulated earlier
    positive -> simulated later
    """
    if event_df is None or event_df.empty:
        return np.nan

    r = event_df.iloc[0]
    doy_a = to_doy(actual_date)
    doy_s = to_doy(r["oil_spot_start"])
    doy_e = to_doy(r["oil_spot_end"])

    if doy_a is None or doy_s is None or doy_e is None:
        return np.nan

    if doy_e < doy_a:
        return doy_e - doy_a
    elif doy_s > doy_a:
        return doy_s - doy_a
    else:
        return 0.0

# ======================================================
# MAIN
# ======================================================
def main():

    df = load_actual()
    best_params = load_best_params()
    cfg = apply_params(BASE_CONFIG, best_params)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    rows = []

    for fold, (_, va) in enumerate(kf.split(df), 1):

        base_errs = []
        multi_errs = []

        print(f"\n=== Fold {fold} ===")

        for row in tqdm(
            df.iloc[va].itertuples(index=False),
            total=len(va),
            desc=f"Fold {fold}"
        ):
            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
            if not os.path.exists(rb_path):
                continue

            with open(rb_path) as f:
                rb = json.load(f)

            # ----- original baseline (early break) -----
            cs0 = CropSeasonFirst(rb, cfg)
            cs0.simulate()
            e0 = calc_error(row.first_infection_date, cs0.primary_chain_unique)
            if np.isfinite(e0):
                base_errs.append(abs(e0))

            # ----- multi-candidate baseline -----
            cs1 = CropSeasonFirstMultiCand(rb, cfg)
            cs1.simulate()
            e1 = calc_error(row.first_infection_date, cs1.primary_chain_unique)
            if np.isfinite(e1):
                multi_errs.append(abs(e1))

        rows.append({
            "fold": fold,
            "baseline_mae": np.mean(base_errs) if base_errs else np.nan,
            "multicand_mae": np.mean(multi_errs) if multi_errs else np.nan,
            "delta_mae": (
                np.mean(base_errs) - np.mean(multi_errs)
                if base_errs and multi_errs else np.nan
            ),
            "n_samples": len(base_errs),
        })

        print(
            f"Fold {fold} | "
            f"baseline MAE = {rows[-1]['baseline_mae']:.2f} | "
            f"multicand MAE = {rows[-1]['multicand_mae']:.2f}"
        )

    # ==================================================
    # SUMMARY
    # ==================================================
    out = pd.DataFrame(rows)

    print("\n=== KFold SUMMARY ===")
    print(out)

    print("\nMean baseline MAE :", out["baseline_mae"].mean())
    print("Mean multicand MAE:", out["multicand_mae"].mean())
    print("Mean ΔMAE         :", out["delta_mae"].mean())

    out_path = os.path.join(
        PROJECT_ROOT, "result", "kfold_baseline_vs_multicand.csv"
    )
    out.to_csv(out_path, index=False)

    print(f"\nSaved to: {out_path}")

# ======================================================
if __name__ == "__main__":
    main()
