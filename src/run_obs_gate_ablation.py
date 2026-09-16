# -*- coding: utf-8 -*-
# LEGACY / 旧版：不属于当前 FSIM 分析入口。
# Status: legacy_analysis; see config/project_registry.json and config/README.md.
# Retained for historical traceability; current results require separate validation.

import os
import sys
import json
import pandas as pd
from copy import deepcopy

# --------------------------------------------------
# path setup
# --------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.crop_season import CropSeasonFirst
from model.config import config
from obs_gate import SporeObsGateWindow   # ← 新增

# --------------------------------------------------
# input rb files
# --------------------------------------------------
RB_FILES = [
    "DA_MPT_2022_grape_rb.json",
    "DA_MPT_2023_grape_rb.json",
    "DA_MPT_2024_grape_rb.json",
    "GL_WK_2022_grape_rb.json",
    "NN_JF_2023_grape_rb.json",
    "NN_JF_2024_grape_rb.json",
]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_OBS_PATH = os.path.join(DATA_DIR, "sporangia_obs.xlsx")

# --------------------------------------------------
# load actual FIRST infection data
# --------------------------------------------------
actual_df = pd.read_excel(ACTUAL_DATA_PATH)
actual_df["bbch_date"] = pd.to_datetime(actual_df["bbch_date"], errors="coerce")
actual_df["first_infection_date"] = pd.to_datetime(
    actual_df["first_infection_date"], errors="coerce"
)

# --------------------------------------------------
# utility: DOY
# --------------------------------------------------
def to_doy(dt):
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear

# --------------------------------------------------
# utility: signed error (DOY, event-based)
# --------------------------------------------------
def calc_signed_error(actual_date, oil_start, oil_end):
    if pd.isna(actual_date) or pd.isna(oil_start) or pd.isna(oil_end):
        return None

    doy_actual = to_doy(actual_date)
    doy_start = to_doy(oil_start)
    doy_end = to_doy(oil_end)

    if doy_actual is None or doy_start is None or doy_end is None:
        return None

    if doy_end < doy_actual:
        return doy_end - doy_actual      # negative → model earlier
    elif doy_start > doy_actual:
        return doy_start - doy_actual    # positive → model later
    else:
        return 0.0

# --------------------------------------------------
# result container
# --------------------------------------------------
results = []

# --------------------------------------------------
# batch run
# --------------------------------------------------
for rb_name in RB_FILES:

    rb_path = os.path.join(DATA_DIR, rb_name)
    if not os.path.exists(rb_path):
        print(f"⚠️ File not found: {rb_name}")
        continue

    with open(rb_path, "r", encoding="utf-8") as f:
        rb = json.load(f)

    city = rb.get("Place", "NA")
    year = rb.get("year", None)

    print(f"\n=== Running {city}-{year} ===")

    # --------------------------------------------------
    # find actual data row (plot + year)
    # --------------------------------------------------
    row_act = actual_df[
        (actual_df["plot"] == city) &
        (actual_df["first_infection_date"].dt.year == year)
    ]

    if row_act.empty:
        bbch_date = None
        actual_date = None
    else:
        row_act = row_act.iloc[0]
        bbch_date = row_act["bbch_date"]
        actual_date = row_act["first_infection_date"]

    # ==================================================
    # 1. BASELINE (no obs_gate)
    # ==================================================
    cs_base = CropSeasonFirst(csrb=deepcopy(rb), config=config)
    cs_base.simulate(use_obs_gate=False)

    df_base = cs_base.primary_chain

    if df_base is None or df_base.empty:
        oil_start_base, oil_end_base = None, None
    else:
        df_eff = df_base.copy()

        if pd.notna(bbch_date):
            df_eff = df_eff[df_eff["oil_spot_start"] >= bbch_date]

        if df_eff.empty:
            oil_start_base, oil_end_base = None, None
        else:
            first = df_eff.sort_values("oil_spot_start").iloc[0]
            oil_start_base = first["oil_spot_start"]
            oil_end_base = first["oil_spot_end"]

    err_base = calc_signed_error(actual_date, oil_start_base, oil_end_base)

    # ==================================================
    # 2. OBS-GATED
    # ==================================================
    obs_gate = SporeObsGateWindow(
        obs_file=SPORE_OBS_PATH,
        site=city,
        year=year,
        w_zero=0.3
    )

    cs_gate = CropSeasonFirst(csrb=deepcopy(rb), config=config)
    cs_gate.simulate(
        use_obs_gate=True,
        obs_gate=obs_gate
    )

    df_gate = cs_gate.primary_chain

    if df_gate is None or df_gate.empty:
        oil_start_gate, oil_end_gate = None, None
    else:
        df_eff = df_gate.copy()

        if pd.notna(bbch_date):
            df_eff = df_eff[df_eff["oil_spot_start"] >= bbch_date]

        if df_eff.empty:
            oil_start_gate, oil_end_gate = None, None
        else:
            first = df_eff.sort_values("oil_spot_start").iloc[0]
            oil_start_gate = first["oil_spot_start"]
            oil_end_gate = first["oil_spot_end"]

    err_gate = calc_signed_error(actual_date, oil_start_gate, oil_end_gate)

    # --------------------------------------------------
    # collect results
    # --------------------------------------------------
    results.append([
        city,
        year,
        bbch_date,
        oil_start_base,
        oil_end_base,
        oil_start_gate,
        oil_end_gate,
        actual_date,
        err_base,
        err_gate
    ])

# --------------------------------------------------
# output table
# --------------------------------------------------
out_df = pd.DataFrame(
    results,
    columns=[
        "plot",
        "year",
        "bbch_date_actual",
        "oil_spot_start_baseline",
        "oil_spot_end_baseline",
        "oil_spot_start_gated",
        "oil_spot_end_gated",
        "actual_first_infection",
        "signed_error_DOY_baseline",
        "signed_error_DOY_gated"
    ]
)

print("\n=== First infection comparison (baseline vs obs-gated) ===")
print(out_df.to_string(index=False))

# --------------------------------------------------
# summary statistics
# --------------------------------------------------
valid_base = out_df["signed_error_DOY_baseline"].dropna()
valid_gate = out_df["signed_error_DOY_gated"].dropna()

print("\n=== Error summary ===")

if not valid_base.empty:
    print(f"Baseline MAE (DOY): {valid_base.abs().mean():.2f}")
else:
    print("Baseline MAE: NA")

if not valid_gate.empty:
    print(f"Obs-gated MAE (DOY): {valid_gate.abs().mean():.2f}")
else:
    print("Obs-gated MAE: NA")

if not valid_base.empty and not valid_gate.empty:
    diff = valid_gate.abs().mean() - valid_base.abs().mean()
    print(f"Δ MAE (gated - baseline): {diff:.2f} days")

# --------------------------------------------------
# optional save
# --------------------------------------------------
out_df.to_csv(
    os.path.join(DATA_DIR, "first_infection_obs_gate_compare.csv"),
    index=False
)
