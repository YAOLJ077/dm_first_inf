# -*- coding: utf-8 -*-
# LEGACY / 旧版：不属于当前 FSIM 分析入口。
# Status: legacy_analysis; see config/project_registry.json and config/README.md.
# Retained for historical traceability; current results require separate validation.

import os
import sys
import json
import pandas as pd

# --------------------------------------------------
# path setup
# --------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.crop_season import CropSeasonFirst
from model.config import config

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

    # --------------------------------------------------
    # run FIRST infection model
    # --------------------------------------------------
    crop_season = CropSeasonFirst(csrb=rb, config=config)
    crop_season.simulate()

    df = crop_season.primary_chain

    if df is None or df.empty:
        oil_start, oil_end = None, None
    else:
        df_eff = df.copy()

        # ---------- phenology gate: observed BBCH date ----------
        if pd.notna(bbch_date):
            df_eff = df_eff[df_eff["oil_spot_start"] >= bbch_date]

        if df_eff.empty:
            oil_start, oil_end = None, None
        else:
            first = df_eff.sort_values("oil_spot_start").iloc[0]
            oil_start = first["oil_spot_start"]
            oil_end = first["oil_spot_end"]

    # --------------------------------------------------
    # compute signed error
    # --------------------------------------------------
    error = calc_signed_error(actual_date, oil_start, oil_end)

    results.append([
        city,
        year,
        bbch_date,
        oil_start,
        oil_end,
        actual_date,
        error
    ])

# --------------------------------------------------
# print result table
# --------------------------------------------------
out_df = pd.DataFrame(
    results,
    columns=[
        "plot",
        "year",
        "bbch_date_actual",
        "oil_spot_start",
        "oil_spot_end",
        "actual_first_infection",
        "signed_error_DOY"
    ]
)

print("\n=== First infection summary (model vs actual) ===")
print(out_df.to_string(index=False))

# --------------------------------------------------
# summary statistics
# --------------------------------------------------
valid_errors = out_df["signed_error_DOY"].dropna()

if valid_errors.empty:
    print("\n⚠️ No valid error values to compute MAE.")
else:
    mae = valid_errors.abs().mean()
    mean_signed = valid_errors.mean()

    print("\n=== Error summary ===")
    print(f"Number of valid cases : {len(valid_errors)}")
    print(f"Mean signed error (DOY): {mean_signed:.2f}")
    print(f"MAE (DOY)              : {mae:.2f}")

