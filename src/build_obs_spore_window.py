# -*- coding: utf-8 -*-
"""
Build spore observation windows (positive + negative)
=====================================================
- One row = one observation window
- Window = [obs_date - 2 days, obs_date]
- Only keep windows within 14 days before first infection
"""

import os
import pandas as pd
from datetime import timedelta

# ======================================================
# PATHS
# ======================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ACTUAL_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_PATH = os.path.join(DATA_DIR, "Spore_Overall.xlsx")  # or .csv
OUT_PATH = os.path.join(DATA_DIR, "spore_observation_windows_14d.csv")

# ======================================================
# LOAD DATA
# ======================================================
actual_df = pd.read_excel(ACTUAL_PATH)
actual_df["first_infection_date"] = pd.to_datetime(
    actual_df["first_infection_date"], errors="coerce"
)

spore_df = pd.read_excel(SPORE_PATH)
spore_df["date"] = pd.to_datetime(spore_df["date"], errors="coerce")

# ======================================================
# BUILD WINDOWS
# ======================================================
rows = []

for _, act in actual_df.iterrows():

    plot = act["plot"]
    year = act["year"]
    inf_date = act["first_infection_date"]

    if pd.isna(inf_date):
        continue

    # 14-day look-back window
    t_start = inf_date - timedelta(days=14)
    t_end = inf_date

    df_site = spore_df[
        (spore_df["plot"] == plot) &
        (spore_df["year"] == year)
    ].copy()

    for _, r in df_site.iterrows():

        obs_date = r["date"]
        if pd.isna(obs_date):
            continue

        window_start = obs_date - timedelta(days=2)
        window_end = obs_date

        # only keep windows ending within [inf_date - 14, inf_date)
        if not (t_start <= window_end < t_end):
            continue

        spore_val = r.get("spore", None)
        label = 1 if pd.notna(spore_val) and spore_val > 0 else 0

        rows.append({
            "plot": plot,
            "year": year,
            "obs_date": obs_date,
            "window_start": window_start,
            "window_end": window_end,
            "label": label,
            "spore_value": spore_val,
        })

# ======================================================
# SAVE
# ======================================================
out_df = pd.DataFrame(rows).sort_values(
    ["plot", "year", "obs_date"]
).reset_index(drop=True)

out_df.to_csv(OUT_PATH, index=False)

print("✅ Spore observation windows saved:")
print(OUT_PATH)
print(out_df.groupby(["plot", "year"])["label"].value_counts())
