# -*- coding: utf-8 -*-

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

from model.crop_season_only import CropSeasonFirst
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
    year = rb.get("year", "NA")
    bbch_53 = pd.to_datetime(rb.get("HHCH_53"), errors="coerce")

    # --------------------------------------------------
    # run FIRST infection model (full biological process)
    # --------------------------------------------------
    crop_season = CropSeasonFirst(csrb=rb, config=config)
    crop_season.simulate()

    # ==================================================
    # USAGE LAYER: read FINAL survivor-only state
    # ==================================================
    df = crop_season.primary_chain

    if df is None or df.empty:
        oil_start, oil_end = None, None

    else:
        df_eff = df.copy()

        # ---------- phenology gate (INTENTIONALLY HERE) ----------
        if pd.notna(bbch_53):
            df_eff = df_eff[df_eff["oil_spot_start"] >= bbch_53]

        if df_eff.empty:
            oil_start, oil_end = None, None
        else:
            first = df_eff.sort_values("oil_spot_start").iloc[0]
            oil_start = first["oil_spot_start"]
            oil_end = first["oil_spot_end"]

    results.append([
        city,
        year,
        bbch_53,
        oil_start,
        oil_end
    ])

# --------------------------------------------------
# print result table
# --------------------------------------------------
out_df = pd.DataFrame(
    results,
    columns=[
        "city",
        "year",
        "date_bbch_53",
        "oil_spot_start",
        "oil_spot_end"
    ]
)

print("\n=== First infection summary (current parameter set) ===")
print(out_df.to_string(index=False))
