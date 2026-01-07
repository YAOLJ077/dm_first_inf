# -*- coding: utf-8 -*-
"""
Build spore absence runs from Spore_Overall.xlsx
with:
1) spore numeric handling (spore > 0 = presence)
2) right-shifted absence run start after last presence (presence + LAG_DAYS)

Absence run definition:
- spore == 0 only
- NaN rows are ignored (do NOT break run)
- spore > 0 breaks run (presence)
- at least MIN_RUN_LENGTH absence samples
- max gap between absence samples <= MAX_GAP_DAYS
"""

import os
import pandas as pd
from datetime import timedelta

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_PATH = os.path.join(DATA_DIR, "Spore_Overall.xlsx")
OUTPUT_PATH = os.path.join(DATA_DIR, "spore_absence_runs.csv")

# ======================================================
# PARAMETERS (absence run construction)
# ======================================================
MIN_RUN_LENGTH = 2
MAX_GAP_DAYS = 5
BUFFER_DAYS = 0

# ======================================================
# BIOLOGICAL LAG AFTER PRESENCE (explicit & tunable)
# ======================================================
LAG_DAYS = 7   # sporangia survival + incubation buffer

# ======================================================
# ABSENCE LEVEL THRESHOLDS (explicit & tunable)
# ======================================================

# Strong absence
STRONG_MIN_SAMPLES = 6
STRONG_MIN_DURATION = 15  # days

# Medium absence
MEDIUM_MIN_SAMPLES = 3
MEDIUM_MAX_SAMPLES = 5
MEDIUM_MIN_DURATION = 7   # days

# Weak absence
WEAK_MIN_SAMPLES = 2
WEAK_MIN_DURATION = 3     # days

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_excel(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
df["spore"] = pd.to_numeric(df["spore"], errors="coerce")  # <<< KEY FIX
df = df.sort_values(["plot", "year", "date"]).reset_index(drop=True)

# ======================================================
# BUILD ABSENCE RUNS
# ======================================================
runs = []

for (plot, year), g in df.groupby(["plot", "year"]):

    g = g.sort_values("date").reset_index(drop=True)

    current_run = []
    last_absence_date = None
    last_presence_date = None

    for _, row in g.iterrows():

        is_presence = pd.notna(row["spore"]) and row["spore"] > 0
        is_absence = pd.notna(row["spore"]) and row["spore"] == 0

        if is_presence:
            last_presence_date = row["date"]
            if len(current_run) >= MIN_RUN_LENGTH:
                runs.append({
                    "run": current_run,
                    "last_presence_date": last_presence_date
                })
            current_run = []
            last_absence_date = None
            continue

        if is_absence:
            if last_absence_date is None:
                current_run = [row]
            else:
                gap = (row["date"] - last_absence_date).days
                if gap <= MAX_GAP_DAYS:
                    current_run.append(row)
                else:
                    if len(current_run) >= MIN_RUN_LENGTH:
                        runs.append({
                            "run": current_run,
                            "last_presence_date": last_presence_date
                        })
                    current_run = [row]

            last_absence_date = row["date"]

        # NaN → ignore completely

    if len(current_run) >= MIN_RUN_LENGTH:
        runs.append({
            "run": current_run,
            "last_presence_date": last_presence_date
        })

# ======================================================
# ABSENCE LEVEL ASSIGNMENT
# ======================================================
def assign_absence_level(n_samples, duration_days):
    if n_samples >= STRONG_MIN_SAMPLES and duration_days >= STRONG_MIN_DURATION:
        return "Strong"
    if MEDIUM_MIN_SAMPLES <= n_samples <= MEDIUM_MAX_SAMPLES and duration_days >= MEDIUM_MIN_DURATION:
        return "Medium"
    if n_samples >= WEAK_MIN_SAMPLES and duration_days >= WEAK_MIN_DURATION:
        return "Weak"
    return "Discard"

# ======================================================
# FORMAT OUTPUT (with right-shifted effective start)
# ======================================================
records = []

for item in runs:

    run = item["run"]
    last_presence = item["last_presence_date"]

    raw_start = run[0]["date"]
    raw_end = run[-1]["date"]

    # ---- Right-shift absence run start after presence ----
    effective_start = raw_start
    days_since_last_presence = None

    if last_presence is not None:
        days_since_last_presence = (raw_start - last_presence).days
        lag_start = last_presence + timedelta(days=LAG_DAYS)
        effective_start = max(raw_start, lag_start)

    # If effective window collapses, discard
    if effective_start > raw_end:
        continue

    n_samples = len(run)
    duration_days = (raw_end - effective_start).days

    absence_level = assign_absence_level(
        n_samples=n_samples,
        duration_days=duration_days
    )

    records.append({
        "plot": run[0]["plot"],
        "year": run[0]["year"],
        "run_start_raw": raw_start,
        "run_start_effective": effective_start,
        "run_end": raw_end,
        "n_samples": n_samples,
        "duration_days": duration_days,
        "last_presence_date": last_presence,
        "days_since_last_presence": days_since_last_presence,
        "exclude_start": effective_start - timedelta(days=BUFFER_DAYS),
        "exclude_end": raw_end + timedelta(days=BUFFER_DAYS),
        "absence_level": absence_level,
    })

out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(out_df)} absence runs to:")
print(OUTPUT_PATH)
