# -*- coding: utf-8 -*-
"""
Compute Signed / MAE Sobol ST ratio
==================================
Based on N=2048 Sobol results
"""

import os
import pandas as pd
import numpy as np

# ======================================================
# PATH SETUP
# ======================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOBOL_DIR = os.path.join(PROJECT_ROOT, "result", "sobol")
OUT_DIR = os.path.join(PROJECT_ROOT, "result", "sobol", "derived")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# INPUT FILES (N=2048)
# ======================================================
MAE_FILE = os.path.join(
    SOBOL_DIR, "sobol_first_inf_ONLY_MAE_N2048.xlsx"
)
SIGNED_FILE = os.path.join(
    SOBOL_DIR, "sobol_first_inf_ONLY_Signed_N2048.xlsx"
)

# ======================================================
# LOAD
# ======================================================
df_mae = pd.read_excel(MAE_FILE)
df_signed = pd.read_excel(SIGNED_FILE)

# safety check
assert set(df_mae["Parameter"]) == set(df_signed["Parameter"])

df = df_mae.merge(
    df_signed[["Parameter", "ST"]],
    on="Parameter",
    suffixes=("_MAE", "_Signed")
)

# ======================================================
# COMPUTE RATIO
# ======================================================
eps = 1e-8  # avoid zero division
df["ST_ratio_Signed_over_MAE"] = df["ST_Signed"] / (df["ST_MAE"] + eps)

# classification (for interpretation & plotting)
def classify(r):
    if np.isnan(r):
        return "inactive"
    if r > 2:
        return "direction-dominant"
    elif r < 0.5:
        return "error-dominant"
    else:
        return "mixed"

df["effect_type"] = df["ST_ratio_Signed_over_MAE"].apply(classify)

# sort for plotting
df = df.sort_values("ST_ratio_Signed_over_MAE", ascending=False)

# ======================================================
# SAVE
# ======================================================
out_csv = os.path.join(OUT_DIR, "sobol_ST_ratio_signed_over_mae_N2048.csv")
out_xlsx = os.path.join(OUT_DIR, "sobol_ST_ratio_signed_over_mae_N2048.xlsx")

df.to_csv(out_csv, index=False)
df.to_excel(out_xlsx, index=False)

print("Saved:")
print(out_csv)
print(out_xlsx)
