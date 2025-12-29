# -*- coding: utf-8 -*-
"""
Figure: Signed / MAE Sobol ST ratio
==================================
SCI-ready, two-column figure
"""

import os
import sys

# ======================================================
# Make project root importable
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.sobol_param_meta import param_to_abbr

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# STYLE (SCI)
# ======================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# colorblind-friendly palette
palette = {
    "direction-dominant": "#0072B2",  # blue
    "mixed": "#009E73",               # green
    "error-dominant": "#D55E00",      # vermillion
    "inactive": "#999999"
}

# ======================================================
# PATH
# ======================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "result", "sobol", "derived",
    "sobol_ST_ratio_signed_over_mae_N2048.csv"
)
FIG_DIR = os.path.join(PROJECT_ROOT, "fig")
os.makedirs(FIG_DIR, exist_ok=True)

# ======================================================
# LOAD
# ======================================================
df = pd.read_csv(DATA_FILE)

# remove inactive (ST≈0 in both)
df_plot = df[df["effect_type"] != "inactive"]

df_plot["Parameter"] = df_plot["Parameter"].apply(param_to_abbr)


# ======================================================
# PLOT
# ======================================================
fig, ax = plt.subplots(figsize=(7.2, 3.8))  # two-column width

sns.barplot(
    data=df_plot,
    y="Parameter",
    x="ST_ratio_Signed_over_MAE",
    hue="effect_type",
    palette=palette,
    dodge=False,
    ax=ax
)

# reference line
ax.axvline(1.0, color="black", linestyle="--", linewidth=1)

ax.set_xlabel("ST(Signed) / ST(MAE)")
ax.set_ylabel("")

# legend
ax.legend(
    title="Parameter role",
    loc="lower right",
    frameon=False
)

sns.despine(ax=ax)

# ======================================================
# SAVE
# ======================================================
out_png = os.path.join(FIG_DIR, "Fig_ST_ratio_Signed_over_MAE.png")
out_pdf = os.path.join(FIG_DIR, "Fig_ST_ratio_Signed_over_MAE.pdf")

plt.tight_layout()
plt.show()
# plt.savefig(out_png, dpi=600)
# plt.savefig(out_pdf)
# plt.close()

# print("Figure saved:")
# print(out_png)
# print(out_pdf)
