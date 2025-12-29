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
import matplotlib as mpl
import numpy as np


mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOBOL_DIR = os.path.join(ROOT, "result", "sobol")
FIG_DIR = os.path.join(ROOT, "fig")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_excel(
    os.path.join(SOBOL_DIR, "sobol_first_inf_ONLY_Signed_N2048.xlsx")
)

df = df.sort_values("ST", ascending=True)

y = np.arange(len(df))
colors = ["#D55E00" if v > 0 else "#0072B2" for v in df["ST"]]

fig, ax = plt.subplots(figsize=(7.0, 3.5))

ax.barh(y, df["ST"], color=colors)

ax.axvline(0, color="black", linewidth=0.8)

ax.set_yticks(y)
ax.set_yticklabels([param_to_abbr(p) for p in df["Parameter"]])
ax.set_xlabel("Signed total-order index")

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(FIG_DIR, "fig3_Signed_ST.png"), dpi=600)
# plt.close()
