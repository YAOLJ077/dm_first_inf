import os
import sys

# ======================================================
# Make project root importable
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.sobol_param_meta import param_to_abbr


# =====================
# style
# =====================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# =====================
# paths
# =====================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOBOL_DIR = os.path.join(ROOT, "result", "sobol")
FIG_DIR = os.path.join(ROOT, "fig")
os.makedirs(FIG_DIR, exist_ok=True)

Ns = [512, 1024, 2048]
files = {
    N: os.path.join(SOBOL_DIR, f"sobol_first_inf_ONLY_MAE_N{N}.xlsx")
    for N in Ns
}

dfs = {N: pd.read_excel(f) for N, f in files.items()}

# select top parameters from largest N
top_params = (
    dfs[2048]
    .sort_values("ST", ascending=False)
    .head(6)["Parameter"]
    .tolist()
)

# Okabe–Ito colors
colors = ["#0072B2", "#D55E00", "#009E73",
          "#CC79A7", "#E69F00", "#56B4E9"]

top_params_abbr = [param_to_abbr(p) for p in top_params]

# =====================
# plot
# =====================
fig, ax = plt.subplots(figsize=(7.0, 3.5))  # double-column width

for p, p_abbr, c in zip(top_params, top_params_abbr, colors):
    ax.plot(
        Ns,
        [dfs[N].set_index("Parameter").loc[p, "ST"] for N in Ns],
        marker="o",
        linewidth=1.5,
        color=c,
        label=param_to_abbr(p)
    )


ax.set_xlabel("Sobol sample size (N)")
ax.set_ylabel("Total-order index (ST)")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(FIG_DIR, "fig1_ST_convergence.png"), dpi=600)
# plt.close()
