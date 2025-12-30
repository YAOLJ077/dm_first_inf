# -*- coding: utf-8 -*-
"""
Combined Sobol Figure (FINAL LOCKED VERSION)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# PATH & IMPORT
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.sobol_param_meta import param_to_abbr

# ======================================================
# STYLE
# ======================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

COL_S1 = "#56B4E9"
COL_ST = "#0072B2"
COL_POS = "#D55E00"
COL_NEG = "#0072B2"

PALETTE_ROLE = {
    "direction-dominant": "#0072B2",
    "mixed": "#009E73",
    "error-dominant": "#D55E00",
}
ROLE_ORDER = ["direction-dominant", "mixed", "error-dominant"]

# ======================================================
# PATHS
# ======================================================
SOBOL_DIR = os.path.join(PROJECT_ROOT, "result", "sobol")
DERIVED_DIR = os.path.join(SOBOL_DIR, "derived")
FIG_DIR = os.path.join(PROJECT_ROOT, "fig")
os.makedirs(FIG_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
df_mae_all = pd.read_excel(
    os.path.join(SOBOL_DIR, "sobol_first_inf_ONLY_MAE_N2048.xlsx")
)
df_signed_all = pd.read_excel(
    os.path.join(SOBOL_DIR, "sobol_first_inf_ONLY_Signed_N2048.xlsx")
)
df_ratio_all = pd.read_csv(
    os.path.join(DERIVED_DIR, "sobol_ST_ratio_signed_over_mae_N2048.csv")
)

# ======================================================
# (a)
# ======================================================
df_a = df_mae_all.sort_values("ST", ascending=False).head(8).copy()
df_a["abbr"] = df_a["Parameter"].apply(param_to_abbr)
order_a = df_a["abbr"].tolist()

# ======================================================
# (b)
# ======================================================
df_b = df_signed_all.copy()
df_b["abs_ST"] = df_b["ST"].abs()
df_b = df_b[df_b["Parameter"].isin(df_a["Parameter"])].sort_values(
    "abs_ST", ascending=False
)
df_b["abbr"] = df_b["Parameter"].apply(param_to_abbr)
order_b = df_b["abbr"].tolist()

# ======================================================
# (c) —— 关键修复版
# ======================================================
df_c = df_ratio_all.copy()
RATIO_COL = "ST_ratio_Signed_over_MAE"

# 1️⃣ 删 Parameter（第一道闸）
DROP_PARAMS = {"STOP", "STMA", "ZVPD", "ZARA"}
df_c = df_c[~df_c["Parameter"].isin(DROP_PARAMS)].copy()

# 2️⃣ 只保留合法 role
df_c = df_c[df_c["effect_type"].isin(ROLE_ORDER)].copy()

# 3️⃣ ratio 强制数值化 + finite
df_c[RATIO_COL] = pd.to_numeric(df_c[RATIO_COL], errors="coerce")
df_c = df_c[np.isfinite(df_c[RATIO_COL])].copy()

# 4️⃣ 生成 abbr
df_c["abbr"] = df_c["Parameter"].apply(param_to_abbr)

# 5️⃣ 再按 abbr 删一次（真正解决你现在的问题）
DROP_ABBR = {param_to_abbr(p) for p in DROP_PARAMS}
df_c = df_c[~df_c["abbr"].isin(DROP_ABBR)].copy()

# 6️⃣ 确保 abbr 是普通字符串（不是 category）
df_c["abbr"] = df_c["abbr"].astype(str)

# 7️⃣ role 顺序
df_c["effect_type"] = pd.Categorical(
    df_c["effect_type"], categories=ROLE_ORDER, ordered=True
)

df_c = df_c.sort_values("effect_type").copy()
order_c = df_c["abbr"].tolist()

# ======================================================
# FIGURE
# ======================================================
fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.6))

def panel(ax, txt):
    ax.text(0.99, 0.95, txt, transform=ax.transAxes,
            ha="right", va="top")

# (a)
ax = axes[0]
y = np.arange(len(order_a))
ax.barh(y - 0.2, df_a["S1"], height=0.35, color=COL_S1, label="S1")
ax.barh(y + 0.2, df_a["ST"], height=0.35, color=COL_ST, label="ST")
ax.set_yticks(y)
ax.set_yticklabels(order_a)
ax.set_xlim(left=0)
ax.set_xlabel("Sobol sensitivity index (MAE)")
ax.invert_yaxis()
ax.legend(loc="lower right", frameon=True)
panel(ax, "(a)")

# (b)
ax = axes[1]
y = np.arange(len(order_b))
colors = [COL_POS if v > 0 else COL_NEG for v in df_b["ST"]]
ax.barh(y, df_b["ST"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(order_b)
ax.set_xlabel("Signed total-order index")
ax.invert_yaxis()
panel(ax, "(b)")

# (c)
ax = axes[2]
sns.barplot(
    data=df_c,
    y="abbr",
    x=RATIO_COL,
    hue="effect_type",
    order=order_c,
    palette=PALETTE_ROLE,
    dodge=False,
    ax=ax
)
ax.axvline(1.0, color="black", ls="--", lw=1)
ax.set_xlabel("ST(Signed) / ST(MAE)")
ax.set_ylabel("")
ax.legend(title="Parameter role", loc="lower right", frameon=True)
panel(ax, "(c)")

# box
for ax in axes:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)

plt.tight_layout(h_pad=0.8)

out_png = os.path.join(FIG_DIR, "Fig_Sobol_Combined_FINAL.png")
plt.savefig(out_png, dpi=300, format="png", bbox_inches="tight")

plt.show()