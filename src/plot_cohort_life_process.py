import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ======================================================
# ================= GLOBAL STYLE ========================
# ======================================================
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 12

# ======================================================
# ============== USER CONFIGURATION =====================
# ======================================================
PLACE = "DA"
YEAR = 2023

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULT_DIR = os.path.join(
    PROJECT_ROOT, "result", "first_infection", f"{PLACE}_{YEAR}"
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(PROJECT_ROOT, "fig")
os.makedirs(FIG_DIR, exist_ok=True)

COHORT_FILE = os.path.join(
    RESULT_DIR, "06_cohort_life_history_gated.xlsx"
)

JSON_FILE = os.path.join(
    DATA_DIR, f"{PLACE}_MPT_{YEAR}_grape_rb.json"
)

OUTPUT_FIG = os.path.join(
    FIG_DIR, f"first_infection_inventory_{PLACE}_{YEAR}.png"
)

# ======================================================
# ================= LOAD DATA ===========================
# ======================================================
df = pd.read_excel(COHORT_FILE, parse_dates=True)

# --- read BBCH_53 (fallback HHCH_53 for compatibility) ---
with open(JSON_FILE, "r", encoding="utf-8") as f:
    js = json.load(f)
HHCH_53 = pd.to_datetime(js.get("BBCH_53", js.get("HHCH_53")))

# ======================================================
# ========== FILTER VALID COHORTS ======================
# ======================================================
df_evt = df[
    (df["infection_effective"] == True) &
    (df["dispersal_datetime"].notna()) &
    (df["dispersal_datetime"] >= HHCH_53)
].copy()

if df_evt.empty:
    raise ValueError("No valid cohort after HHCH_53.")

df_evt = df_evt.sort_values("dispersal_datetime")
highlight_num = df_evt.iloc[0]["Num"]

# ======================================================
# ===== RECONSTRUCT SYSTEM AIRBORNE INVENTORY ==========
# ======================================================
t_start = df_evt["dispersal_datetime"].min().floor("D")
t_end = df_evt["detached_alive_end"].max().ceil("D")

time_index = pd.date_range(t_start, t_end, freq="D")
airborne_ts = pd.Series(0.0, index=time_index)

for _, r in df_evt.iterrows():
    if pd.isna(r["detached_alive_end"]):
        continue
    mask = (
        (airborne_ts.index >= r["dispersal_datetime"]) &
        (airborne_ts.index <= r["detached_alive_end"])
    )
    airborne_ts.loc[mask] += r["airborne_amount"]

# ======================================================
# ==================== PLOTTING =========================
# ======================================================
fig, ax = plt.subplots(figsize=(14, 5))

# ---- airborne inventory curve (BLUE) ----
ax.plot(
    airborne_ts.index,
    airborne_ts.values,
    color="blue",
    linewidth=2,
    label="System airborne sporangia inventory"
)

# ======================================================
# ================ EVENT HEIGHTS =======================
# ======================================================
# 固定 y 轴 → 事件整体上移
y_disp = 4.2   # ~70%
y_dep  = 4.5   # ~75%
y_zre  = 4.8   # ~80%
y_inf  = 5.1   # ~85%
y_oil  = 5.4   # ~90%

# ======================================================
# ================= EVENT SYMBOLS =======================
# ======================================================
for _, c in df_evt.iterrows():

    alpha = 1.0 if c["Num"] == highlight_num else 0.8

    # ---- Dispersal ----
    if c["dispersal_type"] == "rain":
        ax.scatter(
            c["dispersal_datetime"], y_disp,
            marker="v", color="orange",
            s=70, alpha=alpha
        )
    else:
        ax.scatter(
            c["dispersal_datetime"], y_disp,
            marker="v", color="green",
            s=70, alpha=alpha
        )

    # ---- Deposition ----
    if pd.notna(c["deposition_datetime"]):
        ax.scatter(
            c["deposition_datetime"], y_dep,
            marker="s", color="black",
            s=50, alpha=alpha
        )

    # ---- Zoospore release ----
    if pd.notna(c["zoospore_release_datetime"]):
        ax.scatter(
            c["zoospore_release_datetime"], y_zre,
            marker="D", color="purple",
            s=50, alpha=alpha
        )

    # ---- Infection ----
    if pd.notna(c["infection_datetime"]):
        ax.scatter(
            c["infection_datetime"], y_inf,
            marker="x", color="red",
            s=70, alpha=alpha
        )

    # ---- Oil spot window ----
    if pd.notna(c["oil_spot_start"]) and pd.notna(c["oil_spot_end"]):
        ax.hlines(
            y=y_oil,
            xmin=c["oil_spot_start"],
            xmax=c["oil_spot_end"],
            colors="darkred",
            linewidth=4,
            alpha=alpha
        )

# ======================================================
# ================= FORMAT AXES =========================
# ======================================================
ax.set_ylim(0.0, 6.0)
ax.set_xlabel("Date")
ax.set_ylabel("Airborne sporangia inventory (relative units)")

# ax.set_title(
#     f"System-level airborne sporangia inventory and first-infection events\n"
#     f"{PLACE}, {YEAR}"
# )

from matplotlib.lines import Line2D

legend_elements = [

    # inventory
    Line2D([0], [0],
           color="blue", lw=2,
           label="Airborne sporangia inventory"),

    # dispersal
    Line2D([0], [0],
           marker="v", color="w",
           markerfacecolor="orange", markersize=8,
           label="Dispersal – rain"),

    Line2D([0], [0],
           marker="v", color="w",
           markerfacecolor="green", markersize=8,
           label="Dispersal – wind"),

    # deposition
    Line2D([0], [0],
           marker="s", color="black",
           linestyle="None", markersize=7,
           label="Deposition"),

    # zoospore release
    Line2D([0], [0],
           marker="D", color="purple",
           linestyle="None", markersize=7,
           label="Zoospore release (ZRE)"),

    # infection
    Line2D([0], [0],
           marker="x", color="red",
           linestyle="None", markersize=8,
           label="Infection"),

    # oil spot
    Line2D([0], [0],
           color="darkred", lw=4,
           label="Oil spot window")
]

ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=4,
    frameon=False
)


plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()

print(f"✅ Figure saved to: {OUTPUT_FIG}")
