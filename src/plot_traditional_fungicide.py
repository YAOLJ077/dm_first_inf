import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pathlib import Path


# -------------------
# Spray data
spray_no = [1,2,3,4,5,6,7,8,9]
doy = [74,104,114,126,131,139,149,158,195]
# -------------------

# --- Key disease dates ---
first_spore_doy = 101
first_infection_doy = 128

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "fig"

# ================================================================
#                   READ RAINFALL DATA
# ================================================================
rain_path = DATA_DIR / "traditional_fungicide_rainfall_2022_duan_tt_mpt.csv"
rain_daily = pd.read_csv(rain_path)
rain_daily["Date"] = pd.to_datetime(rain_daily["Date"])

rain_dict = dict(zip(rain_daily['DOY'], rain_daily['R']))

rain_doy = rain_daily['DOY'].values
rain_val = rain_daily['R'].values

# ================================================================
#                        FIGURE & AXES
# ================================================================
fig, ax_rain = plt.subplots(figsize=(7.4, 4.2))
ax = ax_rain.twinx()

# ================================================================
#              (2) SHADED AREA: first spray → first infection
# ================================================================
ax.axvspan(doy[0], first_infection_doy,
           color='lightgray', alpha=0.32, zorder=0)

# ================================================================
#                   RAINFALL – thin bar plot
# ================================================================
bar_width = 0.8    
ax_rain.bar(
    rain_doy, rain_val,
    width=bar_width,
    color='skyblue',
    edgecolor='steelblue',
    alpha=0.9,
    zorder=1
)

ax_rain.set_ylabel("Rainfall (mm)", fontsize=12)
ax_rain.set_ylim(0, max(rain_val)*1.6 if len(rain_val)>0 else 10)

# ================================================================
#                EVENT MARKERS AT TOP
# ================================================================
marker_y = 0.88
symbol_size = 120

ax.scatter(first_spore_doy, marker_y, s=symbol_size,
           color='darkorange', edgecolor='black')

ax.text(first_spore_doy-3, marker_y + 0.08,
        "First spore (101)",
        fontsize=12, color='darkorange')

ax.scatter(first_infection_doy, marker_y, s=symbol_size,
           marker='v', color='darkred', edgecolor='black')

ax.text(first_infection_doy - 1.5, marker_y + 0.08,
        "First seasonl infection (symptom onset, 128)",
        fontsize=12, color='darkred')

# ================================================================
#    (4) SPRAY ARROWS — color depends on rainfall in previous 2 days
# ================================================================
fungicide_path = FIG_DIR / "fungicide.jpg"
fungicide_img = mpimg.imread(fungicide_path)

target_fraction = 0.15
xrange = max(doy) - min(doy)
zoom = target_fraction * xrange / fungicide_img.shape[1]

for i, x in enumerate(doy):

    # Check rainfall in previous 2 days
    rain_window = rain_dict.get(x-1, 0) + rain_dict.get(x-2, 0)
    had_rain = (rain_window > 0)

    # If rainy → blue arrows; otherwise → green arrows
    arrow_color = "skyblue" if had_rain else "forestgreen"

    # Draw solid arrow (no dashed lines anymore)
    ax.annotate(
        '',
        xy=(x, 0.55), xytext=(x, 0.68),
        arrowprops=dict(
            arrowstyle='-|>,head_width=0.3,head_length=0.5',
            color=arrow_color,
            linewidth=2,
            linestyle='-'     # always solid
        )
    )

    # Draw the fungicide icon (no green fill)
    ab = AnnotationBbox(
        OffsetImage(fungicide_img, zoom=zoom),
        (x, 0.75),
        frameon=False,
        zorder=3
    )
    ax.add_artist(ab)



# ================================================================
#                         X/Y AXES – NORMAL FRAME
# ================================================================
ax_rain.set_xlabel("Day of Year (DOY)", fontsize=12)

xmin, xmax = min(doy)-10, max(doy)+10
ax_rain.set_xlim(xmin, xmax)
ax.set_xlim(xmin, xmax)

ax.set_ylim(-0.05, 1.05)
ax.set_yticks([])

# ---------- 设置 x / y 刻度字体大小 ----------
ax_rain.tick_params(axis='x', labelsize=12)
ax_rain.tick_params(axis='y', labelsize=12)

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)


# ================================================================
#                 AXIS TICKS → INWARD
# ================================================================
ax_rain.tick_params(axis='both', direction='in', length=4, width=1.2)
ax.tick_params(axis='both', direction='in', length=4, width=1.2)

# ================================================================
#                        LEGEND (only 3 items)
# ================================================================
# 绿色箭头（正常施药）
arrow_green = mlines.Line2D([], [], color='none', marker=r'$\downarrow$',
                            markerfacecolor='forestgreen', markeredgecolor='forestgreen',
                            markersize=12, label='Spray event')

# 蓝色箭头（降雨后施药）
arrow_blue = mlines.Line2D([], [], color='none', marker=r'$\downarrow$',
                           markerfacecolor='skyblue', markeredgecolor='skyblue',
                           markersize=12, label='Spray after rainfall')

# 雨量柱形图示例
rain_patch = mpatches.Patch(color='skyblue', edgecolor='steelblue',
                            label='Rainfall')

ax.legend(handles=[arrow_green, arrow_blue, rain_patch],
          loc='upper center',
          bbox_to_anchor=(0.5, 1.12),
          ncol=3,
          fontsize=11,
          frameon=True,
          columnspacing=3.8,
          handletextpad=0.4,
          handlelength=1.2,
          borderaxespad=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(FIG_DIR / 'fungicide_rainfall.png', dpi=300, bbox_inches='tight')
plt.show()
