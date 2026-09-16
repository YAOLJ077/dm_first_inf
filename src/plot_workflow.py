"""Draw the revised manuscript workflow as four integrated sections."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fig" / "workflow.png"

INK = "#17313C"
TEXT = "#506872"
WHITE = "#FFFFFF"
DATA_BG, DATA_EDGE = "#F1F7F2", "#3F7D58"
MODEL_BG, MODEL_EDGE = "#EEF6F9", "#27708D"
EVAL_BG, EVAL_EDGE = "#FFF8E8", "#B87718"
REGION_BG, REGION_EDGE = "#FAF1ED", "#A95437"


def arrow(ax, start, end, color, dashed=False, width=1.35):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=width, color=color,
        linestyle="--" if dashed else "-",
        shrinkA=2, shrinkB=2, zorder=6,
    ))


def routed_arrow(ax, points, color, dashed=False, width=1.35):
    style = "--" if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=width, linestyle=style, zorder=5)
    arrow(ax, points[-2], points[-1], color, dashed=dashed, width=width)


def section(ax, x, y, w, h, number, title, bg, edge):
    outer = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.8, edgecolor=edge, facecolor=bg, zorder=1,
    )
    ax.add_patch(outer)
    header_h = 0.038
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h,
                           linewidth=0, facecolor=edge, zorder=2))
    ax.text(x + 0.022, y + h - header_h / 2, number,
            ha="left", va="center", fontsize=11.0, fontweight="bold",
            color=WHITE, zorder=3)
    ax.text(x + 0.078, y + h - header_h / 2, title,
            ha="left", va="center", fontsize=14.0, fontweight="bold",
            color=WHITE, zorder=3)
    return y + h - header_h


def cell(ax, x, y, w, h, title, body, edge, title_size=9.9, body_size=7.6):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.007",
        linewidth=1.05, edgecolor=edge, facecolor=WHITE, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025 * w, y + h - 0.16 * h, title,
            ha="left", va="top", fontsize=title_size, fontweight="bold",
            color=INK, zorder=4)
    ax.text(x + 0.025 * w, y + h - 0.43 * h, body,
            ha="left", va="top", fontsize=body_size, color=TEXT,
            linespacing=1.18, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(10, 14), dpi=180)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, width = 0.06, 0.88

    # 01 Field data
    section(ax, left, 0.720, width, 0.250, "01", "Field data", DATA_BG, DATA_EDGE)
    cell(ax, 0.085, 0.875, 0.83, 0.040,
         "Experimental coverage",
         "Three Guangxi vineyards | six independent CK site-years (2022-2024) | repeated observations nested within site-years",
         DATA_EDGE, title_size=9.4, body_size=7.2)
    cell(ax, 0.085, 0.805, 0.39, 0.060,
         "Disease observations",
         "Sequential incidence and severity assessments\nObserved onset interval (L,U]",
         DATA_EDGE)
    cell(ax, 0.525, 0.805, 0.39, 0.060,
         "Phenology observations",
         "Dated BBCH stages across crop development\nObserved phenology for the study site-years",
         DATA_EDGE)
    cell(ax, 0.085, 0.730, 0.39, 0.055,
         "Meteorological observations",
         "Hourly vineyard weather records\nacross the six site-years",
         DATA_EDGE, title_size=9.6, body_size=7.4)
    cell(ax, 0.525, 0.730, 0.39, 0.055,
         "Airborne-sporangia observations",
         "Repeated field slide counts\nacross six site-years",
         DATA_EDGE, title_size=9.4, body_size=7.4)

    # 02 Model construction
    section(ax, left, 0.545, width, 0.165, "02", "Model construction", MODEL_BG, MODEL_EDGE)
    cell(ax, 0.085, 0.575, 0.57, 0.085,
         "FSIM modules",
         "Sporangium production and loss | cohort formation\n"
         "dispersal | deposition | zoospore release and survival\n"
         "infection | incubation | symptom appearance",
         MODEL_EDGE, title_size=10.5, body_size=8.0)
    cell(ax, 0.695, 0.575, 0.22, 0.085,
         "FSIM-S extension",
         "FSIM modules\nobservation-derived relative\nspore-pressure field\nthreshold response",
         MODEL_EDGE, title_size=9.7, body_size=7.4)
    arrow(ax, (0.280, 0.730), (0.280, 0.660), DATA_EDGE)
    routed_arrow(ax, [(0.525, 0.835), (0.500, 0.835), (0.500, 0.675), (0.470, 0.660)], DATA_EDGE)
    arrow(ax, (0.720, 0.730), (0.805, 0.660), DATA_EDGE)
    arrow(ax, (0.655, 0.617), (0.695, 0.617), MODEL_EDGE)
    arrow(ax, (0.500, 0.545), (0.500, 0.515), MODEL_EDGE)

    # 03 Parameterization and evaluation
    section(ax, left, 0.295, width, 0.220, "03", "Parameterization and evaluation", EVAL_BG, EVAL_EDGE)
    cell(ax, 0.085, 0.410, 0.39, 0.060,
         "Biological parameterization",
         "Literature-fixed parameters | bounded perturbations\nStructural audit and Morris screening",
         EVAL_EDGE, title_size=9.7, body_size=7.5)
    cell(ax, 0.525, 0.410, 0.39, 0.060,
         "Staged calibration",
         "Parsimonious staged calibration within training site-years\nFSIM parameters, then one FSIM-S spore-response term",
         EVAL_EDGE, title_size=9.9, body_size=7.5)
    cell(ax, 0.085, 0.325, 0.39, 0.060,
         "Grouped cross-validation",
         "Primary evaluation: leave-one-site-year-out (6 folds)\nAll preprocessing, selection and fitting within training data",
         EVAL_EDGE, title_size=9.6, body_size=7.4)
    cell(ax, 0.525, 0.325, 0.39, 0.060,
         "Disease-onset evaluation",
         "Interval distance | overlap | no-onset predictions\nPaired out-of-fold FSIM-S versus FSIM comparison",
         EVAL_EDGE, title_size=9.1, body_size=7.2)
    routed_arrow(ax, [(0.085, 0.835), (0.035, 0.835), (0.035, 0.355), (0.085, 0.355)],
                 DATA_EDGE, dashed=True)
    arrow(ax, (0.280, 0.410), (0.280, 0.385), EVAL_EDGE)
    arrow(ax, (0.475, 0.440), (0.525, 0.440), EVAL_EDGE)
    arrow(ax, (0.720, 0.410), (0.720, 0.385), EVAL_EDGE)
    arrow(ax, (0.475, 0.355), (0.525, 0.355), EVAL_EDGE)

    # 04 Regional scenario analysis
    section(ax, left, 0.035, width, 0.230, "04", "Retrospective regional scenario", REGION_BG, REGION_EDGE)
    cell(ax, 0.085, 0.170, 0.39, 0.050,
         "Historical meteorological forcing",
         "Gridded hourly data | Guangxi | 2000-2024",
         REGION_EDGE, title_size=9.6, body_size=7.4)
    cell(ax, 0.525, 0.170, 0.39, 0.050,
         "Regional model inputs",
         "Phenology estimates | spore-pressure inputs",
         REGION_EDGE, title_size=9.8, body_size=7.0)
    cell(ax, 0.525, 0.100, 0.39, 0.050,
         "Regional FSIM-S simulations",
         "Infection-risk periods | predicted disease-onset windows",
         REGION_EDGE, title_size=9.6, body_size=7.0)
    cell(ax, 0.085, 0.100, 0.39, 0.050,
         "Recorded spray schedules",
         "Field application dates | reference schedule",
         REGION_EDGE, title_size=9.8, body_size=7.3)
    cell(ax, 0.195, 0.050, 0.61, 0.035,
         "Retrospective management assessment",
         "Risk-aligned windows | potentially avoidable early applications under stated assumptions; no efficacy claim",
         REGION_EDGE, title_size=9.0, body_size=6.8)
    arrow(ax, (0.475, 0.195), (0.525, 0.195), REGION_EDGE)
    arrow(ax, (0.720, 0.170), (0.720, 0.150), REGION_EDGE)
    arrow(ax, (0.280, 0.100), (0.390, 0.085), REGION_EDGE)
    arrow(ax, (0.720, 0.100), (0.610, 0.085), REGION_EDGE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


if __name__ == "__main__":
    main()
