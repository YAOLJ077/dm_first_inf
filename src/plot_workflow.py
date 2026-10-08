"""Create the compact study workflow used as manuscript Figure 1."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (ROOT / "fig" / "workflow.png", ROOT / "fig" / "Figure_01.png")

WHITE = "#FFFFFF"
INK = "#17313C"
TEXT = "#506872"
ARROW = "#5B6E76"
PANELS = (
    ("#F1F7F2", "#3F7D58"),
    ("#EEF6F9", "#27708D"),
    ("#FFF8E8", "#B87718"),
    ("#FAF1ED", "#A95437"),
)


def panel(ax, x, y, width, height, number, title, background, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=background, edgecolor=edge, linewidth=1.4, zorder=1,
        )
    )
    header_height = 0.075
    ax.add_patch(
        Rectangle(
            (x, y + height - header_height), width, header_height,
            facecolor=edge, edgecolor=edge, linewidth=0, zorder=2,
        )
    )
    ax.text(
        x + 0.022, y + height - header_height / 2, number,
        va="center", ha="left", color=WHITE, fontsize=8.0,
        fontweight="bold", zorder=3,
    )
    ax.text(
        x + 0.085, y + height - header_height / 2, title,
        va="center", ha="left", color=WHITE, fontsize=7.6,
        fontweight="bold", zorder=3,
    )


def card(ax, x, y, width, height, title, body, edge, body_size=7.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.003,rounding_size=0.007",
            facecolor=WHITE, edgecolor=edge, linewidth=0.85, zorder=3,
        )
    )
    wrapped_body = "\n".join(
        textwrap.fill(line, width=25, break_long_words=False)
        for line in body.splitlines()
    )
    ax.text(
        x + 0.025 * width, y + height - 0.18 * height, title,
        va="top", ha="left", color=INK, fontsize=6.8,
        fontweight="bold", zorder=4,
    )
    ax.text(
        x + 0.025 * width, y + height - 0.42 * height, wrapped_body,
        va="top", ha="left", color=TEXT, fontsize=min(body_size, 5.7),
        linespacing=1.15, zorder=4,
    )


def arrow(ax, start, end, color=ARROW, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=12,
            linewidth=1.25, color=color,
            linestyle="--" if dashed else "-",
            shrinkA=2, shrinkB=2, zorder=6,
        )
    )


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(figsize=(7.1, 4.65), dpi=180)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y, height = 0.13, 0.72
    widths = (0.235, 0.31, 0.19, 0.19)
    xs = (0.020, 0.270, 0.595, 0.800)

    background, edge = PANELS[0]
    panel(ax, xs[0], y, widths[0], height, "01", "Field data", background, edge)
    card(
        ax, xs[0] + 0.018, 0.705, widths[0] - 0.036, 0.085,
        "Experimental coverage",
        "3 Guangxi vineyards\n6 site-years (2022-2024)", edge, 6.7,
    )
    card(
        ax, xs[0] + 0.018, 0.585, widths[0] - 0.036, 0.095,
        "Disease and phenology",
        "Disease onset intervals\nBBCH 09 / 53 / 65 / 81", edge, 6.55,
    )
    card(
        ax, xs[0] + 0.018, 0.455, widths[0] - 0.036, 0.095,
        "Airborne sporangia",
        "Passive-trap observations\nRepeated slide counts", edge, 6.55,
    )
    card(
        ax, xs[0] + 0.018, 0.285, widths[0] - 0.036, 0.135,
        "Meteorological data",
        "Hourly T, RH, wind and rain\nDew point; VPD / DPD", edge, 6.3,
    )
    card(
        ax, xs[0] + 0.018, 0.175, widths[0] - 0.036, 0.080,
        "Site and management",
        "Cultivars, layout and spray records", edge, 6.25,
    )

    background, edge = PANELS[1]
    panel(ax, xs[1], y, widths[1], height, "02", "Model construction", background, edge)
    card(
        ax, xs[1] + 0.018, 0.490, widths[1] - 0.036, 0.220,
        "FSIM process chain",
        "Weather + phenology\nProduction and loss\nDispersal + deposition\nSurvival / infection / onset",
        edge, 6.0,
    )
    card(
        ax, xs[1] + 0.018, 0.285, widths[1] - 0.036, 0.170,
        "FSIM-S extension",
        "FSIM process chain\nRelative spore-pressure input\nResponse weighting",
        edge, 6.0,
    )
    card(
        ax, xs[1] + 0.018, 0.155, widths[1] - 0.036, 0.095,
        "Model output",
        "Symptom-onset window\nper site-year",
        edge, 6.3,
    )
    arrow(ax, (xs[1] + widths[1] / 2, 0.485), (xs[1] + widths[1] / 2, 0.455), edge)
    arrow(ax, (xs[1] + widths[1] / 2, 0.280), (xs[1] + widths[1] / 2, 0.250), edge)

    background, edge = PANELS[2]
    panel(ax, xs[2], y, widths[2], height, "03", "Validation", background, edge)
    card(
        ax, xs[2] + 0.015, 0.635, widths[2] - 0.030, 0.135,
        "Calibration",
        "Parameter bounds\nSensitivity screening\nFSIM / FSIM-S fit", edge, 6.1,
    )
    card(
        ax, xs[2] + 0.015, 0.435, widths[2] - 0.030, 0.135,
        "LOSO validation",
        "6 site-year folds\n5 train + 1 holdout", edge, 6.1,
    )
    card(
        ax, xs[2] + 0.015, 0.235, widths[2] - 0.030, 0.135,
        "Evaluation",
        "MAE / RMSE\nCity-year errors\nFSIM versus FSIM-S", edge, 6.1,
    )

    background, edge = PANELS[3]
    panel(ax, xs[3], y, widths[3], height, "04", "Regional use", background, edge)
    card(
        ax, xs[3] + 0.015, 0.635, widths[3] - 0.030, 0.135,
        "Historical forcing",
        "Guangxi 2000-2024\n342 grids x 25 years\n3 cultivars", edge, 6.0,
    )
    card(
        ax, xs[3] + 0.015, 0.435, widths[3] - 0.030, 0.135,
        "Onset windows",
        "Grid-year first-symptom\nwindows and disease risk", edge, 6.0,
    )
    card(
        ax, xs[3] + 0.015, 0.235, widths[3] - 0.030, 0.135,
        "Spray assessment",
        "Reference schedule\nRisk-aligned timing\nAvoidable early sprays", edge, 6.0,
    )

    arrow(ax, (xs[0] + widths[0] + 0.003, 0.50), (xs[1] - 0.003, 0.50))
    arrow(ax, (xs[1] + widths[1] + 0.003, 0.50), (xs[2] - 0.003, 0.50))
    arrow(ax, (xs[2] + widths[2] + 0.003, 0.50), (xs[3] - 0.003, 0.50))

    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, facecolor=WHITE)
    plt.close(fig)


if __name__ == "__main__":
    main()
