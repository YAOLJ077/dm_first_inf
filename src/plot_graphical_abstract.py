#!/usr/bin/env python3
"""Draw the PMS graphical abstract at its 50 x 60 mm display size."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from PIL import Image
import numpy as np
import pandas as pd

from plot_table_vine_summary import _load_boundary


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#183D62"
GREEN = "#26725F"
ORANGE = "#B5672B"
INK = "#24323B"
GRAY = "#68767B"


def _results() -> tuple[float, float, float, float]:
    fsim = pd.read_csv(ROOT / "result/fsim_calibration/predictions.csv")
    fsim_s = pd.read_csv(ROOT / "result/fsim_s_calibration/predictions.csv")
    errors_f = fsim.loc[fsim["model"].eq("FSIM"), "error_days"]
    errors_s = fsim_s.loc[fsim_s["is_held_out"].eq(1), "error_days"]
    if len(errors_f) != 6 or len(errors_s) != 6:
        raise ValueError("Expected six held-out predictions per model")
    savings = []
    for profile in ("table", "vine"):
        path = ROOT / "result/guangxi_2000_2024_fungicides" / (
            f"guangxi_2000_2024_fungicide_savings_{profile}.csv"
        )
        data = pd.read_csv(path, usecols=["year", "grid_id", "potential_savable_spray_count"])
        data = data.loc[data["year"].between(2000, 2024)]
        savings.append(data.groupby("grid_id")["potential_savable_spray_count"].mean().mean())
    return float(errors_f.abs().mean()), float(errors_s.abs().mean()), *map(float, savings)


def _artwork(source: Path, crop: tuple[int, int, int, int]) -> Image.Image:
    """Reuse biological illustrations from the supplied image on a white ground."""
    image = Image.open(source).convert("RGBA").crop(crop)
    white = Image.new("RGBA", image.size, "white")
    white.alpha_composite(image)
    return white.convert("RGB")


def draw(output: Path, dpi: int = 1200) -> None:
    mae_f, mae_s, savings_t, savings_w = _results()
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 6.5, "pdf.fonttype": 42,
        "image.composite_image": False,
    })
    fig = plt.figure(figsize=(50 / 25.4, 60 / 25.4), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 100), ylim=(0, 120))
    ax.set_aspect("equal")
    ax.axis("off")
    text_artists = []

    def text(x, y, label, size=6.5, color=INK, weight="normal", **kwargs):
        artist = ax.text(
            x, y, label, fontsize=size, color=color, fontweight=weight,
            ha=kwargs.pop("ha", "center"), va="center", linespacing=1.05, **kwargs,
        )
        text_artists.append(artist)
        return artist

    def arrow(start, end, color=BLUE, width=0.8, style="-|>", **kwargs):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=5,
            linewidth=width, color=color, shrinkA=0, shrinkB=0, **kwargs,
        ))

    def node(x, y, width, height, label, size=6.5):
        ax.add_patch(Rectangle(
            (x - width / 2, y - height / 2), width, height,
            facecolor="#F3F7FA", edgecolor=BLUE, linewidth=0.65,
        ))
        text(x, y, label, size=size, color=BLUE)

    text(50, 115, "First Seasonal Infection Model", size=7.5, weight="bold", color=BLUE)
    text(50, 108.8, "Subtropical double-cropping vineyards", size=6.2, color=GREEN)
    ax.plot((5, 95), (104.5, 104.5), color="#CED9DC", linewidth=0.5)

    # The monitoring branch constrains cohort formation, before all dispersal.
    text(20, 99.6, "FSIM-S", size=7.5, weight="bold", color=GREEN)
    text(68, 99.6, "FSIM", size=7.5, weight="bold", color=BLUE)
    text(68, 94.8, "Weather + host development", size=5.9)

    source = ROOT / "fig/FSIM_FSIM_S.png"
    spores = _artwork(source, (65, 370, 825, 730))
    # Remove the source diagram's old label below the sporangia illustration.
    spores.paste("white", (500, 260, spores.width, spores.height))
    ax.imshow(spores, extent=(6, 33, 87.5, 96.3), aspect="auto", interpolation="none", zorder=2)
    text(20, 83.8, "Spore monitoring", size=6.1, color=GREEN)
    text(20, 78.9, "0: suppress", size=6.2, color=ORANGE)
    text(20, 74.1, "Positive: add", size=6.2, color=GREEN)
    text(20, 69.3, "Missing: no trigger", size=5.7, color=GRAY)
    arrow((34.5, 87.8), (43.5, 87.8), color=GREEN, width=0.95)

    node(68, 87.8, 49, 8, "Sporangial cohorts", size=6.6)
    arrow((68, 83.8), (68, 79.8))
    node(68, 75.8, 49, 8, "Dispersal & deposition", size=6.2)
    arrow((68, 71.8), (68, 67.8))
    node(68, 63, 49, 9.6, "Zoospore release\nInfection", size=6.4)
    arrow((68, 58.2), (68, 50.2))
    text(71.6, 54.7, "Incubation", size=5.9, ha="left")

    lesion = _artwork(source, (2590, 155, 2980, 520))
    ax.imshow(lesion, extent=(28.35, 42.65, 45.39, 58.81), aspect="auto", interpolation="none", zorder=2)
    text(15, 52.1, "Grapevine\ndowny mildew", size=6.0, color=BLUE)
    text(69, 47.1, "First symptom window", size=6.5)

    ax.plot((5, 95), (42, 42), color="#CED9DC", linewidth=0.5)
    text(50, 37.7, "Validation MAE (days)", size=6.4, weight="bold")
    text(22, 30.6, f"{mae_f:.2f}", size=10.5, color=BLUE, weight="bold")
    text(55, 30.6, f"{mae_s:.2f}", size=10.5, color=GREEN, weight="bold")
    arrow((34, 30.6), (43, 30.6), color=GREEN, width=1.0)
    text(22, 24.9, "FSIM", size=6.0, color=BLUE)
    text(55, 24.9, "FSIM-S", size=6.0, color=GREEN)
    text(83, 30.6, f"-{100 * (1 - mae_s / mae_f):.1f}%", size=8.0, color=ORANGE, weight="bold")
    text(83, 24.9, "MAE", size=5.8, color=ORANGE)

    ax.plot((5, 95), (21, 21), color="#CED9DC", linewidth=0.5)
    text(62, 17.0, "Regional scenario reconstruction", size=6.0, weight="bold", color=BLUE)
    text(62, 12.3, "Guangxi, 2000-2024", size=6.0)
    text(62, 7.6, "Model-guided potential savings", size=6.0)
    text(62, 3.0, f"Table {savings_t:.1f} | Wine {savings_w:.1f} sprays/season", size=6.0, color=GREEN, weight="bold")

    # Select the province outline rather than its municipal subdivisions.
    boundary_paths = list((ROOT / "data/shapefile").glob("*.shp"))
    boundary_path = min(boundary_paths, key=lambda path: len(path.stem))
    parts = _load_boundary(boundary_path)
    vertices = np.vstack(parts)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    scale = min(23 / (upper[0] - lower[0]), 16 / (upper[1] - lower[1]))
    for part in parts:
        points = (part - (lower + upper) / 2) * scale + (16, 11)
        ax.add_patch(Polygon(points, closed=True, facecolor="#DCECE5", edgecolor=GREEN, linewidth=0.45))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Verify that final labels remain inside the fixed-size submission image.
    for artist in text_artists:
        if not fig.bbox.contains(*artist.get_window_extent(renderer).get_points()[0]) or not fig.bbox.contains(
            *artist.get_window_extent(renderer).get_points()[1]
        ):
            raise ValueError(f"Text outside graphical abstract: {artist.get_text()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {output} (50 x 60 mm, {dpi} dpi)")
    print(f"Saved {pdf_output}")
    print(f"Validation MAE: {mae_f:.2f} -> {mae_s:.2f} days")
    print(f"Potential sprays saved: table {savings_t:.1f}; wine {savings_w:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "fig/graphical_abstract.png")
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    draw(args.output, args.dpi)


if __name__ == "__main__":
    main()
