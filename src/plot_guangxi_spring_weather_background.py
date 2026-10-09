#!/usr/bin/env python3
"""Plot spring weather background at 180 mm width for a two-column paper.

Top: grid means over 2000-2024. Bottom: equally weighted grid means for
each spring with spatial 25th-75th percentiles (not confidence intervals),
and an ordinary least-squares linear trend reported per decade.
Spring is March-May; nighttime high humidity is RH >=85% at 20:00-07:59
local time, as defined by the weather statistics builder. Map padding is
display-only and never enters the temporal statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_table_vine_summary import (
    ROOT, FONT_FAMILY, _boundary_bounds, _load_boundary, _map_surface,
)
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLES = (
    ("spring_temperature_mean", "Mean temperature", "Temperature (\N{DEGREE SIGN}C)",
     "YlOrRd", "#bf4a30", 2.0),
    ("spring_rainfall_mm", "Cumulative precipitation", "Cumulative precipitation (mm)",
     "YlGnBu", "#2472a4", 200.0),
    ("spring_night_wet_hours_mean", "Nighttime high humidity", "RH\N{GREATER-THAN OR EQUAL TO}85% (h/night)",
     "Greens", "#23834b", 2.0),
)


def load_statistics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = ["year", "grid_id", "longitude", "latitude"] + [item[0] for item in VARIABLES]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing weather columns: {sorted(missing)}")
    frame = frame.loc[frame["year"].between(2000, 2024)].copy()
    if frame.duplicated(["grid_id", "year"]).any():
        raise ValueError("Duplicate grid-year weather records")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Weather statistics contain missing or nonfinite values")
    if sorted(frame["year"].unique()) != list(range(2000, 2025)):
        raise ValueError("Expected all years from 2000 to 2024")
    if not frame.groupby("grid_id")["year"].nunique().eq(25).all():
        raise ValueError("Every source grid must have 25 years of weather statistics")
    if not frame.groupby("grid_id")[["longitude", "latitude"]].nunique().eq(1).all().all():
        raise ValueError("Grid coordinates change across years")
    return frame


def plot_background(frame: pd.DataFrame, boundary: list[np.ndarray], output: Path,
                    width_mm: float = 180.0) -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": 8,
        "axes.labelsize": 8, "axes.linewidth": 0.6,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.unicode_minus": False,
    })
    grids = frame[["grid_id", "longitude", "latitude"]].drop_duplicates("grid_id").reset_index(drop=True)
    baseline = frame.groupby("grid_id")[[item[0] for item in VARIABLES]].mean()
    bounds = _boundary_bounds(boundary)
    clip = MplPath.make_compound_path(*[
        MplPath(part, closed=True) for part in boundary if len(part) >= 3
    ])
    fig = plt.figure(figsize=(width_mm / 25.4, 122 / 25.4), facecolor="white")
    layout = fig.add_gridspec(
        4, 3, height_ratios=[1.18, 0.065, 0.31, 1.15],
        left=0.075, right=0.98, top=0.95, bottom=0.15,
        wspace=0.41, hspace=0.03,
    )
    for index, (column, title, units, cmap, color, step) in enumerate(VARIABLES):
        ax = fig.add_subplot(layout[0, index])
        values = grids["grid_id"].map(baseline[column])
        x, y, surface = _map_surface(grids, values, boundary)
        low = float(np.floor(values.min() / step) * step)
        high = float(np.ceil(values.max() / step) * step)
        mesh = ax.pcolormesh(
            x, y, surface, cmap=cmap, norm=mcolors.Normalize(low, high),
            shading="flat", edgecolors="none", linewidth=0,
            antialiased=False, rasterized=True,
        )
        mesh.set_clip_path(PathPatch(clip, transform=ax.transData))
        for part in boundary:
            ax.plot(part[:, 0], part[:, 1], color="#333333", linewidth=0.45)
        ax.set_xlim(bounds[0] - 0.05, bounds[2] + 0.05)
        ax.set_ylim(bounds[1] - 0.05, bounds[3] + 0.05)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"({chr(97 + index)}) {title}", fontsize=9, loc="left", pad=5)
        cax = fig.add_subplot(layout[1, index])
        position = cax.get_position()
        cax.set_position([position.x0, position.y0 + 0.035,
                          position.width, position.height])
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        tick_step = step * 2 if (high - low) / step > 5 else step
        cbar.set_ticks(np.arange(low, high + step / 2, tick_step))
        cbar.ax.xaxis.set_major_formatter("{x:g}")
        cbar.ax.tick_params(labelsize=7, length=2, pad=2)
        cbar.set_label(units, fontsize=8, labelpad=2)

        ax = fig.add_subplot(layout[3, index])
        grouped = frame.groupby("year")[column]
        means = grouped.mean()
        q25, q75 = grouped.quantile(0.25), grouped.quantile(0.75)
        years = means.index.to_numpy()
        ax.fill_between(years, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(years, means, color=color, linewidth=1, marker="o", markersize=2,
                markeredgewidth=0.4, markerfacecolor="white")
        centered_years = years - years.mean()
        slope, intercept = np.polyfit(centered_years, means.to_numpy(), 1)
        ax.plot(years, slope * centered_years + intercept, color="#444444",
                linestyle="--", linewidth=0.9, zorder=4)
        trend_units = ("\N{DEGREE SIGN}C", "mm", "h/night")[index]
        ax.text(0.97, 0.97, f"Trend: {slope * 10:+.2f} {trend_units}/decade",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                zorder=5)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.14)
        print(f"{column}: OLS trend = {slope * 10:+.4f} {trend_units}/decade")
        ax.set_title(f"({chr(100 + index)}) Annual variability", fontsize=9, loc="left", pad=5)
        ax.set_ylabel(units, labelpad=3)
        ax.set_xlabel("Year", labelpad=3)
        ax.set_xlim(1999.5, 2024.5)
        ax.set_xticks([2000, 2008, 2016, 2024])
        ax.tick_params(length=2.5, pad=2)
        ax.grid(axis="y", color="#dddddd", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[
        Line2D([], [], color="#444444", marker="o", markerfacecolor="white",
               markersize=3, linewidth=1, label="Regional grid mean"),
        Line2D([], [], color="#444444", linestyle="--", linewidth=0.9,
               label="Linear trend"),
        Patch(facecolor="#888888", alpha=0.25, edgecolor="none",
              label="25th-75th percentile range"),
    ], loc="lower center", bbox_to_anchor=(0.53, 0.01), frameon=False,
        ncol=3, fontsize=7.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the specified physical width; tight cropping would change it.
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output} ({width_mm:g} mm wide, 600 dpi)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "result/guangxi_2000_2024_weather/guangxi_grid_year_weather_stats.csv")
    parser.add_argument("--boundary", type=Path,
                        default=ROOT / "data/shapefile/广西壮族自治区.shp")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "fig/figure_12.png")
    parser.add_argument("--width-mm", type=float, default=180.0)
    args = parser.parse_args()
    if args.width_mm <= 0:
        parser.error("--width-mm must be positive")
    plot_background(load_statistics(args.input), _load_boundary(args.boundary),
                    args.output, args.width_mm)


if __name__ == "__main__":
    main()
