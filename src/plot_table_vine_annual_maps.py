#!/usr/bin/env python3
"""Plot annual symptom dates and spray savings for Table and Wine, 2000-2024.

Boundary padding is a nearest-neighbor extension for display only. Original
grid cells without a complete symptom window remain explicitly masked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_table_vine_summary import (
    ROOT, _boundary_bounds, _load_boundary, _map_surface, _read_onset, _read_savings,
)
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _annual_map_surface(
    grids: pd.DataFrame,
    values: pd.Series,
    boundary: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges, y_edges, surface = _map_surface(grids, values, boundary)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2
    missing = ~np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    # Keep missing simulated cells distinct from display-only boundary padding.
    for row in grids.loc[missing].itertuples():
        ix = np.abs(x_centres - row.longitude).argmin()
        iy = np.abs(y_centres - row.latitude).argmin()
        surface[iy, ix] = np.nan
    return x_edges, y_edges, surface


def plot_annual_maps(
    frame: pd.DataFrame,
    grids: pd.DataFrame,
    boundary: list[np.ndarray],
    profile: str,
    output: Path,
    *,
    kind: str = "symptom",
    savings_max: int = 9,
) -> None:
    years = list(range(2000, 2025))
    if kind == "symptom":
        data = frame.loc[frame["complete_window"]]
        value_column = "window_midpoint_doy"
        norm = mcolors.Normalize(50, 300)
        cmap, ticks = "YlGnBu", np.arange(50, 301, 50)
        label = "First symptom-window midpoint (DOY)"
        title = "Annual first symptom-window midpoint"
    elif kind == "savings":
        data = frame
        value_column = "potential_savable_spray_count"
        norm = mcolors.BoundaryNorm(np.arange(-0.5, savings_max + 1.5), 256)
        cmap, ticks = "PuBuGn", np.arange(0, savings_max + 1)
        label = "Potentially avoidable applications per season"
        title = "Annual potential fungicide savings"
    else:
        raise ValueError(f"Unknown annual map kind: {kind}")
    minx, miny, maxx, maxy = _boundary_bounds(boundary)
    clip_path = MplPath.make_compound_path(*[
        MplPath(part, closed=True) for part in boundary if len(part) >= 3
    ])
    colour_map = plt.get_cmap(cmap).copy()
    colour_map.set_bad("#d9d9d9")
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.925,
                        wspace=0.06, hspace=0.23)
    for index, (ax, year) in enumerate(zip(axes.flat, years)):
        yearly = data.loc[data["year"].eq(year)].groupby("grid_id")[value_column].median()
        values = grids["grid_id"].map(yearly)
        x_edges, y_edges, surface = _annual_map_surface(grids, values, boundary)
        mesh = ax.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(surface), cmap=colour_map, norm=norm,
            shading="flat", edgecolors="none", linewidth=0,
            antialiased=False, rasterized=True, zorder=2,
        )
        mesh.set_clip_path(PathPatch(clip_path, transform=ax.transData))
        for part in boundary:
            ax.plot(part[:, 0], part[:, 1], color="#303030", linewidth=0.65, zorder=3)
        ax.set_xlim(minx - 0.08, maxx + 0.08)
        ax.set_ylim(miny - 0.08, maxy + 0.08)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([106, 109, 112])
        ax.set_yticks([22, 24, 26])
        ax.tick_params(labelsize=9, length=2, pad=1,
                       labelbottom=index // 5 == 4, labelleft=index % 5 == 0)
        ax.set_title(str(year), fontsize=11, pad=3)
    fig.text(0.525, 0.115, "Longitude (degrees E)", ha="center", fontsize=10)
    fig.text(0.006, 0.55, "Latitude (degrees N)", va="center", rotation=90, fontsize=10)
    cax = fig.add_axes([0.20, 0.075, 0.65, 0.017])
    cbar = fig.colorbar(mesh, cax=cax, ticks=ticks, orientation="horizontal")
    cbar.set_label(label, fontsize=11, labelpad=3)
    cbar.ax.tick_params(labelsize=10, length=2, pad=2)
    if kind == "symptom":
        fig.legend(handles=[Patch(facecolor="#d9d9d9", label="No complete symptom window")],
                   loc="lower center", bbox_to_anchor=(0.525, -0.005),
                   frameon=False, fontsize=10)
    fig.suptitle(f"{title} - {profile.title()} grapes",
                 fontsize=13, y=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "result/guangxi_2000_2024_onset_windows")
    parser.add_argument("--savings-dir", type=Path,
                        default=ROOT / "result/guangxi_2000_2024_fungicides")
    parser.add_argument("--boundary", type=Path,
                        default=ROOT / "data/shapefile/广西壮族自治区.shp")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "fig/guangxi_2000_2024")
    args = parser.parse_args()
    boundary = _load_boundary(args.boundary)
    profiles = {"table": "table", "vine": "wine"}
    savings = {
        source: _read_savings(
            args.savings_dir / f"guangxi_2000_2024_fungicide_savings_{source}.csv"
        ) for source in profiles
    }
    savings_max = max(1, int(np.ceil(max(
        frame["potential_savable_spray_count"].max() for frame in savings.values()
    ))))
    for source, profile in profiles.items():
        frame, grids = _read_onset(
            args.input_dir / f"guangxi_2000_2024_onset_windows_{source}.csv"
        )
        plot_annual_maps(frame, grids, boundary, profile,
                         args.output_dir / f"01_annual_first_symptom_maps_{profile}.png")
        plot_annual_maps(savings[source], grids, boundary, profile,
                         args.output_dir / f"02_annual_spray_savings_maps_{profile}.png",
                         kind="savings", savings_max=savings_max)


if __name__ == "__main__":
    main()
