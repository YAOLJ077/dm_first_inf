#!/usr/bin/env python3
"""Plot Table/Vine symptom-window timing and spray-saving summaries."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROFILE_COLORS = {"table": "#168f78", "vine": "#756bb1"}
MISSING_COLOR = "#d9d9d9"
BOUNDARY_COLOR = "#202020"
MUNICIPAL_COLOR = "#8a8a8a"
GRID_EDGE = "#ffffff"
FONT_FAMILY = "Times New Roman"
plt.rcParams.update({"font.family": FONT_FAMILY, "axes.unicode_minus": False})


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _read_onset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "year", "grid_id", "latitude", "longitude", "first_symptom_start_doy",
        "first_symptom_end_doy", "symptom_window_complete",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["grid_id"] = pd.to_numeric(frame["grid_id"], errors="raise").astype(int)
    for column in ("latitude", "longitude", "first_symptom_start_doy", "first_symptom_end_doy"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "first_symptom_midpoint_doy" in frame:
        frame["window_midpoint_doy"] = pd.to_numeric(
            frame["first_symptom_midpoint_doy"], errors="coerce"
        )
    else:
        frame["window_midpoint_doy"] = (
            frame["first_symptom_start_doy"] + frame["first_symptom_end_doy"]
        ) / 2.0
    frame["complete_window"] = _bool_series(frame["symptom_window_complete"])
    frame = frame.loc[frame["year"].between(2000, 2024)].copy()
    grids = (
        frame[["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id")
        .sort_values(["latitude", "longitude", "grid_id"])
        .reset_index(drop=True)
    )
    return frame, grids


def _read_savings(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"year", "grid_id", "potential_savable_spray_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["grid_id"] = pd.to_numeric(frame["grid_id"], errors="raise").astype(int)
    frame["potential_savable_spray_count"] = pd.to_numeric(
        frame["potential_savable_spray_count"], errors="coerce"
    )
    return frame.loc[frame["year"].between(2000, 2024)].copy()


def _load_boundary(path: Path) -> list[np.ndarray]:
    """Read polygon/polyline parts from a WGS84 ESRI Shapefile.

    This small reader avoids requiring GeoPandas just for drawing the stored
    Guangxi boundaries. Polygon holes are still valid line parts and are
    intentionally drawn as outlines.
    """
    parts: list[np.ndarray] = []
    with path.open("rb") as handle:
        header = handle.read(100)
        if len(header) != 100:
            raise ValueError(f"Invalid shapefile header: {path}")
        while True:
            record_header = handle.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError(f"Truncated shapefile record header: {path}")
            content_length = struct.unpack(">i", record_header[4:8])[0] * 2
            content = handle.read(content_length)
            if len(content) != content_length:
                raise ValueError(f"Truncated shapefile record: {path}")
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in (3, 5):
                raise ValueError(f"Unsupported shapefile geometry type {shape_type}: {path}")
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            part_indices = struct.unpack(f"<{num_parts}i", content[44:44 + 4 * num_parts])
            points_start = 44 + 4 * num_parts
            points = np.frombuffer(
                content, dtype="<f8", count=2 * num_points, offset=points_start
            ).reshape(num_points, 2)
            for index, start in enumerate(part_indices):
                stop = part_indices[index + 1] if index + 1 < num_parts else num_points
                parts.append(points[start:stop].copy())
    if not parts:
        raise ValueError(f"No drawable geometries found in {path}")
    return parts


def _boundary_bounds(parts: list[np.ndarray]) -> tuple[float, float, float, float]:
    coordinates = np.vstack(parts)
    return (
        float(np.nanmin(coordinates[:, 0])), float(np.nanmin(coordinates[:, 1])),
        float(np.nanmax(coordinates[:, 0])), float(np.nanmax(coordinates[:, 1])),
    )


def _inside_boundary(x: np.ndarray, y: np.ndarray, boundary: list[np.ndarray]) -> np.ndarray:
    points = np.column_stack((x.ravel(), y.ravel()))
    inside = np.zeros(len(points), dtype=bool)
    for part in boundary:
        if len(part) >= 3:
            inside |= MplPath(part, closed=True).contains_points(points)
    return inside.reshape(x.shape)


def _map_surface(
    grids: pd.DataFrame,
    values: pd.Series,
    boundary: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a full display grid with nearest-neighbor padding for clipping."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    longitudes = np.sort(grids["longitude"].unique())
    latitudes = np.sort(grids["latitude"].unique())
    x_index = {value: index for index, value in enumerate(longitudes)}
    y_index = {value: index for index, value in enumerate(latitudes)}
    surface = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    for index, row in grids.reset_index(drop=True).iterrows():
        surface[y_index[row["latitude"]], x_index[row["longitude"]]] = numeric[index]
    dx = float(np.median(np.diff(longitudes))) if len(longitudes) > 1 else 0.25
    dy = float(np.median(np.diff(latitudes))) if len(latitudes) > 1 else 0.25
    # Extend the display grid to the outline; added cells are not observations.
    minx, miny, maxx, maxy = _boundary_bounds(boundary)
    pad_left = max(0, int(np.ceil((longitudes[0] - minx) / dx - 0.5)))
    pad_right = max(0, int(np.ceil((maxx - longitudes[-1]) / dx - 0.5)))
    pad_bottom = max(0, int(np.ceil((latitudes[0] - miny) / dy - 0.5)))
    pad_top = max(0, int(np.ceil((maxy - latitudes[-1]) / dy - 0.5)))
    if pad_left or pad_right or pad_bottom or pad_top:
        surface = np.pad(
            surface,
            ((pad_bottom, pad_top), (pad_left, pad_right)),
            mode="constant", constant_values=np.nan,
        )
        if pad_left:
            longitudes = np.r_[longitudes[0] - dx * np.arange(pad_left, 0, -1), longitudes]
        if pad_right:
            longitudes = np.r_[longitudes, longitudes[-1] + dx * np.arange(1, pad_right + 1)]
        if pad_bottom:
            latitudes = np.r_[latitudes[0] - dy * np.arange(pad_bottom, 0, -1), latitudes]
        if pad_top:
            latitudes = np.r_[latitudes, latitudes[-1] + dy * np.arange(1, pad_top + 1)]
    # Fill display-only cells from the nearest original finite grid point.
    # Statistics below continue to use the original numeric values.
    missing = ~np.isfinite(surface)
    if missing.any():
        grid_x, grid_y = np.meshgrid(longitudes, latitudes)
        known = ~missing
        if not known.any():
            raise ValueError("Cannot draw a map without finite source values")
        longitude_scale = np.cos(np.deg2rad(np.mean(latitudes)))
        distance_squared = (
            ((grid_x[missing, None] - grid_x[known]) * longitude_scale) ** 2
            + (grid_y[missing, None] - grid_y[known]) ** 2
        )
        surface[missing] = surface[known][distance_squared.argmin(axis=1)]
    x_edges = np.r_[longitudes - dx / 2, longitudes[-1] + dx / 2]
    y_edges = np.r_[latitudes - dy / 2, latitudes[-1] + dy / 2]
    return x_edges, y_edges, surface


def _draw_map(
    ax: plt.Axes,
    grids: pd.DataFrame,
    values: pd.Series,
    boundary: list[np.ndarray],
    municipalities: list[np.ndarray] | None,
    *,
    cmap: str,
    norm: mcolors.Normalize,
    title: str,
    panel: str,
    value_label: str,
) -> object:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x_edges, y_edges, surface = _map_surface(grids, values, boundary)
    minx, miny, maxx, maxy = _boundary_bounds(boundary)
    collection = ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(surface), cmap=plt.get_cmap(cmap),
        norm=norm, shading="flat", edgecolors="none", linewidth=0,
        rasterized=True, zorder=2,
    )
    compound = MplPath.make_compound_path(*[
        MplPath(part, closed=True) for part in boundary if len(part) >= 3
    ])
    collection.set_clip_path(PathPatch(compound, transform=ax.transData))
    for part in boundary:
        ax.plot(part[:, 0], part[:, 1], color=BOUNDARY_COLOR, linewidth=0.9, zorder=5)
    if municipalities is not None:
        for part in municipalities:
            ax.plot(part[:, 0], part[:, 1], color=MUNICIPAL_COLOR, linewidth=0.35, zorder=4)
    ax.set_xlim(minx - 0.08, maxx + 0.08)
    ax.set_ylim(miny - 0.08, maxy + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    valid_values = numeric[np.isfinite(numeric)]
    mean = float(np.mean(valid_values)) if len(valid_values) else float("nan")
    median = float(np.median(valid_values)) if len(valid_values) else float("nan")
    decimals = 1 if value_label == "DOY" else 2
    ax.text(
        0.03, 0.96, f"({panel}) {title}", transform=ax.transAxes,
        ha="left", va="top", fontsize=21, fontfamily=FONT_FAMILY,
    )
    ax.text(
        0.03, 0.05, f"Mean = {mean:.{decimals}f}\nMedian = {median:.{decimals}f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=15,
        fontfamily=FONT_FAMILY, bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
    )
    return collection


def _grid_mean(frame: pd.DataFrame, value_column: str, *, complete_only: bool = False) -> pd.Series:
    data = frame.loc[frame["complete_window"]] if complete_only else frame
    values = pd.to_numeric(data[value_column], errors="coerce")
    return data.assign(_value=values).groupby("grid_id")["_value"].mean()


def _bar_axis(
    ax: plt.Axes,
    values: pd.Series,
    *,
    profile: str,
    panel: str,
    kind: str,
    bins: np.ndarray,
) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if kind == "doy":
        numeric = numeric[(numeric >= bins[0]) & (numeric <= bins[-1])]
    else:
        numeric = numeric[(numeric >= 1) & (numeric <= 6)]
    counts, edges = np.histogram(numeric, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(np.min(np.diff(edges)) * (0.82 if kind == "doy" else 0.72))
    ax.bar(
        centers, counts, width=width, color=PROFILE_COLORS[profile],
        edgecolor="#222222", linewidth=0.7,
    )
    mean, median = float(np.mean(numeric)), float(np.median(numeric))
    ax.text(
        0.98, 0.96, f"Mean = {mean:.1f}\nMedian = {median:.1f}" if kind == "doy"
        else f"Mean = {mean:.2f}\nMedian = {median:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=14,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
    )
    title = "Table" if profile == "table" else "Wine"
    ax.text(0.015, 0.96, f"({panel}) {title}", transform=ax.transAxes,
            ha="left", va="top", fontsize=19, fontfamily=FONT_FAMILY)
    ax.set_ylabel("Count", fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    if kind == "doy":
        tick_step = 2 if len(centers) > 16 else 1
        ax.set_xticks(centers[::tick_step])
        ax.set_xticklabels([f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(0, len(edges) - 1, tick_step)],
                           rotation=0, ha="center", fontsize=12)
        ax.set_xlabel("Day of year (DOY)", fontsize=15)
        ax.set_xlim(bins[0], bins[-1])
    else:
        ax.set_xticks(centers)
        ax.set_xticklabels([str(int(round(v))) for v in centers], fontsize=13)
        ax.set_xlabel("Number of Sprays Saved", fontsize=15)
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0, 200)
    ax.grid(axis="y", color="#cfcfcf", linestyle="--", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _style_colorbar(fig: plt.Figure, cax: plt.Axes, mappable: object, label: str, ticks: np.ndarray) -> None:
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.set_label(label, fontsize=14, labelpad=2, fontfamily=FONT_FAMILY)
    cbar.ax.tick_params(labelsize=12, length=2)


def _save_composite(
    output: Path,
    *,
    grids: pd.DataFrame,
    map_values: dict[str, pd.Series],
    bar_values: dict[str, pd.Series],
    boundary: list[np.ndarray],
    municipalities: list[np.ndarray] | None,
    kind: str,
) -> None:
    all_map = np.concatenate([
        pd.to_numeric(map_values[profile], errors="coerce").dropna().to_numpy(dtype=float)
        for profile in ("table", "vine")
    ])
    if kind == "doy":
        vmin = float(np.floor(np.nanmin(all_map) / 10.0) * 10.0)
        vmax = float(np.ceil(np.nanmax(all_map) / 10.0) * 10.0)
        cmap, label = "YlGnBu", "Symptom-window midpoint (DOY)"
        ticks = np.arange(vmin, vmax + 1, 10)
        bins = np.arange(vmin, vmax + 1, 10)
    else:
        vmin, vmax = 1.0, 6.0
        cmap, label = "PuBuGn", "Number of Sprays Saved (model-guided vs. schedule)"
        ticks = np.arange(1, 7, 1)
        bins = np.arange(0.5, 6.6, 1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(13.2, 9.4), facecolor="white")
    layout = fig.add_gridspec(
        4, 2, height_ratios=[1.25, 0.07, 0.12, 1.0],
        left=0.045, right=0.975, bottom=0.075, top=0.965,
        wspace=0.12, hspace=0.025,
    )
    map_axes = [fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1])]
    cbar_axes = [fig.add_subplot(layout[1, 0]), fig.add_subplot(layout[1, 1])]
    bar_axes = [fig.add_subplot(layout[3, 0]), fig.add_subplot(layout[3, 1])]
    profiles = ("table", "vine")
    for index, profile in enumerate(profiles):
        title = "Table" if profile == "table" else "Wine"
        mappable = _draw_map(
            map_axes[index], grids, map_values[profile], boundary, municipalities,
            cmap=cmap, norm=norm, title=title, panel=chr(ord("a") + index), value_label="DOY" if kind == "doy" else "sprays",
        )
        _style_colorbar(fig, cbar_axes[index], mappable, label, ticks)
        _bar_axis(bar_axes[index], bar_values[profile], profile=profile,
                  panel=chr(ord("c") + index), kind=kind, bins=bins)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onset-dir", type=Path, default=ROOT / "result/guangxi_2000_2024_onset_windows")
    parser.add_argument("--savings-dir", type=Path, default=ROOT / "result/guangxi_2000_2024_fungicides")
    parser.add_argument("--boundary", type=Path, default=ROOT / "data/shapefile/广西壮族自治区.shp")
    parser.add_argument("--municipal-boundary", type=Path, default=ROOT / "data/shapefile/广西壮族自治区_市.shp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fig")
    args = parser.parse_args(argv)

    onset: dict[str, pd.DataFrame] = {}
    savings: dict[str, pd.DataFrame] = {}
    grids = None
    for profile in ("table", "vine"):
        onset[profile], current_grids = _read_onset(
            args.onset_dir / f"guangxi_2000_2024_onset_windows_{profile}.csv"
        )
        savings[profile] = _read_savings(
            args.savings_dir / f"guangxi_2000_2024_fungicide_savings_{profile}.csv"
        )
        grids = current_grids if grids is None else grids
    boundary = _load_boundary(args.boundary)
    municipalities = _load_boundary(args.municipal_boundary) if args.municipal_boundary.exists() else None

    window_map = {
        profile: grids["grid_id"].map(_grid_mean(onset[profile], "window_midpoint_doy", complete_only=True))
        for profile in ("table", "vine")
    }
    window_bar = {profile: window_map[profile].dropna() for profile in ("table", "vine")}
    saving_map = {
        profile: grids["grid_id"].map(
            savings[profile].assign(_value=savings[profile]["potential_savable_spray_count"])
            .groupby("grid_id")["_value"].mean()
        ).clip(lower=1.0, upper=6.0)
        for profile in ("table", "vine")
    }
    saving_bar = {profile: saving_map[profile].dropna() for profile in ("table", "vine")}
    _save_composite(
        args.output_dir / "figure_13.png",
        grids=grids, map_values=window_map, bar_values=window_bar,
        boundary=boundary, municipalities=municipalities, kind="doy",
    )
    _save_composite(
        args.output_dir / "figure_14.png",
        grids=grids, map_values=saving_map, bar_values=saving_bar,
        boundary=boundary, municipalities=municipalities, kind="saving",
    )
    print("Saved:", args.output_dir / "figure_13.png")
    print("Saved:", args.output_dir / "figure_14.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
