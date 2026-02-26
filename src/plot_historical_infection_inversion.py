#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, gridspec
from shapely.geometry import box

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False


VARIETY_ORDER = ["WK", "JF", "MPT"]
VARIETY_COLOR = {"WK": "#1b9e77", "JF": "#d95f02", "MPT": "#7570b3"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Guangxi historical inversion outputs (maps + DOY distribution)."
    )
    p.add_argument(
        "--input-csv",
        type=str,
        default="../infection_date_2000_2024/infection_inversion_all_varieties_2000_2024.csv",
    )
    p.add_argument("--gx-shp", type=str, default="data/shapefile/广西壮族自治区.shp")
    p.add_argument("--city-shp", type=str, default="data/shapefile/广西壮族自治区_市.shp")
    p.add_argument("--out-dir", type=str, default="../fig")
    p.add_argument("--out-name", type=str, default="historical_infection_inversion_gx")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--doy-min", type=float, default=np.nan)
    p.add_argument("--doy-max", type=float, default=np.nan)
    p.add_argument("--bbox-buffer", type=float, default=0.5)
    return p.parse_args()


def _resolve_path(project_root: Path, p: str) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    return (project_root / pp).resolve()


def _prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"variety", "year", "grid_id", "latitude", "longitude", "has_event", "oil_spot_start"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input csv: {missing}")

    df = df.copy()
    df["variety"] = df["variety"].astype(str).str.upper()
    df["has_event"] = (
        df["has_event"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )
    df["oil_spot_start"] = pd.to_datetime(df["oil_spot_start"], errors="coerce")
    df["doy"] = df["oil_spot_start"].dt.dayofyear
    return df


def _load_boundaries(gx_path: Path, city_path: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gx = gpd.read_file(gx_path).to_crs("EPSG:4326")
    gx["geometry"] = gx.buffer(0)
    cities = gpd.read_file(city_path).to_crs("EPSG:4326")
    return gx, cities


def _build_non_gx_polygon(gx: gpd.GeoDataFrame, buffer_deg: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = gx.total_bounds
    bbox = box(minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    non_gx = bbox_gdf.difference(gx.unary_union)
    return gpd.GeoDataFrame(geometry=non_gx, crs="EPSG:4326")


def _aggregate_grid(df: pd.DataFrame, variety: str) -> pd.DataFrame:
    sub = df[df["variety"] == variety].copy()
    if sub.empty:
        return sub
    agg = (
        sub.groupby(["grid_id", "latitude", "longitude"], as_index=False)
        .agg(
            mean_doy=("doy", "mean"),
            median_doy=("doy", "median"),
            n_total=("has_event", "count"),
            n_event=("has_event", "sum"),
        )
        .sort_values(["latitude", "longitude"])
    )
    agg["event_rate"] = np.where(agg["n_total"] > 0, agg["n_event"] / agg["n_total"], np.nan)
    return agg


def _make_edges(values: np.ndarray) -> np.ndarray:
    vals = np.sort(np.unique(values.astype(float)))
    if vals.size == 1:
        d = 0.1
        return np.array([vals[0] - d, vals[0] + d], dtype=float)
    mids = (vals[:-1] + vals[1:]) / 2.0
    first = vals[0] - (mids[0] - vals[0])
    last = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _plot_map(
    ax,
    agg: pd.DataFrame,
    gx: gpd.GeoDataFrame,
    cities: gpd.GeoDataFrame,
    non_gx: gpd.GeoDataFrame,
    cmap,
    vmin: float,
    vmax: float,
    panel_label: str,
):
    gx.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=4)
    cities.boundary.plot(ax=ax, color="gray", linewidth=0.55, linestyle="--", zorder=3)

    stat_text = None
    if agg is not None and not agg.empty:
        lon_u = np.sort(agg["longitude"].unique())
        lat_u = np.sort(agg["latitude"].unique())
        lon_edges = _make_edges(lon_u)
        lat_edges = _make_edges(lat_u)
        grid = (
            agg.pivot(index="latitude", columns="longitude", values="mean_doy")
            .reindex(index=lat_u, columns=lon_u)
            .to_numpy()
        )
        # pcolormesh renders contiguous cells and avoids visible white seams.
        img = ax.pcolormesh(
            lon_edges,
            lat_edges,
            grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
            zorder=2,
        )
        stat_text = (
            f"Mean = {np.nanmean(agg['mean_doy']):.1f}\n"
            f"Median = {np.nanmedian(agg['mean_doy']):.1f}"
        )
    else:
        img = None
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12)

    if non_gx is not None and not non_gx.empty:
        non_gx.plot(ax=ax, facecolor="white", edgecolor="none", zorder=5)
    gx.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=6)

    # Draw stats after masking so the label is always readable.
    if stat_text is not None:
        ax.text(
            0.03,
            0.02,
            stat_text,
            transform=ax.transAxes,
            fontsize=10,
            color="black",
            ha="left",
            va="bottom",
            zorder=20,
            bbox=dict(facecolor="none", alpha=0.0, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    minx, miny, maxx, maxy = gx.total_bounds
    ax.set_xlim(minx - 0.2, maxx + 0.2)
    ax.set_ylim(miny - 0.2, maxy + 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    ax.text(
        0.02,
        0.95,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        zorder=40,
        clip_on=False,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )
    # Ensure full thin black frame on all four sides.
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")
        ax.spines[side].set_zorder(30)
    ax.set_frame_on(True)
    return img


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]

    input_csv = _resolve_path(project_root, args.input_csv)
    gx_shp = _resolve_path(project_root, args.gx_shp)
    city_shp = _resolve_path(project_root, args.city_shp)
    out_dir = _resolve_path(project_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_data(input_csv)
    gx, cities = _load_boundaries(gx_shp, city_shp)
    non_gx = _build_non_gx_polygon(gx, float(args.bbox_buffer))

    map_data: Dict[str, pd.DataFrame] = {v: _aggregate_grid(df, v) for v in VARIETY_ORDER}
    mean_doy_all = np.concatenate(
        [
            pd.to_numeric(map_data[v].get("mean_doy", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .to_numpy()
            for v in VARIETY_ORDER
            if v in map_data and map_data[v] is not None and not map_data[v].empty
        ]
    ) if any(v in map_data and map_data[v] is not None and not map_data[v].empty for v in VARIETY_ORDER) else np.array([])
    if mean_doy_all.size == 0:
        raise ValueError("No valid grid-mean DOY found in input data.")

    if np.isfinite(args.doy_min):
        vmin = float(args.doy_min)
    else:
        vmin = float(np.nanpercentile(mean_doy_all, 2))
    if np.isfinite(args.doy_max):
        vmax = float(args.doy_max)
    else:
        vmax = float(np.nanpercentile(mean_doy_all, 98))
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(mean_doy_all)), float(np.nanmax(mean_doy_all))

    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3.05, 1.2], hspace=0.12, wspace=0.02)
    axes_maps = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_hist = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.09)

    cmap = cm.get_cmap("viridis").reversed()
    last_img = None
    for i, variety in enumerate(VARIETY_ORDER):
        agg = map_data.get(variety, pd.DataFrame())
        last_img = _plot_map(
            axes_maps[i],
            agg,
            gx,
            cities,
            non_gx,
            cmap,
            vmin,
            vmax,
            f"({chr(ord('a') + i)}) {variety}",
        )

    bins = np.arange(150, 251, 10)
    centers = (bins[:-1] + bins[1:]) / 2
    width_group = 4.8
    width_single = width_group / 3.5

    for i, variety in enumerate(VARIETY_ORDER):
        agg = map_data.get(variety, pd.DataFrame())
        vals = (
            pd.to_numeric(agg.get("mean_doy", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .to_numpy()
        )
        counts, _ = np.histogram(vals, bins=bins)
        ax_hist.bar(
            centers + (i - 1) * width_single,
            counts,
            width=width_single,
            color=VARIETY_COLOR[variety],
            edgecolor="black",
            label=variety,
        )

    ax_hist.set_xlim(150, 250)
    ax_hist.set_xlabel("Day of Year (DOY)", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.tick_params(axis="x", direction="in", labelsize=10, pad=2)
    ax_hist.tick_params(axis="y", direction="in", labelsize=12)
    ax_hist.set_ylim(0, 250)
    ax_hist.set_yticks([0, 50, 100, 150, 200, 250])
    ax_hist.grid(axis="y", linestyle="--", alpha=0.35)
    ax_hist.set_xticks(centers)
    ax_hist.set_xticklabels([f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)], fontsize=10)

    ax_hist.legend(
        title="Variety",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        frameon=True,
        edgecolor="black",
        fontsize=12,
        title_fontsize=12,
        ncol=3,
        columnspacing=1.2,
    )
    ax_hist.text(
        0.01,
        0.95,
        "(d)",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )

    # Align histogram panel with full top-row width first.
    fig.canvas.draw()
    pos_l = axes_maps[0].get_position()
    pos_r = axes_maps[-1].get_position()
    pos_hist = ax_hist.get_position()
    hist_shift_up = 0.05
    ax_hist.set_position([pos_l.x0, pos_hist.y0 + hist_shift_up, pos_r.x1 - pos_l.x0, pos_hist.height])

    # Then shift histogram panel right so y-label "Count" left edge aligns with maps' left edge.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bbox = ax_hist.yaxis.get_label().get_window_extent(renderer=renderer)
    label_left = label_bbox.x0 / fig.bbox.width
    dx = pos_l.x0 - label_left
    if dx > 0:
        pos_hist = ax_hist.get_position()
        ax_hist.set_position([pos_hist.x0 + dx, pos_hist.y0, pos_hist.width - dx, pos_hist.height])

    # Colorbar aligned to the same combined top-row width.
    fig.canvas.draw()
    pos_hist = ax_hist.get_position()
    map_bottom = pos_l.y0
    hist_top = pos_hist.y1
    cbar_h = 0.018
    gap = max(0.004, map_bottom - hist_top)
    # Keep colorbar below the first-row maps with a visible gap from the map frame.
    cbar_y_target = map_bottom - cbar_h - 0.025
    cbar_y = max(hist_top + 0.001, cbar_y_target)
    cbar_ax = fig.add_axes([pos_l.x0, cbar_y, pos_r.x1 - pos_l.x0, cbar_h])
    cbar = fig.colorbar(last_img, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("First Infection Date (DOY)", fontsize=12, labelpad=4)
    cbar.ax.tick_params(labelsize=12, direction="in", top=True, bottom=True)

    # out_png = out_dir / f"{args.out_name}.png"
    # out_pdf = out_dir / f"{args.out_name}.pdf"
    # plt.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    # plt.savefig(out_pdf, dpi=int(args.dpi), bbox_inches="tight")
    # print(f"[OK] Saved:\n - {out_png}\n - {out_pdf}")
    plt.show()


if __name__ == "__main__":
    main()
