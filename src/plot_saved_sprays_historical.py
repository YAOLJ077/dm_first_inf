#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
        description="Compute and plot 2000-2024 mean saved sprays per grid from historical infection inversion output."
    )
    p.add_argument(
        "--infection-csv",
        type=str,
        default="../infection_date_2000_2024/infection_inversion_all_varieties_2000_2024.csv",
    )
    p.add_argument("--management-xlsx", type=str, default="data/management_data.xlsx")
    p.add_argument("--gx-shp", type=str, default="data/shapefile/广西壮族自治区.shp")
    p.add_argument("--city-shp", type=str, default="data/shapefile/广西壮族自治区_市.shp")
    p.add_argument("--out-table", type=str, default="../infection_date_2000_2024/saved_sprays_grid_mean_2000_2024.csv")
    p.add_argument("--out-detail", type=str, default="../infection_date_2000_2024/saved_sprays_row_level_2000_2024.csv")
    p.add_argument("--out-dir", type=str, default="../fig")
    p.add_argument("--out-name", type=str, default="historical_saved_sprays_gx")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--bbox-buffer", type=float, default=0.5)
    p.add_argument("--year-min", type=int, default=2000)
    p.add_argument("--year-max", type=int, default=2024)
    return p.parse_args()


def _resolve_path(project_root: Path, p: str) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    return (project_root / pp).resolve()


def _load_mgmt_doys(xlsx_path: Path) -> List[int]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Management file not found: {xlsx_path}")
    mgmt_df = pd.read_excel(xlsx_path)
    doy_col = None
    for c in mgmt_df.columns:
        if str(c).strip().lower() == "doy":
            doy_col = c
            break
    if doy_col is None:
        raise KeyError(f"No 'Doy' column found in {xlsx_path}")
    doys = sorted(pd.to_numeric(mgmt_df[doy_col], errors="coerce").dropna().astype(int).unique().tolist())
    if not doys:
        raise ValueError(f"No valid DOY values in {xlsx_path}")
    return doys


def _count_saved_sprays(infection_doy: float, mgmt_doys: List[int]) -> int:
    if pd.isna(infection_doy) or float(infection_doy) <= 0:
        return 0
    threshold = float(infection_doy) - 7.0
    return int(sum(d <= threshold for d in mgmt_doys))


def _prepare_rows(infection_csv: Path, mgmt_doys: List[int], year_min: int, year_max: int) -> pd.DataFrame:
    if not infection_csv.exists():
        raise FileNotFoundError(f"Infection inversion csv not found: {infection_csv}")
    df = pd.read_csv(infection_csv)
    required = {"variety", "year", "grid_id", "latitude", "longitude", "oil_spot_start"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in infection csv: {miss}")

    out = df.copy()
    out["variety"] = out["variety"].astype(str).str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)
    out = out[(out["year"] >= int(year_min)) & (out["year"] <= int(year_max))].copy()

    out["oil_spot_start"] = pd.to_datetime(out["oil_spot_start"], errors="coerce")
    out["infection_doy"] = out["oil_spot_start"].dt.dayofyear
    out["saved_sprays"] = out["infection_doy"].apply(lambda x: _count_saved_sprays(x, mgmt_doys))
    return out


def _aggregate_grid_mean(rows: pd.DataFrame) -> pd.DataFrame:
    agg = (
        rows.groupby(["variety", "grid_id", "latitude", "longitude"], as_index=False)
        .agg(
            n_years=("year", "count"),
            mean_saved_sprays=("saved_sprays", "mean"),
            median_saved_sprays=("saved_sprays", "median"),
            min_saved_sprays=("saved_sprays", "min"),
            max_saved_sprays=("saved_sprays", "max"),
        )
        .sort_values(["variety", "latitude", "longitude"])
    )
    return agg


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
    panel_title: str,
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
            agg.pivot(index="latitude", columns="longitude", values="mean_saved_sprays")
            .reindex(index=lat_u, columns=lon_u)
            .to_numpy()
        )
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
            f"Mean = {np.nanmean(agg['mean_saved_sprays']):.2f}\n"
            f"Median = {np.nanmedian(agg['mean_saved_sprays']):.2f}"
        )
    else:
        img = None
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12)

    if non_gx is not None and not non_gx.empty:
        non_gx.plot(ax=ax, facecolor="white", edgecolor="none", zorder=5)
    gx.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=6)

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
        0.955,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        zorder=40,
        clip_on=False,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )
    ax.text(
        0.12,
        0.94,
        panel_title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        zorder=40,
        clip_on=False,
        bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
    )
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

    infection_csv = _resolve_path(project_root, args.infection_csv)
    mgmt_xlsx = _resolve_path(project_root, args.management_xlsx)
    gx_shp = _resolve_path(project_root, args.gx_shp)
    city_shp = _resolve_path(project_root, args.city_shp)
    out_table = _resolve_path(project_root, args.out_table)
    out_detail = _resolve_path(project_root, args.out_detail)
    out_dir = _resolve_path(project_root, args.out_dir)

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    mgmt_doys = _load_mgmt_doys(mgmt_xlsx)
    rows = _prepare_rows(infection_csv, mgmt_doys, int(args.year_min), int(args.year_max))
    agg = _aggregate_grid_mean(rows)

    rows.to_csv(out_detail, index=False)
    agg.to_csv(out_table, index=False)
    print(f"[OK] Saved detail: {out_detail}")
    print(f"[OK] Saved grid mean: {out_table}")

    gx, cities = _load_boundaries(gx_shp, city_shp)
    non_gx = _build_non_gx_polygon(gx, float(args.bbox_buffer))

    map_data: Dict[str, pd.DataFrame] = {
        v: agg[agg["variety"] == v].copy() for v in VARIETY_ORDER
    }
    all_vals = np.concatenate([
        map_data[v]["mean_saved_sprays"].to_numpy()
        for v in VARIETY_ORDER
        if not map_data[v].empty
    ]) if any(not map_data[v].empty for v in VARIETY_ORDER) else np.array([])
    if all_vals.size == 0:
        raise ValueError("No values available for plotting mean saved sprays.")

    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3.05, 1.2], hspace=0.12, wspace=0.02)
    axes_maps = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_hist = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.09)

    cmap = cm.get_cmap("YlGnBu")
    last_img = None
    for i, variety in enumerate(VARIETY_ORDER):
        last_img = _plot_map(
            axes_maps[i],
            map_data[variety],
            gx,
            cities,
            non_gx,
            cmap,
            vmin,
            vmax,
            f"({chr(ord('a') + i)})",
            variety,
        )

    # Histogram over grid-level mean saved sprays:
    # fixed integer bins starting from 2.
    bmin = 2
    bmax = int(np.ceil(np.nanmax(all_vals)))
    bmax = min(bmax, 6)
    if bmax <= bmin:
        bmax = bmin + 1
    bins = np.arange(bmin - 0.5, bmax + 1.5, 1.0)  # integer-centered bins
    centers = np.arange(bmin, bmax + 1, 1.0)
    width_group = 0.75
    width_single = width_group / 3.5

    for i, variety in enumerate(VARIETY_ORDER):
        vals = pd.to_numeric(map_data[variety].get("mean_saved_sprays", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        counts, _ = np.histogram(vals, bins=bins)
        ax_hist.bar(
            centers + (i - 1) * width_single,
            counts,
            width=width_single,
            color=VARIETY_COLOR[variety],
            edgecolor="black",
            label=variety,
        )

    ax_hist.set_xlim(float(bmin - 0.5), float(bmax + 0.5))
    ax_hist.set_xlabel("Number of Sprays Saved", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.tick_params(axis="x", direction="in", labelsize=11, pad=2)
    ax_hist.tick_params(axis="y", direction="in", labelsize=12)
    ax_hist.set_ylim(0, 250)
    ax_hist.set_yticks([0, 50, 100, 150, 200, 250])
    ax_hist.grid(axis="y", linestyle="--", alpha=0.35)
    ax_hist.set_xticks(centers)
    ax_hist.set_xticklabels([f"{int(x)}" for x in centers], fontsize=11)
    for lbl in ax_hist.get_xticklabels():
        lbl.set_fontsize(11)
        lbl.set_fontname("Times New Roman")

    ax_hist.legend(
        title="Variety",
        loc="upper right",
        bbox_to_anchor=(0.999, 0.98),
        frameon=True,
        edgecolor="black",
        fontsize=11,
        title_fontsize=11,
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

    # Align bottom panel and colorbar to top-row width.
    fig.canvas.draw()
    pos_l = axes_maps[0].get_position()
    pos_r = axes_maps[-1].get_position()
    pos_hist = ax_hist.get_position()
    ax_hist.set_position([pos_l.x0, pos_hist.y0 + 0.025, pos_r.x1 - pos_l.x0, pos_hist.height])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bbox = ax_hist.yaxis.get_label().get_window_extent(renderer=renderer)
    label_left = label_bbox.x0 / fig.bbox.width
    dx = pos_l.x0 - label_left
    if dx > 0:
        pos_hist = ax_hist.get_position()
        ax_hist.set_position([pos_hist.x0 + dx, pos_hist.y0, pos_hist.width - dx, pos_hist.height])

    fig.canvas.draw()
    pos_hist = ax_hist.get_position()
    map_bottom = pos_l.y0
    hist_top = pos_hist.y1
    cbar_h = 0.018
    cbar_y_target = map_bottom - cbar_h - 0.025
    cbar_y = max(hist_top + 0.001, cbar_y_target)
    cbar_ax = fig.add_axes([pos_l.x0, cbar_y, pos_r.x1 - pos_l.x0, cbar_h])
    cbar = fig.colorbar(last_img, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Number of Sprays Saved (model-guided vs. schedule)", fontsize=12, labelpad=4)
    cbar.ax.tick_params(labelsize=12, direction="in", top=True, bottom=True)

    out_png = out_dir / f"{args.out_name}.png"
    out_pdf = out_dir / f"{args.out_name}.pdf"
    plt.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.savefig(out_pdf, dpi=int(args.dpi), bbox_inches="tight")
    print(f"[OK] Saved figure:\n - {out_png}\n - {out_pdf}")
    plt.show()


if __name__ == "__main__":
    main()
