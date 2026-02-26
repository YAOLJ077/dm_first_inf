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

plt.rcParams["font.family"] = "Times New Roman"
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
    p.add_argument("--gx-shp", type=str, default="../data/shapefile/广西壮族自治区.shp")
    p.add_argument("--city-shp", type=str, default="../data/shapefile/广西壮族自治区_市.shp")
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
    title: str,
):
    gx.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=4)
    cities.boundary.plot(ax=ax, color="gray", linewidth=0.55, linestyle="--", zorder=3)

    if agg is not None and not agg.empty:
        img = ax.scatter(
            agg["longitude"],
            agg["latitude"],
            c=agg["mean_doy"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=16,
            marker="s",
            linewidths=0.0,
            zorder=2,
        )
        stat_text = (
            f"Mean = {np.nanmean(agg['mean_doy']):.1f}\n"
            f"Median = {np.nanmedian(agg['mean_doy']):.1f}\n"
            f"Event = {np.nanmean(agg['event_rate']) * 100.0:.1f}%"
        )
        ax.text(
            0.03,
            0.02,
            stat_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"),
        )
    else:
        img = None
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=11)

    if non_gx is not None and not non_gx.empty:
        non_gx.plot(ax=ax, facecolor="white", edgecolor="none", zorder=5)
    gx.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=6)

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
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )
    ax.text(
        0.12,
        0.94,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
    )
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

    event_doy = df.loc[df["has_event"] & df["doy"].notna(), "doy"].values
    if len(event_doy) == 0:
        raise ValueError("No valid event DOY found in input data.")

    if np.isfinite(args.doy_min):
        vmin = float(args.doy_min)
    else:
        vmin = float(np.nanpercentile(event_doy, 2))
    if np.isfinite(args.doy_max):
        vmax = float(args.doy_max)
    else:
        vmax = float(np.nanpercentile(event_doy, 98))
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(event_doy)), float(np.nanmax(event_doy))

    fig = plt.figure(figsize=(7.8, 6.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3.0, 1.25], hspace=0.15, wspace=0.02)
    axes_maps = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_hist = fig.add_subplot(gs[1, :])

    cmap = cm.get_cmap("viridis").reversed()
    last_img = None
    for i, variety in enumerate(VARIETY_ORDER):
        agg = _aggregate_grid(df, variety)
        last_img = _plot_map(
            axes_maps[i],
            agg,
            gx,
            cities,
            non_gx,
            cmap,
            vmin,
            vmax,
            f"({chr(ord('a') + i)})",
            f"{variety}",
        )

    cbar_ax = fig.add_axes([0.075, 0.43, 0.85, 0.02])
    cbar = fig.colorbar(last_img, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Primary Infection Date (DOY)", fontsize=11, labelpad=4)
    cbar.ax.tick_params(labelsize=10, direction="in", top=True, bottom=True)

    bins = np.arange(int(np.floor(vmin / 10.0) * 10), int(np.ceil(vmax / 10.0) * 10) + 10, 10)
    if len(bins) < 3:
        bins = np.arange(60, 241, 10)
    centers = (bins[:-1] + bins[1:]) / 2
    width_group = 6.0
    width_single = width_group / 3.5

    for i, variety in enumerate(VARIETY_ORDER):
        vals = df.loc[
            (df["variety"] == variety) & (df["has_event"]) & (df["doy"].notna()),
            "doy",
        ].values
        counts, _ = np.histogram(vals, bins=bins)
        ax_hist.bar(
            centers + (i - 1) * width_single,
            counts,
            width=width_single,
            color=VARIETY_COLOR[variety],
            edgecolor="black",
            label=variety,
        )

    ax_hist.set_xlim(centers[0] - 10, centers[-1] + 10)
    ax_hist.set_xlabel("Day of Year (DOY)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.tick_params(axis="x", direction="in", labelsize=10, pad=2)
    ax_hist.tick_params(axis="y", direction="in", labelsize=10)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.35)
    ax_hist.set_xticks(centers)
    ax_hist.set_xticklabels([f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)], fontsize=9)

    no_event = (
        df.groupby("variety", as_index=False)
        .agg(n_total=("has_event", "count"), n_event=("has_event", "sum"))
        .assign(no_event_rate=lambda x: (1.0 - x["n_event"] / x["n_total"]) * 100.0)
    )
    txt = "\n".join(
        [f"{r.variety}: no-event {r.no_event_rate:.1f}%" for r in no_event.itertuples(index=False)]
    )
    ax_hist.text(
        0.985,
        0.95,
        txt,
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, boxstyle="round,pad=0.25"),
    )

    ax_hist.legend(
        title="Variety",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        frameon=True,
        edgecolor="black",
        fontsize=9,
        title_fontsize=9,
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
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )

    # out_png = out_dir / f"{args.out_name}.png"
    # out_pdf = out_dir / f"{args.out_name}.pdf"
    # plt.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    # plt.savefig(out_pdf, dpi=int(args.dpi), bbox_inches="tight")
    # print(f"[OK] Saved:\n - {out_png}\n - {out_pdf}")
    plt.show()


if __name__ == "__main__":
    main()
