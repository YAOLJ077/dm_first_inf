"""Plot the core Guangxi gridded FSIM symptom-window figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "result/guangxi_2000_2024_onset_windows/guangxi_2000_2024_onset_windows.csv"
DEFAULT_OUTPUT = ROOT / "fig/guangxi_2000_2024"
DEFAULT_FIG_WIDTH = 6.95
DOY_NORM = mcolors.Normalize(vmin=50, vmax=300)
PERIOD_MEAN_NORM = mcolors.Normalize(vmin=80, vmax=190)
MISSING_COLOR = "#bdbdbd"
MAP_EDGE = "#303030"
GRID_EDGE = "#d0d0d0"


def _load_boundary(path: Path | None) -> gpd.GeoDataFrame:
    if path is not None:
        return gpd.read_file(path)
    candidates = sorted((ROOT / "data/shapefile").glob("*.shp"))
    candidates = [candidate for candidate in candidates if "_市" not in candidate.stem]
    if not candidates:
        raise FileNotFoundError("No Guangxi boundary shapefile was found")
    return gpd.read_file(candidates[0])


def _prepare(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "year", "grid_id", "latitude", "longitude", "first_symptom_start_doy",
        "first_symptom_end_doy", "symptom_window_complete",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Onset table is missing columns: {sorted(missing)}")

    frame = frame.copy()
    complete = frame["symptom_window_complete"]
    if complete.dtype == bool:
        frame["complete"] = complete
    else:
        frame["complete"] = complete.astype(str).str.strip().str.lower().eq("true")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["grid_id"] = pd.to_numeric(frame["grid_id"], errors="raise").astype(int)
    for column in ("latitude", "longitude", "first_symptom_start_doy", "first_symptom_end_doy"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symptom_midpoint_doy"] = (
        frame["first_symptom_start_doy"] + frame["first_symptom_end_doy"]
    ) / 2.0
    if "symptom_duration_days" in frame:
        frame["symptom_duration_days"] = pd.to_numeric(frame["symptom_duration_days"], errors="coerce")
    else:
        frame["symptom_duration_days"] = (
            frame["first_symptom_end_doy"] - frame["first_symptom_start_doy"]
        )

    if "first_symptom_start_date" in frame:
        start_dates = pd.to_datetime(frame["first_symptom_start_date"], errors="coerce")
        frame["symptom_month"] = start_dates.dt.month
    else:
        frame["symptom_month"] = np.nan
    complete_frame = frame.loc[frame["complete"]].copy()
    grids = (
        frame[["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id")
        .sort_values(["latitude", "longitude", "grid_id"])
        .reset_index(drop=True)
    )
    years = sorted(frame["year"].unique().tolist())
    return frame, complete_frame, grids, years


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _map_axes(ax: plt.Axes, boundary: gpd.GeoDataFrame, *, labels: bool = True) -> None:
    boundary.boundary.plot(ax=ax, color=MAP_EDGE, linewidth=0.45, zorder=5)
    minx, miny, maxx, maxy = boundary.total_bounds
    ax.set_xlim(minx - 0.08, maxx + 0.08)
    ax.set_ylim(miny - 0.08, maxy + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID_EDGE, linewidth=0.25, alpha=0.45, zorder=0)
    ax.tick_params(labelsize=6, length=2, pad=1)
    if labels:
        ax.set_xlabel("Longitude (deg E)", fontsize=7, labelpad=1)
        ax.set_ylabel("Latitude (deg N)", fontsize=7, labelpad=1)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")


def _scatter_values(
    ax: plt.Axes,
    grids: pd.DataFrame,
    values: pd.Series,
    *,
    cmap: str | mcolors.Colormap,
    norm: mcolors.Normalize | None = None,
    size: float = 9,
    cell_edgecolor: str = "white",
    cell_linewidth: float = 0.12,
    cell_scale: float = 0.96,
) -> PatchCollection:
    values = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=grids.index)
    missing = values.isna()
    longitudes = grids["longitude"].to_numpy(dtype=float)
    latitudes = grids["latitude"].to_numpy(dtype=float)
    unique_longitudes = np.sort(np.unique(longitudes))
    unique_latitudes = np.sort(np.unique(latitudes))
    dx = float(np.median(np.diff(unique_longitudes))) if len(unique_longitudes) > 1 else 0.25
    dy = float(np.median(np.diff(unique_latitudes))) if len(unique_latitudes) > 1 else 0.25
    cell_width = dx * cell_scale
    cell_height = dy * cell_scale

    def make_patches(mask: pd.Series) -> list[Rectangle]:
        return [
            Rectangle(
                (longitudes[index] - cell_width / 2, latitudes[index] - cell_height / 2),
                cell_width,
                cell_height,
            )
            for index in np.flatnonzero(mask.to_numpy())
        ]

    if missing.any():
        missing_collection = PatchCollection(
            make_patches(missing),
            facecolor=MISSING_COLOR,
            edgecolor=cell_edgecolor,
            linewidth=cell_linewidth,
            zorder=2,
        )
        ax.add_collection(missing_collection)

    valid = ~missing
    collection = PatchCollection(
        make_patches(valid),
        cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        norm=norm,
        edgecolor=cell_edgecolor,
        linewidth=cell_linewidth,
        zorder=3,
    )
    collection.set_array(values.loc[valid].to_numpy(dtype=float))
    ax.add_collection(collection)
    return collection


def plot_annual_maps(complete: pd.DataFrame, grids: pd.DataFrame, years: list[int], boundary: gpd.GeoDataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(DEFAULT_FIG_WIDTH, 7.35))
    layout = fig.add_gridspec(
        5,
        5,
        left=0.035,
        right=0.91,
        bottom=0.045,
        top=0.945,
        wspace=0.0,
        hspace=0.045,
    )
    axes = np.asarray([[fig.add_subplot(layout[row, column]) for column in range(5)] for row in range(5)])
    mappable = None
    for index, (ax, year) in enumerate(zip(axes.flat, years)):
        current = complete.loc[complete["year"].eq(year)].groupby("grid_id")["symptom_midpoint_doy"].median()
        values = grids["grid_id"].map(current)
        _map_axes(ax, boundary, labels=index // 5 == 4 or index % 5 == 0)
        if index % 5 != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if index // 5 != 4:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        mappable = _scatter_values(ax, grids, values, cmap="viridis_r", norm=DOY_NORM, size=7)
        ax.set_title(str(year), fontsize=8, pad=2)
    for ax in axes.flat[len(years):]:
        ax.set_visible(False)
    fig.canvas.draw()
    map_top = axes[0, 0].get_position().y1
    map_bottom = axes[-1, 0].get_position().y0
    cax = fig.add_axes([0.935, map_bottom, 0.014, map_top - map_bottom])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("First symptom date (DOY)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Annual first symptom date across Guangxi", fontsize=10, y=0.985)
    _save(fig, output, "01_annual_first_symptom_maps.png")


def plot_median_iqr(complete: pd.DataFrame, grids: pd.DataFrame, boundary: gpd.GeoDataFrame, output: Path) -> None:
    grouped = complete.groupby("grid_id")["symptom_midpoint_doy"]
    median = grids["grid_id"].map(grouped.median())
    iqr = grids["grid_id"].map(grouped.quantile(0.75) - grouped.quantile(0.25))
    fig = plt.figure(figsize=(DEFAULT_FIG_WIDTH, 3.05))
    layout = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 0.065, 1, 0.065],
        left=0.045,
        right=0.965,
        bottom=0.12,
        top=0.88,
        wspace=0.14,
    )
    axes = [fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 2])]
    colorbar_axes = [fig.add_subplot(layout[0, 1]), fig.add_subplot(layout[0, 3])]
    for index, (ax, cax, values, title, cmap, norm, label) in enumerate((
        (axes[0], colorbar_axes[0], median, "Median first symptom date", "viridis_r", DOY_NORM, "DOY"),
        (axes[1], colorbar_axes[1], iqr, "Interannual IQR", "magma_r", mcolors.Normalize(0, 60), "Days"),
    )):
        _map_axes(ax, boundary, labels=index == 0)
        if index == 1:
            ax.set_yticklabels([])
            ax.set_xlabel("Longitude (deg E)", fontsize=7, labelpad=1)
        mappable = _scatter_values(ax, grids, values, cmap=cmap, norm=norm, size=10)
        ax.set_title(title, fontsize=9, pad=3)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.set_title(label, fontsize=7, pad=2)
        cbar.ax.tick_params(labelsize=6)
    fig.suptitle("Typical timing and interannual variability", fontsize=10, y=0.965)
    _save(fig, output, "02_median_and_iqr_maps.png")


def plot_grid_year_heatmap(complete: pd.DataFrame, grids: pd.DataFrame, years: list[int], output: Path) -> None:
    order = grids["grid_id"].tolist()
    table = complete.pivot_table(index="grid_id", columns="year", values="symptom_midpoint_doy", aggfunc="median")
    table = table.reindex(index=order, columns=years)
    masked = np.ma.masked_invalid(table.to_numpy(dtype=float))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(MISSING_COLOR)
    fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 5.25))
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=DOY_NORM)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Grid point (south to north)", fontsize=8)
    ax.set_xticks(np.arange(len(years)), labels=[str(year) for year in years], rotation=60, ha="right", fontsize=6)
    tick_positions = np.arange(0, len(order), 30)
    ax.set_yticks(tick_positions, labels=[str(order[pos]) for pos in tick_positions], fontsize=6)
    ax.set_title("Grid point by year: first symptom date", fontsize=10, pad=6)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("First symptom date (DOY)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, output, "04_grid_year_first_symptom_heatmap.png")


def _profile_table(complete: pd.DataFrame, key: str, years: list[int]) -> pd.DataFrame:
    return complete.pivot_table(index=key, columns="year", values="symptom_midpoint_doy", aggfunc="median").reindex(columns=years).sort_index()


def plot_lat_lon_profiles(complete: pd.DataFrame, years: list[int], output: Path) -> None:
    lat = _profile_table(complete, "latitude", years)
    lon = _profile_table(complete, "longitude", years)
    fig, axes = plt.subplots(1, 2, figsize=(DEFAULT_FIG_WIDTH, 3.35), squeeze=False)
    for ax, table, label in zip(axes[0], (lat, lon), ("Latitude (deg N)", "Longitude (deg E)")):
        image = ax.imshow(
            np.ma.masked_invalid(table.to_numpy(dtype=float)),
            aspect="auto", interpolation="nearest", cmap="viridis", norm=DOY_NORM,
            extent=[years[0] - 0.5, years[-1] + 0.5, table.index.min(), table.index.max()],
        )
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks(years[::4], labels=[str(year) for year in years[::4]], rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"{label.split(' ')[0]}-time profile", fontsize=9, pad=3)
    cbar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.028, pad=0.02)
    cbar.set_label("First symptom date (DOY)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Latitude and longitude profiles of symptom timing", fontsize=10, y=1.01)
    _save(fig, output, "05_latitude_longitude_time_profiles.png")


def _sen_trend(complete: pd.DataFrame, grids: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grid_id, group in complete.groupby("grid_id"):
        group = group.sort_values("year")
        years = group["year"].to_numpy(dtype=float)
        values = group["symptom_midpoint_doy"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = (values[None, :] - values[:, None]) / (years[None, :] - years[:, None])
        slopes = slopes[np.isfinite(slopes) & ~np.eye(len(values), dtype=bool)]
        slope = float(np.median(slopes)) if len(slopes) else np.nan
        tau, p_value = kendalltau(years, values, nan_policy="omit") if len(values) >= 4 else (np.nan, np.nan)
        rows.append({"grid_id": grid_id, "slope_days_per_decade": slope * 10, "tau": tau, "p_value": p_value, "n_years": len(values)})
    return grids.merge(pd.DataFrame(rows), on="grid_id", how="left")


def plot_trends(frame: pd.DataFrame, complete: pd.DataFrame, grids: pd.DataFrame, years: list[int], boundary: gpd.GeoDataFrame, output: Path) -> None:
    trends = _sen_trend(complete, grids)
    annual = complete.groupby("year")["symptom_midpoint_doy"].agg(
        median="median", q10=lambda x: x.quantile(0.10), q90=lambda x: x.quantile(0.90), n="count"
    ).reindex(years)
    total = frame.groupby("year").size().reindex(years, fill_value=0)
    # Use the full table for the denominator, so no-symptom grid-years remain
    # visible in the annual occurrence fraction.
    occurrence = annual["n"] / total
    fig, axes = plt.subplots(1, 2, figsize=(DEFAULT_FIG_WIDTH, 3.25), squeeze=False)
    axes = axes[0]
    ax = axes[0]
    _map_axes(ax, boundary, labels=True)
    trend_norm = mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    mappable = _scatter_values(ax, grids, trends["slope_days_per_decade"], cmap="RdBu_r", norm=trend_norm, size=10)
    significant = trends["p_value"].lt(0.05) & trends["slope_days_per_decade"].notna()
    ax.scatter(trends.loc[significant, "longitude"], trends.loc[significant, "latitude"], s=13, facecolors="none", edgecolors="black", linewidths=0.45, zorder=6)
    ax.set_title("Sen slope of first symptom date", fontsize=9, pad=3)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Days per decade", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None", markersize=4, label="Mann-Kendall p < 0.05")], loc="lower left", fontsize=6, frameon=True, framealpha=0.85)

    ax = axes[1]
    ax.fill_between(years, annual["q10"], annual["q90"], color="#9ecae1", alpha=0.55, label="Spatial 10th-90th percentile")
    ax.plot(years, annual["median"], color="#08519c", marker="o", markersize=2.6, linewidth=1.1, label="Spatial median")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("First symptom date (DOY)", fontsize=8)
    ax.set_xticks(years[::4], labels=[str(year) for year in years[::4]], rotation=45, ha="right", fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(color=GRID_EDGE, linewidth=0.3, alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(years, occurrence * 100, color="#b30000", linewidth=0.9, linestyle="--", label="Grid points with a window")
    ax2.set_ylabel("Grid points with a window (%)", fontsize=7, color="#7f0000")
    ax2.tick_params(axis="y", labelsize=6, colors="#7f0000")
    ax.set_title("Annual symptom timing and coverage", fontsize=9, pad=3)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=6, frameon=True, framealpha=0.85, loc="best")
    _save(fig, output, "06_symptom_date_trends_and_annual_summary.png")


def plot_risk_months(complete: pd.DataFrame, grids: pd.DataFrame, boundary: gpd.GeoDataFrame, output: Path) -> None:
    months = [3, 4, 5, 6]
    month_probability = pd.DataFrame(index=grids.index, columns=months, dtype=float)
    for month in months:
        month_probability[month] = grids["grid_id"].map(
            complete.assign(is_month=complete["symptom_month"].eq(month)).groupby("grid_id")["is_month"].mean()
        ).to_numpy()
    dominant = complete.groupby("grid_id")["symptom_month"].agg(lambda values: int(values.dropna().astype(int).value_counts().sort_index().idxmax()))
    dominant = grids["grid_id"].map(dominant)
    regional = complete["symptom_month"].value_counts(normalize=True).sort_index()
    month_labels = {2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}

    fig = plt.figure(figsize=(DEFAULT_FIG_WIDTH, 7.0))
    axes = [fig.add_subplot(2, 3, index + 1) for index in range(4)]
    probability_norm = mcolors.Normalize(0, 1)
    mappable = None
    for ax, month in zip(axes, months):
        _map_axes(ax, boundary, labels=month in (3, 5))
        mappable = _scatter_values(ax, grids, month_probability[month], cmap="YlOrRd", norm=probability_norm, size=8)
        ax.set_title(f"{month_labels[month]} probability", fontsize=8, pad=2)
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Probability", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(2, 3, 5)
    month_colors = ["#2166ac", "#67a9cf", "#fdae61", "#d73027", "#762a83", "#1b7837", "#8c510a", "#5e3c99", "#969696"]
    cmap = mcolors.ListedColormap(month_colors)
    norm = mcolors.BoundaryNorm(np.arange(1.5, 11.5, 1), cmap.N)
    _map_axes(ax, boundary, labels=True)
    _scatter_values(ax, grids, dominant, cmap=cmap, norm=norm, size=10)
    ax.set_title("Most frequent symptom month", fontsize=8, pad=2)
    legend_handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=5, label=month_labels[month]) for month, color in zip(range(2, 11), month_colors)]
    ax.legend(handles=legend_handles, ncol=3, fontsize=5.5, loc="lower left", frameon=True, framealpha=0.85, handletextpad=0.2, columnspacing=0.5)

    ax = fig.add_subplot(2, 3, 6)
    all_months = sorted(int(value) for value in complete["symptom_month"].dropna().unique())
    ax.plot(all_months, regional.reindex(all_months, fill_value=0) * 100, color="#8c2d04", marker="o", markersize=2.8, linewidth=1.1)
    ax.set_xticks(all_months, labels=[month_labels.get(month, str(month)) for month in all_months], rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("Month", fontsize=8)
    ax.set_ylabel("Probability (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(color=GRID_EDGE, linewidth=0.3, alpha=0.6)
    ax.set_title("Guangxi-wide symptom-month probability", fontsize=8, pad=2)
    fig.suptitle("Seasonal distribution of first symptom dates", fontsize=10, y=0.995)
    _save(fig, output, "07_symptom_month_risk_maps_and_curve.png")


def plot_period_means(complete: pd.DataFrame, grids: pd.DataFrame, years: list[int], boundary: gpd.GeoDataFrame, output: Path) -> None:
    """Plot three period means and their distribution, following manuscript Figs. 12-13."""
    period_arrays = [chunk for chunk in np.array_split(np.asarray(years, dtype=int), 3) if len(chunk)]
    periods = [(int(chunk[0]), int(chunk[-1]), chunk.tolist()) for chunk in period_arrays]
    period_colors = ["#1b9e77", "#d95f02", "#7570b3"]
    period_values: list[pd.Series] = []

    fig = plt.figure(figsize=(DEFAULT_FIG_WIDTH, 5.45))
    layout = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 0.11, 0.92],
        left=0.055,
        right=0.95,
        bottom=0.105,
        top=0.90,
        wspace=0.07,
        hspace=0.28,
    )
    map_axes = [fig.add_subplot(layout[0, column]) for column in range(3)]
    cax = fig.add_subplot(layout[1, :])
    bar_ax = fig.add_subplot(layout[2, :])
    mappable = None

    for index, (ax, (start, end, period_years)) in enumerate(zip(map_axes, periods)):
        values = (
            complete.loc[complete["year"].isin(period_years)]
            .groupby("grid_id")["symptom_midpoint_doy"]
            .mean()
        )
        values = grids["grid_id"].map(values)
        period_values.append(values)
        _map_axes(ax, boundary, labels=index == 0)
        ax.set_xlabel("")
        if index > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        mappable = _scatter_values(
            ax,
            grids,
            values,
            cmap="viridis_r",
            norm=PERIOD_MEAN_NORM,
            size=10,
            cell_edgecolor="none",
            cell_linewidth=0.0,
            cell_scale=1.0,
        )
        ax.set_title(f"{start}-{end}", fontsize=8.5, pad=3)
        valid_values = values.dropna()
        ax.text(
            0.03,
            0.04,
            f"Mean = {valid_values.mean():.1f}\nMedian = {valid_values.median():.1f}",
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
            zorder=7,
        )

    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("Mean representative symptom date (DOY)", fontsize=7.5, labelpad=2)
    cbar.set_ticks(np.arange(80, 191, 20))
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)

    bins = np.arange(80, 191, 10)
    centers = (bins[:-1] + bins[1:]) / 2
    bar_width = 2.8
    for values, color, (start, end, _) in zip(period_values, period_colors, periods):
        counts, _ = np.histogram(values.dropna().to_numpy(dtype=float), bins=bins)
        offset = (period_colors.index(color) - 1) * bar_width
        bar_ax.bar(
            centers + offset,
            counts,
            width=bar_width * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            label=f"{start}-{end}",
        )
    bar_ax.set_xticks(centers, labels=[f"{int(left)}-{int(right)}" for left, right in zip(bins[:-1], bins[1:])], rotation=45, ha="right", fontsize=6)
    bar_ax.set_xlabel("Period mean symptom date (DOY)", fontsize=8)
    bar_ax.set_ylabel("Grid-cell count", fontsize=8)
    bar_ax.tick_params(axis="y", labelsize=6)
    bar_ax.grid(axis="y", color=GRID_EDGE, linewidth=0.3, alpha=0.6)
    bar_ax.set_axisbelow(True)
    bar_ax.legend(fontsize=6, frameon=True, framealpha=0.9, ncol=3, loc="upper right")
    fig.suptitle("Spatial and temporal variation in mean symptom timing", fontsize=10, y=0.965)
    _save(fig, output, "08_period_mean_symptom_date_maps_and_distribution.png")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--boundary", type=Path, default=None)
    args = parser.parse_args(argv)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
        "axes.titlesize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    frame, complete, grids, years = _prepare(args.input)
    boundary = _load_boundary(args.boundary)
    plot_annual_maps(complete, grids, years, boundary, args.output_dir)
    plot_median_iqr(complete, grids, boundary, args.output_dir)
    plot_period_means(complete, grids, years, boundary, args.output_dir)
    plot_grid_year_heatmap(complete, grids, years, args.output_dir)
    plot_lat_lon_profiles(complete, years, args.output_dir)
    plot_trends(frame, complete, grids, years, boundary, args.output_dir)
    plot_risk_months(complete, grids, boundary, args.output_dir)
    print(f"Plotted {len(years)} years and {len(grids)} grid points from {args.input}")
    print(f"Complete symptom windows: {len(complete)} / {len(frame)}")
    print(f"Wrote PNG figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
