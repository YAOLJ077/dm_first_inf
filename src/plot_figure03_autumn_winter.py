"""Plot autumn-to-winter climate conditions for the three Guangxi sites.

The input files contain hourly weather from September 2021 through December
2024.  The plot uses the three complete September--February cycles:
2021--2022, 2022--2023, and 2023--2024.  Each column is a site and each row
shows a weather variable.  Lines/bars are three-cycle monthly means; ribbons
or points show the between-cycle range.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = ROOT / "data" / "weather_data" / "autumn_winter"
OUTPUT_PNG = ROOT / "fig" / "Figure_03_autumn_winter.png"

# The site order follows the manuscript site table. Colours are drawn from a
# colour-blind-safe palette and are reused across the three rows.
SITES = {
    "Hechi": ("Hechi_2021-09_to_2024-12.json", "#0072B2"),
    "Guilin": ("Guilin_2021-09_to_2024-12.json", "#D55E00"),
    "Nanning": ("Nanning_2021-09_to_2024-12.json", "#009E73"),
}

COMPLETE_CYCLES = (2021, 2022, 2023)
MONTHS = (9, 10, 11, 12, 1, 2)
MONTH_LABELS = ("Sep", "Oct", "Nov", "Dec", "Jan", "Feb")
MONTH_INDEX = {month: i for i, month in enumerate(MONTHS)}


def load_hourly_weather(filename: str) -> pd.DataFrame:
    """Read and validate one autumn--winter hourly JSON file."""

    path = WEATHER_DIR / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(data["weather_hourly"])
    required = {
        "datetime",
        "air_temp",
        "relative_humidity",
        "precip_acc_period_adjusted",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    for column in ("air_temp", "relative_humidity", "precip_acc_period_adjusted"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[["air_temp", "relative_humidity", "precip_acc_period_adjusted"]].isna().any().any():
        raise ValueError(f"{path.name} contains missing climate values")
    if (frame["precip_acc_period_adjusted"] < 0).any():
        raise ValueError(f"{path.name} contains negative precipitation")

    frame = frame.sort_values("datetime").set_index("datetime")
    frame["month"] = frame.index.month
    # September--December belong to the cycle's start year; January--February
    # belong to the preceding September's cycle year.
    frame["cycle_year"] = np.where(
        frame["month"] >= 9,
        frame.index.year,
        frame.index.year - 1,
    )
    return frame.loc[
        frame["month"].isin(MONTHS)
        & frame["cycle_year"].isin(COMPLETE_CYCLES)
    ].copy()


def monthly_cycle_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather into one record per cycle and month."""

    grouped = (
        hourly.groupby(["cycle_year", "month"], sort=True)
        .agg(
            temperature=("air_temp", "mean"),
            humidity=("relative_humidity", "mean"),
            rainfall=("precip_acc_period_adjusted", "sum"),
            hours=("air_temp", "size"),
        )
        .reset_index()
    )
    expected_hours = {
        9: 30 * 24,
        10: 31 * 24,
        11: 30 * 24,
        12: 31 * 24,
        1: 31 * 24,
        2: None,
    }
    for row in grouped.itertuples(index=False):
        expected = expected_hours[row.month]
        if row.month == 2:
            expected = (29 if row.cycle_year + 1 == 2024 else 28) * 24
        if row.hours != expected:
            raise ValueError(
                f"Incomplete hourly month for cycle {row.cycle_year}, "
                f"month {row.month}: {row.hours} rows, expected {expected}"
            )
    grouped["month_index"] = grouped["month"].map(MONTH_INDEX)
    return grouped.sort_values(["cycle_year", "month_index"]).reset_index(drop=True)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.75)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.03,
        0.95,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )


def plot_site_column(
    axes: np.ndarray,
    summary: pd.DataFrame,
    site: str,
    color: str,
    column: int,
) -> None:
    x = np.arange(len(MONTHS), dtype=float)
    by_month = summary.groupby("month_index")

    climate = {
        key: by_month[key].agg(["mean", "min", "max"])
        for key in ("temperature", "humidity", "rainfall")
    }

    for row in range(3):
        style_axis(axes[row, column])
        add_panel_label(axes[row, column], f"({chr(97 + row * 3 + column)})")

    axes[0, column].set_title(site, color=color, fontweight="bold", pad=7)
    axes[0, column].fill_between(
        x,
        climate["temperature"]["min"].to_numpy(),
        climate["temperature"]["max"].to_numpy(),
        color=color,
        alpha=0.16,
        linewidth=0,
    )
    axes[0, column].plot(x, climate["temperature"]["mean"], color=color, linewidth=1.8)
    axes[0, column].scatter(
        summary["month_index"],
        summary["temperature"],
        color=color,
        s=10,
        alpha=0.38,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    axes[1, column].fill_between(
        x,
        climate["humidity"]["min"].to_numpy(),
        climate["humidity"]["max"].to_numpy(),
        color=color,
        alpha=0.16,
        linewidth=0,
    )
    axes[1, column].plot(x, climate["humidity"]["mean"], color=color, linewidth=1.8)
    axes[1, column].scatter(
        summary["month_index"],
        summary["humidity"],
        color=color,
        s=10,
        alpha=0.38,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    axes[2, column].bar(
        x,
        climate["rainfall"]["mean"],
        width=0.68,
        color=color,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    # Thin ranges and points retain the three individual cycle totals without
    # turning the precipitation panels into an overlaid bar chart.
    axes[2, column].vlines(
        x,
        climate["rainfall"]["min"],
        climate["rainfall"]["max"],
        color="#444444",
        linewidth=0.8,
        zorder=3,
    )
    axes[2, column].scatter(
        summary["month_index"],
        summary["rainfall"],
        color="#333333",
        s=8,
        alpha=0.85,
        zorder=4,
    )

    # Use one summary for the complete September--February interval. Climate
    # means are weighted by the number of valid hours; rainfall is the mean
    # cumulative total across the three complete cycles.
    weights = summary["hours"].to_numpy(dtype=float)
    mean_temperature = float(np.average(summary["temperature"], weights=weights))
    mean_humidity = float(np.average(summary["humidity"], weights=weights))
    cumulative_rainfall = float(summary.groupby("cycle_year")["rainfall"].sum().mean())
    axes[0, column].text(
        0.97,
        0.93,
        f"Mean {mean_temperature:.1f} °C",
        transform=axes[0, column].transAxes,
        ha="right",
        va="top",
        color=color,
        fontsize=8.6,
        fontweight="bold",
    )
    axes[1, column].text(
        0.97,
        0.93,
        f"Mean {mean_humidity:.1f}% RH",
        transform=axes[1, column].transAxes,
        ha="right",
        va="top",
        color=color,
        fontsize=8.6,
        fontweight="bold",
    )
    axes[2, column].text(
        0.97,
        0.93,
        f"Σ Sep–Feb {cumulative_rainfall:,.0f} mm",
        transform=axes[2, column].transAxes,
        ha="right",
        va="top",
        color=color,
        fontsize=8.6,
        fontweight="bold",
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.75,
        }
    )

    summaries = {}
    for site, (filename, _) in SITES.items():
        summaries[site] = monthly_cycle_summary(load_hourly_weather(filename))

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.1, 5.55),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.10, "wspace": 0.07},
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.86)
    axes = np.asarray(axes)

    for column, (site, (_, color)) in enumerate(SITES.items()):
        plot_site_column(axes, summaries[site], site, color, column)

    x = np.arange(len(MONTHS))
    for axis in axes[2, :]:
        axis.set_xticks(x, MONTH_LABELS)
    for axis in axes.flat:
        axis.set_xlim(-0.5, 5.5)
    # Figure-level row labels share one x-position so the three labels align
    # cleanly at the left edge of the multi-panel figure.
    fig.text(0.028, 0.735, "Air temperature (°C)", rotation=90, ha="center", va="center", fontsize=10)
    fig.text(0.028, 0.495, "Relative humidity (%)", rotation=90, ha="center", va="center", fontsize=10)
    fig.text(0.028, 0.255, "Monthly rainfall (mm)", rotation=90, ha="center", va="center", fontsize=10)

    axes[0, 0].set_ylim(0, 32)
    axes[0, 0].set_yticks([0, 10, 20, 30])
    axes[1, 0].set_ylim(50, 100)
    axes[1, 0].set_yticks([50, 60, 70, 80, 90, 100])
    rain_max = max(
        float(summary["rainfall"].max())
        for summary in summaries.values()
    )
    rain_limit = max(250.0, np.ceil(rain_max / 50.0) * 50.0)
    axes[2, 0].set_ylim(0, rain_limit)
    axes[2, 0].set_yticks(np.arange(0, rain_limit + 1, 100))

    # The line/bar encodings represent the 3-year summary; points represent
    # the corresponding single-year values.
    mean_handle = mpl.lines.Line2D(
        [], [], color="#666666", linewidth=2.0,
        label="3-year mean (2022–2024)",
    )
    annual_point = mpl.lines.Line2D(
        [], [], marker="o", linestyle="None", color="#333333", markersize=3.5,
        label="Individual year",
    )
    fig.legend(
        [mean_handle, annual_point],
        ["3-year mean (2022–2024)", "Individual year"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.3,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=800, facecolor="white")
    plt.close(fig)

    for site, summary in summaries.items():
        weights = summary["hours"].to_numpy(dtype=float)
        mean_temperature = float(np.average(summary["temperature"], weights=weights))
        mean_humidity = float(np.average(summary["humidity"], weights=weights))
        rainfall = summary.groupby("cycle_year")["rainfall"].sum().mean()
        print(
            f"{site}: Sep-Feb mean {mean_temperature:.1f} °C, "
            f"{mean_humidity:.1f}% RH; cumulative rainfall {rainfall:.0f} mm"
        )
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
