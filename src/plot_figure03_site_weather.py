"""Create the site-faceted weather comparison used as manuscript Figure 3."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = ROOT / "data" / "weather_gx"
OUTPUT_PNG = ROOT / "fig" / "Figure_03.png"
OUTPUT_TIFF = ROOT / "fig" / "Figure_03.tiff"

SITES = {
    "Hechi": ("N1115_duan.csv", "#0072B2"),
    "Guilin": ("N3024_guilin.csv", "#D55E00"),
    "Nanning": ("N1730_mingyang.csv", "#009E73"),
}

START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2025-01-01")
ROLLING_DAYS = 14
RAIN_DAY_THRESHOLD = 0.2


def load_daily_weather(filename: str) -> pd.DataFrame:
    hourly = pd.read_csv(WEATHER_DIR / filename)
    hourly["Datetime"] = pd.to_datetime(hourly["Datetime"])
    hourly = hourly.loc[
        (hourly["Datetime"] >= START) & (hourly["Datetime"] < END)
    ].copy()

    for column in ("TEM", "RHU", "PRE_1h"):
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")

    daily = (
        hourly.set_index("Datetime")
        .resample("D")
        .agg(
            temperature=("TEM", "mean"),
            humidity=("RHU", "mean"),
            rainfall=("PRE_1h", "sum"),
        )
        .reindex(pd.date_range(START, END - pd.Timedelta(days=1), freq="D"))
    )
    daily.index.name = "date"
    return daily


def annual_summary(daily: pd.DataFrame) -> pd.Series:
    annual = daily.groupby(daily.index.year).agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        rainfall=("rainfall", "sum"),
        rainy_days=("rainfall", lambda values: (values > RAIN_DAY_THRESHOLD).sum()),
    )
    return annual.mean()


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.94,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
        }
    )

    daily_by_site = {
        site: load_daily_weather(filename)
        for site, (filename, _) in SITES.items()
    }
    summaries = {site: annual_summary(data) for site, data in daily_by_site.items()}
    max_rain = max(data["rainfall"].max() for data in daily_by_site.values())
    rain_limit = np.ceil(max_rain / 20) * 20

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.1, 7.5),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for column, (site, (_, color)) in enumerate(SITES.items()):
        daily = daily_by_site[site]
        summary = summaries[site]
        rolling = daily[["temperature", "humidity"]].rolling(
            ROLLING_DAYS, center=True, min_periods=7
        ).mean()

        axes[0, column].set_title(site, color=color, fontweight="bold", pad=8)

        axes[0, column].plot(
            daily.index,
            daily["temperature"],
            color=color,
            alpha=0.22,
            linewidth=0.45,
            rasterized=True,
        )
        axes[0, column].plot(
            rolling.index,
            rolling["temperature"],
            color=color,
            linewidth=1.5,
            label=f"{ROLLING_DAYS}-day mean",
        )
        axes[0, column].text(
            0.97,
            0.94,
            f"Mean {summary['temperature']:.1f} °C",
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
            color=color,
            fontweight="bold",
        )

        axes[1, column].plot(
            daily.index,
            daily["humidity"],
            color=color,
            alpha=0.22,
            linewidth=0.45,
            rasterized=True,
        )
        axes[1, column].plot(
            rolling.index,
            rolling["humidity"],
            color=color,
            linewidth=1.5,
        )
        axes[1, column].text(
            0.97,
            0.94,
            f"Mean {summary['humidity']:.1f}%",
            transform=axes[1, column].transAxes,
            ha="right",
            va="top",
            color=color,
            fontweight="bold",
        )

        axes[2, column].bar(
            daily.index,
            daily["rainfall"],
            width=1.0,
            color=color,
            alpha=0.62,
            linewidth=0,
            rasterized=True,
        )
        axes[2, column].text(
            0.97,
            0.94,
            f"{summary['rainfall']:,.0f} mm per year\n"
            f"{summary['rainy_days']:.0f} rainy days per year",
            transform=axes[2, column].transAxes,
            ha="right",
            va="top",
            color=color,
            fontweight="bold",
            linespacing=1.25,
        )

        for row in range(3):
            axis = axes[row, column]
            style_axis(axis)
            add_panel_label(axis, f"({chr(97 + row * 3 + column)})")
            axis.set_xlim(START, END)
            axis.xaxis.set_major_locator(mdates.YearLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axis.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[7]))
            for boundary in pd.to_datetime(["2023-01-01", "2024-01-01"]):
                axis.axvline(boundary, color="#B8B8B8", linewidth=0.7, zorder=0)

    axes[0, 0].set_ylabel("Daily mean temperature (°C)")
    axes[1, 0].set_ylabel("Daily mean relative humidity (%)")
    axes[2, 0].set_ylabel("Daily rainfall (mm)")
    for axis in axes[2, :]:
        axis.set_xlabel("Year")

    axes[0, 0].set_ylim(0, 35)
    axes[0, 0].set_yticks([0, 10, 20, 30])
    axes[1, 0].set_ylim(20, 100)
    axes[1, 0].set_yticks([20, 40, 60, 80, 100])
    axes[2, 0].set_ylim(0, rain_limit)

    raw_handle = plt.Line2D([], [], color="#777777", alpha=0.35, linewidth=1)
    trend_handle = plt.Line2D([], [], color="#222222", linewidth=1.5)
    fig.legend(
        [raw_handle, trend_handle],
        ["Daily value", f"{ROLLING_DAYS}-day moving mean"],
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=2.8,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=800, facecolor="white")
    fig.savefig(
        OUTPUT_TIFF,
        dpi=800,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
