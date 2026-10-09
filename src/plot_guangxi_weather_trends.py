"""Plot the 2000-2024 Guangxi annual weather trend composite figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "result" / "guangxi_2000_2024_weather" / "guangxi_regional_annual_weather_stats.csv"
DEFAULT_OUTPUT = REPO_ROOT / "fig" / "guangxi_2000_2024_weather_trends"


def _trend(years: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    slope, intercept = np.polyfit(years, values, 1)
    return slope * years + intercept, float(slope * 10.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    years = data["year"].to_numpy(float)
    series = [
        ("temperature_mean", "Mean temperature (°C)", "#2166ac"),
        ("humidity_mean", "Mean relative humidity (%)", "#1b7837"),
        ("rainfall_mm", "Cumulative precipitation (mm)", "#b2182b"),
    ]
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "font.size": 8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.55), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.94, wspace=0.32)

    for index, (column, ylabel, color) in enumerate(series):
        axis = axes[index]
        values = data[column].to_numpy(float)
        fitted, slope_decade = _trend(years, values)
        axis.plot(years, values, color=color, linewidth=1.1, marker="o", markersize=2.4,
                  markerfacecolor="white", markeredgewidth=0.7, label="Annual mean")
        axis.plot(years, fitted, color="0.25", linewidth=0.8, linestyle="--", label="Linear trend")
        axis.set_ylabel(ylabel, fontsize=8)
        axis.set_xlim(1999.5, 2024.5)
        axis.set_xticks([2000, 2005, 2010, 2015, 2020, 2024])
        axis.tick_params(labelsize=7, length=3, pad=2)
        axis.grid(axis="y", color="0.85", linewidth=0.45)
        axis.text(0.03, 0.96, f"({chr(97 + index)})", transform=axis.transAxes,
                  ha="left", va="top", fontsize=9, fontweight="bold")
        axis.text(0.97, 0.96, f"trend = {slope_decade:+.2f} / decade",
                  transform=axis.transAxes, ha="right", va="top", fontsize=7)
        if index == 0:
            axis.legend(loc="lower right", frameon=False, fontsize=7, handlelength=2.2)

    axes[1].set_xlabel("Year", fontsize=8)
    output_stem = args.output.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output_stem.with_suffix(".png"))
    print(output_stem.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
