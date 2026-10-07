"""Plot disease onset, airborne sporangia and phenology for Guangxi site-years.

The figure uses only the three local audit workbooks. Numeric zero spore counts
are plotted as observed zeroes, while blank cells remain visibly missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONSET = ROOT / "data/disease_data/grape_downy_mildew_data.xlsx"
DEFAULT_PHENOLOGY = ROOT / "data/phenology_data/phenology.xlsx"
DEFAULT_SPORE = ROOT / "data/spore_data/guangxi_site_years_spore_data.xlsx"
DEFAULT_OUTPUT = ROOT / "fig/data_audit/guangxi_site_year_onset_spore_phenology.png"

COORD_TOLERANCE = 0.01
PLOT_YEAR = 2000  # leap year so every month/day, including Feb 29, is representable
COLORS = {
    "early": "#E69F00",
    "middle": "#009E73",
    "late": "#7B61A8",
    "spore": "#377EB8",
    "onset": "#D55E00",
    "missing": "#8C8C8C",
    "text": "#263238",
}


def _read_onset(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    required = {"Latitude", "Longitude", "Year", "Date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Disease workbook is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame["Latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce")
    frame["Longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce")
    frame = frame.dropna(subset=["Latitude", "Longitude", "Year", "Date"]).copy()
    frame["Year"] = frame["Year"].astype(int)
    if frame.duplicated(["Latitude", "Longitude", "Year"]).any():
        raise ValueError("Disease workbook contains duplicate site-year records")
    return frame.sort_values(["Latitude", "Longitude", "Year"]).reset_index(drop=True)


def _read_phenology(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    required = {"latitude", "longitude", "year", "date", "bbch"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Phenology workbook is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("latitude", "longitude", "year", "bbch"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=list(required)).copy()


def _read_spore(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="CK_Data")
    required = {"latitude", "longitude", "Year", "Date", "Spore"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Spore workbook is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["Spore"] = pd.to_numeric(frame["Spore"], errors="coerce")
    frame = frame.dropna(subset=["latitude", "longitude", "Year", "Date"]).copy()
    # Add the common matching names while retaining workbook column names for
    # the plotting logic and audit traceability.
    frame["year"] = frame["Year"].astype(int)
    return frame


def _match(frame: pd.DataFrame, *, lat: float, lon: float, year: int, prefix: str) -> pd.DataFrame:
    return frame.loc[
        frame[f"{prefix}year"].eq(year)
        & frame[f"{prefix}latitude"].sub(lat).abs().le(COORD_TOLERANCE)
        & frame[f"{prefix}longitude"].sub(lon).abs().le(COORD_TOLERANCE)
    ].copy()


def _stage_summary(phenology: pd.DataFrame) -> dict:
    phenology = phenology.sort_values("date")
    minimum = phenology.iloc[0]
    middle = phenology.loc[phenology["bbch"] >= 53]
    final = phenology.iloc[-1]
    return {
        "minimum_date": minimum["date"],
        "minimum_bbch": int(minimum["bbch"]),
        "middle_date": middle["date"].min() if not middle.empty else pd.NaT,
        "middle_bbch": int(middle.iloc[0]["bbch"]) if not middle.empty else None,
        "final_date": final["date"],
        "final_bbch": int(final["bbch"]),
    }


def _plot_date(value) -> pd.Timestamp:
    """Map a calendar date to the shared reference year used by every panel."""

    stamp = pd.Timestamp(value)
    return pd.Timestamp(PLOT_YEAR, stamp.month, stamp.day)


def _span(ax, start, end, color: str, label: str) -> None:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return
    ax.axvspan(start, end, color=color, alpha=0.13, lw=0, label=label, zorder=0)


def _plot_site(ax, onset_row, phenology, spores, x_start, x_end) -> None:
    lat = float(onset_row["Latitude"])
    lon = float(onset_row["Longitude"])
    year = int(onset_row["Year"])
    onset_label = pd.Timestamp(onset_row["Date"])
    onset = _plot_date(onset_label)
    stages = _stage_summary(phenology)
    stages = {
        **stages,
        "minimum_date": _plot_date(stages["minimum_date"]),
        "middle_date": _plot_date(stages["middle_date"]) if not pd.isna(stages["middle_date"])
        else pd.NaT,
        "final_date": _plot_date(stages["final_date"]),
    }
    spores = spores.copy()
    if not spores.empty:
        spores["Date"] = spores["Date"].map(_plot_date)

    ax.set_xlim(x_start, x_end)

    positive = spores.loc[spores["Spore"].notna() & spores["Spore"].gt(0)]
    zeros = spores.loc[spores["Spore"].notna() & spores["Spore"].eq(0)]
    missing = spores.loc[spores["Spore"].isna()]
    ymax = max(1.0, float(positive["Spore"].max()) if not positive.empty else 1.0)
    ax.set_ylim(0, ymax * 1.28)

    _span(ax, stages["minimum_date"], stages["middle_date"], COLORS["early"], "earliest to BBCH 53+")
    _span(ax, stages["middle_date"], stages["final_date"], COLORS["middle"], "BBCH 53+ to final")
    _span(ax, stages["final_date"], x_end, COLORS["late"], "final to axis end")

    if not positive.empty:
        ax.bar(
            positive["Date"], positive["Spore"], width=1.8,
            color=COLORS["spore"], alpha=0.82, edgecolor="white", linewidth=0.35,
            zorder=3,
        )
    if not zeros.empty:
        ax.scatter(
            zeros["Date"], [0.0] * len(zeros), s=30,
            facecolors="white", edgecolors=COLORS["spore"], linewidths=1.0,
            zorder=5,
        )
    if not missing.empty:
        ax.scatter(
            missing["Date"], [ymax * 0.035] * len(missing), s=10,
            marker="x", color=COLORS["missing"], alpha=0.65, linewidths=0.7,
            zorder=4,
        )

    ax.axvline(onset, color=COLORS["onset"], lw=1.5, ls="--", zorder=6)
    ax.scatter([onset], [ymax * 1.16], marker="v", color=COLORS["onset"], s=26, zorder=7)

    for date, text, color in (
        (stages["minimum_date"], f"min BBCH {stages['minimum_bbch']}", COLORS["early"]),
        (stages["middle_date"], f"BBCH {stages['middle_bbch']}+", COLORS["middle"]),
        (stages["final_date"], f"final BBCH {stages['final_bbch']}", COLORS["late"]),
    ):
        if pd.isna(date):
            continue
        ax.axvline(date, color=color, lw=0.8, alpha=0.75, zorder=2)
        ax.text(
            date, 0.98, text, rotation=90, ha="right", va="top", fontsize=7,
            color=color, transform=ax.get_xaxis_transform(), clip_on=True,
        )

    city = onset_row.get("City", "")
    city = "" if pd.isna(city) else str(city)
    label = f"{city} | {lat:.2f}N, {lon:.2f}E | {year}" if city else f"{lat:.2f}N, {lon:.2f}E | {year}"
    ax.set_title(label, loc="left", fontsize=10, fontweight="bold", color=COLORS["text"])
    ax.text(
        0.995, 0.96, f"onset {onset_label:%b %d}", transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5, color=COLORS["onset"], fontweight="bold",
    )
    ax.grid(axis="y", color="#D9DEE3", lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Spore count" if not spores.empty else "Count", fontsize=8)


def make_figure(onset_path: Path, phenology_path: Path, spore_path: Path, output: Path) -> Path:
    onset = _read_onset(onset_path)
    phenology = _read_phenology(phenology_path)
    spores = _read_spore(spore_path)
    if len(onset) != 6:
        raise ValueError(f"Expected 6 disease-onset site-years, found {len(onset)}")

    # Use one shared leap-year calendar so all six panels have identical
    # x-axis limits and month/day positions despite different observation years.
    all_calendar_dates = []
    for _, row in onset.iterrows():
        all_calendar_dates.append(_plot_date(row["Date"]))
        lat, lon, year = float(row["Latitude"]), float(row["Longitude"]), int(row["Year"])
        ph = _match(phenology, lat=lat, lon=lon, year=year, prefix="")
        sp = _match(spores, lat=lat, lon=lon, year=year, prefix="")
        all_calendar_dates.extend(_plot_date(value) for value in ph["date"])
        all_calendar_dates.extend(_plot_date(value) for value in sp["Date"])
    x_start = min(all_calendar_dates) - pd.Timedelta(days=5)
    x_end = max(all_calendar_dates) + pd.Timedelta(days=5)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, constrained_layout=False)
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, onset.iterrows()):
        lat, lon, year = float(row["Latitude"]), float(row["Longitude"]), int(row["Year"])
        ph = _match(phenology, lat=lat, lon=lon, year=year, prefix="")
        sp = _match(spores, lat=lat, lon=lon, year=year, prefix="")
        if ph.empty:
            raise ValueError(f"No phenology rows for {lat}, {lon}, {year}")
        _plot_site(ax, row, ph, sp, x_start, x_end)

    handles = [
        Patch(facecolor=COLORS["early"], alpha=0.18, label="earliest record to BBCH 53+"),
        Patch(facecolor=COLORS["middle"], alpha=0.18, label="BBCH 53+ to final record"),
        Patch(facecolor=COLORS["late"], alpha=0.18, label="final record to axis end"),
        Patch(facecolor=COLORS["spore"], alpha=0.82, label="positive spore count"),
        Line2D([], [], marker="o", color=COLORS["spore"], markerfacecolor="white", lw=0, label="observed zero"),
        Line2D([], [], marker="x", color=COLORS["missing"], lw=0, label="blank / missing collection"),
        Line2D([], [], color=COLORS["onset"], ls="--", label="observed disease onset"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Guangxi grape downy mildew: disease onset, airborne sporangia and phenology",
        fontsize=15, fontweight="bold", y=1.015,
    )
    fig.text(
        0.5, 0.008,
        "Bars show recorded cumulative counts per collection interval; blank cells are not treated as zero. "
        "Phenology bands use the earliest record, first BBCH 53+ record and final record.",
        ha="center", va="bottom", fontsize=8.5, color="#53616B",
    )
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.06, right=0.98, hspace=0.42, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onset", type=Path, default=DEFAULT_ONSET)
    parser.add_argument("--phenology", type=Path, default=DEFAULT_PHENOLOGY)
    parser.add_argument("--spore", type=Path, default=DEFAULT_SPORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = make_figure(args.onset, args.phenology, args.spore, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
