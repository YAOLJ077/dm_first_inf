"""Load site-year spore counts used by the pressure extension."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_site_year_spore_counts(
    path: str | Path,
    season,
    *,
    coordinate_tolerance: float = 0.005,
) -> pd.DataFrame:
    """Load the spore-count rows matching one weather season.

    This small adapter is used by the pressure extension.  It deliberately
    keeps the original workbook semantics: blank counts are missing and zero
    counts are observed zeros.
    """

    frame = pd.read_excel(path)
    required = {"latitude", "longitude", "Year", "Date", "Spore"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["Year"] = pd.to_numeric(frame["Year"], errors="raise").astype(int)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="raise")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="raise")
    frame["Spore"] = pd.to_numeric(frame["Spore"], errors="coerce")
    matched = frame.loc[
        frame["Year"].eq(int(season.year))
        & frame["latitude"].sub(float(season.latitude)).abs().le(float(coordinate_tolerance))
        & frame["longitude"].sub(float(season.longitude)).abs().le(float(coordinate_tolerance))
    ].copy()
    if matched.empty:
        raise ValueError(
            f"No spore observations match {season.site_id}_{season.year} "
            f"near ({season.latitude}, {season.longitude})"
        )
    matched = matched.sort_values("Date").reset_index(drop=True)
    # CK counts accumulate over the slide-exposure interval.  A blank is a
    # missing independent count, whereas a numeric zero is an observed zero.
    measured = matched.loc[matched["Spore"].notna()].copy()
    if measured.empty:
        raise ValueError(f"No nonblank spore counts match {season.site_id}_{season.year}")
    dates = pd.DatetimeIndex(measured["Date"]).normalize()
    starts = dates.to_series(index=measured.index).shift(1)
    starts.iloc[0] = dates[0] - pd.Timedelta(days=3)
    measured["interval_start"] = starts.to_numpy()
    measured["interval_end"] = dates.to_numpy()
    measured["exposure_days"] = (
        measured["interval_end"] - measured["interval_start"]
    ).dt.total_seconds() / 86400.0
    return measured.reset_index(drop=True)
