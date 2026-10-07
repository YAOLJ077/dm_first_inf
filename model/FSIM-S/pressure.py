"""Observed spore pressure and weather-cohort gating for FSIM-S."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


DEFAULT_PRESSURE_CONFIG = {
    "enabled": False,
    "beta_sp": 0.005,
    "rolling_days": 3,
    "reference_level": 0.5,
    "observed_zero_weather_gate": True,
}


def validate_pressure_config(config: dict | None) -> dict:
    """Validate the final FSIM-S observed-pressure configuration."""

    merged = dict(DEFAULT_PRESSURE_CONFIG)
    if config:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(f"Unsupported pressure configuration fields: {sorted(unknown)}")
        merged.update(config)

    beta_sp = float(merged["beta_sp"])
    if not math.isfinite(beta_sp) or beta_sp < 0:
        raise ValueError("beta_sp must be finite and non-negative")
    rolling_days = int(merged["rolling_days"])
    if rolling_days < 1 or rolling_days % 2 == 0:
        raise ValueError("rolling_days must be a positive odd integer")
    reference_level = float(merged["reference_level"])
    if not math.isfinite(reference_level) or not 0 <= reference_level <= 1:
        raise ValueError("reference_level must be finite and in [0, 1]")
    if not isinstance(merged["observed_zero_weather_gate"], (bool, np.bool_)):
        raise ValueError("observed_zero_weather_gate must be boolean")

    return {
        "enabled": bool(merged["enabled"]),
        "beta_sp": beta_sp,
        "rolling_days": rolling_days,
        "reference_level": reference_level,
        "observed_zero_weather_gate": bool(merged["observed_zero_weather_gate"]),
    }


def _normalize_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        return dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    return dates.dt.normalize()


def observed_zero_dates(observations: pd.DataFrame) -> frozenset:
    """Return dates covered by numeric zero-spore collection intervals."""

    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    required = {"Date", "Spore"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"spore observations are missing columns {sorted(missing)}")

    frame = observations.copy()
    frame["Date"] = _normalize_dates(frame["Date"])
    frame["Spore"] = pd.to_numeric(frame["Spore"], errors="coerce")
    frame = frame.loc[frame["Spore"].notna()].sort_values("Date").reset_index(drop=True)
    if frame.empty:
        return frozenset()

    if {"interval_start", "interval_end"}.issubset(frame.columns):
        starts = _normalize_dates(frame["interval_start"])
        ends = _normalize_dates(frame["interval_end"])
    else:
        ends = frame["Date"]
        starts = ends.shift(1)
        starts.iloc[0] = ends.iloc[0] - pd.Timedelta(days=3)

    zero_dates = set()
    for count, start, end in zip(frame["Spore"], starts, ends):
        if float(count) != 0.0:
            continue
        zero_dates.update(
            timestamp.date()
            for timestamp in pd.date_range(start + pd.Timedelta(days=1), end, freq="D")
        )
    return frozenset(zero_dates)


def build_pressure_field(
    observations: pd.DataFrame,
    timestamps,
    *,
    rolling_days: int = 3,
    reference_level: float = 0.5,
) -> pd.DataFrame:
    """Build the final interval-rate, measured-period z-score pressure field."""

    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    required = {"Date", "Spore"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"spore observations are missing columns {sorted(missing)}")
    rolling_days = int(rolling_days)
    if rolling_days < 1 or rolling_days % 2 == 0:
        raise ValueError("rolling_days must be a positive odd integer")
    reference_level = float(reference_level)
    if not math.isfinite(reference_level):
        raise ValueError("reference_level must be finite")

    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if index.empty:
        return pd.DataFrame(columns=["datetime", "spore_count", "spore_pressure"])
    local = (
        index.tz_convert("Asia/Shanghai")
        if index.tz is not None
        else index.tz_localize("Asia/Shanghai")
    )
    start = local[0].tz_localize(None).normalize()
    end = local[-1].tz_localize(None).normalize()
    calendar = pd.date_range(start, end, freq="D")

    dates = _normalize_dates(observations["Date"])
    counts = pd.to_numeric(observations["Spore"], errors="coerce")
    measured = counts.notna() & np.isfinite(counts.to_numpy(dtype=float))
    if not measured.any():
        raise ValueError("spore observations contain no numeric counts")
    measured_frame = pd.DataFrame({
        "Date": dates.loc[measured].to_numpy(),
        "Spore": counts.loc[measured].to_numpy(float),
    })
    measured_frame = (
        measured_frame.groupby("Date", as_index=False)["Spore"]
        .mean()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    daily = pd.Series(np.nan, index=calendar, dtype=float)
    measured_days = pd.Series(False, index=calendar, dtype=bool)
    interval_end = pd.DatetimeIndex(measured_frame["Date"])
    interval_start = interval_end.to_series(index=measured_frame.index).shift(1)
    interval_start.iloc[0] = interval_end[0] - pd.Timedelta(days=3)
    exposure_days = (
        interval_end.to_series(index=measured_frame.index) - interval_start
    ).dt.total_seconds() / 86400.0
    if (exposure_days <= 0).any():
        raise ValueError("spore observation dates must increase")
    for row, start_date, end_date, duration in zip(
        measured_frame.itertuples(index=False), interval_start, interval_end, exposure_days
    ):
        mask = (calendar > pd.Timestamp(start_date)) & (calendar <= pd.Timestamp(end_date))
        daily.loc[mask] = float(row.Spore) / float(duration)
        measured_days.loc[mask] = True

    raw_daily = daily.fillna(0.0)
    normalization_values = raw_daily.loc[measured_days]
    mean = float(normalization_values.mean())
    std = float(normalization_values.std(ddof=0))
    pressure = (raw_daily - mean) / std if std > 0 else raw_daily * 0.0
    pressure = pressure.rolling(rolling_days, center=True, min_periods=1).mean()
    centered = pressure - reference_level

    by_day = centered.to_dict()
    raw_by_day = raw_daily.to_dict()
    pressure_by_day = pressure.to_dict()
    return pd.DataFrame({
        "datetime": local,
        "spore_count": [float(raw_by_day.get(ts.tz_localize(None).normalize(), 0.0)) for ts in local],
        "pressure_shape": "interval_rate",
        "spore_index": [float(pressure_by_day.get(ts.tz_localize(None).normalize(), 0.0)) for ts in local],
        "spore_pressure": [float(by_day.get(ts.tz_localize(None).normalize(), 0.0)) for ts in local],
        "pressure_reference": reference_level,
        "pressure_normalization": "zscore",
    })


class ObservedDoseProvider:
    """Map positive centered pressure to external sporangia dose."""

    def __init__(
        self,
        pressure_field: pd.DataFrame,
        beta_sp: float,
        *,
        observed_zero_weather_gate: bool = True,
        observed_zero_dates=None,
    ):
        if "datetime" not in pressure_field or "spore_pressure" not in pressure_field:
            raise ValueError("pressure_field must contain datetime and spore_pressure")
        self.time = pd.DatetimeIndex(pd.to_datetime(pressure_field["datetime"]))
        if self.time.tz is not None:
            self.time = self.time.tz_convert("Asia/Shanghai").tz_localize(None)
        self.pressure = pd.to_numeric(pressure_field["spore_pressure"], errors="raise").to_numpy(float)
        self.beta_sp = float(beta_sp)
        if not math.isfinite(self.beta_sp) or self.beta_sp < 0:
            raise ValueError("beta_sp must be finite and non-negative")
        self.observed_zero_weather_gate = bool(observed_zero_weather_gate)
        self.observed_zero_dates = frozenset(observed_zero_dates or ())

    def pressure_at(self, timestamp) -> float:
        stamp = pd.Timestamp(timestamp)
        if stamp.tz is None:
            stamp = stamp.tz_localize("Asia/Shanghai")
        else:
            stamp = stamp.tz_convert("Asia/Shanghai")
        naive = stamp.tz_localize(None)
        index = int(self.time.searchsorted(naive))
        if index >= len(self.time) or self.time[index] != naive:
            return 0.0
        value = float(self.pressure[index])
        return value if math.isfinite(value) else 0.0

    def dose_at(self, timestamp) -> float:
        return self.beta_sp * max(0.0, self.pressure_at(timestamp))

    def suppress_weather_at(self, timestamp) -> bool:
        if not self.observed_zero_weather_gate:
            return False
        stamp = pd.Timestamp(timestamp)
        if stamp.tz is not None:
            stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
        return stamp.date() in self.observed_zero_dates

    def description(self) -> dict:
        return {
            "type": "observed_background_dose",
            "beta_sp": self.beta_sp,
            "hours": len(self.time),
            "dose_rule": "beta_sp * max(spore_pressure, 0)",
            "observed_zero_weather_gate": self.observed_zero_weather_gate,
            "observed_zero_dates": len(self.observed_zero_dates),
        }


__all__ = [
    "DEFAULT_PRESSURE_CONFIG",
    "ObservedDoseProvider",
    "build_pressure_field",
    "observed_zero_dates",
    "validate_pressure_config",
]
