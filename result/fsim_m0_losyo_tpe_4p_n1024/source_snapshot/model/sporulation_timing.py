"""Shared timing and response rules for nocturnal sporangium production."""

from __future__ import annotations

import numpy as np
import pandas as pd
from model.parameter_table import TABLE_CONFIG


_FIXED = TABLE_CONFIG['fixed']
NIGHT_START_HOUR = _FIXED['night']['start']
NIGHT_END_HOUR = _FIXED['night']['end_exclusive']
SPORULATION_AVAILABILITY_HOUR = _FIXED['night']['available']
SPORULATION_MIN_MOIST_HOURS = _FIXED['min_moist_hours']
SPORULATION_MIN_TEMPERATURE, SPORULATION_MAX_TEMPERATURE = _FIXED['temperature_bounds']
_PRODUCTION = _FIXED['production']
_ASYMPTOTE_COEFFICIENTS = _PRODUCTION['asymptote']
_ASYMPTOTE_OFFSET = _PRODUCTION['offset']
_INTERCEPT_COEFFICIENTS = _PRODUCTION['intercept']
_DURATION_COEFFICIENTS = _PRODUCTION['duration']
_RICHARDS_SHAPE = _PRODUCTION['shape']
_REFERENCE_TEMPERATURE = _PRODUCTION['reference_temperature']
_REFERENCE_MOIST_HOURS = _PRODUCTION['reference_hours']


def summarize_sporulation_nights(
    weather_hourly: pd.DataFrame,
    rh_threshold: float = 80.0,
) -> pd.DataFrame:
    """Summarize moist hours from 20:00 to 07:59 by the night-ending date.

    A moist hour follows Brischetto et al. (2021): RH at or above the
    threshold, rain greater than zero, or leaf wetness longer than 30 min.
    The optional leaf-wetness input must be named ``leaf_wetness_minutes``.
    """
    h = weather_hourly.copy()
    hour = h["datetime"].dt.hour
    is_night = (hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)

    rh = pd.to_numeric(h["relative_humidity"], errors="coerce")
    rain = pd.to_numeric(h["precip_acc_period_adjusted"], errors="coerce")
    moist = rh.ge(rh_threshold) | rain.gt(0)
    if "leaf_wetness_minutes" in h.columns:
        leaf_wetness = pd.to_numeric(h["leaf_wetness_minutes"], errors="coerce")
        moist = moist | leaf_wetness.gt(30)

    h["production_date"] = h["datetime"].dt.normalize()
    h.loc[hour >= NIGHT_START_HOUR, "production_date"] += pd.Timedelta(days=1)
    h["moist_night"] = is_night & moist

    moist_hours = h[h["moist_night"]]
    if moist_hours.empty:
        return pd.DataFrame(columns=["production_date", "MP", "TMP"])

    return (
        moist_hours.groupby("production_date", as_index=False)
        .agg(MP=("moist_night", "sum"), TMP=("air_temp", "mean"))
    )


def lalancette_sporangia(temperature: float, moist_hours: float) -> float:
    """Return the Lalancette predicted sporangia per cm2 of lesion.

    This is the pooled Richards response surface reported by Lalancette et al.
    (1988; doi:10.1094/PHYTO-78-1316). Values outside the mathematical domain
    of the fitted surface are returned as zero.
    """
    temperature = float(temperature)
    moist_hours = float(moist_hours)
    if not np.isfinite(temperature) or not np.isfinite(moist_hours):
        return 0.0

    k0, k1, k2 = _ASYMPTOTE_COEFFICIENTS
    asymptote = (
        k0
        + k1 * temperature
        + k2 * temperature**2
        + _ASYMPTOTE_OFFSET
    )
    b0, b1, b2 = _INTERCEPT_COEFFICIENTS
    intercept = b0 + b1 * temperature + b2 * temperature**2
    p0, p1, p2 = _DURATION_COEFFICIENTS
    duration_term = moist_hours * (
        p0 + p1 * temperature + p2 * temperature**2
    )
    richards_base = 1.0 - np.exp(intercept + duration_term)
    if asymptote <= 0.0 or richards_base <= 0.0:
        return 0.0
    return float(asymptote * richards_base ** (1.0 / (1.0 - _RICHARDS_SHAPE)))


_REFERENCE_SPORANGIA = lalancette_sporangia(
    _REFERENCE_TEMPERATURE, _REFERENCE_MOIST_HOURS
)


def normalized_lalancette_sporulation(
    temperature: float, moist_hours: float,
    min_moist_hours: float = SPORULATION_MIN_MOIST_HOURS,
) -> float:
    """Return Brischetto's 0--1 scaling of the Lalancette response surface."""
    if not (
        moist_hours >= min_moist_hours
        and SPORULATION_MIN_TEMPERATURE < temperature < SPORULATION_MAX_TEMPERATURE
    ):
        return 0.0
    relative_dose = lalancette_sporangia(temperature, moist_hours) / _REFERENCE_SPORANGIA
    return float(np.clip(relative_dose, 0.0, 1.0))


def sporulation_dose_for_day(day, min_moist_hours: float = SPORULATION_MIN_MOIST_HOURS) -> float:
    """Return the literature-standardized dose for one completed moist night.

    The calculation deliberately has no repeated-sporulation or lesion-age
    decay term. Mortality after production is handled in a separate model stage.
    """
    return normalized_lalancette_sporulation(day.TMP, day.MP, min_moist_hours)
