"""Cohorts from recurrent asexual production through infection and symptoms."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cropseason import CropSeason, InputError, LOCAL_TIMEZONE


# ---------------------------------------------------------------------------
# Fixed process relations (FSIM provenance; see README.md for source limits)
# ---------------------------------------------------------------------------
# Brischetto et al. (2021) uses the Lalancette response normalized to the
# response at 20 C and 24 h. These coefficients are fixed model relations,
# not calibration parameters for this route.
# Study convention: S starts immediately after the January 31 P cutoff.
# This is not a measured biological activation date.
INITIALIZATION_MONTH_DAY = "02-01"
INITIALIZATION_HOUR = 0
# Night/availability clock times retain the FSIM computational conventions,
# not measured local darkness (fsim_parameter_provenance.csv).
NIGHT_START_HOUR = 20
NIGHT_END_HOUR_EXCLUSIVE = 8
SPORULATION_AVAILABILITY_HOUR = 8
# Defaults are retained as documented process baselines; the active values
# are read from config["sporulation_parameters"] at simulation time.
RH_WET_NIGHT_PCT = 80.0
MINIMUM_MOIST_HOURS = 3.0
SPORULATION_TEMPERATURE_MIN_C = 10.0
SPORULATION_TEMPERATURE_MAX_C = 30.0
LALANCETTE_ASYMPTOTE_COEFFICIENTS = (-873164.0, 114369.0, -2828.0)
LALANCETTE_ASYMPTOTE_OFFSET = 50000.0
LALANCETTE_INTERCEPT_COEFFICIENTS = (-8.96, 1.16, -0.029)
LALANCETTE_DURATION_COEFFICIENTS = (1.51, -0.2, 0.0049)
LALANCETTE_RICHARDS_SHAPE = 0.5
LALANCETTE_REFERENCE_TEMPERATURE_C = 20.0
LALANCETTE_REFERENCE_MOIST_HOURS = 24.0

# Hourly mortality is 1 / [24 * (c0 - c1*x + c2*x^2)], where
# x = T * (1 - RH/100). The coefficient sets are retained from the FSIM
# implementation and their Brischetto et al. (2020, 2021) provenance.
ATTACHED_MORTALITY_COEFFICIENTS = (9.27, 1.12, 0.04)
DETACHED_MORTALITY_COEFFICIENTS = (5.67, 0.47, 0.02)

# Brischetto et al. (2021) represents dispersal and deposition jointly with
# D&DR = 1. This route retains that unit transfer dose at wind dispersal, then
# makes the timing of effective deposition explicit through the rain thresholds
# below. The wind threshold is an explicit structural setting, not a biological
# coefficient inherited from Brischetto.
DISPERSAL_DEPOSITION_RATE = 1.0

# Rain thresholds are explicit route-level parameters.  There is no
# background deposition in the absence of rain: an airborne cohort deposits
# only in the interval [EFFECTIVE_DEPOSITION_MIN_RAIN_MM,
# HEAVY_RAIN_WASHOFF_MM).  Rain at or above the latter threshold removes the
# cohort before effective deposition.
EFFECTIVE_DEPOSITION_MIN_RAIN_MM = 0.2
HEAVY_RAIN_WASHOFF_MM = 10.0

# Fixed DPD leaf-wetness proxy relation used by the shared Rossi downstream
# process. These values are part of the documented process definition, not
# manual calibration parameters for the sporangia route.
DPD_WET_ONSET_C = 2.0
DPD_DRY_OFF_C = 3.8
LEAF_RAIN_THRESHOLD_MM = 0.0
DPD_INITIAL_WET = False

# Rossi et al. (2008) relations reused after a sporangium has deposited on a
# leaf surface. The infection degree-hour requirement remains explicit in
# config.json because it is the main manual sensitivity input for this route.
ROSSI_RELEASE_OFFSET = 1.022
ROSSI_RELEASE_TEMPERATURE_COEFFICIENT = 19.634
ROSSI_ZOOSPORE_SURVIVAL_LIMIT = 1.0
ROSSI_INCUBATION_COEFFICIENTS = {
    "lower": (45.1, 3.45, 0.073),
    "upper": (59.9, 4.55, 0.095),
}

FIXED_RELATIONS = {
    "moist_night": "RH >= sporulation_rh_min_pct OR rain > 0 OR leaf_wetness_minutes > 30 during 20:00-07:59",
    "sporulation_switch": "MP >= minimum_moist_hours and 10 < TMP < 30 C",
    "sporulation_response": "normalized Lalancette Richards surface at 20 C and 24 h",
    "mortality": "1 / (24 * (c0 - c1*x + c2*x^2)), x=T*(1-RH/100)",
    "dispersal_deposition": "Wind threshold creates an airborne detached cohort; rain >= effective_deposition_min_rain_mm and < heavy_rain_washoff_mm deposits its surviving amount; rain >= heavy_rain_washoff_mm washes it off before effective deposition",
    "detached_survival": "Detached mortality starts at wind dispersal and continues through airborne and deposited phases until zoospore release; one mortality update per hour",
    "release": "WD >= exp((-1.022 + 19.634) / TWD); Rossi 2008 printed p.484",
    "zoospore_survival": "elapsed/wet hours after release; irreversible death when ratio > 1",
    "leaf_wetness": "DPD = air temperature - dew point; rain > threshold sets wet, DPD below onset sets wet, DPD above dry-off sets dry, otherwise retain state",
    "infection": "first hourly current S*WD*TWD from release >= infection_degree_hours, on wet hours; S=1 without modulation",
    "incubation": "Rossi Eqs.8-9 cumulative progress >= 1, including infection hour",
}


def local_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize(LOCAL_TIMEZONE)
        if timestamp.tz is None
        else timestamp.tz_convert(LOCAL_TIMEZONE)
    )


def iso_timestamp(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return local_timestamp(value).isoformat()


def _finite(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise InputError(f"{name} must be finite")
    return value


def summarize_sporulation_nights(
    weather: pd.DataFrame,
    *,
    sporulation_rh_min_pct: float = RH_WET_NIGHT_PCT,
) -> pd.DataFrame:
    """Summarize Brischetto moist night hours by the night-ending date."""

    frame = weather.copy()
    hour = frame["datetime"].dt.hour
    is_night = (hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR_EXCLUSIVE)
    rh = pd.to_numeric(frame["relative_humidity"], errors="raise")
    rain = pd.to_numeric(frame["precip_acc_period_adjusted"], errors="raise")
    moist = rh >= float(sporulation_rh_min_pct)
    moist |= rain > 0
    if "leaf_wetness_minutes" in frame:
        leaf_wetness = pd.to_numeric(frame["leaf_wetness_minutes"], errors="coerce")
        moist |= leaf_wetness > 30

    frame["production_date"] = frame["datetime"].dt.normalize()
    frame.loc[hour >= NIGHT_START_HOUR, "production_date"] += pd.Timedelta(days=1)
    frame["moist_night"] = is_night & moist
    selected = frame.loc[frame["moist_night"]]
    if selected.empty:
        return pd.DataFrame(columns=["production_date", "MP", "TMP"])
    return (
        selected.groupby("production_date", as_index=False)
        .agg(MP=("moist_night", "sum"), TMP=("air_temp", "mean"))
    )


def lalancette_sporangia(
    temperature: float,
    moist_hours: float,
) -> float:
    """Lalancette response surface used for relative sporulation dose."""

    temperature = _finite(temperature, "temperature")
    moist_hours = _finite(moist_hours, "moist_hours")
    asymptote = (
        LALANCETTE_ASYMPTOTE_COEFFICIENTS[0]
        + LALANCETTE_ASYMPTOTE_COEFFICIENTS[1] * temperature
        + LALANCETTE_ASYMPTOTE_COEFFICIENTS[2] * temperature**2
        + LALANCETTE_ASYMPTOTE_OFFSET
    )
    intercept = (
        LALANCETTE_INTERCEPT_COEFFICIENTS[0]
        + LALANCETTE_INTERCEPT_COEFFICIENTS[1] * temperature
        + LALANCETTE_INTERCEPT_COEFFICIENTS[2] * temperature**2
    )
    duration = moist_hours * (
        LALANCETTE_DURATION_COEFFICIENTS[0]
        + LALANCETTE_DURATION_COEFFICIENTS[1] * temperature
        + LALANCETTE_DURATION_COEFFICIENTS[2] * temperature**2
    )
    exponent = intercept + duration
    if exponent >= 0:
        return 0.0
    richards_base = 1.0 - math.exp(exponent)
    shape = LALANCETTE_RICHARDS_SHAPE
    if asymptote <= 0 or richards_base <= 0 or shape == 1:
        return 0.0
    return float(asymptote * richards_base ** (1.0 / (1.0 - shape)))


def normalized_sporulation_dose(
    temperature: float,
    moist_hours: float,
    *,
    minimum_moist_hours: float = MINIMUM_MOIST_HOURS,
) -> float:
    """Return the normalized 0--1 dose for a completed moist night."""

    if not (
        moist_hours >= float(minimum_moist_hours)
        and SPORULATION_TEMPERATURE_MIN_C < temperature
        < SPORULATION_TEMPERATURE_MAX_C
    ):
        return 0.0
    reference = lalancette_sporangia(
        LALANCETTE_REFERENCE_TEMPERATURE_C,
        LALANCETTE_REFERENCE_MOIST_HOURS,
    )
    if reference <= 0:
        raise InputError("Sporulation reference response must be positive")
    value = lalancette_sporangia(temperature, moist_hours) / reference
    return float(np.clip(value, 0.0, 1.0))


def mortality_rate(mortality_index: float, *, attached: bool) -> float:
    """Hourly mortality rate from the Brischetto/Blaeser-Weltzien form."""

    index = _finite(mortality_index, "mortality_index")
    c0, c1, c2 = (
        ATTACHED_MORTALITY_COEFFICIENTS if attached else DETACHED_MORTALITY_COEFFICIENTS
    )
    denominator = 24.0 * (c0 - c1 * index + c2 * index**2)
    if denominator <= 0:
        raise InputError("Mortality equation produced a non-positive denominator")
    return 1.0 / denominator


def mortality_index(temperature, relative_humidity):
    temperature = np.asarray(temperature, dtype=float)
    relative_humidity = np.asarray(relative_humidity, dtype=float)
    return temperature * (1.0 - relative_humidity / 100.0)


def physical_vpd_hpa(temperature, relative_humidity):
    """Buck physical vapour-pressure deficit in hPa."""

    temperature = np.asarray(temperature, dtype=float)
    relative_humidity = np.asarray(relative_humidity, dtype=float)
    return 6.1121 * np.exp(17.502 * temperature / (240.97 + temperature)) * (
        1.0 - relative_humidity / 100.0
    )


def dew_point_from_relative_humidity(temperature, relative_humidity):
    """Estimate dew point when a legacy weather row has no dew-point field."""

    temperature = np.asarray(temperature, dtype=float)
    relative_humidity = np.asarray(relative_humidity, dtype=float)
    if not (np.isfinite(temperature).all() and np.isfinite(relative_humidity).all()):
        raise InputError("Temperature and relative humidity must be finite")
    if ((relative_humidity <= 0) | (relative_humidity > 100)).any():
        raise InputError("Relative humidity must be greater than 0 and at most 100%")
    vapor_pressure = physical_vpd_hpa(temperature, relative_humidity)
    saturation_pressure = 6.1121 * np.exp(
        17.502 * temperature / (240.97 + temperature)
    )
    actual_pressure = saturation_pressure - vapor_pressure
    log_ratio = np.log(actual_pressure / 6.1121)
    return 240.97 * log_ratio / (17.502 - log_ratio)


def dpd_leaf_wetness(
    air_temp,
    dew_point,
    rain,
):
    """Causal leaf-wetness proxy for the downstream Rossi process.

    This is implemented inside the sporangia route so the model remains
    self-contained. Rain has priority; between the two DPD thresholds the
    previous wet/dry state is retained.
    """

    air_temp = np.asarray(air_temp, dtype=float)
    dew_point = np.asarray(dew_point, dtype=float)
    rain = np.asarray(rain, dtype=float)
    if DPD_DRY_OFF_C <= DPD_WET_ONSET_C:
        raise InputError("dpd_dry_off_c must be greater than dpd_wet_onset_c")
    if not (
        np.isfinite(air_temp).all()
        and np.isfinite(dew_point).all()
        and np.isfinite(rain).all()
    ):
        raise InputError("Leaf-wetness weather contains non-finite values")
    if (rain < 0).any():
        raise InputError("Rainfall cannot be negative")
    if not (len(air_temp) == len(dew_point) == len(rain)):
        raise InputError("Leaf-wetness weather arrays must have equal length")

    depression = air_temp - dew_point
    wet = np.empty(len(air_temp), dtype=bool)
    state = DPD_INITIAL_WET
    for index, value in enumerate(depression):
        if rain[index] > LEAF_RAIN_THRESHOLD_MM or value < DPD_WET_ONSET_C:
            state = True
        elif value > DPD_DRY_OFF_C:
            state = False
        wet[index] = state
    return depression, wet


def rossi_release(wet_hours: int, mean_temperature: float) -> bool:
    """Rossi's wet-duration/temperature release relation."""

    if wet_hours <= 0 or mean_temperature <= 0:
        return False
    return math.log(wet_hours) * mean_temperature >= (
        -ROSSI_RELEASE_OFFSET + ROSSI_RELEASE_TEMPERATURE_COEFFICIENT
    )


def rossi_zoospore_survival_ratio(elapsed_hours: int, wet_hours: int) -> float:
    """Rossi Eq. (7): elapsed hours divided by subsequent wet hours."""

    if elapsed_hours <= 0:
        return 0.0
    if wet_hours <= 0:
        return float("inf")
    return float(elapsed_hours) / float(wet_hours)


def incubation_rates(temperature):
    """Rossi Eqs. (8)-(9), lower and upper hourly incubation progress."""

    temperature = np.asarray(temperature, dtype=float)
    lower = ROSSI_INCUBATION_COEFFICIENTS["lower"]
    upper = ROSSI_INCUBATION_COEFFICIENTS["upper"]
    return (
        1.0 / (24.0 * (lower[0] - lower[1] * temperature + lower[2] * temperature**2)),
        1.0 / (24.0 * (upper[0] - upper[1] * temperature + upper[2] * temperature**2)),
    )


@dataclass(frozen=True)
class DriverSettings:
    """Weather and transfer assumptions for the sporangia route."""

    dispersal_mode: str = "wind_gated"
    wind_min_m_s: float | None = 0.5


class HourlyDrivers:
    """Validated local-time hourly arrays consumed by the sporangia process."""

    def __init__(self, season: CropSeason, settings: DriverSettings | None = None):
        self.season = season
        self.settings = settings or DriverSettings()
        if self.settings.dispersal_mode not in {"merged", "wind_gated"}:
            raise InputError("dispersal_mode must be merged or wind_gated")
        _validate_transfer(self.settings.dispersal_mode, self.settings.wind_min_m_s)

        frame = season.weather.copy()
        self.frame = frame
        self.time = pd.DatetimeIndex(frame["datetime"]).tz_localize(LOCAL_TIMEZONE)
        if not self.time.equals(pd.date_range(self.time[0], periods=len(self.time), freq="h")):
            raise InputError("Weather timestamps must be consecutive local hours")
        self.temperature = frame["air_temp"].to_numpy(float)
        self.relative_humidity = frame["relative_humidity"].to_numpy(float)
        self.rain = frame["precip_acc_period_adjusted"].to_numpy(float)
        self.wind = frame["wind_speed"].to_numpy(float)
        if "dew_point" in frame:
            self.dew_point = frame["dew_point"].to_numpy(float)
            self.dew_point_source = "input"
        else:
            self.dew_point = dew_point_from_relative_humidity(
                self.temperature, self.relative_humidity
            )
            self.dew_point_source = "estimated_from_temperature_relative_humidity"
        self.mortality_index = mortality_index(self.temperature, self.relative_humidity)
        self.vpd = physical_vpd_hpa(self.temperature, self.relative_humidity)
        self.dpd, self.wet = dpd_leaf_wetness(
            self.temperature,
            self.dew_point,
            self.rain,
        )
        self._incubation_prefix = [
            np.r_[0.0, np.cumsum(rate)] for rate in incubation_rates(self.temperature)
        ]

    def stamp(self, index):
        return self.time[index].isoformat() if index is not None else None

    def symptoms(self, infection_index):
        hits = []
        for prefix in self._incubation_prefix:
            end = int(np.searchsorted(prefix, prefix[infection_index] + 1.0))
            hits.append(end - 1 if end < len(prefix) else None)
        known = sorted(i for i in hits if i is not None)
        return {
            "symptom_start": self.stamp(known[0]) if known else None,
            "symptom_end": self.stamp(known[-1]) if len(known) == 2 else None,
            "symptom_window_complete": len(known) == 2,
            "incubation_censored": len(known) < 2,
        }

    def description(self) -> dict:
        return {
            "timezone": LOCAL_TIMEZONE,
            "dispersal_mode": self.settings.dispersal_mode,
            "wind_min_m_s": self.settings.wind_min_m_s,
            "leaf_wetness_proxy": "DPD hysteresis with rain priority",
            "wetness_parameters": {
                "mode": "dpd",
                "dpd_wet_onset_c": DPD_WET_ONSET_C,
                "dpd_dry_off_c": DPD_DRY_OFF_C,
                "leaf_rain_mm": LEAF_RAIN_THRESHOLD_MM,
                "dpd_initial_wet": DPD_INITIAL_WET,
                "dew_point_source": self.dew_point_source,
            },
            "wet_hours": int(self.wet.sum()),
            "fixed_relations": FIXED_RELATIONS.copy(),
            "weather_hours": len(self.time),
        }


def _validate_transfer(mode, threshold) -> None:
    if mode not in {"merged", "wind_gated"}:
        raise InputError("dispersal_mode must be merged or wind_gated")
    if mode == "merged":
        if threshold is not None:
            raise InputError("merged transfer does not use wind_min_m_s; set it to null")
    elif (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise InputError("wind_gated transfer requires an explicit finite non-negative wind_min_m_s")


def validate_config(config: dict) -> None:
    allowed = {
        "schema_version", "branch", "implementation_status", "reference", "role",
        "source_assumption",
        "initialization_month_day", "initialization_hour", "dispersal_mode", "structure_parameters",
        "sporulation_parameters",
        "deposition_parameters",
        "fixed_relations", "initialization_status", "parameter_status", "biological_parameters",
    }
    unknown = set(config) - allowed
    if unknown:
        raise InputError(f"Unsupported sporangia configuration fields: {sorted(unknown)}")
    if config.get("schema_version") != 3:
        raise InputError("Sporangia requires config schema_version=3; use the current config.json")
    if config.get("branch") != "sporangia":
        raise InputError("Sporangia config must declare branch=sporangia")
    if config.get("initialization_month_day") != INITIALIZATION_MONTH_DAY:
        raise InputError("Sporangia initialization is fixed at 02-01 after the potential window")
    if config.get("initialization_hour") != INITIALIZATION_HOUR:
        raise InputError("Sporangia initialization hour is fixed at 00:00")
    parameters = config.get("structure_parameters")
    if not isinstance(parameters, dict) or set(parameters) != {"wind_min_m_s"}:
        raise InputError("structure_parameters must contain only wind_min_m_s")
    _validate_transfer(config.get("dispersal_mode"), parameters["wind_min_m_s"])
    sporulation = config.get("sporulation_parameters")
    if not isinstance(sporulation, dict) or set(sporulation) != {
        "sporulation_rh_min_pct", "minimum_moist_hours"
    }:
        raise InputError(
            "sporulation_parameters must contain only sporulation_rh_min_pct and minimum_moist_hours"
        )
    rh_min = sporulation["sporulation_rh_min_pct"]
    moist_hours = sporulation["minimum_moist_hours"]
    if (
        isinstance(rh_min, bool)
        or not isinstance(rh_min, (int, float))
        or not math.isfinite(rh_min)
        or not 0 <= rh_min <= 100
    ):
        raise InputError("sporulation_rh_min_pct must be finite and between 0 and 100")
    if (
        isinstance(moist_hours, bool)
        or not isinstance(moist_hours, (int, float))
        or not math.isfinite(moist_hours)
        or not 0 < moist_hours <= (NIGHT_END_HOUR_EXCLUSIVE + (24 - NIGHT_START_HOUR))
    ):
        raise InputError("minimum_moist_hours must be finite, positive, and no greater than 12")
    deposition = config.get("deposition_parameters")
    if not isinstance(deposition, dict) or set(deposition) != {
        "effective_deposition_min_rain_mm", "heavy_rain_washoff_mm"
    }:
        raise InputError(
            "deposition_parameters must contain only effective_deposition_min_rain_mm and heavy_rain_washoff_mm"
        )
    effective_min = deposition["effective_deposition_min_rain_mm"]
    heavy_washoff = deposition["heavy_rain_washoff_mm"]
    for value, name in (
        (effective_min, "effective_deposition_min_rain_mm"),
        (heavy_washoff, "heavy_rain_washoff_mm"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise InputError(f"{name} must be finite and non-negative")
    if heavy_washoff <= effective_min:
        raise InputError("heavy_rain_washoff_mm must be greater than effective_deposition_min_rain_mm")
    biological = config.get("biological_parameters")
    if not isinstance(biological, dict) or set(biological) != {"infection_degree_hours"}:
        raise InputError("biological_parameters must contain only infection_degree_hours")
    if (
        isinstance(biological["infection_degree_hours"], bool)
        or not isinstance(biological["infection_degree_hours"], (int, float))
        or not math.isfinite(biological["infection_degree_hours"])
        or biological["infection_degree_hours"] <= 0
    ):
        raise InputError("infection_degree_hours must be finite and positive")
    if DPD_DRY_OFF_C <= DPD_WET_ONSET_C or LEAF_RAIN_THRESHOLD_MM < 0:
        raise InputError("Invalid fixed DPD wetness relation")


def _remaining(initial_amount: float, cumulative_mortality: float) -> float:
    return max(0.0, initial_amount * (1.0 - cumulative_mortality))


def advance_deposited_cohort(cohort, drivers, infection_degree_hours, susceptibility_modulator=None):
    """Advance one viable deposited cohort; terminal failures never restart.

    A cohort that has already received its detached mortality update at the
    deposition hour (the normal simulation path) is not charged that hour a
    second time. Direct callers without that marker retain the historical
    convention and receive mortality at deposition. Release wetness starts in
    the deposition hour. Dryness resets a pre-release wet run; after release
    it irreversibly kills the cohort under Rossi Eq.7. Infection sums T
    starting at the release hour, on wet hours.
    """
    row = dict(cohort)
    deposition_time = cohort["deposition_datetime"]
    survival_start = cohort.get("dispersal_datetime", deposition_time)
    # ``airborne_amount`` is the viable amount after attached mortality and
    # before detached mortality.  For legacy/direct cohorts the deposited
    # amount is the only available starting amount.
    initial_amount = float(cohort.get("airborne_amount", cohort["deposited_amount"]))
    cumulative = float(cohort.get("detached_mortality_cumulative_at_deposition", 0.0))
    mortality_applied_at_deposition = bool(
        cohort.get("detached_mortality_applied_at_deposition", False)
    )
    row.update(
        zoospore_release=None, infection_time=None,
        symptom_start=None, symptom_end=None, symptom_window_complete=False,
        incubation_censored=None, terminal_time=None,
        status="sporangium_censored", infection_degree_hours_accumulated=0.0,
        detached_survival_start=survival_start,
        detached_survival_end=None, detached_survival_end_reason=None,
        zoospore_survival_ratio=None,
        infection_wet_hours=0, infection_mean_temperature=None,
        infection_environmental_index=0.0,
        susceptibility_modulation_factor=0.0,
        infection_effective_index=0.0,
    )
    start = int(drivers.time.searchsorted(pd.Timestamp(deposition_time)))
    wet_hours, temperature_sum, wet_after_release = 0, 0.0, 0
    release = None
    infection_wet_hours = 0
    infection_temperature_sum = 0.0
    for i in range(start, len(drivers.time)):
        if release is None:
            if i > start or not mortality_applied_at_deposition:
                cumulative += mortality_rate(drivers.mortality_index[i], attached=False)
            row.update(
                detached_mortality_cumulative_at_end=float(min(1.0, cumulative)),
                remaining_amount_at_end=_remaining(initial_amount, cumulative),
            )
            if cumulative >= 1.0:
                row.update(status="sporangium_dead_before_release", terminal_time=drivers.stamp(i),
                           detached_survival_end=drivers.stamp(i), detached_survival_end_reason="dead")
                break
            if drivers.wet[i]:
                wet_hours += 1
                temperature_sum += float(drivers.temperature[i])
                if rossi_release(wet_hours, temperature_sum / wet_hours):
                    release = i
                    row.update(zoospore_release=drivers.stamp(i), status="zoospore_censored",
                               relative_viable_dose_at_release=row["remaining_amount_at_end"],
                               detached_survival_end=drivers.stamp(i), detached_survival_end_reason="released")
            else:
                wet_hours, temperature_sum = 0, 0.0
            if release is None:
                continue

        if i > release:
            wet_after_release += int(drivers.wet[i])
            ratio = rossi_zoospore_survival_ratio(i - release, wet_after_release)
            row["zoospore_survival_ratio"] = ratio if math.isfinite(ratio) else None
            if ratio > ROSSI_ZOOSPORE_SURVIVAL_LIMIT:
                row.update(status="zoospore_dead_by_rossi_eq7", terminal_time=drivers.stamp(i))
                break
        # Rossi (2008) INF is evaluated at each hour.  WD*TWD is the
        # wet-hour temperature product from release to the current hour.
        # The phenology coefficient is the current-hour phenology-stage
        # coefficient applied to that current WD*TWD snapshot; it is not
        # accumulated separately across hours.
        # In this route deposition has already occurred before this loop;
        # the release hour is therefore the first leaf-wetness hour available
        # for infection.  This differs from O, where the explicit splash
        # hour is the Rossi dispersal hour (iota) and is excluded.
        if i >= release and drivers.wet[i]:
            infection_wet_hours += 1
            hourly_temperature = float(drivers.temperature[i])
            infection_temperature_sum += hourly_temperature
            coefficient = (
                susceptibility_modulator.factor(drivers.time[i])
                if susceptibility_modulator is not None else 1.0
            )
            environmental_index = infection_temperature_sum
            effective_index = coefficient * environmental_index
            row.update(
                infection_wet_hours=infection_wet_hours,
                infection_mean_temperature=infection_temperature_sum / infection_wet_hours,
                infection_environmental_index=float(environmental_index),
                susceptibility_modulation_factor=float(coefficient),
                infection_effective_index=float(effective_index),
                infection_degree_hours_accumulated=float(effective_index),
            )
            if effective_index >= infection_degree_hours:
                row.update(infection_time=drivers.stamp(i), terminal_time=drivers.stamp(i),
                           status="infected", **drivers.symptoms(i))
                break
    else:
        row["terminal_time"] = drivers.stamp(len(drivers.time) - 1)
        if release is None:
            row.update(detached_survival_end=row["terminal_time"], detached_survival_end_reason="weather_end")
    row["detached_survival_censored"] = row["detached_survival_end_reason"] == "weather_end"
    return row


def simulate(
    drivers: HourlyDrivers,
    config: dict,
    susceptibility_modulator=None,
    first_window_only: bool = False,
    external_dose_provider=None,
) -> dict:
    """Simulate recurrent asexual cohorts through airborne, deposited and infection states."""

    validate_config(config)
    initial = local_timestamp(
        f"{drivers.time[0].year}-{INITIALIZATION_MONTH_DAY} {INITIALIZATION_HOUR:02d}:00"
    )
    if initial < drivers.time[0] or initial > drivers.time[-1]:
        raise InputError("Sporangia initialization must fall within available weather")

    # Only hours >= initialization contribute to S. The first night uses
    # February 1 00:00--07:00, excluding the January 31 evening hours.
    weather = drivers.frame.loc[drivers.time >= initial]
    sporulation_parameters = config["sporulation_parameters"]
    deposition_parameters = config["deposition_parameters"]
    effective_rain = float(deposition_parameters["effective_deposition_min_rain_mm"])
    heavy_rain = float(deposition_parameters["heavy_rain_washoff_mm"])
    nights = summarize_sporulation_nights(
        weather,
        sporulation_rh_min_pct=sporulation_parameters["sporulation_rh_min_pct"],
    )
    production_events = []
    production_by_date = {}
    for row in nights.itertuples(index=False):
        if row.production_date.year != drivers.season.year:
            continue
        dose = normalized_sporulation_dose(
            row.TMP,
            row.MP,
            minimum_moist_hours=sporulation_parameters["minimum_moist_hours"],
        )
        record = {
            "production_date": pd.Timestamp(row.production_date).date().isoformat(),
            "moist_hours": float(row.MP),
            "mean_moist_temperature_c": float(row.TMP),
            "relative_dose": float(dose),
            "source": "weather_triggered_night",
        }
        production_events.append(record)
        production_by_date[pd.Timestamp(row.production_date).normalize()] = record

    time_index = drivers.time
    attached = []
    airborne = []
    failed_attached = []
    failed_airborne = []
    dispersal = []
    dispersal_events = []
    hourly_inventory = []
    production_added = set()
    next_cohort_id = 0
    availability_hour = SPORULATION_AVAILABILITY_HOUR
    suppressed_weather_production_events = 0

    for index, timestamp in enumerate(time_index):
        if timestamp < initial:
            continue
        current = timestamp
        day = current.tz_localize(None).normalize()
        if current.hour >= availability_hour and day not in production_added:
            event = production_by_date.get(day)
            weather_dose = float(event["relative_dose"]) if event else 0.0
            if (
                weather_dose > 0
                and external_dose_provider is not None
                and hasattr(external_dose_provider, "suppress_weather_at")
                and external_dose_provider.suppress_weather_at(current)
            ):
                weather_dose = 0.0
                suppressed_weather_production_events += 1
            observed_dose = 0.0
            if external_dose_provider is not None:
                observed_dose = float(external_dose_provider.dose_at(current))
                if not math.isfinite(observed_dose) or observed_dose < 0:
                    raise InputError("external observed dose must be finite and non-negative")
            if weather_dose > 0 or observed_dose > 0:
                amount = weather_dose + observed_dose
                attached.append({
                    "cohort_id": next_cohort_id,
                    "source_type": (
                        "weather_plus_observed_background"
                        if weather_dose > 0 and observed_dose > 0
                        else "observed_background"
                        if observed_dose > 0
                        else "weather_triggered_production"
                    ),
                    "production_datetime": current,
                    "initial_amount": amount,
                    "amount": amount,
                    "cumulative_mortality": 0.0,
                    "weather_relative_dose": weather_dose,
                    "observed_background_dose": observed_dose,
                })
                next_cohort_id += 1
            production_added.add(day)

        transfer_allowed = True
        if drivers.settings.dispersal_mode == "wind_gated":
            transfer_allowed = drivers.wind[index] >= drivers.settings.wind_min_m_s

        # Wind moves the attached cohort into a genuine airborne state. The
        # detached mortality clock starts at this same hour.
        if transfer_allowed and attached:
            for cohort in attached:
                if cohort["amount"] <= 0:
                    continue
                airborne.append({
                    "cohort_id": cohort["cohort_id"],
                    "source_type": cohort["source_type"],
                    "production_datetime": iso_timestamp(cohort["production_datetime"]),
                    "dispersal_datetime": current.isoformat(),
                    "dispersal_type": drivers.settings.dispersal_mode,
                    "initial_sporangia_amount": float(cohort["initial_amount"]),
                    "attached_mortality_cumulative": float(cohort["cumulative_mortality"]),
                    "airborne_initial_amount": float(cohort["amount"] * DISPERSAL_DEPOSITION_RATE),
                    "airborne_amount": float(cohort["amount"] * DISPERSAL_DEPOSITION_RATE),
                    "detached_mortality_cumulative": 0.0,
                    "weather_relative_dose": float(cohort.get("weather_relative_dose", cohort["initial_amount"])),
                    "observed_background_dose": float(cohort.get("observed_background_dose", 0.0)),
                })
                dispersal_events.append({
                    "cohort_id": cohort["cohort_id"],
                    "dispersal_datetime": current.isoformat(),
                })
            attached = []
        else:
            surviving = []
            for cohort in attached:
                cohort["cumulative_mortality"] += mortality_rate(
                    drivers.mortality_index[index], attached=True
                )
                cohort["amount"] = _remaining(
                    cohort["initial_amount"], cohort["cumulative_mortality"]
                )
                if cohort["amount"] > 0:
                    surviving.append(cohort)
                else:
                    failed_attached.append({
                        "cohort_id": cohort["cohort_id"],
                        "production_datetime": iso_timestamp(cohort["production_datetime"]),
                        "status": "attached_dead_before_transfer",
                        "terminal_time": current.isoformat(),
                        "infection_time": None,
                        "initial_sporangia_amount": float(cohort["initial_amount"]),
                    })
            attached = surviving

        # Airborne survival and rain-mediated fate are evaluated once per
        # hour. Rain-free hours retain the airborne cohort, so there is no
        # implicit background deposition.
        surviving_airborne = []
        for cohort in airborne:
            cohort["detached_mortality_cumulative"] += mortality_rate(
                drivers.mortality_index[index], attached=False
            )
            cohort["airborne_amount"] = _remaining(
                cohort["airborne_initial_amount"], cohort["detached_mortality_cumulative"]
            )
            rain = max(0.0, float(drivers.rain[index]))
            common = {
                "cohort_id": cohort["cohort_id"],
                "source_type": cohort["source_type"],
                "production_datetime": cohort["production_datetime"],
                "dispersal_datetime": cohort["dispersal_datetime"],
                "dispersal_type": cohort["dispersal_type"],
                "initial_sporangia_amount": cohort["initial_sporangia_amount"],
                "attached_mortality_cumulative": cohort["attached_mortality_cumulative"],
                "detached_mortality_cumulative_at_deposition": cohort["detached_mortality_cumulative"],
                "detached_mortality_applied_at_deposition": True,
                "weather_relative_dose": float(cohort.get("weather_relative_dose", cohort["initial_sporangia_amount"])),
                "observed_background_dose": float(cohort.get("observed_background_dose", 0.0)),
            }
            if cohort["airborne_amount"] <= 0:
                failed_airborne.append({
                    **common,
                    "status": "sporangium_dead_before_deposition",
                    "terminal_time": current.isoformat(),
                    "infection_time": None,
                    "deposition_datetime": None,
                    "airborne_amount": 0.0,
                    "detached_survival_start": cohort["dispersal_datetime"],
                    "detached_survival_end": current.isoformat(),
                    "detached_survival_end_reason": "dead",
                })
            elif rain >= heavy_rain:
                failed_airborne.append({
                    **common,
                    "status": "sporangium_washed_off_before_deposition",
                    "terminal_time": current.isoformat(),
                    "infection_time": None,
                    "deposition_datetime": None,
                    "airborne_amount": float(cohort["airborne_amount"]),
                    "washed_off_rain_mm": rain,
                    "detached_survival_start": cohort["dispersal_datetime"],
                    "detached_survival_end": current.isoformat(),
                    "detached_survival_end_reason": "washed_off",
                })
            elif rain >= effective_rain:
                deposited = {
                    **common,
                    "deposition_datetime": current.isoformat(),
                    "deposition_rain_mm": rain,
                    "airborne_amount": float(cohort["airborne_amount"]),
                    "deposited_amount": float(cohort["airborne_amount"]),
                }
                dispersal.append(deposited)
            else:
                surviving_airborne.append(cohort)
        airborne = surviving_airborne

        hourly_inventory.append({
            "datetime": current.isoformat(),
            "attached_pool": float(sum(c["amount"] for c in attached)),
            "attached_cohort_count": len(attached),
            "airborne_pool": float(sum(c["airborne_amount"] for c in airborne)),
            "airborne_cohort_count": len(airborne),
            "rain_mm": float(max(0.0, drivers.rain[index])),
            "transfer_allowed": bool(transfer_allowed),
        })

    # Weather ending while a cohort is still airborne is a censored detached
    # survival record; it never receives a post-window deposition.
    for cohort in airborne:
        failed_airborne.append({
            "cohort_id": cohort["cohort_id"],
            "source_type": cohort["source_type"],
            "production_datetime": cohort["production_datetime"],
            "dispersal_datetime": cohort["dispersal_datetime"],
            "status": "airborne_censored",
            "terminal_time": time_index[-1].isoformat(),
            "infection_time": None,
            "deposition_datetime": None,
            "initial_sporangia_amount": float(cohort["initial_sporangia_amount"]),
            "airborne_amount": float(cohort["airborne_amount"]),
            "detached_survival_start": cohort["dispersal_datetime"],
            "detached_survival_end": time_index[-1].isoformat(),
            "detached_survival_end_reason": "weather_end",
        })
    for c in attached:
        failed_attached.append({
            "cohort_id": c["cohort_id"],
            "production_datetime": iso_timestamp(c["production_datetime"]),
            "status": "attached_censored",
            "terminal_time": time_index[-1].isoformat(),
            "infection_time": None,
            "initial_sporangia_amount": float(c["initial_amount"]),
        })

    detached = []
    earliest_complete_symptom = None
    early_stop_triggered = False
    for index, cohort in enumerate(dispersal):
        result = advance_deposited_cohort(
            cohort, drivers, config["biological_parameters"]["infection_degree_hours"],
            susceptibility_modulator=susceptibility_modulator,
        )
        detached.append(result)
        if result.get("symptom_window_complete") and result.get("symptom_start"):
            candidate = pd.Timestamp(result["symptom_start"])
            if earliest_complete_symptom is None or candidate < earliest_complete_symptom:
                earliest_complete_symptom = candidate
        if first_window_only and earliest_complete_symptom is not None:
            next_index = index + 1
            if (
                next_index >= len(dispersal)
                or pd.Timestamp(dispersal[next_index]["deposition_datetime"]) > earliest_complete_symptom
            ):
                early_stop_triggered = True
                break
    cohorts = sorted(detached + failed_airborne + failed_attached, key=lambda c: c["cohort_id"])
    infections = sorted([
        {"event_id": c["cohort_id"], "branch": "sporangia", **{key: c[key] for key in (
            "infection_time", "zoospore_release", "symptom_start", "symptom_end",
            "symptom_window_complete", "incubation_censored", "relative_viable_dose_at_release",
            "infection_wet_hours", "infection_mean_temperature", "infection_environmental_index",
            "susceptibility_modulation_factor", "infection_effective_index",
        )}} for c in detached if c["infection_time"] is not None
    ], key=lambda e: (e["infection_time"], e["event_id"]))
    symptom_events = [e for e in infections if e["symptom_start"] is not None]
    first_symptom = min(symptom_events, key=lambda e: e["symptom_start"]) if symptom_events else None

    diagnostics = {
        "initialization": initial.isoformat(),
        "first_night_partial": True,
        "sporulation_parameters": {
            "sporulation_rh_min_pct": float(sporulation_parameters["sporulation_rh_min_pct"]),
            "minimum_moist_hours": float(sporulation_parameters["minimum_moist_hours"]),
        },
        "deposition_parameters": {
            "effective_deposition_min_rain_mm": effective_rain,
            "heavy_rain_washoff_mm": heavy_rain,
        },
        "candidate_cohorts": len(cohorts),
        "candidate_cohorts_before_weather_gate": next_cohort_id,
        "deposited_cohort_count": len(dispersal),
        "infection_events": len(infections),
        "unique_infection_hours": len({e["infection_time"] for e in infections}),
        "cohort_status_counts": dict(Counter(c["status"] for c in cohorts)),
        "first_infection_time": infections[0]["infection_time"] if infections else None,
        "first_symptom_event_id": first_symptom["event_id"] if first_symptom else None,
        "first_symptom_start": first_symptom["symptom_start"] if first_symptom else None,
        "first_symptom_end": first_symptom["symptom_end"] if first_symptom else None,
        "production_events": len(production_events),
        "positive_production_events": sum(e["relative_dose"] > 0 for e in production_events),
        "suppressed_weather_production_events": suppressed_weather_production_events,
        "dispersal_events": len(dispersal_events),
        "deposition_events": len(dispersal),
        "airborne_cohorts": len(dispersal_events),
        "airborne_censored": sum(c["status"] == "airborne_censored" for c in failed_airborne),
        "dead_before_deposition": sum(c["status"] == "sporangium_dead_before_deposition" for c in failed_airborne),
        "washed_off_before_deposition": sum(c["status"] == "sporangium_washed_off_before_deposition" for c in failed_airborne),
        "detached_survival_records": len(detached),
        "total_produced_relative_dose": float(sum(e["relative_dose"] for e in production_events)),
        "total_observed_background_dose": float(
            sum(float(c.get("observed_background_dose", 0.0)) for c in cohorts)
        ),
        "total_deposited_relative_dose": float(sum(c["deposited_amount"] for c in dispersal)),
        "first_production_date": next((e["production_date"] for e in production_events if e["relative_dose"] > 0), None),
        "first_dispersal_datetime": dispersal_events[0]["dispersal_datetime"] if dispersal_events else None,
        "last_dispersal_datetime": dispersal_events[-1]["dispersal_datetime"] if dispersal_events else None,
        "initial_sporangia_pool": 0.0,
        "leaf_wetness_input_available": "leaf_wetness_minutes" in weather.columns,
        "first_window_only": bool(first_window_only),
        "early_stop_triggered": bool(early_stop_triggered),
        "processed_deposited_cohorts": len(detached),
    }

    return {
        "branch": "sporangia",
        "external_dose": {
            "enabled": external_dose_provider is not None,
            "provider": external_dose_provider.description()
            if external_dose_provider is not None and hasattr(external_dose_provider, "description")
            else None,
        },
        "production_events": production_events,
        "dispersal_deposition_cohorts": dispersal,
        "detached_survival": detached,
        "airborne_survival": failed_airborne,
        "cohorts": cohorts,
        "infections": infections,
        "attached_inventory_hourly": hourly_inventory,
        "diagnostics": diagnostics,
    }
