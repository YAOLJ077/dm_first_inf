"""Run the Guangxi S route with an optional observed-dose extension."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from model.FSIM.calculation import DriverSettings, HourlyDrivers, simulate, validate_config
from model.FSIM.cropseason import CropSeason
from model.phenology.susceptibility_modulation import (
    DEFAULT_CONFIG as DEFAULT_SUSCEPTIBILITY_MODULATION_CONFIG,
    SusceptibilityModulator,
)

from .observation import load_site_year_spore_counts
from .pressure import (
    ObservedDoseProvider,
    build_pressure_field,
    observed_zero_dates,
    validate_pressure_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_S_CONFIG = PROJECT_ROOT / "model" / "FSIM" / "config.json"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load and validate the observation-layer configuration."""

    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3 or data.get("model") != "FSIM-S":
        raise ValueError("FSIM-S config must declare model=FSIM-S")
    pressure = validate_pressure_config(data.get("pressure_adjustment"))
    data["pressure_adjustment"] = validate_pressure_config(pressure)
    data["config_path"] = str(config_path)
    return data


def run_spore_plus(
    weather_path: str | Path,
    *,
    s_config_path: str | Path = DEFAULT_S_CONFIG,
    pressure_config: dict | str | Path | None = None,
    pressure_enabled: bool | None = None,
    spore_observations: pd.DataFrame | None = None,
    spore_file: str | Path | None = None,
    susceptibility_modulation_config: str | Path | None = DEFAULT_SUSCEPTIBILITY_MODULATION_CONFIG,
    observed_stage_dates: dict | None = None,
) -> dict:
    """Run FSIM and, when enabled, its strict observed-dose extension."""

    season = CropSeason.read(weather_path)
    config = deepcopy(json.loads(Path(s_config_path).resolve().read_text(encoding="utf-8")))
    validate_config(config)
    settings = DriverSettings(
        dispersal_mode=config["dispersal_mode"],
        wind_min_m_s=config["structure_parameters"]["wind_min_m_s"],
    )
    drivers = HourlyDrivers(season, settings)
    susceptibility_modulator = (
        SusceptibilityModulator(
            season,
            susceptibility_modulation_config,
            observed_stage_dates=observed_stage_dates,
        )
        if susceptibility_modulation_config is not None
        else None
    )
    process = simulate(drivers, config, susceptibility_modulator=susceptibility_modulator)

    spore_config = load_config()
    pressure = dict(spore_config["pressure_adjustment"])
    if isinstance(pressure_config, (str, Path)):
        supplied = json.loads(Path(pressure_config).resolve().read_text(encoding="utf-8"))
        pressure.update(supplied.get("pressure_adjustment", supplied))
    elif pressure_config is not None:
        pressure.update(dict(pressure_config))
    if pressure_enabled is not None:
        pressure["enabled"] = bool(pressure_enabled)
    pressure = validate_pressure_config(pressure)

    pressure_field = None
    spore_process = None
    if pressure["enabled"]:
        if spore_observations is None:
            if spore_file is None:
                raise ValueError("pressure extension requires spore_observations or spore_file")
            spore_observations = load_site_year_spore_counts(spore_file, season)
        pressure_field = build_pressure_field(
            spore_observations,
            drivers.time,
            rolling_days=pressure["rolling_days"],
            reference_level=pressure["reference_level"],
        )
        provider = ObservedDoseProvider(
            pressure_field,
            pressure["beta_sp"],
            observed_zero_weather_gate=pressure["observed_zero_weather_gate"],
            observed_zero_dates=(
                observed_zero_dates(spore_observations)
                if pressure["observed_zero_weather_gate"]
                else None
            ),
        )
        spore_process = simulate(
            drivers,
            config,
            susceptibility_modulator=susceptibility_modulator,
            external_dose_provider=(provider if pressure["beta_sp"] != 0 else None),
        )

    result = {
        "model": "FSIM-S",
        "schema_version": 3,
        "season": {
            "site_id": season.site_id,
            "province": season.province,
            "year": season.year,
            "latitude": season.latitude,
            "longitude": season.longitude,
            "weather_path": str(season.path),
        },
        "parameters": {
            "minimum_moist_hours": float(config["sporulation_parameters"]["minimum_moist_hours"]),
            "sporulation_rh_min_pct": float(config["sporulation_parameters"]["sporulation_rh_min_pct"]),
            "wind_min_m_s": config["structure_parameters"]["wind_min_m_s"],
        },
        "process": process,
        "pressure_config": pressure,
        "susceptibility_modulation": (
            susceptibility_modulator.description()
            if susceptibility_modulator is not None
            else {"enabled": False}
        ),
    }
    if pressure_field is not None:
        pressure_records = pressure_field.copy()
        pressure_records["datetime"] = pressure_records["datetime"].map(lambda value: pd.Timestamp(value).isoformat())
        result["pressure_field"] = pressure_records.to_dict(orient="records")
        result["spore_process"] = spore_process
        result["pressure_adjusted_process"] = spore_process
    return result


__all__ = ["DEFAULT_CONFIG", "DEFAULT_S_CONFIG", "load_config", "run_spore_plus"]
