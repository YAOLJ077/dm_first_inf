"""Public API for the independent FSIM sporangia route."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .calculation import (
    DriverSettings,
    HourlyDrivers,
    simulate,
    validate_config,
)
from .cropseason import CropSeason, InputError
from model.phenology.susceptibility_modulation import DEFAULT_CONFIG as DEFAULT_SUSCEPTIBILITY_MODULATION_CONFIG
from model.phenology.susceptibility_modulation import SusceptibilityModulator


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def describe(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read and validate the sporangia-route configuration."""

    path = Path(config_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data)
    if data.get("implementation_status") != "implemented_research_version":
        raise InputError("Sporangia configuration must declare implementation_status")
    return {
        "config": data,
        "config_path": str(path),
        "capabilities": {
            "weather_only": True,
            "production": True,
            "dispersal_and_deposition": True,
            "attached_and_detached_survival": True,
            "infection_or_incubation": True,
            "calibration": False,
        },
    }


def run_cropseason(
    season: CropSeason,
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    dispersal_mode: str | None = None,
    wind_min_m_s: float | None = None,
    susceptibility_modulation_config: str | Path | None = DEFAULT_SUSCEPTIBILITY_MODULATION_CONFIG,
    observed_stage_dates: dict | None = None,
    first_window_only: bool = False,
) -> dict:
    """Run recurrent asexual cohorts through infection and symptom windows."""

    description = describe(config_path)
    config = deepcopy(description["config"])
    if dispersal_mode is not None:
        config["dispersal_mode"] = dispersal_mode
        if dispersal_mode == "merged":
            config["structure_parameters"]["wind_min_m_s"] = None
    if wind_min_m_s is not None:
        config["structure_parameters"]["wind_min_m_s"] = wind_min_m_s
    validate_config(config)
    settings = DriverSettings(
        dispersal_mode=config["dispersal_mode"],
        wind_min_m_s=config["structure_parameters"]["wind_min_m_s"],
    )
    drivers = HourlyDrivers(season, settings)
    susceptibility_modulator = (
        SusceptibilityModulator(season, susceptibility_modulation_config,
                                observed_stage_dates=observed_stage_dates)
        if susceptibility_modulation_config is not None else None
    )
    process = simulate(
        drivers,
        config,
        susceptibility_modulator=susceptibility_modulator,
        first_window_only=first_window_only,
    )
    return {
        "schema_version": 3,
        "model": "fsim_sporangia_infection_route",
        "reference": config["reference"],
        "scope": "production_to_symptom_appearance",
        "season": {
            "site_id": season.site_id,
            "province": season.province,
            "year": season.year,
            "latitude": season.latitude,
            "longitude": season.longitude,
            "cultivation_method": season.cultivation_method,
            "timezone": season.timezone,
            "weather_path": str(season.path),
        },
        "config": config,
        "drivers": drivers.description(),
        "susceptibility_modulation": susceptibility_modulator.description() if susceptibility_modulator is not None else {"enabled": False},
        "process": process,
        "provenance": {"weather": season.fingerprint(), "config": description["config_path"]},
    }
