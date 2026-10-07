"""Host susceptibility modulation for the hourly infection check.

The multiplier is intentionally kept separate from the chill/force model
configuration.  It describes how the calibrated BBCH anchors are converted
to a 0--1 host-progress multiplier during the infection check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .model import simulate as simulate_phenology, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("susceptibility_modulation.json")
DEFAULT_OBSERVED_PHENOLOGY = PROJECT_ROOT / "data" / "phenology_data" / "phenology.xlsx"


@dataclass(frozen=True)
class ModulationNode:
    bbch: int
    name: str
    coefficient: float


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_modulation_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load and validate a host susceptibility modulation configuration."""

    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported susceptibility modulation schema_version")
    if raw.get("interpolation") != "linear":
        raise ValueError("Only linear interpolation is supported")
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("susceptibility modulation requires at least two nodes")
    parsed = []
    previous = -1
    for item in nodes:
        bbch = int(item["bbch"])
        coefficient = float(item["coefficient"])
        if bbch <= previous or not 0.0 <= coefficient <= 1.0:
            raise ValueError("nodes must have increasing BBCH and coefficients in [0, 1]")
        parsed.append({"bbch": bbch, "name": str(item.get("name", f"BBCH{bbch:02d}")),
                       "coefficient": coefficient})
        previous = bbch
    raw["nodes"] = parsed
    phenology_configs = raw.get("phenology_configs")
    if phenology_configs is not None:
        if not isinstance(phenology_configs, dict) or set(phenology_configs) != {"north", "south"}:
            raise ValueError("phenology_configs must provide north and south paths")
        raw["latitude_cutoff_n"] = float(raw.get("latitude_cutoff_n", 32.0))
        if not 0.0 < raw["latitude_cutoff_n"] < 90.0:
            raise ValueError("latitude_cutoff_n must be between 0 and 90")
    elif "phenology_config" not in raw:
        raise ValueError("Provide phenology_config or regional phenology_configs")
    raw["config_path"] = str(config_path)
    return raw


@lru_cache(maxsize=4)
def _read_observed_phenology(path: str) -> pd.DataFrame:
    """Read the local phenology workbook once per process."""

    frame = pd.read_excel(path)
    required = {"latitude", "longitude", "year", "date", "bbch"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Observed phenology is missing columns: {sorted(missing)}")
    result = frame.loc[:, sorted(required)].copy()
    result["latitude"] = pd.to_numeric(result["latitude"], errors="raise")
    result["longitude"] = pd.to_numeric(result["longitude"], errors="raise")
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    result["bbch"] = pd.to_numeric(result["bbch"], errors="raise")
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.date
    return result


def observed_stage_anchors(
    season,
    path: str | Path = DEFAULT_OBSERVED_PHENOLOGY,
    *,
    tolerance: float = 1e-6,
    node_bbchs: tuple[int, ...] = (9, 53, 65, 81, 89),
) -> dict[int, date]:
    """Return first locally observed date at or above each configured BBCH.

    Site-year matching uses the coordinates and year carried by the weather
    season.  A missing stage is intentionally omitted so the caller can retain
    its model-derived fallback rather than inventing an observation.
    """

    observed = _read_observed_phenology(str(Path(path).resolve()))
    selected = observed.loc[
        (observed["year"] == int(season.year))
        & (abs(observed["latitude"] - float(season.latitude)) <= tolerance)
        & (abs(observed["longitude"] - float(season.longitude)) <= tolerance)
    ].sort_values("date")
    if selected.empty:
        return {}
    anchors: dict[int, date] = {}
    for node_bbch in node_bbchs:
        candidates = selected.loc[selected["bbch"] >= node_bbch, "date"]
        if not candidates.empty:
            anchors[node_bbch] = candidates.iloc[0]
    return anchors


class SusceptibilityModulator:
    """Map a site-year hourly timestamp to the configured host multiplier."""

    def __init__(self, season, config_path: str | Path = DEFAULT_CONFIG,
                 observed_stage_dates: dict[int, date] | None = None):
        self.config = load_modulation_config(config_path)
        self.enabled = bool(self.config.get("enabled", True))
        self.stage_dates: dict[int, date | None] = {}
        self._interval_progress: dict[tuple[int, int], dict[pd.Timestamp, float]] = {}
        self.region: str | None = None
        self.phenology_config_path: str | None = None
        self.observed_phenology_path: str | None = None
        if not self.enabled:
            return

        observed_config = self.config.get("observed_phenology")
        if observed_stage_dates is None and observed_config and observed_config.get("enabled", True):
            observed_path = _resolve_project_path(
                observed_config.get("path", DEFAULT_OBSERVED_PHENOLOGY)
            )
            if observed_path.exists():
                node_bbchs = tuple(int(node["bbch"]) for node in self.config["nodes"])
                observed_stage_dates = observed_stage_anchors(
                    season, observed_path, node_bbchs=node_bbchs
                )
                if observed_stage_dates:
                    self.observed_phenology_path = str(observed_path)

        if "phenology_configs" in self.config:
            cutoff = float(self.config["latitude_cutoff_n"])
            self.region = "north" if float(season.latitude) >= cutoff else "south"
            phenology_path = _resolve_project_path(self.config["phenology_configs"][self.region])
        else:
            phenology_path = _resolve_project_path(self.config["phenology_config"])
        self.phenology_config_path = str(phenology_path)
        phenology = load_config(phenology_path)
        daily = (
            season.weather.assign(day=season.weather["datetime"].dt.date)
            .groupby("day", as_index=False)["air_temp"]
            .mean()
            .rename(columns={"day": "date", "air_temp": "temperature_c"})
        )
        try:
            simulation = simulate_phenology(
                [{"date": row.date.isoformat(), "temperature_c": float(row.temperature_c)}
                 for row in daily.itertuples(index=False)],
                phenology,
                year=season.year,
            )
        except ValueError as error:
            # Disease weather files cover the crop year only.  A Chill+Force
            # run needs the previous September–December; when those days are
            # unavailable, preserve the calibrated forcing requirements and
            # use Jan 1 as a documented operational fallback.
            if phenology.model_type != "sequential_chill_force" or "continuous" not in str(error):
                raise
            fallback = replace(phenology, model_type="force_only", chill_requirement=None)
            simulation = simulate_phenology(
                [{"date": row.date.isoformat(), "temperature_c": float(row.temperature_c)}
                 for row in daily.itertuples(index=False)],
                fallback,
                year=season.year,
            )
            self.phenology_source = "regional_model_missing_chill_fallback"
        target_by_bbch = {target.bbch: target for target in phenology.stage_targets}
        dates = simulation["stage_dates"]
        for node in self.config["nodes"]:
            target = target_by_bbch.get(node["bbch"])
            self.stage_dates[node["bbch"]] = (
                pd.Timestamp(dates[target.name]).date() if target and dates.get(target.name) else None
            )
        if observed_stage_dates:
            for bbch, observed in observed_stage_dates.items():
                if int(bbch) in self.stage_dates and observed is not None:
                    self.stage_dates[int(bbch)] = pd.Timestamp(observed).date()
            self.phenology_source = "observed_anchors"
        elif not hasattr(self, "phenology_source"):
            self.phenology_source = "regional_model"

        # If a configured stage is absent from local observations, retain the
        # auxiliary rule as an explicit model-derived fallback.  For the six
        # Guangxi site-years, the observed BBCH53-or-above anchors take
        # precedence and this branch is not used.
        for bbch_text, rule in self.config.get("auxiliary_stage_rules", {}).items():
            bbch = int(bbch_text)
            if self.stage_dates.get(bbch) is not None:
                continue
            left = int(rule["left_bbch"])
            right = int(rule["right_bbch"])
            fraction = float(rule["forcing_fraction"])
            left_date = self.stage_dates.get(left)
            right_date = self.stage_dates.get(right)
            right_target = target_by_bbch.get(right)
            if not left_date or not right_date or right_target is None:
                continue
            threshold = float(right_target.requirement) * fraction
            base = float(
                phenology.early_base_c
                if phenology.early_base_c is not None and right <= 65
                else phenology.forcing_base_c
            )
            forcing = 0.0
            for row in daily.itertuples(index=False):
                day = row.date
                if day <= left_date or day > right_date:
                    continue
                forcing += max(float(row.temperature_c) - base, 0.0)
                if forcing >= threshold:
                    self.stage_dates[bbch] = day
                    break

        # Build a forcing-progress lookup for every adjacent node interval.
        # Each interval uses the calibrated requirement of its right-hand
        # stage, matching the sequential force-only phenology model.
        base_default = float(phenology.forcing_base_c)
        for left, right in zip(self.config["nodes"], self.config["nodes"][1:]):
            left_date = self.stage_dates.get(left["bbch"])
            right_date = self.stage_dates.get(right["bbch"])
            if not left_date or not right_date or right_date <= left_date:
                continue
            requirement = self._interval_requirement(
                left["bbch"], right["bbch"], target_by_bbch
            )
            if requirement is None:
                continue
            base = float(
                phenology.early_base_c
                if phenology.early_base_c is not None and right["bbch"] <= 65
                else base_default
            )
            forcing = 0.0
            interval = {}
            for row in season.weather.itertuples(index=False):
                timestamp = pd.Timestamp(row.datetime).tz_localize("Asia/Shanghai")
                if timestamp.date() <= left_date:
                    continue
                if timestamp.date() >= right_date:
                    continue
                # Phenology requirements are degree-days, while the route
                # process is evaluated hourly.  Convert each hourly forcing
                # contribution from degree-hours to degree-days here.
                forcing += max(float(row.air_temp) - base, 0.0) / 24.0
                interval[timestamp] = max(0.0, min(1.0, forcing / requirement))
            self._interval_progress[(left["bbch"], right["bbch"])] = interval

    def _interval_requirement(self, left_bbch: int, right_bbch: int, target_by_bbch) -> float | None:
        """Return forcing needed between two configured susceptibility nodes."""

        direct = target_by_bbch.get(right_bbch)
        if direct is not None:
            # If the left node is an auxiliary split of this target, retain
            # only the remaining fraction after the split.
            for rule in self.config.get("auxiliary_stage_rules", {}).values():
                if (int(rule["left_bbch"]) == left_bbch
                        and int(rule["right_bbch"]) == right_bbch):
                    return float(direct.requirement) * (1.0 - float(rule["forcing_fraction"]))
            return float(direct.requirement)

        rule = self.config.get("auxiliary_stage_rules", {}).get(str(right_bbch))
        if rule is None or int(rule["left_bbch"]) != left_bbch:
            return None
        target = target_by_bbch.get(int(rule["right_bbch"]))
        if target is None:
            return None
        return float(target.requirement) * float(rule["forcing_fraction"])

    def factor(self, timestamp) -> float:
        """Return the multiplier at one local weather timestamp."""

        if not self.enabled:
            return 1.0
        current = pd.Timestamp(timestamp)
        if current.tz is None:
            current = current.tz_localize("Asia/Shanghai")
        else:
            current = current.tz_convert("Asia/Shanghai")
        nodes = self.config["nodes"]
        values = self.stage_dates
        first_date = values.get(nodes[0]["bbch"])
        if first_date is None:
            return float(self.config.get("missing_stage_coefficient", 1.0))
        if current.date() < first_date:
            return float(nodes[0]["coefficient"])
        for left, right in zip(nodes, nodes[1:]):
            left_date = values.get(left["bbch"])
            right_date = values.get(right["bbch"])
            if not left_date or not right_date or right_date <= left_date:
                continue
            if current.date() < right_date:
                progress = self._interval_progress.get(
                    (left["bbch"], right["bbch"]), {}
                ).get(current, 0.0)
                start_value = float(left["coefficient"])
                end_value = float(right["coefficient"])
                return max(0.0, min(1.0, start_value + progress * (end_value - start_value)))
        return float(nodes[-1]["coefficient"])

    def description(self) -> dict:
        return {
            "enabled": self.enabled,
            "config_path": self.config.get("config_path"),
            "phenology_region": self.region,
            "phenology_config_path": self.phenology_config_path,
            "phenology_source": getattr(self, "phenology_source", "regional_model"),
            "observed_phenology_path": self.observed_phenology_path,
            "stage_dates": {f"{bbch:02d}": value.isoformat() if value else None
                            for bbch, value in self.stage_dates.items()},
            "nodes": self.config.get("nodes", []),
            "progress_basis": self.config.get("progress_basis"),
        }


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OBSERVED_PHENOLOGY",
    "SusceptibilityModulator",
    "load_modulation_config",
    "observed_stage_anchors",
]
