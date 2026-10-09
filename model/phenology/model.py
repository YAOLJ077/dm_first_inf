"""Small sequential chilling-forcing and forcing-only models.

The model follows the sequential structure described in
``reference/phenology/manuscript_ylj.docx``:

1. accumulate chilling from 1 September of the previous year;
2. start forcing on the day after the chilling requirement is reached;
3. reset forcing after each stage and detect the next stage when its
   requirement is reached.

The code deliberately leaves biological requirements as explicit inputs. They
must be calibrated from the prepared phenology data before operational use.
No cultivar, regional offset, residual-learning, or disease-data parameter is
included here.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_CONFIG = Path(__file__).with_name("observed_anchor_fallback.json")


@dataclass(frozen=True)
class StageTarget:
    """One ordered BBCH target and its forcing requirement."""

    name: str
    bbch: int
    requirement: float


@dataclass(frozen=True)
class PhenologyConfig:
    """Model settings and calibrated requirements."""

    chill_requirement: float | None
    stage_targets: tuple[StageTarget, ...]
    chill_start_month: int = 9
    chill_start_day: int = 1
    chill_min_c: float = -1.7
    chill_opt_c: float = 1.7
    chill_max_c: float = 13.2
    forcing_base_c: float = 0.0
    simulation_end_month: int = 12
    simulation_end_day: int = 31
    model_type: str = "sequential_chill_force"
    early_base_c: float | None = None
    forcing_start_month: int = 1
    forcing_start_day: int = 1
    forcing_start_doy: int | None = None

    def start_date(self, year: int) -> date:
        """Forcing-only uses a fixed crop-year date; CF starts in prior autumn."""
        if self.model_type == "force_only" and self.forcing_start_doy is not None:
            return date(year, 1, 1) + timedelta(days=self.forcing_start_doy - 1)
        return (date(year, self.forcing_start_month, self.forcing_start_day) if self.model_type == "force_only" else
                date(year - 1, self.chill_start_month, self.chill_start_day))

    def validate(self) -> None:
        if self.model_type not in {"sequential_chill_force", "force_only"}:
            raise ValueError("Unknown phenology model_type")
        if self.model_type == "force_only":
            if self.forcing_start_doy is not None and not 1 <= self.forcing_start_doy <= 365:
                raise ValueError("Forcing start DOY must be between 1 and 365")
            start = self.start_date(2001)
            if start > date(2001, self.simulation_end_month, self.simulation_end_day):
                raise ValueError("Forcing start must not be after simulation end")
        values = [self.chill_min_c, self.chill_opt_c,
                  self.chill_max_c, self.forcing_base_c]
        if self.early_base_c is not None:
            values.append(self.early_base_c)
        if self.model_type == "sequential_chill_force":
            if self.chill_requirement is None or self.chill_requirement <= 0:
                raise ValueError("chill_requirement must be positive for the CF model")
            values.append(self.chill_requirement)
        elif self.chill_requirement is not None:
            raise ValueError("force_only must have chill_requirement=null")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Model parameters must be finite")
        if len({t.name for t in self.stage_targets}) != len(self.stage_targets):
            raise ValueError("Stage names must be unique")
        if not self.stage_targets:
            raise ValueError("stage_targets cannot be empty")
        if self.chill_min_c >= self.chill_opt_c or self.chill_opt_c >= self.chill_max_c:
            raise ValueError("chilling cardinals must be min < optimum < max")
        previous_bbch = -1
        for target in self.stage_targets:
            if target.bbch <= previous_bbch:
                raise ValueError("stage BBCH targets must be strictly increasing")
            if not math.isfinite(target.requirement) or target.requirement <= 0:
                raise ValueError("stage requirements must be positive")
            previous_bbch = target.bbch


def _parse_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _chill_response(temp_c: float, config: PhenologyConfig) -> float:
    """Triangular daily chilling response with fixed cardinals."""

    if temp_c <= config.chill_min_c or temp_c >= config.chill_max_c:
        return 0.0
    if temp_c <= config.chill_opt_c:
        return (temp_c - config.chill_min_c) / (config.chill_opt_c - config.chill_min_c)
    return (config.chill_max_c - temp_c) / (config.chill_max_c - config.chill_opt_c)


def _force_response(temp_c: float, config: PhenologyConfig, bbch: int | None = None) -> float:
    base = (config.early_base_c if config.early_base_c is not None and bbch is not None and bbch <= 65
            else config.forcing_base_c)
    return max(float(temp_c) - base, 0.0)


def load_config(path: str | Path = DEFAULT_CONFIG) -> PhenologyConfig:
    """Load a calibrated JSON configuration."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Adapt the calibrated Table/Vine schema to the existing sequential model.
    if raw.get("profile") in {"table", "vine"}:
        model = raw.get("model", {})
        if model.get("family") != "linear_gdd" or model.get("temperature_response") != "linear":
            raise ValueError("profile configuration must use the linear_gdd family")
        start_doy = int(model.get("start_doy", 1))
        start = date(2001, 1, 1) + timedelta(days=start_doy - 1)
        profile_stage_names = {9: "budburst", 65: "flowering", 81: "veraison"}
        targets = tuple(
            StageTarget(
                profile_stage_names.get(
                    int(item["bbch"]),
                    str(item.get("name", f"BBCH{int(item['bbch']):02d}")),
                ),
                int(item["bbch"]),
                float(item["incremental_gdd"]),
            )
            for item in raw.get("stage_thresholds", [])
        )
        config = PhenologyConfig(
            chill_requirement=None,
            stage_targets=targets,
            model_type="force_only",
            forcing_base_c=float(model["base_temperature_c"]),
            forcing_start_month=start.month,
            forcing_start_day=start.day,
            forcing_start_doy=start_doy,
            simulation_end_month=12,
            simulation_end_day=31,
        )
        config.validate()
        return config
    kind = raw.get("model")
    if kind not in {"sequential_chill_force", "force_only"}:
        raise ValueError("configuration model must be sequential_chill_force or force_only")
    if kind == "sequential_chill_force" and raw.get("chill_requirement") is None:
        raise ValueError("chill_requirement is not calibrated")
    targets = []
    for item in raw.get("stage_targets", []):
        if item.get("requirement") is None:
            raise ValueError(f"stage requirement is not calibrated: {item.get('name')}")
        targets.append(StageTarget(item["name"], int(item["bbch"]), float(item["requirement"])))
    config = PhenologyConfig(
        chill_requirement=float(raw["chill_requirement"]) if raw.get("chill_requirement") is not None else None,
        stage_targets=tuple(targets),
        model_type=kind,
        early_base_c=float(raw["early_base_c"]) if raw.get("early_base_c") is not None else None,
        forcing_start_month=int(raw.get("forcing_start_month", 1)),
        forcing_start_day=int(raw.get("forcing_start_day", 1)),
        chill_start_month=int(raw.get("chill_start_month", 9)),
        chill_start_day=int(raw.get("chill_start_day", 1)),
        chill_min_c=float(raw.get("chill_min_c", -1.7)),
        chill_opt_c=float(raw.get("chill_opt_c", 1.7)),
        chill_max_c=float(raw.get("chill_max_c", 13.2)),
        forcing_base_c=float(raw.get("forcing_base_c", 0.0)),
        simulation_end_month=int(raw.get("simulation_end_month", 12)),
        simulation_end_day=int(raw.get("simulation_end_day", 31)),
    )
    config.validate()
    return config


def daily_weather_from_csv(
    path: str | Path,
    *,
    site_year: str | None = None,
    datetime_column: str = "datetime_utc",
    temperature_column: str = "temperature_C",
) -> list[dict[str, object]]:
    """Read hourly weather CSV and return daily mean temperatures.

    The prepared ERA5 table contains ``datetime_utc`` and ``temperature_C``.
    When the table contains multiple site-years, pass ``site_year`` to select
    one series before daily aggregation. Any CSV with equivalent columns can
    be used. Dates are grouped using the date component of the supplied
    datetime column.
    """

    values: dict[date, list[float]] = defaultdict(list)
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if site_year is not None:
                if "site_year" not in row:
                    raise ValueError(f"Missing site_year column in {path}")
                if row.get("site_year") != str(site_year):
                    continue
            if not row.get(datetime_column) or not row.get(temperature_column):
                raise ValueError(f"Missing timestamp or temperature in {path}")
            dt = datetime.fromisoformat(row[datetime_column].replace("Z", "+00:00"))
            temp = float(row[temperature_column])
            if dt in seen or dt.minute or dt.second or dt.microsecond or not math.isfinite(temp):
                raise ValueError(f"Duplicate/nonhourly timestamp or nonfinite temperature in {path}")
            seen.add(dt)
            values[dt.date()].append(temp)
    if not values:
        selected = f" for site-year {site_year}" if site_year is not None else ""
        raise ValueError(f"No hourly weather rows found in {path}{selected}")
    if any(len(temps) != 24 for temps in values.values()):
        selected = f" for site-year {site_year}" if site_year is not None else ""
        raise ValueError(f"Require 24 valid hourly temperatures per day in {path}{selected}")
    return [
        {"date": day.isoformat(), "temperature_c": sum(temps) / len(temps), "n_hours": len(temps)}
        for day, temps in sorted(values.items())
    ]


def simulate(
    daily_weather: Iterable[Mapping[str, object]],
    config: PhenologyConfig,
    *,
    year: int,
) -> dict[str, object]:
    """Simulate one crop year from daily mean temperatures.

    ``daily_weather`` must cover ``config.start_date(year)`` through the end
    of ``year``. The output includes chilling completion, stage dates, and a
    daily broad-stage label that can be joined to infection events later.
    """

    config.validate()
    rows = sorted(
        ({"date": _parse_date(row["date"]), "temperature_c": float(row["temperature_c"])} for row in daily_weather),
        key=lambda row: row["date"],
    )
    if not rows:
        raise ValueError("daily_weather is empty")
    simulation_start = config.start_date(year)
    simulation_end = date(year, config.simulation_end_month, config.simulation_end_day)
    rows = [row for row in rows if simulation_start <= row["date"] <= simulation_end]
    if not rows:
        raise ValueError("daily_weather does not cover the requested crop year")
    expected = [simulation_start + timedelta(days=i)
                for i in range((simulation_end - simulation_start).days + 1)]
    if [row["date"] for row in rows] != expected:
        raise ValueError("Require continuous, unique daily weather over the full simulation period")
    if not all(math.isfinite(row["temperature_c"]) for row in rows):
        raise ValueError("Daily temperatures must be finite")

    chill_total = 0.0
    chill_completion_date: date | None = None
    for row in rows:
        if config.model_type == "force_only":
            break
        if chill_completion_date is None:
            chill_total += _chill_response(row["temperature_c"], config)
            if chill_total >= config.chill_requirement:
                chill_completion_date = row["date"]

    stage_dates: dict[str, str | None] = {target.name: None for target in config.stage_targets}
    forcing_total = 0.0
    target_index = 0
    forcing_start = (
        chill_completion_date + timedelta(days=1) if chill_completion_date is not None else None
    )
    if config.model_type == "force_only":
        forcing_start = simulation_start
    if forcing_start is not None:
        for row in rows:
            if row["date"] < forcing_start or target_index >= len(config.stage_targets):
                continue
            target = config.stage_targets[target_index]
            forcing_total += _force_response(row["temperature_c"], config, target.bbch)
            if forcing_total >= target.requirement:
                stage_dates[target.name] = row["date"].isoformat()
                forcing_total = 0.0
                target_index += 1

    # Three observed anchors define four broad daily intervals.
    stage_order = ("budburst", "flowering", "veraison")
    stage_names = ("pre_budburst", "budburst_to_flowering", "flowering_to_veraison", "veraison_onward")
    daily_stage = []
    for row in rows:
        day = row["date"]
        stage_id = max((i + 1 for i, name in enumerate(stage_order)
                        if stage_dates.get(name) is not None and day >= _parse_date(stage_dates[name])), default=0)
        daily_stage.append({"date": day.isoformat(), "stage_id": stage_id, "stage": stage_names[stage_id]})

    return {
        "year": year,
        "model": config.model_type,
        "chill_start": simulation_start.isoformat() if config.model_type == "sequential_chill_force" else None,
        "chill_completion_date": chill_completion_date.isoformat() if chill_completion_date else None,
        "chill_units": round(chill_total, 6) if config.model_type == "sequential_chill_force" else None,
        "forcing_start": forcing_start.isoformat() if forcing_start else None,
        "stage_dates": stage_dates,
        "daily_stage": daily_stage,
    }
