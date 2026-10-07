"""Build daily temperature inputs and simulated phenology for Guangxi grids.

The run is weather-only: observed phenology anchors are not read. Results are
written outside the repository as one daily-input JSON and one phenology JSON
per grid-year.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model.phenology.model import PhenologyConfig, load_config, simulate


DEFAULT_WEATHER_ROOT = Path(r"D:\yaolinjia\Grape\input_weather_json\guangxi")
DEFAULT_OUTPUT_ROOT = Path(r"D:\yaolinjia\Grape\guangxi_phenology")
DEFAULT_CONFIG = REPO_ROOT / "model" / "phenology" / "config_south.json"

_CONFIG: PhenologyConfig | None = None
_CONFIG_PATH: str | None = None


def _init_worker(config_path: str) -> None:
    global _CONFIG, _CONFIG_PATH
    _CONFIG = load_config(config_path)
    _CONFIG_PATH = config_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_53_date(
    daily_rows: list[dict[str, object]],
    stage_dates: dict[str, str | None],
    config: PhenologyConfig,
) -> str | None:
    """Apply the configured 10% BBCH53 fallback between BBCH9 and BBCH65."""

    left = stage_dates.get("budburst")
    right = stage_dates.get("flowering")
    if not left or not right:
        return None
    target = next(item for item in config.stage_targets if item.bbch == 65)
    threshold = target.requirement * 0.1
    left_day = date.fromisoformat(left)
    right_day = date.fromisoformat(right)
    forcing = 0.0
    for row in daily_rows:
        day = date.fromisoformat(str(row["date"]))
        if day <= left_day or day > right_day:
            continue
        forcing += max(float(row["temperature_c"]) - config.forcing_base_c, 0.0)
        if forcing >= threshold:
            return day.isoformat()
    return None


def _daily_input(weather_path: Path, year: int) -> tuple[dict, list[dict[str, object]]]:
    data = json.loads(weather_path.read_text(encoding="utf-8"))
    if int(data.get("year", -1)) != year:
        raise ValueError(f"year mismatch in {weather_path}")
    hourly = data.get("weather_hourly")
    if not isinstance(hourly, list) or not hourly:
        raise ValueError(f"missing weather_hourly in {weather_path}")

    by_day: dict[date, list[float]] = {}
    for row in hourly:
        timestamp = datetime.fromisoformat(str(row["datetime"]))
        if timestamp.year != year:
            raise ValueError(f"weather timestamp outside {year}: {weather_path}")
        temperature = float(row["air_temp"])
        if not math.isfinite(temperature):
            raise ValueError(f"non-finite air temperature in {weather_path}")
        by_day.setdefault(timestamp.date(), []).append(temperature)

    end = date(year, 12, 31)
    expected_days = [date(year, 1, 1) + timedelta(days=i)
                     for i in range((end - date(year, 1, 1)).days + 1)]
    if list(sorted(by_day)) != expected_days:
        raise ValueError(f"weather dates are not continuous in {weather_path}")
    if any(len(by_day[day]) != 24 for day in expected_days):
        raise ValueError(f"weather does not contain 24 hourly temperatures per day: {weather_path}")

    daily = [
        {
            "date": day.isoformat(),
            "temperature_c": sum(by_day[day]) / 24.0,
            "n_hours": 24,
        }
        for day in expected_days
    ]
    metadata = {
        "year": year,
        "grid_id": int(data["grid_id"]),
        "latitude": float(data["latitude"]),
        "longitude": float(data["longitude"]),
        "source_weather": str(weather_path),
    }
    return metadata, daily


def _process_one(task: tuple[str, str, bool]) -> dict[str, object]:
    if _CONFIG is None or _CONFIG_PATH is None:
        raise RuntimeError("phenology worker was not initialized")
    weather_text, output_text, overwrite = task
    weather_path = Path(weather_text)
    output_root = Path(output_text)
    year = int(weather_path.parent.name)
    metadata, daily = _daily_input(weather_path, year)
    stem = weather_path.stem
    daily_path = output_root / "temperature_daily" / str(year) / f"{stem}.json"
    phenology_path = output_root / "phenology" / str(year) / f"{stem}.json"
    if not overwrite and daily_path.exists() and phenology_path.exists():
        return {"status": "skipped", "year": year, "grid_id": metadata["grid_id"]}

    simulation = simulate(daily, _CONFIG, year=year)
    stage_dates = dict(simulation["stage_dates"])
    stage_dates["inflorescence_emergence"] = _stage_53_date(daily, stage_dates, _CONFIG)
    daily_payload = {
        "schema_version": 1,
        "model_input": "daily_mean_temperature",
        **metadata,
        "daily_weather": daily,
    }
    phenology_payload = {
        "schema_version": 1,
        "model": _CONFIG.model_type,
        "config_path": _CONFIG_PATH,
        **metadata,
        "forcing_start": simulation["forcing_start"],
        "stage_dates": stage_dates,
        "stage_targets": [
            {"name": item.name, "bbch": item.bbch, "requirement": item.requirement}
            for item in _CONFIG.stage_targets
        ],
        "bbch53_rule": {
            "left_bbch": 9,
            "right_bbch": 65,
            "forcing_fraction": 0.1,
            "source": "model-derived fallback",
        },
        "daily_input_path": str(daily_path),
    }
    _write_json(daily_path, daily_payload)
    _write_json(phenology_path, phenology_payload)
    return {"status": "written", "year": year, "grid_id": metadata["grid_id"]}


def _tasks(weather_root: Path, output_root: Path, start_year: int, end_year: int,
           overwrite: bool) -> list[tuple[str, str, bool]]:
    tasks = []
    for year in range(start_year, end_year + 1):
        paths = sorted((weather_root / str(year)).glob("*.json"))
        if len(paths) != 342:
            raise ValueError(f"expected 342 weather files for {year}, found {len(paths)}")
        tasks.extend((str(path), str(output_root), overwrite) for path in paths)
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    tasks = _tasks(args.weather_root.resolve(), args.output_root.resolve(),
                   args.start_year, args.end_year, args.overwrite)
    counts = {"written": 0, "skipped": 0}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.config.resolve()),),
    ) as executor:
        for index, result in enumerate(executor.map(_process_one, tasks, chunksize=1), start=1):
            counts[result["status"]] += 1
            if index == 1 or index % 100 == 0 or index == len(tasks):
                print(f"phenology {index}/{len(tasks)} {counts}", flush=True)
    print(f"completed phenology tasks: {len(tasks)}; {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
