"""Summarize Guangxi hourly weather by grid-year and analysis period."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEATHER_ROOT = Path(r"D:\yaolinjia\Grape\input_weather_json\guangxi")
DEFAULT_PHENOLOGY_ROOT = Path(r"D:\yaolinjia\Grape\guangxi_phenology")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "guangxi_2000_2024_weather"

_RAIN_DAY_THRESHOLD = 0.1
_WET_RH_THRESHOLD = 85.0


def _number(value: float | int | None) -> float | None:
    return None if value is None or not np.isfinite(float(value)) else float(value)


def _metrics(
    frame: pd.DataFrame,
    hour_mask: np.ndarray,
    daily_rain: pd.Series,
    daily_wet_hours: pd.Series,
    day_mask: pd.Series,
) -> dict[str, float | int | None]:
    temperatures = frame.loc[hour_mask, "air_temp"].to_numpy(float)
    humidities = frame.loc[hour_mask, "relative_humidity"].to_numpy(float)
    day_mask_array = np.asarray(day_mask, dtype=bool)
    selected_rain = daily_rain.iloc[day_mask_array]
    selected_wet = daily_wet_hours.iloc[day_mask_array]
    if len(temperatures) == 0 or len(selected_rain) == 0:
        return {
            "temperature_mean": None, "temperature_max": None, "temperature_min": None,
            "humidity_mean": None, "humidity_max": None, "humidity_min": None,
            "rainfall_mm": None, "rain_days": None,
            "night_wet_hours_mean": None, "night_wet_hours_max": None,
            "night_wet_hours_min": None, "night_wet_hours_total": None,
        }
    return {
        "temperature_mean": float(np.mean(temperatures)),
        "temperature_max": float(np.max(temperatures)),
        "temperature_min": float(np.min(temperatures)),
        "humidity_mean": float(np.mean(humidities)),
        "humidity_max": float(np.max(humidities)),
        "humidity_min": float(np.min(humidities)),
        "rainfall_mm": float(selected_rain.sum()),
        "rain_days": int((selected_rain >= _RAIN_DAY_THRESHOLD).sum()),
        "night_wet_hours_mean": float(selected_wet.mean()),
        "night_wet_hours_max": float(selected_wet.max()),
        "night_wet_hours_min": float(selected_wet.min()),
        "night_wet_hours_total": float(selected_wet.sum()),
    }


def _add_period(row: dict[str, object], prefix: str,
                metrics: dict[str, float | int | None]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def _process_one(task: tuple[str, str]) -> dict[str, object]:
    weather_path, phenology_path = map(Path, task)
    data = json.loads(weather_path.read_text(encoding="utf-8-sig"))
    year = int(data["year"])
    frame = pd.DataFrame(data["weather_hourly"])
    timestamps = pd.to_datetime(frame["datetime"], errors="raise")
    frame["datetime"] = timestamps
    frame["date"] = timestamps.dt.normalize()
    frame["hour"] = timestamps.dt.hour
    frame["air_temp"] = pd.to_numeric(frame["air_temp"], errors="raise")
    frame["relative_humidity"] = pd.to_numeric(frame["relative_humidity"], errors="raise")
    frame["rain"] = pd.to_numeric(frame["precip_acc_period_adjusted"], errors="raise")

    daily_rain = frame.groupby("date", sort=True)["rain"].sum()
    night = (frame["hour"] >= 20) | (frame["hour"] < 8)
    wet = night & (frame["relative_humidity"] >= _WET_RH_THRESHOLD)
    daily_wet = wet.astype(int).groupby(frame["date"], sort=True).sum()
    dates = frame["date"]
    day_index = daily_rain.index

    phenology = json.loads(phenology_path.read_text(encoding="utf-8-sig"))
    stage_dates = phenology.get("stage_dates", {})
    bbch9_text = stage_dates.get("budburst")
    bbch65_text = stage_dates.get("flowering")
    bbch9 = pd.Timestamp(bbch9_text).normalize() if bbch9_text else None
    bbch65 = pd.Timestamp(bbch65_text).normalize() if bbch65_text else None

    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {
        "annual": (pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)),
        "spring": (pd.Timestamp(year, 3, 1), pd.Timestamp(year, 5, 31)),
        "bbch9_65": (bbch9, bbch65) if bbch9 is not None and bbch65 is not None else None,
    }
    row: dict[str, object] = {
        "year": year,
        "grid_id": int(data["grid_id"]),
        "site_id": str(data.get("site_id", weather_path.stem)),
        "province": str(data.get("province", "Guangxi")),
        "latitude": float(data["latitude"]),
        "longitude": float(data["longitude"]),
        "bbch9_date": bbch9.date().isoformat() if bbch9 is not None else "",
        "bbch65_date": bbch65.date().isoformat() if bbch65 is not None else "",
        "weather_path": str(weather_path),
        "phenology_path": str(phenology_path),
    }
    for prefix, bounds in periods.items():
        if bounds is None:
            _add_period(row, prefix, _metrics(
                frame, np.zeros(len(frame), dtype=bool), daily_rain, daily_wet,
                pd.Series(False, index=day_index),
            ))
            continue
        start, end = bounds
        hour_mask = ((dates >= start) & (dates <= end)).to_numpy()
        day_mask = (day_index >= start) & (day_index <= end)
        _add_period(row, prefix, _metrics(frame, hour_mask, daily_rain, daily_wet, day_mask))
    return row


def _tasks(weather_root: Path, phenology_root: Path) -> list[tuple[str, str]]:
    tasks = []
    for year in range(2000, 2025):
        weather_paths = sorted((weather_root / str(year)).glob("*.json"))
        if len(weather_paths) != 342:
            raise ValueError(f"expected 342 weather files for {year}, found {len(weather_paths)}")
        for weather_path in weather_paths:
            phenology_path = phenology_root / "phenology" / str(year) / weather_path.name
            if not phenology_path.exists():
                raise FileNotFoundError(phenology_path)
            tasks.append((str(weather_path), str(phenology_path)))
    return tasks


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER_ROOT)
    parser.add_argument("--phenology-root", type=Path, default=DEFAULT_PHENOLOGY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    tasks = _tasks(args.weather_root.resolve(), args.phenology_root.resolve())
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(_process_one, tasks, chunksize=1), start=1):
            rows.append(row)
            if index == 1 or index % 200 == 0 or index == len(tasks):
                print(f"weather statistics {index}/{len(tasks)}", flush=True)

    output_root = args.output_root.resolve()
    _write_csv(output_root / "guangxi_grid_year_weather_stats.csv", rows)

    frame = pd.DataFrame(rows)
    numeric = [
        column for column in frame.select_dtypes(include=["number"]).columns
        if column not in {"year", "grid_id", "latitude", "longitude"}
    ]
    baseline = frame.groupby("grid_id", as_index=False)[numeric].mean(numeric_only=True)
    coords = frame.groupby("grid_id", as_index=False)[["site_id", "province", "latitude", "longitude"]].first()
    baseline = coords.merge(baseline, on="grid_id", how="left")
    _write_csv(output_root / "guangxi_grid_weather_baseline_2000_2024.csv", baseline.to_dict("records"))

    annual = (
        frame.groupby("year", as_index=False)
        .agg(
            temperature_mean=("annual_temperature_mean", "mean"),
            temperature_max=("annual_temperature_max", "mean"),
            temperature_min=("annual_temperature_min", "mean"),
            humidity_mean=("annual_humidity_mean", "mean"),
            humidity_max=("annual_humidity_max", "mean"),
            humidity_min=("annual_humidity_min", "mean"),
            rainfall_mm=("annual_rainfall_mm", "mean"),
            rain_days=("annual_rain_days", "mean"),
            spring_temperature_mean=("spring_temperature_mean", "mean"),
            spring_humidity_mean=("spring_humidity_mean", "mean"),
            spring_rainfall_mm=("spring_rainfall_mm", "mean"),
            spring_rain_days=("spring_rain_days", "mean"),
            bbch9_65_temperature_mean=("bbch9_65_temperature_mean", "mean"),
            bbch9_65_humidity_mean=("bbch9_65_humidity_mean", "mean"),
            bbch9_65_rainfall_mm=("bbch9_65_rainfall_mm", "mean"),
            bbch9_65_rain_days=("bbch9_65_rain_days", "mean"),
            annual_night_wet_hours_mean=("annual_night_wet_hours_mean", "mean"),
            spring_night_wet_hours_mean=("spring_night_wet_hours_mean", "mean"),
            bbch9_65_night_wet_hours_mean=("bbch9_65_night_wet_hours_mean", "mean"),
        )
    )
    _write_csv(output_root / "guangxi_regional_annual_weather_stats.csv", annual.to_dict("records"))

    summary = {
        "schema_version": 1,
        "n_years": 25,
        "n_grids": 342,
        "n_grid_year_rows": len(rows),
        "periods": {
            "annual": "January 1 through December 31",
            "spring": "March 1 through May 31",
            "bbch9_65": "inclusive calendar dates from modeled BBCH9 to modeled BBCH65",
        },
        "rain_day_threshold_mm": _RAIN_DAY_THRESHOLD,
        "night_hours": "20:00-23:00 and 00:00-07:00 local time",
        "wet_rh_threshold_pct": _WET_RH_THRESHOLD,
        "outputs": [
            "guangxi_grid_year_weather_stats.csv",
            "guangxi_grid_weather_baseline_2000_2024.csv",
            "guangxi_regional_annual_weather_stats.csv",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
