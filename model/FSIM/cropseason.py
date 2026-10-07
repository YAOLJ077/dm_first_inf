"""Crop-season input handling and site-year weather conversion for S.

The source CSV files are already expressed in Beijing local clock time despite
the historical ``time_utc`` column name.  Conversion therefore preserves the
clock values and only renames fields to the weather JSON contract used here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np


LOCAL_TIMEZONE = "Asia/Shanghai"
WEATHER_COLUMNS = (
    "datetime",
    "dew_point",
    "precip_acc_period_adjusted",
    "relative_humidity",
    "air_temp",
    "wind_speed",
)
CSV_COLUMNS = (
    "time_utc",
    "relative_humidity_pct",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_m_s",
    "dew_point_c",
    "latitude",
    "longitude",
)


class InputError(ValueError):
    """Raised when a crop-season or weather input violates its contract."""


def _finite_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise InputError(f"Weather {column} contains missing/non-finite values")
        frame[column] = values.astype(float)


def validate_weather_frame(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Validate and localize an hourly weather frame without shifting clocks."""

    missing = set(WEATHER_COLUMNS) - set(frame.columns)
    if missing:
        raise InputError(f"Weather is missing columns: {sorted(missing)}")
    result = frame.loc[:, list(WEATHER_COLUMNS)].copy()
    dates = pd.to_datetime(result["datetime"], errors="raise")
    if dates.dt.tz is not None:
        raise InputError("Weather JSON must contain naive local timestamps")
    expected = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left"
    )
    if not pd.DatetimeIndex(dates).equals(expected):
        raise InputError(
            f"Weather must contain every local hour of {year} in order "
            f"({len(expected)} rows expected, {len(result)} received)"
        )
    result["datetime"] = dates
    _finite_numeric(
        result,
        ["dew_point", "precip_acc_period_adjusted", "relative_humidity", "air_temp", "wind_speed"],
    )
    if not result["relative_humidity"].between(0, 100).all():
        raise InputError("Relative humidity must be between 0 and 100 percent")
    if (result[["precip_acc_period_adjusted", "wind_speed"]] < 0).any().any():
        raise InputError("Rainfall and wind speed cannot be negative")
    return result


@dataclass(frozen=True)
class CropSeason:
    """One Guangxi site-year weather season used by both S routes."""

    path: Path
    site_id: str
    province: str
    year: int
    latitude: float
    longitude: float
    timezone: str
    cultivation_method: str
    weather: pd.DataFrame
    metadata: dict

    @classmethod
    def read(cls, filename: str | Path) -> "CropSeason":
        path = Path(filename).resolve()
        if not path.exists():
            raise InputError(f"Weather JSON not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise InputError(f"Invalid weather JSON: {path}") from error

        required = {"year", "latitude", "longitude", "time_zone", "weather_hourly"}
        missing = required - set(data)
        if missing:
            raise InputError(f"Weather JSON missing fields: {sorted(missing)}")
        if not isinstance(data["year"], int) or isinstance(data["year"], bool):
            raise InputError("Weather year must be an integer")
        timezone = str(data["time_zone"]).lower()
        if timezone not in {"asia/beijing", "asia/shanghai"}:
            raise InputError("Weather must declare Beijing/Shanghai local time")
        hourly = validate_weather_frame(pd.DataFrame(data["weather_hourly"]), data["year"])
        try:
            latitude = float(data["latitude"])
            longitude = float(data["longitude"])
        except (TypeError, ValueError) as error:
            raise InputError("Weather latitude/longitude must be numeric") from error
        return cls(
            path=path,
            site_id=str(data.get("site_id", path.stem)),
            province=str(data.get("province", "China")),
            year=data["year"],
            latitude=latitude,
            longitude=longitude,
            timezone=LOCAL_TIMEZONE,
            cultivation_method=str(data.get("cultivation_method", "open field")),
            weather=hourly,
            metadata=data,
        )

    def fingerprint(self) -> dict:
        return {"path": str(self.path), "sha256": sha256(self.path.read_bytes()).hexdigest()}


def _json_number(value):
    value = float(value)
    return int(value) if value.is_integer() else value


def convert_site_year_csv(
    source: str | Path,
    output: str | Path | None = None,
    *,
    request_datetime: str | None = None,
    province: str = "China",
    overwrite: bool = False,
) -> Path:
    """Convert one site-year CSV to the DA_MPT-style weather JSON contract."""

    source_path = Path(source).resolve()
    target = Path(output).resolve() if output else source_path.with_suffix(".json")
    if target.exists() and not overwrite:
        raise InputError(f"Output already exists: {target}; use --overwrite")
    frame = pd.read_csv(source_path)
    missing = set(CSV_COLUMNS) - set(frame.columns)
    if missing:
        raise InputError(f"{source_path.name} missing CSV columns: {sorted(missing)}")
    timestamps = pd.to_datetime(frame["time_utc"], errors="raise")
    if timestamps.dt.tz is not None:
        raise InputError("Source CSV timestamps must be naive Beijing local time")
    year = int(timestamps.iloc[0].year)
    precipitation = pd.to_numeric(frame["precipitation_mm"], errors="raise")
    missing_precipitation = int(precipitation.isna().sum())
    # The four affected CSVs have short missing rainfall blocks while all
    # other weather variables are present.  A missing hourly accumulation is
    # treated as 0 mm and recorded in the JSON metadata; source CSVs are not
    # edited.
    precipitation = precipitation.fillna(0.0)
    weather = pd.DataFrame(
        {
            "datetime": timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "dew_point": frame["dew_point_c"].map(_json_number),
            "precip_acc_period_adjusted": precipitation.map(_json_number),
            "relative_humidity": frame["relative_humidity_pct"].map(_json_number),
            "air_temp": frame["temperature_c"].map(_json_number),
            "wind_speed": frame["wind_speed_m_s"].map(_json_number),
        }
    )
    validated = validate_weather_frame(weather, year)
    latitudes = pd.to_numeric(frame["latitude"], errors="raise").dropna().unique()
    longitudes = pd.to_numeric(frame["longitude"], errors="raise").dropna().unique()
    if len(latitudes) != 1 or len(longitudes) != 1:
        raise InputError("Each site-year CSV must contain one latitude/longitude")
    request_datetime = request_datetime or date.today().isoformat()
    data = {
        "province": str(province),
        "crop_eppo_code": "VITVI",
        "time_zone": "asia/beijing",
        "cultivation_method": "open field",
        "year": year,
        "latitude": _json_number(latitudes[0]),
        "longitude": _json_number(longitudes[0]),
        "site_id": source_path.stem.rsplit("_", 1)[0],
        "request_datetime": request_datetime,
        "weather_daily": None,
        "crop_establishment": {
            "date": f"{year}-01-01",
            "method": "weather_year_baseline",
            "growth_stage": None,
            "growth_stage_scale": None,
        },
        "weather_hourly": [
            {
                "datetime": row.datetime.strftime("%Y-%m-%dT%H:%M:%S"),
                "dew_point": row.dew_point,
                "precip_acc_period_adjusted": row.precip_acc_period_adjusted,
                "relative_humidity": row.relative_humidity,
                "air_temp": row.air_temp,
                "wind_speed": row.wind_speed,
            }
            for row in validated.itertuples(index=False)
        ],
        "source_csv": str(source_path),
        "source_time_column_note": "time_utc values preserved as Beijing local clock time per project metadata",
        "source_missing_value_handling": {
            "precipitation_mm_filled_as_zero": missing_precipitation,
            "other_fields_filled": 0,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def convert_site_year_directory(
    directory: str | Path,
    *,
    request_datetime: str | None = None,
    province_lookup: Callable[[Path], str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every CSV in a site-year directory in deterministic order."""

    paths = sorted(Path(directory).glob("*.csv"))
    if not paths:
        raise InputError(f"No site-year CSV files found in {directory}")
    return [
        convert_site_year_csv(
            path,
            request_datetime=request_datetime,
            province=province_lookup(path) if province_lookup else "China",
            overwrite=overwrite,
        )
        for path in paths
    ]
