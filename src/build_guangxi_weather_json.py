"""Build one unified Guangxi weather archive from the jf files and GEE CSVs.

The shapefile grid IDs are one-based, while the existing weather JSON files
use the zero-based full-grid ID in their filenames and JSON metadata. The
script joins them through ``weather_grid_id`` and keeps the existing JSON
weather contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "weather_data"
DEFAULT_SOURCE_ROOT = DEFAULT_INPUT_ROOT / "jf"
DEFAULT_TARGET_ROOT = DEFAULT_INPUT_ROOT / "regional"
DEFAULT_GRID_INDEX = REPO_ROOT / "data" / "shapefile" / "gx_grid_inside_0.25.csv"
DEFAULT_GEE_DIR = REPO_ROOT / "tmp"

CSV_REQUIRED = {
    "weather_grid_id",
    "grid_id",
    "latitude",
    "longitude",
    "year",
    "datetime",
    "air_temp",
    "dew_point",
    "relative_humidity",
    "precip_acc_period_adjusted",
    "wind_speed",
}
WEATHER_FIELDS = (
    "datetime",
    "dew_point",
    "precip_acc_period_adjusted",
    "relative_humidity",
    "air_temp",
    "wind_speed",
)


def _number(value: str) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return int(number) if number.is_integer() else number


def load_grid_map(index_path: Path, source_root: Path, year: int) -> dict[int, dict[str, object]]:
    """Load the 342-cell mask and resolve each cell to an existing jf filename."""

    grids: dict[int, dict[str, object]] = {}
    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            boundary_id = int(row["id"])
            weather_id = boundary_id - 1
            if weather_id in grids:
                raise ValueError(f"duplicate boundary grid id: {boundary_id}")
            matches = sorted((source_root / str(year)).glob(f"grid_{weather_id}_*.json"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected one source file for weather_grid_id={weather_id}, "
                    f"found {len(matches)}"
                )
            grids[weather_id] = {
                "boundary_id": boundary_id,
                "weather_grid_id": weather_id,
                "latitude": round(float(row["lat"]), 3),
                "longitude": round(float(row["lon"]), 3),
                "filename": matches[0].name,
            }
    if len(grids) != 342:
        raise ValueError(f"expected 342 Guangxi cells, found {len(grids)}")
    return grids


def _json_prefix(metadata: dict[str, object]) -> str:
    text = json.dumps(metadata, ensure_ascii=False, indent=4)
    return text[:-2] + ',\n    "weather_hourly": [\n'


def convert_gee_year(
    csv_paths: list[Path],
    target_root: Path,
    grids: dict[int, dict[str, object]],
    year: int,
) -> int:
    """Stream GEE batch CSVs into one JSON file per grid."""

    output_dir = target_root / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, object] = {}
    counts = {weather_id: 0 for weather_id in grids}
    expected_next = {weather_id: datetime(year, 1, 1) for weather_id in grids}

    try:
        for csv_path in sorted(csv_paths):
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                missing = CSV_REQUIRED - fields
                if missing:
                    raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")
                for row_number, row in enumerate(reader, start=2):
                    weather_id = int(row["weather_grid_id"])
                    if weather_id not in grids:
                        raise ValueError(
                            f"{csv_path.name}:{row_number} contains non-Guangxi weather_grid_id "
                            f"{weather_id}"
                        )
                    if int(row["year"]) != year:
                        raise ValueError(f"{csv_path.name}:{row_number} has unexpected year")
                    if int(row["grid_id"]) != weather_id + 1:
                        raise ValueError(f"{csv_path.name}:{row_number} has inconsistent grid IDs")

                    timestamp = datetime.fromisoformat(row["datetime"])
                    if timestamp != expected_next[weather_id]:
                        raise ValueError(
                            f"{csv_path.name}:{row_number} has non-contiguous weather for "
                            f"grid {weather_id}: expected {expected_next[weather_id].isoformat()}, "
                            f"got {timestamp.isoformat()}"
                        )
                    expected_next[weather_id] += timedelta(hours=1)

                    values = {
                        "datetime": timestamp.strftime("%Y-%m-%dT%H:%M"),
                        "dew_point": _number(row["dew_point"]),
                        # ERA5 exports can contain tiny negative floating-point
                        # residuals around zero; FSIM requires non-negative rain.
                        "precip_acc_period_adjusted": max(
                            0.0, float(row["precip_acc_period_adjusted"])
                        ),
                        "relative_humidity": _number(row["relative_humidity"]),
                        "air_temp": _number(row["air_temp"]),
                        "wind_speed": _number(row["wind_speed"]),
                    }
                    if weather_id not in handles:
                        metadata = {
                            "province": "Guang Xi",
                            "crop_eppo_code": "VITVI",
                            "time_zone": "asia/beijing",
                            "cultivation_method": "open field",
                            "year": year,
                            "latitude": grids[weather_id]["latitude"],
                            "longitude": grids[weather_id]["longitude"],
                            "grid_id": weather_id,
                        }
                        output_path = output_dir / str(grids[weather_id]["filename"])
                        output = output_path.open("w", encoding="utf-8", newline="\n")
                        output.write(_json_prefix(metadata))
                        handles[weather_id] = output
                    if counts[weather_id]:
                        handles[weather_id].write(",\n")
                    handles[weather_id].write(
                        "        " + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
                    )
                    counts[weather_id] += 1
    finally:
        for output in handles.values():
            output.write("\n    ]\n}\n")
            output.close()

    expected_hours = 8784 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 8760
    incomplete = {weather_id: count for weather_id, count in counts.items() if count != expected_hours}
    if incomplete:
        raise ValueError(f"incomplete {year} GEE grids: {incomplete}")
    if any(value != datetime(year + 1, 1, 1) for value in expected_next.values()):
        raise ValueError(f"unexpected final timestamps in {year} GEE data")
    return len(handles)


def copy_existing_year(
    source_root: Path,
    target_root: Path,
    grids: dict[int, dict[str, object]],
    year: int,
) -> int:
    output_dir = target_root / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    for weather_id, grid in grids.items():
        source = source_root / str(year) / str(grid["filename"])
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / str(grid["filename"]))
    return len(grids)


def build_archive(
    source_root: Path,
    target_root: Path,
    grid_index: Path,
    gee_dir: Path,
    start_year: int,
    end_year: int,
) -> None:
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    target_root.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        grids = load_grid_map(grid_index, source_root, year)
        if year == 2004:
            csv_paths = sorted(gee_dir.glob("ERA5_GX_2004_batch_*.csv"))
            if len(csv_paths) != 7:
                raise FileNotFoundError(
                    f"expected 7 GEE CSV batches for 2004, found {len(csv_paths)}"
                )
            count = convert_gee_year(csv_paths, target_root, grids, year)
        else:
            count = copy_existing_year(source_root, target_root, grids, year)
        print(f"{year}: wrote {count} Guangxi grid JSON files", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--grid-index", type=Path, default=DEFAULT_GRID_INDEX)
    parser.add_argument("--gee-dir", type=Path, default=DEFAULT_GEE_DIR)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    build_archive(
        args.source_root.resolve(),
        args.target_root.resolve(),
        args.grid_index.resolve(),
        args.gee_dir.resolve(),
        args.start_year,
        args.end_year,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
