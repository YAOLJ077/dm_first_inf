"""Run the deployed FSIM route for every Guangxi grid-year.

The host-susceptibility dates come from the weather-only phenology files.  No
observed phenology workbook is used.  One compact JSON is written per grid and
year, with the first symptom window as the primary result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.FSIM.cropseason import CropSeason
from model.FSIM.model import run_cropseason


DEFAULT_WEATHER_ROOT = Path(r"D:\yaolinjia\Grape\input_weather_json\guangxi")
DEFAULT_PHENOLOGY_ROOT = Path(r"D:\yaolinjia\Grape\guangxi_phenology")
DEFAULT_OUTPUT_ROOT = Path(r"D:\yaolinjia\Grape\guangxi_disease_window")
DEFAULT_CONFIG = REPO_ROOT / "result" / "deployment_parameters" / "fsim_deployment_config.json"
DEFAULT_SUSCEPTIBILITY = REPO_ROOT / "model" / "phenology" / "susceptibility_modulation.json"
PROFILE_CONFIGS = {
    "table": REPO_ROOT / "model" / "phenology" / "profile_config_table.json",
    "vine": REPO_ROOT / "model" / "phenology" / "profile_config_vine.json",
}

_CONFIG_PATH: str | None = None
_SUSCEPTIBILITY_PATH: str | None = None


def _init_worker(config_path: str, susceptibility_path: str) -> None:
    global _CONFIG_PATH, _SUSCEPTIBILITY_PATH
    _CONFIG_PATH = config_path
    _SUSCEPTIBILITY_PATH = susceptibility_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_dates(phenology: dict) -> dict[int, str]:
    names = {
        "budburst": 9,
        "inflorescence_emergence": 53,
        "flowering": 65,
        "veraison": 81,
        "post_veraison": 89,
    }
    dates = phenology.get("stage_dates", {})
    result: dict[int, str] = {}
    for name, bbch in names.items():
        value = dates.get(name)
        if value:
            result[bbch] = str(value)
    return result


def _process_one(task: tuple[str, str, str, str, str, bool]) -> dict[str, object]:
    if _CONFIG_PATH is None or _SUSCEPTIBILITY_PATH is None:
        raise RuntimeError("FSIM worker was not initialized")
    weather_text, phenology_text, output_text, profile, profile_config_text, overwrite = task
    weather_path = Path(weather_text)
    phenology_path = Path(phenology_text)
    output_root = Path(output_text)
    year = int(weather_path.parent.name)
    output_path = output_root / str(year) / f"{weather_path.stem}.json"
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "year": year, "grid_id": weather_path.stem}

    season = CropSeason.read(weather_path)
    phenology = json.loads(phenology_path.read_text(encoding="utf-8"))
    if (phenology.get("profile") != profile or phenology["year"] != season.year
            or phenology["grid_id"] != season.metadata["grid_id"]
            or Path(phenology["config_path"]).resolve() != Path(profile_config_text).resolve()):
        raise ValueError(f"Phenology profile, grid-year, or configuration mismatch: {phenology_path}")
    stage_dates = _stage_dates(phenology)
    result = run_cropseason(
        season,
        _CONFIG_PATH,
        susceptibility_modulation_config=_SUSCEPTIBILITY_PATH,
        observed_stage_dates=stage_dates,
        phenology_config_path=profile_config_text,
        first_window_only=True,
    )
    diagnostics = result["process"]["diagnostics"]
    susceptibility = result["susceptibility_modulation"]
    # The FSIM API uses the observed_stage_dates argument for any externally
    # supplied anchors.  Here those dates are simulated from hourly weather,
    # so make the provenance explicit in the persisted result.
    susceptibility["phenology_source"] = "weather_only_precomputed"
    susceptibility["observed_phenology_path"] = None
    if (phenology["bbch53_rule"]["forcing_fraction"]
            != susceptibility["auxiliary_stage_rules"]["53"]["forcing_fraction"]):
        raise ValueError(f"Precomputed BBCH53 rule differs from disease configuration: {phenology_path}")
    payload = {
        "schema_version": 1,
        "model": "fsim_sporangia_infection_route",
        "result_type": "first_symptom_window",
        "year": season.year,
        "grid_id": int(season.metadata.get("grid_id", season.site_id.rsplit("_", 1)[-1])),
        "site_id": season.site_id,
        "province": season.province,
        "latitude": season.latitude,
        "longitude": season.longitude,
        "weather_path": str(weather_path),
        "phenology_path": str(phenology_path),
        "phenology_source": "weather_only_precomputed",
        "phenology_profile": profile,
        "phenology_profile_config": str(Path(profile_config_text).resolve()),
        "phenology_stage_dates": {str(bbch): value for bbch, value in sorted(stage_dates.items())},
        "bbch53_rule": phenology["bbch53_rule"],
        "deployment_config": str(Path(_CONFIG_PATH).resolve()),
        "susceptibility_config": str(Path(_SUSCEPTIBILITY_PATH).resolve()),
        "first_symptom_start": diagnostics["first_symptom_start"],
        "first_symptom_end": diagnostics["first_symptom_end"],
        "symptom_window_complete": diagnostics["first_symptom_start"] is not None and diagnostics["first_symptom_end"] is not None,
        "first_infection_time": diagnostics["first_infection_time"],
        "diagnostics": diagnostics,
        "susceptibility_modulation": susceptibility,
    }
    _write_json(output_path, payload)
    return {
        "status": "written",
        "year": year,
        "grid_id": payload["grid_id"],
        "symptom": payload["first_symptom_start"] is not None,
    }


def _tasks(weather_root: Path, phenology_root: Path, output_root: Path,
           profile: str, profile_config: Path, start_year: int, end_year: int,
           overwrite: bool) -> list[tuple[str, str, str, str, str, bool]]:
    tasks = []
    for year in range(start_year, end_year + 1):
        weather_paths = sorted((weather_root / str(year)).glob("*.json"))
        phenology_dir = phenology_root / profile / "phenology" / str(year)
        if len(weather_paths) != 342:
            raise ValueError(f"expected 342 weather files for {year}, found {len(weather_paths)}")
        for weather_path in weather_paths:
            phenology_path = phenology_dir / weather_path.name
            if not phenology_path.exists():
                raise FileNotFoundError(f"missing phenology file: {phenology_path}")
            tasks.append((str(weather_path), str(phenology_path), str(output_root / profile),
                          profile, str(profile_config.resolve()), overwrite))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-root", type=Path, default=DEFAULT_WEATHER_ROOT)
    parser.add_argument("--phenology-root", type=Path, default=DEFAULT_PHENOLOGY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), required=True,
                        help="Regional cultivar profile used for phenology and disease")
    parser.add_argument("--profile-config", type=Path, default=None,
                        help="Optional explicit profile configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--susceptibility-config", type=Path, default=DEFAULT_SUSCEPTIBILITY)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    profile_config = (args.profile_config or PROFILE_CONFIGS[args.profile]).resolve()
    if json.loads(profile_config.read_text(encoding="utf-8")).get("profile") != args.profile:
        parser.error("Selected configuration must match --profile")
    tasks = _tasks(args.weather_root.resolve(), args.phenology_root.resolve(),
                   args.output_root.resolve(), args.profile, profile_config,
                   args.start_year, args.end_year, args.overwrite)
    counts = {"written": 0, "skipped": 0, "symptom": 0}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.config.resolve()), str(args.susceptibility_config.resolve())),
    ) as executor:
        for index, result in enumerate(executor.map(_process_one, tasks, chunksize=1), start=1):
            counts[result["status"]] += 1
            if result.get("symptom"):
                counts["symptom"] += 1
            if index == 1 or index % 100 == 0 or index == len(tasks):
                print(f"disease {index}/{len(tasks)} {counts}", flush=True)
    print(f"completed disease tasks: {len(tasks)}; {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
