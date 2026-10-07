"""Run asexual sporangia cohorts through infection and symptom appearance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.FSIM.cropseason import CropSeason, InputError  # noqa: E402
from model.FSIM.model import run_cropseason  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weather",
        action="append",
        help="One site-year JSON; repeat this option for selected files",
    )
    parser.add_argument(
        "--weather-dir",
        default=str(PROJECT_ROOT / "data" / "weather_data" / "site_year"),
        help="Directory of site-year JSON files when --weather is omitted",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "result" / "FSIM"),
        help="Directory for one result JSON per site-year",
    )
    parser.add_argument("--config", default=None, help="Override model/FSIM/config.json")
    parser.add_argument("--susceptibility-modulation-config", default="model/phenology/susceptibility_modulation.json",
                        help="Host susceptibility modulation config; use 'none' to disable")
    parser.add_argument("--dispersal-mode", choices=["merged", "wind_gated"], default=None)
    parser.add_argument("--wind-min-m-s", type=float, default=None,
                        help="Explicit threshold for wind_gated transfer; no literature default")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    weather_paths = (
        [Path(path).resolve() for path in args.weather]
        if args.weather
        else sorted(Path(args.weather_dir).resolve().glob("*.json"))
    )
    if args.limit is not None:
        weather_paths = weather_paths[: args.limit]
    if not weather_paths:
        parser.error("No weather JSON files found; use src/convert_site_year_weather.py first")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for weather_path in weather_paths:
        target = output_dir / f"{weather_path.stem}.json"
        if target.exists() and not args.overwrite:
            parser.error(f"Output exists: {target}; use --overwrite")
        season = CropSeason.read(weather_path)
        try:
            result = run_cropseason(
                season,
                args.config or PROJECT_ROOT / "model" / "FSIM" / "config.json",
                dispersal_mode=args.dispersal_mode,
                wind_min_m_s=args.wind_min_m_s,
                susceptibility_modulation_config=(None if args.susceptibility_modulation_config == "none"
                                                   else args.susceptibility_modulation_config),
            )
        except InputError as error:
            parser.error(str(error))
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        diagnostics = result["process"]["diagnostics"]
        row = {
            "site_id": season.site_id,
            "province": season.province,
            "year": season.year,
            "weather_file": str(weather_path),
            "result_file": str(target),
            "production_events": diagnostics["production_events"],
            "positive_production_events": diagnostics["positive_production_events"],
            "dispersal_events": diagnostics["dispersal_events"],
            "total_deposited_relative_dose": diagnostics["total_deposited_relative_dose"],
            "first_dispersal_datetime": diagnostics["first_dispersal_datetime"],
            "infection_events": diagnostics["infection_events"],
            "first_infection_time": diagnostics["first_infection_time"],
            "first_symptom_event_id": diagnostics["first_symptom_event_id"],
            "first_symptom_start": diagnostics["first_symptom_start"],
            "first_symptom_end": diagnostics["first_symptom_end"],
        }
        rows.append(row)
        print(
            f"{season.site_id}_{season.year}: "
            f"{row['positive_production_events']} positive production events, "
            f"{row['dispersal_events']} dispersal events, "
            f"{row['infection_events']} infected cohorts -> {target.name}",
            flush=True,
        )

    manifest = {
        "schema_version": 3,
        "model": "FSIM",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "production_to_symptom_appearance",
        "results": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} FSIM simulations to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
