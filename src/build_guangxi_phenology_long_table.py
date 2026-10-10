"""Build a separate Guangxi grid-year phenology table for one profile."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHENOLOGY_ROOT = ROOT / "result" / "regional_phenology"
DEFAULT_OUTPUT_ROOT = ROOT / "result/guangxi_2000_2024_onset_windows"
STAGES = ("budburst", "inflorescence_emergence", "flowering", "veraison")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("table", "vine"), required=True)
    parser.add_argument("--phenology-root", type=Path, default=DEFAULT_PHENOLOGY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    config_path = ROOT / f"model/phenology/profile_config_{args.profile}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    for year in range(2000, 2025):
        paths = sorted((args.phenology_root / args.profile / "phenology" / str(year)).glob("*.json"))
        if len(paths) != 342:
            raise ValueError(f"Expected 342 phenology files in {year}, found {len(paths)}")
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["profile"] != args.profile or data["year"] != year:
                raise ValueError(f"Profile/year mismatch: {path}")
            row = {
                "phenology_profile": args.profile,
                "year": year,
                "grid_id": data["grid_id"],
                "latitude": data["latitude"],
                "longitude": data["longitude"],
                "forcing_start": data["forcing_start"],
                "base_temperature_c": config["model"]["base_temperature_c"],
                "start_doy": config["model"]["start_doy"],
                "bbch53_forcing_fraction": data["bbch53_rule"]["forcing_fraction"],
            }
            for stage in STAGES:
                value = data["stage_dates"].get(stage)
                row[f"{stage}_date"] = value or ""
                row[f"{stage}_doy"] = date.fromisoformat(value).timetuple().tm_yday if value else ""
            for target in data["stage_targets"]:
                row[f"bbch{target['bbch']:02d}_incremental_gdd"] = target["requirement"]
            row.update({
                "weather_path": data["source_weather"],
                "daily_input_path": data["daily_input_path"],
                "phenology_path": str(path.resolve()),
                "phenology_config_path": data["config_path"],
            })
            rows.append(row)
    if len({(row["year"], row["grid_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate phenology grid-years")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"guangxi_2000_2024_phenology_{args.profile}.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} {args.profile} phenology rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
