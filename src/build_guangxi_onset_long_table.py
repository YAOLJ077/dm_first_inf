"""Build one long CSV row for every Guangxi grid-year FSIM result."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISEASE_ROOT = REPO_ROOT / "result" / "regional_disease_window"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "guangxi_2000_2024_onset_windows"
PROFILES = ("table", "vine")

STAGES = {
    "9": "budburst",
    "53": "inflorescence_emergence",
    "65": "flowering",
    "81": "veraison",
    "89": "post_veraison",
}

FIELDS = [
    "phenology_profile", "bbch53_forcing_fraction", "year", "grid_id", "site_id", "province", "latitude", "longitude",
    "first_symptom_start", "first_symptom_end",
    "first_symptom_start_date", "first_symptom_end_date",
    "first_symptom_start_doy", "first_symptom_end_doy",
    "first_symptom_midpoint_doy",
    "symptom_duration_hours", "symptom_duration_days",
    "symptom_window_complete", "window_status",
    "first_infection_time", "first_infection_date", "first_infection_doy",
    "infection_to_symptom_hours", "infection_to_symptom_days",
    "budburst_date", "budburst_doy",
    "inflorescence_emergence_date", "inflorescence_emergence_doy",
    "flowering_date", "flowering_doy", "veraison_date", "veraison_doy",
    "post_veraison_date", "post_veraison_doy",
    "production_events", "positive_production_events", "dispersal_events",
    "deposition_events", "infection_events", "first_production_date",
    "first_dispersal_datetime", "early_stop_triggered",
    "phenology_source", "weather_path", "phenology_path",
]


def _parsed(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _date(value: str | None) -> str:
    parsed = _parsed(value)
    return parsed.date().isoformat() if parsed else ""


def _doy(value: str | None) -> str:
    parsed = _parsed(value)
    return str(parsed.timetuple().tm_yday) if parsed else ""


def _duration_hours(start: str | None, end: str | None) -> str:
    first, last = _parsed(start), _parsed(end)
    if first is None or last is None:
        return ""
    return f"{(last - first).total_seconds() / 3600.0:.6f}"


def _row(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    diagnostics = data.get("diagnostics", {})
    stage_dates = data.get("phenology_stage_dates", {})
    start = data.get("first_symptom_start")
    end = data.get("first_symptom_end")
    infection = data.get("first_infection_time")
    duration = _duration_hours(start, end)
    lag = _duration_hours(infection, start)
    if start and end:
        status = "complete"
    elif start:
        status = "incomplete"
    else:
        status = "no_symptom"

    row: dict[str, object] = {
        "year": data["year"],
        "phenology_profile": data.get("phenology_profile", ""),
        "bbch53_forcing_fraction": data["bbch53_rule"]["forcing_fraction"],
        "grid_id": data["grid_id"],
        "site_id": data["site_id"],
        "province": data["province"],
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "first_symptom_start": start or "",
        "first_symptom_end": end or "",
        "first_symptom_start_date": _date(start),
        "first_symptom_end_date": _date(end),
        "first_symptom_start_doy": _doy(start),
        "first_symptom_end_doy": _doy(end),
        "first_symptom_midpoint_doy": (int(_doy(start)) + int(_doy(end))) / 2 if start and end else "",
        "symptom_duration_hours": duration,
        "symptom_duration_days": f"{float(duration) / 24.0:.6f}" if duration else "",
        "symptom_window_complete": bool(data.get("symptom_window_complete", False)),
        "window_status": status,
        "first_infection_time": infection or "",
        "first_infection_date": _date(infection),
        "first_infection_doy": _doy(infection),
        "infection_to_symptom_hours": lag,
        "infection_to_symptom_days": f"{float(lag) / 24.0:.6f}" if lag else "",
        "production_events": diagnostics.get("production_events", ""),
        "positive_production_events": diagnostics.get("positive_production_events", ""),
        "dispersal_events": diagnostics.get("dispersal_events", ""),
        "deposition_events": diagnostics.get("deposition_events", ""),
        "infection_events": diagnostics.get("infection_events", ""),
        "first_production_date": diagnostics.get("first_production_date", "") or "",
        "first_dispersal_datetime": diagnostics.get("first_dispersal_datetime", "") or "",
        "early_stop_triggered": diagnostics.get("early_stop_triggered", ""),
        "phenology_source": data.get("phenology_source", ""),
        "weather_path": data.get("weather_path", ""),
        "phenology_path": data.get("phenology_path", ""),
    }
    for bbch, name in STAGES.items():
        value = stage_dates.get(bbch) or ""
        row[f"{name}_date"] = value
        row[f"{name}_doy"] = _doy(value)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease-root", type=Path, default=DEFAULT_DISEASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    disease_root = args.disease_root.resolve() / args.profile
    output_root = args.output_root.resolve()
    files = []
    for year in range(2000, 2025):
        year_files = sorted((disease_root / str(year)).glob("*.json"))
        if len(year_files) != 342:
            raise ValueError(f"expected 342 disease files for {year}, found {len(year_files)}")
        files.extend(year_files)
    rows = [_row(path) for path in files]
    if any(row["phenology_profile"] != args.profile for row in rows):
        raise ValueError(f"Disease files contain a different profile from {args.profile}")
    fractions = {row["bbch53_forcing_fraction"] for row in rows}
    if len(fractions) != 1:
        raise ValueError("Disease files contain different BBCH53 forcing fractions")
    if len({(row["year"], row["grid_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate grid-year disease results")
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"guangxi_2000_2024_onset_windows_{args.profile}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "source": str(disease_root),
        "output": str(csv_path),
        "profile": args.profile,
        "bbch53_forcing_fraction": fractions.pop(),
        "years": list(range(2000, 2025)),
        "n_years": 25,
        "n_grids": 342,
        "n_rows": len(rows),
        "complete_windows": sum(row["window_status"] == "complete" for row in rows),
        "no_symptom": sum(row["window_status"] == "no_symptom" for row in rows),
        "incomplete_windows": sum(row["window_status"] == "incomplete" for row in rows),
        "date_convention": "ISO-8601 local Asia/Shanghai timestamps; DOY uses local calendar date",
    }
    (output_root / f"summary_{args.profile}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
