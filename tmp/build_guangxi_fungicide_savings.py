"""Add potential fungicide savings estimates to the Guangxi onset long table.

The management schedule is the fixed 2022 Duan schedule from the workbook.
Sprays on the seven calendar days before first symptom (day -7 through day -1)
are treated as the effective prevention window. Sprays earlier than that window
are counted as potentially avoidable. For a grid-year without a modeled symptom
window, all scheduled sprays are counted under the explicit no-symptom scenario
and the row is marked accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONSET_TABLE = (
    REPO_ROOT / "result" / "guangxi_2000_2024_onset_windows"
)
DEFAULT_MANAGEMENT = REPO_ROOT / "data" / "management_data" / "management_data.xlsx"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "guangxi_2000_2024_fungicides"
PROFILES = ("table", "vine")
XML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

EXTRA_FIELDS = [
    "management_schedule_basis_year",
    "management_schedule_basis_site",
    "management_spray_count",
    "management_spray_doys",
    "effective_window_start_date",
    "effective_window_end_date",
    "effective_window_start_doy",
    "effective_window_end_doy",
    "effective_window_spray_count",
    "effective_window_spray_doys",
    "potential_savable_spray_count",
    "potential_savable_spray_fraction",
    "potential_savable_spray_doys",
    "potential_saving_status",
]


def _shared_strings(zfile: ZipFile) -> list[str]:
    root = ET.fromstring(zfile.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{XML_NS}}}t"))
        for item in root.findall(f"{{{XML_NS}}}si")
    ]


def _read_management_doys(path: Path) -> list[int]:
    """Read the workbook with stdlib XML so openpyxl is not required."""

    with ZipFile(path) as zfile:
        strings = _shared_strings(zfile)
        root = ET.fromstring(zfile.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in root.findall(f".//{{{XML_NS}}}sheetData/{{{XML_NS}}}row"):
        values: dict[str, str] = {}
        for cell in row.findall(f"{{{XML_NS}}}c"):
            ref = cell.get("r", "")
            column = re.match(r"[A-Z]+", ref)
            if not column:
                continue
            value_node = cell.find(f"{{{XML_NS}}}v")
            value = "" if value_node is None else (value_node.text or "")
            if cell.get("t") == "s" and value:
                value = strings[int(value)]
            values[column.group()] = value
        rows.append([values.get("A", ""), values.get("B", "")])
    if not rows or rows[0] != ["Date", "Doy"]:
        raise ValueError(f"Unexpected management workbook header: {rows[:1]}")
    doys = [int(float(row[1])) for row in rows[1:] if row[1] != ""]
    if not doys or any(doy < 1 or doy > 366 for doy in doys):
        raise ValueError(f"Invalid management DOY values: {doys}")
    return doys


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value).date()


def _format_doys(values: list[int]) -> str:
    return ";".join(str(value) for value in values)


def _format_dates(values: list[date]) -> str:
    return ";".join(value.isoformat() for value in values)


def _augment(row: dict[str, str], management_doys: list[int]) -> dict[str, object]:
    year = int(row["year"])
    start = _parse_date(row.get("first_symptom_start_date", ""))
    schedule = [date(year, 1, 1) + timedelta(days=doy - 1) for doy in management_doys]
    if start is None:
        effective_start = effective_end = None
        effective = []
        saved = schedule
        status = "no_modeled_symptom_all_sprays_counted"
    else:
        effective_start = start - timedelta(days=7)
        effective_end = start - timedelta(days=1)
        effective = [day for day in schedule if effective_start <= day <= effective_end]
        saved = [day for day in schedule if day < effective_start]
        status = "complete_symptom_window"

    row.update({
        "management_schedule_basis_year": 2022,
        "management_schedule_basis_site": "Duan",
        "management_spray_count": len(schedule),
        "management_spray_doys": _format_doys(management_doys),
        "effective_window_start_date": effective_start.isoformat() if effective_start else "",
        "effective_window_end_date": effective_end.isoformat() if effective_end else "",
        "effective_window_start_doy": effective_start.timetuple().tm_yday if effective_start else "",
        "effective_window_end_doy": effective_end.timetuple().tm_yday if effective_end else "",
        "effective_window_spray_count": len(effective),
        "effective_window_spray_doys": _format_doys([day.timetuple().tm_yday for day in effective]),
        "potential_savable_spray_count": len(saved),
        "potential_savable_spray_fraction": f"{len(saved) / len(schedule):.6f}",
        "potential_savable_spray_doys": _format_doys([day.timetuple().tm_yday for day in saved]),
        "potential_saving_status": status,
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onset-table", type=Path, default=DEFAULT_ONSET_TABLE)
    parser.add_argument("--management", type=Path, default=DEFAULT_MANAGEMENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()

    onset_table = args.onset_table.resolve()
    if onset_table.is_dir():
        onset_table = onset_table / f"guangxi_2000_2024_onset_windows_{args.profile}.csv"
    with onset_table.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"phenology_profile", "bbch53_forcing_fraction", "year", "grid_id", "first_symptom_start_date"}
    missing = required - set(base_fields)
    if missing:
        raise ValueError(f"Onset table is missing columns: {sorted(missing)}")
    if len(rows) != 8550:
        raise ValueError(f"Expected 8550 onset rows, found {len(rows)}")
    if any(row["phenology_profile"] != args.profile for row in rows):
        raise ValueError(f"Onset table contains a different profile from {args.profile}")
    fractions = {float(row["bbch53_forcing_fraction"]) for row in rows}
    if len(fractions) != 1:
        raise ValueError("Onset table contains different BBCH53 forcing fractions")
    if len({(row["year"], row["grid_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate grid-year onset rows")
    management_doys = _read_management_doys(args.management.resolve())
    result_rows = [_augment(row, management_doys) for row in rows]

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"guangxi_2000_2024_fungicide_savings_{args.profile}.csv"
    fields = base_fields + [field for field in EXTRA_FIELDS if field not in base_fields]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)

    summary = {
        "schema_version": 1,
        "profile": args.profile,
        "bbch53_forcing_fraction": fractions.pop(),
        "source_onset_table": str(onset_table),
        "source_management_workbook": str(args.management.resolve()),
        "management_schedule_basis_year": 2022,
        "management_schedule_basis_site": "Duan",
        "management_spray_doys": management_doys,
        "management_spray_count": len(management_doys),
        "effective_window": "first_symptom_date - 7 days through first_symptom_date - 1 day",
        "n_rows": len(result_rows),
        "complete_symptom_rows": sum(row["potential_saving_status"] == "complete_symptom_window" for row in result_rows),
        "no_symptom_rows": sum(row["potential_saving_status"] == "no_modeled_symptom_all_sprays_counted" for row in result_rows),
        "total_potential_savable_sprays": sum(int(row["potential_savable_spray_count"]) for row in result_rows),
        "mean_potential_savable_sprays_per_grid_year": sum(int(row["potential_savable_spray_count"]) for row in result_rows) / len(result_rows),
        "output": str(output_path),
    }
    (output_root / f"summary_{args.profile}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
