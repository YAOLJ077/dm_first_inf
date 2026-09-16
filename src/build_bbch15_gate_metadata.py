#!/usr/bin/env python3
"""Build auditable BBCH15 output-gate metadata and inject it into RB inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OBS_PATH = ROOT / "data" / "grape_experiment_CK_site_years.xlsx"
OUT_PATH = ROOT / "data" / "fsim_bbch15_gate_metadata.csv"

PLACE_MAP = {"Duan": "DA", "Guilin": "GL", "Mingyang": "NN"}


def main() -> None:
    obs = pd.read_excel(OBS_PATH, sheet_name="CK_Data")
    obs["Date"] = pd.to_datetime(obs["Date"], errors="coerce").dt.normalize()
    obs["GS"] = pd.to_numeric(obs["GS"], errors="coerce")

    keys = obs[["Place", "Year", "Variety"]].drop_duplicates().sort_values(
        ["Place", "Year", "Variety"]
    )
    rows = []
    for place, year, variety in keys.itertuples(index=False):
        group = obs[
            (obs["Place"] == place)
            & (obs["Year"] == year)
            & (obs["Variety"] == variety)
        ].sort_values("Date")
        observed = group[group["GS"] >= 15]
        if not observed.empty:
            date = observed.iloc[0]["Date"]
            source = "observed_GS15"
            method = "first recorded GS >= 15"
        else:
            refs = obs[
                (obs["Place"] == place)
                & (obs["Variety"] == variety)
                & (obs["GS"] >= 15)
            ].copy()
            # One phenology date per reference site-year prevents repeated
            # disease/spore rows from weighting the estimate.
            refs = (
                refs.sort_values("Date")
                .groupby(["Place", "Year", "Variety"], as_index=False)
                .first()
            )
            refs["doy"] = refs["Date"].dt.dayofyear
            if refs.empty:
                date = pd.NaT
                source = "no_gate"
                method = "no same-site same-variety BBCH15 reference"
            else:
                # Preserve the median seasonal timing while retaining the target year.
                doy = int(round(refs["doy"].median()))
                date = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=doy - 1)
                source = "inferred_site_variety_median"
                method = "median DOY of same-site same-variety observed GS >= 15"

        input_key = f"{PLACE_MAP[place]}_{variety}_{int(year)}"
        rb_path = ROOT / "data" / f"{input_key}_grape_rb.json"
        if rb_path.exists():
            with rb_path.open(encoding="utf-8") as handle:
                rb = json.load(handle)
            rb["BBCH_15"] = date.strftime("%Y-%m-%d") if pd.notna(date) else None
            with rb_path.open("w", encoding="utf-8") as handle:
                json.dump(rb, handle, ensure_ascii=False, separators=(",", ":"))

        rows.append(
            {
                "input_key": input_key,
                "Place": place,
                "Year": int(year),
                "Variety": variety,
                "BBCH15_date": date.strftime("%Y-%m-%d") if pd.notna(date) else "",
                "source": source,
                "method": method,
            }
        )

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
