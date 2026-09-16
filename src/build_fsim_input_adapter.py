#!/usr/bin/env python3
"""Build model-ready site-year manifests and observation files.

The adapter keeps the FSIM implementation unchanged. It standardizes the
current CK observations, links each site-year to its weather JSON, and joins
the fixed disease-onset endpoint table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBS = ROOT / "data" / "grape_experiment_CK_site_years.xlsx"
DEFAULT_ENDPOINTS = ROOT / "data" / "site_year_onset_endpoints.csv"
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "result" / "fsim_inputs"

PLACE_MAP = {"Duan": "DA", "Guilin": "GL", "Mingyang": "NN"}
VARIETY_MAP = {"MPT": "MPT", "WK": "WK", "JF": "JF"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, default=DEFAULT_OBS)
    p.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def load_observations(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="CK_Data")
    required = {"Place", "Year", "Plot", "Variety", "Date", "disease", "Spore", "GS"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Observation file is missing columns: {sorted(missing)}")
    coordinate = "latitude_longitude" if "latitude_longitude" in df.columns else "坐标"
    if coordinate not in df.columns:
        raise ValueError("Observation file must contain latitude_longitude or 坐标")
    df = df.rename(columns={coordinate: "coordinates"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    if df["Date"].isna().any():
        raise ValueError("Observation file contains invalid dates")
    if not df["Plot"].astype(str).str.strip().eq("CK").all():
        raise ValueError("Observation file must contain CK observations only")
    if not df["Place"].isin(PLACE_MAP).all():
        raise ValueError("Observation file contains an unmapped Place value")
    if not df["Variety"].isin(VARIETY_MAP).all():
        raise ValueError("Observation file contains an unmapped Variety value")
    df["place_code"] = df["Place"].map(PLACE_MAP)
    df["model_variety"] = df["Variety"].map(VARIETY_MAP)
    return df


def load_endpoints(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Place", "place_code", "Year", "Plot", "Variety", "model_variety",
        "L", "L_DOY", "U", "U_DOY", "interval_width_days",
        "L_status", "U_status", "onset_censoring",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Endpoint file is missing columns: {sorted(missing)}")
    for col in ["L", "U"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if df["L"].isna().any() or df["U"].isna().any():
        raise ValueError("Current endpoint table must contain both L and U for every site-year")
    if not df["onset_censoring"].eq("interval-censored").all():
        raise ValueError("Current endpoint table contains a non-interval-censored record")
    return df


def weather_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        rb = json.load(f)
    hourly = pd.DataFrame(rb.get("weather_hourly", []))
    if hourly.empty or "datetime" not in hourly:
        raise ValueError(f"Weather JSON has no usable weather_hourly data: {path}")
    dt = pd.to_datetime(hourly["datetime"], errors="coerce")
    return {
        "weather_json": str(path.relative_to(ROOT)),
        "weather_rows": int(len(hourly)),
        "weather_start": dt.min().strftime("%Y-%m-%d"),
        "weather_end": dt.max().strftime("%Y-%m-%d"),
        "rb_place": rb.get("Place", ""),
        "rb_variety": rb.get("variety", ""),
        "rb_bbch53": rb.get("HHCH_53", rb.get("BBCH_53", "")),
        "rb_bbch15": rb.get("BBCH_15", ""),
    }


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    obs = load_observations(args.observations)
    endpoints = load_endpoints(args.endpoints)
    key_cols = ["Place", "Year", "Plot", "Variety", "model_variety"]
    obs_keys = obs[key_cols].drop_duplicates()
    ep_keys = endpoints[key_cols].drop_duplicates()
    if len(obs_keys) != len(endpoints):
        raise ValueError("Observation groups and endpoint rows do not have the same count")
    if not obs_keys.merge(ep_keys, on=key_cols, how="outer", indicator=True)["_merge"].eq("both").all():
        raise ValueError("Observation groups and endpoint rows do not match")

    out_obs = args.output_dir / "observations"
    out_obs.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for key, group in obs.groupby(key_cols, sort=False):
        place, year, plot, variety, model_variety = key
        ep = endpoints[
            (endpoints["Place"] == place)
            & (endpoints["Year"] == year)
            & (endpoints["Plot"] == plot)
            & (endpoints["Variety"] == variety)
        ]
        if len(ep) != 1:
            raise ValueError(f"Expected one endpoint row for {key}")
        ep = ep.iloc[0]
        input_key = f"{PLACE_MAP[place]}_{model_variety}_{int(year)}"
        rb_path = args.data_dir / f"{input_key}_grape_rb.json"
        weather = weather_metadata(rb_path)
        group = group.sort_values("Date").copy()
        group["Date"] = group["Date"].dt.strftime("%Y-%m-%d")
        group = group[["Place", "place_code", "Year", "Plot", "Variety", "model_variety", "Date", "disease", "Spore", "GS", "coordinates"]]
        obs_path = out_obs / f"{input_key}_observations.csv"
        group.to_csv(obs_path, index=False, encoding="utf-8-sig")
        manifest_rows.append({
            "input_key": input_key,
            "Place": place,
            "place_code": PLACE_MAP[place],
            "Year": int(year),
            "Plot": plot,
            "Variety": variety,
            "model_variety": model_variety,
            "observations_csv": str(obs_path.relative_to(ROOT)),
            "L": ep["L"].strftime("%Y-%m-%d"),
            "L_DOY": int(ep["L_DOY"]),
            "U": ep["U"].strftime("%Y-%m-%d"),
            "U_DOY": int(ep["U_DOY"]),
            "interval_width_days": int(ep["interval_width_days"]),
            "onset_censoring": ep["onset_censoring"],
            **weather,
        })

    manifest = pd.DataFrame(manifest_rows).sort_values(["place_code", "Year"]).reset_index(drop=True)
    manifest_path = args.output_dir / "site_year_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    config = {
        "observations_source": str(args.observations.relative_to(ROOT)),
        "endpoints_source": str(args.endpoints.relative_to(ROOT)),
        "place_map": PLACE_MAP,
        "variety_map": VARIETY_MAP,
        "n_site_years": int(len(manifest)),
        "endpoint_definition": "(L, U] with observed last-negative L and first-positive U",
    }
    (args.output_dir / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return manifest, manifest_path


if __name__ == "__main__":
    args = parse_args()
    manifest, path = build(args)
    print(f"Saved {len(manifest)} site-year inputs to {path}")
    print(manifest[["input_key", "L", "U", "interval_width_days", "weather_json"]].to_string(index=False))
