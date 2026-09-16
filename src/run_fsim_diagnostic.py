#!/usr/bin/env python3
"""Run fixed-parameter FSIM diagnostics for the six study site-years."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "result" / "fsim_inputs" / "site_year_manifest.csv"
DEFAULT_OUTPUT = ROOT / "result" / "fsim_diagnostic_vpd"

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from model.config import config as FSIM_CONFIG  # noqa: E402
from model.crop_season import CropSeasonFirst  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compare-no-bbch15-gate",
        action="store_true",
        help="Also run FSIM with BBCH15 removed from the RB metadata.",
    )
    parser.add_argument(
        "--use-wind-gate", action=argparse.BooleanOptionalAction, default=None,
        help="Enable M1 or use --no-use-wind-gate for M0; default follows model config.",
    )
    return parser.parse_args()


def frame_count(value: Any) -> int:
    return int(len(value)) if isinstance(value, pd.DataFrame) else 0


def first_window(df: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.NaT, pd.NaT
    required = {"oil_spot_start", "oil_spot_end"}
    if not required.issubset(df.columns):
        return pd.NaT, pd.NaT
    valid = df.dropna(subset=["oil_spot_start", "oil_spot_end"]).copy()
    if valid.empty:
        return pd.NaT, pd.NaT
    valid["oil_spot_start"] = pd.to_datetime(valid["oil_spot_start"])
    valid["oil_spot_end"] = pd.to_datetime(valid["oil_spot_end"])
    row = valid.sort_values("oil_spot_start").iloc[0]
    return row["oil_spot_start"], row["oil_spot_end"]


def interval_metrics(
    observed_l: pd.Timestamp,
    observed_u: pd.Timestamp,
    predicted_start: pd.Timestamp,
    predicted_end: pd.Timestamp,
) -> dict[str, Any]:
    if any(pd.isna(x) for x in [predicted_start, predicted_end]):
        return {
            "has_prediction": 0,
            "overlap": np.nan,
            "signed_boundary_error_days": np.nan,
            "absolute_boundary_error_days": np.nan,
            "observed_midpoint": observed_l + (observed_u - observed_l) / 2,
            "predicted_midpoint": pd.NaT,
            "midpoint_error_days": np.nan,
            "absolute_midpoint_error_days": np.nan,
            "predicted_window_width_days": np.nan,
        }

    p_start = pd.Timestamp(predicted_start)
    p_end = pd.Timestamp(predicted_end)
    observed_mid = observed_l + (observed_u - observed_l) / 2
    predicted_mid = p_start + (p_end - p_start) / 2

    if p_end < observed_l:
        signed = (p_end - observed_l).total_seconds() / 86400
        overlap = 0
    elif p_start > observed_u:
        signed = (p_start - observed_u).total_seconds() / 86400
        overlap = 0
    else:
        signed = 0.0
        overlap = 1

    midpoint_error = (predicted_mid - observed_mid).total_seconds() / 86400
    return {
        "has_prediction": 1,
        "overlap": overlap,
        "signed_boundary_error_days": signed,
        "absolute_boundary_error_days": abs(signed),
        "observed_midpoint": observed_mid,
        "predicted_midpoint": predicted_mid,
        "midpoint_error_days": midpoint_error,
        "absolute_midpoint_error_days": abs(midpoint_error),
        "predicted_window_width_days": (p_end - p_start).total_seconds() / 86400,
    }


def format_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M")
    return out


def run_case(row: pd.Series, compare_no_gate: bool = False, use_wind_gate=None) -> tuple[dict[str, Any], dict[str, Any]]:
    rb_path = ROOT / row["weather_json"]
    with rb_path.open(encoding="utf-8") as f:
        rb = json.load(f)

    observed_l = pd.to_datetime(row["L"])
    observed_u = pd.to_datetime(row["U"])
    cs = CropSeasonFirst(csrb=rb, config=copy.deepcopy(FSIM_CONFIG))
    status = "ok"
    error_stage = ""
    error_type = ""
    error_message = ""
    try:
        error_stage = "simulate"
        cs.simulate(use_wind_gate=use_wind_gate)
        error_stage = ""
    except Exception as exc:  # Preserve partial process counts for diagnosis.
        status = "error"
        error_type = type(exc).__name__
        error_message = str(exc)

    precheck_start, precheck_end = first_window(cs.primary_chain)
    predicted_start, predicted_end = first_window(cs.primary_chain_unique)
    metrics = interval_metrics(observed_l, observed_u, predicted_start, predicted_end)
    if status == "ok" and not metrics["has_prediction"]:
        status = "no_prediction"

    common = {
        "input_key": row["input_key"],
        "Place": row["Place"],
        "place_code": row["place_code"],
        "Year": int(row["Year"]),
        "model_variety": row["model_variety"],
        "status": status,
        "error_stage": error_stage,
        "error_type": error_type,
        "error_message": error_message,
    }
    prediction = {
        **common,
        "observed_L": observed_l,
        "observed_U": observed_u,
        "observed_interval_width_days": int(row["interval_width_days"]),
        "rb_bbch15": pd.to_datetime(rb.get("BBCH_15"), errors="coerce"),
        "season_start": cs.season_start,
        "first_before_output_check_start": precheck_start,
        "first_before_output_check_end": precheck_end,
        "predicted_start": predicted_start,
        "predicted_end": predicted_end,
        **metrics,
    }
    if compare_no_gate:
        rb_no_gate = copy.deepcopy(rb)
        rb_no_gate["BBCH_15"] = None
        cs_no_gate = CropSeasonFirst(csrb=rb_no_gate, config=copy.deepcopy(FSIM_CONFIG))
        try:
            cs_no_gate.simulate(use_wind_gate=use_wind_gate)
            no_gate_start, no_gate_end = first_window(cs_no_gate.primary_chain_unique)
        except Exception:
            no_gate_start, no_gate_end = pd.NaT, pd.NaT
        no_gate_metrics = interval_metrics(observed_l, observed_u, no_gate_start, no_gate_end)
        prediction.update({
            "no_gate_predicted_start": no_gate_start,
            "no_gate_predicted_end": no_gate_end,
            "no_gate_overlap": no_gate_metrics["overlap"],
            "no_gate_signed_boundary_error_days": no_gate_metrics["signed_boundary_error_days"],
            "no_gate_midpoint_error_days": no_gate_metrics["midpoint_error_days"],
        })
    process = {
        **common,
        "weather_rows": int(len(cs.weather_hourly)),
        "attached_inventory_days": frame_count(cs.attached_inventory),
        "dispersal_cohorts": frame_count(cs.dispersal_cohorts),
        "deposition_cohorts": frame_count(cs.deposition),
        "surviving_sporangia_cohorts": frame_count(cs.detached_survival),
        "primary_chain_events": frame_count(cs.primary_chain),
        "bbch15_gated_events": frame_count(cs.primary_chain_gated),
        "use_wind_gate": cs.use_wind_gate,
        "unique_infection_events": frame_count(cs.primary_chain_unique),
    }
    return prediction, process


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    required = {
        "input_key", "Place", "place_code", "Year", "model_variety",
        "L", "U", "interval_width_days", "weather_json", "rb_bbch53",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    predictions = []
    processes = []
    for _, row in manifest.iterrows():
        prediction, process = run_case(row, compare_no_gate=args.compare_no_bbch15_gate, use_wind_gate=args.use_wind_gate)
        predictions.append(prediction)
        processes.append(process)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_df = pd.DataFrame(predictions)
    process_df = pd.DataFrame(processes)
    format_datetime_columns(prediction_df).to_csv(
        args.output_dir / "fsim_site_year_predictions.csv", index=False, encoding="utf-8-sig"
    )
    process_df.to_csv(
        args.output_dir / "fsim_process_counts.csv", index=False, encoding="utf-8-sig"
    )

    valid = prediction_df[prediction_df["has_prediction"].eq(1)]
    summary = {
        "parameter_source": "model/config.py",
        "n_site_years": int(len(prediction_df)),
        "n_predictions": int(prediction_df["has_prediction"].sum()),
        "n_no_predictions": int(prediction_df["has_prediction"].eq(0).sum()),
        "n_overlaps": int(valid["overlap"].sum()) if not valid.empty else 0,
        "overlap_rate": float(valid["overlap"].mean()) if not valid.empty else None,
        "mean_absolute_boundary_error_days": float(valid["absolute_boundary_error_days"].mean()) if not valid.empty else None,
        "mean_absolute_midpoint_error_days": float(valid["absolute_midpoint_error_days"].mean()) if not valid.empty else None,
        "mean_predicted_window_width_days": float(valid["predicted_window_width_days"].mean()) if not valid.empty else None,
    }
    (args.output_dir / "fsim_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    columns = [
        "input_key", "status", "observed_L", "observed_U", "rb_bbch15",
        "predicted_start", "predicted_end", "overlap",
        "signed_boundary_error_days", "midpoint_error_days",
    ]
    if args.compare_no_bbch15_gate:
        columns.extend([
            "no_gate_predicted_start", "no_gate_predicted_end",
            "no_gate_overlap", "no_gate_signed_boundary_error_days",
            "no_gate_midpoint_error_days",
        ])
    print(format_datetime_columns(prediction_df[columns]).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
