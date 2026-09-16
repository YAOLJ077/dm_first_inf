#!/usr/bin/env python3
"""Compare two FSIM merged-transfer structures at fixed parameters.

Only viable, available attached cohorts enter transfer. The candidates differ
solely in whether a single wind gate is enabled. Transfer and relative leaf
availability share a timestamp; no physical travel time or deposition efficiency
is inferred. This retrospective diagnostic neither fits nor selects a model.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import config as BASE_CONFIG  # noqa: E402
from model.crop_season import CropSeasonFirst  # noqa: E402


MANIFEST = ROOT / "result" / "fsim_inputs" / "site_year_manifest.csv"
OUTPUT_DIR = ROOT / "result" / "fsim_merged_transfer_vpd"

SCENARIOS = {
    "M0_merged_transfer": {
        "use_wind_gate": False,
        "description": "Merged transfer of viable available cohorts without a wind gate.",
        "role": "Ungated structural candidate",
    },
    "M1_merged_transfer_wind_gate": {
        "use_wind_gate": True,
        "wind_threshold_status": "Historical fitted value; not recalibrated or a biological constant",
        "description": "Merged transfer of viable available cohorts with one wind gate.",
        "role": "Wind-gated structural candidate",
    },
}


def interval_distance(
    observed_l: pd.Timestamp,
    observed_u: pd.Timestamp,
    predicted_l: pd.Timestamp,
    predicted_u: pd.Timestamp,
) -> float:
    if predicted_u < observed_l:
        return (observed_l - predicted_u).total_seconds() / 86400.0
    if predicted_l > observed_u:
        return (predicted_l - observed_u).total_seconds() / 86400.0
    return 0.0


def run_scenario(rb: dict, scenario: dict, config: dict | None = None) -> CropSeasonFirst:
    """Use the ordinary model entry point with caller-supplied parameters."""
    cfg = copy.deepcopy(BASE_CONFIG if config is None else config)
    cs = CropSeasonFirst(csrb=rb, config=cfg)
    cs.simulate(use_wind_gate=scenario["use_wind_gate"])
    return cs


def weather_audit(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    all_wind = []
    wind_min = BASE_CONFIG["dispersal"]["wind_min"]
    for row in manifest.itertuples(index=False):
        with open(ROOT / row.weather_json, encoding="utf-8") as handle:
            rb = json.load(handle)
        weather = pd.DataFrame(rb["weather_hourly"])
        wind = pd.to_numeric(weather["wind_speed"], errors="coerce")
        valid = np.isfinite(wind) & (wind >= 0)
        all_wind.append(wind[valid])
        records.append({
            "input_key": row.input_key,
            "n_hours": len(weather),
            "n_invalid_wind_hours": int((~valid).sum()),
            "wind_min_m_per_s": wind_min,
            "wind_ge_threshold_valid_hours_pct": 100 * (wind[valid] >= wind_min).mean(),
            "wind_max_m_per_s": wind[valid].max(),
        })
    wind = pd.concat(all_wind, ignore_index=True)
    quantiles = [{
        "variable": "wind_speed",
        "unit": "m/s (numerically cross-checked; original request metadata unavailable)",
        "n": len(wind),
        "p50": wind.quantile(0.50),
        "p90": wind.quantile(0.90),
        "p95": wind.quantile(0.95),
        "p99": wind.quantile(0.99),
        "p99_9": wind.quantile(0.999),
        "maximum": wind.max(),
    }]
    return pd.DataFrame(records), pd.DataFrame(quantiles)


def main() -> None:
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    site_weather, weather_quantiles = weather_audit(manifest)
    site_weather.to_csv(OUTPUT_DIR / "weather_threshold_context.csv", index=False)
    weather_quantiles.to_csv(OUTPUT_DIR / "weather_quantiles.csv", index=False)

    scenario_metadata = pd.DataFrame(
        [{
            "scenario": key,
            **value,
            "wind_min_m_per_s": BASE_CONFIG["dispersal"]["wind_min"] if value["use_wind_gate"] else np.nan,
            "leaf_moisture_proxy": "physical VPD only",
            "incubation_completion": 1.0,
            "hourly_state_accounting": "transfer before aging; one state per hour",
            "transfer_quantity": "relative viable available inoculum",
            "transfer_and_leaf_availability": "same timestamp; no physical travel time inferred",
            "parameter_estimation": "none; all other parameters fixed",
        } for key, value in SCENARIOS.items()]
    )
    scenario_metadata.to_csv(OUTPUT_DIR / "scenario_definitions.csv", index=False)

    predictions = []
    for row in manifest.itertuples(index=False):
        with open(ROOT / row.weather_json, encoding="utf-8") as handle:
            rb = json.load(handle)
        observed_l = pd.Timestamp(row.L)
        observed_u = pd.Timestamp(row.U)
        observed_mid = observed_l + (observed_u - observed_l) / 2
        for scenario_name, scenario in SCENARIOS.items():
            cs = run_scenario(rb, scenario)
            candidates = cs.primary_chain_unique
            first = None if candidates is None or candidates.empty else candidates.sort_values("oil_spot_start").iloc[0]
            record = {
                "scenario": scenario_name,
                "input_key": row.input_key,
                "observed_L": observed_l.date(),
                "observed_U": observed_u.date(),
                "n_dispersal_cohorts": 0 if cs.dispersal_cohorts is None else len(cs.dispersal_cohorts),
                "n_deposited_cohorts": 0 if cs.deposition is None else len(cs.deposition),
                "n_post_gate_candidate_windows": 0 if candidates is None else len(candidates),
                "predicted_L": pd.NaT,
                "predicted_U": pd.NaT,
                "overlap": False,
                "interval_distance_days": np.nan,
                "signed_midpoint_error_days": np.nan,
            }
            if first is not None:
                predicted_l = pd.Timestamp(first["oil_spot_start"])
                predicted_u = pd.Timestamp(first["oil_spot_end"])
                predicted_mid = predicted_l + (predicted_u - predicted_l) / 2
                distance = interval_distance(observed_l, observed_u, predicted_l, predicted_u)
                record.update(
                    {
                        "predicted_L": predicted_l,
                        "predicted_U": predicted_u,
                        "overlap": distance == 0,
                        "interval_distance_days": distance,
                        "signed_midpoint_error_days": (predicted_mid - observed_mid).total_seconds() / 86400.0,
                    }
                )
            predictions.append(record)

    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv(OUTPUT_DIR / "scenario_predictions.csv", index=False)
    summary = (
        prediction_df.groupby("scenario", sort=False)
        .agg(
            n_site_years=("input_key", "size"),
            n_predictions=("predicted_L", "count"),
            n_overlaps=("overlap", "sum"),
            mean_interval_distance_days=("interval_distance_days", "mean"),
            mean_absolute_midpoint_error_days=(
                "signed_midpoint_error_days", lambda x: x.abs().mean()
            ),
            mean_signed_midpoint_error_days=("signed_midpoint_error_days", "mean"),
            mean_dispersal_cohorts=("n_dispersal_cohorts", "mean"),
            mean_deposited_cohorts=("n_deposited_cohorts", "mean"),
            mean_post_gate_candidate_windows=("n_post_gate_candidate_windows", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved audit outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
