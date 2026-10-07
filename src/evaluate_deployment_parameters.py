"""Evaluate the selected FSIM and FSIM-S deployment configurations.

The evaluation uses the six Guangxi site-years and the same interval-censored
onset metric used by calibration: the observed onset is the three-day window
from the last negative survey date through the first positive survey date.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibrate_spore_pressure_s import (
    DEFAULT_HOST_CONFIG,
    DEFAULT_OBSERVATIONS,
    DEFAULT_PHENOLOGY_OBSERVATIONS,
    DEFAULT_SPORE_FILE,
    DEFAULT_WEATHER_DIR,
)
from src.sobol_sensitivity import _build_records


DEFAULT_DEPLOYMENT_DIR = PROJECT_ROOT / "result/deployment_parameters"
DEFAULT_FSIM_CONFIG = DEFAULT_DEPLOYMENT_DIR / "fsim_deployment_config.json"
DEFAULT_FSIM_S_CONFIG = DEFAULT_DEPLOYMENT_DIR / "fsim_s_deployment_config.json"
DEFAULT_OUTPUT_CSV = DEFAULT_DEPLOYMENT_DIR / "deployment_predictions.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_DEPLOYMENT_DIR / "deployment_summary.json"

run_spore_plus = importlib.import_module("model.FSIM-S.model").run_spore_plus


def _first_window(result: dict[str, Any], *, pressure: bool) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    process = result.get("spore_process" if pressure else "process")
    if not process:
        return None, None
    events = [
        event
        for event in process.get("infections", [])
        if event.get("symptom_window_complete")
        and event.get("symptom_start") is not None
        and event.get("symptom_end") is not None
    ]
    if not events:
        return None, None
    event = min(events, key=lambda item: item["symptom_start"])
    return pd.Timestamp(event["symptom_start"]), pd.Timestamp(event["symptom_end"])


def _error_row(record: dict[str, Any], result: dict[str, Any], *, pressure: bool) -> dict[str, Any]:
    observed = pd.Timestamp(record["observed_date"]).date()
    observed_start = observed - pd.Timedelta(days=3)
    observed_end = observed
    predicted_start, predicted_end = _first_window(result, pressure=pressure)
    row: dict[str, Any] = {
        "model": "FSIM-S" if pressure else "FSIM",
        "site_year": record["site_year"],
        "observed_date": observed.isoformat(),
        "observed_window_start": observed_start.isoformat(),
        "observed_window_end": observed_end.isoformat(),
        "complete_window": int(predicted_start is not None and predicted_end is not None),
        "predicted_start": predicted_start.isoformat() if predicted_start is not None else None,
        "predicted_end": predicted_end.isoformat() if predicted_end is not None else None,
    }
    if predicted_start is None or predicted_end is None:
        row["error_days"] = None
        row["absolute_error_days"] = None
        return row

    start_date, end_date = predicted_start.date(), predicted_end.date()
    if start_date > observed_end:
        error = (start_date - observed_end).days
    elif end_date < observed_start:
        error = (end_date - observed_start).days
    else:
        error = 0
    row["error_days"] = int(error)
    row["absolute_error_days"] = int(abs(error))
    return row


def _run_model(
    record: dict[str, Any],
    *,
    fsim_config: Path,
    fsim_s_config: Path,
    pressure: bool,
) -> dict[str, Any]:
    return run_spore_plus(
        record["weather_path"],
        s_config_path=fsim_config,
        pressure_config=fsim_s_config if pressure else None,
        pressure_enabled=pressure,
        spore_file=DEFAULT_SPORE_FILE if pressure else None,
        susceptibility_modulation_config=DEFAULT_HOST_CONFIG,
        observed_stage_dates=record.get("observed_stage_dates"),
    )


def evaluate(
    *,
    fsim_config: Path = DEFAULT_FSIM_CONFIG,
    fsim_s_config: Path = DEFAULT_FSIM_S_CONFIG,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = _build_records(
        DEFAULT_WEATHER_DIR,
        DEFAULT_OBSERVATIONS,
        DEFAULT_PHENOLOGY_OBSERVATIONS,
    )
    rows: list[dict[str, Any]] = []
    for pressure in (False, True):
        for record in records:
            result = _run_model(
                record,
                fsim_config=fsim_config,
                fsim_s_config=fsim_s_config,
                pressure=pressure,
            )
            rows.append(_error_row(record, result, pressure=pressure))

    frame = pd.DataFrame(rows)
    summaries: dict[str, dict[str, Any]] = {}
    for model_name, group in frame.groupby("model", sort=False):
        errors = pd.to_numeric(group["error_days"], errors="coerce").dropna().to_numpy(dtype=float)
        summaries[model_name] = {
            "n_site_years": int(len(group)),
            "complete_windows": int(len(errors)),
            "mae_days": float(np.mean(np.abs(errors))) if len(errors) else None,
            "rmse_days": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "mean_signed_error_days": float(np.mean(errors)) if len(errors) else None,
        }

    summary = {
        "schema_version": 1,
        "metric": {
            "observed_onset_window": "observed_date minus 3 days through observed_date",
            "error_rule": "zero for overlapping predicted and observed windows; otherwise signed nearest-boundary distance",
        },
        "configs": {
            "FSIM": str(fsim_config.resolve()),
            "FSIM-S": str(fsim_s_config.resolve()),
        },
        "models": summaries,
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frame, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsim-config", type=Path, default=DEFAULT_FSIM_CONFIG)
    parser.add_argument("--fsim-s-config", type=Path, default=DEFAULT_FSIM_S_CONFIG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args(argv)
    frame, summary = evaluate(
        fsim_config=args.fsim_config,
        fsim_s_config=args.fsim_s_config,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(frame[["model", "site_year", "error_days", "absolute_error_days"]].to_string(index=False))
    for model_name, values in summary["models"].items():
        print(f"{model_name}: MAE={values['mae_days']:.6g} days; RMSE={values['rmse_days']:.6g} days")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
