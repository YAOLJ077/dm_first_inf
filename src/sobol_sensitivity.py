"""Sobol sensitivity analysis for the six Guangxi sporangia site-years.

The analysis samples the six explicit parameters in the current S-path
configuration, runs the process model with observed phenology anchors, and
evaluates complete symptom windows against the disease-onset workbook. The
observed onset is treated as the interval ``[observed_date - 3 days,
observed_date]``; a prediction overlapping that interval has zero timing
error, while a non-overlapping prediction is scored by its nearest boundary.
The parameter domains use the explicit bounded S-route sensitivity ranges.
This module deliberately contains only the current sporangia route.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import date, datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import os
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.phenology.susceptibility_modulation import (  # noqa: E402
    DEFAULT_CONFIG as DEFAULT_SUSCEPTIBILITY_CONFIG,
    observed_stage_anchors,
)
from model.FSIM.cropseason import CropSeason  # noqa: E402
from model.FSIM.model import run_cropseason  # noqa: E402


DEFAULT_WEATHER_DIR = ROOT / "data/weather_data/site_year"
DEFAULT_OBSERVATIONS = ROOT / "data/disease_data/grape_downy_mildew_data.xlsx"
DEFAULT_PHENOLOGY_OBSERVATIONS = ROOT / "data/phenology_data/phenology.xlsx"
DEFAULT_SPORANGIA_CONFIG = ROOT / "model/FSIM/config.json"
DEFAULT_OUTPUT_ROOT = ROOT / "result/sobol_sensitivity"
DEFAULT_MISSING_PENALTY = 60.0
MIN_POSITIVE = 1e-6

S_PARAMETER_NAMES = (
    "infection_degree_hours",
    "wind_min_m_s",
    "effective_deposition_min_rain_mm",
    "sporulation_rh_min_pct",
    "minimum_moist_hours",
    "heavy_rain_washoff_mm",
)

WORKER_STATE: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _temporary_config(config: dict[str, Any]) -> Path:
    handle, name = tempfile.mkstemp(prefix="sobol_sporangia_", suffix=".json")
    os.close(handle)
    path = Path(name)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@lru_cache(maxsize=32)
def _read_season(path: str) -> CropSeason:
    return CropSeason.read(path)


def _build_records(
    weather_dir: Path,
    observations_path: Path,
    phenology_path: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Match disease observations to the current weather seasons."""

    observations = pd.read_excel(observations_path)
    required = {"Latitude", "Longitude", "Year", "Date"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Disease workbook is missing columns: {sorted(missing)}")
    observations = observations.copy()
    observations["Latitude"] = pd.to_numeric(observations["Latitude"], errors="coerce")
    observations["Longitude"] = pd.to_numeric(observations["Longitude"], errors="coerce")
    observations["Year"] = pd.to_numeric(observations["Year"], errors="coerce")
    observations["Date"] = pd.to_datetime(observations["Date"], errors="coerce")
    observations = observations.dropna(subset=["Latitude", "Longitude", "Year", "Date"])

    seasons = [_read_season(str(path.resolve())) for path in sorted(weather_dir.glob("*.json"))]
    records: list[dict[str, Any]] = []
    for row in observations.itertuples(index=False):
        matches = [
            season for season in seasons
            if season.year == int(row.Year)
            and abs(season.latitude - float(row.Latitude)) <= 0.005
            and abs(season.longitude - float(row.Longitude)) <= 0.005
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one weather season for ({row.Latitude}, {row.Longitude}, {int(row.Year)}), "
                f"found {len(matches)}"
            )
        season = matches[0]
        records.append({
            "site_year": f"{season.site_id}_{season.year}",
            "site_id": season.site_id,
            "year": int(season.year),
            "weather_path": str(season.path),
            "observed_date": pd.Timestamp(row.Date).date().isoformat(),
            "observed_stage_dates": observed_stage_anchors(season, phenology_path),
        })
    records.sort(key=lambda item: item["site_year"])
    if limit is not None:
        records = records[:limit]
    if len(records) != 6 and limit is None:
        raise ValueError(f"Expected six Guangxi site-years, found {len(records)}")
    if not records:
        raise ValueError("No disease-onset site-years matched the weather directory")
    return records


def _prior_defaults(base_config: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(base_config["biological_parameters"]["infection_degree_hours"]),
        float(base_config["structure_parameters"]["wind_min_m_s"]),
        float(base_config["deposition_parameters"]["effective_deposition_min_rain_mm"]),
        float(base_config["sporulation_parameters"]["sporulation_rh_min_pct"]),
        float(base_config["sporulation_parameters"]["minimum_moist_hours"]),
        float(base_config["deposition_parameters"]["heavy_rain_washoff_mm"]),
    )


def _expanded_bounds(base_config: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    """Return the fixed S-route Sobol domains requested for this analysis."""

    # The current configuration remains the prior reference, while these
    # explicit domains define the bounded sensitivity experiment.
    return (
        (60.0, 200.0),    # infection_degree_hours
        (0.2, 1.0),       # wind_min_m_s
        (1.0, 4.9),       # effective_deposition_min_rain_mm
        (80.0, 100.0),    # sporulation_rh_min_pct
        (3.0, 10.0),      # minimum_moist_hours
        (5.0, 10.0),      # heavy_rain_washoff_mm
    )


def _resolved_vector(vector: list[float]) -> list[float]:
    resolved = list(map(float, vector))
    resolved[0] = max(resolved[0], MIN_POSITIVE)
    resolved[1] = max(resolved[1], 0.0)
    resolved[2] = max(resolved[2], 0.0)
    resolved[3] = min(max(resolved[3], 0.0), 100.0)
    resolved[4] = min(max(resolved[4], MIN_POSITIVE), 12.0)
    resolved[5] = max(resolved[5], resolved[2] + MIN_POSITIVE)
    return resolved


def _candidate_config(vector: list[float], base_config: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolved_vector(vector)
    config = deepcopy(base_config)
    config["biological_parameters"]["infection_degree_hours"] = resolved[0]
    config["structure_parameters"]["wind_min_m_s"] = resolved[1]
    config["deposition_parameters"]["effective_deposition_min_rain_mm"] = resolved[2]
    config["sporulation_parameters"]["sporulation_rh_min_pct"] = resolved[3]
    config["sporulation_parameters"]["minimum_moist_hours"] = resolved[4]
    config["deposition_parameters"]["heavy_rain_washoff_mm"] = resolved[5]
    return config


def _first_window(result: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    events = [
        event for event in result["process"].get("infections", [])
        if event.get("symptom_window_complete")
        and event.get("symptom_start") is not None
        and event.get("symptom_end") is not None
    ]
    if not events:
        return None, None
    event = min(events, key=lambda item: item["symptom_start"])
    return pd.Timestamp(event["symptom_start"]), pd.Timestamp(event["symptom_end"])


def _window_row(record: dict[str, Any], result: dict[str, Any], missing_penalty: float) -> dict[str, Any]:
    observed = pd.Timestamp(record["observed_date"]).date()
    observed_window_start = observed - pd.Timedelta(days=3)
    observed_window_end = observed
    start, end = _first_window(result)
    if start is None or end is None:
        year = int(record["year"])
        censor_day = int((date(year, 12, 31) - date(year, 1, 1)).days + 2)
        return {
            "site_year": record["site_year"], "year": year,
            "observed_date": observed.isoformat(), "complete_window": 0,
            "observed_window_start": observed_window_start.isoformat(),
            "observed_window_end": observed_window_end.isoformat(),
            "predicted_start": None, "predicted_end": None,
            "predicted_start_day": censor_day, "signed_error_days": None,
            "absolute_error_days": None, "interval_distance_days": None,
            "penalized_absolute_error_days": float(missing_penalty),
        }

    year = int(record["year"])
    start_date, end_date = start.date(), end.date()
    if start_date > observed_window_end:
        signed_error = (start_date - observed_window_end).days
        distance = signed_error
    elif end_date < observed_window_start:
        signed_error = (end_date - observed_window_start).days
        distance = -signed_error
    else:
        signed_error = 0
        distance = 0
    return {
        "site_year": record["site_year"], "year": year,
        "observed_date": observed.isoformat(), "complete_window": 1,
        "observed_window_start": observed_window_start.isoformat(),
        "observed_window_end": observed_window_end.isoformat(),
        "predicted_start": start.isoformat(), "predicted_end": end.isoformat(),
        "predicted_start_day": int((start_date - date(year, 1, 1)).days + 1),
        "signed_error_days": int(signed_error),
        "absolute_error_days": int(abs(signed_error)),
        "interval_distance_days": int(distance),
        "penalized_absolute_error_days": int(abs(signed_error)),
    }


def _aggregate(rows: list[dict[str, Any]], missing_penalty: float) -> dict[str, float | int | None]:
    frame = pd.DataFrame(rows)
    complete = frame["complete_window"].astype(bool)
    n = len(frame)
    n_complete = int(complete.sum())
    missing = int(n - n_complete)
    absolute = pd.to_numeric(frame.loc[complete, "absolute_error_days"], errors="coerce")
    signed = pd.to_numeric(frame.loc[complete, "signed_error_days"], errors="coerce")
    interval = pd.to_numeric(frame.loc[complete, "interval_distance_days"], errors="coerce")
    return {
        "n_site_years": n, "n_complete": n_complete,
        "complete_window_fraction": float(n_complete / n) if n else None,
        "missing_windows": missing,
        "mean_first_window_day_censored": float(frame["predicted_start_day"].mean()),
        "mean_first_window_day_complete": float(frame.loc[complete, "predicted_start_day"].mean()) if n_complete else None,
        "mae_days_complete": float(absolute.mean()) if n_complete else None,
        "mae_days_penalized": float((absolute.sum() + missing * missing_penalty) / n) if n else None,
        "mean_signed_error_days_complete": float(signed.mean()) if n_complete else None,
        "mean_interval_distance_days_complete": float(interval.mean()) if n_complete else None,
    }


def _init_worker(state: dict[str, Any]) -> None:
    global WORKER_STATE
    WORKER_STATE = state


def _evaluate_vector(vector: list[float]) -> dict[str, Any]:
    if WORKER_STATE is None:
        raise RuntimeError("Sobol worker was not initialized")
    state = WORKER_STATE
    config = _candidate_config(vector, state["base_config"])
    resolved = _resolved_vector(vector)
    config_path = _temporary_config(config)
    rows = []
    try:
        for record in state["records"]:
            season = _read_season(record["weather_path"])
            result = run_cropseason(
                season,
                config_path,
                susceptibility_modulation_config=state["susceptibility_config"],
                observed_stage_dates=record.get("observed_stage_dates"),
                first_window_only=True,
            )
            rows.append(_window_row(record, result, state["missing_penalty"]))
    finally:
        config_path.unlink(missing_ok=True)
    return {
        "rows": rows,
        "aggregate": _aggregate(rows, state["missing_penalty"]),
        "applied_vector": resolved,
    }


def _analyse_output(problem: dict[str, Any], values: list[dict[str, Any]], key: str, args):
    raw = np.asarray([
        np.nan if item["aggregate"][key] is None else float(item["aggregate"][key])
        for item in values
    ], dtype=float)
    finite = np.isfinite(raw)
    metadata = {
        "output": key, "finite_samples": int(finite.sum()),
        "total_samples": int(len(raw)),
        "variance": float(np.nanvar(raw)) if finite.any() else None,
        "min": float(np.nanmin(raw)) if finite.any() else None,
        "max": float(np.nanmax(raw)) if finite.any() else None,
    }
    if not finite.all() or np.nanvar(raw) <= 1e-12:
        metadata["status"] = "skipped_nonfinite_or_zero_variance"
        return [], [], metadata
    result = sobol_analyze.analyze(
        problem, raw, calc_second_order=args.second_order,
        num_resamples=args.resamples, conf_level=args.confidence,
        print_to_console=False, seed=args.seed,
    )
    first_rows = [{
        "output": key, "parameter": name,
        "S1": float(result["S1"][i]), "S1_conf": float(result["S1_conf"][i]),
        "ST": float(result["ST"][i]), "ST_conf": float(result["ST_conf"][i]),
    } for i, name in enumerate(problem["names"])]
    second_rows = []
    if args.second_order:
        for i, name_i in enumerate(problem["names"]):
            for j in range(i + 1, len(problem["names"])):
                value, confidence = result["S2"][i, j], result["S2_conf"][i, j]
                second_rows.append({
                    "output": key, "parameter_i": name_i,
                    "parameter_j": problem["names"][j],
                    "S2": float(value) if np.isfinite(value) else None,
                    "S2_conf": float(confidence) if np.isfinite(confidence) else None,
                })
    metadata["status"] = "analysed"
    return first_rows, second_rows, metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run_one(args, sample_n: int) -> Path:
    base_config = _load_json(args.sporangia_config)
    defaults = _prior_defaults(base_config)
    bounds = _expanded_bounds(base_config)
    names = list(S_PARAMETER_NAMES)
    problem = {"num_vars": len(names), "names": names, "bounds": [list(item) for item in bounds]}
    samples = sobol_sample.sample(problem, sample_n, calc_second_order=args.second_order, seed=args.seed)
    records = _build_records(args.weather_dir, args.observations, args.phenology_observations, args.limit)
    state = {
        "records": records,
        "base_config": base_config,
        "susceptibility_config": str(args.susceptibility_config.resolve()),
        "missing_penalty": float(args.missing_penalty),
    }
    context = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(state,)) if args.workers > 1 else None
    if context is None:
        _init_worker(state)
    try:
        iterator = context.map(_evaluate_vector, samples.tolist(), chunksize=1) if context else map(_evaluate_vector, samples.tolist())
        evaluated = list(tqdm(iterator, total=len(samples), desc=f"Sobol S N={sample_n}", unit="sample"))
    finally:
        if context is not None:
            context.shutdown()

    output_dir = (args.output_dir or DEFAULT_OUTPUT_ROOT / f"S_N{sample_n}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows, site_rows = [], []
    for index, (vector, evaluation) in enumerate(zip(samples, evaluated)):
        aggregate_rows.append({
            "sample_index": index,
            **dict(zip(names, map(float, vector))),
            **{f"applied_{name}": value for name, value in zip(names, evaluation["applied_vector"])},
            **evaluation["aggregate"],
        })
        site_rows.extend({"sample_index": index, **row} for row in evaluation["rows"])
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "sample_outputs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(site_rows).to_csv(output_dir / "site_year_outputs.csv", index=False, encoding="utf-8-sig")

    output_keys = [
        "mean_first_window_day_censored", "mean_first_window_day_complete",
        "complete_window_fraction", "missing_windows", "mae_days_complete",
        "mae_days_penalized", "mean_signed_error_days_complete",
        "mean_interval_distance_days_complete",
    ]
    first_rows, second_rows, output_metadata = [], [], []
    for key in output_keys:
        first, second, metadata = _analyse_output(problem, evaluated, key, args)
        first_rows.extend(first)
        second_rows.extend(second)
        output_metadata.append(metadata)
    pd.DataFrame(first_rows, columns=["output", "parameter", "S1", "S1_conf", "ST", "ST_conf"]).to_csv(output_dir / "sobol_indices.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(second_rows, columns=["output", "parameter_i", "parameter_j", "S2", "S2_conf"]).to_csv(output_dir / "sobol_second_order.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(output_metadata).to_csv(output_dir / "sobol_output_summary.csv", index=False, encoding="utf-8-sig")

    _write_json(output_dir / "problem.json", {
        "route": "S", "model": "sporangia", "parameter_names": names,
        "prior_defaults": dict(zip(names, defaults)),
        "sampling_bounds": {name: list(bound) for name, bound in zip(names, bounds)},
        "sampling_domain_note": "Fixed user-specified S-route domains; heavy wash-off is projected above effective deposition when needed.",
        "domain_definition": "fixed user-specified bounds",
        "sample_size_N": sample_n,
        "sobol_sample_count": int(len(samples)), "second_order": args.second_order,
        "seed": args.seed, "records": [record["site_year"] for record in records],
        "record_count": len(records), "missing_penalty_days": args.missing_penalty,
        "observed_window": "[observed_date - 3 days, observed_date]",
        "error_definition": "zero for overlapping observed and predicted windows; otherwise signed distance to the nearest window boundary",
    })
    _write_json(output_dir / "manifest.json", {
        "schema_version": 1, "model": "sporangia_sobol", "route": "S", "N": sample_n,
        "workers": args.workers, "second_order": args.second_order,
        "resamples": args.resamples, "confidence": args.confidence,
        "output_dir": str(output_dir), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_metadata": output_metadata,
        "observed_window": "[observed_date - 3 days, observed_date]",
        "error_definition": "zero for overlapping observed and predicted windows; otherwise signed distance to the nearest window boundary",
        "inputs": {
            "weather_dir": str(args.weather_dir.resolve()),
            "observations": str(args.observations.resolve()),
            "phenology_observations": str(args.phenology_observations.resolve()),
            "sporangia_config": str(args.sporangia_config.resolve()),
            "susceptibility_config": str(args.susceptibility_config.resolve()),
        },
    })
    print(f"Completed S Sobol analysis: {output_dir}")
    print(pd.DataFrame(output_metadata).to_string(index=False))
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=["S", "sporangia"], default="S", help="Compatibility flag; only S is available.")
    parser.add_argument("--N", "--n", dest="N", type=int, nargs="+", required=True, help="Sobol base sample size(s), powers of two")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--resamples", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--missing-penalty", type=float, default=DEFAULT_MISSING_PENALTY)
    parser.add_argument("--weather-dir", type=Path, default=DEFAULT_WEATHER_DIR)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--phenology-observations", type=Path, default=DEFAULT_PHENOLOGY_OBSERVATIONS)
    parser.add_argument("--sporangia-config", type=Path, default=DEFAULT_SPORANGIA_CONFIG)
    parser.add_argument("--susceptibility-config", type=Path, default=DEFAULT_SUSCEPTIBILITY_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit site-years for a smoke test")
    parser.add_argument("--no-second-order", dest="second_order", action="store_false")
    parser.set_defaults(second_order=True)
    args = parser.parse_args(argv)

    if args.route.lower() not in {"s", "sporangia"}:
        parser.error("only the current S/sporangia route is supported")
    if any(sample_n < 2 or sample_n & (sample_n - 1) for sample_n in args.N):
        parser.error("every --N value must be a power of two")
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    if args.resamples < 1 or not 0 < args.confidence < 1:
        parser.error("--resamples must be positive and --confidence must be in (0, 1)")
    if not np.isfinite(args.missing_penalty) or args.missing_penalty <= 0:
        parser.error("--missing-penalty must be finite and positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    output_dirs = [_run_one(args, sample_n) for sample_n in args.N]
    print("Completed directories:", ", ".join(str(path) for path in output_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
