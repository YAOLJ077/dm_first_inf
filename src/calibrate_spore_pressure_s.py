"""Nested grouped leave-one-out calibration for the current FSIM-S route.

The primary outer fold holds out one Guangxi site-year. Optional grouped
stress-test folds hold out one site or one year at a time. Three open Sporangia
process parameters are fitted on the remaining records using one finite
Cartesian grid; wind, heavy-rain wash-off and minimum moist duration are fixed.
An optional shared effective-deposition threshold can also be fixed across folds
for a cross-fold stability scenario. Among training candidates
within a fixed MAE tolerance of the optimum, selection uses the normalized
component-wise median of that near-optimal training candidate set as a
data-derived stability center. The selected S configuration is then fixed
while the single active FSIM-S pressure coefficient, ``beta_sp``, is calibrated
on the same training records. Every record in the held-out group is evaluated
with the fixed S configuration and the selected pressure coefficient.

The pressure field is the configured interval-rate/z-score field. This script
uses the observation-zero weather-cohort gate: numeric zero-spore collection
intervals suppress weather-generated cohorts, while missing observations do
not suppress weather production and any observed-background dose is retained.

Disease observations are interval-censored by the three-day survey interval:
the observed onset window is ``[observed_date - 3 days, observed_date]``.
Timing error is zero when this window overlaps the predicted symptom window;
otherwise it is the signed distance to the nearest window boundary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import importlib
from itertools import product
import json
from pathlib import Path
import os
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

run_spore_plus = importlib.import_module("model.FSIM-S.model").run_spore_plus  # noqa: E402
from src.sobol_sensitivity import (  # noqa: E402
    S_PARAMETER_NAMES,
    _build_records,
    _load_json,
)


DEFAULT_WEATHER_DIR = PROJECT_ROOT / "data/weather_data/site_year"
DEFAULT_OBSERVATIONS = PROJECT_ROOT / "data/disease_data/grape_downy_mildew_data.xlsx"
DEFAULT_PHENOLOGY_OBSERVATIONS = PROJECT_ROOT / "data/phenology_data/phenology.xlsx"
DEFAULT_S_CONFIG = PROJECT_ROOT / "model/FSIM/config.json"
DEFAULT_SPORE_CONFIG = PROJECT_ROOT / "model/FSIM-S/config.json"
DEFAULT_SPORE_FILE = PROJECT_ROOT / "data/spore_data/guangxi_site_years_spore_data.xlsx"
DEFAULT_HOST_CONFIG = PROJECT_ROOT / "model/phenology/susceptibility_modulation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "result/fsim_s_calibration"
DEFAULT_MISSING_PENALTY = 60.0
DEFAULT_MAE_TOLERANCE_DAYS = 0.5
CV_SCHEMES = ("site_year", "site", "year")

FULL_PARAMETER_NAMES = tuple(S_PARAMETER_NAMES)
FIXED_PARAMETERS = {
    "wind_min_m_s": 0.5,
    "heavy_rain_washoff_mm": 5.0,
    "minimum_moist_hours": 7.0,
}
PARAMETER_NAMES = tuple(name for name in FULL_PARAMETER_NAMES if name not in FIXED_PARAMETERS)
PRESSURE_PARAMETER_NAME = "beta_sp"

PARAMETER_BOUNDS = {
    "infection_degree_hours": (150.0, 200.0),
    "effective_deposition_min_rain_mm": (3.8, 3.95),
    "sporulation_rh_min_pct": (80.0, 100.0),
    "heavy_rain_washoff_mm": (4.0, 10.0),
}

# The FSIM process parameters are evaluated on one prespecified grid.
PARAMETER_GRIDS = {
    "infection_degree_hours": tuple(float(value) for value in range(150, 201, 10)),
    "effective_deposition_min_rain_mm": (3.80, 3.85, 3.90, 3.95),
    "sporulation_rh_min_pct": (80.0, 85.0, 90.0, 95.0, 100.0),
    "heavy_rain_washoff_mm": (4.0, 5.0, 6.0, 8.0, 10.0),
}

BETA_BOUNDS = (0.0, 1.0)
BETA_COARSE_GRID = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
BETA_FINE_STEP = 0.05

WORKER_STATE: dict[str, Any] | None = None


def _temporary_config(config: dict[str, Any]) -> Path:
    handle, name = tempfile.mkstemp(prefix="calibrate_spore_s_", suffix=".json")
    os.close(handle)
    path = Path(name)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _resolved_vector(vector: Iterable[float]) -> tuple[float, ...]:
    values = list(map(float, vector))
    if len(values) != len(PARAMETER_NAMES):
        raise ValueError(f"Expected {len(PARAMETER_NAMES)} open S parameters, found {len(values)}")
    resolved = []
    for value, name in zip(values, PARAMETER_NAMES):
        lower, upper = PARAMETER_BOUNDS[name]
        if not np.isfinite(value):
            raise ValueError(f"Non-finite value for {name}: {value}")
        resolved.append(min(max(value, lower), upper))
    return tuple(resolved)


def _candidate_config(vector: Iterable[float], base_config: dict[str, Any]) -> dict[str, Any]:
    values = _resolved_vector(vector)
    parameters = dict(FIXED_PARAMETERS)
    parameters.update(zip(PARAMETER_NAMES, values))
    config = deepcopy(base_config)
    config["biological_parameters"]["infection_degree_hours"] = parameters["infection_degree_hours"]
    config["structure_parameters"]["wind_min_m_s"] = parameters["wind_min_m_s"]
    config["deposition_parameters"]["effective_deposition_min_rain_mm"] = parameters["effective_deposition_min_rain_mm"]
    config["sporulation_parameters"]["sporulation_rh_min_pct"] = parameters["sporulation_rh_min_pct"]
    config["sporulation_parameters"]["minimum_moist_hours"] = parameters["minimum_moist_hours"]
    config["deposition_parameters"]["heavy_rain_washoff_mm"] = parameters["heavy_rain_washoff_mm"]
    return config


def _first_window(
    result: dict[str, Any], *, pressure: bool = False
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    process_key = "spore_process" if pressure else "process"
    process = result.get(process_key)
    if not process:
        return None, None
    events = [
        event for event in process.get("infections", [])
        if event.get("symptom_window_complete")
        and event.get("symptom_start") is not None
        and event.get("symptom_end") is not None
    ]
    if not events:
        return None, None
    event = min(events, key=lambda item: item["symptom_start"])
    return pd.Timestamp(event["symptom_start"]), pd.Timestamp(event["symptom_end"])


def _metric_for_record(
    record: dict[str, Any],
    result: dict[str, Any],
    missing_penalty: float,
    *,
    pressure: bool = False,
) -> dict[str, Any]:
    observed = pd.Timestamp(record["observed_date"]).date()
    observed_window_start = observed - pd.Timedelta(days=3)
    observed_window_end = observed
    start, end = _first_window(result, pressure=pressure)
    if start is None or end is None:
        return {
            "site_year": record["site_year"],
            "observed_date": observed.isoformat(),
            "observed_window_start": observed_window_start.isoformat(),
            "observed_window_end": observed_window_end.isoformat(),
            "complete_window": 0,
            "predicted_start": None,
            "predicted_end": None,
            "error_days": None,
            "absolute_error_days": None,
            "penalized_absolute_error_days": float(missing_penalty),
        }
    start_date, end_date = start.date(), end.date()
    if start_date > observed_window_end:
        error = (start_date - observed_window_end).days
    elif end_date < observed_window_start:
        error = (end_date - observed_window_start).days
    else:
        error = 0
    return {
        "site_year": record["site_year"],
        "observed_date": observed.isoformat(),
        "observed_window_start": observed_window_start.isoformat(),
        "observed_window_end": observed_window_end.isoformat(),
        "complete_window": 1,
        "predicted_start": start.isoformat(),
        "predicted_end": end.isoformat(),
        "error_days": int(error),
        "absolute_error_days": int(abs(error)),
        "penalized_absolute_error_days": int(abs(error)),
    }


def _aggregate(rows: list[dict[str, Any]], missing_penalty: float) -> dict[str, Any]:
    errors = np.asarray(
        [row["error_days"] for row in rows if row["error_days"] is not None],
        dtype=float,
    )
    missing = len(rows) - len(errors)
    return {
        "n": len(rows),
        "complete": len(errors),
        "missing": missing,
        "window_coverage": float(len(errors) / len(rows)) if rows else 0.0,
        "mae_days": float(np.mean(np.abs(errors))) if len(errors) else None,
        "bias_days": float(np.mean(errors)) if len(errors) else None,
        "rmse_days": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "selection_mae_days": float(
            (np.abs(errors).sum() + missing * missing_penalty) / len(rows)
        ) if rows else None,
    }


def _selection_key(
    result: dict[str, Any],
    vector: Iterable[float] | None = None,
) -> tuple[float, int, float, tuple[float, ...]]:
    aggregate = result["aggregate"]
    bias = aggregate["bias_days"]
    values = tuple(result["vector"] if vector is None else vector)
    # Primary search objective is training penalized MAE. Deterministic ties
    # prefer non-positive bias, then bias closest to zero, then lexicographic
    # parameter order. No configuration prior participates in this key.
    return (
        float(aggregate["selection_mae_days"]),
        1 if bias is None or float(bias) > 0 else 0,
        abs(float(bias)) if bias is not None else float("inf"),
        values,
    )


def _select_stable_candidate(
    results: Iterable[dict[str, Any]],
    names: tuple[str, ...],
    bounds: tuple[tuple[float, float], ...],
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose a stable candidate using only near-optimal training results."""

    candidates = list(results)
    if not candidates:
        raise ValueError("Cannot select from an empty candidate bank")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("MAE tolerance must be finite and nonnegative")
    widths = np.asarray([upper - lower for lower, upper in bounds], dtype=float)
    lower_bounds = np.asarray([lower for lower, _ in bounds], dtype=float)
    if len(widths) != len(names) or (widths <= 0).any():
        raise ValueError("Parameter names and positive domain widths must match")
    minimum_missing = min(item["aggregate"]["missing"] for item in candidates)
    feasible = [item for item in candidates if item["aggregate"]["missing"] == minimum_missing]
    minimum_mae = min(float(item["aggregate"]["selection_mae_days"]) for item in feasible)
    near = [
        item for item in feasible
        if float(item["aggregate"]["selection_mae_days"]) <= minimum_mae + tolerance + 1e-10
    ]

    vectors = np.asarray([item["vector"] for item in near], dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != len(names):
        raise ValueError("Candidate vectors do not match the parameter domain")
    normalized_vectors = (vectors - lower_bounds) / widths
    normalized_center = np.median(normalized_vectors, axis=0)

    def distance(item):
        vector = np.asarray(item["vector"], dtype=float)
        if vector.shape != widths.shape:
            raise ValueError("Candidate vector does not match the parameter domain")
        normalized = (vector - lower_bounds) / widths
        return float(np.sum((normalized - normalized_center) ** 2))

    def key(item):
        bias = item["aggregate"]["bias_days"]
        return (
            distance(item),
            float(item["aggregate"]["selection_mae_days"]),
            1 if bias is None or float(bias) > 0 else 0,
            abs(float(bias)) if bias is not None else float("inf"),
            tuple(item["vector"]),
        )

    selected = min(near, key=key)
    center_parameters = lower_bounds + normalized_center * widths
    audit = {
        "candidate_count": len(candidates),
        "minimum_missing": int(minimum_missing),
        "minimum_training_penalized_mae_days": minimum_mae,
        "mae_tolerance_days": float(tolerance),
        "near_optimal_count": len(near),
        "selected_parameters": dict(zip(names, map(float, selected["vector"]))),
        "stability_center_method": "componentwise_median_of_normalized_near_optimal_training_candidates",
        "normalized_stability_center": dict(zip(names, map(float, normalized_center))),
        "stability_center_parameters": dict(zip(names, map(float, center_parameters))),
        "selected_training_penalized_mae_days": float(selected["aggregate"]["selection_mae_days"]),
        "selected_mae_increase_days": float(selected["aggregate"]["selection_mae_days"]) - minimum_mae,
        "selected_normalized_squared_center_distance": distance(selected),
        "near_optimal_ranges": {
            name: [float(vectors[:, i].min()), float(vectors[:, i].max())]
            for i, name in enumerate(names)
        },
    }
    return selected, audit


def _init_worker(state: dict[str, Any]) -> None:
    global WORKER_STATE, FIXED_PARAMETERS, PARAMETER_NAMES, PARAMETER_BOUNDS
    if "fixed_parameters" in state:
        FIXED_PARAMETERS = dict(state["fixed_parameters"])
    if "parameter_names" in state:
        PARAMETER_NAMES = tuple(state["parameter_names"])
    if "parameter_bounds" in state:
        PARAMETER_BOUNDS = dict(state["parameter_bounds"])
    WORKER_STATE = state


def _evaluate_s_candidate(vector: tuple[float, ...]) -> dict[str, Any]:
    if WORKER_STATE is None:
        raise RuntimeError("Calibration worker was not initialized")
    state = WORKER_STATE
    resolved = _resolved_vector(vector)
    config_path = _temporary_config(_candidate_config(resolved, state["s_config"]))
    try:
        rows = []
        for record in state["records"]:
            result = run_spore_plus(
                record["weather_path"],
                s_config_path=config_path,
                pressure_config={**state["pressure_config"], "enabled": False},
                pressure_enabled=False,
                susceptibility_modulation_config=state["host_config"],
                observed_stage_dates=record.get("observed_stage_dates"),
            )
            rows.append(_metric_for_record(record, result, state["missing_penalty"]))
    finally:
        config_path.unlink(missing_ok=True)
    return {
        "vector": list(resolved),
        "rows": rows,
        "aggregate": _aggregate(rows, state["missing_penalty"]),
    }


def _evaluate_beta_candidate(value: float | tuple[float, ...]) -> dict[str, Any]:
    if WORKER_STATE is None:
        raise RuntimeError("Calibration worker was not initialized")
    state = WORKER_STATE
    beta = float(value[0] if isinstance(value, tuple) else value)
    beta = float(min(max(beta, BETA_BOUNDS[0]), BETA_BOUNDS[1]))
    pressure_config = dict(state["pressure_config"])
    pressure_config.update(
        enabled=True,
        beta_sp=beta,
        observed_zero_weather_gate=True,
    )
    rows = []
    for record in state["records"]:
        result = run_spore_plus(
            record["weather_path"],
            s_config_path=state["fitted_config_path"],
            spore_file=state["spore_file"],
            pressure_config=pressure_config,
            pressure_enabled=True,
            susceptibility_modulation_config=state["host_config"],
            observed_stage_dates=record.get("observed_stage_dates"),
        )
        rows.append(_metric_for_record(record, result, state["missing_penalty"], pressure=True))
    return {
        "vector": [beta],
        "beta_sp": beta,
        "rows": rows,
        "aggregate": _aggregate(rows, state["missing_penalty"]),
    }


def _unique_vectors(vectors: Iterable[Iterable[float]]) -> list[tuple[float, ...]]:
    unique = []
    seen = set()
    for vector in vectors:
        key = tuple(round(float(value), 8) for value in _resolved_vector(vector))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _fine_beta_grid(center: float) -> tuple[float, ...]:
    lower, upper = BETA_BOUNDS
    values = np.arange(
        center - 2 * BETA_FINE_STEP,
        center + 2 * BETA_FINE_STEP + BETA_FINE_STEP * 0.5,
        BETA_FINE_STEP,
    )
    return tuple(round(float(min(max(value, lower), upper)), 10) for value in values)


def _close_progress(progress: tqdm | None) -> None:
    """Close an aggregate progress bar after resolving clipped duplicates."""

    if progress is None:
        return
    progress.total = progress.n
    progress.refresh()
    progress.close()


def _parallel_evaluate(
    vectors: list[Any],
    *,
    context: ProcessPoolExecutor | None,
    evaluator,
    description: str,
    progress: tqdm | None = None,
) -> list[dict[str, Any]]:
    if context is not None:
        iterator = context.map(evaluator, vectors, chunksize=1)
    else:
        iterator = map(evaluator, vectors)
    if progress is None:
        return list(tqdm(iterator, total=len(vectors), desc=description, unit="candidate"))
    results = []
    for result in iterator:
        results.append(result)
        progress.update(1)
    return results


def _progress_bar(total: int, description: str) -> tqdm:
    return tqdm(
        total=total,
        desc=description,
        unit="candidate",
        dynamic_ncols=True,
        leave=True,
    )


def _fit_s_training(
    training_records: list[dict[str, Any]],
    *,
    s_config: dict[str, Any],
    pressure_config: dict[str, Any],
    host_config: str,
    workers: int,
    missing_penalty: float,
    mae_tolerance: float = DEFAULT_MAE_TOLERANCE_DAYS,
    progress_label: str | None = None,
) -> dict[str, Any]:
    state = {
        "records": training_records,
        "s_config": s_config,
        "pressure_config": pressure_config,
        "host_config": host_config,
        "missing_penalty": missing_penalty,
        "fixed_parameters": dict(FIXED_PARAMETERS),
        "parameter_names": tuple(PARAMETER_NAMES),
        "parameter_bounds": dict(PARAMETER_BOUNDS),
    }
    context = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(state,),
    ) if workers > 1 else None
    if context is None:
        _init_worker(state)
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    parameter_vectors = list(product(*(PARAMETER_GRIDS[name] for name in PARAMETER_NAMES)))
    progress = _progress_bar(len(parameter_vectors), f"{progress_label} parameter grid") if progress_label else None

    def evaluate(vectors: Iterable[Iterable[float]], stage: str, progress: tqdm | None):
        normalized = _unique_vectors(vectors)
        pending = [vector for vector in normalized if vector not in cache]
        if pending:
            results = _parallel_evaluate(
                pending,
                context=context,
                evaluator=_evaluate_s_candidate,
                description=stage,
                progress=progress,
            )
            for result in results:
                cache[tuple(result["vector"])] = result
            if progress is not None:
                progress.update(len(normalized) - len(pending))
        elif progress is not None:
            progress.update(len(normalized))
        return [cache[vector] for vector in normalized]

    try:
        # Evaluate the complete prespecified FSIM parameter grid once.
        evaluate(parameter_vectors, "S parameter grid", progress)
    finally:
        _close_progress(progress)
        if context is not None:
            context.shutdown()

    selected, audit = _select_stable_candidate(
        cache.values(), PARAMETER_NAMES,
        tuple(PARAMETER_BOUNDS[name] for name in PARAMETER_NAMES), mae_tolerance,
    )
    return {"selected": selected, "cache": cache, "selection_audit": audit}


def _fit_beta_training(
    training_records: list[dict[str, Any]],
    *,
    fitted_config_path: Path,
    spore_file: str,
    pressure_config: dict[str, Any],
    host_config: str,
    workers: int,
    fine_passes: int,
    missing_penalty: float,
    mae_tolerance: float = DEFAULT_MAE_TOLERANCE_DAYS,
    progress_label: str | None = None,
) -> dict[str, Any]:
    state = {
        "records": training_records,
        "fitted_config_path": str(fitted_config_path),
        "spore_file": spore_file,
        "pressure_config": pressure_config,
        "host_config": host_config,
        "missing_penalty": missing_penalty,
        "fixed_parameters": dict(FIXED_PARAMETERS),
        "parameter_names": tuple(PARAMETER_NAMES),
        "parameter_bounds": dict(PARAMETER_BOUNDS),
    }
    context = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(state,),
    ) if workers > 1 else None
    if context is None:
        _init_worker(state)
    cache: dict[float, dict[str, Any]] = {}
    coarse_progress = _progress_bar(len(BETA_COARSE_GRID) + 1, f"{progress_label} coarse") if progress_label else None
    fine_progress = None

    def evaluate(values: Iterable[float], stage: str, progress: tqdm | None):
        normalized = []
        seen = set()
        for value in values:
            key = round(float(min(max(value, BETA_BOUNDS[0]), BETA_BOUNDS[1])), 8)
            if key not in seen:
                seen.add(key)
                normalized.append(key)
        pending = [value for value in normalized if value not in cache]
        if pending:
            results = _parallel_evaluate(
                [(value,) for value in pending],
                context=context,
                evaluator=_evaluate_beta_candidate,
                description=stage,
                progress=progress,
            )
            for result in results:
                cache[round(float(result["beta_sp"]), 8)] = result
            if progress is not None:
                progress.update(len(normalized) - len(pending))
        elif progress is not None:
            progress.update(len(normalized))
        return [cache[value] for value in normalized]

    try:
        beta_anchor = sum(BETA_BOUNDS) / 2
        selected = min(
            evaluate((*BETA_COARSE_GRID, beta_anchor), "beta coarse", coarse_progress),
            key=_selection_key,
        )
        _close_progress(coarse_progress)
        coarse_progress = None
        current = float(selected["beta_sp"])
        if fine_passes > 0:
            fine_progress = _progress_bar(5 * fine_passes, f"{progress_label} fine") if progress_label else None
            for iteration in range(1, fine_passes + 1):
                selected = min(
                    evaluate(_fine_beta_grid(current), f"beta fine {iteration}/{fine_passes}", fine_progress),
                    key=_selection_key,
                )
                current = float(selected["beta_sp"])
    finally:
        _close_progress(coarse_progress)
        _close_progress(fine_progress)
        if context is not None:
            context.shutdown()
    selected, audit = _select_stable_candidate(
        cache.values(), (PRESSURE_PARAMETER_NAME,),
        (BETA_BOUNDS,), mae_tolerance,
    )
    return {"selected": selected, "cache": cache, "selection_audit": audit}


def _evaluate_held_out(
    record: dict[str, Any],
    *,
    fitted_config_path: Path,
    beta_sp: float,
    spore_file: str,
    pressure_config: dict[str, Any],
    host_config: str,
    missing_penalty: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # One pressure-enabled run supplies both outputs: ``process`` is the
    # fixed S baseline and ``spore_process`` is S+Spore.
    pressure_settings = dict(pressure_config)
    pressure_settings.update(
        enabled=True,
        beta_sp=beta_sp,
        observed_zero_weather_gate=True,
    )
    result = run_spore_plus(
        record["weather_path"],
        s_config_path=fitted_config_path,
        spore_file=spore_file,
        pressure_config=pressure_settings,
        pressure_enabled=True,
        susceptibility_modulation_config=host_config,
        observed_stage_dates=record.get("observed_stage_dates"),
    )
    return (
        _metric_for_record(record, result, missing_penalty, pressure=False),
        _metric_for_record(record, result, missing_penalty, pressure=True),
    )


def _cv_group(record: dict[str, Any], scheme: str) -> str:
    """Return the outer-CV group identifier for one record."""

    if scheme == "site_year":
        return str(record["site_year"])
    if scheme == "site":
        return str(record["site_id"])
    if scheme == "year":
        return str(record["year"])
    raise ValueError(f"Unknown CV scheme: {scheme}")


def _run_calibration(args) -> Path:
    global FIXED_PARAMETERS, PARAMETER_NAMES, PARAMETER_BOUNDS, PARAMETER_GRIDS
    if args.open_sporulation_rh_min_pct:
        FIXED_PARAMETERS = dict(FIXED_PARAMETERS)
        FIXED_PARAMETERS.pop("sporulation_rh_min_pct", None)
    if args.open_heavy_rain_washoff_mm:
        FIXED_PARAMETERS = dict(FIXED_PARAMETERS)
        FIXED_PARAMETERS.pop("heavy_rain_washoff_mm", None)
    if args.fixed_effective_deposition_min_rain_mm is not None:
        FIXED_PARAMETERS = dict(FIXED_PARAMETERS)
        FIXED_PARAMETERS["effective_deposition_min_rain_mm"] = float(
            args.fixed_effective_deposition_min_rain_mm
        )
    PARAMETER_NAMES = tuple(
        name for name in FULL_PARAMETER_NAMES if name not in FIXED_PARAMETERS
    )
    if args.effective_deposition_grid is not None:
        if "effective_deposition_min_rain_mm" not in PARAMETER_NAMES:
            raise ValueError("An effective-deposition grid requires the parameter to be open")
        values = tuple(sorted({round(float(value), 10) for value in args.effective_deposition_grid}))
        if len(values) < 2 or not all(np.isfinite(value) for value in values):
            raise ValueError("The effective-deposition grid must contain at least two finite values")
        PARAMETER_GRIDS = dict(PARAMETER_GRIDS)
        PARAMETER_BOUNDS = dict(PARAMETER_BOUNDS)
        PARAMETER_GRIDS["effective_deposition_min_rain_mm"] = values
        PARAMETER_BOUNDS["effective_deposition_min_rain_mm"] = (values[0], values[-1])
    if args.infection_degree_hours_grid is not None:
        if "infection_degree_hours" not in PARAMETER_NAMES:
            raise ValueError("An infection-degree grid requires the parameter to be open")
        values = tuple(sorted({round(float(value), 10) for value in args.infection_degree_hours_grid}))
        if len(values) < 2 or not all(np.isfinite(value) for value in values):
            raise ValueError("The infection-degree grid must contain at least two finite values")
        PARAMETER_GRIDS = dict(PARAMETER_GRIDS)
        PARAMETER_BOUNDS = dict(PARAMETER_BOUNDS)
        PARAMETER_GRIDS["infection_degree_hours"] = values
        PARAMETER_BOUNDS["infection_degree_hours"] = (values[0], values[-1])
    if args.sporulation_rh_grid is not None:
        if "sporulation_rh_min_pct" not in PARAMETER_NAMES:
            raise ValueError("A sporulation-RH grid requires the parameter to be open")
        values = tuple(sorted({round(float(value), 10) for value in args.sporulation_rh_grid}))
        if len(values) < 2 or not all(np.isfinite(value) for value in values):
            raise ValueError("The sporulation-RH grid must contain at least two finite values")
        PARAMETER_GRIDS = dict(PARAMETER_GRIDS)
        PARAMETER_BOUNDS = dict(PARAMETER_BOUNDS)
        PARAMETER_GRIDS["sporulation_rh_min_pct"] = values
        PARAMETER_BOUNDS["sporulation_rh_min_pct"] = (values[0], values[-1])
    if args.heavy_rain_washoff_grid is not None:
        if "heavy_rain_washoff_mm" not in PARAMETER_NAMES:
            raise ValueError("A wash-off grid requires the parameter to be open")
        values = tuple(sorted({round(float(value), 10) for value in args.heavy_rain_washoff_grid}))
        if len(values) < 2 or not all(np.isfinite(value) for value in values):
            raise ValueError("The wash-off grid must contain at least two finite values")
        PARAMETER_GRIDS = dict(PARAMETER_GRIDS)
        PARAMETER_BOUNDS = dict(PARAMETER_BOUNDS)
        PARAMETER_GRIDS["heavy_rain_washoff_mm"] = values
        PARAMETER_BOUNDS["heavy_rain_washoff_mm"] = (values[0], values[-1])
    s_config = _load_json(args.s_config)
    spore_config = _load_json(args.spore_config)
    pressure_template = dict(spore_config.get("pressure_adjustment", {}))
    # The accepted FSIM-S version uses the observed-zero weather-cohort gate
    # in every pressure-enabled calibration evaluation.
    pressure_template["observed_zero_weather_gate"] = True
    records = _build_records(
        args.weather_dir,
        args.observations,
        args.phenology_observations,
        args.limit,
    )
    records = sorted(records, key=lambda item: item["site_year"])
    if args.limit is None and len(records) != 6:
        raise ValueError(f"Expected six Guangxi site-years, found {len(records)}")

    spore_file = str(args.spore_file.resolve())
    host_config = str(args.host_config.resolve())
    schemes = list(dict.fromkeys(args.cv_scheme))
    if not schemes:
        raise ValueError("At least one CV scheme is required")
    fold_parameters: list[dict[str, Any]] = []
    s_candidate_rows: list[dict[str, Any]] = []
    beta_candidate_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    training_metric_rows: dict[str, list[dict[str, Any]]] = {
        "FSIM": [],
        "FSIM-S": [],
    }
    selection_audits = []

    for scheme in schemes:
        groups = sorted({_cv_group(record, scheme) for record in records})
        for fold_index, held_out in enumerate(groups, start=1):
            held_out_records = [record for record in records if _cv_group(record, scheme) == held_out]
            training = [record for record in records if _cv_group(record, scheme) != held_out]
            if not training:
                raise ValueError(f"CV scheme {scheme!r}, held-out group {held_out!r} leaves no training records")

            s_fit = _fit_s_training(
                training,
                s_config=s_config,
                pressure_config=pressure_template,
                host_config=host_config,
                workers=args.workers,
                missing_penalty=args.missing_penalty,
                mae_tolerance=args.mae_tolerance,
                progress_label=f"{scheme} fold {fold_index}/{len(groups)} {held_out} FSIM",
            )
            selected_s = s_fit["selected"]
            training_metric_rows["FSIM"].extend(
                {**row, "cv_scheme": scheme, "held_out_group": held_out}
                for row in selected_s["rows"]
            )
            fitted_s_config = _candidate_config(selected_s["vector"], s_config)
            fitted_s_path = _temporary_config(fitted_s_config)
            try:
                for result in s_fit["cache"].values():
                    s_candidate_rows.append({
                        "cv_scheme": scheme,
                        "held_out_group": held_out,
                        "stage": "S_training",
                        **{f"fixed_{name}": value for name, value in FIXED_PARAMETERS.items()},
                        **{name: value for name, value in zip(PARAMETER_NAMES, result["vector"])},
                        **result["aggregate"],
                    })

                beta_fit = _fit_beta_training(
                    training,
                    fitted_config_path=fitted_s_path,
                    spore_file=spore_file,
                    pressure_config=pressure_template,
                    host_config=host_config,
                    workers=args.workers,
                    fine_passes=args.beta_fine_passes,
                    missing_penalty=args.missing_penalty,
                    mae_tolerance=args.mae_tolerance,
                    progress_label=f"{scheme} fold {fold_index}/{len(groups)} {held_out} FSIM-S pressure",
                )
                selected_beta = float(beta_fit["selected"]["beta_sp"])
                selection_audits.append({
                    "cv_scheme": scheme,
                    "held_out_group": held_out,
                    "held_out_site_years": [record["site_year"] for record in held_out_records],
                    "FSIM": s_fit["selection_audit"],
                    "FSIM-S_beta": beta_fit["selection_audit"],
                })
                training_metric_rows["FSIM-S"].extend(
                    {**row, "cv_scheme": scheme, "held_out_group": held_out}
                    for row in beta_fit["selected"]["rows"]
                )
                for result in beta_fit["cache"].values():
                    beta_candidate_rows.append({
                        "cv_scheme": scheme,
                        "held_out_group": held_out,
                        "stage": "beta_training",
                        **{f"fixed_{name}": value for name, value in FIXED_PARAMETERS.items()},
                        **{name: value for name, value in zip(PARAMETER_NAMES, selected_s["vector"])},
                        "beta_sp": result["beta_sp"],
                        **result["aggregate"],
                    })

                held_out_metrics = []
                for held_out_record in held_out_records:
                    baseline, pressure = _evaluate_held_out(
                        held_out_record,
                        fitted_config_path=fitted_s_path,
                        beta_sp=selected_beta,
                        spore_file=spore_file,
                        pressure_config=pressure_template,
                        host_config=host_config,
                        missing_penalty=args.missing_penalty,
                    )
                    held_out_metrics.append((baseline, pressure))
                    for model, metric in (("FSIM", baseline), ("FSIM-S", pressure)):
                        prediction_rows.append({
                            "cv_scheme": scheme,
                            "held_out_group": held_out,
                            "held_out_site_year": held_out,
                            "site_year": held_out_record["site_year"],
                            "model": model,
                            **FIXED_PARAMETERS,
                            **{name: value for name, value in zip(PARAMETER_NAMES, selected_s["vector"])},
                            "beta_sp": selected_beta,
                            **metric,
                        })
                baseline_metrics = [item[0] for item in held_out_metrics]
                pressure_metrics = [item[1] for item in held_out_metrics]
                baseline_aggregate = _aggregate(baseline_metrics, args.missing_penalty)
                pressure_aggregate = _aggregate(pressure_metrics, args.missing_penalty)
                fold_parameters.append({
                    "cv_scheme": scheme,
                    "held_out_group": held_out,
                    "held_out_n": len(held_out_records),
                **{f"s_{name}": value for name, value in FIXED_PARAMETERS.items()},
                **{f"s_{name}": value for name, value in zip(PARAMETER_NAMES, selected_s["vector"])},
                "s_training_mae_days": selected_s["aggregate"]["mae_days"],
                "s_training_rmse_days": selected_s["aggregate"]["rmse_days"],
                "s_training_bias_days": selected_s["aggregate"]["bias_days"],
                "s_training_window_coverage": selected_s["aggregate"]["window_coverage"],
                "s_minimum_training_mae_days": s_fit["selection_audit"]["minimum_training_penalized_mae_days"],
                "s_selected_mae_increase_days": s_fit["selection_audit"]["selected_mae_increase_days"],
                "s_near_optimal_count": s_fit["selection_audit"]["near_optimal_count"],
                "beta_sp": selected_beta,
                "beta_training_mae_days": beta_fit["selected"]["aggregate"]["mae_days"],
                "beta_training_rmse_days": beta_fit["selected"]["aggregate"]["rmse_days"],
                "beta_training_bias_days": beta_fit["selected"]["aggregate"]["bias_days"],
                "beta_training_window_coverage": beta_fit["selected"]["aggregate"]["window_coverage"],
                "beta_selected_mae_increase_days": beta_fit["selection_audit"]["selected_mae_increase_days"],
                "beta_near_optimal_count": beta_fit["selection_audit"]["near_optimal_count"],
                "s_held_out_mae_days": baseline_aggregate["mae_days"],
                "s_held_out_bias_days": baseline_aggregate["bias_days"],
                "s_held_out_window_coverage": baseline_aggregate["window_coverage"],
                "fsim_s_held_out_mae_days": pressure_aggregate["mae_days"],
                "fsim_s_held_out_bias_days": pressure_aggregate["bias_days"],
                "fsim_s_held_out_window_coverage": pressure_aggregate["window_coverage"],
                "s_held_out_error_days": baseline_metrics[0]["error_days"] if len(baseline_metrics) == 1 else None,
                "s_held_out_complete_window": baseline_metrics[0]["complete_window"] if len(baseline_metrics) == 1 else None,
                "fsim_s_held_out_error_days": pressure_metrics[0]["error_days"] if len(pressure_metrics) == 1 else None,
                "fsim_s_held_out_complete_window": pressure_metrics[0]["complete_window"] if len(pressure_metrics) == 1 else None,
                })
            finally:
                fitted_s_path.unlink(missing_ok=True)

    predictions = pd.DataFrame(prediction_rows)
    summaries = pd.DataFrame([
        {
            "cv_scheme": scheme,
            "model": model,
            **_aggregate(
                [row for row in prediction_rows if row["cv_scheme"] == scheme and row["model"] == model],
                args.missing_penalty,
            ),
        }
        for scheme in schemes
        for model in ("FSIM", "FSIM-S")
    ])
    terminal_summary_rows = []
    for scheme in schemes:
        for dataset in ("training_folds_pooled", "held_out_validation"):
            rows_by_model = {
                model: (
                    [row for row in training_metric_rows[model] if row["cv_scheme"] == scheme]
                    if dataset == "training_folds_pooled"
                    else [row for row in prediction_rows if row["cv_scheme"] == scheme and row["model"] == model]
                )
                for model in ("FSIM", "FSIM-S")
            }
            for model in ("FSIM", "FSIM-S"):
                terminal_summary_rows.append({
                    "cv_scheme": scheme,
                    "dataset": dataset,
                    "model": model,
                    **_aggregate(rows_by_model[model], args.missing_penalty),
                })
    terminal_summary = pd.DataFrame(terminal_summary_rows)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(s_candidate_rows).to_csv(output_dir / "s_candidate_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(beta_candidate_rows).to_csv(output_dir / "beta_candidate_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_parameters).to_csv(output_dir / "fold_parameters.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {**row, "model": model, "is_held_out": 0}
        for model, rows in training_metric_rows.items()
        for row in rows
    ]).to_csv(output_dir / "training_predictions.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    terminal_summary.to_csv(output_dir / "training_validation_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "selection_audit.json").write_text(
        json.dumps(selection_audits, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    stability_rows = []
    for scheme in schemes:
        scheme_folds = [row for row in fold_parameters if row["cv_scheme"] == scheme]
        for name in (*PARAMETER_NAMES, PRESSURE_PARAMETER_NAME):
            column = "beta_sp" if name == PRESSURE_PARAMETER_NAME else f"s_{name}"
            values = np.asarray([row[column] for row in scheme_folds], dtype=float)
            lower, upper = BETA_BOUNDS if name == PRESSURE_PARAMETER_NAME else PARAMETER_BOUNDS[name]
            stability_rows.append({
                "cv_scheme": scheme, "parameter": name, "n_folds": len(values),
                "mean": float(values.mean()), "median": float(np.median(values)),
                "min": float(values.min()), "max": float(values.max()),
                "std": float(values.std(ddof=0)),
                "normalized_std": float(values.std(ddof=0) / (upper - lower)),
            })
    pd.DataFrame(stability_rows).to_csv(output_dir / "parameter_stability.csv", index=False)

    manifest = {
        "schema_version": 4,
        "model": "nested_fsim_s_grouped_cv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_cv_scheme": "site_year",
        "cv_schemes": schemes,
        "supplementary_cv_schemes": [scheme for scheme in schemes if scheme != "site_year"],
        "site_years": [record["site_year"] for record in records],
        "fold_training": "fit three open FSIM parameters on all records outside the held-out group with wind, wash-off and minimum moist hours fixed; then fix FSIM and fit beta_sp on the same training records",
        "held_out_evaluation": "fixed fold S parameters for Sporangia and fixed fold S plus beta_sp for Spore",
        "observed_window": "[observed_date - 3 days, observed_date]",
        "error_definition": "zero for overlapping observed and predicted windows; otherwise signed distance to the nearest window boundary",
        "selection_target": "minimum missing predictions; among candidates within a fixed tolerance of minimum training penalized MAE, minimize domain-normalized squared distance to the component-wise median of the near-optimal training candidates; then MAE, non-positive bias, absolute bias, lexicographic vector",
        "mae_tolerance_days": args.mae_tolerance,
        "search_navigation": "evaluate one complete finite Cartesian FSIM parameter grid; select from the near-optimal training candidate set using its normalized median center",
        "selection_scope": "training site-years only; no held-out labels, configuration prior, prior exploratory optimum, or cross-fold parameter consensus used for selection",
        "evaluation_status": "exploratory protocol revision after inspection of earlier LOSYO results; not independent confirmatory validation",
        "stability_definition": "component-wise median center of the near-optimal training candidate set within each fold; actual between-fold dispersion reported separately",
        "training_summary_unit": "pooled fold/site-year predictions within each CV scheme; repeated observations, not independent site-years",
        "missing_penalty_days": float(args.missing_penalty),
        "parameter_names": list(PARAMETER_NAMES),
        "fixed_parameters": FIXED_PARAMETERS,
        "parameter_bounds": PARAMETER_BOUNDS,
        "parameter_grid": PARAMETER_GRIDS,
        "beta_fine_passes": int(args.beta_fine_passes),
        "pressure_parameter": PRESSURE_PARAMETER_NAME,
        "observed_zero_weather_gate": True,
        "pressure_bounds": list(BETA_BOUNDS),
        "pressure_coarse_grid": BETA_COARSE_GRID,
        "pressure_fine_step": BETA_FINE_STEP,
        "pressure_preprocessing": {
            "shape": "interval_rate",
            "normalization": "zscore",
            "rolling_days": pressure_template["rolling_days"],
            "outside_fill": "zero",
            "reference_level": pressure_template["reference_level"],
            "observed_zero_weather_gate": pressure_template["observed_zero_weather_gate"],
        },
        "inputs": {
            "weather_dir": str(args.weather_dir.resolve()),
            "observations": str(args.observations.resolve()),
            "phenology_observations": str(args.phenology_observations.resolve()),
            "spore_file": spore_file,
            "s_config": str(args.s_config.resolve()),
            "spore_config": str(args.spore_config.resolve()),
            "host_config": host_config,
        },
        "outputs": [
            "s_candidate_scores.csv",
            "beta_candidate_scores.csv",
            "fold_parameters.csv",
            "predictions.csv",
            "summary.csv",
            "training_validation_summary.csv",
            "selection_audit.json",
            "parameter_stability.csv",
            "manifest.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Calibration summary:", flush=True)
    print(terminal_summary.to_string(index=False), flush=True)
    print(f"Results: {output_dir}", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-dir", type=Path, default=DEFAULT_WEATHER_DIR)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--phenology-observations", type=Path, default=DEFAULT_PHENOLOGY_OBSERVATIONS)
    parser.add_argument("--spore-file", type=Path, default=DEFAULT_SPORE_FILE)
    parser.add_argument("--s-config", type=Path, default=DEFAULT_S_CONFIG)
    parser.add_argument("--spore-config", type=Path, default=DEFAULT_SPORE_CONFIG)
    parser.add_argument("--host-config", type=Path, default=DEFAULT_HOST_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--beta-fine-passes", "--fine-passes", dest="beta_fine_passes", type=int, default=1,
        help="Optional local refinement passes for the separate beta_sp calibration",
    )
    parser.add_argument("--missing-penalty", type=float, default=DEFAULT_MISSING_PENALTY)
    parser.add_argument("--mae-tolerance", type=float, default=DEFAULT_MAE_TOLERANCE_DAYS,
                        help="Fixed near-optimal training MAE tolerance in days for data-derived stability selection")
    parser.add_argument(
        "--fixed-effective-deposition-min-rain-mm", type=float, default=None,
        help="Fix the effective deposition threshold across folds instead of fitting it in each fold",
    )
    parser.add_argument(
        "--open-sporulation-rh-min-pct", action="store_true",
        help="Retained for compatibility; the sporulation RH threshold is open by default",
    )
    parser.add_argument(
        "--effective-deposition-grid", type=float, nargs="+", default=None,
        help="Optional replacement grid for the open effective-deposition threshold",
    )
    parser.add_argument(
        "--infection-degree-hours-grid", type=float, nargs="+", default=None,
        help="Optional replacement grid for the open infection degree-hour threshold",
    )
    parser.add_argument(
        "--sporulation-rh-grid", type=float, nargs="+", default=None,
        help="Optional replacement grid for the open sporulation relative-humidity threshold",
    )
    parser.add_argument(
        "--open-heavy-rain-washoff-mm", action="store_true",
        help="Fit the heavy-rain wash-off threshold as an additional S parameter",
    )
    parser.add_argument(
        "--heavy-rain-washoff-grid", type=float, nargs="+", default=None,
        help="Optional replacement grid for the open heavy-rain wash-off threshold",
    )
    parser.add_argument(
        "--cv-scheme", nargs="+", choices=CV_SCHEMES, default=["site_year"],
        help="Outer grouped leave-out schemes; site_year is primary, site and year are supplementary evaluations",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of site-years processed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    if args.beta_fine_passes < 0:
        parser.error("beta-fine-passes must be nonnegative")
    if args.missing_penalty <= 0 or not np.isfinite(args.missing_penalty):
        parser.error("missing-penalty must be finite and positive")
    if args.mae_tolerance < 0 or not np.isfinite(args.mae_tolerance):
        parser.error("mae-tolerance must be finite and nonnegative")
    _run_calibration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
