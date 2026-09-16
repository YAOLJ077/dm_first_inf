#!/usr/bin/env python3
"""Current FSIM Morris screening for BBCH15-gated onset-window prediction.

This script deliberately does not use historical infection dates, legacy
calibration files, or the old exact-date objective.  It screens only the
continuous parameters that remain plausible for the current sporangium-pool
formulation.  Dispersal/deposition structures are handled separately.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from SALib.analyze import morris as morris_analyze
    from SALib.sample import morris as morris_sample
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("SALib is required: install it with `python -m pip install SALib`") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import config as BASE_CONFIG  # noqa: E402
from model.crop_season import CropSeasonFirst  # noqa: E402


# These ranges are the current, literature-constrained screening space.
# Structural dispersal/deposition alternatives are intentionally excluded.
PARAM_RANGES = {
    "zoospore_alive__vpd": (1.0, 5.0),
    "infection__infection_condition": (45.0, 90.0),
}

DEFAULT_MANIFEST = ROOT / "result" / "fsim_inputs" / "site_year_manifest.csv"
DEFAULT_OUTPUT = ROOT / "result" / "morris_fsim_current"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-traj", type=int, default=64)
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, choices=(2, 4), default=2)
    p.add_argument(
        "--use-wind-gate", action=argparse.BooleanOptionalAction, default=True,
        help="Select M1; --no-use-wind-gate selects M0. Wind threshold is held fixed here.",
    )
    return p.parse_args()


def apply_params(values: np.ndarray, names: list[str]) -> dict:
    cfg = copy.deepcopy(BASE_CONFIG)
    for value, key in zip(values, names):
        group, field = key.split("__", 1)
        cfg[group][field] = float(value)
    return cfg


def interval_distance(l: pd.Timestamp, u: pd.Timestamp,
                      start: pd.Timestamp, end: pd.Timestamp) -> float:
    if end < l:
        return float((l - end).total_seconds() / 86400.0)
    if start > u:
        return float((start - u).total_seconds() / 86400.0)
    return 0.0


def load_cases(manifest_path: Path) -> list[dict]:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"input_key", "L", "U", "weather_json"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    cases = []
    for row in manifest.itertuples(index=False):
        cases.append({
            "input_key": row.input_key,
            "L": pd.Timestamp(row.L),
            "U": pd.Timestamp(row.U),
            "weather_json": str(ROOT / row.weather_json),
        })
    return cases


def evaluate_sample(task: tuple[int, np.ndarray, list[str], list[dict], bool]) -> dict:
    sample_id, values, names, cases, use_wind_gate = task
    cfg = apply_params(values, names)
    successes = 0
    distances: list[float] = []
    midpoint_errors: list[float] = []
    candidate_counts: list[int] = []
    for case in cases:
        with open(case["weather_json"], encoding="utf-8") as handle:
            rb = json.load(handle)
        try:
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate(use_wind_gate=use_wind_gate)
            candidates = cs.primary_chain_unique
            n_candidates = 0 if candidates is None else len(candidates)
            candidate_counts.append(n_candidates)
            if candidates is None or candidates.empty:
                continue
            first = candidates.sort_values("oil_spot_start").iloc[0]
            start = pd.Timestamp(first["oil_spot_start"])
            end = pd.Timestamp(first["oil_spot_end"])
            successes += 1
            distances.append(interval_distance(case["L"], case["U"], start, end))
            obs_mid = case["L"] + (case["U"] - case["L"]) / 2
            pred_mid = start + (end - start) / 2
            midpoint_errors.append((pred_mid - obs_mid).total_seconds() / 86400.0)
        except Exception:
            candidate_counts.append(0)

    n_cases = len(cases)
    coverage = successes / n_cases if n_cases else 0.0
    # Timing loss is conditional on a prediction. Coverage is reported
    # separately so no arbitrary penalty is assigned to missing predictions.
    conditional_loss = float(np.mean(distances)) if distances else np.nan
    conditional_midpoint = float(np.mean(np.abs(midpoint_errors))) if midpoint_errors else np.nan
    return {
        "sample_id": sample_id,
        "coverage": coverage,
        "n_predictions": successes,
        "n_no_predictions": n_cases - successes,
        "conditional_interval_distance_days": conditional_loss,
        "conditional_abs_midpoint_error_days": conditional_midpoint,
        "mean_candidate_windows": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }


def analyze_response(problem: dict, X: np.ndarray, y: np.ndarray,
                     label: str, levels: int) -> pd.DataFrame:
    valid = np.isfinite(y)
    # Morris elementary effects require complete trajectories. Conditional
    # timing responses with missing predictions are therefore reported as
    # unavailable rather than analyzed on a filtered, non-Morris design.
    if valid.sum() < len(y) or valid.sum() < max(20, problem["num_vars"] * 4):
        return pd.DataFrame({"parameter": problem["names"], "mu": np.nan,
                             "mu_star": np.nan, "sigma": np.nan,
                             "response": label, "n_valid": int(valid.sum())})
    si = morris_analyze.analyze(
        problem, X[valid], y[valid], num_levels=levels, print_to_console=False
    )
    return pd.DataFrame({
        "parameter": problem["names"],
        "mu": si["mu"],
        "mu_star": si["mu_star"],
        "sigma": si["sigma"],
        "response": label,
        "n_valid": int(valid.sum()),
    }).sort_values("mu_star", ascending=False, na_position="last")


def main() -> None:
    args = parse_args()
    if args.n_traj <= 0 or args.levels < 4:
        raise ValueError("n-traj must be positive and levels must be at least 4")
    cases = load_cases(args.manifest)
    names = list(PARAM_RANGES)
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [PARAM_RANGES[name] for name in names],
    }
    X = morris_sample.sample(problem, N=args.n_traj, num_levels=args.levels, seed=args.seed)
    tasks = [(i, X[i], names, cases, args.use_wind_gate) for i in range(len(X))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(evaluate_sample, tasks))
    sample_df = pd.DataFrame(records).sort_values("sample_id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.output_dir / f"morris_samples_N{args.n_traj}_seed{args.seed}.csv", index=False)
    np.save(args.output_dir / f"morris_X_N{args.n_traj}_seed{args.seed}.npy", X)
    sensitivity = pd.concat([
        analyze_response(problem, X, sample_df["coverage"].to_numpy(), "coverage", args.levels),
        analyze_response(problem, X, sample_df["conditional_interval_distance_days"].to_numpy(), "conditional_interval_distance", args.levels),
        analyze_response(problem, X, sample_df["conditional_abs_midpoint_error_days"].to_numpy(), "conditional_abs_midpoint_error", args.levels),
        analyze_response(problem, X, sample_df["mean_candidate_windows"].to_numpy(), "candidate_windows", args.levels),
    ], ignore_index=True)
    sensitivity.to_csv(args.output_dir / f"morris_sensitivity_N{args.n_traj}_seed{args.seed}.csv", index=False)
    (args.output_dir / "parameter_space.json").write_text(
        json.dumps({"problem": problem, "n_site_years": len(cases),
                    "gate": "BBCH15 infection gate", "workers": args.workers,
                    "use_wind_gate": args.use_wind_gate,
                    "wind_threshold_screened": False,
                    "leaf_moisture_proxy": "physical VPD only",
                    "incubation_completion": 1.0}, indent=2),
        encoding="utf-8",
    )
    print(sensitivity.to_string(index=False))
    print(f"Saved Morris outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
