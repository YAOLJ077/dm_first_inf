#!/usr/bin/env python3
"""Calibrate the historical grapevine phenology model with grouped CV.

North and south use separate Force-only parameter vectors. Folds are assigned by
complete site-year within each region, so every held-out prediction is out-of-sample.
The default inputs are the curated historical phenology tables in this repository.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.phenology.calibration import (  # noqa: E402
    DEFAULT_EXCLUDED_SITE_YEARS, TARGETS, LATITUDE_CUTOFF, errors,
    fit_thresholds, predict_force as predict, prepare_targets, prepare_weather,
    sha256, EARLY_BASE_C, build_force_config,
)


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "data/phenology_data/historical_data/phenology_model_observations.csv",
    )
    p.add_argument(
        "--site-years",
        type=Path,
        default=ROOT / "data/phenology_data/historical_data/phenology_model_site_years.csv",
    )
    p.add_argument("--config", type=Path, default=ROOT / "model/phenology/config.json")
    p.add_argument("--exclude-site-years", nargs="*", default=list(DEFAULT_EXCLUDED_SITE_YEARS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260924, 20260925, 20260926])
    p.add_argument("--chill-bounds", type=float, nargs=2, default=[20., 180.])
    p.add_argument("--force-bounds", type=float, nargs=2, default=[.001, 3000.])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "result/historical_calibration_phenology",
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.folds < 2 or a.maxiter < 0 or a.popsize < 1 or a.workers < 1:
        p.error("folds >= 2, maxiter >= 0, popsize >= 1 and workers >= 1 are required")
    if not 0 <= a.seed < 2**32 or any(not 0 <= s < 2**32 for s in a.seeds):
        p.error("seeds must be nonnegative 32-bit integers")
    if not 0 < a.chill_bounds[0] < a.chill_bounds[1] or not 0 < a.force_bounds[0] < a.force_bounds[1]:
        p.error("bounds must satisfy 0 < low < high")
    return a


def assign_folds(sites, folds, seed):
    if len(sites) < folds:
        raise ValueError(f"{len(sites)} site-years cannot form {folds} folds")
    rng = np.random.default_rng(seed)
    values = np.array(sorted(sites), dtype=object); rng.shuffle(values)
    return {str(v): int(i % folds) for i, v in enumerate(values)}


def rows_for(targets, predictions, fold, region):
    values, interval, midpoint = errors(targets, predictions)
    out = targets[["site_year", "year", "bbch", "n_varieties", "n_records"]].copy()
    out["fold"] = fold; out["region"] = region
    out["predicted_date"] = [date.fromordinal(int(v)).isoformat() if np.isfinite(v) else "" for v in values]
    out["interval_error_days"] = interval; out["midpoint_error_days"] = midpoint
    out["attained"] = np.isfinite(values)
    out["observed_start"] = targets.lower.map(lambda v: pd.Timestamp(date.fromordinal(int(v))).isoformat())
    out["observed_end"] = targets.upper.map(lambda v: pd.Timestamp(date.fromordinal(int(v))).isoformat())
    return out


def metrics(pred):
    x = pred.copy(); x["abs_error_days"] = x.midpoint_error_days.abs()
    return x.groupby(["region", "fold", "bbch"], as_index=False).agg(
        n=("bbch", "size"), attained=("attained", "sum"), mae_days=("abs_error_days", "mean"),
        bias_days=("midpoint_error_days", "mean"), median_abs_days=("abs_error_days", "median"),
        within_7d=("abs_error_days", lambda z: float((z <= 7).mean())),
        within_14d=("abs_error_days", lambda z: float((z <= 14).mean())))


def fit_south(targets, seasons, a, label):
    best, records = fit_thresholds(targets, seasons, [a.force_bounds] * 3,
                                   seeds=a.seeds, maxiter=a.maxiter,
                                   popsize=a.popsize, workers=a.workers)
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError(f"No finite south fit for {label}")
    return best, records


def make_cf_config(template, vector):
    raw = json.loads(template.read_text(encoding="utf-8"))
    raw.update(model="sequential_chill_force", chill_min_c=CHILL_MIN,
               chill_opt_c=CHILL_OPT, chill_max_c=CHILL_MAX,
               chill_requirement=float(vector[0]), forcing_base_c=0.0,
               forcing_start_month=1, forcing_start_day=1, early_base_c=EARLY_BASE_C)
    raw["stage_targets"] = [{"name": name, "bbch": bbch, "requirement": float(value)}
                            for value, (bbch, name) in zip(vector[1:], TARGETS.items())]
    raw["status"] = "mixed_regional_cross_validation_full_fit"
    return raw


def make_force_config(template, vector):
    raw = json.loads(template.read_text(encoding="utf-8"))
    raw.update(model="force_only", chill_requirement=None, forcing_base_c=0.0,
               forcing_start_month=1, forcing_start_day=1, early_base_c=EARLY_BASE_C)
    raw["stage_targets"] = [{"name": name, "bbch": bbch, "requirement": float(value)}
                            for value, (bbch, name) in zip(vector, TARGETS.items())]
    raw["status"] = "mixed_regional_cross_validation_full_fit"
    return raw


def main(a):
    out = a.output_dir.resolve()
    if out.exists() and not a.overwrite:
        raise FileExistsError(f"Output exists: {out}; use --overwrite")
    targets, audit = prepare_targets(a.observations, region="all", exclude_site_years=a.exclude_site_years)
    index = pd.read_csv(a.site_years, encoding="utf-8-sig", dtype={"site_year": str})
    region_map = index.set_index("site_year")["latitude"].map(
        lambda z: "north" if float(z) >= LATITUDE_CUTOFF else "south").to_dict()
    targets["region"] = targets.site_year.map(region_map)
    if targets.region.isna().any():
        raise ValueError("Missing region for target site-year")
    force_config = build_force_config(a.config)
    force_seasons = prepare_weather(a.site_years, targets, force_config)
    region_sites = {r: sorted(targets.loc[targets.region.eq(r), "site_year"].unique()) for r in ("north", "south")}
    if any(len(v) < a.folds for v in region_sites.values()):
        raise ValueError(f"Not enough site-years for {a.folds} folds: { {k:len(v) for k,v in region_sites.items()} }")
    fold_maps = {"north": assign_folds(region_sites["north"], a.folds, a.seed),
                 "south": assign_folds(region_sites["south"], a.folds, a.seed + 1)}
    prediction_tables, fold_params = [], []
    for fold in range(a.folds):
        for region in ("north", "south"):
            test_sites = [s for s in region_sites[region] if fold_maps[region][s] == fold]
            train_sites = [s for s in region_sites[region] if s not in test_sites]
            train = targets[targets.site_year.isin(train_sites)].drop(columns="region")
            test = targets[targets.site_year.isin(test_sites)].drop(columns="region")
            best, restarts = fit_thresholds(
                train, {s: force_seasons[s] for s in train_sites}, [a.force_bounds] * 3,
                seeds=a.seeds, maxiter=a.maxiter, popsize=a.popsize,
                workers=a.workers, kind="force"
            )
            pred = predict({s: force_seasons[s] for s in test_sites}, best.x)
            names = ("H09", "H09_65", "H65_81")
            prediction_tables.append(rows_for(test, pred, fold, region))
            fold_params.append({"region": region, "fold": fold, **dict(zip(names, best.x)),
                                "objective_days": float(best.fun), "optimizer_converged": bool(best.success),
                                "restarts": json.dumps(restarts, ensure_ascii=False)})
    pred = pd.concat(prediction_tables, ignore_index=True)
    fold_metrics = metrics(pred)
    final_vectors = {}
    final_params = []
    for region in ("north", "south"):
        sites = region_sites[region]; train = targets[targets.site_year.isin(sites)].drop(columns="region")
        best, restarts = fit_thresholds(
            train, {s: force_seasons[s] for s in sites}, [a.force_bounds] * 3,
            seeds=a.seeds, maxiter=a.maxiter, popsize=a.popsize,
            workers=a.workers, kind="force"
        )
        names = ("H09", "H09_65", "H65_81")
        final_vectors[region] = best.x
        final_params.append({"region": region, **dict(zip(names, best.x)), "objective_days": float(best.fun),
                             "optimizer_converged": bool(best.success), "restarts": json.dumps(restarts, ensure_ascii=False)})
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "cv_predictions.csv", index=False); fold_metrics.to_csv(out / "cv_metrics.csv", index=False)
    pd.DataFrame(fold_params).to_csv(out / "fold_parameters.csv", index=False)
    pd.DataFrame(final_params).to_csv(out / "final_parameters.csv", index=False)
    (out / "final_config_north.json").write_text(json.dumps(make_force_config(a.config, final_vectors["north"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "final_config_south.json").write_text(json.dumps(make_force_config(a.config, final_vectors["south"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit.to_csv(out / "observation_usage.csv", index=False)
    x = pred.copy(); x["abs_error_days"] = x.midpoint_error_days.abs()
    overall = x.groupby(["region", "bbch"], as_index=False).agg(
        n=("bbch", "size"), attained=("attained", "sum"), mae_days=("abs_error_days", "mean"),
        bias_days=("midpoint_error_days", "mean"), median_abs_days=("abs_error_days", "median"),
        within_7d=("abs_error_days", lambda z: float((z <= 7).mean())),
        within_14d=("abs_error_days", lambda z: float((z <= 14).mean())))
    overall.to_csv(out / "cv_summary_by_stage.csv", index=False)
    summary = {"model_north": "Force-only_Tb0", "model_south": "Force-only_Tb0",
               "latitude_cutoff_n": LATITUDE_CUTOFF, "folds": a.folds, "fold_seed": a.seed,
               "site_year_count_all_targets": int(targets.site_year.nunique()),
               "site_year_count_by_region": {r: len(v) for r, v in region_sites.items()},
               "grouped_targets": int(len(targets)), "stage_counts": {f"{b:02d}": int(targets.bbch.eq(b).sum()) for b in TARGETS},
               "workers": a.workers, "excluded_site_years": list(a.exclude_site_years),
               "forcing_bounds": list(a.force_bounds),
               "inputs": {str(p.resolve()): sha256(p) for p in (a.observations, a.site_years, a.config)}}
    (out / "cv_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["分区域物候模型交叉验证：北方 Force-only，南方 Force-only", "", overall.to_string(index=False), "", "最终参数："]
    report.extend(pd.DataFrame(final_params).drop(columns="restarts").to_string(index=False).splitlines())
    (out / "cv_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main(args())
