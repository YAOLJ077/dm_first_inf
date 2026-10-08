#!/usr/bin/env python3
"""Compare compact temperature-response phenology models with grouped CV.

This calibration deliberately keeps only three fitted stage requirements
(BBCH09, BBCH65 and BBCH81).  BBCH53 is retained in the target table and
reported as an auxiliary interpolation diagnostic.  Exact observations,
right/left censoring and interval observations are scored differently, so a
first observation at BBCH13 is not incorrectly treated as BBCH09.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[2]
STAGES = (9, 65, 81)
STAGE_NAMES = {9: "budburst", 65: "flowering", 81: "veraison"}
STAGE_WEIGHT = {9: 1.0, 65: 1.25, 81: 0.35}
LOCAL_MULTIPLIER = 4.0
UNATTAINED_PENALTY = 180.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/phenology_data/calibration_data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "result/phenology_table_vine_no_guilin")
    p.add_argument("--maxiter", type=int, default=55)
    p.add_argument("--popsize", type=int, default=7)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261008, 20261009])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def load_data(data_dir: Path, targets_path: Path | None = None):
    targets_file = targets_path or (data_dir / "phenology_calibration_targets.csv")
    targets = pd.read_csv(targets_file, encoding="utf-8-sig")
    daily = pd.read_csv(data_dir / "phenology_weather_daily_bjt.csv", encoding="utf-8-sig")
    targets["target_stage"] = targets.target_stage.astype(int)
    targets["observed_start_ordinal"] = pd.to_numeric(targets.observed_start_ordinal, errors="coerce")
    targets["observed_end_ordinal"] = pd.to_numeric(targets.observed_end_ordinal, errors="coerce")
    if "calibration_lower_ordinal" in targets:
        targets["calibration_lower_ordinal"] = pd.to_numeric(targets.calibration_lower_ordinal, errors="coerce")
    else:
        targets["calibration_lower_ordinal"] = targets.observed_start_ordinal
    targets["quality_weight"] = pd.to_numeric(targets.quality_weight, errors="coerce").fillna(.5)
    targets = targets.loc[targets.target_stage.isin(STAGES)].copy()
    daily["date_bjt_dt"] = pd.to_datetime(daily.date_bjt)
    daily["temperature_c"] = pd.to_numeric(daily.temperature_c, errors="raise")
    weather = {}
    for key, group in daily.groupby("site_year", sort=True):
        rows = group.sort_values("date_bjt_dt")
        year = int(rows.year.iloc[0])
        rows = rows.loc[rows.date_bjt_dt.dt.year.eq(year)].copy().reset_index(drop=True)
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        if not np.array_equal(rows.date_bjt_dt.to_numpy(dtype="datetime64[ns]"), expected.to_numpy(dtype="datetime64[ns]")):
            raise ValueError(f"Incomplete or nonunique target-year weather for {key}")
        weather[str(key)] = {
            "year": year,
            "dates": rows.date_bjt_dt.dt.date.map(lambda x: x.toordinal()).to_numpy(int),
            "doy": rows.date_bjt_dt.dt.dayofyear.to_numpy(int),
            "temp": rows.temperature_c.to_numpy(float),
        }
    local_sites = set(targets.loc[targets.target_source.fillna("").str.contains("phenology_xlsx"), "site_year"])
    if not local_sites:
        raise ValueError("No phenology.xlsx site-years found in target table")
    return targets, weather, local_sites


def linear_response(temp, base):
    return np.maximum(np.asarray(temp, dtype=float) - float(base), 0.0)


def beta_response(temp, cardinals=(0.0, 25.4, 35.0)):
    """Normalized Fila/Wang-Engel-like response."""
    temp = np.asarray(temp, dtype=float)
    tmin, topt, tmax = map(float, cardinals)
    if not (tmin < topt < tmax):
        return np.zeros_like(temp)
    out = np.zeros_like(temp)
    valid = (temp > tmin) & (temp < tmax)
    x = (temp[valid] - tmin) / (tmax - tmin)
    alpha = (topt - tmin) / (tmax - tmin)
    beta = (tmax - topt) / (tmax - tmin)
    raw = np.power(np.maximum(x, 1e-12), alpha) * np.power(np.maximum(1.0 - x, 1e-12), beta)
    peak = alpha**alpha * beta**beta
    out[valid] = raw / peak
    return out


def decode(vector, family, start_mode):
    pos = 0
    base = 0.0
    cardinals = (0.0, 25.4, 35.0)
    if family == "linear_free_base":
        base = float(vector[pos]); pos += 1
    elif family.startswith("linear_fixed_base_"):
        base = float(family.rsplit("_", 1)[1])
    elif family == "beta_free_cardinals":
        cardinals = tuple(float(v) for v in vector[pos:pos + 3]); pos += 3
    if start_mode == "free_start":
        start_doy = int(round(vector[pos])); pos += 1
    else:
        start_doy = int(start_mode.replace("start_", ""))
    thresholds = np.asarray(vector[pos:pos + 3], dtype=float)
    return {"base_c": base, "start_doy": start_doy, "thresholds": thresholds, "cardinals": cardinals}


def responses(temp, family, params):
    if family.startswith("linear"):
        return linear_response(temp, params["base_c"])
    return beta_response(temp, params["cardinals"])


def simulate_one(weather, params, family):
    response = responses(weather["temp"], family, params)
    start = max(1, min(365, int(params["start_doy"])))
    start_idx = int(np.searchsorted(weather["doy"], start, side="left"))
    dates = weather["dates"]
    out = {stage: np.nan for stage in STAGES}
    cursor = start_idx
    for stage, requirement in zip(STAGES, params["thresholds"]):
        if cursor >= len(response):
            break
        cumulative = np.cumsum(response[cursor:])
        hit = np.flatnonzero(cumulative >= float(requirement))
        if len(hit) == 0:
            break
        idx = cursor + int(hit[0])
        out[stage] = float(dates[idx])
        cursor = idx + 1
    # BBCH53 is an auxiliary interpolation target.  It is deliberately not a
    # fourth free threshold because only six site-years contain it directly.
    out[53] = np.nan
    if np.isfinite(out[9]) and np.isfinite(out[65]):
        idx9 = int(np.searchsorted(dates, int(out[9])))
        idx65 = int(np.searchsorted(dates, int(out[65])))
        if idx65 > idx9:
            total = np.cumsum(response[idx9 + 1:idx65 + 1])
            threshold = .15 * params["thresholds"][1]
            hit = np.flatnonzero(total >= threshold)
            if len(hit):
                out[53] = float(dates[idx9 + 1 + int(hit[0])])
    return out


def predictions(weather, sites, vector, family, start_mode):
    params = decode(vector, family, start_mode)
    return {site: simulate_one(weather[site], params, family) for site in sites}


def target_error(row, predicted):
    if not np.isfinite(predicted):
        return UNATTAINED_PENALTY
    kind = row.censor_type
    lo, hi = row.observed_start_ordinal, row.observed_end_ordinal
    if kind in {"exact", "exact_assumed"}:
        return abs(float(predicted) - float(lo))
    if kind == "interval_censored":
        if predicted < lo:
            return float(lo - predicted)
        if predicted > hi:
            return float(predicted - hi)
        return 0.0
    if kind == "right_censored":
        # A first observation at BBCH13/15/17 is right-censored for BBCH09,
        # but a prediction months earlier is biologically implausible.  The
        # target builder supplies a conservative 14-day lower bound used only
        # during calibration; the observed censoring fields remain unchanged.
        lower = getattr(row, "calibration_lower_ordinal", np.nan)
        if np.isfinite(lower) and predicted < lower:
            return float(lower - predicted)
        return max(float(predicted) - float(hi), 0.0)
    if kind == "left_censored":
        return max(float(lo) - float(predicted), 0.0)
    raise ValueError(f"Unknown censor type: {kind}")


def objective(vector, targets, weather, sites, family, start_mode):
    params = decode(vector, family, start_mode)
    if family == "beta_free_cardinals" and not (params["cardinals"][0] < params["cardinals"][1] < params["cardinals"][2]):
        return UNATTAINED_PENALTY
    pred = predictions(weather, sites, vector, family, start_mode)
    scores = []
    for row in targets.loc[targets.site_year.isin(sites)].itertuples():
        if row.target_stage not in STAGES:
            continue
        value = target_error(row, pred[row.site_year][row.target_stage])
        local = LOCAL_MULTIPLIER if "phenology_xlsx" in str(row.target_source) else 1.0
        scores.append(value * float(row.quality_weight) * STAGE_WEIGHT[row.target_stage] * local)
    if not scores:
        return UNATTAINED_PENALTY
    return float(np.mean(scores))


def bounds_for(family, start_mode):
    bounds = []
    if family == "linear_free_base":
        bounds.append((0.0, 10.0))
    if family == "beta_free_cardinals":
        bounds.extend([(-2.0, 10.0), (15.0, 32.0), (30.0, 45.0)])
    if start_mode == "free_start":
        bounds.append((1.0, 75.0))
    if family in {"beta_fixed_cardinals", "beta_free_cardinals"}:
        # The normalized beta response is at most one unit per day.
        bounds.extend([(5.0, 180.0), (5.0, 180.0), (5.0, 240.0)])
    else:
        bounds.extend([(100.0, 2200.0), (100.0, 1800.0), (100.0, 1800.0)])
    return bounds


def fit_model(targets, weather, sites, family, start_mode, args):
    best = None
    records = []
    bounds = bounds_for(family, start_mode)
    for seed in args.seeds:
        result = differential_evolution(
            lambda x: objective(x, targets, weather, sites, family, start_mode),
            bounds=bounds, seed=seed, maxiter=args.maxiter, popsize=args.popsize,
            tol=1e-5, polish=True, updating="immediate", workers=1,
        )
        records.append({"seed": seed, "objective": float(result.fun), "success": bool(result.success),
                        "iterations": int(result.nit), "evaluations": int(result.nfev)})
        if best is None or result.fun < best.fun:
            best = result
    return best, records


def assign_folds(sites, n_folds, seed):
    values = np.array(sorted(sites), dtype=object)
    rng = np.random.default_rng(seed); rng.shuffle(values)
    return {str(site): int(i % n_folds) for i, site in enumerate(values)}


def metric_rows(targets, pred, sites, split, model):
    rows = []
    for row in targets.loc[targets.site_year.isin(sites)].itertuples():
        if row.target_stage not in STAGES:
            continue
        p = pred[row.site_year][row.target_stage]
        err = target_error(row, p)
        exact_error = abs(float(p) - float(row.observed_start_ordinal)) if row.censor_type in {"exact", "exact_assumed"} and np.isfinite(p) else np.nan
        rows.append({"model": model, "split": split, "site_year": row.site_year,
                     "target_stage": row.target_stage, "censor_type": row.censor_type,
                     "observed_start": row.observed_start, "observed_end": row.observed_end,
                     "predicted_date": pd.Timestamp.fromordinal(int(p)).date().isoformat() if np.isfinite(p) else "",
                     "interval_error_days": err, "exact_abs_error_days": exact_error,
                     "target_source": row.target_source})
    return rows


def aggregate_metrics(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["within_7d"] = frame.interval_error_days.le(7)
    frame["within_10d"] = frame.interval_error_days.le(10)
    return (frame.groupby(["model", "split", "target_stage"], as_index=False)
            .agg(n=("target_stage", "size"), mae_days=("interval_error_days", "mean"),
                 median_error_days=("interval_error_days", "median"),
                 exact_n=("exact_abs_error_days", "count"), exact_mae_days=("exact_abs_error_days", "mean"),
                 within_7d=("within_7d", "mean"), within_10d=("within_10d", "mean")))


def make_model_name(family, start_mode):
    return f"{family}_{start_mode}"


def main(args):
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets, weather, local_sites = load_data(args.data_dir)
    all_sites = sorted(weather)
    specs = [
        ("linear_fixed_base", "start_1"),
        ("linear_fixed_base", "free_start"),
        ("linear_free_base", "free_start"),
        ("beta_fixed_cardinals", "start_1"),
        ("beta_fixed_cardinals", "free_start"),
    ]
    # Alias used by response dispatch; fixed-base linear uses base 0 C.
    models = []
    all_prediction_rows = []
    summary_rows = []
    fold_map = assign_folds(all_sites, args.folds, 20261008)
    for family, start_mode in specs:
        name = make_model_name(family, start_mode)
        train_sites = all_sites
        best, restarts = fit_model(targets, weather, train_sites, family, start_mode, args)
        full_pred = predictions(weather, all_sites, best.x, family, start_mode)
        all_prediction_rows.extend(metric_rows(targets, full_pred, all_sites, "apparent_full", name))
        params = decode(best.x, family, start_mode)
        model_dir = args.output_dir / name
        model_dir.mkdir(exist_ok=True)
        (model_dir / "final_parameters.json").write_text(json.dumps({
            "model": name, "family": family, "start_mode": start_mode,
            "parameters": {"base_c": params["base_c"], "start_doy": params["start_doy"],
                           "H09": float(params["thresholds"][0]), "H09_65": float(params["thresholds"][1]),
                           "H65_81": float(params["thresholds"][2])},
            "restarts": restarts,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Random grouped CV across all supported site-years.
        for fold in range(args.folds):
            test = sorted(s for s in all_sites if fold_map[s] == fold)
            train = sorted(s for s in all_sites if s not in test)
            fitted, _ = fit_model(targets, weather, train, family, start_mode, args)
            pred = predictions(weather, test, fitted.x, family, start_mode)
            all_prediction_rows.extend(metric_rows(targets, pred, test, f"all_5fold_{fold}", name))

        # Leave-one-local-site-year-out validation is the key selection test.
        for test_site in sorted(local_sites):
            train = sorted(s for s in all_sites if s != test_site)
            fitted, _ = fit_model(targets, weather, train, family, start_mode, args)
            pred = predictions(weather, [test_site], fitted.x, family, start_mode)
            all_prediction_rows.extend(metric_rows(targets, pred, [test_site], "local_loso", name))

    pred_frame = pd.DataFrame(all_prediction_rows)
    pred_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics = aggregate_metrics(all_prediction_rows)
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)

    # Model choice emphasizes local leave-one-out BBCH09/65 while retaining
    # the complete grouped-CV result as a transferability check.
    select = []
    for name in sorted(pred_frame.model.unique()):
        local = pred_frame.loc[(pred_frame.model == name) & (pred_frame.split == "local_loso") & pred_frame.target_stage.isin([9, 65])]
        global_cv = pred_frame.loc[(pred_frame.model == name) & pred_frame.split.str.startswith("all_5fold") & pred_frame.target_stage.isin([9, 65])]
        exact_local = local.loc[local.censor_type == "exact", "exact_abs_error_days"]
        select.append({"model": name, "local_primary_interval_mae": local.interval_error_days.mean(),
                      "local_primary_exact_mae": exact_local.mean() if len(exact_local) else np.nan,
                      "global_primary_interval_mae": global_cv.interval_error_days.mean(),
                      "selection_score": .7 * local.interval_error_days.mean() + .3 * global_cv.interval_error_days.mean()})
    comparison = pd.DataFrame(select).sort_values("selection_score")
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    selected = str(comparison.iloc[0].model)
    (args.output_dir / "selected_model.txt").write_text(selected + "\n", encoding="utf-8")

    report = ["Phenology multi-model calibration", "", comparison.to_string(index=False), "", f"Selected model: {selected}", ""]
    report.append(metrics.loc[metrics.model.eq(selected)].to_string(index=False))
    (args.output_dir / "calibration_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main(parse_args())
