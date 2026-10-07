"""Shared calibration helpers for the fixed mixed regional phenology workflow."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date, timedelta
import hashlib
import multiprocessing as mp
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm

from .model import PhenologyConfig, StageTarget, _chill_response, _force_response
from .model import daily_weather_from_csv

TARGETS = {9: "budburst", 65: "flowering", 81: "veraison"}
LATITUDE_CUTOFF = 32.0
EARLY_BASE_C = 0.0
DEFAULT_EXCLUDED_SITE_YEARS = (
    "22.610000_108.230000_2023", "22.610000_108.230000_2024",
    "23.830000_108.100000_2022", "23.830000_108.100000_2023",
    "23.830000_108.100000_2024")
CHILL_MIN, CHILL_OPT, CHILL_MAX = -1.7, 1.7, 13.2
PENALTY = 1000.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_targets(path, *, region="all", exclude_site_years=()):
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"record_id": str, "site_year": str})
    required = {"record_id", "site_year", "year", "bbch", "date_start", "date_end",
                "variety", "observation_type", "source", "fruit_crop", "latitude"}
    if required - set(frame):
        raise ValueError(f"Missing observation columns: {sorted(required-set(frame))}")
    if frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise ValueError("record_id must be nonmissing and unique")
    frame["bbch"] = pd.to_numeric(frame.bbch, errors="raise").astype(int)
    frame["latitude"] = pd.to_numeric(frame.latitude, errors="raise")
    if region == "north": in_region = frame.latitude >= LATITUDE_CUTOFF
    elif region == "south": in_region = frame.latitude < LATITUDE_CUTOFF
    else: in_region = pd.Series(True, index=frame.index)
    excluded = frame.site_year.isin(set(exclude_site_years))
    event = frame.observation_type.eq("单年观测") & ~frame.source.fillna("").str.lower().isin(["expoeriment", "experiment"])
    first_crop = ~frame.fruit_crop.fillna("").str.contains("二次|二季|双季", regex=True)
    included = frame.bbch.isin(TARGETS) & in_region & ~excluded & event & first_crop
    audit = frame[["record_id", "site_year", "bbch", "observation_type"]].copy()
    audit["use"] = np.where(included, "calibration", "excluded")
    audit["reason"] = np.select([excluded, ~in_region, ~first_crop, ~event, ~frame.bbch.isin(TARGETS)],
        ["excluded_site_year", "outside_region", "secondary_crop", "not_single_year_event", "not_anchor_bbch"], default="exact_anchor_event")
    selected = frame.loc[included].copy()
    if set(selected.bbch) != set(TARGETS):
        raise ValueError("Each of BBCH09,65,81 needs at least one target")
    for col in ("date_start", "date_end"):
        d = pd.to_datetime(selected[col], errors="raise")
        selected[col] = d.map(lambda x: x.date().toordinal())
    selected["year"] = pd.to_numeric(selected.year, errors="raise").astype(int)
    selected["variety"] = selected.variety.fillna("").astype(str)
    keys = ["site_year", "year", "bbch"]
    varieties = selected.groupby(keys + ["variety"], as_index=False).agg(lower=("date_start", "median"), upper=("date_end", "median"))
    targets = varieties.groupby(keys, as_index=False).agg(lower=("lower", "median"), upper=("upper", "median"), n_varieties=("variety", "size"))
    targets = targets.merge(selected.groupby(keys, as_index=False).agg(n_records=("record_id", "size")), on=keys, validate="one_to_one")
    return targets.sort_values(keys).reset_index(drop=True), audit


def build_force_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhenologyConfig(None, tuple(StageTarget(n, b, 1.0) for b, n in TARGETS.items()),
        model_type="force_only", early_base_c=EARLY_BASE_C,
        chill_start_month=int(raw.get("chill_start_month", 9)), chill_start_day=int(raw.get("chill_start_day", 1)),
        chill_min_c=float(raw.get("chill_min_c", CHILL_MIN)), chill_opt_c=float(raw.get("chill_opt_c", CHILL_OPT)),
        chill_max_c=float(raw.get("chill_max_c", CHILL_MAX)), forcing_base_c=0.0,
        forcing_start_month=1, forcing_start_day=1, simulation_end_month=12, simulation_end_day=31)


def prepare_weather(path, targets, config):
    index = pd.read_csv(path, encoding="utf-8-sig", dtype={"site_year": str})
    seasons = {}
    for row in index[index.site_year.isin(targets.site_year.unique())].itertuples(index=False):
        file = Path(row.weather_file); file = file if file.is_absolute() else Path(__file__).resolve().parents[2] / file
        year = int(row.year); daily = daily_weather_from_csv(file, site_year=str(row.site_year))
        start, end = date(year, 1, 1), date(year, 12, 31)
        daily = [r for r in daily if start.isoformat() <= r["date"] <= end.isoformat()]
        expected = [start + timedelta(days=i) for i in range((end-start).days+1)]
        if [date.fromisoformat(r["date"]) for r in daily] != expected: raise ValueError(f"Incomplete weather: {row.site_year}")
        seasons[str(row.site_year)] = {"year": year, "daily": daily, "path": str(file), "dates": np.array([d.toordinal() for d in expected]),
            "force": np.r_[0., np.cumsum([_force_response(r["temperature_c"], config) for r in daily])],
            "early_force": np.r_[0., np.cumsum([_force_response(r["temperature_c"], config, 9) for r in daily])]}
    return seasons


def predict_force(seasons, vector):
    out = {}
    for key, w in seasons.items():
        values = np.full(3, np.nan); start = 0
        for i, req in enumerate(vector):
            prefix = w.get("early_force", w["force"]) if i < 2 else w["force"]
            k = int(np.searchsorted(prefix, prefix[start] + req, side="left"))
            if k > len(w["dates"]): break
            values[i] = w["dates"][k-1]; start = k
        out[key] = values
    return out


# Public names used by the retained mixed workflow and lightweight diagnostics.
predict = predict_force


def loss(targets, seasons, vector):
    return objective(targets, seasons, vector, "force")


def configured(template, vector):
    """Return a force-only three-anchor config from a template mapping."""
    raw = json.loads(json.dumps(template))
    raw["model"] = "force_only"
    raw["chill_requirement"] = None
    raw["early_base_c"] = EARLY_BASE_C
    raw["stage_targets"] = [{"name": name, "bbch": bbch, "requirement": float(v)}
                            for v, (bbch, name) in zip(vector, TARGETS.items())]
    return raw


def errors(targets, predictions):
    positions = {b: i for i, b in enumerate(TARGETS)}
    values = np.array([predictions[r.site_year][positions[r.bbch]] for r in targets.itertuples()])
    lower, upper = targets.lower.to_numpy(), targets.upper.to_numpy()
    interval = np.where(values < lower, values-lower, np.where(values > upper, values-upper, 0.0)); interval[~np.isfinite(values)] = np.nan
    return values, interval, values-(lower+upper)/2


_WORKER = None
def _init_worker(targets, seasons, kind):
    global _WORKER; _WORKER = (targets, seasons, kind)

def _objective(v):
    targets, seasons, kind = _WORKER
    return objective(targets, seasons, v, kind)

def objective(targets, seasons, vector, kind):
    pred = predict_force(seasons, vector) if kind == "force" else {k: predict_cf_one(seasons[k], vector) for k in seasons}
    values, interval, _ = errors(targets, pred); score = np.where(np.isfinite(values), np.abs(interval), PENALTY)
    return float(np.mean([score[targets.bbch.eq(b)].mean() for b in TARGETS]))


def fit_thresholds(targets, seasons, bounds, *, seeds, maxiter, popsize, workers, kind="force"):
    global _WORKER; _WORKER = (targets, seasons, kind)
    ctx = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(targets, seasons, kind)) if workers > 1 else nullcontext(None)
    best, records = None, []
    with ctx as pool:
        func = _objective if pool else lambda v: objective(targets, seasons, v, kind)
        for seed in seeds:
            with tqdm(total=maxiter, desc=f"Fit {kind} seed={seed}", unit="gen", dynamic_ncols=True) as bar:
                def cb(x, convergence): bar.update(1); bar.set_postfix(loss=f"{objective(targets,seasons,x,kind):.3f}")
                result = differential_evolution(func, bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=.001, polish=False,
                    workers=pool.map if pool else 1, updating="deferred" if pool else "immediate", callback=cb)
            records.append({"seed": seed, "objective": float(result.fun), "converged": bool(result.success), "iterations": int(result.nit), "evaluations": int(result.nfev), "parameters": dict(zip(("chill_requirement","H09","H09_65","H65_81") if kind=="cf" else ("H09","H09_65","H65_81"), result.x.tolist()))})
            if best is None or result.fun < best.fun: best = result
    return best, records


def load_cf_seasons(path, targets):
    index = pd.read_csv(path, encoding="utf-8-sig", dtype={"site_year": str}); seasons = {}
    for row in index[index.site_year.isin(targets.site_year.unique())].itertuples(index=False):
        file = Path(row.weather_file); file = file if file.is_absolute() else Path(__file__).resolve().parents[2] / file
        year = int(row.year); daily_all = daily_weather_from_csv(file, site_year=str(row.site_year)); start, end = date(year-1,9,1), date(year,12,31)
        lookup = {date.fromisoformat(r["date"]): r for r in daily_all}; expected=[start+timedelta(days=i) for i in range((end-start).days+1)]
        if any(d not in lookup for d in expected): raise ValueError(f"Incomplete Chill+Force weather: {row.site_year}")
        seasons[str(row.site_year)]={"year":year,"daily":[lookup[d] for d in expected],"path":str(file)}
    return seasons


def _cf_config():
    return PhenologyConfig(1., (StageTarget("budburst",9,1.),), model_type="sequential_chill_force", early_base_c=EARLY_BASE_C,
        chill_start_month=9,chill_start_day=1,chill_min_c=CHILL_MIN,chill_opt_c=CHILL_OPT,chill_max_c=CHILL_MAX,forcing_base_c=0.,forcing_start_month=1,forcing_start_day=1,simulation_end_month=12,simulation_end_day=31)


def predict_cf_one(weather, vector):
    req, h09, h0965, h6581 = map(float, vector); cfg=_cf_config(); total=0.; completion=None
    for i,row in enumerate(weather["daily"]):
        total += _chill_response(float(row["temperature_c"]), cfg)
        if completion is None and total >= req: completion=i
    if completion is None: return np.full(3,np.nan)
    dates=np.array([date.fromisoformat(r["date"]).toordinal() for r in weather["daily"]]); out=np.full(3,np.nan); start=completion+1
    for stage, requirement in enumerate((h09,h0965,h6581)):
        total=0.; bbch=9 if stage<2 else 81
        for i in range(start,len(dates)):
            total += _force_response(float(weather["daily"][i]["temperature_c"]), cfg, bbch)
            if total >= requirement: out[stage]=dates[i]; start=i+1; break
        else: break
    return out


def fit_cf(targets, seasons, bounds, *, seeds, maxiter, popsize, workers):
    return fit_thresholds(targets, seasons, bounds, seeds=seeds, maxiter=maxiter, popsize=popsize, workers=workers, kind="cf")
