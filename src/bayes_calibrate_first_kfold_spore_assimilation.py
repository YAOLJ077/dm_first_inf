# -*- coding: utf-8 -*-
"""
Bayes + TPE + k-fold (single-stage) with strong spore assimilation.

Key idea:
  - Use all observed spore windows (between consecutive observed dates).
  - Simulated spore window = [dispersal_datetime, min(deposition_datetime, sporangia_alive_end)].
  - Likelihood uses count-based (NB or Poisson) with optional zero-inflation.
  - Objective: train_loss = train_mae + lambda * train_spore_nll.
"""

import os
import platform
import sys
import json
import copy
import argparse
import math
from typing import Dict, List, Tuple
from collections import OrderedDict

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import time
from joblib import Parallel, delayed

# ======================================================
# PATHS
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_DATA_PATH = os.path.join(DATA_DIR, "Spore_Overall.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season import CropSeasonFirst
from model.config import config

# ======================================================
# SETTINGS
# ======================================================
PENALTY_DAYS = 120.0
RANDOM_SEED = 42
FOLD_SEED = 42
N_JOBS = 32
RB_CACHE_MAX = 128
TOPK_PCT = 5.0
JOBLIB_BATCH = 8  # trials per batch submission

# Spore assimilation hyperparams (defaults)
SPORE_LAMBDA = 0.3
SPORE_K_SCALE = 0.6        # scale from simulated intensity to expected count
SPORE_NB_DISP = 2.5        # NB dispersion (r); higher = closer to Poisson
SPORE_ZERO_INFL = 0.15     # zero-inflation probability
SPORE_RAIN_WEIGHT = 0.2    # down-weight rain-splash events vs wind
USE_NB = True              # if False -> Poisson
EARLY_STOP_MAE = 0.0       # set >0 to early-stop bad trials (e.g. 30.0)
NO_OBS_NLL = PENALTY_DAYS * 0.5  # penalty when no usable obs windows

# Hyper-parameter ranges (for TPE)
HYPER_RANGES = {
    "spore_lambda": (0.1, 1.0),
    "spore_k_scale": (0.2, 1.5),
    "spore_zero_infl": (0.05, 0.4),
}

# Base ranges (global)
BASE_RANGES = {
    "mp_tmp__rh_wet_night": (85, 95),
    "sporulation__MP_min": (3, 6),
    "sporulation__Tmin": (8, 15),
    "sporulation__Topt": (18, 24),
    "sporulation__Tmax": (25, 35),
    "dispersal__rain_min": (6.0, 12.0),
    "dispersal__wind_min": (4.0, 12.0),
    "deposition__rain_min": (4.0, 12.0),
    "zoospore_alive__rain": (3.0, 6.0),
    "zoospore_alive__vpd": (1.5, 3.5),
    "infection__infection_condition": (50, 200),
    "incubation__spl_end": (1.0, 1.5),
    "thresholds__min_spores": (0.02, 0.06)
}


def expand_range_1p5(lo: float, hi: float) -> Tuple[float, float]:
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    new_half = 1.5 * half
    return center - new_half, center + new_half


def build_param_ranges() -> Dict[str, Tuple[float, float]]:
    return {k: expand_range_1p5(v[0], v[1]) for k, v in BASE_RANGES.items()}


def load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)


def apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in values.items():
        if "__" not in k:
            continue
        group, sub = k.split("__", 1)
        cfg[group][sub] = float(v)
    return cfg


_RB_CACHE: Dict[str, dict] = {}


def load_rb_cached(path: str) -> dict:
    if path not in _RB_CACHE:
        with open(path, "r") as f:
            _RB_CACHE[path] = json.load(f)
    return _RB_CACHE[path]


def _load_rb_cached(path: str, cache: Dict[str, dict]) -> dict:
    if path not in cache:
        with open(path, "r") as f:
            cache[path] = json.load(f)
    return cache[path]


def _calc_error(actual_date, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return np.nan
    first = event_df.iloc[0]
    doy_actual = pd.to_datetime(actual_date).dayofyear
    doy_start = pd.to_datetime(first["oil_spot_start"]).dayofyear
    doy_end = pd.to_datetime(first["oil_spot_end"]).dayofyear
    if doy_end < doy_actual:
        return float(doy_end - doy_actual)
    if doy_start > doy_actual:
        return float(doy_start - doy_actual)
    return 0.0


def compute_error_stats(cfg: dict, rows: List[pd.Series]) -> Tuple[float, float, float]:
    errors = []
    for row in rows:
        key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue
        try:
            rb = load_rb_cached(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            err = _calc_error(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            err = np.nan
        if np.isfinite(err):
            errors.append(float(err))
        else:
            errors.append(float(PENALTY_DAYS))
    if not errors:
        return PENALTY_DAYS, 0.0, PENALTY_DAYS
    abs_err = np.abs(np.array(errors, dtype=float))
    mae = float(abs_err.mean())
    std = float(abs_err.std(ddof=0))
    rmse = float(np.sqrt(np.mean(np.square(abs_err))))
    return mae, std, rmse


def _normalize_spore_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    required = {"plot", "year", "date", "spore"}
    missing = required - set(cols.keys())
    if missing:
        raise ValueError(f"Spore data missing columns: {missing}")
    out = df.rename(columns={
        cols["plot"]: "plot",
        cols["year"]: "year",
        cols["date"]: "date",
        cols["spore"]: "spore",
    })
    out["date"] = pd.to_datetime(out["date"])
    out["spore"] = pd.to_numeric(out["spore"], errors="coerce")
    return out[["plot", "year", "date", "spore"]]


def load_spore_obs_windows() -> Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]]:
    df = pd.read_excel(SPORE_DATA_PATH)
    df = _normalize_spore_columns(df)
    windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]] = {}
    for (plot, year), g in df.groupby(["plot", "year"]):
        g = g.sort_values("date")
        g = g[g["spore"].notna()]
        prev = None
        for _, row in g.iterrows():
            cur = row["date"].normalize()
            y = float(row["spore"])
            if prev is not None:
                windows.setdefault((str(plot), int(year)), []).append((prev, cur, y))
            prev = cur
    return windows


def _nb_logpmf(y: int, mu: float, r: float) -> float:
    mu = max(mu, 1e-8)
    r = max(r, 1e-6)
    p = r / (r + mu)
    return (
        math.lgamma(y + r)
        - math.lgamma(r)
        - math.lgamma(y + 1)
        + r * math.log(p)
        + y * math.log(1 - p)
    )


def _pois_logpmf(y: int, mu: float) -> float:
    mu = max(mu, 1e-8)
    return y * math.log(mu) - mu - math.lgamma(y + 1)


def _zi_logpmf(y: int, logpmf: float, zi: float, logpmf_zero: float) -> float:
    zi = float(np.clip(zi, 1e-6, 1 - 1e-6))
    if y == 0:
        a = math.log(zi)
        b = math.log(1 - zi) + logpmf_zero
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))
    return math.log(1 - zi) + logpmf


def _pick_bbch53_date(row: pd.Series, rb: dict) -> pd.Timestamp:
    cand = getattr(row, "bbch_date", pd.NaT)
    if pd.notna(cand):
        return pd.to_datetime(cand)
    rb_dt = rb.get("HHCH_53")
    if rb_dt:
        return pd.to_datetime(rb_dt, errors="coerce")
    return pd.NaT


def _calc_assim_start(obs_windows, start_candidate: pd.Timestamp) -> pd.Timestamp:
    if not obs_windows:
        return start_candidate
    if pd.isna(start_candidate):
        return obs_windows[0][0]
    for (s, e, _) in obs_windows:
        if e >= start_candidate:
            return max(s, start_candidate)
    return obs_windows[-1][0]


def compute_spore_assim_nll(
    cfg: dict,
    rows: List[pd.Series],
    obs_windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]],
    k_scale: float,
    nb_disp: float,
    zi: float,
    use_nb: bool,
    rain_weight: float,
    end_mode: str,
    no_obs_nll: float,
    load_rb_fn
) -> float:
    nlls: List[float] = []
    for row in rows:
        key = (str(row.plot), int(row.first_infection_date.year))
        win_obs = obs_windows.get(key, [])
        if not win_obs:
            nlls.append(float(no_obs_nll))
            continue
        rb_key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(DATA_DIR, f"{rb_key}_grape_rb.json")
        if not os.path.exists(rb_path):
            nlls.append(float(no_obs_nll))
            continue
        try:
            rb = load_rb_fn(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            cohorts = cs.detached_survival
            if cohorts is None or cohorts.empty:
                nlls.append(float(no_obs_nll))
                continue
        except Exception:
            nlls.append(float(no_obs_nll))
            continue

        bbch53 = _pick_bbch53_date(row, rb)
        start_candidate = bbch53 - pd.Timedelta(days=14) if pd.notna(bbch53) else pd.NaT
        start_date = _calc_assim_start(win_obs, start_candidate)

        if end_mode == "sim":
            if cs.primary_chain_unique is not None and not cs.primary_chain_unique.empty:
                end_date = pd.to_datetime(cs.primary_chain_unique.iloc[0]["oil_spot_start"])
            else:
                end_date = pd.NaT
        else:
            end_date = pd.to_datetime(row.first_infection_date)

        if pd.isna(end_date):
            end_date = win_obs[-1][1]

        sim_windows = []
        for _, c in cohorts.iterrows():
            s = pd.to_datetime(c.get("dispersal_datetime")).normalize()
            dep = c.get("deposition_datetime")
            alive_end = c.get("sporangia_alive_end")
            end = None
            if pd.notna(dep):
                end = pd.to_datetime(dep).normalize()
            if pd.notna(alive_end):
                ae = pd.to_datetime(alive_end).normalize()
                end = ae if end is None else min(end, ae)
            if end is None:
                continue
            if end < s:
                continue
            dtype = c.get("dispersal_type", "wind")
            w = rain_weight if dtype == "rain" else 1.0
            sim_windows.append((s, end, float(c.get("airborne_amount", 0.0)) * w))

        if not sim_windows:
            nlls.append(float(no_obs_nll))
            continue

        total_nll = 0.0
        used = 0
        for (s_obs, e_obs, y) in win_obs:
            if e_obs < start_date or s_obs > end_date:
                continue
            s_obs = max(s_obs, start_date)
            e_obs = min(e_obs, end_date)
            if e_obs < s_obs:
                continue
            used += 1
            y_int = int(round(max(y, 0.0)))
            intensity = 0.0
            window_days = (e_obs - s_obs).days + 1
            if window_days <= 0:
                continue
            for (s_sim, e_sim, amt) in sim_windows:
                start = max(s_obs, s_sim)
                end = min(e_obs, e_sim)
                if end < start:
                    continue
                overlap_days = (end - start).days + 1
                overlap_ratio = float(overlap_days) / float(window_days)
                intensity += amt * overlap_ratio

            mu = max(0.0, k_scale * intensity)
            if use_nb:
                logpmf = _nb_logpmf(y_int, mu, nb_disp)
                logpmf_zero = _nb_logpmf(0, mu, nb_disp)
            else:
                logpmf = _pois_logpmf(y_int, mu)
                logpmf_zero = _pois_logpmf(0, mu)

            logp = _zi_logpmf(y_int, logpmf, zi, logpmf_zero)
            total_nll += -logp

        if used == 0:
            nlls.append(float(no_obs_nll))
        else:
            nlls.append(total_nll / float(used))
    return float(np.mean(nlls)) if nlls else 0.0


def _spore_nll_for_row(
    row: pd.Series,
    rb: dict,
    cs: CropSeasonFirst,
    win_obs: List[Tuple[pd.Timestamp, pd.Timestamp, float]],
    k_scale: float,
    nb_disp: float,
    zi: float,
    use_nb: bool,
    rain_weight: float,
    end_mode: str,
    no_obs_nll: float
) -> float:
    if not win_obs:
        return float(no_obs_nll)

    cohorts = cs.detached_survival
    if cohorts is None or cohorts.empty:
        return float(no_obs_nll)

    bbch53 = _pick_bbch53_date(row, rb)
    start_candidate = bbch53 - pd.Timedelta(days=14) if pd.notna(bbch53) else pd.NaT
    start_date = _calc_assim_start(win_obs, start_candidate)

    if end_mode == "sim":
        if cs.primary_chain_unique is not None and not cs.primary_chain_unique.empty:
            end_date = pd.to_datetime(cs.primary_chain_unique.iloc[0]["oil_spot_start"])
        else:
            end_date = pd.NaT
    else:
        end_date = pd.to_datetime(row.first_infection_date)

    if pd.isna(end_date):
        end_date = win_obs[-1][1]

    sim_windows = []
    for _, c in cohorts.iterrows():
        s = pd.to_datetime(c.get("dispersal_datetime")).normalize()
        dep = c.get("deposition_datetime")
        alive_end = c.get("sporangia_alive_end")
        end = None
        if pd.notna(dep):
            end = pd.to_datetime(dep).normalize()
        if pd.notna(alive_end):
            ae = pd.to_datetime(alive_end).normalize()
            end = ae if end is None else min(end, ae)
        if end is None:
            continue
        if end < s:
            continue
        dtype = c.get("dispersal_type", "wind")
        w = rain_weight if dtype == "rain" else 1.0
        sim_windows.append((s, end, float(c.get("airborne_amount", 0.0)) * w))

    if not sim_windows:
        return float(no_obs_nll)

    total_nll = 0.0
    used = 0
    for (s_obs, e_obs, y) in win_obs:
        if e_obs < start_date or s_obs > end_date:
            continue
        s_obs = max(s_obs, start_date)
        e_obs = min(e_obs, end_date)
        if e_obs < s_obs:
            continue
        used += 1
        y_int = int(round(max(y, 0.0)))
        intensity = 0.0
        window_days = (e_obs - s_obs).days + 1
        if window_days <= 0:
            continue
        for (s_sim, e_sim, amt) in sim_windows:
            start = max(s_obs, s_sim)
            end = min(e_obs, e_sim)
            if end < start:
                continue
            overlap_days = (end - start).days + 1
            overlap_ratio = float(overlap_days) / float(window_days)
            intensity += amt * overlap_ratio

        mu = max(0.0, k_scale * intensity)
        if use_nb:
            logpmf = _nb_logpmf(y_int, mu, nb_disp)
            logpmf_zero = _nb_logpmf(0, mu, nb_disp)
        else:
            logpmf = _pois_logpmf(y_int, mu)
            logpmf_zero = _pois_logpmf(0, mu)

        logp = _zi_logpmf(y_int, logpmf, zi, logpmf_zero)
        total_nll += -logp

    if used == 0:
        return float(no_obs_nll)
    return float(total_nll / float(used))


# ======================================================
# Joblib backend (lazy load per worker)
# ======================================================
_JB_ROWS = None
_JB_SPORE_WINDOWS = None
_JB_RB_CACHE = OrderedDict()
_JB_RB_CACHE_MAX = 64


def _jb_ensure_loaded(data_dir: str, rb_cache_max: int) -> None:
    global _JB_ROWS, _JB_SPORE_WINDOWS, _JB_RB_CACHE, _JB_RB_CACHE_MAX
    if _JB_ROWS is not None and _JB_SPORE_WINDOWS is not None:
        return
    _JB_RB_CACHE = OrderedDict()
    _JB_RB_CACHE_MAX = rb_cache_max
    actual_path = os.path.join(data_dir, "actual_data.xlsx")
    spore_path = os.path.join(data_dir, "Spore_Overall.xlsx")
    df = pd.read_excel(actual_path)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    _JB_ROWS = list(df.itertuples(index=False))
    df_spore = pd.read_excel(spore_path)
    df_spore = _normalize_spore_columns(df_spore)
    windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]] = {}
    for (plot, year), g in df_spore.groupby(["plot", "year"]):
        g = g.sort_values("date")
        g = g[g["spore"].notna()]
        prev = None
        for _, row in g.iterrows():
            cur = row["date"].normalize()
            y = float(row["spore"])
            if prev is not None:
                windows.setdefault((str(plot), int(year)), []).append((prev, cur, y))
            prev = cur
    _JB_SPORE_WINDOWS = windows


def _jb_load_rb_cached_lru(path: str) -> dict:
    global _JB_RB_CACHE, _JB_RB_CACHE_MAX
    if path in _JB_RB_CACHE:
        _JB_RB_CACHE.move_to_end(path)
        return _JB_RB_CACHE[path]
    with open(path, "r") as f:
        data = json.load(f)
    _JB_RB_CACHE[path] = data
    _JB_RB_CACHE.move_to_end(path)
    if _JB_RB_CACHE_MAX > 0 and len(_JB_RB_CACHE) > _JB_RB_CACHE_MAX:
        _JB_RB_CACHE.popitem(last=False)
    return data


def _jb_compute_stats_idx(
    cfg: dict,
    idxs: List[int],
    k_scale: float,
    nb_disp: float,
    zi: float,
    use_nb: bool,
    rain_weight: float,
    end_mode: str,
    no_obs_nll: float,
    data_dir: str,
    rb_cache_max: int
) -> Tuple[float, float, float, float]:
    _jb_ensure_loaded(data_dir, rb_cache_max)
    errors = []
    nlls = []
    for i in idxs:
        row = _JB_ROWS[i]
        key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(data_dir, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            errors.append(float(PENALTY_DAYS))
            nlls.append(float(no_obs_nll))
            continue
        try:
            rb = _jb_load_rb_cached_lru(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            err = _calc_error(row.first_infection_date, cs.primary_chain_unique)
            if not np.isfinite(err):
                err = float(PENALTY_DAYS)
            nll = _spore_nll_for_row(
                row, rb, cs,
                _JB_SPORE_WINDOWS.get((str(row.plot), int(row.first_infection_date.year)), []),
                k_scale, nb_disp, zi, use_nb, rain_weight, end_mode, no_obs_nll
            )
        except Exception:
            err = float(PENALTY_DAYS)
            nll = float(no_obs_nll)

        errors.append(float(err))
        nlls.append(float(nll))

    if not errors:
        return PENALTY_DAYS, 0.0, PENALTY_DAYS, float(no_obs_nll)
    abs_err = np.abs(np.array(errors, dtype=float))
    mae = float(abs_err.mean())
    std = float(abs_err.std(ddof=0))
    rmse = float(np.sqrt(np.mean(np.square(abs_err))))
    nll = float(np.mean(nlls)) if nlls else float(no_obs_nll)
    return mae, std, rmse, nll


def eval_mae_joblib(
    params: Dict[str, float],
    hypers: Dict[str, float],
    train_idx: List[int],
    val_idx: List[int],
    nb_disp: float,
    use_nb: bool,
    rain_weight: float,
    end_mode: str,
    early_stop_mae: float,
    early_stop_nll: float,
    no_obs_nll: float,
    data_dir: str,
    rb_cache_max: int
) -> Tuple[float, float, float, float, float, float, float, float]:
    cfg = apply_params(config, params)
    k_scale = float(hypers.get("spore_k_scale", SPORE_K_SCALE))
    zi = float(hypers.get("spore_zero_infl", SPORE_ZERO_INFL))
    train_mae, train_std, train_rmse, train_nll = _jb_compute_stats_idx(
        cfg, train_idx, k_scale, nb_disp, zi, use_nb, rain_weight, end_mode, no_obs_nll, data_dir, rb_cache_max
    )
    if early_stop_mae > 0 and train_mae >= early_stop_mae:
        val_mae, val_std, val_rmse, _val_nll = _jb_compute_stats_idx(
            cfg, val_idx, k_scale, nb_disp, zi, use_nb, rain_weight, end_mode, no_obs_nll, data_dir, rb_cache_max
        )
        train_nll = early_stop_nll
        val_nll = early_stop_nll
        return train_mae, val_mae, train_std, val_std, train_rmse, val_rmse, train_nll, val_nll
    val_mae, val_std, val_rmse, val_nll = _jb_compute_stats_idx(
        cfg, val_idx, k_scale, nb_disp, zi, use_nb, rain_weight, end_mode, no_obs_nll, data_dir, rb_cache_max
    )
    return train_mae, val_mae, train_std, val_std, train_rmse, val_rmse, train_nll, val_nll


def build_folds(n: int, k: int, seed: int) -> List[List[int]]:
    if k <= 1 or k > n:
        k = n
    folds = [[] for _ in range(k)]
    rng = np.random.RandomState(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    for i, idx in enumerate(idxs):
        folds[i % k].append(idx)
    return folds


def _progress_print(
    done: int,
    total: int,
    running: int,
    running_max: int,
    prefix: str,
    last_ts: float,
    interval_s: float
) -> float:
    now = time.time()
    if now - last_ts >= interval_s:
        msg = f"Progress: {done}/{total} completed | running: {running}/{running_max}"
        if prefix:
            msg = f"{prefix} {msg}"
        print(msg, flush=True)
        return now
    return last_ts


def _stage_start(prefix: str, total: int, running: int, running_max: int) -> None:
    msg = f"{prefix} Start | total trials: {total} | running: {running}/{running_max}"
    print(msg, flush=True)


def define_search_space(
    trial: optuna.Trial,
    ranges: Dict[str, Tuple[float, float]],
    hyper_ranges: Dict[str, Tuple[float, float]]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    params = {}
    for param_name, (lo, hi) in ranges.items():
        params[param_name] = trial.suggest_float(param_name, lo, hi)
    hypers = {}
    for h_name, (lo, hi) in hyper_ranges.items():
        hypers[h_name] = trial.suggest_float(h_name, lo, hi)
    return params, hypers


def _topk_by_train(df: pd.DataFrame, pct: float, loss_col: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty trials dataframe.")
    df_sorted = df.sort_values(loss_col).reset_index(drop=False)
    n_top = max(1, int(math.ceil(len(df_sorted) * pct / 100.0)))
    return df_sorted.iloc[:n_top]


def select_topk_val_best_constrained(df: pd.DataFrame, pct: float) -> pd.Series:
    if df.empty:
        raise ValueError("Empty trials dataframe.")
    if "train_loss" not in df.columns:
        raise KeyError("Missing train_loss in trials dataframe.")
    df_sorted = df.sort_values("train_loss").reset_index(drop=True)
    n_top = max(1, int(math.ceil(len(df_sorted) * pct / 100.0)))
    top = df_sorted.iloc[:n_top].copy()
    valid = top[top["val_mae"] > top["train_mae"]]
    if not valid.empty:
        best = valid.sort_values("val_mae").iloc[0]
    else:
        equal = top[top["val_mae"] == top["train_mae"]]
        if not equal.empty:
            best = equal.sort_values("val_mae").iloc[0]
        else:
            diff = (top["val_mae"] - top["train_mae"]).abs()
            best = top.iloc[diff.values.argmin()]
    if "train_std" in top.columns:
        same = top[top["val_mae"] == best["val_mae"]]
        if len(same) > 1:
            best = same.sort_values("train_std").iloc[0]
    return best


def run_baseline_kfold_joblib(
    rows: List[pd.Series],
    k: int,
    stage1_ranges: Dict[str, Tuple[float, float]],
    hyper_ranges: Dict[str, Tuple[float, float]],
    spore_windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]],
    n_jobs: int,
    restart_every: int,
    n_trials_stage1: int,
    n_startup_stage1: int,
    progress_interval: float,
    seed: int,
    seed_tag: str = "",
    nb_disp: float = SPORE_NB_DISP,
    use_nb: bool = USE_NB,
    rain_weight: float = SPORE_RAIN_WEIGHT,
    end_mode: str = "actual",
    early_stop_mae: float = 0.0,
    early_stop_nll: float = PENALTY_DAYS,
    no_obs_nll: float = NO_OBS_NLL,
) -> Tuple[List[Dict[str, float]], pd.DataFrame, Dict[str, float]]:
    folds = build_folds(len(rows), k, FOLD_SEED)
    fold_best_rows = []
    fold_params = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = [i for i in range(len(rows)) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        sampler = TPESampler(seed=seed + fold_id, multivariate=True, n_startup_trials=n_startup_stage1)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        trial_rows = []
        last_ts = 0.0
        completed = 0
        max_conc = max(1, n_jobs)
        prefix = f"{seed_tag}[Assimilation][Fold {fold_id}/{k}]"
        _stage_start(prefix, n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage1, 0, max_conc, prefix, last_ts, progress_interval)
        filled_logged = False
        joblib_batch = min(JOBLIB_BATCH, max_conc)

        batch_id = 0
        while completed < n_trials_stage1:
            batch_id += 1
            batch = min(joblib_batch, n_trials_stage1 - completed)
            trials = []
            params_list = []
            hypers_list = []
            for _ in range(batch):
                trial = study.ask()
                params, hypers = define_search_space(trial, stage1_ranges, hyper_ranges)
                trials.append(trial)
                params_list.append(params)
                hypers_list.append(hypers)

            if batch == max_conc and not filled_logged:
                print(f"{prefix} Dispatch | filled {batch}/{max_conc}", flush=True)
                filled_logged = True
            print(f"{prefix} Batch {batch_id} | size={batch}", flush=True)

            results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                delayed(eval_mae_joblib)(
                    params_list[i], hypers_list[i], train_idx, val_idx,
                    nb_disp, use_nb, rain_weight, end_mode,
                    early_stop_mae, early_stop_nll, no_obs_nll,
                    DATA_DIR, RB_CACHE_MAX
                )
                for i in range(batch)
            )

            for i, res in enumerate(results):
                train_mae, val_mae, train_std, val_std, train_rmse, val_rmse, train_nll, val_nll = res
                spore_lambda = float(hypers_list[i].get("spore_lambda", SPORE_LAMBDA))
                train_loss = float(train_mae + spore_lambda * train_nll)
                val_loss = float(val_mae + spore_lambda * val_nll)

                study.tell(trials[i], train_loss)
                trial_rows.append({
                    **params_list[i],
                    **hypers_list[i],
                    "train_mae": float(train_mae),
                    "val_mae": float(val_mae),
                    "train_std": float(train_std),
                    "val_std": float(val_std),
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse),
                    "train_spore_nll": float(train_nll),
                    "val_spore_nll": float(val_nll),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "timeout": 0,
                })

                completed += 1
                last_ts = _progress_print(completed, n_trials_stage1, max_conc, max_conc, prefix, last_ts, progress_interval)

        df_trials = pd.DataFrame(trial_rows)
        fold_dir = os.path.join(RESULT_DIR, f"seed_{seed}_assimilation")
        os.makedirs(fold_dir, exist_ok=True)
        fold_csv = os.path.join(fold_dir, f"baseline_kfold_fold_{fold_id}_trials_assimilation.csv")
        df_trials.to_csv(fold_csv, index=False)

        best = select_topk_val_best_constrained(df_trials, TOPK_PCT)
        fold_best_rows.append({
            "fold": fold_id,
            "train_mae": float(best["train_mae"]),
            "val_mae": float(best["val_mae"]),
            "train_std": float(best["train_std"]),
            "val_std": float(best["val_std"]),
            "train_rmse": float(best["train_rmse"]),
            "val_rmse": float(best["val_rmse"]),
            "train_spore_nll": float(best["train_spore_nll"]),
            "val_spore_nll": float(best["val_spore_nll"]),
            "train_loss": float(best["train_loss"]),
            "val_loss": float(best["val_loss"]),
        })

        fold_params.append({k: float(best[k]) for k in stage1_ranges.keys()})

        cfg_best = apply_params(config, fold_params[-1])
        spore_lambda = float(best.get("spore_lambda", SPORE_LAMBDA))
        k_scale = float(best.get("spore_k_scale", SPORE_K_SCALE))
        zi = float(best.get("spore_zero_infl", SPORE_ZERO_INFL))
        train_mae, train_std, train_rmse = compute_error_stats(cfg_best, train_rows)
        val_mae, val_std, val_rmse = compute_error_stats(cfg_best, val_rows)
        fold_best_rows[-1].update({
            "train_mae_full": float(train_mae),
            "val_mae_full": float(val_mae),
            "train_std_full": float(train_std),
            "val_std_full": float(val_std),
            "train_rmse_full": float(train_rmse),
            "val_rmse_full": float(val_rmse),
            "spore_lambda": spore_lambda,
            "spore_k_scale": k_scale,
            "spore_zero_infl": zi,
        })

        print(
            f"{prefix} Done | train_mae={fold_best_rows[-1]['train_mae']:.2f} "
            f"val_mae={fold_best_rows[-1]['val_mae']:.2f} "
            f"train_loss={fold_best_rows[-1]['train_loss']:.2f} "
            f"val_loss={fold_best_rows[-1]['val_loss']:.2f}",
            flush=True
        )

    df_best = pd.DataFrame(fold_best_rows)
    summary = {
        "mean_train_mae": float(df_best["train_mae"].mean()),
        "mean_val_mae": float(df_best["val_mae"].mean()),
        "mean_train_loss": float(df_best["train_loss"].mean()),
        "mean_val_loss": float(df_best["val_loss"].mean()),
    }
    return fold_params, df_best, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=25000)
    parser.add_argument("--n-startup", type=int, default=8000)
    parser.add_argument("--n-jobs", type=int, default=256)
    parser.add_argument("--restart-every", type=int, default=100)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--seed-list", type=str, default="42,123,2024")
    parser.add_argument("--assim-end", type=str, default="sim", choices=["actual", "sim"])
    parser.add_argument("--spore-nb-disp", type=float, default=SPORE_NB_DISP)
    parser.add_argument("--spore-rain-weight", type=float, default=SPORE_RAIN_WEIGHT)
    parser.add_argument("--use-poisson", action="store_true")
    parser.add_argument("--early-stop-mae", type=float, default=EARLY_STOP_MAE)
    parser.add_argument("--early-stop-nll", type=float, default=PENALTY_DAYS)
    parser.add_argument("--no-obs-nll", type=float, default=NO_OBS_NLL)
    args = parser.parse_args()

    if platform.system().lower().startswith("win"):
        max_win_jobs = 32
        if args.n_jobs > max_win_jobs:
            print(f"[Info] Windows max_workers capped to {max_win_jobs} (was {args.n_jobs})", flush=True)
            args.n_jobs = max_win_jobs
    print(f"[Info] Effective n_jobs={args.n_jobs} | JOBLIB_BATCH={JOBLIB_BATCH}", flush=True)

    t0 = time.time()
    print("[Init] Loading actual_data.xlsx...", flush=True)
    actual_df = load_actual()
    print(f"[Init] Loaded actual_data.xlsx in {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    print("[Init] Loading Spore_Overall.xlsx...", flush=True)
    rows = list(actual_df.itertuples(index=False))
    spore_windows = load_spore_obs_windows()
    print(f"[Init] Loaded Spore_Overall.xlsx in {time.time() - t1:.1f}s", flush=True)

    seeds = [int(x) for x in args.seed_list.split(",") if x.strip()] if args.seed_list else [args.seed]
    ranges = build_param_ranges()

    for seed in seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        _, df_best, summary = run_baseline_kfold_joblib(
            rows,
            k=args.k,
            stage1_ranges=ranges,
            hyper_ranges=HYPER_RANGES,
            spore_windows=spore_windows,
            n_jobs=args.n_jobs,
            restart_every=args.restart_every,
            n_trials_stage1=args.n_trials,
            n_startup_stage1=args.n_startup,
            progress_interval=args.progress_interval,
            seed=seed,
            seed_tag=f"[seed={seed}] ",
            nb_disp=args.spore_nb_disp,
            use_nb=not args.use_poisson,
            rain_weight=args.spore_rain_weight,
            end_mode=args.assim_end,
            early_stop_mae=args.early_stop_mae,
            early_stop_nll=args.early_stop_nll,
            no_obs_nll=args.no_obs_nll,
        )

        out_dir = os.path.join(RESULT_DIR, f"seed_{seed}_assimilation")
        os.makedirs(out_dir, exist_ok=True)
        out_best = os.path.join(out_dir, "baseline_kfold_best_assimilation.csv")
        df_best.to_csv(out_best, index=False)
        print("Saved:", out_best)
        print("Summary:", summary)


if __name__ == "__main__":
    main()
