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
import ray
import optuna
from optuna.samplers import TPESampler
import time

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
N_JOBS = 128
RB_CACHE_MAX = 128
TOPK_PCT = 5.0

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
    no_obs_nll: float
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
            rb = load_rb_cached(rb_path)
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


@ray.remote(num_cpus=1)
class ModelWorker:
    def __init__(self, rows_ref, spore_windows_ref):
        if isinstance(rows_ref, ray.ObjectRef):
            self.rows = ray.get(rows_ref)
        else:
            self.rows = rows_ref
        if isinstance(spore_windows_ref, ray.ObjectRef):
            self.spore_windows = ray.get(spore_windows_ref)
        else:
            self.spore_windows = spore_windows_ref
        self.rb_cache: "OrderedDict[str, dict]" = OrderedDict()
        self.rb_cache_max = RB_CACHE_MAX

    def _load_rb_cached_lru(self, path: str) -> dict:
        if path in self.rb_cache:
            self.rb_cache.move_to_end(path)
            return self.rb_cache[path]
        with open(path, "r") as f:
            data = json.load(f)
        self.rb_cache[path] = data
        self.rb_cache.move_to_end(path)
        if self.rb_cache_max > 0 and len(self.rb_cache) > self.rb_cache_max:
            self.rb_cache.popitem(last=False)
        return data

    def _compute_error_stats_idx(self, cfg: dict, idxs: List[int]) -> Tuple[float, float, float]:
        errors = []
        for i in idxs:
            row = self.rows[i]
            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
            if not os.path.exists(rb_path):
                continue
            try:
                rb = self._load_rb_cached_lru(rb_path)
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

    def eval_mae(
        self,
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
        no_obs_nll: float
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        cfg = apply_params(config, params)
        k_scale = float(hypers.get("spore_k_scale", SPORE_K_SCALE))
        zi = float(hypers.get("spore_zero_infl", SPORE_ZERO_INFL))
        train_mae, train_std, train_rmse = self._compute_error_stats_idx(cfg, train_idx)
        if early_stop_mae > 0 and train_mae >= early_stop_mae:
            val_mae, val_std, val_rmse = self._compute_error_stats_idx(cfg, val_idx)
            train_nll = early_stop_nll
            val_nll = early_stop_nll
            return train_mae, val_mae, train_std, val_std, train_rmse, val_rmse, train_nll, val_nll
        val_mae, val_std, val_rmse = self._compute_error_stats_idx(cfg, val_idx)
        train_rows = [self.rows[i] for i in train_idx]
        val_rows = [self.rows[i] for i in val_idx]
        train_nll = compute_spore_assim_nll(
            cfg, train_rows, self.spore_windows, k_scale, nb_disp, zi, use_nb, rain_weight, end_mode, no_obs_nll
        )
        val_nll = compute_spore_assim_nll(
            cfg, val_rows, self.spore_windows, k_scale, nb_disp, zi, use_nb, rain_weight, end_mode, no_obs_nll
        )
        return train_mae, val_mae, train_std, val_std, train_rmse, val_rmse, train_nll, val_nll


def _make_workers(n_jobs: int, rows_ref, spore_windows_ref) -> List[ray.actor.ActorHandle]:
    return [ModelWorker.remote(rows_ref, spore_windows_ref) for _ in range(n_jobs)]


def _replace_worker(workers: List[ray.actor.ActorHandle], old_worker, rows_ref, spore_windows_ref):
    try:
        idx = workers.index(old_worker)
    except ValueError:
        idx = None
    ray.kill(old_worker, no_restart=True)
    new_worker = ModelWorker.remote(rows_ref, spore_windows_ref)
    if idx is not None:
        workers[idx] = new_worker
    else:
        workers.append(new_worker)
    return new_worker


def run_baseline_kfold(
    rows: List[pd.Series],
    k: int,
    stage1_ranges: Dict[str, Tuple[float, float]],
    hyper_ranges: Dict[str, Tuple[float, float]],
    spore_windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]],
    workers: List[ray.actor.ActorHandle],
    rows_ref,
    spore_windows_ref,
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
    trial_timeout_s: float = 0.0,
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
        pending = {}
        completed = 0
        max_conc = max(1, n_jobs)
        available = list(workers)
        next_restart = restart_every if restart_every > 0 else None
        prefix = f"{seed_tag}[Assimilation][Fold {fold_id}/{k}]"
        _stage_start(prefix, n_trials_stage1, 0, max_conc)
        last_ts = _progress_print(0, n_trials_stage1, 0, max_conc, prefix, last_ts, progress_interval)

        while completed < n_trials_stage1:
            while available and len(pending) < max_conc and completed + len(pending) < n_trials_stage1:
                trial = study.ask()
                params, hypers = define_search_space(trial, stage1_ranges, hyper_ranges)
                worker = available.pop(0)
                fut = worker.eval_mae.remote(
                    params, hypers, train_idx, val_idx, nb_disp, use_nb, rain_weight, end_mode,
                    early_stop_mae, early_stop_nll, no_obs_nll
                )
                pending[fut] = (trial, params, hypers, worker, time.time())

            if not pending:
                continue

            done_futs, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=1.0)

            if trial_timeout_s and trial_timeout_s > 0:
                now = time.time()
                timed_out = [
                    fut for fut, (_, _, _, _, ts) in pending.items()
                    if now - ts >= trial_timeout_s
                ]
                for fut in timed_out:
                    trial, params, hypers, worker, _ts = pending.pop(fut)
                    try:
                        ray.cancel(fut, force=True)
                    except Exception:
                        pass
                    spore_lambda = float(hypers.get("spore_lambda", SPORE_LAMBDA))
                    train_mae = PENALTY_DAYS
                    val_mae = PENALTY_DAYS
                    train_std = 0.0
                    val_std = 0.0
                    train_rmse = PENALTY_DAYS
                    val_rmse = PENALTY_DAYS
                    train_nll = no_obs_nll
                    val_nll = no_obs_nll
                    train_loss = float(train_mae + spore_lambda * train_nll)
                    val_loss = float(val_mae + spore_lambda * val_nll)
                    study.tell(trial, train_loss)
                    trial_rows.append({
                        **params,
                        **hypers,
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
                        "timeout": 1,
                    })
                    completed += 1
                    new_worker = _replace_worker(workers, worker, rows_ref, spore_windows_ref)
                    available.append(new_worker)
            for fut in done_futs:
                trial, params, hypers, worker, _ts = pending.pop(fut)
                try:
                    train_mae, val_mae, train_std, val_std, train_rmse, val_rmse, train_nll, val_nll = ray.get(fut)
                except Exception:
                    new_worker = _replace_worker(workers, worker, rows_ref, spore_windows_ref)
                    available.append(new_worker)
                    continue

                spore_lambda = float(hypers.get("spore_lambda", SPORE_LAMBDA))
                train_loss = float(train_mae + spore_lambda * train_nll)
                val_loss = float(val_mae + spore_lambda * val_nll)

                study.tell(trial, train_loss)
                trial_rows.append({
                    **params,
                    **hypers,
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
                available.append(worker)
                last_ts = _progress_print(completed, n_trials_stage1, len(pending), max_conc, prefix, last_ts, progress_interval)

                if next_restart is not None and completed >= next_restart:
                    next_restart += restart_every
                    for w in list(workers):
                        new_w = _replace_worker(workers, w, rows_ref, spore_windows_ref)
                        available = [aw if aw != w else new_w for aw in available]

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

        # Evaluate best on full fold sets for reporting
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
    parser.add_argument("--n-startup", type=int, default=10000)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
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
    parser.add_argument("--trial-timeout", type=float, default=180.0)
    args = parser.parse_args()

    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))
    spore_windows = load_spore_obs_windows()

    print("Initializing Ray...", flush=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=args.n_jobs)
    try:
        print("Ray resources:", ray.available_resources(), flush=True)
    except Exception:
        pass
    rows_ref = ray.put(rows)
    spore_ref = ray.put(spore_windows)
    print(f"Creating {args.n_jobs} workers...", flush=True)
    workers = _make_workers(args.n_jobs, rows_ref, spore_ref)
    print("Workers ready.", flush=True)

    seeds = [int(x) for x in args.seed_list.split(",") if x.strip()] if args.seed_list else [args.seed]
    ranges = build_param_ranges()

    for seed in seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        _, df_best, summary = run_baseline_kfold(
            rows,
            k=args.k,
            stage1_ranges=ranges,
            hyper_ranges=HYPER_RANGES,
            spore_windows=spore_windows,
            workers=workers,
            rows_ref=rows_ref,
            spore_windows_ref=spore_ref,
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
            trial_timeout_s=args.trial_timeout,
        )

        out_dir = os.path.join(RESULT_DIR, f"seed_{seed}_assimilation")
        os.makedirs(out_dir, exist_ok=True)
        out_best = os.path.join(out_dir, "baseline_kfold_best_assimilation.csv")
        df_best.to_csv(out_best, index=False)
        print("Saved:", out_best)
        print("Summary:", summary)


if __name__ == "__main__":
    main()
