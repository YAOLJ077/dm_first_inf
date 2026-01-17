# -*- coding: utf-8 -*-
"""
LOO calibration with two stages:

Stage 1 (spore-process calibration):
  - Calibrate spore-process parameters using window matching between
    simulated capture windows (wind dispersal -> deposition/survival end)
    and observed spore windows (spore>0).
  - Objective: Loss_spore = alpha*FNP + beta*FPP within actual_date-30d window.

Stage 2 (MAE calibration):
  - Fix or softly relax stage1 params, calibrate remaining params by MAE.
"""

import os
import sys
import json
import copy
import argparse
import math
from typing import Dict, List, Tuple

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

import numpy as np
import time
import pandas as pd
import ray
import optuna
from optuna.samplers import TPESampler

# ======================================================
# PATHS
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_DATA_PATH = os.path.join(DATA_DIR, "Spore_overall.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season import CropSeasonFirst
from model.config import config

# ======================================================
# SETTINGS
# ======================================================
PENALTY_DAYS = 120.0
N_JOBS = 128
RANDOM_SEED = 42
FOLD_SEED = 42

SPORE_WINDOW_DAYS = 30
SPORE_ALPHA = 2.0
SPORE_BETA = 1.0
STAGE1_RELAX = 0.1
# Stage1 parameters (spore-process calibration)
# - mp_tmp__rh_wet_night
# - sporulation__MP_min
# - sporulation__Tmin
# - sporulation__Topt
# - sporulation__Tmax
# - thresholds__min_spores
# - dispersal__wind_min
# - deposition__rain_min
#
# Optional per-parameter absolute relax (overrides STAGE1_RELAX percent).
# Example A (temperature-heavy):
# STAGE1_RELAX_ABS = {
#     "sporulation__Tmin": 0.5,
#     "sporulation__Topt": 0.5,
#     "sporulation__Tmax": 2.0,
# }
# Example B (dispersal-heavy):
# STAGE1_RELAX_ABS = {
#     "dispersal__wind_min": 0.5,
#     "deposition__rain_min": 1.0,
#     "thresholds__min_spores": 0.01,
# }
STAGE1_RELAX_ABS = {
    "mp_tmp__rh_wet_night": 3.0,
    "sporulation__MP_min": 1.0,
    "sporulation__Tmin": 2.0,
    "sporulation__Topt": 2.0,
    "sporulation__Tmax": 2.0,
    "thresholds__min_spores": 0.01,
    "dispersal__wind_min": 2.0,
    "deposition__rain_min": 2.0,
}


# Detached mortality (from model.first_infection)
DET_B0 = 5.67
DET_B1 = 0.47
DET_B2 = 0.02

# Base ranges from Morris settings
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
    "thresholds__min_spores": (0.02, 0.06),
}

STAGE1_PARAMS = {
    "mp_tmp__rh_wet_night",
    "sporulation__MP_min",
    "sporulation__Tmin",
    "sporulation__Topt",
    "sporulation__Tmax",
    "thresholds__min_spores",
    "dispersal__wind_min",
    "deposition__rain_min",
}

STAGE2_PARAMS = {
    "zoospore_alive__rain",
    "zoospore_alive__vpd",
    "infection__infection_condition",
    "incubation__spl_end",
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


def load_spores() -> pd.DataFrame:
    df = pd.read_excel(SPORE_DATA_PATH)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df["spore"] = pd.to_numeric(df["spore"], errors="coerce").fillna(0)
    df["present"] = df["spore"] > 0
    return df


def build_spore_windows(df: pd.DataFrame) -> Dict[Tuple[str, int], List[Tuple[object, object, bool]]]:
    windows = {}
    for (plot, year), g in df.groupby(["plot", "year"]):
        g = g.sort_values("date")
        prev_date = None
        entries = []
        for _, row in g.iterrows():
            cur_date = row["date"].date()
            start = prev_date or cur_date
            end = cur_date
            entries.append((start, end, bool(row["present"])))
            prev_date = cur_date
        windows[(str(plot), int(year))] = entries
    return windows


def apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in values.items():
        group, sub = k.split("__")
        cfg[group][sub] = float(v)
    return cfg


def compute_mae(cfg: dict, rows: List[pd.Series]) -> float:
    errors = []
    for row in rows:
        key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue
        try:
            with open(rb_path, "r") as f:
                rb = json.load(f)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            err = _calc_error(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            err = np.nan
        if np.isfinite(err):
            errors.append(abs(err))
        else:
            errors.append(PENALTY_DAYS)
    if not errors:
        return PENALTY_DAYS
    return float(np.mean(errors))


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


def _compute_vpd(wh: pd.DataFrame) -> np.ndarray:
    T = wh["air_temp"].values
    RH = wh["relative_humidity"].values
    return 6.11 * np.exp((17.47 * T) / (239 + T)) * (1 - RH / 100)


def _alive_end(wh: pd.DataFrame, vpd: np.ndarray, start_dt, D0: float, th: float):
    time_index = wh["datetime"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(time_index, np.datetime64(start_dt))
    D = float(D0)
    end_time = wh.datetime.iloc[-1]
    for i in range(idx, len(wh)):
        hr = wh.iloc[i]
        if 8 <= hr.datetime.hour <= 18:
            lam = max(0.0, DET_B0 - DET_B1 * vpd[i] + DET_B2 * vpd[i] ** 2) / 24.0
            D *= math.exp(-lam)
            if D < th:
                end_time = hr.datetime
                break
    return end_time


def build_sim_windows(cs: CropSeasonFirst) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    if cs.dispersal_cohorts is None or cs.dispersal_cohorts.empty:
        return windows
    disp = cs.dispersal_cohorts.copy()
    dep = cs.deposition if cs.deposition is not None else pd.DataFrame()
    dep_map = {}
    if dep is not None and not dep.empty:
        dep_map = dict(zip(dep["Num"].values, dep["deposition_datetime"].values))
    wh = cs.weather_hourly
    vpd = _compute_vpd(wh)
    th = cs.config["thresholds"]["min_spores"]
    for _, c in disp.iterrows():
        if c.get("dispersal_type") != "wind":
            continue
        start = c.dispersal_datetime
        end = dep_map.get(c.Num)
        if pd.isna(end):
            end = _alive_end(wh, vpd, start, c.airborne_amount, th)
        if pd.notna(end):
            windows.append((pd.to_datetime(start), pd.to_datetime(end)))
    return windows


def _adjust_rb_start(rb: dict, bbch_date, actual_date, window_days: int) -> dict:
    rb = copy.deepcopy(rb)
    bbch = pd.to_datetime(bbch_date, errors="coerce")
    actual = pd.to_datetime(actual_date, errors="coerce")
    if pd.isna(bbch) or pd.isna(actual):
        return rb
    if bbch >= actual:
        start = actual - pd.Timedelta(days=window_days)
    else:
        start = bbch - pd.Timedelta(days=15)
    # CropSeasonFirst uses HHCH_53 - 7 days as gate start.
    rb["HHCH_53"] = (start + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return rb


def spore_loss(
    cfg: dict,
    rows: List[pd.Series],
    spore_windows: Dict[Tuple[str, int], List[Tuple[object, object, bool]]],
    window_days: int,
    alpha: float,
    beta: float,
) -> float:
    tp = fp = fn = 0
    for row in rows:
        key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue
        try:
            with open(rb_path, "r") as f:
                rb = json.load(f)
            rb = _adjust_rb_start(rb, getattr(row, "bbch_date", None), row.first_infection_date, window_days)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            sim_windows = build_sim_windows(cs)
        except Exception:
            sim_windows = []

        actual = pd.to_datetime(row.first_infection_date)
        start = (actual - pd.Timedelta(days=window_days)).date()
        end = actual.date()

        obs = []
        for w_start, w_end, present in spore_windows.get((str(row.plot), int(row.first_infection_date.year)), []):
            if not present:
                continue
            if w_end < start or w_start > end:
                continue
            obs.append((w_start, w_end))

        sim_in = []
        for s, e in sim_windows:
            if e.date() < start or s.date() > end:
                continue
            sim_in.append((s.date(), e.date()))

        matched_obs = set()
        for s, e in sim_in:
            hit = False
            for i, (os_s, os_e) in enumerate(obs):
                if not (e < os_s or s > os_e):
                    hit = True
                    matched_obs.add(i)
                    break
            if hit:
                tp += 1
            else:
                fp += 1

        fn += max(0, len(obs) - len(matched_obs))

    fpp = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    fnp = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return alpha * fnp + beta * fpp


def define_search_space(trial: optuna.Trial, ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    params = {}
    for param_name, (lo, hi) in ranges.items():
        params[param_name] = trial.suggest_float(param_name, lo, hi)
    return params


@ray.remote
def eval_spore(
    params: Dict[str, float],
    train_rows: List[pd.Series],
    val_rows: List[pd.Series],
    spore_windows,
    window_days,
    alpha,
    beta
) -> Tuple[float, float]:
    cfg = apply_params(config, params)
    train_loss = spore_loss(cfg, train_rows, spore_windows, window_days, alpha, beta)
    val_loss = spore_loss(cfg, val_rows, spore_windows, window_days, alpha, beta)
    return train_loss, val_loss


@ray.remote
def eval_mae(params: Dict[str, float], train_rows: List[pd.Series], val_rows: List[pd.Series]) -> Tuple[float, float]:
    cfg = apply_params(config, params)
    return compute_mae(cfg, train_rows), compute_mae(cfg, val_rows)


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


def _progress_print(done: int, total: int, prefix: str, last_ts: float, interval_s: float) -> float:
    now = time.time()
    if now - last_ts >= interval_s:
        msg = f"{prefix} Progress: {done}/{total} completed"
        print(msg, flush=True)
        return now
    return last_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials-spore", type=int, default=10000)
    parser.add_argument("--n-startup-spore", type=int, default=1000)
    parser.add_argument("--n-trials-mae", type=int, default=8000)
    parser.add_argument("--n-startup-mae", type=int, default=800)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--train-top-pct", type=float, default=1.0)
    parser.add_argument("--kfold", type=int, default=6)
    parser.add_argument("--spore-window-days", type=int, default=SPORE_WINDOW_DAYS)
    parser.add_argument("--spore-alpha", type=float, default=SPORE_ALPHA)
    parser.add_argument("--spore-beta", type=float, default=SPORE_BETA)
    parser.add_argument("--stage1-relax", type=float, default=STAGE1_RELAX)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    args = parser.parse_args()

    ranges = build_param_ranges()
    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))
    spore_df = load_spores()
    spore_windows = build_spore_windows(spore_df)

    ray.init(num_cpus=args.n_jobs, include_dashboard=False)

    n_samples = len(rows)
    k = args.kfold if args.kfold > 0 else n_samples
    folds = build_folds(n_samples, k, FOLD_SEED)

    stage1_fold_rows = []
    stage2_fold_rows = []
    stage1_fold_params = []
    stage2_fold_params = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = [i for i in range(n_samples) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        # ---------------- Stage 1: spore process ----------------
        stage1_ranges = {k: v for k, v in ranges.items() if k in STAGE1_PARAMS}
        sampler1 = TPESampler(seed=args.seed + fold_id, multivariate=True, n_startup_trials=args.n_startup_spore)
        study1 = optuna.create_study(sampler=sampler1, direction="minimize")
        stage1_trials = []
        last_ts = 0.0
        pending = {}
        completed = 0
        max_conc = max(1, int(args.n_jobs))
        while completed < args.n_trials_spore:
            while len(pending) < max_conc and completed + len(pending) < args.n_trials_spore:
                trial = study1.ask()
                params = define_search_space(trial, stage1_ranges)
                fut = eval_spore.remote(
                    params,
                    train_rows,
                    val_rows,
                    spore_windows,
                    args.spore_window_days,
                    args.spore_alpha,
                    args.spore_beta
                )
                pending[fut] = (trial, params)

            done_ids, _ = ray.wait(list(pending.keys()), num_returns=max(1, min(len(pending), max_conc)))
            for fut in done_ids:
                trial, params = pending.pop(fut)
                train_loss, val_loss = ray.get(fut)
                study1.tell(trial, train_loss)
                stage1_trials.append({
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    **params
                })
                completed += 1

            last_ts = _progress_print(
                completed,
                args.n_trials_spore,
                f"[Fold {fold_id}/{k}][Stage1]",
                last_ts,
                args.progress_interval
            )

        df_stage1 = pd.DataFrame(stage1_trials).sort_values("train_loss")
        n_keep1 = max(1, int(len(df_stage1) * args.train_top_pct / 100.0))
        pool1 = df_stage1.head(n_keep1)
        best1 = pool1.sort_values("val_loss").iloc[0]
        stage1_best = {
            param_name: best1[param_name]
            for param_name in best1.index
            if param_name not in ("train_loss", "val_loss")
        }
        stage1_fold_params.append(stage1_best)
        stage1_fold_rows.append({
            "fold": fold_id,
            "train_loss": float(best1["train_loss"]),
            "val_loss": float(best1["val_loss"]),
        })
        stage1_out = os.path.join(RESULT_DIR, f"loo_spore_stage1_fold{fold_id}.csv")
        df_stage1.to_csv(stage1_out, index=False)
        stage1_best_out = os.path.join(RESULT_DIR, f"loo_spore_stage1_best_fold{fold_id}.json")
        with open(stage1_best_out, "w") as f:
            json.dump({
                "fold": fold_id,
                "train_loss": float(best1["train_loss"]),
                "val_loss": float(best1["val_loss"]),
                "params": stage1_best,
            }, f, indent=2)

        # ---------------- Stage 2: MAE calibration ----------------
        stage2_ranges = {k: v for k, v in ranges.items() if k in STAGE2_PARAMS}
        if args.stage1_relax > 0 or STAGE1_RELAX_ABS:
            for param_name, param_value in stage1_best.items():
                g_lo, g_hi = ranges[param_name]
                if param_name in STAGE1_RELAX_ABS:
                    delta = float(STAGE1_RELAX_ABS[param_name])
                    lo = max(g_lo, param_value - delta)
                    hi = min(g_hi, param_value + delta)
                else:
                    lo = max(g_lo, param_value * (1 - args.stage1_relax))
                    hi = min(g_hi, param_value * (1 + args.stage1_relax))
                stage2_ranges[param_name] = (lo, hi)

        sampler2 = TPESampler(seed=args.seed + fold_id + 1000, multivariate=True, n_startup_trials=args.n_startup_mae)
        study2 = optuna.create_study(sampler=sampler2, direction="minimize")
        trial_rows = []
        last_ts = 0.0
        pending = {}
        completed = 0
        while completed < args.n_trials_mae:
            while len(pending) < max_conc and completed + len(pending) < args.n_trials_mae:
                trial = study2.ask()
                params = define_search_space(trial, stage2_ranges)
                params = {**stage1_best, **params}
                fut = eval_mae.remote(params, train_rows, val_rows)
                pending[fut] = (trial, params)

            done_ids, _ = ray.wait(list(pending.keys()), num_returns=max(1, min(len(pending), max_conc)))
            for fut in done_ids:
                trial, params = pending.pop(fut)
                train_mae, val_mae = ray.get(fut)
                study2.tell(trial, train_mae)
                trial_rows.append({"train_mae": train_mae, "val_mae": val_mae, **params})
                completed += 1

            last_ts = _progress_print(
                completed,
                args.n_trials_mae,
                f"[Fold {fold_id}/{k}][Stage2]",
                last_ts,
                args.progress_interval
            )

        df_trials = pd.DataFrame(trial_rows).sort_values("train_mae")
        stage2_trials_out = os.path.join(RESULT_DIR, f"loo_spore_stage2_fold{fold_id}.csv")
        df_trials.to_csv(stage2_trials_out, index=False)
        n_keep = max(1, int(len(df_trials) * args.train_top_pct / 100.0))
        pool = df_trials.head(n_keep)
        best_val_row = pool.sort_values("val_mae").iloc[0]
        best_params = {
            param_name: best_val_row[param_name]
            for param_name in best_val_row.index
            if param_name not in ("train_mae", "val_mae")
        }
        stage2_fold_params.append(best_params)
        stage2_fold_rows.append({
            "fold": fold_id,
            "avg_train_mae": float(best_val_row["train_mae"]),
            "val_mae": float(best_val_row["val_mae"]),
        })
        stage2_best_out = os.path.join(RESULT_DIR, f"loo_spore_stage2_best_fold{fold_id}.json")
        with open(stage2_best_out, "w") as f:
            json.dump({
                "fold": fold_id,
                "avg_train_mae": float(best_val_row["train_mae"]),
                "val_mae": float(best_val_row["val_mae"]),
                "params": best_params,
            }, f, indent=2)

    stage1_summary = pd.DataFrame(stage1_fold_rows)
    stage1_out = os.path.join(RESULT_DIR, "loo_spore_stage1_summary.csv")
    stage1_summary.to_csv(stage1_out, index=False)
    stage1_stats = {
        "train_loss_mean": float(stage1_summary["train_loss"].mean()),
        "val_loss_mean": float(stage1_summary["val_loss"].mean()),
        "spore_alpha": float(args.spore_alpha),
        "spore_beta": float(args.spore_beta),
        "spore_window_days": int(args.spore_window_days),
    }
    stage1_json = os.path.join(RESULT_DIR, "loo_spore_stage1_summary.json")
    with open(stage1_json, "w") as f:
        json.dump(stage1_stats, f, indent=2)

    stage2_summary = pd.DataFrame(stage2_fold_rows)
    stage2_all = []
    for fold_id in range(1, k + 1):
        fp = os.path.join(RESULT_DIR, f"loo_spore_stage2_fold{fold_id}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df["fold"] = fold_id
            stage2_all.append(df)
    if stage2_all:
        stage2_all_df = pd.concat(stage2_all, ignore_index=True)
        stage2_all_out = os.path.join(RESULT_DIR, "loo_spore_stage2_all_folds.csv")
        stage2_all_df.to_csv(stage2_all_out, index=False)
    stage2_out = os.path.join(RESULT_DIR, "loo_spore_stage2_summary.csv")
    stage2_summary.to_csv(stage2_out, index=False)
    stage2_stats = {
        "train_mae_mean": float(stage2_summary["avg_train_mae"].mean()),
        "val_mae_mean": float(stage2_summary["val_mae"].mean()),
        "stage1_relax": float(args.stage1_relax),
    }
    stage2_json = os.path.join(RESULT_DIR, "loo_spore_stage2_summary.json")
    with open(stage2_json, "w") as f:
        json.dump(stage2_stats, f, indent=2)

    # Global best from stage2 candidates.
    global_rows = []
    for i, params in enumerate(stage2_fold_params, start=1):
        cfg = apply_params(config, params)
        global_mae = compute_mae(cfg, rows)
        global_rows.append({"fold": i, "global_mae": float(global_mae)})
    df_global = pd.DataFrame(global_rows)
    best_idx = int(df_global["global_mae"].idxmin())
    best_fold = int(df_global.loc[best_idx, "fold"])
    best_params = stage2_fold_params[best_idx]
    best_global = {
        "best_from_fold": best_fold,
        "global_mae": float(df_global.loc[best_idx, "global_mae"]),
        "avg_train_mae": float(stage2_summary.loc[best_idx, "avg_train_mae"]),
        "val_mae": float(stage2_summary.loc[best_idx, "val_mae"]),
        "params": best_params,
    }
    global_out = os.path.join(RESULT_DIR, "loo_spore_stage2_best_global.json")
    with open(global_out, "w") as f:
        json.dump(best_global, f, indent=2)

    ray.shutdown()


if __name__ == "__main__":
    main()
