# -*- coding: utf-8 -*-
import os
import sys
import json
import copy
import argparse
import math
import time
from typing import Dict, List, Tuple

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from joblib import Parallel, delayed

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
SPORE_DATA_PATH = os.path.join(DATA_DIR, "Spore_Overall.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration_hazard")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season import CropSeasonFirst
from model.config import config

PENALTY_DAYS = 120.0
FOLD_SEED = 42
TOPK_PCT = 5.0
DEFAULT_N_JOBS = 0
DEFAULT_BATCH_SIZE = 0
DEFAULT_JOBLIB_BACKEND = "loky"
DEFAULT_RESTART_EVERY = 200

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

    # === updated hazard parameter ranges ===
    "hazard__alpha": (1e-4, 10.0),
    "hazard__p": (0.8, 1.8),
    "hazard__q": (0.8, 1.4),
    "hazard__threshold": (0.1, 100.0),

    "thresholds__min_spores": (0.02, 0.06),
    "incubation__spl_end": (1.0, 1.5),
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


def _rb_path_for_row(row: pd.Series) -> str:
    key = f"{row.plot}_{row.variety}_{int(row.year)}"
    return os.path.join(DATA_DIR, f"{key}_grape_rb.json")


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


def load_spore_windows() -> Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]]:
    df = pd.read_excel(SPORE_DATA_PATH)
    df = _normalize_spore_columns(df)
    windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]] = {}
    for (plot, year), g in df.groupby(["plot", "year"]):
        g = g.sort_values("date")
        g = g[g["spore"].notna()]
        prev = None
        for _, row in g.iterrows():
            cur = pd.to_datetime(row["date"]).normalize()
            y = float(row["spore"])
            if prev is not None:
                windows.setdefault((str(plot), int(year)), []).append((prev, cur, y))
            prev = cur
    return windows


def build_folds(n: int, k: int, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    return folds


def define_search_space(trial: optuna.Trial, ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    params = {}
    for k, (lo, hi) in ranges.items():
        if k == "hazard__alpha":
            lo2 = max(lo, 1e-8)
            params[k] = trial.suggest_float(k, lo2, hi, log=True)
        elif k == "hazard__threshold":
            lo2 = max(lo, 1e-8)
            params[k] = trial.suggest_float(k, lo2, hi, log=True)
        else:
            params[k] = trial.suggest_float(k, lo, hi)
    return params


def _calc_time_error(actual_date, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return float(PENALTY_DAYS)
    try:
        first = event_df.iloc[0]
        oil_spot_start = pd.to_datetime(first["oil_spot_start"]).normalize()
        oil_spot_end = pd.to_datetime(first["oil_spot_end"]).normalize()
        actual = pd.to_datetime(actual_date).normalize()
        if pd.isna(oil_spot_start) or pd.isna(oil_spot_end) or pd.isna(actual):
            return float(PENALTY_DAYS)
        if oil_spot_end < actual:
            return float((oil_spot_end - actual).days)
        if oil_spot_start > actual:
            return float((oil_spot_start - actual).days)
        return 0.0
    except Exception:
        return float(PENALTY_DAYS)


def _sim_spore_df(cs: CropSeasonFirst) -> pd.DataFrame:
    if cs.dispersal_cohorts is None or cs.dispersal_cohorts.empty:
        return pd.DataFrame(columns=["date", "signal"])
    df = cs.dispersal_cohorts.copy()
    df["date"] = pd.to_datetime(df["dispersal_datetime"]).dt.normalize()
    df["signal"] = pd.to_numeric(df["airborne_amount"], errors="coerce").fillna(0.0)
    return df[["date", "signal"]]


def _window_sim_value(sim_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if sim_df.empty:
        return 0.0
    mask = (sim_df["date"] >= start) & (sim_df["date"] <= end)
    if not mask.any():
        return 0.0
    return float(sim_df.loc[mask, "signal"].sum())


def compute_metrics(
    cfg: dict,
    rows: List[pd.Series],
    spore_windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]],
    rb_cache: Dict[str, dict] | None = None
) -> Tuple[float, float]:
    scale = 100.0
    time_errors = []
    spore_sq_errors_log = []
    for row in rows:
        rb_path = _rb_path_for_row(row)
        if not os.path.exists(rb_path):
            time_errors.append(float(PENALTY_DAYS))
            continue
        try:
            if rb_cache is not None and rb_path in rb_cache:
                rb = rb_cache[rb_path]
            else:
                rb = load_rb_cached(rb_path)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            time_err = _calc_time_error(row.first_infection_date, cs.primary_chain_unique)
            time_errors.append(float(time_err))
            win_obs = spore_windows.get((str(row.plot), int(row.year)), [])
            if win_obs:
                sim_df = cs.supply_wind if hasattr(cs, "supply_wind") else None
                if sim_df is None or sim_df.empty:
                    continue
                for (s, e, y) in win_obs:
                    s = pd.to_datetime(s).normalize()
                    e = pd.to_datetime(e).normalize()
                    if e < s:
                        continue
                    sim_val = _window_sim_value(sim_df, s, e)
                    obs_log = math.log(float(y) + 1.0)
                    sim_log = math.log(scale * float(sim_val) + 1.0)
                    spore_sq_errors_log.append(float((sim_log - obs_log) ** 2))
        except Exception:
            time_errors.append(float(PENALTY_DAYS))

    if not time_errors:
        time_mae = float(PENALTY_DAYS)
    else:
        time_mae = float(np.mean(np.abs(np.array(time_errors, dtype=float))))

    if not spore_sq_errors_log:
        spore_mse_log = 0.0
    else:
        spore_mse_log = float(np.mean(np.array(spore_sq_errors_log, dtype=float)))

    return time_mae, spore_mse_log


def eval_trial(
    params: Dict[str, float],
    train_rows: List[pd.Series],
    val_rows: List[pd.Series],
    spore_windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]],
    spore_lambda: float,
    rb_cache: Dict[str, dict] | None = None
) -> Dict[str, float]:
    cfg = apply_params(config, params)
    train_time_mae, train_spore_mse_log = compute_metrics(cfg, train_rows, spore_windows, rb_cache=rb_cache)
    val_time_mae, _ = compute_metrics(cfg, val_rows, spore_windows, rb_cache=rb_cache)
    train_loss = float(train_time_mae + spore_lambda * train_spore_mse_log)
    out = {
        "train_time_mae": float(train_time_mae),
        "val_time_mae": float(val_time_mae),
        "train_spore_mse_log": float(train_spore_mse_log),
        "train_loss": float(train_loss),
    }
    for k, v in params.items():
        out[k] = float(v)
    return out


def select_best_trial(df: pd.DataFrame, topk_pct: float) -> pd.Series:
    if df.empty:
        raise ValueError("No trials to select from")
    k = max(1, int(math.ceil(len(df) * topk_pct / 100.0)))
    top_df = df.nsmallest(k, "train_loss")
    constrained = top_df[top_df["val_time_mae"] >= top_df["train_time_mae"]]
    if not constrained.empty:
        return constrained.sort_values("val_time_mae").iloc[0]
    tmp = top_df.copy()
    tmp["gap"] = tmp["train_time_mae"] - tmp["val_time_mae"]
    return tmp.sort_values("gap").iloc[0]


def run_kfold(
    rows: List[pd.Series],
    k: int,
    ranges: Dict[str, Tuple[float, float]],
    spore_windows: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, pd.Timestamp, float]]],
    n_trials: int,
    n_startup: int,
    n_jobs: int,
    batch_size: int,
    joblib_backend: str,
    restart_every: int,
    spore_lambda: float,
    seed: int,
    out_dir: str
) -> Dict[str, float]:
    folds = build_folds(len(rows), k, FOLD_SEED)
    fold_rows = []
    os.makedirs(out_dir, exist_ok=True)
    fold_durations: List[float] = []

    for fold_id, val_idx in enumerate(folds, start=1):
        fold_start = time.time()
        print(f"[Seed {seed}] Fold {fold_id}/{k} start", flush=True)
        train_idx = [i for i in range(len(rows)) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        sampler = TPESampler(seed=seed + fold_id, multivariate=True, n_startup_trials=n_startup)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        trial_rows = []
        completed = 0
        next_restart = restart_every if restart_every > 0 else None

        fold_rb_cache: Dict[str, dict] = {}
        for row in train_rows + val_rows:
            rb_path = _rb_path_for_row(row)
            if not os.path.exists(rb_path):
                continue
            if rb_path in fold_rb_cache:
                continue
            try:
                with open(rb_path, "r") as f:
                    fold_rb_cache[rb_path] = json.load(f)
            except Exception:
                continue

        while completed < n_trials:
            with Parallel(n_jobs=n_jobs, backend=joblib_backend) as parallel:
                while completed < n_trials:
                    batch = min(batch_size, n_trials - completed)
                    trials = [study.ask() for _ in range(batch)]
                    params_list = [define_search_space(t, ranges) for t in trials]
                    results = parallel(
                        delayed(eval_trial)(p, train_rows, val_rows, spore_windows, spore_lambda, fold_rb_cache)
                        for p in params_list
                    )
                    for trial, res in zip(trials, results):
                        study.tell(trial, res["train_loss"])
                        res["trial_number"] = int(trial.number)
                        res["fold"] = int(fold_id)
                        trial_rows.append(res)
                        completed += 1
                    if completed > 0:
                        elapsed = time.time() - fold_start
                        rate = completed / max(elapsed, 1e-6)
                        remaining = (n_trials - completed) / max(rate, 1e-6)
                        print(
                            f"[Seed {seed}] Fold {fold_id}/{k} {completed}/{n_trials} "
                            f"elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m",
                            flush=True
                        )
                    if next_restart is not None and completed >= next_restart:
                        break
            if next_restart is not None and completed >= next_restart:
                next_restart += restart_every

        df_trials = pd.DataFrame(trial_rows)
        df_trials = df_trials.sort_values("train_loss")
        fold_path = os.path.join(out_dir, f"fold_{fold_id}.csv")
        df_trials.to_csv(fold_path, index=False)

        best = select_best_trial(df_trials, TOPK_PCT)
        fold_rows.append({
            "fold": int(fold_id),
            "train_time_mae": float(best["train_time_mae"]),
            "val_time_mae": float(best["val_time_mae"]),
            "train_spore_mse_log": float(best["train_spore_mse_log"]),
            "train_loss": float(best["train_loss"]),
        })
        fold_elapsed = time.time() - fold_start
        fold_durations.append(fold_elapsed)
        avg_fold = sum(fold_durations) / len(fold_durations)
        remaining_folds = k - fold_id
        eta_folds = avg_fold * remaining_folds
        print(
            f"[Seed {seed}] Fold {fold_id}/{k} done in {fold_elapsed/60:.1f}m | "
            f"ETA remaining folds {eta_folds/60:.1f}m",
            flush=True
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.loc["mean"] = fold_df.mean(numeric_only=True)
    fold_summary_path = os.path.join(out_dir, "fold_summary.csv")
    fold_df.to_csv(fold_summary_path, index=False)
    seed_summary = {
        "seed": int(seed),
        "seed_mean_val_mae": float(fold_df["val_time_mae"].mean()),
        "seed_mean_train_loss": float(fold_df["train_loss"].mean()),
        "seed_mean_spore_mse_log": float(fold_df["train_spore_mse_log"].mean()),
    }
    seed_summary_path = os.path.join(out_dir, "seed_summary.csv")
    pd.DataFrame([seed_summary]).to_csv(seed_summary_path, index=False)
    return seed_summary

'''1'''
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=20000)
    parser.add_argument("--n-startup", type=int, default=8000)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--joblib-backend", type=str, default=DEFAULT_JOBLIB_BACKEND)
    parser.add_argument("--restart-every", type=int, default=DEFAULT_RESTART_EVERY)
    parser.add_argument("--lambda", dest="spore_lambda", type=float, default=0.3)
    parser.add_argument("--seed-list", type=str, default="42,123,2024")
    args = parser.parse_args()

    seed_list = [int(x) for x in args.seed_list.split(",") if x.strip()]
    if not seed_list:
        seed_list = [42]

    n_jobs = args.n_jobs
    if n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    if os.name == "nt":
        n_jobs = min(n_jobs, 61)
    batch_size = args.batch_size if args.batch_size and args.batch_size > 0 else n_jobs
    if batch_size <= 0:
        batch_size = 1

    rows = list(load_actual().itertuples(index=False))
    ranges = build_param_ranges()
    spore_windows = load_spore_windows()

    base_out = RESULT_DIR
    os.makedirs(base_out, exist_ok=True)
    seed_summaries = []

    for seed in seed_list:
        seed_start = time.time()
        print(f"[Seed {seed}] start", flush=True)
        seed_dir = os.path.join(base_out, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        summary = run_kfold(
            rows=rows,
            k=args.k,
            ranges=ranges,
            spore_windows=spore_windows,
            n_trials=args.n_trials,
            n_startup=args.n_startup,
            n_jobs=n_jobs,
            batch_size=batch_size,
            joblib_backend=args.joblib_backend,
            restart_every=args.restart_every,
            spore_lambda=float(args.spore_lambda),
            seed=seed,
            out_dir=seed_dir
        )
        seed_summaries.append(summary)
        seed_elapsed = time.time() - seed_start
        print(f"[Seed {seed}] done in {seed_elapsed/60:.1f}m", flush=True)

    if seed_summaries:
        df_global = pd.DataFrame(seed_summaries)
        best_idx = int(df_global["seed_mean_val_mae"].idxmin())
        best_seed = int(df_global.loc[best_idx, "seed"])
        df_global["is_best"] = df_global["seed"] == best_seed
        df_global["global_best_seed"] = best_seed
        global_path = os.path.join(base_out, "global_summary.csv")
        df_global.to_csv(global_path, index=False)


if __name__ == "__main__":
    main()
