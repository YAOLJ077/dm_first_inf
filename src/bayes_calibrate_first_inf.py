# -*- coding: utf-8 -*-
"""
Bayesian calibration (Optuna) for FIRST infection model.
Objective: minimize LOOCV MAE with penalty for missing events.
"""

import os
import sys
import json
import copy
import argparse
from typing import Dict, List, Tuple

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")

import numpy as np
import pandas as pd

import ray
import optuna
from optuna.samplers import TPESampler
import time

# ======================================================
# Make project root importable
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# PATH SETUP
# ======================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season import CropSeasonFirst
from model.config import config

# ======================================================
# SETTINGS
# ======================================================
PENALTY_DAYS = 120.0
N_JOBS = 64
RANDOM_SEED = 42

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
    "thresholds__min_spores": (0.02, 0.08),
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
    return df


def to_doy(dt):
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear


def calc_error(actual_date, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return np.nan

    first = event_df.iloc[0]
    doy_actual = to_doy(actual_date)
    doy_start = to_doy(first["oil_spot_start"])
    doy_end = to_doy(first["oil_spot_end"])

    if doy_actual is None or doy_start is None or doy_end is None:
        return np.nan

    if doy_end < doy_actual:
        return doy_end - doy_actual
    if doy_start > doy_actual:
        return doy_start - doy_actual
    return 0.0


_RB_CACHE: Dict[str, dict] = {}


def load_rb_cached(rb_path: str) -> dict:
    if rb_path not in _RB_CACHE:
        with open(rb_path, "r") as f:
            _RB_CACHE[rb_path] = json.load(f)
    return _RB_CACHE[rb_path]


def apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in values.items():
        group, sub = k.split("__")
        cfg[group][sub] = float(v)
    return cfg


_GLOBAL_RANGES: Dict[str, Tuple[float, float]] = {}

def short_trial_dirname(trial):
    return f"trial_{trial.trial_id}"


def _progress_print(done: int, total: int, running: int, prefix: str, last_ts: float, interval_s: float) -> float:
    now = time.time()
    if now - last_ts >= interval_s:
        msg = f"Progress: {done}/{total} completed | running: {running}"
        if prefix:
            msg = f"{prefix} {msg}"
        print(msg, flush=True)
        return now
    return last_ts

def define_search_space(trial: optuna.Trial) -> Dict[str, float]:
    """
    Optuna define-by-run search space with Tmin < Topt < Tmax.
    """
    params = {}

    tmin_lo, tmin_hi = _GLOBAL_RANGES["sporulation__Tmin"]
    topt_lo, topt_hi = _GLOBAL_RANGES["sporulation__Topt"]
    tmax_lo, tmax_hi = _GLOBAL_RANGES["sporulation__Tmax"]

    Tmin = trial.suggest_float("sporulation__Tmin", tmin_lo, tmin_hi)
    Topt = trial.suggest_float(
        "sporulation__Topt",
        max(topt_lo, Tmin + 0.1),
        topt_hi
    )
    Tmax = trial.suggest_float(
        "sporulation__Tmax",
        max(tmax_lo, Topt + 0.1),
        tmax_hi
    )

    params["sporulation__Tmin"] = Tmin
    params["sporulation__Topt"] = Topt
    params["sporulation__Tmax"] = Tmax

    for k, (lo, hi) in _GLOBAL_RANGES.items():
        if k in ("sporulation__Tmin", "sporulation__Topt", "sporulation__Tmax"):
            continue
        params[k] = trial.suggest_float(k, lo, hi)

    return params


def compute_mae(cfg: dict, rows: List[pd.Series]) -> float:
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
            err = calc_error(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            err = np.nan

        if np.isfinite(err):
            errors.append(abs(err))
        else:
            errors.append(PENALTY_DAYS)

    if not errors:
        return PENALTY_DAYS

    return float(np.mean(errors))


def build_folds(n: int, k: int) -> List[List[int]]:
    if k <= 1 or k > n:
        k = n
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(range(n)):
        folds[i % k].append(idx)
    return folds


@ray.remote
def eval_params(params: Dict[str, float], train_rows: List[pd.Series], val_rows: List[pd.Series]) -> Tuple[float, float]:
    model_cfg = apply_params(config, params)
    train_mae = compute_mae(model_cfg, train_rows)
    val_mae = compute_mae(model_cfg, val_rows)
    return train_mae, val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=6000)
    parser.add_argument("--n-startup", type=int, default=600)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--cpus-per-trial", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "result", "ray_results")
    )
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--no-quiet", dest="quiet", action="store_false")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--train-top-k", type=int, default=100)
    parser.add_argument("--train-top-pct", type=float, default=5.0)
    parser.add_argument("--kfold", type=int, default=3)
    args = parser.parse_args()

    ranges = build_param_ranges()
    _GLOBAL_RANGES.update(ranges)
    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))

    ray.init(num_cpus=args.n_jobs, include_dashboard=False)

    n_samples = len(rows)
    k = args.kfold if args.kfold > 0 else n_samples
    folds = build_folds(n_samples, k)

    fold_rows = []
    overall_best = {
        "best_train_params": None,
        "best_train_mae": None,
        "best_train_val_mae": None,
        "best_val_params": None,
        "best_val_mae": None,
        "best_val_train_mae": None,
    }
    fold_best_train_params = []
    fold_best_val_params = []
    fold_best_train_weights = []
    fold_best_val_weights = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = [i for i in range(n_samples) if i not in val_idx]
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        sampler = TPESampler(
            seed=args.seed + fold_id,
            multivariate=True,
            n_startup_trials=args.n_startup,
            warn_independent_sampling=False
        )

        study = optuna.create_study(
            sampler=sampler,
            direction="minimize"
        )
        max_conc = args.max_concurrent
        if max_conc <= 0:
            max_conc = max(1, args.n_jobs // max(1, args.cpus_per_trial))

        pending = []
        completed = 0
        last_ts = 0.0
        prefix = f"[Fold {fold_id}/{k}]"
        trial_rows = []

        while completed < args.n_trials:
            while len(pending) < max_conc and completed + len(pending) < args.n_trials:
                trial = study.ask()
                params = define_search_space(trial)
                fut = eval_params.remote(params, train_rows, val_rows)
                pending.append((trial, params, fut))

            done_ids, _ = ray.wait([f for _, _, f in pending], num_returns=1)
            for trial, params, fut in list(pending):
                if fut in done_ids:
                    train_mae, val_mae = ray.get(fut)
                    study.tell(trial, train_mae)
                    trial_rows.append({
                        "trial_number": trial.number,
                        "train_mae": float(train_mae),
                        "val_mae": float(val_mae),
                        **params
                    })
                    pending.remove((trial, params, fut))
                    completed += 1
                    break

            running = len(pending)
            last_ts = _progress_print(
                completed,
                args.n_trials,
                running,
                prefix,
                last_ts,
                args.progress_interval
            )

        df_trials = pd.DataFrame(trial_rows)
        fold_trials_path = os.path.join(RESULT_DIR, f"fold_{fold_id}_trials.csv")
        df_trials.to_csv(fold_trials_path, index=False)

        best_trial = study.best_trial
        best_train = float(best_trial.value)
        best_cfg = best_trial.params
        best_val = best_train
        for r in trial_rows:
            if r["trial_number"] == best_trial.number:
                best_val = float(r["val_mae"])
                break
        fold_best_train_params.append(best_cfg)
        fold_best_train_weights.append(float(best_val))

        pool = df_trials.dropna(subset=["train_mae", "val_mae"]).copy()
        pool = pool.sort_values("train_mae", ascending=True)
        if args.train_top_pct > 0:
            n_keep = max(1, int(len(pool) * args.train_top_pct / 100.0))
        else:
            n_keep = min(args.train_top_k, len(pool))
        pool = pool.head(n_keep)

        pool_out = os.path.join(RESULT_DIR, f"fold_{fold_id}_train_top_pool.csv")
        pool.to_csv(pool_out, index=False)

        best_val_row = None
        if not pool.empty:
            best_val_row = pool.sort_values("val_mae", ascending=True).iloc[0]
            fold_best_val_params.append({
                k: best_val_row[k]
                for k in best_val_row.index
                if k not in ("trial_number", "train_mae", "val_mae")
            })
            fold_best_val_weights.append(float(best_val_row["val_mae"]))

        fold_rows.append({
            "fold": fold_id,
            "train_mae_best_train": float(best_train),
            "val_mae_at_best_train": float(best_val),
            "train_mae_best_val": float(best_val_row["train_mae"]) if best_val_row is not None else np.nan,
            "val_mae_best_val": float(best_val_row["val_mae"]) if best_val_row is not None else np.nan,
        })

        if overall_best["best_train_mae"] is None or best_train < overall_best["best_train_mae"]:
            overall_best["best_train_mae"] = float(best_train)
            overall_best["best_train_val_mae"] = float(best_val)
            overall_best["best_train_params"] = best_cfg

        if best_val_row is not None:
            if overall_best["best_val_mae"] is None or best_val_row["val_mae"] < overall_best["best_val_mae"]:
                overall_best["best_val_mae"] = float(best_val_row["val_mae"])
                overall_best["best_val_train_mae"] = float(best_val_row["train_mae"])
                overall_best["best_val_params"] = {
                    k: best_val_row[k]
                    for k in best_val_row.index
                    if k not in ("trial_number", "train_mae", "val_mae")
                }

    summary_df = pd.DataFrame(fold_rows)
    summary_out = os.path.join(RESULT_DIR, "nested_loocv_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    def weighted_median(vals: List[float], weights: List[float]) -> float:
        pairs = sorted(zip(vals, weights), key=lambda x: x[0])
        total = sum(weights)
        if total <= 0:
            return float(np.median(vals))
        csum = 0.0
        for v, w in pairs:
            csum += w
            if csum >= 0.5 * total:
                return float(v)
        return float(pairs[-1][0])

    def aggregate_params(param_list: List[Dict[str, float]], fold_vals: List[float]) -> Dict[str, float]:
        if not param_list:
            return {}
        # lower val_mae -> higher weight
        weights = [1.0 / max(1e-6, v) for v in fold_vals]
        keys = sorted(param_list[0].keys())
        out = {}
        for k in keys:
            vals = [p[k] for p in param_list]
            out[k] = weighted_median(vals, weights)

        # enforce Tmin < Topt < Tmax by sorting only
        tmin = out.get("sporulation__Tmin")
        topt = out.get("sporulation__Topt")
        tmax = out.get("sporulation__Tmax")
        if tmin is not None and topt is not None and tmax is not None:
            tmin, topt, tmax = sorted([tmin, topt, tmax])
            out["sporulation__Tmin"] = tmin
            out["sporulation__Topt"] = topt
            out["sporulation__Tmax"] = tmax

        return out

    def aggregate_params_mean(param_list: List[Dict[str, float]], fold_vals: List[float]) -> Dict[str, float]:
        if not param_list:
            return {}
        weights = [1.0 / max(1e-6, v) for v in fold_vals]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0] * len(param_list)
            total_w = float(len(param_list))
        keys = sorted(param_list[0].keys())
        out = {}
        for k in keys:
            vals = [p[k] for p in param_list]
            out[k] = float(sum(v * w for v, w in zip(vals, weights)) / total_w)

        tmin = out.get("sporulation__Tmin")
        topt = out.get("sporulation__Topt")
        tmax = out.get("sporulation__Tmax")
        if tmin is not None and topt is not None and tmax is not None:
            tmin, topt, tmax = sorted([tmin, topt, tmax])
            out["sporulation__Tmin"] = tmin
            out["sporulation__Topt"] = topt
            out["sporulation__Tmax"] = tmax

        return out

    agg_train_params = aggregate_params(fold_best_train_params, fold_best_train_weights)
    agg_val_params = aggregate_params(fold_best_val_params, fold_best_val_weights)
    agg_train_params_mean = aggregate_params_mean(fold_best_train_params, fold_best_train_weights)
    agg_val_params_mean = aggregate_params_mean(fold_best_val_params, fold_best_val_weights)

    if args.train_top_pct > 0:
        tag = f"top{args.train_top_pct:g}pct"
    else:
        tag = f"topk{args.train_top_k}"

    def simple_mean_params(param_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not param_list:
            return {}
        keys = sorted(param_list[0].keys())
        out = {}
        for k in keys:
            vals = [p[k] for p in param_list]
            out[k] = float(np.mean(vals))
        tmin = out.get("sporulation__Tmin")
        topt = out.get("sporulation__Topt")
        tmax = out.get("sporulation__Tmax")
        if tmin is not None and topt is not None and tmax is not None:
            tmin, topt, tmax = sorted([tmin, topt, tmax])
            out["sporulation__Tmin"] = tmin
            out["sporulation__Topt"] = topt
            out["sporulation__Tmax"] = tmax
        return out

    # per-fold best params outputs
    for i, params in enumerate(fold_best_train_params, start=1):
        out_path = os.path.join(RESULT_DIR, f"best_params_train_fold{i}.json")
        with open(out_path, "w") as f:
            json.dump(params, f, indent=2)
    for i, params in enumerate(fold_best_val_params, start=1):
        out_path = os.path.join(RESULT_DIR, f"best_params_top{tag}_fold{i}.json")
        with open(out_path, "w") as f:
            json.dump(params, f, indent=2)

    # kfold detail and summary
    kfold_detail_train = summary_df[["fold", "train_mae_best_train", "val_mae_at_best_train"]].copy()
    kfold_detail_train.columns = ["fold", "train_mae", "val_mae"]
    kfold_detail_val = summary_df[["fold", "train_mae_best_val", "val_mae_best_val"]].copy()
    kfold_detail_val.columns = ["fold", "train_mae", "val_mae"]

    kfold_detail_train_path = os.path.join(RESULT_DIR, "kfold_detailed_train.csv")
    kfold_detail_val_path = os.path.join(RESULT_DIR, f"kfold_detailed_top{tag}.csv")
    kfold_detail_train.to_csv(kfold_detail_train_path, index=False)
    kfold_detail_val.to_csv(kfold_detail_val_path, index=False)

    kfold_summary_train = {
        "train_mae_mean": float(kfold_detail_train["train_mae"].mean()),
        "train_mae_std": float(kfold_detail_train["train_mae"].std()),
        "val_mae_mean": float(kfold_detail_train["val_mae"].mean()),
        "val_mae_std": float(kfold_detail_train["val_mae"].std()),
        "n_folds": int(len(kfold_detail_train)),
    }
    kfold_summary_val = {
        "train_mae_mean": float(kfold_detail_val["train_mae"].mean()),
        "train_mae_std": float(kfold_detail_val["train_mae"].std()),
        "val_mae_mean": float(kfold_detail_val["val_mae"].mean()),
        "val_mae_std": float(kfold_detail_val["val_mae"].std()),
        "n_folds": int(len(kfold_detail_val)),
    }

    kfold_summary_train_path = os.path.join(RESULT_DIR, "kfold_summary_train.json")
    kfold_summary_val_path = os.path.join(RESULT_DIR, f"kfold_summary_top{tag}.json")
    with open(kfold_summary_train_path, "w") as f:
        json.dump(kfold_summary_train, f, indent=2)
    with open(kfold_summary_val_path, "w") as f:
        json.dump(kfold_summary_val, f, indent=2)

    # aggregates (weighted median/mean/simple mean)
    train_agg = {
        "params_weighted_median": agg_train_params,
        "params_weighted_mean": agg_train_params_mean,
        "params_mean": simple_mean_params(fold_best_train_params),
    }
    val_agg = {
        "params_weighted_median": agg_val_params,
        "params_weighted_mean": agg_val_params_mean,
        "params_mean": simple_mean_params(fold_best_val_params),
    }
    train_agg_path = os.path.join(RESULT_DIR, "best_train_params_agg.json")
    val_agg_path = os.path.join(RESULT_DIR, f"best_val_params_top{tag}_agg.json")
    with open(train_agg_path, "w") as f:
        json.dump(train_agg, f, indent=2)
    with open(val_agg_path, "w") as f:
        json.dump(val_agg, f, indent=2)

    print(f"Saved per-fold summary: {summary_out}")
    print(f"Saved kfold detail train: {kfold_detail_train_path}")
    print(f"Saved kfold detail top{tag}: {kfold_detail_val_path}")
    print(f"Saved kfold summary train: {kfold_summary_train_path}")
    print(f"Saved kfold summary top{tag}: {kfold_summary_val_path}")
    print(f"Saved train aggregates: {train_agg_path}")
    print(f"Saved val aggregates ({tag}): {val_agg_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
