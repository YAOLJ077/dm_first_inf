# -*- coding: utf-8 -*-
"""
Postprocess evaluation for aggregated parameters.

Method A:
  - Use aggregated params (weighted mean) and compute k-fold val MAE mean.
Method B:
  - From per-fold top% pools, evaluate each candidate on full dataset
    and pick the global best MAE.
"""

import os
import sys
import json
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
OUT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")

from model.crop_season import CropSeasonFirst
from model.config import config as BASE_CONFIG

try:
    import ray  # type: ignore
except Exception:
    ray = None


def load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df.reset_index(drop=True)


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
        return float(doy_end - doy_actual)
    if doy_start > doy_actual:
        return float(doy_start - doy_actual)
    return 0.0


def apply_params(base_cfg: dict, params: Dict[str, float]) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    for k, v in params.items():
        g, sub = k.split("__")
        cfg[g][sub] = float(v)
    return cfg


def _row_get(row, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key)


def compute_mae(rows: List, cfg: dict, penalty: float = 90.0) -> float:
    errs = []
    for row in rows:
        plot = _row_get(row, "plot")
        variety = _row_get(row, "variety")
        date = _row_get(row, "first_infection_date")
        key = f"{plot}_{variety}_{pd.to_datetime(date).year}"
        rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
        if not os.path.exists(rb_path):
            continue
        try:
            with open(rb_path, "r") as f:
                rb = json.load(f)
            cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            e = calc_error(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            e = np.nan
        if np.isfinite(e):
            errs.append(abs(e))
        else:
            errs.append(penalty)
    if not errs:
        return penalty
    return float(np.mean(errs))


def detect_top_tag() -> str:
    candidates = []
    for pat in ("best_val_params_top*pct_agg.json", "kfold_detailed_top*pct.csv", "kfold_summary_top*pct.json"):
        for path in glob.glob(os.path.join(OUT_DIR, pat)):
            base = os.path.basename(path)
            if "top" in base and "pct" in base:
                tag = base.split("top", 1)[1].split("pct", 1)[0]
                candidates.append(f"{tag}pct")
    if not candidates:
        return "5pct"
    # Prefer the largest percentage if multiple are present.
    def _tag_val(t: str) -> float:
        try:
            return float(t.replace("pct", ""))
        except ValueError:
            return -1.0
    candidates = sorted(set(candidates), key=_tag_val, reverse=True)
    return candidates[0]


def method_a_weighted_mean(kfold: int = 3) -> Dict[str, float]:
    agg_path = os.path.join(OUT_DIR, "best_train_params_agg.json")
    if not os.path.exists(agg_path):
        return {"status": "missing", "reason": f"Missing agg params: {agg_path}"}
    with open(agg_path) as f:
        agg = json.load(f)
    params = agg.get("params_weighted_mean") or agg.get("params_weighted_median") or agg.get("params_mean")
    if not params:
        return {"status": "missing", "reason": f"No params_* field found in {agg_path}"}

    df = load_actual()
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    fold_mae = []
    for tr, va in kf.split(df):
        val_rows = [r for r in df.iloc[va].itertuples(index=False)]
        cfg = apply_params(BASE_CONFIG, params)
        fold_mae.append(compute_mae(val_rows, cfg))
    return {
        "status": "ok",
        "val_mae_mean": float(np.mean(fold_mae)),
        "val_mae_std": float(np.std(fold_mae)),
    }


def _params_key(params: Dict[str, float]) -> str:
    parts = [f"{k}={params[k]:.6g}" for k in sorted(params.keys())]
    return "|".join(parts)


def method_b_top_pool_global(tag: str) -> Dict[str, float]:
    cv_pool = os.path.join(OUT_DIR, "cv_train_top_pool.csv")
    if os.path.exists(cv_pool):
        files = [cv_pool]
    else:
        pattern = os.path.join(OUT_DIR, f"fold_*_train_top_pool.csv")
        files = sorted([p for p in glob.glob(pattern)])
        if not files:
            return {"status": "missing", "reason": f"No pool files found: {pattern}"}

    df_all = []
    for fp in files:
        df = pd.read_csv(fp)
        df_all.append(df)
    pool = pd.concat(df_all, ignore_index=True)
    if pool.empty:
        return {"status": "missing", "reason": "Top pool is empty."}

    df = load_actual()
    rows = [r._asdict() for r in df.itertuples(index=False)]

    best = {"mae": 1e9, "params": None}
    cache: Dict[str, float] = {}
    unique = {}
    for _, row in pool.iterrows():
        params = {}
        for k in row.index:
            if k in ("trial_number", "train_mae", "val_mae"):
                continue
            key = k.replace("config/", "")
            params[key] = row[k]
        cache_key = _params_key(params)
        if cache_key not in unique:
            unique[cache_key] = params

    total = int(len(unique))
    if total == 0:
        return {"status": "missing", "reason": "Top pool is empty."}

    if ray is not None:
        if not ray.is_initialized():
            ray.init(num_cpus=128, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        @ray.remote
        def eval_params_remote(params, rows_dicts):
            cfg = apply_params(BASE_CONFIG, params)
            return compute_mae(rows_dicts, cfg)

        futures = {}
        for k, params in unique.items():
            futures[k] = eval_params_remote.remote(params, rows)

        done = 0
        while futures:
            done_ids, _ = ray.wait(list(futures.values()), num_returns=1)
            finished = done_ids[0]
            for k, fut in list(futures.items()):
                if fut == finished:
                    mae = ray.get(fut)
                    cache[k] = mae
                    params = unique[k]
                    if mae < best["mae"]:
                        best["mae"] = mae
                        best["params"] = params
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"[method_b] Progress: {done}/{total} | best_mae={best['mae']:.3f}", flush=True)
                    del futures[k]
                    break
        ray.shutdown()
    else:
        done = 0
        for k, params in unique.items():
            cfg = apply_params(BASE_CONFIG, params)
            mae = compute_mae(rows, cfg)
            cache[k] = mae
            if mae < best["mae"]:
                best["mae"] = mae
                best["params"] = params
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[method_b] Progress: {done}/{total} | best_mae={best['mae']:.3f}", flush=True)

    return {
        "status": "ok",
        "tag": tag,
        "global_mae": float(best["mae"]),
        "params": best["params"],
    }


def main():
    out = {}
    out["method_a_weighted_mean"] = method_a_weighted_mean(kfold=3)
    tag = detect_top_tag()
    out["method_b_top_pool_global"] = method_b_top_pool_global(tag)

    out_path = os.path.join(OUT_DIR, "postprocess_agg_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
