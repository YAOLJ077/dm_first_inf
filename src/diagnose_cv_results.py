# -*- coding: utf-8 -*-
"""
Diagnostics for CV trial outputs.

Reads fold_*_trials.csv and cv_train_top_pool.csv (if present) and reports:
- fraction of rows with train==val
- fraction of rows at penalty value
- basic stats for train/val columns
- size of top pool and unique parameter count
"""

import os
import glob
import json
from typing import Dict, List

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")


def _load_folds() -> List[pd.DataFrame]:
    pattern = os.path.join(RESULT_DIR, "fold_*_trials.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No fold trial files found: {pattern}")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["fold"] = int(os.path.basename(fp).split("_")[1])
        dfs.append(df)
    return dfs


def _basic_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def main():
    dfs = _load_folds()
    df_all = pd.concat(dfs, ignore_index=True)

    train_eq_val = df_all["train_mae"] == df_all["val_mae"]
    penalty = 90.0
    train_is_pen = df_all["train_mae"] == penalty
    val_is_pen = df_all["val_mae"] == penalty
    both_pen = train_is_pen & val_is_pen

    diag = {
        "rows_total": int(len(df_all)),
        "train_eq_val_frac": float(train_eq_val.mean()),
        "train_penalty_frac": float(train_is_pen.mean()),
        "val_penalty_frac": float(val_is_pen.mean()),
        "both_penalty_frac": float(both_pen.mean()),
        "train_stats": _basic_stats(df_all["train_mae"]),
        "val_stats": _basic_stats(df_all["val_mae"]),
        "fold_counts": df_all["fold"].value_counts().sort_index().to_dict(),
    }

    cv_pool_path = os.path.join(RESULT_DIR, "cv_train_top_pool.csv")
    if os.path.exists(cv_pool_path):
        pool = pd.read_csv(cv_pool_path)
        ignore_cols = {"trial_number", "avg_train_mae", "avg_val_mae"}
        param_cols = [c for c in pool.columns if c not in ignore_cols]
        uniq = pool[param_cols].drop_duplicates().shape[0] if param_cols else 0
        diag["cv_pool"] = {
            "rows": int(len(pool)),
            "unique_params": int(uniq),
            "avg_train_stats": _basic_stats(pool["avg_train_mae"]),
            "avg_val_stats": _basic_stats(pool["avg_val_mae"]),
        }
    else:
        diag["cv_pool"] = {"rows": 0}

    out_path = os.path.join(RESULT_DIR, "cv_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2)

    print(f"Saved: {out_path}")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
