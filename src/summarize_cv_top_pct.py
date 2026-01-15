# -*- coding: utf-8 -*-
"""
Summarize CV results without re-simulation.

Reads fold_1_trials.csv, fold_2_trials.csv, fold_3_trials.csv and computes
mean train/val MAE across folds for the same parameter set. Then reports
top% summaries based on mean train MAE.
"""

import os
import sys
import glob
import json
from typing import List

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


def main():
    dfs = _load_folds()
    df_all = pd.concat(dfs, ignore_index=True)

    # Identify parameter columns (everything except these fields).
    ignore_cols = {"trial_number", "train_mae", "val_mae", "fold"}
    param_cols = [c for c in df_all.columns if c not in ignore_cols]
    if not param_cols:
        raise RuntimeError("No parameter columns found in fold trials.")

    group_cols = ["trial_number"] + param_cols
    grouped = df_all.groupby(group_cols, dropna=False, as_index=False).agg(
        train_mae_mean=("train_mae", "mean"),
        val_mae_mean=("val_mae", "mean"),
        n_folds=("fold", "nunique")
    )

    grouped = grouped.sort_values("train_mae_mean", ascending=True)

    out_table = os.path.join(RESULT_DIR, "cv_trial_means.csv")
    grouped.to_csv(out_table, index=False)

    top_pcts = [1, 2, 3, 5]
    summary_rows = []
    n_total = len(grouped)
    for pct in top_pcts:
        n_keep = max(1, int(n_total * pct / 100.0))
        top = grouped.head(n_keep)
        summary_rows.append({
            "top_pct": pct,
            "n_keep": int(n_keep),
            "train_mae_mean": float(top["train_mae_mean"].mean()),
            "val_mae_mean": float(top["val_mae_mean"].mean()),
            "train_mae_min": float(top["train_mae_mean"].min()),
            "val_mae_min": float(top["val_mae_mean"].min()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULT_DIR, "cv_top_pct_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved: {out_table}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
