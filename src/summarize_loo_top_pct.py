# -*- coding: utf-8 -*-
"""
Summarize LOO fold results for multiple top% thresholds.

For each fold:
  - Use loo_fold_{i}_trials.csv
  - Sort by train_mae, keep top%
  - Record mean train_mae and mean val_mae in that pool

Outputs a summary table across folds for top% in [1,2,3,4,5].
"""

import os
import glob
from typing import List

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")


def _load_folds() -> List[pd.DataFrame]:
    pattern = os.path.join(RESULT_DIR, "loo_fold_*_trials.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No LOO trial files found: {pattern}")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["fold"] = int(os.path.basename(fp).split("_")[2])
        dfs.append(df)
    return dfs


def main():
    dfs = _load_folds()
    top_pcts = [1, 2, 3, 4, 5]

    rows = []
    for pct in top_pcts:
        fold_rows = []
        for df in dfs:
            fold_id = int(df["fold"].iloc[0])
            pool = df.dropna(subset=["train_mae", "val_mae"]).copy()
            pool = pool.sort_values("train_mae", ascending=True)
            n_keep = max(1, int(len(pool) * pct / 100.0))
            top = pool.head(n_keep)
            best_row = top.sort_values("val_mae", ascending=True).iloc[0]
            fold_rows.append({
                "fold": fold_id,
                "top_pct": pct,
                "n_keep": int(n_keep),
                "train_mae_mean": float(best_row["train_mae"]),
                "val_mae_mean": float(best_row["val_mae"]),
            })
        df_fold = pd.DataFrame(fold_rows)
        rows.append({
            "top_pct": pct,
            "train_mae_mean": float(df_fold["train_mae_mean"].mean()),
            "val_mae_mean": float(df_fold["val_mae_mean"].mean()),
            "train_mae_std": float(df_fold["train_mae_mean"].std()),
            "val_mae_std": float(df_fold["val_mae_mean"].std()),
        })

        fold_out = os.path.join(RESULT_DIR, f"loo_top{pct}pct_by_fold.csv")
        df_fold.to_csv(fold_out, index=False)

    summary = pd.DataFrame(rows)
    summary_out = os.path.join(RESULT_DIR, "loo_top_pct_summary.csv")
    summary.to_csv(summary_out, index=False)

    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
