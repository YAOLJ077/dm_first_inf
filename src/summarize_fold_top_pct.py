# -*- coding: utf-8 -*-
"""
Summarize top-percent train_mae pools per fold.
"""

import os
import glob
import pandas as pd


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration")
OUT_PATH = os.path.join(RESULT_DIR, "fold_top_pct_summary.csv")

TOP_PCTS = [1, 3, 5]


def main():
    pattern = os.path.join(RESULT_DIR, "fold_*_trials.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No fold trials found: {pattern}")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        df = df.dropna(subset=["train_mae", "val_mae"]).copy()
        if df.empty:
            continue

        fold_id = os.path.basename(fp).split("_")[1]
        df = df.sort_values("train_mae", ascending=True)

        for pct in TOP_PCTS:
            n_keep = max(1, int(len(df) * pct / 100.0))
            pool = df.head(n_keep)
            best_val_row = pool.sort_values("val_mae", ascending=True).iloc[0]
            rows.append({
                "fold": fold_id,
                "top_pct": pct,
                "pool_size": n_keep,
                "best_train_mae": float(best_val_row["train_mae"]),
                "best_val_mae": float(best_val_row["val_mae"]),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
