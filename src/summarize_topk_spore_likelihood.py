#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize top-k selection for spore-likelihood runs.

Selection rule:
1) Top-k by train_loss.
2) In top-k, pick min val_mae with constraint val_mae > train_mae.
   If none, pick min |val_mae - train_mae|.
3) If multiple share the same val_mae, tie-break by train_mae (asc),
   then train_loss (asc).
"""

import argparse
import math
import os
from typing import List

import pandas as pd


def _load_fold_csv(result_dir: str, fold: int) -> pd.DataFrame:
    path = os.path.join(result_dir, f"baseline_kfold_fold_{fold}_trials_spore.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing fold file: {path}")
    return pd.read_csv(path)


def _topk_by_train_loss(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty trials dataframe.")
    df_sorted = df.sort_values("train_loss").reset_index(drop=True)
    n_top = max(1, int(math.ceil(len(df_sorted) * pct / 100.0)))
    return df_sorted.iloc[:n_top].copy()


def select_topk_val_best_constrained(df: pd.DataFrame, pct: float) -> pd.Series:
    top = _topk_by_train_loss(df, pct)
    valid = top[top["val_mae"] > top["train_mae"]]
    if not valid.empty:
        candidates = valid.sort_values(["val_mae", "train_mae", "train_loss"])
    else:
        diff = (top["val_mae"] - top["train_mae"]).abs()
        candidates = top.iloc[diff.values.argsort()]
    best = candidates.iloc[0]
    return best


def summarize_topk(result_dir: str, folds: List[int], topk_list: List[float], seed: int) -> pd.DataFrame:
    rows = []
    for fold in folds:
        df = _load_fold_csv(result_dir, fold)
        for pct in topk_list:
            best = select_topk_val_best_constrained(df, pct)
            rows.append({
                "seed": seed,
                "fold": fold,
                "pct": pct,
                "n_total": int(len(df)),
                "n_top": int(max(1, math.ceil(len(df) * pct / 100.0))),
                "train_mae": float(best["train_mae"]),
                "val_mae": float(best["val_mae"]),
                "train_loss": float(best["train_loss"]),
                "train_spore_nll": float(best["train_spore_nll"]),
                "val_spore_nll": float(best["val_spore_nll"]),
                "train_spore_consistency": float(best.get("train_spore_consistency", float("nan"))),
                "val_spore_consistency": float(best.get("val_spore_consistency", float("nan"))),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=os.path.join("result", "calibration"))
    parser.add_argument("--seed-list", type=str, default="42,123,2024")
    parser.add_argument("--folds", type=str, default="1,2,3")
    parser.add_argument("--topk", type=str, default="1,2,3,5,10")
    parser.add_argument("--out", type=str, default="summarize_topk_spore_likelihood.csv")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    topk_list = [float(x) for x in args.topk.split(",") if x.strip()]

    all_rows = []
    out_root = os.path.join(args.result_dir, "summarize_topk_seed_fold")
    os.makedirs(out_root, exist_ok=True)
    for seed in seeds:
        seed_dir = os.path.join(args.result_dir, f"seed_{seed}_spore")
        df = summarize_topk(seed_dir, folds, topk_list, seed)
        out_path = os.path.join(out_root, f"seed_{seed}_{args.out}")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        all_rows.append(df)

        agg = (
            df.groupby("pct")[["train_mae", "val_mae", "train_loss",
                               "train_spore_consistency", "val_spore_consistency"]]
            .mean()
            .reset_index()
        )
        agg_path = os.path.join(out_root, f"seed_{seed}_summarize_topk_spore_likelihood_agg.csv")
        agg.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path}")

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_out = os.path.join(out_root, "summarize_topk_spore_likelihood_all_seeds.csv")
        all_df.to_csv(all_out, index=False)
        print(f"Saved: {all_out}")

        all_agg = (
            all_df.groupby(["seed", "pct"])[["train_mae", "val_mae", "train_loss",
                                             "train_spore_consistency", "val_spore_consistency"]]
            .mean()
            .reset_index()
        )
        all_agg_out = os.path.join(out_root, "summarize_topk_spore_likelihood_all_seeds_agg.csv")
        all_agg.to_csv(all_agg_out, index=False)
        print(f"Saved: {all_agg_out}")


if __name__ == "__main__":
    main()
