#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate assimilation results across seeds.

Reads:
  result/calibration/seed_{seed}_assimilation/baseline_kfold_best_assimilation.csv
Writes:
  result/calibration/summarize_spore_assimilation_seeds.csv
"""

import argparse
import os
from typing import List

import pandas as pd


def load_seed_file(result_dir: str, seed: int) -> pd.DataFrame:
    path = os.path.join(
        result_dir,
        f"seed_{seed}_assimilation",
        "baseline_kfold_best_assimilation.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing seed file: {path}")
    df = pd.read_csv(path)
    df["seed"] = int(seed)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=os.path.join("result", "calibration"))
    parser.add_argument("--seed-list", type=str, default="42,123,2024")
    parser.add_argument("--out", type=str, default="summarize_spore_assimilation_seeds.csv")
    args = parser.parse_args()

    seeds: List[int] = [int(x) for x in args.seed_list.split(",") if x.strip()]
    if not seeds:
        raise ValueError("Empty seed list.")

    frames = []
    for seed in seeds:
        frames.append(load_seed_file(args.result_dir, seed))

    out_df = pd.concat(frames, ignore_index=True)
    if "fold" in out_df.columns:
        out_df = out_df.sort_values(["seed", "fold"]).reset_index(drop=True)
    else:
        out_df = out_df.sort_values(["seed"]).reset_index(drop=True)

    out_path = os.path.join(args.result_dir, args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
