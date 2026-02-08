#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize calibration outputs across seeds for baseline and spore models.
Writes per-seed summaries, cross-seed aggregates, and baseline-vs-spore comparisons.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def _list_seed_dirs(root: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(root):
        return []
    seeds = []
    for name in os.listdir(root):
        m = SEED_DIR_RE.match(name)
        if not m:
            continue
        seed = int(m.group(1))
        seeds.append((seed, os.path.join(root, name)))
    return sorted(seeds, key=lambda x: x[0])


def _read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _flatten_params(info: Dict) -> Dict:
    params = info.get("params") or info.get("best_params") or {}
    out = {}
    for k, v in params.items():
        out[k] = v
    return out


def _read_seed_summary(seed_dir: str) -> Dict:
    summary_json = os.path.join(seed_dir, "baseline_kfold_summary.json")
    summary_csv = os.path.join(seed_dir, "baseline_kfold_summary.csv")
    if os.path.exists(summary_json):
        return _read_json(summary_json)
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        return df.mean(numeric_only=True).to_dict()
    return {}


def _read_seed_best_global(seed_dir: str) -> Dict:
    best_path = os.path.join(seed_dir, "baseline_kfold_best_global.json")
    if os.path.exists(best_path):
        return _read_json(best_path)
    return {}


def _seed_summary_table(root: str) -> pd.DataFrame:
    rows = []
    for seed, sdir in _list_seed_dirs(root):
        summary = _read_seed_summary(sdir)
        if not summary:
            continue
        summary["seed"] = seed
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("seed")


def _seed_best_table(root: str) -> pd.DataFrame:
    rows = []
    for seed, sdir in _list_seed_dirs(root):
        info = _read_seed_best_global(sdir)
        if not info:
            continue
        row = {k: v for k, v in info.items() if k != "params"}
        row["seed"] = seed
        row.update(_flatten_params(info))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("seed")


def _aggregate_numeric(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    if df.empty:
        return out
    for col in df.columns:
        if col == "seed":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std(ddof=0))
    out["n_seeds"] = int(len(df))
    return out


def _write_json(path: str, obj: Dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _compare_baseline_spore(b_df: pd.DataFrame, s_df: pd.DataFrame) -> pd.DataFrame:
    if b_df.empty or s_df.empty:
        return pd.DataFrame()
    merged = b_df.merge(s_df, on="seed", suffixes=("_baseline", "_spore"))
    for metric in ["val_mae_mean", "train_mae_mean", "val_rmse", "train_rmse"]:
        b_col = f"{metric}_baseline"
        s_col = f"{metric}_spore"
        if b_col in merged.columns and s_col in merged.columns:
            merged[f"{metric}_delta"] = merged[s_col] - merged[b_col]
            merged[f"{metric}_pct"] = (merged[s_col] - merged[b_col]) / merged[b_col] * 100.0
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=str, default="result/calibration")
    parser.add_argument("--spore-dir", type=str, default="result/calibration_spore_infthr")
    parser.add_argument("--out-dir", type=str, default="result/calibration_sum")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    baseline_seed_summary = _seed_summary_table(args.baseline_dir)
    spore_seed_summary = _seed_summary_table(args.spore_dir)

    baseline_best = _seed_best_table(args.baseline_dir)
    spore_best = _seed_best_table(args.spore_dir)

    if not baseline_seed_summary.empty:
        baseline_seed_summary.to_csv(os.path.join(args.out_dir, "baseline_seed_summary.csv"), index=False)
        _write_json(
            os.path.join(args.out_dir, "baseline_seed_agg.json"),
            _aggregate_numeric(baseline_seed_summary),
        )
    if not spore_seed_summary.empty:
        spore_seed_summary.to_csv(os.path.join(args.out_dir, "spore_seed_summary.csv"), index=False)
        _write_json(
            os.path.join(args.out_dir, "spore_seed_agg.json"),
            _aggregate_numeric(spore_seed_summary),
        )

    if not baseline_best.empty:
        baseline_best.to_csv(os.path.join(args.out_dir, "baseline_seed_best_global.csv"), index=False)
    if not spore_best.empty:
        spore_best.to_csv(os.path.join(args.out_dir, "spore_seed_best_global.csv"), index=False)

    compare = _compare_baseline_spore(baseline_seed_summary, spore_seed_summary)
    if not compare.empty:
        compare.to_csv(os.path.join(args.out_dir, "baseline_vs_spore_seed_compare.csv"), index=False)
        _write_json(
            os.path.join(args.out_dir, "baseline_vs_spore_agg.json"),
            _aggregate_numeric(compare),
        )

    print(f"Saved summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
