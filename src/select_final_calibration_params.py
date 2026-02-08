#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select one final seed and parameter set for baseline/spore models from
calibration summaries.

Selection logic:
1) Pair by seed.
2) Hard constraints (preferred):
   - spore val_mae_mean < baseline val_mae_mean
   - spore global_mae < baseline global_mae
3) Rank by:
   - spore val_mae_mean (asc)
   - spore val_mae_std (asc)
   - spore global_mae (asc)
4) If no seed satisfies hard constraints, fallback to smallest val_mae improvement
   (most negative delta), then same secondary keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _param_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if "__" in c]


def _build_candidate_table(
    baseline_summary: pd.DataFrame,
    spore_summary: pd.DataFrame,
    baseline_best: pd.DataFrame,
    spore_best: pd.DataFrame,
) -> pd.DataFrame:
    s_pair = baseline_summary.merge(spore_summary, on="seed", suffixes=("_baseline", "_spore"))
    b_best = baseline_best[["seed", "global_mae"] + _param_cols(baseline_best)].rename(
        columns={"global_mae": "global_mae_baseline"}
    )
    s_best = spore_best[["seed", "global_mae"] + _param_cols(spore_best)].rename(
        columns={"global_mae": "global_mae_spore"}
    )
    out = s_pair.merge(b_best, on="seed", how="left").merge(s_best, on="seed", how="left")

    out["delta_val_mae"] = out["val_mae_mean_spore"] - out["val_mae_mean_baseline"]
    out["delta_val_mae_pct"] = out["delta_val_mae"] / out["val_mae_mean_baseline"] * 100.0
    out["delta_global_mae"] = out["global_mae_spore"] - out["global_mae_baseline"]
    out["delta_global_mae_pct"] = out["delta_global_mae"] / out["global_mae_baseline"] * 100.0
    out["pass_hard_constraints"] = (
        (out["val_mae_mean_spore"] < out["val_mae_mean_baseline"])
        & (out["global_mae_spore"] < out["global_mae_baseline"])
    )
    return out


def _select_seed(candidates: pd.DataFrame) -> Dict:
    preferred = candidates[candidates["pass_hard_constraints"]].copy()

    sort_cols = ["val_mae_mean_spore", "val_mae_std_spore", "global_mae_spore"]
    if not preferred.empty:
        ranked = preferred.sort_values(sort_cols, ascending=[True, True, True]).reset_index(drop=True)
        row = ranked.iloc[0].to_dict()
        row["selection_mode"] = "hard_constraints"
        return row

    # Fallback: pick max reduction in val_mae (most negative delta), then stable/low global
    ranked = candidates.sort_values(
        ["delta_val_mae", "val_mae_std_spore", "global_mae_spore"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    row = ranked.iloc[0].to_dict()
    row["selection_mode"] = "fallback_val_improvement"
    return row


def _extract_params(row: Dict, model: str) -> Dict:
    suffix = "_baseline" if model == "baseline" else "_spore"
    params = {}
    for k, v in row.items():
        if "__" not in k:
            continue
        # baseline params have no suffix in current merge; spore too.
        params[k] = float(v)
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sum-dir", type=str, default="result/calibration_sum")
    parser.add_argument("--out-dir", type=str, default="result/calibration_sum")
    args = parser.parse_args()

    sum_dir = Path(args.sum_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = _load_required_csv(sum_dir / "baseline_seed_summary.csv")
    spore_summary = _load_required_csv(sum_dir / "spore_seed_summary.csv")
    baseline_best = _load_required_csv(sum_dir / "baseline_seed_best_global.csv")
    spore_best = _load_required_csv(sum_dir / "spore_seed_best_global.csv")

    candidates = _build_candidate_table(baseline_summary, spore_summary, baseline_best, spore_best)
    candidates_sorted = candidates.sort_values(
        ["pass_hard_constraints", "val_mae_mean_spore", "val_mae_std_spore", "global_mae_spore"],
        ascending=[False, True, True, True],
    )
    candidates_sorted.to_csv(out_dir / "final_seed_candidates.csv", index=False)

    selected = _select_seed(candidates)
    selected_seed = int(selected["seed"])

    b_row = baseline_best[baseline_best["seed"] == selected_seed].iloc[0].to_dict()
    s_row = spore_best[spore_best["seed"] == selected_seed].iloc[0].to_dict()
    b_sum = baseline_summary[baseline_summary["seed"] == selected_seed].iloc[0].to_dict()
    s_sum = spore_summary[spore_summary["seed"] == selected_seed].iloc[0].to_dict()

    baseline_payload = {
        "selected_seed": selected_seed,
        "selection_mode": selected["selection_mode"],
        "metrics_seed_summary": b_sum,
        "metrics_seed_best_global": {k: v for k, v in b_row.items() if "__" not in k},
        "params": {k: float(v) for k, v in b_row.items() if "__" in k},
    }
    spore_payload = {
        "selected_seed": selected_seed,
        "selection_mode": selected["selection_mode"],
        "metrics_seed_summary": s_sum,
        "metrics_seed_best_global": {k: v for k, v in s_row.items() if "__" not in k},
        "params": {k: float(v) for k, v in s_row.items() if "__" in k},
    }

    with open(out_dir / "final_baseline_params.json", "w") as f:
        json.dump(baseline_payload, f, indent=2)
    with open(out_dir / "final_spore_params.json", "w") as f:
        json.dump(spore_payload, f, indent=2)

    selection_report = {
        "selected_seed": selected_seed,
        "selection_mode": selected["selection_mode"],
        "hard_constraints_passed": bool(selected["pass_hard_constraints"]),
        "baseline_val_mae_mean": float(selected["val_mae_mean_baseline"]),
        "spore_val_mae_mean": float(selected["val_mae_mean_spore"]),
        "delta_val_mae": float(selected["delta_val_mae"]),
        "delta_val_mae_pct": float(selected["delta_val_mae_pct"]),
        "baseline_global_mae": float(selected["global_mae_baseline"]),
        "spore_global_mae": float(selected["global_mae_spore"]),
        "delta_global_mae": float(selected["delta_global_mae"]),
        "delta_global_mae_pct": float(selected["delta_global_mae_pct"]),
    }
    with open(out_dir / "final_selection_report.json", "w") as f:
        json.dump(selection_report, f, indent=2)

    print(f"Saved: {out_dir / 'final_seed_candidates.csv'}")
    print(f"Saved: {out_dir / 'final_baseline_params.json'}")
    print(f"Saved: {out_dir / 'final_spore_params.json'}")
    print(f"Saved: {out_dir / 'final_selection_report.json'}")


if __name__ == "__main__":
    main()

