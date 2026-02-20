#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _load_bounds(best_json: Path) -> Dict[str, Tuple[float, float]]:
    with open(best_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    bounds = obj.get("param_bounds", {})
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in bounds.items():
        if isinstance(v, list) and len(v) == 2:
            out[k] = (float(v[0]), float(v[1]))
    return out


def _suggest_bound(
    lo: float,
    hi: float,
    p_lower: float,
    p_upper: float,
    best_v: float,
    expand_ratio: float,
    pressure_thr: float,
    best_eps: float,
) -> Tuple[float, float, str]:
    width = hi - lo
    if width <= 0:
        return lo, hi, "keep"
    action = "keep"
    near_lower = (best_v - lo) / width <= best_eps
    near_upper = (hi - best_v) / width <= best_eps

    new_lo, new_hi = lo, hi
    if p_lower >= pressure_thr or near_lower:
        new_lo = lo - expand_ratio * width
        action = "expand_lower"
    if p_upper >= pressure_thr or near_upper:
        new_hi = hi + expand_ratio * width
        action = "expand_upper" if action == "keep" else "expand_both"
    return new_lo, new_hi, action


def main() -> None:
    p = argparse.ArgumentParser(description="Check boundary pressure from spore_proxy_trials and suggest new bounds.")
    p.add_argument(
        "--trials",
        default="result/historical_inputs/spore_pressure_calibration/spore_proxy_trials.csv",
    )
    p.add_argument(
        "--best-json",
        default="result/historical_inputs/spore_pressure_calibration/spore_proxy_best.json",
    )
    p.add_argument("--top-frac", type=float, default=0.10, help="Top fraction (by smallest loss) used to estimate boundary pressure.")
    p.add_argument("--edge-eps", type=float, default=0.05, help="Near-boundary width ratio.")
    p.add_argument("--pressure-thr", type=float, default=0.20, help="Pressure threshold to trigger expansion.")
    p.add_argument("--expand-ratio", type=float, default=0.25, help="Expand bound by this fraction of current width.")
    p.add_argument(
        "--out-json",
        default="result/historical_inputs/spore_pressure_calibration/spore_param_bounds_suggested.json",
    )
    args = p.parse_args()

    trials_path = Path(args.trials)
    best_json = Path(args.best_json)
    if not trials_path.exists():
        raise FileNotFoundError(trials_path)
    if not best_json.exists():
        raise FileNotFoundError(best_json)

    df = pd.read_csv(trials_path)
    if "value" not in df.columns:
        raise KeyError("trials csv must contain 'value' column.")
    bounds = _load_bounds(best_json)
    with open(best_json, "r", encoding="utf-8") as f:
        best_obj = json.load(f)
    best_params = best_obj.get("best_params", {})

    n_top = max(1, int(np.ceil(len(df) * float(args.top_frac))))
    top = df.sort_values("value", ascending=True).head(n_top).copy()

    rows = []
    suggested = {}
    for k, (lo, hi) in bounds.items():
        if k not in top.columns:
            continue
        col = pd.to_numeric(top[k], errors="coerce").dropna()
        if col.empty:
            continue
        width = hi - lo
        if width <= 0:
            continue
        x = (col.values - lo) / width
        p_lower = float(np.mean(x <= float(args.edge_eps)))
        p_upper = float(np.mean(x >= 1.0 - float(args.edge_eps)))
        best_v = float(best_params.get(k, np.nan))
        new_lo, new_hi, action = _suggest_bound(
            lo=lo,
            hi=hi,
            p_lower=p_lower,
            p_upper=p_upper,
            best_v=best_v,
            expand_ratio=float(args.expand_ratio),
            pressure_thr=float(args.pressure_thr),
            best_eps=float(args.edge_eps),
        )
        if k == "roll_window":
            new_lo = int(np.floor(new_lo))
            new_hi = int(np.ceil(new_hi))
        suggested[k] = [new_lo, new_hi]
        rows.append(
            {
                "param": k,
                "lo": lo,
                "hi": hi,
                "best": best_v,
                "p_near_lower": p_lower,
                "p_near_upper": p_upper,
                "action": action,
                "suggest_lo": new_lo,
                "suggest_hi": new_hi,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["action", "param"]).reset_index(drop=True)
    print(out_df.to_string(index=False))

    out_obj = {
        "source_trials": str(trials_path),
        "source_best_json": str(best_json),
        "top_frac": float(args.top_frac),
        "edge_eps": float(args.edge_eps),
        "pressure_thr": float(args.pressure_thr),
        "expand_ratio": float(args.expand_ratio),
        "suggested_bounds": suggested,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

