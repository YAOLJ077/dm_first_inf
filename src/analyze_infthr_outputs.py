import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ast
import re
from typing import Optional


def _load_base_ranges() -> Dict[str, Tuple[float, float]]:
    src_dir = Path(__file__).resolve().parent
    target = src_dir / "bayes_calibrate_first_kfold_spore_infthr.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing: {target}")
    text = target.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "BASE_RANGES":
                    return ast.literal_eval(node.value)
    raise ValueError("Could not parse BASE_RANGES from bayes_calibrate_first_kfold_spore_infthr.py")


def _load_fold_best_params(seed_dir: Path) -> List[Dict]:
    fold_dirs = sorted(seed_dir.glob("baseline_fold_*"))
    out = []
    for fdir in fold_dirs:
        best_path = fdir / "baseline_kfold_best_params.json"
        if not best_path.exists():
            continue
        with open(best_path, "r") as f:
            data = json.load(f)
        params = data.get("params", {})
        params["_fold_dir"] = fdir.name
        params["_seed_dir"] = seed_dir.name
        out.append(params)
    return out


def _load_seed_summary(seed_dir: Path) -> pd.DataFrame | None:
    path = seed_dir / "baseline_kfold_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["seed_dir"] = seed_dir.name
    return df


def _collect_results(result_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_dirs = sorted([p for p in result_dir.glob("seed_*") if p.is_dir()])
    fold_params = []
    summaries = []
    for seed_dir in seed_dirs:
        fold_params.extend(_load_fold_best_params(seed_dir))
        s = _load_seed_summary(seed_dir)
        if s is not None:
            summaries.append(s)
    params_df = pd.DataFrame(fold_params)
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return params_df, summary_df


def _boundary_analysis(params_df: pd.DataFrame, base_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for k, (lo, hi) in base_ranges.items():
        if k not in params_df.columns:
            continue
        vals = params_df[k].astype(float)
        if vals.empty:
            continue
        q10, q50, q90 = vals.quantile([0.1, 0.5, 0.9]).tolist()
        vmin, vmax = float(vals.min()), float(vals.max())
        span = hi - lo
        lo_margin = (q10 - lo) / span if span > 0 else np.nan
        hi_margin = (hi - q90) / span if span > 0 else np.nan
        rows.append({
            "param": k,
            "lo": lo,
            "hi": hi,
            "min": vmin,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "max": vmax,
            "lo_margin": lo_margin,
            "hi_margin": hi_margin,
        })
    return pd.DataFrame(rows)


def _suggest_range(row: pd.Series) -> str:
    lo_margin = row["lo_margin"]
    hi_margin = row["hi_margin"]
    if pd.isna(lo_margin) or pd.isna(hi_margin):
        return ""
    # small margin => near boundary
    if lo_margin < 0.05 and hi_margin < 0.05:
        return "expand both"
    if lo_margin < 0.05:
        return "expand lower"
    if hi_margin < 0.05:
        return "expand upper"
    # values well inside
    if lo_margin > 0.2 and hi_margin > 0.2:
        return "shrink both"
    return "keep"


def _window_compare(stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, df in stats.items():
        if df.empty or "val_mae" not in df.columns:
            continue
        rows.append({
            "label": label,
            "val_mae_mean": float(df["val_mae"].mean()),
            "val_mae_median": float(df["val_mae"].median()),
            "val_mae_std": float(df["val_mae"].std(ddof=0)),
            "train_mae_mean": float(df["train_mae"].mean()) if "train_mae" in df else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, default="result/calibration_spore_infthr")
    parser.add_argument(
        "--compare-dir",
        type=str,
        default="",
        help="Optional second result dir (e.g., window=7) to compare"
    )
    parser.add_argument("--out", type=str, default="result/analysis_infthr_bounds.csv")
    args = parser.parse_args()

    res_dirs = {"w3": Path(args.result_dir)}
    if args.compare_dir.strip():
        res_dirs["w7"] = Path(args.compare_dir)

    base_ranges = _load_base_ranges()
    all_stats = {}
    for label, rdir in res_dirs.items():
        if not rdir.exists():
            print(f"[WARN] Missing result dir: {rdir}")
            continue
        params_df, summary_df = _collect_results(rdir)
        if params_df.empty and summary_df.empty:
            print(f"[WARN] No results found in: {rdir}")
            continue

        bounds = _boundary_analysis(params_df, base_ranges)
        if not bounds.empty:
            bounds["suggestion"] = bounds.apply(_suggest_range, axis=1)
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bounds.to_csv(out_path.with_name(f"{out_path.stem}_{label}.csv"), index=False)

        all_stats[label] = summary_df

    if all_stats:
        compare = _window_compare(all_stats)
        if not compare.empty:
            print(compare)


if __name__ == "__main__":
    main()
