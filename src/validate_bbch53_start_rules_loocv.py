#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "result" / "bbch53_start_patterns"


def _load_actual(actual_path: Path) -> pd.DataFrame:
    df = pd.read_excel(actual_path).copy()
    df["bbch_date"] = pd.to_datetime(df["bbch_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["plot", "variety", "year", "bbch_date"]).reset_index(drop=True)
    return df


def _load_weather_hourly(data_dir: Path, plot: str, variety: str, year: int) -> pd.DataFrame:
    rb_path = data_dir / f"{plot}_{variety}_{year}_grape_rb.json"
    if not rb_path.exists():
        raise FileNotFoundError(f"Missing weather file: {rb_path}")
    with rb_path.open("r", encoding="utf-8") as f:
        rb = json.load(f)
    wh = rb.get("weather_hourly", [])
    if not wh:
        raise ValueError(f"No weather_hourly records in: {rb_path}")
    df = pd.DataFrame(wh)
    if "datetime" not in df.columns or "air_temp" not in df.columns:
        raise ValueError(f"{rb_path.name} missing required weather columns.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["air_temp"] = pd.to_numeric(df["air_temp"], errors="coerce")
    df = df.dropna(subset=["datetime", "air_temp"]).sort_values("datetime").reset_index(drop=True)
    return df


def _daily_from_hourly(hourly: pd.DataFrame, t_base: float, rgdd_windows: List[int]) -> pd.DataFrame:
    d = hourly.copy()
    d["date"] = d["datetime"].dt.normalize()
    daily = d.groupby("date", as_index=False).agg(
        tmean=("air_temp", "mean"),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["gdd"] = (daily["tmean"] - float(t_base)).clip(lower=0.0)
    daily["agdd"] = daily["gdd"].cumsum()
    for w in sorted(set(int(x) for x in rgdd_windows if int(x) > 0)):
        daily[f"rgdd_{w}"] = daily["gdd"].rolling(w, min_periods=w).sum()
    return daily


def _rolling_run_length(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.size, dtype=int)
    run = 0
    for i, m in enumerate(mask):
        if m:
            run += 1
        else:
            run = 0
        out[i] = run
    return out


def _predict_start(
    sample: Dict,
    rule: Dict,
) -> Dict[str, float]:
    d = sample["daily"]
    bbch_date = sample["bbch_date"]
    rgdd_window = int(rule["rgdd_window"])
    rgdd_col = f"rgdd_{rgdd_window}"
    if rgdd_col not in d.columns:
        return {"lead_days": np.nan, "rgdd_threshold_effective": np.nan, "pred_start_date": pd.NaT}

    x = d[d["date"] <= bbch_date].copy()
    if x.empty:
        return {"lead_days": np.nan, "rgdd_threshold_effective": np.nan, "pred_start_date": pd.NaT}

    mode = str(rule["rule_mode"])
    rule_value = float(rule["rule_value"])
    if mode == "absolute":
        rgdd_thr = rule_value
    else:
        hit = x[x["date"] == bbch_date]
        if hit.empty:
            return {"lead_days": np.nan, "rgdd_threshold_effective": np.nan, "pred_start_date": pd.NaT}
        rgdd_at_bbch = float(hit[rgdd_col].iloc[0])
        if not np.isfinite(rgdd_at_bbch):
            return {"lead_days": np.nan, "rgdd_threshold_effective": np.nan, "pred_start_date": pd.NaT}
        rgdd_thr = rule_value * rgdd_at_bbch

    warm = _rolling_run_length((x["tmean"].values >= float(rule["tmean_threshold"])))
    cond = (x[rgdd_col].values >= rgdd_thr) & (warm >= int(rule["run_days"]))
    idx = np.where(cond)[0]
    if len(idx) == 0:
        return {"lead_days": np.nan, "rgdd_threshold_effective": float(rgdd_thr), "pred_start_date": pd.NaT}
    pred_start = pd.to_datetime(x.iloc[int(idx[0])]["date"])
    lead_days = float((bbch_date - pred_start).days)
    return {
        "lead_days": lead_days,
        "rgdd_threshold_effective": float(rgdd_thr),
        "pred_start_date": pred_start,
    }


def _score_on_training(train_samples: List[Dict], rule: Dict) -> Dict[str, float]:
    leads = []
    for s in train_samples:
        p = _predict_start(s, rule)
        if np.isfinite(p["lead_days"]):
            leads.append(float(p["lead_days"]))
    found = len(leads)
    n = len(train_samples)
    if found == 0:
        return {
            "coverage": 0.0,
            "lead_std_days": np.nan,
            "lead_iqr_days": np.nan,
            "lead_mean_days": np.nan,
            "lead_median_days": np.nan,
            "stability_score": -1e9,
        }
    arr = np.array(leads, dtype=float)
    std = float(arr.std(ddof=0))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    score = 100.0 * (found / n) - 0.8 * std - 0.4 * iqr
    return {
        "coverage": float(found / n),
        "lead_std_days": std,
        "lead_iqr_days": iqr,
        "lead_mean_days": float(arr.mean()),
        "lead_median_days": float(np.median(arr)),
        "stability_score": float(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LOOCV validation for BBCH53 start-rule candidates.")
    parser.add_argument(
        "--rules-file",
        type=str,
        default=str(OUT_DIR / "bbch53_start_rule_topk.csv"),
        help="Candidate rules CSV from extract_bbch53_start_patterns.py",
    )
    parser.add_argument("--actual-file", type=str, default=str(DATA_DIR / "actual_data.xlsx"))
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--t-base", type=float, default=10.0)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = pd.read_csv(args.rules_file).copy()
    if "rgdd_window" not in rules.columns:
        rules["rgdd_window"] = 7
    required = {"rule_mode", "rule_value", "tmean_threshold", "run_days", "rgdd_window"}
    miss = sorted(required - set(rules.columns))
    if miss:
        raise ValueError(f"Rules file missing columns: {miss}")
    rules = rules[list(required)].drop_duplicates().reset_index(drop=True)

    actual = _load_actual(Path(args.actual_file))
    rgdd_windows = sorted({int(x) for x in rules["rgdd_window"].tolist()})

    samples: List[Dict] = []
    for r in actual.itertuples(index=False):
        plot = str(r.plot)
        variety = str(r.variety)
        year = int(r.year)
        bbch_date = pd.to_datetime(r.bbch_date).normalize()
        hourly = _load_weather_hourly(Path(args.data_dir), plot, variety, year)
        daily = _daily_from_hourly(hourly, t_base=float(args.t_base), rgdd_windows=rgdd_windows)
        samples.append(
            {
                "sample_id": f"{plot}_{variety}_{year}",
                "plot": plot,
                "variety": variety,
                "year": year,
                "bbch_date": bbch_date,
                "daily": daily,
            }
        )

    fold_rows = []
    chosen_counter: Dict[str, int] = {}

    for i in range(len(samples)):
        test = samples[i]
        train = [samples[j] for j in range(len(samples)) if j != i]

        train_scored = []
        for rr in rules.to_dict(orient="records"):
            m = _score_on_training(train, rr)
            train_scored.append({**rr, **m})
        score_df = pd.DataFrame(train_scored).sort_values(
            ["coverage", "stability_score", "lead_std_days"],
            ascending=[False, False, True],
        )
        best = score_df.iloc[0].to_dict()
        pred = _predict_start(test, best)
        abs_err = abs(pred["lead_days"]) if np.isfinite(pred["lead_days"]) else np.nan
        key = (
            f"w{int(best['rgdd_window'])}|{best['rule_mode']}|v{float(best['rule_value']):.6g}|"
            f"T{float(best['tmean_threshold']):.6g}|k{int(best['run_days'])}"
        )
        chosen_counter[key] = chosen_counter.get(key, 0) + 1
        fold_rows.append(
            {
                "test_sample_id": test["sample_id"],
                "plot": test["plot"],
                "variety": test["variety"],
                "year": test["year"],
                "bbch_date": test["bbch_date"],
                "selected_rgdd_window": int(best["rgdd_window"]),
                "selected_rule_mode": str(best["rule_mode"]),
                "selected_rule_value": float(best["rule_value"]),
                "selected_tmean_threshold": float(best["tmean_threshold"]),
                "selected_run_days": int(best["run_days"]),
                "train_coverage": float(best["coverage"]),
                "train_stability_score": float(best["stability_score"]),
                "train_lead_std_days": float(best["lead_std_days"]),
                "pred_start_date": pred["pred_start_date"],
                "lead_days": float(pred["lead_days"]) if np.isfinite(pred["lead_days"]) else np.nan,
                "abs_lead_days": float(abs_err) if np.isfinite(abs_err) else np.nan,
                "rgdd_threshold_effective": pred["rgdd_threshold_effective"],
            }
        )

    folds_df = pd.DataFrame(fold_rows).sort_values(["plot", "year"]).reset_index(drop=True)
    folds_df.to_csv(out_dir / "bbch53_start_rule_loocv_folds.csv", index=False)

    valid = folds_df["lead_days"].dropna().to_numpy(dtype=float)
    abs_valid = np.abs(valid) if valid.size else np.array([])
    summary = {
        "n_samples": int(len(samples)),
        "n_valid": int(abs_valid.size),
        "loocv_mae_lead_days": float(abs_valid.mean()) if abs_valid.size else None,
        "loocv_rmse_lead_days": float(np.sqrt(np.mean(np.square(abs_valid)))) if abs_valid.size else None,
        "loocv_bias_days": float(valid.mean()) if valid.size else None,
        "chosen_rule_frequency": chosen_counter,
        "outputs": {
            "folds": "bbch53_start_rule_loocv_folds.csv",
            "rule_frequency": "bbch53_start_rule_loocv_rule_frequency.csv",
        },
    }
    (out_dir / "bbch53_start_rule_loocv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    freq_df = pd.DataFrame(
        [{"rule_key": k, "selected_count": v} for k, v in chosen_counter.items()]
    ).sort_values("selected_count", ascending=False)
    freq_df.to_csv(out_dir / "bbch53_start_rule_loocv_rule_frequency.csv", index=False)

    print(f"Saved: {out_dir / 'bbch53_start_rule_loocv_folds.csv'}")
    print(f"Saved: {out_dir / 'bbch53_start_rule_loocv_rule_frequency.csv'}")
    print(f"Saved: {out_dir / 'bbch53_start_rule_loocv_summary.json'}")


if __name__ == "__main__":
    main()

