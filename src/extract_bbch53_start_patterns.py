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
ACTUAL_PATH = DATA_DIR / "actual_data.xlsx"
OUT_DIR = PROJECT_ROOT / "result" / "bbch53_start_patterns"


def _load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_PATH).copy()
    df["bbch_date"] = pd.to_datetime(df["bbch_date"], errors="coerce")
    df = df.dropna(subset=["plot", "variety", "year", "bbch_date"]).reset_index(drop=True)
    return df


def _load_weather_hourly(plot: str, variety: str, year: int) -> pd.DataFrame:
    rb_path = DATA_DIR / f"{plot}_{variety}_{year}_grape_rb.json"
    if not rb_path.exists():
        raise FileNotFoundError(f"Missing weather file: {rb_path}")
    with rb_path.open("r", encoding="utf-8") as f:
        rb = json.load(f)
    wh = rb.get("weather_hourly", [])
    if not wh:
        raise ValueError(f"No weather_hourly records in: {rb_path}")
    df = pd.DataFrame(wh)
    needed = ["datetime", "air_temp"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{rb_path.name} missing fields: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["air_temp"] = pd.to_numeric(df["air_temp"], errors="coerce")
    df = df.dropna(subset=["datetime", "air_temp"]).sort_values("datetime").reset_index(drop=True)
    return df


def _daily_from_hourly(hourly: pd.DataFrame, t_base: float, rgdd_windows: List[int]) -> pd.DataFrame:
    d = hourly.copy()
    d["date"] = d["datetime"].dt.normalize()
    daily = d.groupby("date", as_index=False).agg(
        tmean=("air_temp", "mean"),
        tmax=("air_temp", "max"),
        tmin=("air_temp", "min"),
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


def _sample_metrics(
    sample_daily: pd.DataFrame,
    bbch_date: pd.Timestamp,
    lookback_days: int,
    rgdd_windows: List[int],
) -> Dict[str, float]:
    win_start = bbch_date - pd.Timedelta(days=lookback_days)
    w = sample_daily[(sample_daily["date"] >= win_start) & (sample_daily["date"] <= bbch_date)].copy()
    if w.empty:
        return {}
    warm10 = _rolling_run_length((w["tmean"].values >= 10.0))
    warm12 = _rolling_run_length((w["tmean"].values >= 12.0))
    out = {
        "gdd_total": float(w["gdd"].sum(skipna=True)),
        "tmean_mean": float(w["tmean"].mean(skipna=True)),
        "tmean_min": float(w["tmean"].min(skipna=True)),
        "tmean_max": float(w["tmean"].max(skipna=True)),
        "tmax_max": float(w["tmax"].max(skipna=True)),
        "tmin_min": float(w["tmin"].min(skipna=True)),
        "days_tmean_ge_8": int((w["tmean"] >= 8.0).sum()),
        "days_tmean_ge_10": int((w["tmean"] >= 10.0).sum()),
        "days_tmean_ge_12": int((w["tmean"] >= 12.0).sum()),
        "warm10_longest_run": int(warm10.max() if warm10.size else 0),
        "warm12_longest_run": int(warm12.max() if warm12.size else 0),
    }
    for rw in sorted(set(int(x) for x in rgdd_windows if int(x) > 0)):
        col = f"rgdd_{rw}"
        if col not in w.columns:
            continue
        out[f"rgdd{rw}_bbch"] = float(w.loc[w["date"] == bbch_date, col].iloc[0]) if (w["date"] == bbch_date).any() else np.nan
        out[f"rgdd{rw}_min"] = float(w[col].min(skipna=True))
        out[f"rgdd{rw}_mean"] = float(w[col].mean(skipna=True))
        out[f"rgdd{rw}_max"] = float(w[col].max(skipna=True))
    return out


def _first_start_date(
    d: pd.DataFrame,
    bbch_date: pd.Timestamp,
    rgdd_window: int,
    rgdd_threshold: float,
    tmean_threshold: float,
    run_days: int,
) -> pd.Timestamp:
    x = d[d["date"] <= bbch_date].copy()
    if x.empty:
        return pd.NaT
    col = f"rgdd_{int(rgdd_window)}"
    if col not in x.columns:
        return pd.NaT
    warm = _rolling_run_length((x["tmean"].values >= tmean_threshold))
    cond = (x[col].values >= rgdd_threshold) & (warm >= run_days)
    idx = np.where(cond)[0]
    if len(idx) == 0:
        return pd.NaT
    return pd.to_datetime(x.iloc[int(idx[0])]["date"])


def _rule_scores(
    samples: List[Dict],
    rgdd_windows: List[int],
    rgdd_thresholds: List[float],
    p_ratios: List[float],
    tmean_thresholds: List[float],
    run_days_list: List[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_rule: List[Dict] = []
    rows_dates: List[Dict] = []
    for rgdd_window in sorted(set(int(x) for x in rgdd_windows if int(x) > 0)):
        rgdd_bbch_col = f"rgdd{rgdd_window}_bbch"
        for mode in ["absolute", "relative"]:
            for t_thr in tmean_thresholds:
                for run_days in run_days_list:
                    for s_val in (rgdd_thresholds if mode == "absolute" else p_ratios):
                        lead_days = []
                        found = 0
                        for s in samples:
                            d = s["daily"]
                            bbch_date = s["bbch_date"]
                            if mode == "absolute":
                                rgdd_thr = float(s_val)
                            else:
                                rgdd_bbch = s["metrics"].get(rgdd_bbch_col, np.nan)
                                if not np.isfinite(rgdd_bbch):
                                    rgdd_thr = np.nan
                                else:
                                    rgdd_thr = float(s_val) * float(rgdd_bbch)
                            if not np.isfinite(rgdd_thr):
                                pred_start = pd.NaT
                                lead = np.nan
                            else:
                                pred_start = _first_start_date(
                                    d=d,
                                    bbch_date=bbch_date,
                                    rgdd_window=int(rgdd_window),
                                    rgdd_threshold=rgdd_thr,
                                    tmean_threshold=float(t_thr),
                                    run_days=int(run_days),
                                )
                                lead = (bbch_date - pred_start).days if pd.notna(pred_start) else np.nan
                            if np.isfinite(lead):
                                found += 1
                                lead_days.append(float(lead))
                            rows_dates.append(
                                {
                                    "rgdd_window": int(rgdd_window),
                                    "rule_mode": mode,
                                    "rule_value": float(s_val),
                                    "tmean_threshold": float(t_thr),
                                    "run_days": int(run_days),
                                    "plot": s["plot"],
                                    "year": int(s["year"]),
                                    "variety": s["variety"],
                                    "bbch_date": s["bbch_date"],
                                    "pred_start_date": pred_start,
                                    "lead_days": float(lead) if np.isfinite(lead) else np.nan,
                                    "rgdd_threshold_effective": float(rgdd_thr) if np.isfinite(rgdd_thr) else np.nan,
                                }
                            )
                        if lead_days:
                            arr = np.array(lead_days, dtype=float)
                            lead_mean = float(arr.mean())
                            lead_std = float(arr.std(ddof=0))
                            lead_median = float(np.median(arr))
                            lead_iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
                            lead_min = float(arr.min())
                            lead_max = float(arr.max())
                            stability_score = 100.0 * (found / len(samples)) - 0.8 * lead_std - 0.4 * lead_iqr
                        else:
                            lead_mean = np.nan
                            lead_std = np.nan
                            lead_median = np.nan
                            lead_iqr = np.nan
                            lead_min = np.nan
                            lead_max = np.nan
                            stability_score = -1e9
                        rows_rule.append(
                            {
                                "rgdd_window": int(rgdd_window),
                                "rule_mode": mode,
                                "rule_value": float(s_val),
                                "tmean_threshold": float(t_thr),
                                "run_days": int(run_days),
                                "n_samples": int(len(samples)),
                                "found_samples": int(found),
                                "coverage": float(found / len(samples)),
                                "lead_mean_days": lead_mean,
                                "lead_median_days": lead_median,
                                "lead_std_days": lead_std,
                                "lead_iqr_days": lead_iqr,
                                "lead_min_days": lead_min,
                                "lead_max_days": lead_max,
                                "stability_score": float(stability_score),
                            }
                        )
    rule_df = pd.DataFrame(rows_rule).sort_values(
        ["coverage", "stability_score", "lead_std_days"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    date_df = pd.DataFrame(rows_dates)
    return rule_df, date_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract meteorological/agrometeorological start signals before observed BBCH53."
    )
    parser.add_argument("--t-base", type=float, default=10.0, help="Base temperature for GDD.")
    parser.add_argument("--lookback-days", type=int, default=45, help="Window length before BBCH53.")
    parser.add_argument("--topk", type=int, default=20, help="How many top rules to save.")
    parser.add_argument("--rgdd-windows", type=str, default="3,5,7,10,14", help="Comma-separated RGDD windows in days.")
    parser.add_argument("--rgdd-abs-min", type=float, default=10.0)
    parser.add_argument("--rgdd-abs-max", type=float, default=120.0)
    parser.add_argument("--rgdd-abs-step", type=float, default=5.0)
    parser.add_argument("--rgdd-ratios", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--tmean-thresholds", type=str, default="8,10,12")
    parser.add_argument("--run-days", type=str, default="3,5,7")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = _load_actual()

    rgdd_windows = [int(x) for x in args.rgdd_windows.split(",") if x.strip()]

    samples: List[Dict] = []
    metric_rows: List[Dict] = []
    for r in actual.itertuples(index=False):
        plot = str(r.plot)
        variety = str(r.variety)
        year = int(r.year)
        bbch_date = pd.to_datetime(r.bbch_date).normalize()
        hourly = _load_weather_hourly(plot, variety, year)
        daily = _daily_from_hourly(hourly, t_base=float(args.t_base), rgdd_windows=rgdd_windows)
        metrics = _sample_metrics(
            daily,
            bbch_date=bbch_date,
            lookback_days=int(args.lookback_days),
            rgdd_windows=rgdd_windows,
        )
        samples.append(
            {
                "plot": plot,
                "variety": variety,
                "year": year,
                "bbch_date": bbch_date,
                "daily": daily,
                "metrics": metrics,
            }
        )
        metric_rows.append(
            {
                "plot": plot,
                "variety": variety,
                "year": year,
                "bbch_date": bbch_date,
                **metrics,
            }
        )

    metric_df = pd.DataFrame(metric_rows).sort_values(["plot", "year"]).reset_index(drop=True)
    metric_df.to_csv(OUT_DIR / "bbch53_prewindow_metrics.csv", index=False)

    rgdd_abs = np.arange(args.rgdd_abs_min, args.rgdd_abs_max + 1e-12, args.rgdd_abs_step).tolist()
    rgdd_ratios = [float(x) for x in args.rgdd_ratios.split(",") if x.strip()]
    tmean_thrs = [float(x) for x in args.tmean_thresholds.split(",") if x.strip()]
    run_days = [int(x) for x in args.run_days.split(",") if x.strip()]

    rule_df, date_df = _rule_scores(
        samples=samples,
        rgdd_windows=rgdd_windows,
        rgdd_thresholds=rgdd_abs,
        p_ratios=rgdd_ratios,
        tmean_thresholds=tmean_thrs,
        run_days_list=run_days,
    )
    rule_df.to_csv(OUT_DIR / "bbch53_start_rule_candidates.csv", index=False)
    topk_df = rule_df.head(max(1, int(args.topk))).copy()
    topk_df.to_csv(OUT_DIR / "bbch53_start_rule_topk.csv", index=False)

    topk_keys = set(
        (int(x.rgdd_window), x.rule_mode, float(x.rule_value), float(x.tmean_threshold), int(x.run_days))
        for x in topk_df.itertuples(index=False)
    )
    date_topk = date_df[
        date_df.apply(
            lambda x: (
                int(x["rgdd_window"]),
                x["rule_mode"],
                float(x["rule_value"]),
                float(x["tmean_threshold"]),
                int(x["run_days"]),
            ) in topk_keys,
            axis=1,
        )
    ].copy()
    date_topk = date_topk.sort_values(
        ["rgdd_window", "rule_mode", "rule_value", "tmean_threshold", "run_days", "plot", "year"]
    )
    date_topk.to_csv(OUT_DIR / "bbch53_start_dates_topk_rules.csv", index=False)

    summary = {
        "n_site_year": int(len(samples)),
        "t_base": float(args.t_base),
        "lookback_days": int(args.lookback_days),
        "rgdd_windows": rgdd_windows,
        "n_rules_total": int(len(rule_df)),
        "n_rules_topk": int(len(topk_df)),
        "outputs": {
            "metrics": "bbch53_prewindow_metrics.csv",
            "rules_all": "bbch53_start_rule_candidates.csv",
            "rules_topk": "bbch53_start_rule_topk.csv",
            "dates_topk": "bbch53_start_dates_topk_rules.csv",
        },
    }
    (OUT_DIR / "bbch53_start_signal_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {OUT_DIR / 'bbch53_prewindow_metrics.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_start_rule_candidates.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_start_rule_topk.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_start_dates_topk_rules.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_start_signal_summary.json'}")


if __name__ == "__main__":
    main()
