#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

try:
    import ray
except Exception:  # pragma: no cover
    ray = None

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.bbch53_module import compute_agdd_hourly, estimate_bbch53_date, load_hourly_weather_from_json
except Exception:  # pragma: no cover
    from bbch53_module import compute_agdd_hourly, estimate_bbch53_date, load_hourly_weather_from_json


DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "result" / "historical_inputs" / "bbch53_calibration"
PLOT_TO_VAR_CODE = {"GL": "WK", "NN": "JF", "DA": "MPT"}


def _load_actual() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "actual_data.xlsx")
    df["bbch_date"] = pd.to_datetime(df["bbch_date"], errors="coerce")
    return df.dropna(subset=["bbch_date"]).reset_index(drop=True)


def _load_site_year_hourly(row) -> pd.DataFrame:
    rb_path = DATA_DIR / f"{row.plot}_{row.variety}_{int(row.year)}_grape_rb.json"
    if not rb_path.exists():
        raise FileNotFoundError(f"Missing weather json: {rb_path}")
    df = load_hourly_weather_from_json(rb_path)
    y0 = int(row.year)
    return df[df["datetime"].dt.year == y0].reset_index(drop=True)


def _build_cache(rows: List) -> List[Dict]:
    cache = []
    for r in rows:
        plot = str(r.plot)
        cache.append(
            {
                "plot": plot,
                "year": int(r.year),
                "obs_bbch53": pd.to_datetime(r.bbch_date).normalize(),
                "hourly": _load_site_year_hourly(r),
                "variety_code": PLOT_TO_VAR_CODE.get(plot, "UNK"),
            }
        )
    return cache


def _start_dt_for_year_from_doy(year: int, doy: int) -> pd.Timestamp:
    y = int(year)
    d = int(doy)
    return pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=d - 1)


def _backcalc_theta_at_obs(cache: List[Dict], t_base: float, start_doy: int) -> pd.DataFrame:
    rows = []
    for s in cache:
        obs = pd.to_datetime(s["obs_bbch53"]).normalize()
        start_dt = _start_dt_for_year_from_doy(s["year"], start_doy)
        ag = compute_agdd_hourly(s["hourly"], t_base=t_base, start_datetime=start_dt)
        ag = ag[ag["datetime"] <= (obs + pd.Timedelta(hours=23, minutes=59, seconds=59))]
        theta_obs = float(ag["agdd"].iloc[-1]) if not ag.empty else np.nan
        rows.append(
            {
                "plot": s["plot"],
                "year": s["year"],
                "obs_bbch53": obs,
                "t_base_main": float(t_base),
                "theta_obs": theta_obs,
            }
        )
    return pd.DataFrame(rows)


def _backcalc_rgdd_at_obs(
    cache: List[Dict],
    t_base: float,
    start_doy: int,
    rgdd_window: int,
) -> pd.DataFrame:
    rows = []
    for s in cache:
        obs = pd.to_datetime(s["obs_bbch53"]).normalize()
        start_dt = _start_dt_for_year_from_doy(s["year"], start_doy)
        ag = compute_agdd_hourly(s["hourly"], t_base=t_base, start_datetime=start_dt)
        if ag.empty:
            rgdd_obs = np.nan
        else:
            d = (
                ag.assign(date=ag["datetime"].dt.normalize())
                .groupby("date", as_index=False)
                .agg(gdd_day=("gdd_hour", "sum"))
                .sort_values("date")
            )
            d["rgdd_window"] = d["gdd_day"].rolling(rgdd_window, min_periods=rgdd_window).sum()
            hit = d[d["date"] == obs]
            rgdd_obs = float(hit["rgdd_window"].iloc[0]) if not hit.empty else np.nan
        rows.append(
            {
                "plot": s["plot"],
                "year": s["year"],
                "obs_bbch53": obs,
                "t_base_main": float(t_base),
                "rgdd_window_obs": rgdd_obs,
            }
        )
    return pd.DataFrame(rows)


def _estimate_bbch53_date_with_gate(
    hourly_df: pd.DataFrame,
    theta_agdd: float,
    t_base: float,
    start_datetime: pd.Timestamp,
    use_rgdd_gate: bool,
    rgdd_min: float,
    rgdd_window: int,
) -> pd.Timestamp:
    ag = compute_agdd_hourly(hourly_df, t_base=t_base, start_datetime=start_datetime)
    if ag.empty:
        return pd.NaT
    d = (
        ag.assign(date=ag["datetime"].dt.normalize())
        .groupby("date", as_index=False)
        .agg(gdd_day=("gdd_hour", "sum"), agdd_end=("agdd", "max"))
        .sort_values("date")
    )
    if use_rgdd_gate:
        d["rgdd_window"] = d["gdd_day"].rolling(rgdd_window, min_periods=rgdd_window).sum()
        mask = (d["agdd_end"] >= float(theta_agdd)) & (d["rgdd_window"] >= float(rgdd_min))
    else:
        mask = d["agdd_end"] >= float(theta_agdd)
    hit = d[mask]
    if hit.empty:
        return pd.NaT
    return pd.to_datetime(hit.iloc[0]["date"]).normalize()


def _eval_params(
    cache: List[Dict],
    t_base: float,
    theta: float,
    start_doy: int,
    use_rgdd_gate: bool,
    rgdd_min: float,
    rgdd_window: int,
    use_variety_delta: bool,
    delta_wk: float,
    delta_jf: float,
    delta_mpt: float,
    penalty_days: float = 120.0,
) -> Tuple[float, float, List[Dict]]:
    errs = []
    detail = []
    for s in cache:
        if use_variety_delta:
            dmap = {"WK": float(delta_wk), "JF": float(delta_jf), "MPT": float(delta_mpt)}
            delta_var = dmap.get(str(s.get("variety_code", "UNK")), 0.0)
            theta_eff = float(theta) * (1.0 + delta_var)
        else:
            delta_var = 0.0
            theta_eff = float(theta)
        start_dt = _start_dt_for_year_from_doy(s["year"], start_doy)
        pred = _estimate_bbch53_date_with_gate(
            s["hourly"],
            theta_agdd=theta_eff,
            t_base=t_base,
            start_datetime=start_dt,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=rgdd_min,
            rgdd_window=rgdd_window,
        )
        obs = s["obs_bbch53"]
        if pd.isna(pred):
            err = float(penalty_days)
        else:
            err = float((pred - obs).days)
        errs.append(err)
        detail.append(
            {
                "plot": s["plot"],
                "year": s["year"],
                "obs_bbch53": obs,
                "pred_bbch53": pred,
                "error_days": err,
                "abs_error_days": abs(err),
                "rgdd_gate_used": bool(use_rgdd_gate),
                "rgdd_min": float(rgdd_min) if use_rgdd_gate else np.nan,
                "variety_code": str(s.get("variety_code", "UNK")),
                "start_doy": int(start_doy),
                "start_date_used": start_dt,
                "theta_eff": float(theta_eff),
                "delta_variety": float(delta_var) if use_variety_delta else np.nan,
            }
        )
    abs_err = np.abs(np.asarray(errs, dtype=float))
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(abs_err**2)))
    return mae, rmse, detail


if ray is not None:
    @ray.remote
    def _eval_combo_remote(
        cache_ref,
        t_base: float,
        theta: float,
        rgdd_min: float,
        delta_wk: float,
        delta_jf: float,
        delta_mpt: float,
        start_doy: int,
        use_rgdd_gate: bool,
        rgdd_window: int,
        use_variety_delta: bool,
    ) -> Dict:
        # Ray may pass either the resolved value or an ObjectRef depending on call path.
        cache = ray.get(cache_ref) if isinstance(cache_ref, ray.ObjectRef) else cache_ref
        mae, rmse, _ = _eval_params(
            cache,
            float(t_base),
            float(theta),
            int(start_doy),
            bool(use_rgdd_gate),
            float(rgdd_min),
            int(rgdd_window),
            bool(use_variety_delta),
            float(delta_wk),
            float(delta_jf),
            float(delta_mpt),
        )
        return {
            "t_base": float(t_base),
            "theta_agdd": float(theta),
            "start_doy": int(start_doy),
            "rgdd_min": float(rgdd_min),
            "delta_wk": float(delta_wk),
            "delta_jf": float(delta_jf),
            "delta_mpt": float(delta_mpt),
            "mae_days": float(mae),
            "rmse_days": float(rmse),
        }


def _grid_search(
    cache: List[Dict],
    tbase_grid: np.ndarray,
    theta_grid: np.ndarray,
    start_doy_grid: np.ndarray,
    rgdd_grid: np.ndarray,
    delta_wk_grid: np.ndarray,
    delta_jf_grid: np.ndarray,
    delta_mpt_grid: np.ndarray,
    n_jobs: int,
    use_rgdd_gate: bool,
    rgdd_window: int,
    use_variety_delta: bool,
) -> Tuple[Dict, pd.DataFrame]:
    combos = [
        (float(tb), float(th), int(sd), float(rg), float(dw), float(dj), float(dm))
        for tb in tbase_grid
        for th in theta_grid
        for sd in start_doy_grid
        for rg in rgdd_grid
        for dw in delta_wk_grid
        for dj in delta_jf_grid
        for dm in delta_mpt_grid
    ]
    if ray is None or n_jobs <= 1:
        out = []
        for tb, th, sd, rg, dw, dj, dm in tqdm(combos, desc="BBCH grid", total=len(combos)):
            mae, rmse, _ = _eval_params(
                cache, tb, th, sd, use_rgdd_gate, rg, rgdd_window,
                use_variety_delta, dw, dj, dm
            )
            out.append(
                {
                    "t_base": tb, "theta_agdd": th, "start_doy": int(sd), "rgdd_min": rg,
                    "delta_wk": dw, "delta_jf": dj, "delta_mpt": dm,
                    "mae_days": mae, "rmse_days": rmse,
                }
            )
        trials = pd.DataFrame(out).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)
        return trials.iloc[0].to_dict(), trials

    cache_ref = ray.put(cache)
    pending = {}
    rows_out: List[Dict] = []
    done = 0
    total = len(combos)
    cursor = 0
    pbar = tqdm(total=total, desc="BBCH grid")
    while done < total:
        while cursor < total and len(pending) < max(1, n_jobs):
            tb, th, sd, rg, dw, dj, dm = combos[cursor]
            fut = _eval_combo_remote.remote(
                cache_ref, tb, th, rg, dw, dj, dm,
                sd, use_rgdd_gate, rgdd_window, use_variety_delta
            )
            pending[fut] = True
            cursor += 1
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        for r in ready:
            rows_out.append(ray.get(r))
            pending.pop(r, None)
            done += 1
            pbar.update(1)
    pbar.close()
    trials = pd.DataFrame(rows_out).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)
    return trials.iloc[0].to_dict(), trials


def _loocv(
    rows: List,
    tbase_grid: np.ndarray,
    theta_grid: np.ndarray,
    start_doy_grid: np.ndarray,
    rgdd_grid: np.ndarray,
    delta_wk_grid: np.ndarray,
    delta_jf_grid: np.ndarray,
    delta_mpt_grid: np.ndarray,
    n_jobs: int,
    use_rgdd_gate: bool,
    rgdd_window: int,
    use_variety_delta: bool,
) -> pd.DataFrame:
    out = []
    for i in tqdm(range(len(rows)), desc="BBCH LOOCV"):
        train = [rows[j] for j in range(len(rows)) if j != i]
        test = rows[i]
        train_cache = _build_cache(train)
        best, _ = _grid_search(
            train_cache,
            tbase_grid,
            theta_grid,
            start_doy_grid,
            rgdd_grid,
            delta_wk_grid,
            delta_jf_grid,
            delta_mpt_grid,
            n_jobs=n_jobs,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_window=rgdd_window,
            use_variety_delta=use_variety_delta,
        )
        test_cache = _build_cache([test])
        mae, rmse, detail = _eval_params(
            test_cache,
            best["t_base"],
            best["theta_agdd"],
            start_doy=int(best["start_doy"]),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=best["rgdd_min"],
            rgdd_window=rgdd_window,
            use_variety_delta=use_variety_delta,
            delta_wk=float(best.get("delta_wk", 0.0)),
            delta_jf=float(best.get("delta_jf", 0.0)),
            delta_mpt=float(best.get("delta_mpt", 0.0)),
        )
        d = detail[0]
        out.append(
            {
                "plot": d["plot"],
                "year": d["year"],
                "obs_bbch53": d["obs_bbch53"],
                "pred_bbch53": d["pred_bbch53"],
                "error_days": d["error_days"],
                "abs_error_days": d["abs_error_days"],
                "fit_t_base": best["t_base"],
                "fit_theta_agdd": best["theta_agdd"],
                "fit_start_doy": int(best["start_doy"]),
                "fit_rgdd_min": best["rgdd_min"],
                "fit_delta_wk": best.get("delta_wk", 0.0),
                "fit_delta_jf": best.get("delta_jf", 0.0),
                "fit_delta_mpt": best.get("delta_mpt", 0.0),
                "fit_train_mae_days": best["mae_days"],
                "fit_train_rmse_days": best["rmse_days"],
                "test_mae_days": mae,
                "test_rmse_days": rmse,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate BBCH53 GDD inversion module.")
    parser.add_argument("--tbase-min", type=float, default=10.0)
    parser.add_argument("--tbase-max", type=float, default=12.0)
    parser.add_argument("--tbase-step", type=float, default=1.0)
    parser.add_argument("--theta-min", type=float, default=100.0)
    parser.add_argument("--theta-max", type=float, default=1100.0)
    parser.add_argument("--theta-step", type=float, default=10.0)
    parser.add_argument("--tbase-main", type=float, default=10.0, help="Main setting for publication scenario.")
    parser.add_argument(
        "--theta-quantiles",
        type=str,
        default="0.25,0.5,0.75",
        help="Quantiles for main-setting threshold scenarios.",
    )
    parser.add_argument("--n-jobs", type=int, default=512)
    parser.add_argument("--start-doy-min", type=int, default=32, help="Minimum DOY for start search (default Feb-1).")
    parser.add_argument("--start-doy-max", type=int, default=46, help="Maximum DOY for start search (default Feb-15).")
    parser.add_argument("--start-doy-step", type=int, default=1, help="Step size for start DOY grid.")
    parser.add_argument("--use-rgdd-gate", action="store_true", help="Enable condition-B gate: AGDD and RGDD-window.")
    parser.add_argument("--rgdd-window", type=int, default=7, help="Rolling window days for RGDD gate.")
    parser.add_argument("--rgdd-min", type=float, default=0.0, help="RGDD gate lower bound (used if auto grid is off).")
    parser.add_argument("--rgdd-max", type=float, default=80.0, help="RGDD gate upper bound (used if auto grid is off).")
    parser.add_argument("--rgdd-step", type=float, default=1.0, help="RGDD gate step (used if auto grid is off).")
    parser.add_argument(
        "--rgdd-auto-grid",
        action="store_true",
        help="Build RGDD grid from observed RGDD quantile range (Q10-Q90) at BBCH53.",
    )
    parser.add_argument(
        "--search-tbase",
        action="store_true",
        help="If set, optimize both t_base and theta; otherwise fix t_base=tbase_main and optimize theta only.",
    )
    parser.add_argument("--delta-min", type=float, default=-0.5, help="Lower bound for variety delta.")
    parser.add_argument("--delta-max", type=float, default=0.5, help="Upper bound for variety delta.")
    parser.add_argument("--delta-step", type=float, default=0.1, help="Step for variety delta grid.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Fixed mode: always use global theta + variety delta (WK/JF/MPT).
    use_variety_delta = True

    actual = _load_actual()
    rows = list(actual.itertuples(index=False))
    cache = _build_cache(rows)
    tbase_grid_full = np.arange(args.tbase_min, args.tbase_max + 1e-12, args.tbase_step)
    tbase_grid = tbase_grid_full if args.search_tbase else np.array([float(args.tbase_main)])
    theta_grid = np.arange(args.theta_min, args.theta_max + 1e-12, args.theta_step)
    start_doy_grid = np.arange(args.start_doy_min, args.start_doy_max + 1, args.start_doy_step, dtype=int)
    if args.use_rgdd_gate:
        if args.rgdd_auto_grid:
            rgdd_obs = _backcalc_rgdd_at_obs(
                cache,
                t_base=args.tbase_main,
                start_doy=int(np.median(start_doy_grid)),
                rgdd_window=args.rgdd_window,
            )
            valid = rgdd_obs["rgdd_window_obs"].dropna()
            if valid.empty:
                rgdd_grid = np.arange(args.rgdd_min, args.rgdd_max + 1e-12, args.rgdd_step)
            else:
                q10 = float(valid.quantile(0.10))
                q90 = float(valid.quantile(0.90))
                lo = max(0.0, np.floor(q10))
                hi = max(lo + args.rgdd_step, np.ceil(q90))
                rgdd_grid = np.arange(lo, hi + 1e-12, args.rgdd_step)
        else:
            rgdd_grid = np.arange(args.rgdd_min, args.rgdd_max + 1e-12, args.rgdd_step)
    else:
        rgdd_grid = np.array([0.0])
    if use_variety_delta:
        delta_grid = np.arange(args.delta_min, args.delta_max + 1e-12, args.delta_step)
    else:
        delta_grid = np.array([0.0])
    delta_wk_grid = delta_grid
    delta_jf_grid = delta_grid
    delta_mpt_grid = delta_grid

    ray_used = ray is not None and args.n_jobs > 1
    if ray_used:
        ray.init(num_cpus=max(1, args.n_jobs), include_dashboard=False, logging_level="ERROR")

    best, trials = _grid_search(
        cache,
        tbase_grid,
        theta_grid,
        start_doy_grid,
        rgdd_grid,
        delta_wk_grid,
        delta_jf_grid,
        delta_mpt_grid,
        n_jobs=args.n_jobs,
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_window=args.rgdd_window,
        use_variety_delta=use_variety_delta,
    )
    final_mae, final_rmse, final_detail = _eval_params(
        cache,
        best["t_base"],
        best["theta_agdd"],
        start_doy=int(best["start_doy"]),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_min=float(best.get("rgdd_min", 0.0)),
        rgdd_window=args.rgdd_window,
        use_variety_delta=use_variety_delta,
        delta_wk=float(best.get("delta_wk", 0.0)),
        delta_jf=float(best.get("delta_jf", 0.0)),
        delta_mpt=float(best.get("delta_mpt", 0.0)),
    )
    loocv = _loocv(
        rows,
        tbase_grid,
        theta_grid,
        start_doy_grid,
        rgdd_grid,
        delta_wk_grid,
        delta_jf_grid,
        delta_mpt_grid,
        n_jobs=args.n_jobs,
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_window=args.rgdd_window,
        use_variety_delta=use_variety_delta,
    )

    if ray_used:
        ray.shutdown()

    q_values = [float(x) for x in args.theta_quantiles.split(",") if x.strip()]
    start_doy_for_scenarios = int(best["start_doy"])
    theta_obs_df = _backcalc_theta_at_obs(
        cache,
        t_base=args.tbase_main,
        start_doy=start_doy_for_scenarios,
    ).dropna(subset=["theta_obs"])
    theta_obs_df.to_csv(OUT_DIR / "bbch53_theta_obs_tbase_main.csv", index=False)

    scenario_rows = []
    for q in q_values:
        theta_q = float(theta_obs_df["theta_obs"].quantile(q))
        mae_q, rmse_q, _ = _eval_params(
            cache,
            t_base=args.tbase_main,
            theta=theta_q,
            start_doy=start_doy_for_scenarios,
            use_rgdd_gate=args.use_rgdd_gate,
            rgdd_min=float(best.get("rgdd_min", 0.0)),
            rgdd_window=args.rgdd_window,
            use_variety_delta=use_variety_delta,
            delta_wk=float(best.get("delta_wk", 0.0)),
            delta_jf=float(best.get("delta_jf", 0.0)),
            delta_mpt=float(best.get("delta_mpt", 0.0)),
        )
        scenario_rows.append(
            {
                "t_base_main": float(args.tbase_main),
                "theta_quantile": q,
                "theta_agdd": theta_q,
                "mae_days": mae_q,
                "rmse_days": rmse_q,
            }
        )
    scenario_df = pd.DataFrame(scenario_rows).sort_values("theta_quantile").reset_index(drop=True)
    scenario_df.to_csv(OUT_DIR / "bbch53_main_setting_scenarios.csv", index=False)

    trials.to_csv(OUT_DIR / "bbch53_trials.csv", index=False)
    pd.DataFrame(final_detail).to_csv(OUT_DIR / "bbch53_fit_detail.csv", index=False)
    loocv.to_csv(OUT_DIR / "bbch53_loocv.csv", index=False)

    summary = {
        "best_t_base": float(best["t_base"]),
        "best_theta_agdd": float(best["theta_agdd"]),
        "best_start_doy": int(best["start_doy"]),
        "fit_mae_days": float(final_mae),
        "fit_rmse_days": float(final_rmse),
        "loocv_mae_days": float(loocv["abs_error_days"].mean()),
        "loocv_rmse_days": float(np.sqrt(np.mean(loocv["abs_error_days"].values ** 2))),
        "t_base_main_setting": float(args.tbase_main),
        "main_theta_quantiles": q_values,
        "main_theta_scenarios_file": "bbch53_main_setting_scenarios.csv",
        "best_rgdd_min": float(best.get("rgdd_min", 0.0)) if args.use_rgdd_gate else None,
        "best_delta_wk": float(best.get("delta_wk", 0.0)),
        "best_delta_jf": float(best.get("delta_jf", 0.0)),
        "best_delta_mpt": float(best.get("delta_mpt", 0.0)),
        "use_rgdd_gate": bool(args.use_rgdd_gate),
        "use_variety_delta": True,
        "rgdd_window": int(args.rgdd_window),
        "start_doy_min": int(args.start_doy_min),
        "start_doy_max": int(args.start_doy_max),
        "start_doy_step": int(args.start_doy_step),
        "calibration_mode": (
            "search_tbase_startdoy_theta_variety_delta"
            if args.search_tbase
            else "fixed_tbase_startdoy_theta_variety_delta"
        ),
        "t_base_search_enabled": bool(args.search_tbase),
        "t_base_grid_size": int(len(tbase_grid)),
        "start_doy_grid_size": int(len(start_doy_grid)),
        "theta_grid_size": int(len(theta_grid)),
        "rgdd_grid_size": int(len(rgdd_grid)),
        "delta_grid_size": int(len(delta_grid)),
        "variety_codes": ["WK", "JF", "MPT"],
        "n_site_year": int(len(rows)),
    }
    with open(OUT_DIR / "bbch53_best.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {OUT_DIR / 'bbch53_best.json'}")
    print(f"Saved: {OUT_DIR / 'bbch53_trials.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_fit_detail.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_loocv.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_theta_obs_tbase_main.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_main_setting_scenarios.csv'}")


if __name__ == "__main__":
    main()
