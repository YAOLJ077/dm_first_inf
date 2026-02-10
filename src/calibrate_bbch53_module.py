#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import ray
except Exception:  # pragma: no cover
    ray = None

from bbch53_module import compute_agdd_hourly, estimate_bbch53_date, load_hourly_weather_from_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "result" / "historical_inputs" / "bbch53_calibration"


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
        cache.append(
            {
                "plot": str(r.plot),
                "year": int(r.year),
                "obs_bbch53": pd.to_datetime(r.bbch_date).normalize(),
                "hourly": _load_site_year_hourly(r),
            }
        )
    return cache


def _backcalc_theta_at_obs(cache: List[Dict], t_base: float) -> pd.DataFrame:
    rows = []
    for s in cache:
        obs = pd.to_datetime(s["obs_bbch53"]).normalize()
        ag = compute_agdd_hourly(s["hourly"], t_base=t_base)
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


def _eval_params(cache: List[Dict], t_base: float, theta: float, penalty_days: float = 120.0) -> Tuple[float, float, List[Dict]]:
    errs = []
    detail = []
    for s in cache:
        pred = estimate_bbch53_date(s["hourly"], theta_agdd=theta, t_base=t_base)
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
            }
        )
    abs_err = np.abs(np.asarray(errs, dtype=float))
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(abs_err**2)))
    return mae, rmse, detail


if ray is not None:
    @ray.remote
    def _eval_combo_remote(cache_ref, t_base: float, theta: float) -> Dict:
        cache = ray.get(cache_ref)
        mae, rmse, _ = _eval_params(cache, float(t_base), float(theta))
        return {
            "t_base": float(t_base),
            "theta_agdd": float(theta),
            "mae_days": float(mae),
            "rmse_days": float(rmse),
        }


def _grid_search(
    cache: List[Dict],
    tbase_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_jobs: int,
) -> Tuple[Dict, pd.DataFrame]:
    combos = [(float(tb), float(th)) for tb in tbase_grid for th in theta_grid]
    if ray is None or n_jobs <= 1:
        out = []
        for tb, th in combos:
            mae, rmse, _ = _eval_params(cache, tb, th)
            out.append({"t_base": tb, "theta_agdd": th, "mae_days": mae, "rmse_days": rmse})
        trials = pd.DataFrame(out).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)
        return trials.iloc[0].to_dict(), trials

    cache_ref = ray.put(cache)
    pending = {}
    rows_out: List[Dict] = []
    done = 0
    total = len(combos)
    cursor = 0
    while done < total:
        while cursor < total and len(pending) < max(1, n_jobs):
            tb, th = combos[cursor]
            fut = _eval_combo_remote.remote(cache_ref, tb, th)
            pending[fut] = True
            cursor += 1
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        for r in ready:
            rows_out.append(ray.get(r))
            pending.pop(r, None)
            done += 1
    trials = pd.DataFrame(rows_out).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)
    return trials.iloc[0].to_dict(), trials


def _loocv(rows: List, tbase_grid: np.ndarray, theta_grid: np.ndarray, n_jobs: int) -> pd.DataFrame:
    out = []
    for i in range(len(rows)):
        train = [rows[j] for j in range(len(rows)) if j != i]
        test = rows[i]
        train_cache = _build_cache(train)
        best, _ = _grid_search(train_cache, tbase_grid, theta_grid, n_jobs=n_jobs)
        test_cache = _build_cache([test])
        mae, rmse, detail = _eval_params(test_cache, best["t_base"], best["theta_agdd"])
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
                "fit_train_mae_days": best["mae_days"],
                "fit_train_rmse_days": best["rmse_days"],
                "test_mae_days": mae,
                "test_rmse_days": rmse,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate BBCH53 GDD inversion module.")
    parser.add_argument("--tbase-min", type=float, default=6.0)
    parser.add_argument("--tbase-max", type=float, default=14.0)
    parser.add_argument("--tbase-step", type=float, default=0.25)
    parser.add_argument("--theta-min", type=float, default=120.0)
    parser.add_argument("--theta-max", type=float, default=900.0)
    parser.add_argument("--theta-step", type=float, default=10.0)
    parser.add_argument("--tbase-main", type=float, default=10.0, help="Main setting for publication scenario.")
    parser.add_argument(
        "--theta-quantiles",
        type=str,
        default="0.25,0.5,0.75",
        help="Quantiles for main-setting threshold scenarios.",
    )
    parser.add_argument("--n-jobs", type=int, default=64)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    actual = _load_actual()
    rows = list(actual.itertuples(index=False))
    cache = _build_cache(rows)
    tbase_grid = np.arange(args.tbase_min, args.tbase_max + 1e-12, args.tbase_step)
    theta_grid = np.arange(args.theta_min, args.theta_max + 1e-12, args.theta_step)

    ray_used = ray is not None and args.n_jobs > 1
    if ray_used:
        ray.init(num_cpus=max(1, args.n_jobs), include_dashboard=False, logging_level="ERROR")

    best, trials = _grid_search(cache, tbase_grid, theta_grid, n_jobs=args.n_jobs)
    final_mae, final_rmse, final_detail = _eval_params(cache, best["t_base"], best["theta_agdd"])
    loocv = _loocv(rows, tbase_grid, theta_grid, n_jobs=args.n_jobs)

    if ray_used:
        ray.shutdown()

    q_values = [float(x) for x in args.theta_quantiles.split(",") if x.strip()]
    theta_obs_df = _backcalc_theta_at_obs(cache, t_base=args.tbase_main).dropna(subset=["theta_obs"])
    theta_obs_df.to_csv(OUT_DIR / "bbch53_theta_obs_tbase_main.csv", index=False)

    scenario_rows = []
    for q in q_values:
        theta_q = float(theta_obs_df["theta_obs"].quantile(q))
        mae_q, rmse_q, _ = _eval_params(cache, t_base=args.tbase_main, theta=theta_q)
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
        "fit_mae_days": float(final_mae),
        "fit_rmse_days": float(final_rmse),
        "loocv_mae_days": float(loocv["abs_error_days"].mean()),
        "loocv_rmse_days": float(np.sqrt(np.mean(loocv["abs_error_days"].values ** 2))),
        "t_base_main_setting": float(args.tbase_main),
        "main_theta_quantiles": q_values,
        "main_theta_scenarios_file": "bbch53_main_setting_scenarios.csv",
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

