#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.bbch53_module import compute_agdd_hourly, load_hourly_weather_from_json
except Exception:  # pragma: no cover
    from bbch53_module import compute_agdd_hourly, load_hourly_weather_from_json

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "result" / "historical_inputs" / "bbch53_calibration"
PLOT_TO_VAR_CODE = {"GL": "WK", "NN": "JF", "DA": "MPT"}

# Multiprocessing worker globals
_W_SAMPLES: List["Sample"] = []
_W_USE_RGDD_GATE: bool = False
_W_RGDD_WINDOW: int = 7
_W_USE_VARIETY_DELTA: bool = False
_W_FIXED_TBASE: float = 10.0
_W_FIXED_THETA: float = 400.0
_W_FIXED_START_DOY: int = 32
_W_FIXED_RGDD_MIN: float = 0.0


@dataclass
class Sample:
    plot: str
    year: int
    obs_bbch53: pd.Timestamp
    hourly: pd.DataFrame
    variety_code: str


def _load_actual() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "actual_data.xlsx")
    df["bbch_date"] = pd.to_datetime(df["bbch_date"], errors="coerce")
    return df.dropna(subset=["bbch_date"]).reset_index(drop=True)


def _load_site_year_hourly(plot: str, variety: str, year: int) -> pd.DataFrame:
    rb_path = DATA_DIR / f"{plot}_{variety}_{year}_grape_rb.json"
    if not rb_path.exists():
        raise FileNotFoundError(f"Missing weather json: {rb_path}")
    df = load_hourly_weather_from_json(rb_path)
    return df[df["datetime"].dt.year == int(year)].reset_index(drop=True)


def _build_samples(df: pd.DataFrame) -> List[Sample]:
    out: List[Sample] = []
    for r in df.itertuples(index=False):
        plot = str(r.plot)
        year = int(r.year)
        hourly = _load_site_year_hourly(plot, str(r.variety), year)
        out.append(
            Sample(
                plot=plot,
                year=year,
                obs_bbch53=pd.to_datetime(r.bbch_date).normalize(),
                hourly=hourly,
                variety_code=PLOT_TO_VAR_CODE.get(plot, "UNK"),
            )
        )
    return out


def _start_dt_for_year_from_doy(year: int, doy: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=int(doy) - 1)


def _estimate_date_with_gate(
    hourly_df: pd.DataFrame,
    theta_agdd: float,
    t_base: float,
    start_dt: pd.Timestamp,
    use_rgdd_gate: bool,
    rgdd_min: float,
    rgdd_window: int,
) -> pd.Timestamp:
    ag = compute_agdd_hourly(hourly_df, t_base=t_base, start_datetime=start_dt)
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
    samples: List[Sample],
    t_base: float,
    theta: float,
    start_doy: int,
    use_rgdd_gate: bool,
    rgdd_min: float,
    rgdd_window: int,
    delta_wk: float,
    delta_jf: float,
    delta_mpt: float,
    use_variety_delta: bool,
    penalty_days: float = 120.0,
) -> Tuple[float, float, List[Dict]]:
    dmap = {"WK": float(delta_wk), "JF": float(delta_jf), "MPT": float(delta_mpt)}
    errs: List[float] = []
    details: List[Dict] = []
    for s in samples:
        delta = dmap.get(s.variety_code, 0.0) if use_variety_delta else 0.0
        theta_eff = float(theta) * (1.0 + delta)
        start_dt = _start_dt_for_year_from_doy(s.year, start_doy)
        pred = _estimate_date_with_gate(
            s.hourly,
            theta_agdd=theta_eff,
            t_base=float(t_base),
            start_dt=start_dt,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(rgdd_min),
            rgdd_window=int(rgdd_window),
        )
        err = float((pred - s.obs_bbch53).days) if pd.notna(pred) else float(penalty_days)
        errs.append(err)
        details.append(
            {
                "plot": s.plot,
                "year": s.year,
                "obs_bbch53": s.obs_bbch53,
                "pred_bbch53": pred,
                "error_days": err,
                "abs_error_days": abs(err),
                "variety_code": s.variety_code,
                "start_doy": int(start_doy),
                "start_date_used": start_dt,
                "theta_eff": float(theta_eff),
                "delta_variety": float(delta),
                "rgdd_gate_used": bool(use_rgdd_gate),
                "rgdd_min": float(rgdd_min) if use_rgdd_gate else np.nan,
            }
        )
    abs_err = np.abs(np.asarray(errs, dtype=float))
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(abs_err**2)))
    return mae, rmse, details


def _init_stage1_worker(samples: List[Sample], use_rgdd_gate: bool, rgdd_window: int) -> None:
    global _W_SAMPLES, _W_USE_RGDD_GATE, _W_RGDD_WINDOW, _W_USE_VARIETY_DELTA
    _W_SAMPLES = samples
    _W_USE_RGDD_GATE = bool(use_rgdd_gate)
    _W_RGDD_WINDOW = int(rgdd_window)
    _W_USE_VARIETY_DELTA = False


def _init_stage2_worker(
    samples: List[Sample],
    use_rgdd_gate: bool,
    rgdd_window: int,
    fixed_tbase: float,
    fixed_theta: float,
    fixed_start_doy: int,
    fixed_rgdd_min: float,
) -> None:
    global _W_SAMPLES, _W_USE_RGDD_GATE, _W_RGDD_WINDOW, _W_USE_VARIETY_DELTA
    global _W_FIXED_TBASE, _W_FIXED_THETA, _W_FIXED_START_DOY, _W_FIXED_RGDD_MIN
    _W_SAMPLES = samples
    _W_USE_RGDD_GATE = bool(use_rgdd_gate)
    _W_RGDD_WINDOW = int(rgdd_window)
    _W_USE_VARIETY_DELTA = True
    _W_FIXED_TBASE = float(fixed_tbase)
    _W_FIXED_THETA = float(fixed_theta)
    _W_FIXED_START_DOY = int(fixed_start_doy)
    _W_FIXED_RGDD_MIN = float(fixed_rgdd_min)


def _worker_stage1(task: Tuple[float, float, int, float]) -> Dict:
    tb, th, sd, rg = task
    mae, rmse, _ = _eval_params(
        _W_SAMPLES,
        t_base=float(tb),
        theta=float(th),
        start_doy=int(sd),
        use_rgdd_gate=_W_USE_RGDD_GATE,
        rgdd_min=float(rg),
        rgdd_window=_W_RGDD_WINDOW,
        delta_wk=0.0,
        delta_jf=0.0,
        delta_mpt=0.0,
        use_variety_delta=False,
    )
    return {
        "t_base": float(tb),
        "theta_agdd": float(th),
        "start_doy": int(sd),
        "rgdd_min": float(rg),
        "mae_days": float(mae),
        "rmse_days": float(rmse),
    }


def _worker_stage2(task: Tuple[float, float, float]) -> Dict:
    dw, dj, dm = task
    mae, rmse, _ = _eval_params(
        _W_SAMPLES,
        t_base=_W_FIXED_TBASE,
        theta=_W_FIXED_THETA,
        start_doy=_W_FIXED_START_DOY,
        use_rgdd_gate=_W_USE_RGDD_GATE,
        rgdd_min=_W_FIXED_RGDD_MIN,
        rgdd_window=_W_RGDD_WINDOW,
        delta_wk=float(dw),
        delta_jf=float(dj),
        delta_mpt=float(dm),
        use_variety_delta=True,
    )
    return {
        "delta_wk": float(dw),
        "delta_jf": float(dj),
        "delta_mpt": float(dm),
        "mae_days": float(mae),
        "rmse_days": float(rmse),
    }


def _grid_search_stage1(
    samples: List[Sample],
    tbase_grid: np.ndarray,
    theta_grid: np.ndarray,
    start_doy_grid: np.ndarray,
    rgdd_grid: np.ndarray,
    use_rgdd_gate: bool,
    rgdd_window: int,
) -> Tuple[Dict, pd.DataFrame]:
    out: List[Dict] = []
    total = len(tbase_grid) * len(theta_grid) * len(start_doy_grid) * len(rgdd_grid)
    iterator = product(tbase_grid, theta_grid, start_doy_grid, rgdd_grid)
    if n_jobs <= 1:
        for task in tqdm(iterator, total=total, desc="Stage1 grid"):
            out.append(_worker_stage1(task))
    else:
        pbar = tqdm(total=total, desc="Stage1 grid")
        with ProcessPoolExecutor(
            max_workers=int(n_jobs),
            initializer=_init_stage1_worker,
            initargs=(samples, use_rgdd_gate, int(rgdd_window)),
        ) as ex:
            inflight_max = max(2 * int(n_jobs), 1)
            pending = {}
            task_iter = iter(iterator)
            while len(pending) < inflight_max:
                try:
                    t = next(task_iter)
                except StopIteration:
                    break
                pending[ex.submit(_worker_stage1, t)] = True
            while pending:
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    out.append(fut.result())
                    pending.pop(fut, None)
                    pbar.update(1)
                    try:
                        t = next(task_iter)
                        pending[ex.submit(_worker_stage1, t)] = True
                    except StopIteration:
                        pass
        pbar.close()
    trials = pd.DataFrame(out).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)
    return trials.iloc[0].to_dict(), trials


def _grid_search_stage2(
    samples: List[Sample],
    fixed_t_base: float,
    fixed_theta: float,
    fixed_start_doy: int,
    fixed_rgdd_min: float,
    delta_grid: np.ndarray,
    use_rgdd_gate: bool,
    rgdd_window: int,
) -> Tuple[Dict, pd.DataFrame]:
    out: List[Dict] = []
    total = len(delta_grid) ** 3
    iterator = product(delta_grid, delta_grid, delta_grid)
    if n_jobs <= 1:
        _init_stage2_worker(
            samples=samples,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_window=int(rgdd_window),
            fixed_tbase=float(fixed_t_base),
            fixed_theta=float(fixed_theta),
            fixed_start_doy=int(fixed_start_doy),
            fixed_rgdd_min=float(fixed_rgdd_min),
        )
        for task in tqdm(iterator, total=total, desc="Stage2 delta-grid"):
            out.append(_worker_stage2(task))
    else:
        pbar = tqdm(total=total, desc="Stage2 delta-grid")
        with ProcessPoolExecutor(
            max_workers=int(n_jobs),
            initializer=_init_stage2_worker,
            initargs=(
                samples,
                use_rgdd_gate,
                int(rgdd_window),
                float(fixed_t_base),
                float(fixed_theta),
                int(fixed_start_doy),
                float(fixed_rgdd_min),
            ),
        ) as ex:
            inflight_max = max(2 * int(n_jobs), 1)
            pending = {}
            task_iter = iter(iterator)
            while len(pending) < inflight_max:
                try:
                    t = next(task_iter)
                except StopIteration:
                    break
                pending[ex.submit(_worker_stage2, t)] = True
            while pending:
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    out.append(fut.result())
                    pending.pop(fut, None)
                    pbar.update(1)
                    try:
                        t = next(task_iter)
                        pending[ex.submit(_worker_stage2, t)] = True
                    except StopIteration:
                        pass
        pbar.close()
    trials = pd.DataFrame(out).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)
    best = trials.iloc[0].to_dict()
    best.update(
        {
            "t_base": float(fixed_t_base),
            "theta_agdd": float(fixed_theta),
            "start_doy": int(fixed_start_doy),
            "rgdd_min": float(fixed_rgdd_min),
        }
    )
    return best, trials


def _loocv(
    samples: List[Sample],
    tbase_grid: np.ndarray,
    theta_grid: np.ndarray,
    start_doy_grid: np.ndarray,
    rgdd_grid: np.ndarray,
    delta_grid: np.ndarray,
    use_rgdd_gate: bool,
    rgdd_window: int,
) -> pd.DataFrame:
    rows = []
    for i in tqdm(range(len(samples)), desc="LOOCV"):
        train = [samples[j] for j in range(len(samples)) if j != i]
        test = [samples[i]]
        b1, _ = _grid_search_stage1(
            train,
            tbase_grid=tbase_grid,
            theta_grid=theta_grid,
            start_doy_grid=start_doy_grid,
            rgdd_grid=rgdd_grid,
            n_jobs=n_jobs,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_window=rgdd_window,
        )
        b2, _ = _grid_search_stage2(
            train,
            fixed_t_base=float(b1["t_base"]),
            fixed_theta=float(b1["theta_agdd"]),
            fixed_start_doy=int(b1["start_doy"]),
            fixed_rgdd_min=float(b1["rgdd_min"]),
            delta_grid=delta_grid,
            n_jobs=n_jobs,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_window=rgdd_window,
        )
        mae, rmse, detail = _eval_params(
            test,
            t_base=float(b1["t_base"]),
            theta=float(b1["theta_agdd"]),
            start_doy=int(b1["start_doy"]),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(b1["rgdd_min"]),
            rgdd_window=rgdd_window,
            delta_wk=float(b2["delta_wk"]),
            delta_jf=float(b2["delta_jf"]),
            delta_mpt=float(b2["delta_mpt"]),
            use_variety_delta=True,
        )
        d = detail[0]
        rows.append(
            {
                "plot": d["plot"],
                "year": d["year"],
                "obs_bbch53": d["obs_bbch53"],
                "pred_bbch53": d["pred_bbch53"],
                "error_days": d["error_days"],
                "abs_error_days": d["abs_error_days"],
                "fit_t_base": float(b1["t_base"]),
                "fit_theta_agdd": float(b1["theta_agdd"]),
                "fit_start_doy": int(b1["start_doy"]),
                "fit_rgdd_min": float(b1["rgdd_min"]),
                "fit_delta_wk": float(b2["delta_wk"]),
                "fit_delta_jf": float(b2["delta_jf"]),
                "fit_delta_mpt": float(b2["delta_mpt"]),
                "fit_stage1_train_mae_days": float(b1["mae_days"]),
                "fit_stage1_train_rmse_days": float(b1["rmse_days"]),
                "fit_stage2_train_mae_days": float(b2["mae_days"]),
                "fit_stage2_train_rmse_days": float(b2["rmse_days"]),
                "test_mae_days": float(mae),
                "test_rmse_days": float(rmse),
            }
        )
    return pd.DataFrame(rows)


def _backcalc_theta_at_obs(samples: List[Sample], t_base: float, start_doy: int) -> pd.DataFrame:
    rows = []
    for s in samples:
        start_dt = _start_dt_for_year_from_doy(s.year, start_doy)
        ag = compute_agdd_hourly(s.hourly, t_base=t_base, start_datetime=start_dt)
        ag = ag[ag["datetime"] <= (s.obs_bbch53 + pd.Timedelta(hours=23, minutes=59, seconds=59))]
        theta_obs = float(ag["agdd"].iloc[-1]) if not ag.empty else np.nan
        rows.append(
            {
                "plot": s.plot,
                "year": s.year,
                "obs_bbch53": s.obs_bbch53,
                "t_base_main": float(t_base),
                "theta_obs": theta_obs,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage BBCH53 calibration (global then variety delta).")
    parser.add_argument("--tbase-min", type=float, default=10.0)
    parser.add_argument("--tbase-max", type=float, default=12.0)
    parser.add_argument("--tbase-step", type=float, default=1.0)
    parser.add_argument("--theta-min", type=float, default=100.0)
    parser.add_argument("--theta-max", type=float, default=1100.0)
    parser.add_argument("--theta-step", type=float, default=10.0)
    parser.add_argument("--search-tbase", action="store_true", default=True)
    parser.add_argument("--tbase-main", type=float, default=10.0)
    parser.add_argument("--start-doy-min", type=int, default=32)
    parser.add_argument("--start-doy-max", type=int, default=46)
    parser.add_argument("--start-doy-step", type=int, default=1)
    parser.add_argument("--use-rgdd-gate", action="store_true")
    parser.add_argument("--rgdd-window", type=int, default=7)
    parser.add_argument("--rgdd-min", type=float, default=0.0)
    parser.add_argument("--rgdd-max", type=float, default=80.0)
    parser.add_argument("--rgdd-step", type=float, default=1.0)
    parser.add_argument("--delta-min", type=float, default=-0.5)
    parser.add_argument("--delta-max", type=float, default=0.5)
    parser.add_argument("--delta-step", type=float, default=0.1)
    parser.add_argument("--theta-quantiles", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--n-jobs", type=int, default=128, help="Parallel workers; each worker evaluates one parameter combo per task.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = _load_actual()
    samples = _build_samples(actual)

    tbase_grid_full = np.arange(args.tbase_min, args.tbase_max + 1e-12, args.tbase_step)
    tbase_grid = tbase_grid_full if args.search_tbase else np.array([float(args.tbase_main)])
    theta_grid = np.arange(args.theta_min, args.theta_max + 1e-12, args.theta_step)
    start_doy_grid = np.arange(args.start_doy_min, args.start_doy_max + 1, args.start_doy_step, dtype=int)
    rgdd_grid = (
        np.arange(args.rgdd_min, args.rgdd_max + 1e-12, args.rgdd_step)
        if args.use_rgdd_gate
        else np.array([0.0])
    )
    delta_grid = np.arange(args.delta_min, args.delta_max + 1e-12, args.delta_step)

    stage1_total = len(tbase_grid) * len(theta_grid) * len(start_doy_grid) * len(rgdd_grid)
    stage2_total = len(delta_grid) ** 3
    print(
        f"[BBCH] Stage1 grid={stage1_total:,} | Stage2 grid={stage2_total:,} | "
        f"LOOCV total≈{(stage1_total + stage2_total) * len(samples):,} | n_jobs={args.n_jobs}",
        flush=True,
    )

    # Initialize worker globals for serial mode.
    _init_stage1_worker(samples, args.use_rgdd_gate, int(args.rgdd_window))

    best_stage1, trials_stage1 = _grid_search_stage1(
        samples,
        tbase_grid=tbase_grid,
        theta_grid=theta_grid,
        start_doy_grid=start_doy_grid,
        rgdd_grid=rgdd_grid,
        n_jobs=int(args.n_jobs),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_window=args.rgdd_window,
    )

    best_stage2, trials_stage2 = _grid_search_stage2(
        samples,
        fixed_t_base=float(best_stage1["t_base"]),
        fixed_theta=float(best_stage1["theta_agdd"]),
        fixed_start_doy=int(best_stage1["start_doy"]),
        fixed_rgdd_min=float(best_stage1["rgdd_min"]),
        delta_grid=delta_grid,
        n_jobs=int(args.n_jobs),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_window=args.rgdd_window,
    )

    fit_mae, fit_rmse, fit_detail = _eval_params(
        samples,
        t_base=float(best_stage1["t_base"]),
        theta=float(best_stage1["theta_agdd"]),
        start_doy=int(best_stage1["start_doy"]),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_min=float(best_stage1["rgdd_min"]),
        rgdd_window=args.rgdd_window,
        delta_wk=float(best_stage2["delta_wk"]),
        delta_jf=float(best_stage2["delta_jf"]),
        delta_mpt=float(best_stage2["delta_mpt"]),
        use_variety_delta=True,
    )

    loocv = _loocv(
        samples,
        tbase_grid=tbase_grid,
        theta_grid=theta_grid,
        start_doy_grid=start_doy_grid,
        rgdd_grid=rgdd_grid,
        delta_grid=delta_grid,
        n_jobs=int(args.n_jobs),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_window=args.rgdd_window,
    )

    q_values = [float(x) for x in args.theta_quantiles.split(",") if x.strip()]
    theta_obs_df = _backcalc_theta_at_obs(samples, t_base=args.tbase_main, start_doy=int(best_stage1["start_doy"]))
    theta_obs_df = theta_obs_df.dropna(subset=["theta_obs"])

    scenario_rows = []
    for q in q_values:
        theta_q = float(theta_obs_df["theta_obs"].quantile(q))
        mae_q, rmse_q, _ = _eval_params(
            samples,
            t_base=args.tbase_main,
            theta=theta_q,
            start_doy=int(best_stage1["start_doy"]),
            use_rgdd_gate=args.use_rgdd_gate,
            rgdd_min=float(best_stage1["rgdd_min"]),
            rgdd_window=args.rgdd_window,
            delta_wk=float(best_stage2["delta_wk"]),
            delta_jf=float(best_stage2["delta_jf"]),
            delta_mpt=float(best_stage2["delta_mpt"]),
            use_variety_delta=True,
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

    trials_stage1.to_csv(OUT_DIR / "bbch53_trials_stage1_global.csv", index=False)
    trials_stage2.to_csv(OUT_DIR / "bbch53_trials_stage2_delta.csv", index=False)
    pd.DataFrame([{
        "t_base": float(best_stage1["t_base"]),
        "theta_agdd": float(best_stage1["theta_agdd"]),
        "start_doy": int(best_stage1["start_doy"]),
        "rgdd_min": float(best_stage1["rgdd_min"]),
        "delta_wk": float(best_stage2["delta_wk"]),
        "delta_jf": float(best_stage2["delta_jf"]),
        "delta_mpt": float(best_stage2["delta_mpt"]),
        "mae_days": float(fit_mae),
        "rmse_days": float(fit_rmse),
    }]).to_csv(OUT_DIR / "bbch53_trials.csv", index=False)

    pd.DataFrame(fit_detail).to_csv(OUT_DIR / "bbch53_fit_detail.csv", index=False)
    loocv.to_csv(OUT_DIR / "bbch53_loocv.csv", index=False)
    theta_obs_df.to_csv(OUT_DIR / "bbch53_theta_obs_tbase_main.csv", index=False)
    scenario_df.to_csv(OUT_DIR / "bbch53_main_setting_scenarios.csv", index=False)

    stage1_best = {
        "best_t_base": float(best_stage1["t_base"]),
        "best_theta_agdd": float(best_stage1["theta_agdd"]),
        "best_start_doy": int(best_stage1["start_doy"]),
        "best_rgdd_min": float(best_stage1["rgdd_min"]) if args.use_rgdd_gate else None,
        "fit_mae_days": float(best_stage1["mae_days"]),
        "fit_rmse_days": float(best_stage1["rmse_days"]),
        "grid_size": int(stage1_total),
    }
    (OUT_DIR / "bbch53_stage1_global_best.json").write_text(json.dumps(stage1_best, indent=2), encoding="utf-8")

    stage2_best = {
        "fixed_t_base": float(best_stage1["t_base"]),
        "fixed_theta_agdd": float(best_stage1["theta_agdd"]),
        "fixed_start_doy": int(best_stage1["start_doy"]),
        "fixed_rgdd_min": float(best_stage1["rgdd_min"]) if args.use_rgdd_gate else None,
        "best_delta_wk": float(best_stage2["delta_wk"]),
        "best_delta_jf": float(best_stage2["delta_jf"]),
        "best_delta_mpt": float(best_stage2["delta_mpt"]),
        "fit_mae_days": float(best_stage2["mae_days"]),
        "fit_rmse_days": float(best_stage2["rmse_days"]),
        "grid_size": int(stage2_total),
    }
    (OUT_DIR / "bbch53_stage2_delta_best.json").write_text(json.dumps(stage2_best, indent=2), encoding="utf-8")

    summary = {
        "best_t_base": float(best_stage1["t_base"]),
        "best_theta_agdd": float(best_stage1["theta_agdd"]),
        "best_start_doy": int(best_stage1["start_doy"]),
        "fit_mae_days": float(fit_mae),
        "fit_rmse_days": float(fit_rmse),
        "loocv_mae_days": float(loocv["abs_error_days"].mean()),
        "loocv_rmse_days": float(np.sqrt(np.mean(loocv["abs_error_days"].values ** 2))),
        "t_base_main_setting": float(args.tbase_main),
        "main_theta_quantiles": q_values,
        "main_theta_scenarios_file": "bbch53_main_setting_scenarios.csv",
        "best_rgdd_min": float(best_stage1["rgdd_min"]) if args.use_rgdd_gate else None,
        "best_delta_wk": float(best_stage2["delta_wk"]),
        "best_delta_jf": float(best_stage2["delta_jf"]),
        "best_delta_mpt": float(best_stage2["delta_mpt"]),
        "use_rgdd_gate": bool(args.use_rgdd_gate),
        "use_variety_delta": True,
        "rgdd_window": int(args.rgdd_window),
        "start_doy_min": int(args.start_doy_min),
        "start_doy_max": int(args.start_doy_max),
        "start_doy_step": int(args.start_doy_step),
        "calibration_mode": "two_stage_global_then_variety_delta",
        "t_base_search_enabled": bool(args.search_tbase),
        "t_base_grid_size": int(len(tbase_grid)),
        "start_doy_grid_size": int(len(start_doy_grid)),
        "theta_grid_size": int(len(theta_grid)),
        "rgdd_grid_size": int(len(rgdd_grid)),
        "delta_grid_size": int(len(delta_grid)),
        "stage1_total_grid": int(stage1_total),
        "stage2_total_grid": int(stage2_total),
        "variety_codes": ["WK", "JF", "MPT"],
        "n_site_year": int(len(samples)),
    }
    (OUT_DIR / "bbch53_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_DIR / 'bbch53_best.json'}")
    print(f"Saved: {OUT_DIR / 'bbch53_stage1_global_best.json'}")
    print(f"Saved: {OUT_DIR / 'bbch53_stage2_delta_best.json'}")
    print(f"Saved: {OUT_DIR / 'bbch53_trials_stage1_global.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_trials_stage2_delta.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_trials.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_fit_detail.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_loocv.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_theta_obs_tbase_main.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_main_setting_scenarios.csv'}")


if __name__ == "__main__":
    main()
