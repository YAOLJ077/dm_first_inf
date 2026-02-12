#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

'''1'''
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
    from src.bbch53_module import compute_agdd_hourly, load_hourly_weather_from_json
except Exception:  # pragma: no cover
    from bbch53_module import compute_agdd_hourly, load_hourly_weather_from_json

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "result" / "historical_inputs" / "bbch53_calibration"
PLOT_TO_VAR_CODE = {"GL": "WK", "NN": "JF", "DA": "MPT"}


@dataclass
class Sample:
    plot: str
    year: int
    obs_bbch53: pd.Timestamp
    hourly: pd.DataFrame
    variety_code: str
    year_signal: float = 0.0


FIXED_WK_PRIOR = {"delta": -0.05, "shift": 6}


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
    raw_year_signal: List[float] = []
    for r in df.itertuples(index=False):
        plot = str(r.plot)
        year = int(r.year)
        hourly = _load_site_year_hourly(plot, str(r.variety), year)
        jan_feb = hourly[
            (hourly["datetime"].dt.year == year)
            & (hourly["datetime"].dt.month >= 1)
            & (hourly["datetime"].dt.month <= 2)
        ]
        raw_sig = float(jan_feb["air_temp"].mean()) if not jan_feb.empty else float(hourly["air_temp"].mean())
        raw_year_signal.append(raw_sig)
        out.append(
            Sample(
                plot=plot,
                year=year,
                obs_bbch53=pd.to_datetime(r.bbch_date).normalize(),
                hourly=hourly,
                variety_code=PLOT_TO_VAR_CODE.get(plot, "UNK"),
                year_signal=0.0,
            )
        )
    arr = np.asarray(raw_year_signal, dtype=float)
    mu = float(np.nanmean(arr)) if len(arr) > 0 else 0.0
    sd = float(np.nanstd(arr)) if len(arr) > 0 else 0.0
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    for i, s in enumerate(out):
        z = (float(raw_year_signal[i]) - mu) / sd if np.isfinite(raw_year_signal[i]) else 0.0
        s.year_signal = float(z)
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
    k_mpt: float = 0.0,
    start_shift_wk: int = 0,
    start_shift_jf: int = 0,
    start_shift_mpt: int = 0,
    penalty_days: float = 120.0,
) -> Tuple[float, float, List[Dict]]:
    dmap = {"WK": float(delta_wk), "JF": float(delta_jf), "MPT": float(delta_mpt)}
    smap = {"WK": int(start_shift_wk), "JF": int(start_shift_jf), "MPT": int(start_shift_mpt)}
    errs: List[float] = []
    details: List[Dict] = []
    for s in samples:
        delta = dmap.get(s.variety_code, 0.0)
        sshift = int(smap.get(s.variety_code, 0))
        year_term = float(k_mpt) * float(s.year_signal) if str(s.variety_code) == "MPT" else 0.0
        scale = 1.0 + float(delta) + year_term
        scale = float(np.clip(scale, 0.2, 3.0))
        theta_eff = float(theta) * scale
        start_doy_eff = int(np.clip(int(start_doy) + int(sshift), 1, 366))
        start_dt = _start_dt_for_year_from_doy(s.year, start_doy_eff)
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
                "start_doy": int(start_doy_eff),
                "start_doy_base": int(start_doy),
                "start_date_used": start_dt,
                "theta_eff": float(theta_eff),
                "delta_variety": float(delta),
                "k_mpt": float(k_mpt) if str(s.variety_code) == "MPT" else 0.0,
                "year_signal": float(s.year_signal),
                "start_shift_variety": int(sshift),
                "rgdd_gate_used": bool(use_rgdd_gate),
                "rgdd_min": float(rgdd_min) if use_rgdd_gate else np.nan,
            }
        )
    abs_err = np.abs(np.asarray(errs, dtype=float))
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(abs_err**2)))
    return mae, rmse, details


if ray is not None:
    @ray.remote
    def _eval_combo_ray(
        samples_ref,
        params: Tuple[float, float, int, float, float, float, float, int, int, int],
        settings: Dict,
    ) -> Dict:
        samples = ray.get(samples_ref) if isinstance(samples_ref, ray.ObjectRef) else samples_ref
        tb, th, sd, rg, dw, dj, dm, sw, sj, sm = params
        mae, rmse, _ = _eval_params(
            samples=samples,
            t_base=float(tb),
            theta=float(th),
            start_doy=int(sd),
            use_rgdd_gate=bool(settings["use_rgdd_gate"]),
            rgdd_min=float(rg),
            rgdd_window=int(settings["rgdd_window"]),
            delta_wk=float(dw),
            delta_jf=float(dj),
            delta_mpt=float(dm),
            start_shift_wk=int(sw),
            start_shift_jf=int(sj),
            start_shift_mpt=int(sm),
        )
        return {
            "t_base": float(tb),
            "theta_agdd": float(th),
            "start_doy": int(sd),
            "rgdd_min": float(rg),
            "delta_wk": float(dw),
            "delta_jf": float(dj),
            "delta_mpt": float(dm),
            "start_shift_wk": int(sw),
            "start_shift_jf": int(sj),
            "start_shift_mpt": int(sm),
            "mae_days": float(mae),
            "rmse_days": float(rmse),
        }


def _run_grid(
    samples: List[Sample],
    tbase_grid: np.ndarray,
    theta_grid: np.ndarray,
    start_doy_grid: np.ndarray,
    rgdd_grid: np.ndarray,
    delta_grid: np.ndarray,
    shift_grid: np.ndarray,
    n_jobs: int,
    use_rgdd_gate: bool,
    rgdd_window: int,
    backend: str,
    desc: str,
) -> pd.DataFrame:
    total = (
        len(tbase_grid)
        * len(theta_grid)
        * len(start_doy_grid)
        * len(rgdd_grid)
        * len(delta_grid)
        * len(delta_grid)
        * len(delta_grid)
        * len(shift_grid)
        * len(shift_grid)
        * len(shift_grid)
    )
    combos = product(
        tbase_grid,
        theta_grid,
        start_doy_grid,
        rgdd_grid,
        delta_grid,
        delta_grid,
        delta_grid,
        shift_grid,
        shift_grid,
        shift_grid,
    )
    rows: List[Dict] = []

    use_ray = backend == "ray" and ray is not None and n_jobs > 1
    if not use_ray:
        for tb, th, sd, rg, dw, dj, dm, sw, sj, sm in tqdm(combos, total=total, desc=desc):
            mae, rmse, _ = _eval_params(
                samples=samples,
                t_base=float(tb),
                theta=float(th),
                start_doy=int(sd),
                use_rgdd_gate=use_rgdd_gate,
                rgdd_min=float(rg),
                rgdd_window=int(rgdd_window),
                delta_wk=float(dw),
                delta_jf=float(dj),
                delta_mpt=float(dm),
                start_shift_wk=int(sw),
                start_shift_jf=int(sj),
                start_shift_mpt=int(sm),
            )
            rows.append(
                {
                    "t_base": float(tb),
                    "theta_agdd": float(th),
                    "start_doy": int(sd),
                    "rgdd_min": float(rg),
                    "delta_wk": float(dw),
                    "delta_jf": float(dj),
                    "delta_mpt": float(dm),
                    "start_shift_wk": int(sw),
                    "start_shift_jf": int(sj),
                    "start_shift_mpt": int(sm),
                    "mae_days": float(mae),
                    "rmse_days": float(rmse),
                }
            )
    else:
        host_cpus = os.cpu_count() or 1
        win_cap = 61 if os.name == "nt" else host_cpus
        effective_jobs = max(1, min(int(n_jobs), int(host_cpus), int(win_cap)))
        ray.init(num_cpus=effective_jobs, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True)
        settings_ref = ray.put({"use_rgdd_gate": bool(use_rgdd_gate), "rgdd_window": int(rgdd_window)})
        samples_ref = ray.put(samples)
        pending = {}
        pbar = tqdm(total=total, desc=desc)
        inflight_max = max(2 * effective_jobs, 1)
        done_submitting = False

        def _submit_one(it):
            nonlocal done_submitting
            try:
                c = next(it)
            except StopIteration:
                done_submitting = True
                return
            fut = _eval_combo_ray.remote(samples_ref, c, settings_ref)
            pending[fut] = True

        combo_iter = iter(combos)
        while len(pending) < inflight_max and not done_submitting:
            _submit_one(combo_iter)

        while pending:
            ready, _ = ray.wait(list(pending.keys()), num_returns=1)
            for r in ready:
                rows.append(ray.get(r))
                pending.pop(r, None)
                pbar.update(1)
                if not done_submitting:
                    _submit_one(combo_iter)
        pbar.close()
        ray.shutdown()

    return pd.DataFrame(rows).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)


def _topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0:
        return df
    return df.head(min(k, len(df))).copy()


def _build_shift_candidates(
    t_base: float,
    theta_agdd: float,
    start_doy: int,
    rgdd_min: float,
    delta_wk: float,
    delta_jf: float,
    delta_mpt: float,
    shift_grid: np.ndarray,
    desc: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    s = shift_grid.astype(int)
    it = product(s, s, s)
    total = int(len(s) ** 3)
    if desc:
        it = tqdm(it, total=total, desc=desc)
    for sw, sj, sm in it:
        rows.append(
            {
                "t_base": float(t_base),
                "theta_agdd": float(theta_agdd),
                "start_doy": int(start_doy),
                "rgdd_min": float(rgdd_min),
                "delta_wk": float(delta_wk),
                "delta_jf": float(delta_jf),
                "delta_mpt": float(delta_mpt),
                "start_shift_wk": int(sw),
                "start_shift_jf": int(sj),
                "start_shift_mpt": int(sm),
            }
        )
    return pd.DataFrame(rows)


def _normalize_variety_params_for_train(
    train_samples: List[Sample],
    delta_wk: float,
    delta_jf: float,
    delta_mpt: float,
    start_shift_wk: int,
    start_shift_jf: int,
    start_shift_mpt: int,
    unseen_variety_strategy: str,
    prior_delta_wk: float,
    prior_delta_jf: float,
    prior_delta_mpt: float,
    prior_start_shift_wk: int,
    prior_start_shift_jf: int,
    prior_start_shift_mpt: int,
) -> Tuple[float, float, float, int, int, int]:
    present = {str(s.variety_code) for s in train_samples}
    use_prior = str(unseen_variety_strategy).lower() == "global"
    dw = float(delta_wk) if "WK" in present else (float(prior_delta_wk) if use_prior else 0.0)
    dj = float(delta_jf) if "JF" in present else (float(prior_delta_jf) if use_prior else 0.0)
    dm = float(delta_mpt) if "MPT" in present else (float(prior_delta_mpt) if use_prior else 0.0)
    sw = int(start_shift_wk) if "WK" in present else (int(prior_start_shift_wk) if use_prior else 0)
    sj = int(start_shift_jf) if "JF" in present else (int(prior_start_shift_jf) if use_prior else 0)
    sm = int(start_shift_mpt) if "MPT" in present else (int(prior_start_shift_mpt) if use_prior else 0)
    return dw, dj, dm, sw, sj, sm


def _score_candidates_on_train(
    train_samples: List[Sample],
    candidates: pd.DataFrame,
    use_rgdd_gate: bool,
    rgdd_window: int,
    unseen_variety_strategy: str,
    prior_delta_wk: float,
    prior_delta_jf: float,
    prior_delta_mpt: float,
    prior_start_shift_wk: int,
    prior_start_shift_jf: int,
    prior_start_shift_mpt: int,
) -> pd.DataFrame:
    scored = []
    for r in candidates.itertuples(index=False):
        dw, dj, dm, sw, sj, sm = _normalize_variety_params_for_train(
            train_samples=train_samples,
            delta_wk=float(r.delta_wk),
            delta_jf=float(r.delta_jf),
            delta_mpt=float(r.delta_mpt),
            start_shift_wk=int(r.start_shift_wk),
            start_shift_jf=int(r.start_shift_jf),
            start_shift_mpt=int(r.start_shift_mpt),
            unseen_variety_strategy=unseen_variety_strategy,
            prior_delta_wk=float(prior_delta_wk),
            prior_delta_jf=float(prior_delta_jf),
            prior_delta_mpt=float(prior_delta_mpt),
            prior_start_shift_wk=int(prior_start_shift_wk),
            prior_start_shift_jf=int(prior_start_shift_jf),
            prior_start_shift_mpt=int(prior_start_shift_mpt),
        )
        mae, rmse, _ = _eval_params(
            samples=train_samples,
            t_base=float(r.t_base),
            theta=float(r.theta_agdd),
            start_doy=int(r.start_doy),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(r.rgdd_min),
            rgdd_window=int(rgdd_window),
            delta_wk=float(dw),
            delta_jf=float(dj),
            delta_mpt=float(dm),
            start_shift_wk=int(sw),
            start_shift_jf=int(sj),
            start_shift_mpt=int(sm),
        )
        d = {
            "t_base": float(r.t_base),
            "theta_agdd": float(r.theta_agdd),
            "start_doy": int(r.start_doy),
            "rgdd_min": float(r.rgdd_min),
            "delta_wk": float(dw),
            "delta_jf": float(dj),
            "delta_mpt": float(dm),
            "start_shift_wk": int(sw),
            "start_shift_jf": int(sj),
            "start_shift_mpt": int(sm),
            "mae_days": float(mae),
            "rmse_days": float(rmse),
        }
        scored.append(d)
    return pd.DataFrame(scored).sort_values(["mae_days", "rmse_days"]).reset_index(drop=True)


def _pick_candidate_train_topk_then_val(
    train_samples: List[Sample],
    test_samples: List[Sample],
    candidates: pd.DataFrame,
    loocv_topk_frac: float,
    use_rgdd_gate: bool,
    rgdd_window: int,
    unseen_variety_strategy: str,
    prior_delta_wk: float,
    prior_delta_jf: float,
    prior_delta_mpt: float,
    prior_start_shift_wk: int,
    prior_start_shift_jf: int,
    prior_start_shift_mpt: int,
    desc: Optional[str] = None,
) -> Dict:
    scored = _score_candidates_on_train(
        train_samples=train_samples,
        candidates=candidates,
        use_rgdd_gate=use_rgdd_gate,
        rgdd_window=rgdd_window,
        unseen_variety_strategy=unseen_variety_strategy,
        prior_delta_wk=prior_delta_wk,
        prior_delta_jf=prior_delta_jf,
        prior_delta_mpt=prior_delta_mpt,
        prior_start_shift_wk=prior_start_shift_wk,
        prior_start_shift_jf=prior_start_shift_jf,
        prior_start_shift_mpt=prior_start_shift_mpt,
    )
    k = max(1, int(np.ceil(len(scored) * float(loocv_topk_frac))))
    top = scored.head(k).copy()

    val_rows = []
    it = top.itertuples(index=False)
    if desc:
        it = tqdm(it, total=len(top), desc=desc)
    for r in it:
        val_mae, val_rmse, detail = _eval_params(
            samples=test_samples,
            t_base=float(r.t_base),
            theta=float(r.theta_agdd),
            start_doy=int(r.start_doy),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(r.rgdd_min),
            rgdd_window=rgdd_window,
            delta_wk=float(r.delta_wk),
            delta_jf=float(r.delta_jf),
            delta_mpt=float(r.delta_mpt),
            start_shift_wk=int(r.start_shift_wk),
            start_shift_jf=int(r.start_shift_jf),
            start_shift_mpt=int(r.start_shift_mpt),
        )
        val_rows.append(
            {
                "t_base": float(r.t_base),
                "theta_agdd": float(r.theta_agdd),
                "start_doy": int(r.start_doy),
                "rgdd_min": float(r.rgdd_min),
                "delta_wk": float(r.delta_wk),
                "delta_jf": float(r.delta_jf),
                "delta_mpt": float(r.delta_mpt),
                "start_shift_wk": int(r.start_shift_wk),
                "start_shift_jf": int(r.start_shift_jf),
                "start_shift_mpt": int(r.start_shift_mpt),
                "train_mae_days": float(r.mae_days),
                "train_rmse_days": float(r.rmse_days),
                "val_mae_days": float(val_mae),
                "val_rmse_days": float(val_rmse),
                "val_abs_error_days": float(detail[0]["abs_error_days"]),
            }
        )
    val_df = pd.DataFrame(val_rows)
    valid = val_df[val_df["val_abs_error_days"] >= val_df["train_mae_days"]]
    choose_df = valid if not valid.empty else val_df
    best = choose_df.sort_values(["val_abs_error_days", "train_mae_days", "train_rmse_days"]).iloc[0].to_dict()
    best["topk_pool_size"] = int(k)
    return best


def _select_variety_params_fold(
    train_samples: List[Sample],
    test_samples: List[Sample],
    variety_code: str,
    t_base: float,
    theta_agdd: float,
    start_doy: int,
    rgdd_min: float,
    kmpt_grid: np.ndarray,
    delta_grid: np.ndarray,
    shift_grid: np.ndarray,
    loocv_topk_frac: float,
    use_rgdd_gate: bool,
    rgdd_window: int,
    unseen_variety_strategy: str,
    prior_delta: float,
    prior_shift: int,
    penalty_delta_l1: float = 0.0,
    penalty_shift_l1: float = 0.0,
    desc: Optional[str] = None,
) -> Dict:
    train_v = [s for s in train_samples if str(s.variety_code) == str(variety_code)]
    if not train_v:
        if str(unseen_variety_strategy).lower() == "global":
            return {"delta": float(prior_delta), "shift": int(prior_shift), "k_mpt": 0.0, "topk_pool_size": 0, "train_mae_days": np.nan, "train_rmse_days": np.nan}
        return {"delta": 0.0, "shift": 0, "k_mpt": 0.0, "topk_pool_size": 0, "train_mae_days": np.nan, "train_rmse_days": np.nan}

    cand_rows = []
    sgrid = shift_grid.astype(int)
    kgrid = kmpt_grid if str(variety_code) == "MPT" else np.array([0.0], dtype=float)
    it = product(delta_grid, sgrid, kgrid)
    if desc:
        it = tqdm(it, total=int(len(delta_grid) * len(sgrid) * len(kgrid)), desc=desc)
    for d, s, km in it:
        dw = float(d) if variety_code == "WK" else 0.0
        dj = float(d) if variety_code == "JF" else 0.0
        dm = float(d) if variety_code == "MPT" else 0.0
        sw = int(s) if variety_code == "WK" else 0
        sj = int(s) if variety_code == "JF" else 0
        sm = int(s) if variety_code == "MPT" else 0
        mae, rmse, _ = _eval_params(
            samples=train_v,
            t_base=float(t_base),
            theta=float(theta_agdd),
            start_doy=int(start_doy),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(rgdd_min),
            rgdd_window=int(rgdd_window),
            delta_wk=float(dw),
            delta_jf=float(dj),
            delta_mpt=float(dm),
            k_mpt=float(km) if str(variety_code) == "MPT" else 0.0,
            start_shift_wk=int(sw),
            start_shift_jf=int(sj),
            start_shift_mpt=int(sm),
        )
        obj = float(mae) + float(penalty_delta_l1) * abs(float(d)) + float(penalty_shift_l1) * abs(float(s))
        cand_rows.append(
            {
                "delta": float(d),
                "shift": int(s),
                "k_mpt": float(km) if str(variety_code) == "MPT" else 0.0,
                "train_obj_days": float(obj),
                "train_mae_days": float(mae),
                "train_rmse_days": float(rmse),
            }
        )
    cdf = pd.DataFrame(cand_rows).sort_values(["train_obj_days", "train_mae_days", "train_rmse_days"]).reset_index(drop=True)

    test_v = [s for s in test_samples if str(s.variety_code) == str(variety_code)]
    if not test_v:
        b = cdf.iloc[0].to_dict()
        b["topk_pool_size"] = 0
        return b

    k = max(1, int(np.ceil(len(cdf) * float(loocv_topk_frac))))
    top = cdf.head(k).copy()
    vals = []
    for r in top.itertuples(index=False):
        dw = float(r.delta) if variety_code == "WK" else 0.0
        dj = float(r.delta) if variety_code == "JF" else 0.0
        dm = float(r.delta) if variety_code == "MPT" else 0.0
        sw = int(r.shift) if variety_code == "WK" else 0
        sj = int(r.shift) if variety_code == "JF" else 0
        sm = int(r.shift) if variety_code == "MPT" else 0
        vmae, vrmse, det = _eval_params(
            samples=test_v,
            t_base=float(t_base),
            theta=float(theta_agdd),
            start_doy=int(start_doy),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(rgdd_min),
            rgdd_window=int(rgdd_window),
            delta_wk=float(dw),
            delta_jf=float(dj),
            delta_mpt=float(dm),
            k_mpt=float(r.k_mpt) if str(variety_code) == "MPT" else 0.0,
            start_shift_wk=int(sw),
            start_shift_jf=int(sj),
            start_shift_mpt=int(sm),
        )
        vals.append(
            {
                "delta": float(r.delta),
                "shift": int(r.shift),
                "k_mpt": float(r.k_mpt) if str(variety_code) == "MPT" else 0.0,
                "train_mae_days": float(r.train_mae_days),
                "train_rmse_days": float(r.train_rmse_days),
                "train_obj_days": float(r.train_obj_days),
                "val_mae_days": float(vmae),
                "val_rmse_days": float(vrmse),
                "val_abs_error_days": float(np.mean([float(x["abs_error_days"]) for x in det])),
            }
        )
    vdf = pd.DataFrame(vals)
    valid = vdf[vdf["val_abs_error_days"] >= vdf["train_mae_days"]]
    choose = valid if not valid.empty else vdf
    b = choose.sort_values(["val_abs_error_days", "train_obj_days", "train_mae_days", "train_rmse_days"]).iloc[0].to_dict()
    b["topk_pool_size"] = int(k)
    return b


def _fit_variety_prior_from_all_samples(
    samples: List[Sample],
    variety_code: str,
    t_base: float,
    theta_agdd: float,
    start_doy: int,
    rgdd_min: float,
    kmpt_grid: np.ndarray,
    delta_grid: np.ndarray,
    shift_grid: np.ndarray,
    use_rgdd_gate: bool,
    rgdd_window: int,
    penalty_delta_l1: float = 0.0,
    penalty_shift_l1: float = 0.0,
) -> Dict:
    subset = [s for s in samples if str(s.variety_code) == str(variety_code)]
    if not subset:
        return {"delta": 0.0, "shift": 0, "k_mpt": 0.0, "mae_days": np.nan, "rmse_days": np.nan}
    rows = []
    kgrid = kmpt_grid if str(variety_code) == "MPT" else np.array([0.0], dtype=float)
    for d, s, km in product(delta_grid, shift_grid.astype(int), kgrid):
        dw = float(d) if variety_code == "WK" else 0.0
        dj = float(d) if variety_code == "JF" else 0.0
        dm = float(d) if variety_code == "MPT" else 0.0
        sw = int(s) if variety_code == "WK" else 0
        sj = int(s) if variety_code == "JF" else 0
        sm = int(s) if variety_code == "MPT" else 0
        mae, rmse, _ = _eval_params(
            samples=subset,
            t_base=float(t_base),
            theta=float(theta_agdd),
            start_doy=int(start_doy),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(rgdd_min),
            rgdd_window=int(rgdd_window),
            delta_wk=float(dw),
            delta_jf=float(dj),
            delta_mpt=float(dm),
            k_mpt=float(km) if str(variety_code) == "MPT" else 0.0,
            start_shift_wk=int(sw),
            start_shift_jf=int(sj),
            start_shift_mpt=int(sm),
        )
        obj = float(mae) + float(penalty_delta_l1) * abs(float(d)) + float(penalty_shift_l1) * abs(float(s))
        rows.append({"delta": float(d), "shift": int(s), "k_mpt": float(km) if str(variety_code) == "MPT" else 0.0, "obj_days": float(obj), "mae_days": float(mae), "rmse_days": float(rmse)})
    best = pd.DataFrame(rows).sort_values(["obj_days", "mae_days", "rmse_days"]).iloc[0].to_dict()
    return {"delta": float(best["delta"]), "shift": int(best["shift"]), "k_mpt": float(best["k_mpt"]), "mae_days": float(best["mae_days"]), "rmse_days": float(best["rmse_days"])}


def _build_variety_stratified_folds(samples: List[Sample], n_folds: int) -> List[List[int]]:
    n_folds = int(n_folds)
    if n_folds >= len(samples):
        # LOOCV: one sample per fold
        return [[i] for i in range(len(samples))]
    n_folds = max(2, n_folds)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    by_var: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_var.setdefault(str(s.variety_code), []).append(i)
    for k in by_var:
        by_var[k] = sorted(by_var[k], key=lambda idx: (samples[idx].year, samples[idx].plot))
    for _, ids in sorted(by_var.items()):
        for j, idx in enumerate(ids):
            folds[j % n_folds].append(idx)
    return [sorted(f) for f in folds]


def _kfold_full_grid(
    samples: List[Sample],
    tbase_grid: np.ndarray,
    theta_grid: np.ndarray,
    start_doy_grid: np.ndarray,
    rgdd_grid: np.ndarray,
    kmpt_grid: np.ndarray,
    delta_grid: np.ndarray,
    shift_grid: np.ndarray,
    loocv_topk_frac: float,
    n_jobs: int,
    use_rgdd_gate: bool,
    rgdd_window: int,
    backend: str,
    unseen_variety_strategy: str,
    prior_delta_wk: float,
    prior_delta_jf: float,
    prior_delta_mpt: float,
    prior_start_shift_wk: int,
    prior_start_shift_jf: int,
    prior_start_shift_mpt: int,
    prior_k_mpt: float,
    wk_fixed_enable: bool,
    wk_fixed_delta: float,
    wk_fixed_shift: int,
    penalty_delta_l1: float,
    penalty_shift_l1: float,
    n_folds: int,
) -> pd.DataFrame:
    folds = _build_variety_stratified_folds(samples=samples, n_folds=n_folds)
    rows = []
    cv_desc = "LOOCV" if int(n_folds) >= len(samples) else f"{int(n_folds)}-fold CV"
    for i in tqdm(range(len(folds)), desc=cv_desc):
        test_idx = set(folds[i])
        train = [samples[j] for j in range(len(samples)) if j not in test_idx]
        test = [samples[j] for j in sorted(test_idx)]

        trials_stage1_fold = _run_grid(
            samples=train,
            tbase_grid=tbase_grid,
            theta_grid=theta_grid,
            start_doy_grid=start_doy_grid,
            rgdd_grid=rgdd_grid,
            delta_grid=np.array([0.0]),
            shift_grid=np.array([0], dtype=int),
            n_jobs=int(n_jobs),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_window=rgdd_window,
            backend=backend,
            desc=f"CV fold {i + 1}/{len(folds)} stage1",
        )
        best_stage1_fold = trials_stage1_fold.iloc[0].to_dict()

        # Stage2: per-variety independent calibration of (delta, start_shift)
        if wk_fixed_enable:
            wk = {
                "delta": float(wk_fixed_delta),
                "shift": int(wk_fixed_shift),
                "k_mpt": 0.0,
                "topk_pool_size": 0,
                "train_mae_days": np.nan,
                "train_rmse_days": np.nan,
            }
        else:
            wk = _select_variety_params_fold(
                train_samples=train,
                test_samples=test,
                variety_code="WK",
                t_base=float(best_stage1_fold["t_base"]),
                theta_agdd=float(best_stage1_fold["theta_agdd"]),
                start_doy=int(best_stage1_fold["start_doy"]),
                rgdd_min=float(best_stage1_fold["rgdd_min"]),
                kmpt_grid=kmpt_grid,
                delta_grid=delta_grid,
                shift_grid=shift_grid,
                loocv_topk_frac=loocv_topk_frac,
                use_rgdd_gate=use_rgdd_gate,
                rgdd_window=rgdd_window,
                unseen_variety_strategy=unseen_variety_strategy,
                prior_delta=float(prior_delta_wk),
                prior_shift=int(prior_start_shift_wk),
                penalty_delta_l1=penalty_delta_l1,
                penalty_shift_l1=penalty_shift_l1,
                desc=f"CV fold {i + 1}/{len(folds)} stage2-WK",
            )
        jf = _select_variety_params_fold(
            train_samples=train,
            test_samples=test,
            variety_code="JF",
            t_base=float(best_stage1_fold["t_base"]),
            theta_agdd=float(best_stage1_fold["theta_agdd"]),
            start_doy=int(best_stage1_fold["start_doy"]),
            rgdd_min=float(best_stage1_fold["rgdd_min"]),
            kmpt_grid=kmpt_grid,
            delta_grid=delta_grid,
            shift_grid=shift_grid,
            loocv_topk_frac=loocv_topk_frac,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_window=rgdd_window,
            unseen_variety_strategy=unseen_variety_strategy,
            prior_delta=float(prior_delta_jf),
            prior_shift=int(prior_start_shift_jf),
            penalty_delta_l1=penalty_delta_l1,
            penalty_shift_l1=penalty_shift_l1,
            desc=f"CV fold {i + 1}/{len(folds)} stage2-JF",
        )
        mpt = _select_variety_params_fold(
            train_samples=train,
            test_samples=test,
            variety_code="MPT",
            t_base=float(best_stage1_fold["t_base"]),
            theta_agdd=float(best_stage1_fold["theta_agdd"]),
            start_doy=int(best_stage1_fold["start_doy"]),
            rgdd_min=float(best_stage1_fold["rgdd_min"]),
            kmpt_grid=kmpt_grid,
            delta_grid=delta_grid,
            shift_grid=shift_grid,
            loocv_topk_frac=loocv_topk_frac,
            use_rgdd_gate=use_rgdd_gate,
            rgdd_window=rgdd_window,
            unseen_variety_strategy=unseen_variety_strategy,
            prior_delta=float(prior_delta_mpt),
            prior_shift=int(prior_start_shift_mpt),
            penalty_delta_l1=penalty_delta_l1,
            penalty_shift_l1=penalty_shift_l1,
            desc=f"CV fold {i + 1}/{len(folds)} stage2-MPT",
        )
        best_fold = {
            "t_base": float(best_stage1_fold["t_base"]),
            "theta_agdd": float(best_stage1_fold["theta_agdd"]),
            "start_doy": int(best_stage1_fold["start_doy"]),
            "rgdd_min": float(best_stage1_fold["rgdd_min"]),
            "delta_wk": float(wk["delta"]),
            "delta_jf": float(jf["delta"]),
            "delta_mpt": float(mpt["delta"]),
            "k_mpt": float(mpt.get("k_mpt", prior_k_mpt)),
            "start_shift_wk": int(wk["shift"]),
            "start_shift_jf": int(jf["shift"]),
            "start_shift_mpt": int(mpt["shift"]),
            "train_mae_days": float(np.nanmean([wk["train_mae_days"], jf["train_mae_days"], mpt["train_mae_days"]])),
            "train_rmse_days": float(np.nanmean([wk["train_rmse_days"], jf["train_rmse_days"], mpt["train_rmse_days"]])),
            "topk_pool_size": int(max(wk["topk_pool_size"], jf["topk_pool_size"], mpt["topk_pool_size"])),
        }
        mae, rmse, detail = _eval_params(
            samples=test,
            t_base=float(best_fold["t_base"]),
            theta=float(best_fold["theta_agdd"]),
            start_doy=int(best_fold["start_doy"]),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(best_fold["rgdd_min"]),
            rgdd_window=rgdd_window,
            delta_wk=float(best_fold["delta_wk"]),
            delta_jf=float(best_fold["delta_jf"]),
            delta_mpt=float(best_fold["delta_mpt"]),
            k_mpt=float(best_fold["k_mpt"]),
            start_shift_wk=int(best_fold["start_shift_wk"]),
            start_shift_jf=int(best_fold["start_shift_jf"]),
            start_shift_mpt=int(best_fold["start_shift_mpt"]),
        )

        for d in detail:
            rows.append(
                {
                    "plot": d["plot"],
                    "year": d["year"],
                    "obs_bbch53": d["obs_bbch53"],
                    "pred_bbch53": d["pred_bbch53"],
                    "error_days": d["error_days"],
                    "abs_error_days": d["abs_error_days"],
                    "fit_t_base": float(best_fold["t_base"]),
                    "fit_theta_agdd": float(best_fold["theta_agdd"]),
                    "fit_start_doy": int(best_fold["start_doy"]),
                    "fit_rgdd_min": float(best_fold["rgdd_min"]),
                    "fit_k_mpt": float(best_fold["k_mpt"]),
                    "fit_delta_wk": float(best_fold["delta_wk"]),
                    "fit_delta_jf": float(best_fold["delta_jf"]),
                    "fit_delta_mpt": float(best_fold["delta_mpt"]),
                    "fit_start_shift_wk": int(best_fold["start_shift_wk"]),
                    "fit_start_shift_jf": int(best_fold["start_shift_jf"]),
                    "fit_start_shift_mpt": int(best_fold["start_shift_mpt"]),
                    "fit_train_mae_days": float(best_fold["train_mae_days"]),
                    "fit_train_rmse_days": float(best_fold["train_rmse_days"]),
                    "fit_topk_pool_size": int(best_fold["topk_pool_size"]),
                    "test_mae_days": float(mae),
                    "test_rmse_days": float(rmse),
                    "cv_fold": int(i + 1),
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
        rows.append({"plot": s.plot, "year": s.year, "obs_bbch53": s.obs_bbch53, "t_base_main": float(t_base), "theta_obs": theta_obs})
    return pd.DataFrame(rows)


def _build_candidate_leaderboard_from_cv_rows(
    samples: List[Sample],
    cv_df: pd.DataFrame,
    n_folds: int,
    use_rgdd_gate: bool,
    rgdd_window: int,
) -> pd.DataFrame:
    key_cols = [
        "fit_t_base",
        "fit_theta_agdd",
        "fit_start_doy",
        "fit_rgdd_min",
        "fit_k_mpt",
        "fit_delta_wk",
        "fit_delta_jf",
        "fit_delta_mpt",
        "fit_start_shift_wk",
        "fit_start_shift_jf",
        "fit_start_shift_mpt",
    ]
    gate_df = (
        cv_df.groupby(key_cols, dropna=False, as_index=False)
        .agg(
            gate_cv_mae_days=("abs_error_days", "mean"),
            gate_cv_rmse_days=("abs_error_days", lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))),
            n_selected_folds=("cv_fold", "nunique"),
        )
        .reset_index(drop=True)
    )
    rows = []
    for r in gate_df.itertuples(index=False):
        fit_mae, fit_rmse, _ = _eval_params(
            samples=samples,
            t_base=float(r.fit_t_base),
            theta=float(r.fit_theta_agdd),
            start_doy=int(r.fit_start_doy),
            use_rgdd_gate=use_rgdd_gate,
            rgdd_min=float(r.fit_rgdd_min),
            rgdd_window=int(rgdd_window),
            k_mpt=float(r.fit_k_mpt),
            delta_wk=float(r.fit_delta_wk),
            delta_jf=float(r.fit_delta_jf),
            delta_mpt=float(r.fit_delta_mpt),
            start_shift_wk=int(r.fit_start_shift_wk),
            start_shift_jf=int(r.fit_start_shift_jf),
            start_shift_mpt=int(r.fit_start_shift_mpt),
        )
        rows.append(
            {
                "fit_t_base": float(r.fit_t_base),
                "fit_theta_agdd": float(r.fit_theta_agdd),
                "fit_start_doy": int(r.fit_start_doy),
                "fit_rgdd_min": float(r.fit_rgdd_min),
                "fit_k_mpt": float(r.fit_k_mpt),
                "fit_delta_wk": float(r.fit_delta_wk),
                "fit_delta_jf": float(r.fit_delta_jf),
                "fit_delta_mpt": float(r.fit_delta_mpt),
                "fit_start_shift_wk": int(r.fit_start_shift_wk),
                "fit_start_shift_jf": int(r.fit_start_shift_jf),
                "fit_start_shift_mpt": int(r.fit_start_shift_mpt),
                "gate_cv_mae_days": float(r.gate_cv_mae_days),
                "gate_cv_rmse_days": float(r.gate_cv_rmse_days),
                "n_selected_folds": int(r.n_selected_folds),
                "fit_mae_days": float(fit_mae),
                "fit_rmse_days": float(fit_rmse),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _pick_final_from_leaderboard(
    leaderboard: pd.DataFrame,
    selector: str,
    min_selected_folds: int = 1,
    allow_relax: bool = True,
) -> Optional[Dict]:
    eligible = leaderboard[leaderboard["n_selected_folds"] >= int(min_selected_folds)].copy()
    relaxed = False
    if eligible.empty:
        if not allow_relax:
            return None
        eligible = leaderboard.copy()
        relaxed = True
    if str(selector).lower() == "fit":
        b = eligible.sort_values(["fit_mae_days", "fit_rmse_days", "gate_cv_mae_days", "gate_cv_rmse_days"]).iloc[0]
    else:
        b = eligible.sort_values(["gate_cv_mae_days", "gate_cv_rmse_days", "fit_mae_days", "fit_rmse_days"]).iloc[0]
    return {
        "t_base": float(b["fit_t_base"]),
        "theta_agdd": float(b["fit_theta_agdd"]),
        "start_doy": int(b["fit_start_doy"]),
        "rgdd_min": float(b["fit_rgdd_min"]),
        "k_mpt": float(b["fit_k_mpt"]),
        "delta_wk": float(b["fit_delta_wk"]),
        "delta_jf": float(b["fit_delta_jf"]),
        "delta_mpt": float(b["fit_delta_mpt"]),
        "start_shift_wk": int(b["fit_start_shift_wk"]),
        "start_shift_jf": int(b["fit_start_shift_jf"]),
        "start_shift_mpt": int(b["fit_start_shift_mpt"]),
        "selected_by_cv_mae_days": float(b["gate_cv_mae_days"]),
        "selected_by_cv_rmse_days": float(b["gate_cv_rmse_days"]),
        "selected_by_fit_mae_days": float(b["fit_mae_days"]),
        "selected_by_fit_rmse_days": float(b["fit_rmse_days"]),
        "selected_n_folds": int(b["n_selected_folds"]),
        "gate_min_selected_folds": int(min_selected_folds),
        "gate_constraint_relaxed": bool(relaxed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage BBCH53 calibration (global + per-variety) with LOOCV-aware model selection.")
    parser.add_argument("--tbase-min", type=float, default=8.0)
    parser.add_argument("--tbase-max", type=float, default=12.0)
    parser.add_argument("--tbase-step", type=float, default=1.0)
    parser.add_argument("--theta-min", type=float, default=100.0)
    parser.add_argument("--theta-max", type=float, default=1100.0)
    parser.add_argument("--theta-step", type=float, default=10.0)
    parser.add_argument("--search-tbase", action="store_true", default=True)
    parser.add_argument("--tbase-main", type=float, default=10.0)
    parser.add_argument("--start-doy-min", type=int, default=24)
    parser.add_argument("--start-doy-max", type=int, default=65)
    parser.add_argument("--start-doy-step", type=int, default=1)
    parser.add_argument("--use-rgdd-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgdd-window", type=int, default=7)
    parser.add_argument("--rgdd-min", type=float, default=0.0)
    parser.add_argument("--rgdd-max", type=float, default=160.0)
    parser.add_argument("--rgdd-step", type=float, default=10.0)
    parser.add_argument("--kmpt-min", type=float, default=-0.5)
    parser.add_argument("--kmpt-max", type=float, default=0.3)
    parser.add_argument("--kmpt-step", type=float, default=0.1)
    parser.add_argument("--delta-min", type=float, default=-0.75)
    parser.add_argument("--delta-max", type=float, default=0.75)
    parser.add_argument("--delta-step", type=float, default=0.1)
    parser.add_argument("--penalty-delta-l1", type=float, default=0.25)
    parser.add_argument("--start-shift-min", type=int, default=-16)
    parser.add_argument("--start-shift-max", type=int, default=16)
    parser.add_argument("--start-shift-step", type=int, default=1)
    parser.add_argument("--penalty-start-shift-l1", type=float, default=0.10)
    parser.add_argument("--theta-quantiles", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--n-jobs", type=int, default=512)
    parser.add_argument("--backend", type=str, default="ray", choices=["ray", "serial"])
    parser.add_argument("--loocv-topk-frac", type=float, default=0.10, help="Top fraction by train MAE before val selection.")
    parser.add_argument("--loocv-unseen-variety-strategy", type=str, default="global", choices=["zero", "global"])
    parser.add_argument("--cv-folds", type=int, default=0, help="0 means LOOCV; otherwise use K-fold.")
    parser.add_argument("--gate-min-selected-folds", type=int, default=4, help="Minimum fold coverage for gate/cv selector.")
    parser.add_argument("--wk-fixed-enable", action=argparse.BooleanOptionalAction, default=True, help="Use fixed WK delta/start_shift in stage2.")
    parser.add_argument("--wk-fixed-delta", type=float, default=float(FIXED_WK_PRIOR["delta"]))
    parser.add_argument("--wk-fixed-start-shift", type=int, default=int(FIXED_WK_PRIOR["shift"]))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _build_samples(_load_actual())

    tbase_grid_full = np.arange(args.tbase_min, args.tbase_max + 1e-12, args.tbase_step)
    tbase_grid = tbase_grid_full if args.search_tbase else np.array([float(args.tbase_main)])
    theta_grid = np.arange(args.theta_min, args.theta_max + 1e-12, args.theta_step)
    start_doy_grid = np.arange(args.start_doy_min, args.start_doy_max + 1, args.start_doy_step, dtype=int)
    rgdd_grid = np.arange(args.rgdd_min, args.rgdd_max + 1e-12, args.rgdd_step) if args.use_rgdd_gate else np.array([0.0])
    kmpt_grid = np.arange(args.kmpt_min, args.kmpt_max + 1e-12, args.kmpt_step)
    delta_grid = np.arange(args.delta_min, args.delta_max + 1e-12, args.delta_step)
    shift_grid = np.arange(args.start_shift_min, args.start_shift_max + 1, args.start_shift_step, dtype=int)

    stage1_total = len(tbase_grid) * len(theta_grid) * len(start_doy_grid) * len(rgdd_grid)
    stage2_total = (2 * len(delta_grid) * len(shift_grid)) + (len(delta_grid) * len(shift_grid) * len(kmpt_grid))
    n_folds = len(samples) if int(args.cv_folds) <= 1 else int(args.cv_folds)
    cv_label = "LOOCV" if n_folds >= len(samples) else f"{n_folds}-fold CV"
    print(
        f"[BBCH] {cv_label}-first | Stage1 grid={stage1_total:,} | Stage2(delta+start-shift+MPT-k) grid={stage2_total:,} | "
        f"CV topk_frac={args.loocv_topk_frac:.3f}, unseen_variety={args.loocv_unseen_variety_strategy} | "
        f"n_jobs={args.n_jobs} | backend={args.backend}",
        flush=True,
    )

    prior_source = "wk_fixed_only"
    prior_stage1 = {
        "t_base": float(args.tbase_main),
        "theta_agdd": float(np.median(theta_grid)),
        "start_doy": int(np.median(start_doy_grid)),
        "rgdd_min": float(np.median(rgdd_grid)),
    }
    prior_wk = {"delta": float(args.wk_fixed_delta), "shift": int(args.wk_fixed_start_shift), "k_mpt": 0.0}
    # JF/MPT are not fixed; keep neutral priors only for fallback safety.
    prior_jf = {"delta": 0.0, "shift": 0, "k_mpt": 0.0}
    prior_mpt = {"delta": 0.0, "shift": 0, "k_mpt": 0.0}

    cv_df = _kfold_full_grid(
        samples=samples,
        tbase_grid=tbase_grid,
        theta_grid=theta_grid,
        start_doy_grid=start_doy_grid,
        rgdd_grid=rgdd_grid,
        kmpt_grid=kmpt_grid,
        delta_grid=delta_grid,
        shift_grid=shift_grid,
        loocv_topk_frac=float(args.loocv_topk_frac),
        n_jobs=int(args.n_jobs),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_window=args.rgdd_window,
        backend=args.backend,
        unseen_variety_strategy=str(args.loocv_unseen_variety_strategy),
        prior_delta_wk=float(prior_wk["delta"]),
        prior_delta_jf=float(prior_jf["delta"]),
        prior_delta_mpt=float(prior_mpt["delta"]),
        prior_start_shift_wk=int(prior_wk["shift"]),
        prior_start_shift_jf=int(prior_jf["shift"]),
        prior_start_shift_mpt=int(prior_mpt["shift"]),
        prior_k_mpt=float(prior_mpt.get("k_mpt", 0.0)),
        wk_fixed_enable=bool(args.wk_fixed_enable),
        wk_fixed_delta=float(args.wk_fixed_delta),
        wk_fixed_shift=int(args.wk_fixed_start_shift),
        penalty_delta_l1=float(args.penalty_delta_l1),
        penalty_shift_l1=float(args.penalty_start_shift_l1),
        n_folds=int(n_folds),
    )
    leaderboard_df = _build_candidate_leaderboard_from_cv_rows(
        samples=samples,
        cv_df=cv_df,
        n_folds=int(n_folds),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_window=args.rgdd_window,
    )
    best_final_main = _pick_final_from_leaderboard(
        leaderboard=leaderboard_df,
        selector="fit",
        min_selected_folds=int(args.gate_min_selected_folds),
        allow_relax=True,
    )
    best_final_gate = _pick_final_from_leaderboard(
        leaderboard=leaderboard_df,
        selector="cv",
        min_selected_folds=int(args.gate_min_selected_folds),
        allow_relax=False,
    )
    best_final = best_final_main

    fit_mae, fit_rmse, fit_detail = _eval_params(
        samples=samples,
        t_base=float(best_final["t_base"]),
        theta=float(best_final["theta_agdd"]),
        start_doy=int(best_final["start_doy"]),
        use_rgdd_gate=args.use_rgdd_gate,
        rgdd_min=float(best_final["rgdd_min"]),
        rgdd_window=args.rgdd_window,
        k_mpt=float(best_final["k_mpt"]),
        delta_wk=float(best_final["delta_wk"]),
        delta_jf=float(best_final["delta_jf"]),
        delta_mpt=float(best_final["delta_mpt"]),
        start_shift_wk=int(best_final["start_shift_wk"]),
        start_shift_jf=int(best_final["start_shift_jf"]),
        start_shift_mpt=int(best_final["start_shift_mpt"]),
    )
    gate_available = best_final_gate is not None
    if gate_available:
        gate_fit_mae, gate_fit_rmse, gate_fit_detail = _eval_params(
            samples=samples,
            t_base=float(best_final_gate["t_base"]),
            theta=float(best_final_gate["theta_agdd"]),
            start_doy=int(best_final_gate["start_doy"]),
            use_rgdd_gate=args.use_rgdd_gate,
            rgdd_min=float(best_final_gate["rgdd_min"]),
            rgdd_window=args.rgdd_window,
            k_mpt=float(best_final_gate["k_mpt"]),
            delta_wk=float(best_final_gate["delta_wk"]),
            delta_jf=float(best_final_gate["delta_jf"]),
            delta_mpt=float(best_final_gate["delta_mpt"]),
            start_shift_wk=int(best_final_gate["start_shift_wk"]),
            start_shift_jf=int(best_final_gate["start_shift_jf"]),
            start_shift_mpt=int(best_final_gate["start_shift_mpt"]),
        )
    else:
        gate_fit_mae, gate_fit_rmse, gate_fit_detail = np.nan, np.nan, []

    q_values = [float(x) for x in args.theta_quantiles.split(",") if x.strip()]
    theta_obs_df = _backcalc_theta_at_obs(samples, t_base=args.tbase_main, start_doy=int(best_final["start_doy"]))
    theta_obs_df = theta_obs_df.dropna(subset=["theta_obs"])

    scenario_rows = []
    for q in q_values:
        theta_q = float(theta_obs_df["theta_obs"].quantile(q))
        mae_q, rmse_q, _ = _eval_params(
            samples=samples,
            t_base=args.tbase_main,
            theta=theta_q,
            start_doy=int(best_final["start_doy"]),
            use_rgdd_gate=args.use_rgdd_gate,
            rgdd_min=float(best_final["rgdd_min"]),
            rgdd_window=args.rgdd_window,
            k_mpt=float(best_final["k_mpt"]),
            delta_wk=float(best_final["delta_wk"]),
            delta_jf=float(best_final["delta_jf"]),
            delta_mpt=float(best_final["delta_mpt"]),
            start_shift_wk=int(best_final["start_shift_wk"]),
            start_shift_jf=int(best_final["start_shift_jf"]),
            start_shift_mpt=int(best_final["start_shift_mpt"]),
        )
        scenario_rows.append({"t_base_main": float(args.tbase_main), "theta_quantile": q, "theta_agdd": theta_q, "mae_days": mae_q, "rmse_days": rmse_q})
    scenario_df = pd.DataFrame(scenario_rows).sort_values("theta_quantile").reset_index(drop=True)

    trial_rows = [
        {
                "selector": "main_fit",
                "t_base": float(best_final["t_base"]),
                "theta_agdd": float(best_final["theta_agdd"]),
                "start_doy": int(best_final["start_doy"]),
                "rgdd_min": float(best_final["rgdd_min"]),
                "k_mpt": float(best_final["k_mpt"]),
                "delta_wk": float(best_final["delta_wk"]),
                "delta_jf": float(best_final["delta_jf"]),
                "delta_mpt": float(best_final["delta_mpt"]),
                "start_shift_wk": int(best_final["start_shift_wk"]),
                "start_shift_jf": int(best_final["start_shift_jf"]),
                "start_shift_mpt": int(best_final["start_shift_mpt"]),
                "mae_days": float(fit_mae),
                "rmse_days": float(fit_rmse),
                "selected_by_cv_mae_days": float(best_final["selected_by_cv_mae_days"]),
                "selected_by_cv_rmse_days": float(best_final["selected_by_cv_rmse_days"]),
                "selected_by_fit_mae_days": float(best_final["selected_by_fit_mae_days"]),
                "selected_by_fit_rmse_days": float(best_final["selected_by_fit_rmse_days"]),
                "selected_n_folds": int(best_final["selected_n_folds"]),
                "gate_min_selected_folds": int(best_final["gate_min_selected_folds"]),
                "gate_constraint_relaxed": bool(best_final["gate_constraint_relaxed"]),
            }
    ]
    if gate_available:
        trial_rows.append(
            {
                "selector": "gate_cv",
                "t_base": float(best_final_gate["t_base"]),
                "theta_agdd": float(best_final_gate["theta_agdd"]),
                "start_doy": int(best_final_gate["start_doy"]),
                "rgdd_min": float(best_final_gate["rgdd_min"]),
                "k_mpt": float(best_final_gate["k_mpt"]),
                "delta_wk": float(best_final_gate["delta_wk"]),
                "delta_jf": float(best_final_gate["delta_jf"]),
                "delta_mpt": float(best_final_gate["delta_mpt"]),
                "start_shift_wk": int(best_final_gate["start_shift_wk"]),
                "start_shift_jf": int(best_final_gate["start_shift_jf"]),
                "start_shift_mpt": int(best_final_gate["start_shift_mpt"]),
                "mae_days": float(gate_fit_mae),
                "rmse_days": float(gate_fit_rmse),
                "selected_by_cv_mae_days": float(best_final_gate["selected_by_cv_mae_days"]),
                "selected_by_cv_rmse_days": float(best_final_gate["selected_by_cv_rmse_days"]),
                "selected_by_fit_mae_days": float(best_final_gate["selected_by_fit_mae_days"]),
                "selected_by_fit_rmse_days": float(best_final_gate["selected_by_fit_rmse_days"]),
                "selected_n_folds": int(best_final_gate["selected_n_folds"]),
                "gate_min_selected_folds": int(best_final_gate["gate_min_selected_folds"]),
                "gate_constraint_relaxed": bool(best_final_gate["gate_constraint_relaxed"]),
            }
        )
    pd.DataFrame(trial_rows).to_csv(OUT_DIR / "bbch53_trials.csv", index=False)

    pd.DataFrame(fit_detail).to_csv(OUT_DIR / "bbch53_fit_detail.csv", index=False)
    if gate_available:
        pd.DataFrame(gate_fit_detail).to_csv(OUT_DIR / "bbch53_fit_detail_gate.csv", index=False)
    cv_df.to_csv(OUT_DIR / "bbch53_cv.csv", index=False)
    leaderboard_df.sort_values(["fit_mae_days", "fit_rmse_days", "gate_cv_mae_days", "gate_cv_rmse_days"]).to_csv(
        OUT_DIR / "bbch53_selection_compare.csv", index=False
    )
    theta_obs_df.to_csv(OUT_DIR / "bbch53_theta_obs_tbase_main.csv", index=False)
    scenario_df.to_csv(OUT_DIR / "bbch53_main_setting_scenarios.csv", index=False)

    summary = {
        "best_t_base": float(best_final["t_base"]),
        "best_theta_agdd": float(best_final["theta_agdd"]),
        "best_start_doy": int(best_final["start_doy"]),
        "best_k_mpt": float(best_final["k_mpt"]),
        "fit_mae_days": float(fit_mae),
        "fit_rmse_days": float(fit_rmse),
        "cv_mae_days": float(cv_df["abs_error_days"].mean()),
        "cv_rmse_days": float(np.sqrt(np.mean(cv_df["abs_error_days"].values ** 2))),
        "t_base_main_setting": float(args.tbase_main),
        "main_theta_quantiles": q_values,
        "main_theta_scenarios_file": "bbch53_main_setting_scenarios.csv",
        "best_rgdd_min": float(best_final["rgdd_min"]) if args.use_rgdd_gate else None,
        "best_delta_wk": float(best_final["delta_wk"]),
        "best_delta_jf": float(best_final["delta_jf"]),
        "best_delta_mpt": float(best_final["delta_mpt"]),
        "best_start_shift_wk": int(best_final["start_shift_wk"]),
        "best_start_shift_jf": int(best_final["start_shift_jf"]),
        "best_start_shift_mpt": int(best_final["start_shift_mpt"]),
        "selected_by_cv_mae_days": float(best_final["selected_by_cv_mae_days"]),
        "selected_by_cv_rmse_days": float(best_final["selected_by_cv_rmse_days"]),
        "selected_by_fit_mae_days": float(best_final["selected_by_fit_mae_days"]),
        "selected_by_fit_rmse_days": float(best_final["selected_by_fit_rmse_days"]),
        "selected_n_folds": int(best_final["selected_n_folds"]),
        "gate_min_selected_folds": int(args.gate_min_selected_folds),
        "main_selector_constraint_relaxed": bool(best_final["gate_constraint_relaxed"]),
        "gate_available": bool(gate_available),
        "gate_t_base": float(best_final_gate["t_base"]) if gate_available else None,
        "gate_theta_agdd": float(best_final_gate["theta_agdd"]) if gate_available else None,
        "gate_start_doy": int(best_final_gate["start_doy"]) if gate_available else None,
        "gate_k_mpt": float(best_final_gate["k_mpt"]) if gate_available else None,
        "gate_rgdd_min": (float(best_final_gate["rgdd_min"]) if args.use_rgdd_gate else None) if gate_available else None,
        "gate_delta_wk": float(best_final_gate["delta_wk"]) if gate_available else None,
        "gate_delta_jf": float(best_final_gate["delta_jf"]) if gate_available else None,
        "gate_delta_mpt": float(best_final_gate["delta_mpt"]) if gate_available else None,
        "gate_start_shift_wk": int(best_final_gate["start_shift_wk"]) if gate_available else None,
        "gate_start_shift_jf": int(best_final_gate["start_shift_jf"]) if gate_available else None,
        "gate_start_shift_mpt": int(best_final_gate["start_shift_mpt"]) if gate_available else None,
        "gate_fit_mae_days": float(gate_fit_mae) if gate_available else None,
        "gate_fit_rmse_days": float(gate_fit_rmse) if gate_available else None,
        "gate_selected_by_cv_mae_days": float(best_final_gate["selected_by_cv_mae_days"]) if gate_available else None,
        "gate_selected_by_cv_rmse_days": float(best_final_gate["selected_by_cv_rmse_days"]) if gate_available else None,
        "gate_selected_by_fit_mae_days": float(best_final_gate["selected_by_fit_mae_days"]) if gate_available else None,
        "gate_selected_by_fit_rmse_days": float(best_final_gate["selected_by_fit_rmse_days"]) if gate_available else None,
        "gate_selected_n_folds": int(best_final_gate["selected_n_folds"]) if gate_available else None,
        "gate_selector_constraint_relaxed": (bool(best_final_gate["gate_constraint_relaxed"]) if gate_available else None),
        "penalty_delta_l1": float(args.penalty_delta_l1),
        "penalty_start_shift_l1": float(args.penalty_start_shift_l1),
        "use_rgdd_gate": bool(args.use_rgdd_gate),
        "use_variety_delta": True,
        "use_variety_start_shift": True,
        "rgdd_window": int(args.rgdd_window),
        "kmpt_min": float(args.kmpt_min),
        "kmpt_max": float(args.kmpt_max),
        "kmpt_step": float(args.kmpt_step),
        "start_doy_min": int(args.start_doy_min),
        "start_doy_max": int(args.start_doy_max),
        "start_doy_step": int(args.start_doy_step),
        "start_shift_min": int(args.start_shift_min),
        "start_shift_max": int(args.start_shift_max),
        "start_shift_step": int(args.start_shift_step),
        "calibration_mode": "loocv_first_two_stage_global_then_variety_delta_start_shift_and_mpt_k",
        "t_base_search_enabled": bool(args.search_tbase),
        "t_base_grid_size": int(len(tbase_grid)),
        "start_doy_grid_size": int(len(start_doy_grid)),
        "theta_grid_size": int(len(theta_grid)),
        "rgdd_grid_size": int(len(rgdd_grid)),
        "kmpt_grid_size": int(len(kmpt_grid)),
        "delta_grid_size": int(len(delta_grid)),
        "start_shift_grid_size": int(len(shift_grid)),
        "stage1_total_grid": int(stage1_total),
        "stage2_total_grid": int(stage2_total),
        "cv_folds": int(n_folds),
        "loocv_topk_frac": float(args.loocv_topk_frac),
        "loocv_unseen_variety_strategy": str(args.loocv_unseen_variety_strategy),
        "prior_stage1_t_base": float(prior_stage1["t_base"]),
        "prior_stage1_theta_agdd": float(prior_stage1["theta_agdd"]),
        "prior_stage1_start_doy": int(prior_stage1["start_doy"]),
        "prior_source": str(prior_source),
        "wk_fixed_enable": bool(args.wk_fixed_enable),
        "wk_fixed_delta": float(args.wk_fixed_delta),
        "wk_fixed_start_shift": int(args.wk_fixed_start_shift),
        "prior_k_mpt": float(prior_mpt["k_mpt"]),
        "prior_delta_wk": float(prior_wk["delta"]),
        "prior_delta_jf": float(prior_jf["delta"]),
        "prior_delta_mpt": float(prior_mpt["delta"]),
        "prior_start_shift_wk": int(prior_wk["shift"]),
        "prior_start_shift_jf": int(prior_jf["shift"]),
        "prior_start_shift_mpt": int(prior_mpt["shift"]),
        "cv_mode": "loocv_train_topk_then_val_then_finalize" if n_folds >= len(samples) else "kfold_train_topk_then_val_then_finalize",
        "variety_codes": ["WK", "JF", "MPT"],
        "n_site_year": int(len(samples)),
    }
    (OUT_DIR / "bbch53_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_DIR / 'bbch53_best.json'}")
    print(f"Saved: {OUT_DIR / 'bbch53_trials.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_fit_detail.csv'}")
    if gate_available:
        print(f"Saved: {OUT_DIR / 'bbch53_fit_detail_gate.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_cv.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_selection_compare.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_theta_obs_tbase_main.csv'}")
    print(f"Saved: {OUT_DIR / 'bbch53_main_setting_scenarios.csv'}")


if __name__ == "__main__":
    main()
