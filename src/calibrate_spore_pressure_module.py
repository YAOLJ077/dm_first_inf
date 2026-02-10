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
try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

from bbch53_module import load_hourly_weather_from_json
from spore_pressure_module import build_daily_features, latent_to_pressure, simulate_latent_spore_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "result" / "historical_inputs" / "spore_pressure_calibration"


def _load_actual() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "actual_data.xlsx")
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"], errors="coerce")
    return df.reset_index(drop=True)


def _load_spore_obs() -> pd.DataFrame:
    s = pd.read_excel(DATA_DIR / "Spore_Overall.xlsx")
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s["spore"] = pd.to_numeric(s["spore"], errors="coerce")
    return s.dropna(subset=["date"]).reset_index(drop=True)


def _site_year_rows() -> List[Tuple]:
    a = _load_actual()
    return list(a.itertuples(index=False))


def _prepare_site_year_data(r, spore_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rb_path = DATA_DIR / f"{r.plot}_{r.variety}_{int(r.year)}_grape_rb.json"
    if not rb_path.exists():
        raise FileNotFoundError(rb_path)

    wh = load_hourly_weather_from_json(rb_path)
    wh = wh[wh["datetime"].dt.year == int(r.year)].reset_index(drop=True)
    daily = build_daily_features(wh, rh_wet_night=85.0)

    obs = spore_df[(spore_df["plot"] == r.plot) & (spore_df["year"] == int(r.year)) & (spore_df["spore"].notna())].copy()
    if pd.notna(r.first_infection_date):
        obs = obs[obs["date"] <= pd.to_datetime(r.first_infection_date)]
    obs["date"] = obs["date"].dt.normalize()
    obs = obs.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return daily, obs


def _normalized_obs(obs: pd.DataFrame) -> pd.Series:
    if obs.empty:
        return pd.Series(dtype=float)
    p95 = float(obs["spore"].quantile(0.95))
    if p95 <= 0:
        return pd.Series(np.zeros(len(obs)), index=obs["date"].values, dtype=float)
    return pd.Series((np.log1p(obs["spore"].values) / np.log1p(p95)).clip(0.0, 1.0), index=obs["date"].values, dtype=float)


def _evaluate_params(site_year_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]], params: Dict[str, float]) -> Tuple[float, pd.DataFrame]:
    rows = []
    losses = []
    for (plot, year), (daily, obs) in site_year_cache.items():
        if obs.empty:
            continue
        latent = simulate_latent_spore_state(daily, params)
        pressure = latent_to_pressure(
            latent_daily=latent,
            roll_window=int(round(params["roll_window"])),
            baseline_q=float(params["baseline_q"]),
        )
        pred = pressure.set_index("date")["S_day_roll"]
        pred = pred.reindex(obs["date"].values).astype(float).fillna(0.0)
        obs_norm = _normalized_obs(obs)
        mae = float(np.mean(np.abs(pred.values - obs_norm.values)))
        losses.append(mae)
        rows.append({"plot": plot, "year": year, "mae": mae})
    if not losses:
        return 999.0, pd.DataFrame(rows)
    return float(np.mean(losses)), pd.DataFrame(rows)


if ray is not None:
    @ray.remote
    class SporeEvalWorker:
        def __init__(self, cache_ref):
            self.cache = ray.get(cache_ref)

        def eval_loss(self, params: Dict[str, float]) -> float:
            loss, _ = _evaluate_params(self.cache, params)
            return float(loss)


def _suggest_params(trial: optuna.Trial) -> Dict[str, float]:
    tmin = trial.suggest_float("tmin", 5.0, 16.0)
    topt = trial.suggest_float("topt", tmin + 2.0, 30.0)
    tmax = trial.suggest_float("tmax", topt + 2.0, 40.0)
    return {
        "rho": trial.suggest_float("rho", 0.2, 0.99),
        "beta_mp": trial.suggest_float("beta_mp", 0.1, 4.0),
        "beta_rain": trial.suggest_float("beta_rain", 0.0, 0.3),
        "beta_wind": trial.suggest_float("beta_wind", 0.0, 0.3),
        "gamma_vpd": trial.suggest_float("gamma_vpd", 0.0, 1.5),
        "k_mp": trial.suggest_float("k_mp", 0.03, 1.0),
        "tmin": tmin,
        "topt": topt,
        "tmax": tmax,
        "roll_window": trial.suggest_int("roll_window", 3, 7),
        "baseline_q": trial.suggest_float("baseline_q", 0.45, 0.75),
    }


def _sample_params_rng(rng: np.random.Generator) -> Dict[str, float]:
    tmin = float(rng.uniform(5.0, 16.0))
    topt = float(rng.uniform(tmin + 2.0, 30.0))
    tmax = float(rng.uniform(topt + 2.0, 40.0))
    return {
        "rho": float(rng.uniform(0.2, 0.99)),
        "beta_mp": float(rng.uniform(0.1, 4.0)),
        "beta_rain": float(rng.uniform(0.0, 0.3)),
        "beta_wind": float(rng.uniform(0.0, 0.3)),
        "gamma_vpd": float(rng.uniform(0.0, 1.5)),
        "k_mp": float(rng.uniform(0.03, 1.0)),
        "tmin": tmin,
        "topt": topt,
        "tmax": tmax,
        "roll_window": int(rng.integers(3, 8)),
        "baseline_q": float(rng.uniform(0.45, 0.75)),
    }


def _parallel_random_search(
    site_year_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]],
    n_trials: int,
    seed: int,
    n_jobs: int,
) -> Tuple[Dict[str, float], float, List[Dict]]:
    if ray is None or n_jobs <= 1:
        rng = np.random.default_rng(seed)
        trial_rows = []
        best_loss = float("inf")
        best_params = None
        for _ in range(int(n_trials)):
            p = _sample_params_rng(rng)
            loss, _ = _evaluate_params(site_year_cache, p)
            trial_rows.append({"value": float(loss), **p})
            if loss < best_loss:
                best_loss = float(loss)
                best_params = p
        return best_params, best_loss, trial_rows

    rng = np.random.default_rng(seed)
    cache_ref = ray.put(site_year_cache)
    workers = [SporeEvalWorker.remote(cache_ref) for _ in range(max(1, n_jobs))]
    available = workers[:]
    pending = {}
    trial_rows = []
    submitted = 0
    done = 0
    best_loss = float("inf")
    best_params = None

    while done < n_trials:
        while submitted < n_trials and available:
            p = _sample_params_rng(rng)
            w = available.pop(0)
            fut = w.eval_loss.remote(p)
            pending[fut] = (p, w)
            submitted += 1
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        for f in ready:
            p, w = pending.pop(f)
            loss = float(ray.get(f))
            trial_rows.append({"value": loss, **p})
            if loss < best_loss:
                best_loss = loss
                best_params = p
            available.append(w)
            done += 1
    return best_params, best_loss, trial_rows


def _parallel_optuna_search(
    site_year_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]],
    n_trials: int,
    n_startup: int,
    seed: int,
    n_jobs: int,
) -> Tuple[Dict[str, float], float, List[Dict]]:
    if ray is None or n_jobs <= 1:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=n_startup)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        trial_rows = []

        def objective(trial: optuna.Trial) -> float:
            p = _suggest_params(trial)
            loss, _ = _evaluate_params(site_year_cache, p)
            trial_rows.append({"value": float(loss), **p})
            return float(loss)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return _suggest_params(study.best_trial), float(study.best_value), trial_rows

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=n_startup)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    cache_ref = ray.put(site_year_cache)
    workers = [SporeEvalWorker.remote(cache_ref) for _ in range(max(1, n_jobs))]
    available = workers[:]
    pending = {}
    trial_rows = []
    submitted = 0
    completed = 0

    while completed < n_trials:
        while submitted < n_trials and available:
            tr = study.ask()
            p = _suggest_params(tr)
            w = available.pop(0)
            fut = w.eval_loss.remote(p)
            pending[fut] = (tr, p, w)
            submitted += 1
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        for f in ready:
            tr, p, w = pending.pop(f)
            loss = float(ray.get(f))
            study.tell(tr, loss)
            trial_rows.append({"value": loss, **p})
            available.append(w)
            completed += 1

    best_params = _suggest_params(study.best_trial)
    return best_params, float(study.best_value), trial_rows


def _loocv(rows: List, spore_df: pd.DataFrame, n_trials: int, seed: int) -> pd.DataFrame:
    out = []
    for i in range(len(rows)):
        test = rows[i]
        train_rows = [rows[j] for j in range(len(rows)) if j != i]
        train_cache = {}
        for r in train_rows:
            train_cache[(str(r.plot), int(r.year))] = _prepare_site_year_data(r, spore_df)

        if optuna is not None:
            sampler = optuna.samplers.TPESampler(seed=seed + i, multivariate=True, n_startup_trials=max(20, n_trials // 4))
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def obj(trial: optuna.Trial) -> float:
                p = _suggest_params(trial)
                loss, _ = _evaluate_params(train_cache, p)
                return float(loss)

            study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
            pbest = _suggest_params(study.best_trial)
            fit_train_mae = float(study.best_value)
        else:
            rng = np.random.default_rng(seed + i)
            best_loss = float("inf")
            pbest = None
            for _ in range(int(n_trials)):
                p = _sample_params_rng(rng)
                loss, _ = _evaluate_params(train_cache, p)
                if loss < best_loss:
                    best_loss = loss
                    pbest = p
            fit_train_mae = float(best_loss)

        test_cache = {(str(test.plot), int(test.year)): _prepare_site_year_data(test, spore_df)}
        test_loss, _ = _evaluate_params(test_cache, pbest)
        out.append(
            {
                "plot": str(test.plot),
                "year": int(test.year),
                "test_mae": float(test_loss),
                "fit_train_mae": fit_train_mae,
                **pbest,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate standalone weather->spore pressure module.")
    parser.add_argument("--n-trials", type=int, default=1200)
    parser.add_argument("--n-startup", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--run-loocv", action="store_true")
    parser.add_argument("--loocv-trials", type=int, default=200)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spore_df = _load_spore_obs()
    rows = _site_year_rows()

    site_year_cache = {}
    for r in rows:
        site_year_cache[(str(r.plot), int(r.year))] = _prepare_site_year_data(r, spore_df)

    ray_used = ray is not None and args.n_jobs > 1
    if ray_used:
        ray.init(num_cpus=max(1, args.n_jobs), include_dashboard=False, logging_level="ERROR")
    if optuna is not None:
        best_params, best_loss, trial_rows = _parallel_optuna_search(
            site_year_cache=site_year_cache,
            n_trials=int(args.n_trials),
            n_startup=int(args.n_startup),
            seed=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
    else:
        best_params, best_loss, trial_rows = _parallel_random_search(
            site_year_cache=site_year_cache,
            n_trials=int(args.n_trials),
            seed=int(args.seed),
            n_jobs=int(args.n_jobs),
        )

    best_loss, detail = _evaluate_params(site_year_cache, best_params)

    pd.DataFrame(trial_rows).sort_values("value").to_csv(OUT_DIR / "spore_proxy_trials.csv", index=False)
    detail.to_csv(OUT_DIR / "spore_proxy_fit_detail.csv", index=False)

    summary = {
        "best_loss_mae": float(best_loss),
        "n_site_year": int(len(site_year_cache)),
        "best_params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in best_params.items()},
    }
    with open(OUT_DIR / "spore_proxy_best.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.run_loocv:
        loocv_df = _loocv(rows, spore_df, n_trials=args.loocv_trials, seed=args.seed + 1000)
        loocv_df.to_csv(OUT_DIR / "spore_proxy_loocv.csv", index=False)
    if ray_used:
        ray.shutdown()

    print(f"Saved: {OUT_DIR / 'spore_proxy_best.json'}")
    print(f"Saved: {OUT_DIR / 'spore_proxy_trials.csv'}")
    print(f"Saved: {OUT_DIR / 'spore_proxy_fit_detail.csv'}")
    if args.run_loocv:
        print(f"Saved: {OUT_DIR / 'spore_proxy_loocv.csv'}")


if __name__ == "__main__":
    main()
