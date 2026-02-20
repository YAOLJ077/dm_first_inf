#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

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
try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

if TYPE_CHECKING:  # pragma: no cover
    import optuna as optuna_typing

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
    from src.bbch53_module import load_hourly_weather_from_json
    from src.spore_pressure_module import build_daily_features, latent_to_pressure, simulate_latent_spore_state
except Exception:  # pragma: no cover
    from bbch53_module import load_hourly_weather_from_json
    from spore_pressure_module import build_daily_features, latent_to_pressure, simulate_latent_spore_state


DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "result" / "historical_inputs" / "spore_pressure_calibration"

# tightened search bounds (1.5x expansion disabled)
PARAM_BOUNDS = {
    "rho": (0.08, 0.99),
    "beta_mp": (0.0, 4.0),
    "beta_rain": (0.0, 0.30),
    "beta_wind": (0.0, 0.30),
    "gamma_vpd": (0.0, 1.4),
    "k_mp": (0.01, 1.0),
    "tmin": (4.0, 17.0),
    # fixed spaces (no dynamic bounds): order is enforced after sampling
    "topt": (6.0, 34.0),
    "tmax": (8.0, 42.0),
    "roll_window": (2, 10),
    "baseline_q": (0.40, 0.80),
}


def _load_actual() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "actual_data.xlsx")
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"], errors="coerce")
    return df.reset_index(drop=True)


def _load_spore_obs() -> pd.DataFrame:
    s = pd.read_excel(DATA_DIR / "Spore_Overall.xlsx")
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s["spore"] = pd.to_numeric(s["spore"], errors="coerce")
    return s.dropna(subset=["date"]).reset_index(drop=True)


def _require_optuna():
    if optuna is None:
        raise RuntimeError("optuna is required for this path but is not installed.")
    return optuna


def _site_year_rows() -> List[Tuple]:
    a = _load_actual()
    return list(a.itertuples(index=False))


def _prepare_site_year_data(r, spore_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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
    obs_norm = _normalized_obs(obs)
    obs_dates = obs["date"].values
    return daily, obs_dates, obs_norm.values.astype(float)


def _normalized_obs(obs: pd.DataFrame) -> pd.Series:
    if obs.empty:
        return pd.Series(dtype=float)
    p95 = float(obs["spore"].quantile(0.95))
    if p95 <= 0:
        return pd.Series(np.zeros(len(obs)), index=obs["date"].values, dtype=float)
    return pd.Series((np.log1p(obs["spore"].values) / np.log1p(p95)).clip(0.0, 1.0), index=obs["date"].values, dtype=float)


def _evaluate_params(
    site_year_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    params: Dict[str, float],
) -> Tuple[float, pd.DataFrame]:
    rows = []
    losses = []
    for (plot, year), (daily, obs_dates, obs_norm_vals) in site_year_cache.items():
        if obs_dates is None or len(obs_dates) == 0:
            continue
        latent = simulate_latent_spore_state(daily, params)
        pressure = latent_to_pressure(
            latent_daily=latent,
            roll_window=int(round(params["roll_window"])),
            baseline_q=float(params["baseline_q"]),
        )
        pred = pressure.set_index("date")["S_day_roll"]
        pred_vals = pred.reindex(obs_dates).astype(float).fillna(0.0).values
        mae = float(np.mean(np.abs(pred_vals - obs_norm_vals)))
        losses.append(mae)
        rows.append({"plot": plot, "year": year, "mae": mae})
    if not losses:
        return 999.0, pd.DataFrame(rows)
    return float(np.mean(losses)), pd.DataFrame(rows)


if ray is not None:
    @ray.remote
    class SporeEvalWorker:
        def __init__(self, cache_ref):
            # Ray versions/transport may pass either ObjectRef or already-materialized object.
            if isinstance(cache_ref, dict):
                self.cache = cache_ref
            else:
                self.cache = ray.get(cache_ref)

        def eval_loss(self, params: Dict[str, float]) -> float:
            loss, _ = _evaluate_params(self.cache, params)
            return float(loss)

        def eval_loss_batch(self, params_batch: List[Dict[str, float]]) -> List[float]:
            out = []
            for p in params_batch:
                loss, _ = _evaluate_params(self.cache, p)
                out.append(float(loss))
            return out


def _suggest_params(trial: "optuna_typing.trial.Trial") -> Dict[str, float]:
    tmin_raw = trial.suggest_float("tmin", PARAM_BOUNDS["tmin"][0], PARAM_BOUNDS["tmin"][1])
    topt_raw = trial.suggest_float("topt", PARAM_BOUNDS["topt"][0], PARAM_BOUNDS["topt"][1])
    tmax_raw = trial.suggest_float("tmax", PARAM_BOUNDS["tmax"][0], PARAM_BOUNDS["tmax"][1])
    # Enforce biological order under fixed search spaces.
    tmin = float(tmin_raw)
    topt = float(max(topt_raw, tmin + 2.0))
    tmax = float(max(tmax_raw, topt + 2.0))
    if tmax > PARAM_BOUNDS["tmax"][1]:
        tmax = float(PARAM_BOUNDS["tmax"][1])
        topt = min(float(topt), tmax - 2.0)
        topt = max(float(topt), tmin + 2.0)
    return {
        "rho": trial.suggest_float("rho", PARAM_BOUNDS["rho"][0], PARAM_BOUNDS["rho"][1]),
        "beta_mp": trial.suggest_float("beta_mp", PARAM_BOUNDS["beta_mp"][0], PARAM_BOUNDS["beta_mp"][1]),
        "beta_rain": trial.suggest_float("beta_rain", PARAM_BOUNDS["beta_rain"][0], PARAM_BOUNDS["beta_rain"][1]),
        "beta_wind": trial.suggest_float("beta_wind", PARAM_BOUNDS["beta_wind"][0], PARAM_BOUNDS["beta_wind"][1]),
        "gamma_vpd": trial.suggest_float("gamma_vpd", PARAM_BOUNDS["gamma_vpd"][0], PARAM_BOUNDS["gamma_vpd"][1]),
        "k_mp": trial.suggest_float("k_mp", PARAM_BOUNDS["k_mp"][0], PARAM_BOUNDS["k_mp"][1]),
        "tmin": tmin,
        "topt": topt,
        "tmax": tmax,
        "roll_window": trial.suggest_int("roll_window", PARAM_BOUNDS["roll_window"][0], PARAM_BOUNDS["roll_window"][1]),
        "baseline_q": trial.suggest_float("baseline_q", PARAM_BOUNDS["baseline_q"][0], PARAM_BOUNDS["baseline_q"][1]),
    }


def _sample_params_rng(rng: np.random.Generator) -> Dict[str, float]:
    tmin_raw = float(rng.uniform(PARAM_BOUNDS["tmin"][0], PARAM_BOUNDS["tmin"][1]))
    topt_raw = float(rng.uniform(PARAM_BOUNDS["topt"][0], PARAM_BOUNDS["topt"][1]))
    tmax_raw = float(rng.uniform(PARAM_BOUNDS["tmax"][0], PARAM_BOUNDS["tmax"][1]))
    tmin = float(tmin_raw)
    topt = float(max(topt_raw, tmin + 2.0))
    tmax = float(max(tmax_raw, topt + 2.0))
    if tmax > PARAM_BOUNDS["tmax"][1]:
        tmax = float(PARAM_BOUNDS["tmax"][1])
        topt = min(float(topt), tmax - 2.0)
        topt = max(float(topt), tmin + 2.0)
    return {
        "rho": float(rng.uniform(PARAM_BOUNDS["rho"][0], PARAM_BOUNDS["rho"][1])),
        "beta_mp": float(rng.uniform(PARAM_BOUNDS["beta_mp"][0], PARAM_BOUNDS["beta_mp"][1])),
        "beta_rain": float(rng.uniform(PARAM_BOUNDS["beta_rain"][0], PARAM_BOUNDS["beta_rain"][1])),
        "beta_wind": float(rng.uniform(PARAM_BOUNDS["beta_wind"][0], PARAM_BOUNDS["beta_wind"][1])),
        "gamma_vpd": float(rng.uniform(PARAM_BOUNDS["gamma_vpd"][0], PARAM_BOUNDS["gamma_vpd"][1])),
        "k_mp": float(rng.uniform(PARAM_BOUNDS["k_mp"][0], PARAM_BOUNDS["k_mp"][1])),
        "tmin": tmin,
        "topt": topt,
        "tmax": tmax,
        "roll_window": int(rng.integers(PARAM_BOUNDS["roll_window"][0], PARAM_BOUNDS["roll_window"][1] + 1)),
        "baseline_q": float(rng.uniform(PARAM_BOUNDS["baseline_q"][0], PARAM_BOUNDS["baseline_q"][1])),
    }


def _parallel_random_search(
    site_year_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    n_trials: int,
    seed: int,
    n_jobs: int,
    batch_size: int = 8,
) -> Tuple[Dict[str, float], float, List[Dict]]:
    if ray is None or n_jobs <= 1:
        rng = np.random.default_rng(seed)
        trial_rows = []
        best_loss = float("inf")
        best_params = None
        for _ in tqdm(range(int(n_trials)), desc="Spore random search"):
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
    pbar = tqdm(total=int(n_trials), desc="Spore random search")

    while done < n_trials:
        while submitted < n_trials and available:
            n_submit = min(int(batch_size), int(n_trials - submitted))
            p_batch = [_sample_params_rng(rng) for _ in range(n_submit)]
            w = available.pop(0)
            fut = w.eval_loss_batch.remote(p_batch)
            pending[fut] = (p_batch, w)
            submitted += n_submit
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        for f in ready:
            p_batch, w = pending.pop(f)
            loss_batch = [float(x) for x in ray.get(f)]
            for p, loss in zip(p_batch, loss_batch):
                trial_rows.append({"value": loss, **p})
                if loss < best_loss:
                    best_loss = loss
                    best_params = p
                done += 1
                pbar.update(1)
            available.append(w)
    pbar.close()
    return best_params, best_loss, trial_rows


def _parallel_optuna_search(
    site_year_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    n_trials: int,
    n_startup: int,
    seed: int,
    n_jobs: int,
    batch_size: int = 8,
) -> Tuple[Dict[str, float], float, List[Dict]]:
    opt = _require_optuna()
    if ray is None or n_jobs <= 1:
        sampler = opt.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=n_startup)
        study = opt.create_study(direction="minimize", sampler=sampler)
        trial_rows = []

        def objective(trial: "optuna_typing.trial.Trial") -> float:
            p = _suggest_params(trial)
            loss, _ = _evaluate_params(site_year_cache, p)
            trial_rows.append({"value": float(loss), **p})
            return float(loss)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return _suggest_params(study.best_trial), float(study.best_value), trial_rows

    sampler = opt.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=n_startup)
    study = opt.create_study(direction="minimize", sampler=sampler)
    cache_ref = ray.put(site_year_cache)
    workers = [SporeEvalWorker.remote(cache_ref) for _ in range(max(1, n_jobs))]
    available = workers[:]
    pending = {}
    trial_rows = []
    submitted = 0
    completed = 0
    pbar = tqdm(total=int(n_trials), desc="Spore TPE search")

    while completed < n_trials:
        while submitted < n_trials and available:
            n_submit = min(int(batch_size), int(n_trials - submitted))
            tr_batch = []
            p_batch = []
            for _ in range(n_submit):
                tr = study.ask()
                p = _suggest_params(tr)
                tr_batch.append(tr)
                p_batch.append(p)
            w = available.pop(0)
            fut = w.eval_loss_batch.remote(p_batch)
            pending[fut] = (tr_batch, p_batch, w)
            submitted += n_submit
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        for f in ready:
            tr_batch, p_batch, w = pending.pop(f)
            loss_batch = [float(x) for x in ray.get(f)]
            for tr, p, loss in zip(tr_batch, p_batch, loss_batch):
                study.tell(tr, loss)
                trial_rows.append({"value": loss, **p})
                completed += 1
                pbar.update(1)
            available.append(w)
    pbar.close()

    best_params = _suggest_params(study.best_trial)
    return best_params, float(study.best_value), trial_rows


def _loocv(rows: List, spore_df: pd.DataFrame, n_trials: int, seed: int) -> pd.DataFrame:
    out = []
    for i in tqdm(range(len(rows)), desc="Spore LOOCV"):
        test = rows[i]
        train_rows = [rows[j] for j in range(len(rows)) if j != i]
        train_cache = {}
        for r in train_rows:
            train_cache[(str(r.plot), int(r.year))] = _prepare_site_year_data(r, spore_df)

        if optuna is not None:
            opt = _require_optuna()
            sampler = opt.samplers.TPESampler(seed=seed + i, multivariate=True, n_startup_trials=max(20, n_trials // 4))
            study = opt.create_study(direction="minimize", sampler=sampler)

            def obj(trial: "optuna_typing.trial.Trial") -> float:
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


def _extract_params_from_trial_row(tr: Dict) -> Dict[str, float]:
    return {
        "rho": float(tr["rho"]),
        "beta_mp": float(tr["beta_mp"]),
        "beta_rain": float(tr["beta_rain"]),
        "beta_wind": float(tr["beta_wind"]),
        "gamma_vpd": float(tr["gamma_vpd"]),
        "k_mp": float(tr["k_mp"]),
        "tmin": float(tr["tmin"]),
        "topt": float(tr["topt"]),
        "tmax": float(tr["tmax"]),
        "roll_window": int(round(float(tr["roll_window"]))),
        "baseline_q": float(tr["baseline_q"]),
    }


def _select_from_topk_with_constraint(candidates: List[Dict], topk_frac: float) -> Dict:
    if not candidates:
        raise ValueError("No fold candidates to select from.")
    cand_df = pd.DataFrame(candidates).sort_values(["train_mae", "val_mae"]).reset_index(drop=True)
    k = max(1, int(np.ceil(len(cand_df) * float(topk_frac))))
    top = cand_df.head(k).copy()
    valid = top[top["val_mae"] >= top["train_mae"]]
    if not valid.empty:
        pick = valid.sort_values(["val_mae", "train_mae"]).iloc[0].to_dict()
        pick["constraint_satisfied"] = True
        pick["topk_pool_size"] = int(k)
        return pick
    top["gap_abs"] = (top["val_mae"] - top["train_mae"]).abs()
    pick = top.sort_values(["gap_abs", "val_mae", "train_mae"]).iloc[0].to_dict()
    pick["constraint_satisfied"] = False
    pick["topk_pool_size"] = int(k)
    return pick


def _loocv_train_topk_then_val(
    rows: List,
    spore_df: pd.DataFrame,
    full_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    n_trials: int,
    n_startup: int,
    seed: int,
    n_jobs: int,
    topk_frac: float,
) -> pd.DataFrame:
    out = []
    for i in tqdm(range(len(rows)), desc="Spore LOOCV"):
        test = rows[i]
        train_rows = [rows[j] for j in range(len(rows)) if j != i]
        train_cache = {}
        for r in train_rows:
            train_cache[(str(r.plot), int(r.year))] = _prepare_site_year_data(r, spore_df)
        test_cache = {(str(test.plot), int(test.year)): _prepare_site_year_data(test, spore_df)}

        if optuna is not None:
            _, _, trial_rows = _parallel_optuna_search(
                site_year_cache=train_cache,
                n_trials=int(n_trials),
                n_startup=max(10, int(n_startup)),
                seed=int(seed + i),
                n_jobs=int(n_jobs),
                batch_size=8,
            )
        else:
            _, _, trial_rows = _parallel_random_search(
                site_year_cache=train_cache,
                n_trials=int(n_trials),
                seed=int(seed + i),
                n_jobs=int(n_jobs),
                batch_size=8,
            )

        fold_candidates = []
        for tr in trial_rows:
            p = _extract_params_from_trial_row(tr)
            train_mae = float(tr["value"])
            val_mae, _ = _evaluate_params(test_cache, p)
            fold_candidates.append({"train_mae": train_mae, "val_mae": float(val_mae), **p})

        best_fold = _select_from_topk_with_constraint(fold_candidates, topk_frac=topk_frac)
        pbest = _extract_params_from_trial_row(best_fold)
        fit_mae_full, _ = _evaluate_params(full_cache, pbest)
        out.append(
            {
                "plot": str(test.plot),
                "year": int(test.year),
                "fit_train_mae": float(best_fold["train_mae"]),
                "test_mae": float(best_fold["val_mae"]),
                "constraint_satisfied": bool(best_fold["constraint_satisfied"]),
                "topk_pool_size": int(best_fold["topk_pool_size"]),
                "fit_mae_full": float(fit_mae_full),
                **pbest,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate standalone weather->spore pressure module.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loocv-trials", type=int, default=18000)
    parser.add_argument("--cv-topk-frac", type=float, default=0.10)
    parser.add_argument("--min-selected-folds", type=int, default=2)
    parser.add_argument("--param-round-decimals", type=int, default=4)
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
    # LOOCV-only calibration flow.
    loocv_df = _loocv_train_topk_then_val(
        rows=rows,
        spore_df=spore_df,
        full_cache=site_year_cache,
        n_trials=int(args.loocv_trials),
        n_startup=max(20, int(args.loocv_trials) // 4),
        seed=int(args.seed + 1000),
        n_jobs=int(args.n_jobs),
        topk_frac=float(args.cv_topk_frac),
    )
    loocv_df.to_csv(OUT_DIR / "spore_proxy_loocv.csv", index=False)
    # Keep this output for compatibility with existing downstream readers.
    loocv_df.sort_values(["test_mae", "fit_train_mae"]).to_csv(OUT_DIR / "spore_proxy_trials.csv", index=False)

    select_cols = [
        "rho", "beta_mp", "beta_rain", "beta_wind", "gamma_vpd", "k_mp",
        "tmin", "topt", "tmax", "roll_window", "baseline_q",
    ]
    loocv_group = loocv_df.copy()
    # Reduce floating-point fragmentation before grouping fold-selected candidates.
    dec = int(max(0, args.param_round_decimals))
    for c in select_cols:
        if c == "roll_window":
            loocv_group[c] = pd.to_numeric(loocv_group[c], errors="coerce").round().astype("Int64")
        else:
            loocv_group[c] = pd.to_numeric(loocv_group[c], errors="coerce").round(dec)
    leaderboard = (
        loocv_group.groupby(select_cols, dropna=False, as_index=False)
        .agg(
            cv_train_mae_mean=("fit_train_mae", "mean"),
            cv_val_mae_mean=("test_mae", "mean"),
            cv_train_mae_std=("fit_train_mae", "std"),
            cv_val_mae_std=("test_mae", "std"),
            n_selected_folds=("plot", "count"),
            fit_mae_full=("fit_mae_full", "mean"),
            constraint_satisfied_rate=("constraint_satisfied", "mean"),
        )
        .sort_values(["cv_val_mae_mean", "fit_mae_full", "cv_train_mae_mean"])
        .reset_index(drop=True)
    )
    min_folds = int(max(1, args.min_selected_folds))
    leaderboard_gate = leaderboard[leaderboard["n_selected_folds"] >= min_folds].copy()
    selection_constraint_relaxed = False
    if leaderboard_gate.empty:
        leaderboard_gate = leaderboard.copy()
        selection_constraint_relaxed = True
    leaderboard.to_csv(OUT_DIR / "spore_proxy_selection_compare.csv", index=False)
    best_row = leaderboard_gate.iloc[0].to_dict()
    best_params = _extract_params_from_trial_row(best_row)

    best_loss, detail = _evaluate_params(site_year_cache, best_params)
    detail.to_csv(OUT_DIR / "spore_proxy_fit_detail.csv", index=False)

    summary = {
        "best_loss_mae": float(best_loss),
        "n_site_year": int(len(site_year_cache)),
        "best_params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in best_params.items()},
        "cv_topk_frac": float(args.cv_topk_frac),
        "param_bounds": PARAM_BOUNDS,
        "selection_mode": "loocv_train_topk_then_val_then_min_full_fit_mae",
        "cv_train_mae_mean": float(loocv_df["fit_train_mae"].mean()),
        "cv_val_mae_mean": float(loocv_df["test_mae"].mean()),
        "cv_train_mae_std": float(loocv_df["fit_train_mae"].std(ddof=0)),
        "cv_val_mae_std": float(loocv_df["test_mae"].std(ddof=0)),
        "cv_constraint_satisfied_rate": float(loocv_df["constraint_satisfied"].mean()),
        "selected_n_folds": int(best_row.get("n_selected_folds", 0)),
        "min_selected_folds": int(min_folds),
        "selection_constraint_relaxed": bool(selection_constraint_relaxed),
        "param_round_decimals": int(dec),
    }
    with open(OUT_DIR / "spore_proxy_best.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Directly consumable config for historical inversion scripts.
    inversion_cfg = {
        "spore_pressure_params": {
            "rho": float(best_params["rho"]),
            "beta_mp": float(best_params["beta_mp"]),
            "beta_rain": float(best_params["beta_rain"]),
            "beta_wind": float(best_params["beta_wind"]),
            "gamma_vpd": float(best_params["gamma_vpd"]),
            "k_mp": float(best_params["k_mp"]),
            "tmin": float(best_params["tmin"]),
            "topt": float(best_params["topt"]),
            "tmax": float(best_params["tmax"]),
            "roll_window": int(round(best_params["roll_window"])),
            "baseline_q": float(best_params["baseline_q"]),
        },
        "fit_loss_mae": float(best_loss),
        "n_site_year": int(len(site_year_cache)),
        "source": "calibrate_spore_pressure_module.py",
        "files": {
            "best_summary": "spore_proxy_best.json",
            "fit_detail": "spore_proxy_fit_detail.csv",
            "trials": "spore_proxy_trials.csv",
            "loocv": "spore_proxy_loocv.csv",
            "selection_compare": "spore_proxy_selection_compare.csv",
        },
    }
    with open(OUT_DIR / "spore_pressure_params_for_inversion.json", "w", encoding="utf-8") as f:
        json.dump(inversion_cfg, f, indent=2)

    if ray_used:
        ray.shutdown()

    print(f"Saved: {OUT_DIR / 'spore_proxy_best.json'}")
    print(f"Saved: {OUT_DIR / 'spore_pressure_params_for_inversion.json'}")
    print(f"Saved: {OUT_DIR / 'spore_proxy_trials.csv'}")
    print(f"Saved: {OUT_DIR / 'spore_proxy_fit_detail.csv'}")
    print(f"Saved: {OUT_DIR / 'spore_proxy_loocv.csv'}")
    print(f"Saved: {OUT_DIR / 'spore_proxy_selection_compare.csv'}")


if __name__ == "__main__":
    main()
