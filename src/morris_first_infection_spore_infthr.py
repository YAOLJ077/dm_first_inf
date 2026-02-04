# -*- coding: utf-8 -*-
"""
Morris sensitivity analysis for FIRST infection model (ONLY architecture)
=========================================================================
Outputs:
- MAE (absolute error) sensitivities
- Signed error (directional bias) sensitivities
"""

import os
import sys
import json
import copy
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

# ======================================================
# Make project root importable
# ======================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# PATH SETUP
# ======================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "morris_spore_infthr_v2")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season_spore_infthr import CropSeasonFirstSporeInfThr
from model.config import config

# ======================================================
# GLOBAL SETTINGS
# ======================================================
N_JOBS = min(128, os.cpu_count())
RANDOM_SEED = 42
PENALTY_DAYS = 120.0

# Morris settings
N_TRAJ_LIST = [256, 512]
NUM_LEVELS = 6
SEEDS = [42, 123, 2024]
OPT_TRAJ = None  # set int to optimize trajectories

# ======================================================
# PARAMETER SPACE
# ======================================================
PARAM_RANGES = {
    "mp_tmp__rh_wet_night": (85, 95),
    "sporulation__MP_min": (3, 6),
    "sporulation__Tmin": (8, 15),
    "sporulation__Topt": (18, 24),
    "sporulation__Tmax": (25, 35),
    "dispersal__rain_min": (6.0, 12.0),
    "dispersal__wind_min": (4.0, 12.0),
    "deposition__rain_min": (4.0, 12.0),
    "zoospore_alive__rain": (3.0, 8.0),
    "zoospore_alive__vpd": (1.5, 4.0),
    "infection__infection_condition": (50, 150),
    "incubation__spl_end": (1.0, 1.5),
    "thresholds__min_spores": (0.05, 0.30),
    "spore__alpha_pos": (0.3, 2.5),
    "spore__alpha_neg": (0.0, 1.4),
}

# ======================================================
# LOAD ACTUAL DATA
# ======================================================
def load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"])
    return df

# ======================================================
# APPLY PARAMS
# ======================================================
def apply_params(base_cfg: dict, values: np.ndarray, names: List[str]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for v, k in zip(values, names):
        group, sub = k.split("__")
        if group not in cfg:
            cfg[group] = {}
        cfg[group][sub] = float(v)
    return cfg


def _normalize_spore_series(spore_vals: pd.Series) -> pd.Series:
    vals = spore_vals.dropna().astype(float)
    if vals.empty:
        return pd.Series(dtype=float)
    p95 = float(vals.quantile(0.95))
    if p95 <= 0:
        return spore_vals.fillna(0.0).astype(float) * 0.0
    norm = (np.log1p(spore_vals.astype(float)) / np.log1p(p95)).clip(lower=0.0, upper=1.0)
    # Weak detrend: short window + low quantile baseline to avoid over-subtraction
    obs = spore_vals[spore_vals.notna()]
    if len(obs.index) >= 2:
        gaps = obs.index.to_series().diff().dt.days.dropna()
        median_gap = float(gaps.median()) if not gaps.empty else 3.0
    else:
        median_gap = 3.0
    window = int(max(3, min(9, round(median_gap * 2))))
    rolling_norm = norm.rolling(window=window, min_periods=1, center=True).mean()
    if rolling_norm.dropna().size:
        target_pos = 0.4
        candidates = np.linspace(0.45, 0.75, 31)
        pos_fracs = [(rolling_norm > q).mean() for q in candidates]
        q_idx = int(np.argmin([abs(p - target_pos) for p in pos_fracs]))
        global_baseline = float(candidates[q_idx])
    else:
        global_baseline = 0.6
    return rolling_norm - global_baseline


def build_spore_day_map(actual_df: pd.DataFrame) -> Dict[Tuple[str, int], pd.Series]:
    spore_path = os.path.join(DATA_DIR, "Spore_Overall.xlsx")
    spore = pd.read_excel(spore_path)
    spore["date"] = pd.to_datetime(spore["date"])

    actual_df = actual_df.copy()
    actual_df["first_infection_date"] = pd.to_datetime(actual_df["first_infection_date"], errors="coerce")
    actual_inf = {
        (str(r.plot), int(r.year)): r.first_infection_date
        for r in actual_df.itertuples(index=False)
    }

    spore_day_map: Dict[Tuple[str, int], pd.Series] = {}
    for (plot, year), g in spore.groupby(["plot", "year"]):
        key = (str(plot), int(year))
        if key not in actual_inf:
            continue
        g = g.sort_values("date")
        obs = g[g["spore"].notna()].copy()
        if obs.empty:
            continue
        start = obs["date"].min().normalize()
        end = obs["date"].max().normalize()
        obs_inf = actual_inf.get(key)
        if pd.notna(obs_inf):
            end = min(end, obs_inf.normalize())
        if end < start:
            continue

        daily_index = pd.date_range(start=start, end=end, freq="D")
        series = pd.Series(index=daily_index, dtype=float)
        obs = obs[(obs["date"] >= start) & (obs["date"] <= end)]
        if obs.empty:
            continue
        series.loc[obs["date"].dt.normalize()] = obs["spore"].values
        series = series.interpolate(method="time", limit_direction="both")
        series = _normalize_spore_series(series)
        spore_day_map[key] = series

    return spore_day_map

# ======================================================
# DOY UTILITY
# ======================================================
def to_doy(dt) -> int:
    if pd.isna(dt):
        return None
    return pd.to_datetime(dt).dayofyear

# ======================================================
# ERROR FUNCTION (EVENT-BASED)
# ======================================================
def calc_error(actual_date, event_df: pd.DataFrame) -> float:
    """
    Signed error in DOY:
    negative → simulated earlier
    positive → simulated later
    """
    if event_df is None or event_df.empty:
        return np.nan

    first = event_df.iloc[0]
    doy_actual = to_doy(actual_date)
    doy_start = to_doy(first["oil_spot_start"])
    doy_end = to_doy(first["oil_spot_end"])

    if doy_actual is None or doy_start is None or doy_end is None:
        return np.nan

    if doy_end < doy_actual:
        return doy_end - doy_actual
    if doy_start > doy_actual:
        return doy_start - doy_actual
    return 0.0

# ======================================================
# ---- PROCESS-LOCAL JSON CACHE ----
# ======================================================
_RB_CACHE: Dict[str, dict] = {}

def load_rb_cached(rb_path: str) -> dict:
    if rb_path not in _RB_CACHE:
        with open(rb_path, "r") as f:
            _RB_CACHE[rb_path] = json.load(f)
    return _RB_CACHE[rb_path]

# ======================================================
# MORRIS EVALUATION
# ======================================================
def evaluate(
    X: np.ndarray,
    param_names: List[str],
    actual_df: pd.DataFrame,
    spore_day_map: Dict[Tuple[str, int], pd.Series]
) -> Tuple[np.ndarray, np.ndarray]:

    rows = list(actual_df.itertuples(index=False))
    n_samples = len(X)

    def run_one_sample(i):
        cfg = apply_params(config, X[i], param_names)
        signed_errs = []
        abs_errs = []
        n_total = 0
        n_valid = 0
        early = 0
        late = 0
        ontime = 0

        for row in rows:
            key = f"{row.plot}_{row.variety}_{row.first_infection_date.year}"
            rb_path = os.path.join(DATA_DIR, f"{key}_grape_rb.json")
            if not os.path.exists(rb_path):
                continue

            n_total += 1

            try:
                rb = load_rb_cached(rb_path)
                spore_day = spore_day_map.get((str(row.plot), int(row.first_infection_date.year)))
                cs = CropSeasonFirstSporeInfThr(csrb=rb, config=cfg, spore_day=spore_day)
                cs.simulate()
                e = calc_error(row.first_infection_date, cs.primary_chain_unique)
                if np.isfinite(e):
                    signed_errs.append(e)
                    abs_errs.append(abs(e))
                    if e < 0:
                        early += 1
                    elif e > 0:
                        late += 1
                    else:
                        ontime += 1
                    n_valid += 1
            except Exception:
                continue

        if n_valid == 0:
            return PENALTY_DAYS, PENALTY_DAYS, 0.0, 0.0, 0.0, 0

        return (
            float(np.mean(abs_errs)),
            float(np.mean(signed_errs)),
            float(early / n_valid),
            float(late / n_valid),
            float(ontime / n_valid),
            int(n_valid),
        )

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        batch_size=1,
        pre_dispatch="2*n_jobs"
    )(
        delayed(run_one_sample)(i)
        for i in tqdm(range(n_samples), desc="Morris samples")
    )

    results = np.asarray(results, dtype=object)
    y_mae = results[:, 0].astype(float)
    y_signed = results[:, 1].astype(float)
    stats = results[:, 2:].astype(float)
    return y_mae, y_signed, stats

# ======================================================
# MAIN
# ======================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Morris sensitivity analysis for FIRST infection model"
    )
    parser.add_argument(
        "--n_traj",
        type=int,
        default=None,
        help="Override N_TRAJ list with a single value (e.g., 256 or 512)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override SEEDS list with a single value (e.g., 2024)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    actual_df = load_actual()
    spore_day_map = build_spore_day_map(actual_df)

    param_names = list(PARAM_RANGES.keys())
    bounds = [PARAM_RANGES[k] for k in param_names]

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds
    }

    summary_rows = []

    n_traj_list = [args.n_traj] if args.n_traj is not None else N_TRAJ_LIST
    seed_list = [args.seed] if args.seed is not None else SEEDS

    for n_traj in n_traj_list:
        for seed in seed_list:
            np.random.seed(seed)
            print(
                f"\n=== Morris N_TRAJ={n_traj}, "
                f"NUM_LEVELS={NUM_LEVELS}, SEED={seed} ==="
            )

            X = morris_sample.sample(
                problem,
                n_traj,
                num_levels=NUM_LEVELS,
                optimal_trajectories=OPT_TRAJ,
                seed=seed
            )

            Y_mae, Y_signed, stats = evaluate(X, param_names, actual_df, spore_day_map)

            for label, Y in [("MAE", Y_mae), ("Signed", Y_signed)]:
                print(f"{label}: mean={np.mean(Y):.2f}, std={np.std(Y):.2f}")

                Si = morris_analyze.analyze(
                    problem,
                    X,
                    Y,
                    num_levels=NUM_LEVELS,
                    print_to_console=False
                )

                df = pd.DataFrame({
                    "Parameter": param_names,
                    "mu": Si["mu"],
                    "mu_star": Si["mu_star"],
                    "sigma": Si["sigma"],
                }).sort_values("mu_star", ascending=False)

                out = os.path.join(
                    RESULT_DIR,
                    f"morris_first_inf_spore_infthr_ONLY_{label}_N{n_traj}_seed{seed}.xlsx"
                )
                df.to_excel(out, index=False)
                print(f"Saved: {out}")

                for _, r in df.iterrows():
                    summary_rows.append({
                        "N_TRAJ": n_traj,
                        "seed": seed,
                        "metric": label,
                        "Parameter": r["Parameter"],
                        "mu": r["mu"],
                        "mu_star": r["mu_star"],
                        "sigma": r["sigma"],
                    })

            stats_out = os.path.join(
                RESULT_DIR,
                f"morris_first_inf_spore_infthr_early_late_stats_N{n_traj}_seed{seed}.csv"
            )
            stats_df = pd.DataFrame({
                "early_frac": stats[:, 0],
                "late_frac": stats[:, 1],
                "ontime_frac": stats[:, 2],
                "n_valid": stats[:, 3],
            })
            stats_df.to_csv(stats_out, index=False)
            print(f"Saved: {stats_out}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_out = os.path.join(
            RESULT_DIR,
            "morris_first_inf_spore_infthr_summary.csv"
        )
        summary_df.to_csv(summary_out, index=False)
        print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
