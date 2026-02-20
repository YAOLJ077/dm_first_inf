#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ACTUAL_DATA_PATH = DATA_DIR / "actual_data.xlsx"

from model.config import config as base_config
from model.crop_season import CropSeasonFirst
from model.crop_season_spore_infthr import CropSeasonFirstSporeInfThr


def load_actual() -> pd.DataFrame:
    df = pd.read_excel(ACTUAL_DATA_PATH)
    df["first_infection_date"] = pd.to_datetime(df["first_infection_date"], errors="coerce")
    return df.reset_index(drop=True)


def apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    for k, v in values.items():
        if "__" not in k:
            continue
        group, sub = k.split("__", 1)
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
    obs = spore_vals[spore_vals.notna()]
    if len(obs.index) >= 2:
        gaps = obs.index.to_series().diff().dt.days.dropna()
        median_gap = float(gaps.median()) if not gaps.empty else 3.0
    else:
        median_gap = 3.0
    window = int(max(3, min(9, round(median_gap * 2))))
    rolling_norm = norm.rolling(window=window, min_periods=1, center=True).mean()
    global_baseline = 0.6
    if rolling_norm.dropna().size:
        target_pos = 0.4
        candidates = np.linspace(0.45, 0.75, 31)
        pos_fracs = [(rolling_norm > q).mean() for q in candidates]
        q_idx = int(np.argmin([abs(p - target_pos) for p in pos_fracs]))
        global_baseline = float(candidates[q_idx])
    return rolling_norm - global_baseline


def build_spore_day_map(actual_df: pd.DataFrame) -> Dict[tuple[str, int], pd.Series]:
    spore_path = DATA_DIR / "Spore_Overall.xlsx"
    spore = pd.read_excel(spore_path)
    spore["date"] = pd.to_datetime(spore["date"], errors="coerce")

    actual_inf = {
        (str(r.plot), int(r.year)): pd.to_datetime(r.first_infection_date, errors="coerce")
        for r in actual_df.itertuples(index=False)
    }

    out: Dict[tuple[str, int], pd.Series] = {}
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
        out[key] = _normalize_spore_series(series)
    return out


def _calc_error(actual_date: pd.Timestamp, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return np.nan
    first = event_df.iloc[0]
    doy_actual = pd.to_datetime(actual_date).dayofyear
    doy_start = pd.to_datetime(first["oil_spot_start"]).dayofyear
    doy_end = pd.to_datetime(first["oil_spot_end"]).dayofyear
    if doy_end < doy_actual:
        return float(doy_end - doy_actual)
    if doy_start > doy_actual:
        return float(doy_start - doy_actual)
    return 0.0


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _simulate_one(
    rb_path: Path,
    actual_date: pd.Timestamp,
    cfg: Dict,
    use_spore: bool,
    spore_day: pd.Series | None = None,
) -> Dict:
    if not rb_path.exists():
        return {
            "pred_start": pd.NaT,
            "pred_end": pd.NaT,
            "error_days": np.nan,
            "abs_error_days": np.nan,
            "status": "missing_rb",
        }
    try:
        with open(rb_path, "r", encoding="utf-8") as f:
            rb = json.load(f)
        if use_spore:
            cs = CropSeasonFirstSporeInfThr(csrb=rb, config=cfg, spore_day=spore_day)
        else:
            cs = CropSeasonFirst(csrb=rb, config=cfg)
        cs.simulate()
        ev = cs.primary_chain_unique
        if ev is None or ev.empty:
            return {
                "pred_start": pd.NaT,
                "pred_end": pd.NaT,
                "error_days": np.nan,
                "abs_error_days": np.nan,
                "status": "no_event",
            }
        first = ev.iloc[0]
        pred_start = pd.to_datetime(first["oil_spot_start"]).normalize()
        pred_end = pd.to_datetime(first["oil_spot_end"]).normalize()
        err = _calc_error(actual_date, ev)
        return {
            "pred_start": pred_start,
            "pred_end": pred_end,
            "error_days": float(err),
            "abs_error_days": float(abs(err)) if np.isfinite(err) else np.nan,
            "status": "ok",
        }
    except Exception as e:  # pragma: no cover
        return {
            "pred_start": pd.NaT,
            "pred_end": pd.NaT,
            "error_days": np.nan,
            "abs_error_days": np.nan,
            "status": f"error:{type(e).__name__}",
        }


def _iter_rows(df: pd.DataFrame):
    if tqdm is not None:
        return tqdm(list(df.itertuples(index=False)), desc="Simulating", total=len(df))
    return df.itertuples(index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Run FIM/FIM-S with final selected params and export obs-vs-sim dates.")
    p.add_argument("--baseline-json", default=str(PROJECT_ROOT / "result" / "calibration_sum" / "final_baseline_params.json"))
    p.add_argument("--spore-json", default=str(PROJECT_ROOT / "result" / "calibration_sum" / "final_spore_params.json"))
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "result" / "calibration_sum"))
    p.add_argument("--out-name", default="final_fim_fims_obs_vs_sim_long.csv")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b_json = _load_json(Path(args.baseline_json))
    s_json = _load_json(Path(args.spore_json))
    b_params = b_json.get("params", {})
    s_params = s_json.get("params", {})
    if not b_params or not s_params:
        raise ValueError("Missing 'params' in final parameter json files.")

    cfg_b = apply_params(base_config, b_params)
    cfg_s = apply_params(base_config, s_params)

    actual_df = load_actual()
    spore_day_map = build_spore_day_map(actual_df)

    rows_out: List[Dict] = []
    for row in _iter_rows(actual_df):
        plot = str(row.plot)
        variety = str(row.variety)
        year = int(pd.to_datetime(row.first_infection_date).year)
        actual_date = pd.to_datetime(row.first_infection_date).normalize()
        rb_path = DATA_DIR / f"{plot}_{variety}_{year}_grape_rb.json"

        pred_b = _simulate_one(
            rb_path=rb_path,
            actual_date=actual_date,
            cfg=cfg_b,
            use_spore=False,
            spore_day=None,
        )
        pred_s = _simulate_one(
            rb_path=rb_path,
            actual_date=actual_date,
            cfg=cfg_s,
            use_spore=True,
            spore_day=spore_day_map.get((plot, year)),
        )

        rows_out.append(
            {
                "model": "FIM",
                "plot": plot,
                "variety": variety,
                "year": year,
                "obs_first_infection_date": actual_date,
                "pred_start": pred_b["pred_start"],
                "pred_end": pred_b["pred_end"],
                "error_days": pred_b["error_days"],
                "abs_error_days": pred_b["abs_error_days"],
                "status": pred_b["status"],
            }
        )
        rows_out.append(
            {
                "model": "FIM-S",
                "plot": plot,
                "variety": variety,
                "year": year,
                "obs_first_infection_date": actual_date,
                "pred_start": pred_s["pred_start"],
                "pred_end": pred_s["pred_end"],
                "error_days": pred_s["error_days"],
                "abs_error_days": pred_s["abs_error_days"],
                "status": pred_s["status"],
            }
        )

    out_df = pd.DataFrame(rows_out).sort_values(["model", "plot", "year"]).reset_index(drop=True)
    out_csv = out_dir / args.out_name
    out_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
