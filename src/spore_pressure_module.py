#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_vpd(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    return 6.11 * np.exp((17.47 * temp_c) / (239 + temp_c)) * (1.0 - rh / 100.0)


def build_daily_features(
    hourly_df: pd.DataFrame,
    rh_wet_night: float = 85.0,
    night_start: int = 20,
    night_end: int = 8,
) -> pd.DataFrame:
    df = hourly_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["air_temp"] = pd.to_numeric(df["air_temp"], errors="coerce")
    df["relative_humidity"] = pd.to_numeric(df["relative_humidity"], errors="coerce")
    df["precip_acc_period_adjusted"] = pd.to_numeric(df["precip_acc_period_adjusted"], errors="coerce").fillna(0.0)
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["datetime", "air_temp", "relative_humidity"]).reset_index(drop=True)

    hour = df["datetime"].dt.hour.values
    is_night = (hour >= night_start) | (hour < night_end)
    is_day = (hour >= night_end) & (hour < night_start)
    wet_night = is_night & (df["relative_humidity"].values >= rh_wet_night)
    vpd = compute_vpd(df["air_temp"].values, df["relative_humidity"].values)

    df["date"] = df["datetime"].dt.normalize()
    df["is_night"] = is_night
    df["is_day"] = is_day
    df["wet_night"] = wet_night
    df["vpd"] = vpd

    df["tmp_night_only"] = np.where(df["is_night"], df["air_temp"], np.nan)
    df["vpd_day_only"] = np.where(df["is_day"], df["vpd"], np.nan)

    g = df.groupby("date", as_index=False)
    daily = g.agg(
        MP_night=("wet_night", "sum"),
        TMP_night=("tmp_night_only", "mean"),
        Rain_day=("precip_acc_period_adjusted", "sum"),
        Wind_day=("wind_speed", "mean"),
        VPD_day=("vpd_day_only", "mean"),
    )
    daily["TMP_night"] = daily["TMP_night"].fillna(daily["TMP_night"].median())
    daily["VPD_day"] = daily["VPD_day"].fillna(daily["VPD_day"].median())
    return daily.sort_values("date").reset_index(drop=True)


def temp_response_triangular(t: np.ndarray, tmin: float, topt: float, tmax: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = (t > tmin) & (t < tmax) & (tmax > topt > tmin)
    out[mask] = ((t[mask] - tmin) / (topt - tmin)) * ((tmax - t[mask]) / (tmax - topt))
    out = np.clip(out, 0.0, 1.0)
    return out


def moisture_response(mp: np.ndarray, k_mp: float) -> np.ndarray:
    return np.clip(1.0 - np.exp(-k_mp * np.maximum(mp, 0.0)), 0.0, 1.0)


def simulate_latent_spore_state(daily_df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    d = daily_df.copy().sort_values("date").reset_index(drop=True)

    rho = float(params["rho"])
    beta_mp = float(params["beta_mp"])
    beta_rain = float(params["beta_rain"])
    beta_wind = float(params["beta_wind"])
    gamma_vpd = float(params["gamma_vpd"])
    k_mp = float(params["k_mp"])
    tmin = float(params["tmin"])
    topt = float(params["topt"])
    tmax = float(params["tmax"])

    f_t = temp_response_triangular(d["TMP_night"].values, tmin=tmin, topt=topt, tmax=tmax)
    f_m = moisture_response(d["MP_night"].values, k_mp=k_mp)

    source = beta_mp * f_t * f_m + beta_rain * np.maximum(d["Rain_day"].values, 0.0) + beta_wind * np.maximum(d["Wind_day"].values, 0.0)
    loss = gamma_vpd * np.maximum(d["VPD_day"].values, 0.0)

    A = np.zeros(len(d), dtype=float)
    a_prev = 0.0
    for i in range(len(d)):
        a_now = max(0.0, rho * a_prev + source[i] - loss[i])
        A[i] = a_now
        a_prev = a_now

    d["A_latent"] = A
    d["source"] = source
    d["loss"] = loss
    return d


def latent_to_pressure(
    latent_daily: pd.DataFrame,
    roll_window: int = 3,
    baseline_q: float = 0.60,
) -> pd.DataFrame:
    out = latent_daily.copy().sort_values("date").reset_index(drop=True)
    a = out["A_latent"].astype(float)
    p95 = float(a.quantile(0.95)) if len(a) > 0 else 0.0
    if p95 <= 0:
        norm = pd.Series(np.zeros(len(a)), index=out.index, dtype=float)
    else:
        norm = (np.log1p(a) / np.log1p(p95)).clip(lower=0.0, upper=1.0)

    roll = norm.rolling(window=int(max(1, roll_window)), min_periods=1, center=True).mean()
    baseline = float(roll.quantile(float(baseline_q))) if len(roll) > 0 else 0.0
    out["S_day_raw"] = norm
    out["S_day_roll"] = roll
    out["S_day"] = roll - baseline
    return out
