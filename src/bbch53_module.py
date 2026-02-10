#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd


def load_hourly_weather_from_json(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "weather_hourly" in data:
        rows = data["weather_hourly"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported weather json format: {json_path}")
    df = pd.DataFrame(rows)
    if "datetime" not in df.columns or "air_temp" not in df.columns:
        raise ValueError(f"Missing required fields in {json_path}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["air_temp"] = pd.to_numeric(df["air_temp"], errors="coerce")
    df = df.dropna(subset=["datetime", "air_temp"]).sort_values("datetime").reset_index(drop=True)
    return df


def compute_agdd_hourly(
    hourly_df: pd.DataFrame,
    t_base: float = 10.0,
    start_datetime: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = hourly_df.copy()
    if start_datetime is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_datetime)]
    df = df.sort_values("datetime").reset_index(drop=True)
    df["gdd_hour"] = np.maximum(df["air_temp"].values - float(t_base), 0.0) / 24.0
    df["agdd"] = df["gdd_hour"].cumsum()
    return df


def estimate_bbch53_datetime(
    hourly_df: pd.DataFrame,
    theta_agdd: float,
    t_base: float = 10.0,
    start_datetime: Optional[pd.Timestamp] = None,
) -> pd.Timestamp:
    agdd_df = compute_agdd_hourly(hourly_df, t_base=t_base, start_datetime=start_datetime)
    idx = np.where(agdd_df["agdd"].values >= float(theta_agdd))[0]
    if len(idx) == 0:
        return pd.NaT
    return pd.to_datetime(agdd_df.iloc[int(idx[0])]["datetime"])


def estimate_bbch53_date(
    hourly_df: pd.DataFrame,
    theta_agdd: float,
    t_base: float = 10.0,
    start_datetime: Optional[pd.Timestamp] = None,
) -> pd.Timestamp:
    dt = estimate_bbch53_datetime(
        hourly_df=hourly_df,
        theta_agdd=theta_agdd,
        t_base=t_base,
        start_datetime=start_datetime,
    )
    if pd.isna(dt):
        return pd.NaT
    return pd.to_datetime(dt).normalize()

