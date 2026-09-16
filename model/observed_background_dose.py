"""External observed-background airborne dose provider."""
from __future__ import annotations
import numpy as np
import pandas as pd


class ObservedBackgroundDose:
    """Convert interval-expanded relative pressure into external dose."""
    def __init__(self, pressure: pd.Series | None, beta_sp: float = 0.0):
        self.pressure = pressure.copy() if pressure is not None else pd.Series(dtype=float)
        self.pressure.index = pd.to_datetime(self.pressure.index).normalize()
        self.beta_sp = float(beta_sp)
        if not np.isfinite(self.beta_sp) or self.beta_sp < 0:
            raise ValueError("beta_sp must be finite and non-negative")

    def dose_at(self, datetime) -> float:
        return self.beta_sp * self.pressure_at(datetime)

    def pressure_at(self, datetime) -> float:
        value = float(self.pressure.get(pd.Timestamp(datetime).normalize(), 0.0))
        if not np.isfinite(value):
            raise ValueError("background pressure must be finite")
        return value
