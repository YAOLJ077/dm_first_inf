# -*- coding: utf-8 -*-
"""
Sporangia-based support function for infection events
=====================================================
Pure observation logic.
NO model logic, NO MAE, NO calibration.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


class SporangiaSupport:
    """
    Compute sporangia support score for a given datetime
    """

    def __init__(
        self,
        obs_df: pd.DataFrame,
        place: str,
        year: int,
        window_expand_days: int = 3,
        decay: float = 0.5,
    ):
        """
        Parameters
        ----------
        obs_df : DataFrame
            sporangia_obs with columns [Place, Year, Date, Spore]
        place : str
        year : int
        window_expand_days : int
            ± days around observed sporangia dates
        decay : float
            exponential decay rate outside exact window
        """

        self.decay = decay

        df = obs_df[
            (obs_df["Place"] == place)
            & (obs_df["Year"] == year)
            & (obs_df["Spore"] > 0)
        ].copy()

        self.windows = []

        for d in df["Date"]:
            self.windows.append((
                d - timedelta(days=window_expand_days),
                d + timedelta(days=window_expand_days)
            ))

    def support(self, dt: pd.Timestamp) -> float:
        """
        Compute support score ∈ (0, 1]
        """

        if not self.windows:
            return 0.0

        dt = pd.to_datetime(dt)

        scores = []

        for start, end in self.windows:
            if start <= dt <= end:
                # inside window → maximum support
                scores.append(1.0)
            else:
                # distance-based decay
                dist = min(abs((dt - start).days), abs((dt - end).days))
                scores.append(np.exp(-self.decay * dist))

        return float(max(scores))
