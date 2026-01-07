import pandas as pd
import numpy as np
from datetime import timedelta


class SporangiaObservation:
    """
    Observation operator for airborne sporangia (fork sampler)
    Produces obs_support(t) ∈ [0, 1]
    """

    def __init__(
        self,
        obs_df: pd.DataFrame,
        place: str,
        year: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        window_days: int = 1,
        w_positive: float = 1.0,
        w_negative: float = 0.1,
        w_unknown: float = 0.5,
    ):
        """
        Parameters
        ----------
        obs_df : full sporangia_obs dataframe
        place, year : identify site-season
        start_date : BBCH53 or first sampling date
        end_date : real first infection date
        window_days : half-window around sampling date
        w_positive : support when spore > 0
        w_negative : support when spore == 0
        w_unknown : support when no sampling info
        """

        self.place = place
        self.year = year
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.window = timedelta(days=window_days)

        self.w_positive = w_positive
        self.w_negative = w_negative
        self.w_unknown = w_unknown

        df = obs_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        df = df[
            (df["Place"] == place)
            & (df["Year"] == year)
            & (df["Date"] >= self.start_date)
            & (df["Date"] <= self.end_date)
        ].reset_index(drop=True)

        self.df = df

    # --------------------------------------------------
    def support(self, t: pd.Timestamp) -> float:
        """
        Return observational support at time t
        """

        if t < self.start_date or t > self.end_date:
            return self.w_unknown

        hits = self.df[
            (self.df["Date"] >= t - self.window)
            & (self.df["Date"] <= t + self.window)
        ]

        if hits.empty:
            return self.w_unknown

        # Priority: any positive detection dominates
        if (hits["Spore"] > 0).any():
            return self.w_positive

        # Explicit negative detections only
        if (hits["Spore"] == 0).all():
            return self.w_negative

        return self.w_unknown
