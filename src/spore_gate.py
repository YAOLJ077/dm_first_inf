import pandas as pd
import numpy as np


class SporeObsGateWindow:
    """
    Observational soft gate based on sporangia monitoring windows.

    Rule:
    - Only days with observed spore value (spore not NaN) generate a window
    - Window = [date - 2 days, date]
    - spore == 0  -> suppress weight
    - spore > 0   -> support (weight = 1.0)
    - Outside windows -> no constraint (weight = 1.0)
    """

    def __init__(
        self,
        obs_file: str,
        site: str,
        year: int,
        w_zero: float = 0.3
    ):
        """
        Parameters
        ----------
        obs_file : str
            Path to spore_obs file
        site : str
            Site code (e.g. 'GL', 'DA', 'NN')
        year : int
            Year (e.g. 2022)
        w_zero : float
            Weight applied in windows where spore == 0
        """

        self.w_zero = w_zero

        # ---------- load ----------
        df = pd.read_csv(obs_file, sep=r"\s+|,", engine="python")
        df["date"] = pd.to_datetime(df["date"])

        # ---------- filter site-year ----------
        df = df[(df["site"] == site) & (df["year"] == year)].copy()

        if df.empty:
            self.windows = []
            return

        # ---------- keep only observed days (spore not NaN) ----------
        df_obs = df[~df["spore"].isna()].copy()

        self.windows = []

        for _, r in df_obs.iterrows():
            d = r["date"].normalize()
            spore = r["spore"]

            window_start = d - pd.Timedelta(days=2)
            window_end = d

            if spore > 0:
                w = 1.0
            else:
                w = self.w_zero

            self.windows.append({
                "start": window_start,
                "end": window_end,
                "weight": w
            })

    # --------------------------------------------------
    def support(self, t):
        """
        Return support weight for datetime t.
        """

        if not self.windows:
            return 1.0

        d = pd.to_datetime(t).normalize()

        for w in self.windows:
            if w["start"] <= d <= w["end"]:
                return w["weight"]

        return 1.0
