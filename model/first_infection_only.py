import pandas as pd
import numpy as np
import math


class FirstInfection:
    """
    FIRST infection model
    (engine-optimized, biology-identical)
    """

    def __init__(self, cs, config):

        self.cs = cs
        self.cfg = config

        # ===== weather =====
        self.wh = cs.weather_hourly
        self.wd = cs.weather_daily

        # ===== TIME INDEX (CRITICAL FIX) =====
        # MUST be numpy datetime64 for searchsorted
        self.time_index = self.wh["datetime"].to_numpy(dtype="datetime64[ns]")

        # ===== Rossi fixed parameters =====
        self._ATTACHED_MORTALITY = {
            "a0": 9.27, "a1": 1.12, "a2": 0.04,
            "start_hour": 8, "end_hour": 18
        }

        self._DETACHED_MORTALITY = {
            "b0": 5.67, "b1": 0.47, "b2": 0.02
        }

        self._INCUBATION_PARAMS = {
            "a1": 45.1, "b1": 3.45, "c1": 0.073,
            "a2": 59.9, "b2": 4.55, "c2": 0.095
        }

        # ===== precompute meteorology =====
        self._precompute_weather_fields()
        self._build_daily_index()

    # ======================================================
    # PRECOMPUTE WEATHER FIELDS
    # ======================================================
    def _precompute_weather_fields(self):

        wh = self.wh

        hour = wh["datetime"].dt.hour.values
        self.is_night = (hour >= 20) | (hour < 8)

        T = wh["air_temp"].values
        RH = wh["relative_humidity"].values

        self.VPD = (
            0.6108 * np.exp((17.27 * T) / (T + 237.3))
            * (1 - RH / 100)
        )

        self.wet = wh["precip_acc_period_adjusted"].values > 0

    def _build_daily_index(self):
        """cache hourly indices per day"""
        dates = self.wh["date"].values
        self.day_to_idxs = {
            d: np.where(dates == d)[0] for d in np.unique(dates)
        }

    # ======================================================
    # S1–S2. ATTACHED INVENTORY + DISPERSAL
    # ======================================================
    def stage_attached_and_dispersal(self):

        cfg = self.cfg
        sp = cfg["sporulation"]
        disp = cfg["dispersal"]
        th = cfg["thresholds"]["min_spores"]
        k = 0.2

        attached_pool = 0.0
        cohorts = []
        cohort_num = 0

        h = self.wh.copy()
        h["is_night"] = self.is_night
        h["wet_night"] = h["relative_humidity"] >= cfg["mp_tmp"]["rh_wet_night"]

        night_wet = h[h.is_night & h.wet_night]

        d = (
            night_wet.groupby("date")
            .agg(MP=("wet_night", "sum"), TMP=("air_temp", "mean"))
            .reset_index()
        )

        wd = self.wd.merge(d, left_on="datetime", right_on="date", how="left")
        wd[["MP", "TMP"]] = wd[["MP", "TMP"]].fillna(0)

        inventory = []

        for _, day in wd.iterrows():
            date = day.datetime

            if day.MP >= sp["MP_min"] and sp["Tmin"] < day.TMP < sp["Tmax"]:
                fT = ((day.TMP - sp["Tmin"]) / (sp["Topt"] - sp["Tmin"])) * \
                     ((sp["Tmax"] - day.TMP) / (sp["Tmax"] - sp["Topt"]))
                fM = 1 - math.exp(-k * day.MP)
                attached_pool += max(0.0, min(1.0, fT * fM))

            for i in self.day_to_idxs[date]:
                hr = self.wh.iloc[i]

                if 8 <= hr.datetime.hour <= 18:
                    lam = max(
                        0.0,
                        self._ATTACHED_MORTALITY["a0"]
                        - self._ATTACHED_MORTALITY["a1"] * self.VPD[i]
                        + self._ATTACHED_MORTALITY["a2"] * self.VPD[i] ** 2
                    ) / 24.0
                    attached_pool *= math.exp(-lam)

                if attached_pool < th:
                    continue

                if hr.precip_acc_period_adjusted >= disp["rain_min"]:
                    dtype = "rain"
                elif hr.get("wind_speed", 0) >= disp["wind_min"]:
                    dtype = "wind"
                else:
                    continue

                cohorts.append({
                    "Num": cohort_num,
                    "dispersal_datetime": hr.datetime,
                    "dispersal_type": dtype,
                    "airborne_amount": attached_pool
                })
                cohort_num += 1
                attached_pool = 0.0

            inventory.append({
                "date": day.datetime,
                "attached_pool": attached_pool
            })

        self.cs.attached_inventory = pd.DataFrame(inventory)
        self.cs.dispersal_cohorts = pd.DataFrame(cohorts)

    # ======================================================
    # S3. DEPOSITION
    # ======================================================
    def stage_deposition(self):

        dep = self.cfg["deposition"]
        cohorts = self.cs.dispersal_cohorts.copy()

        dep_dt, dep_amt = [], []

        for _, c in cohorts.iterrows():

            if c.dispersal_type == "rain":
                dep_dt.append(c.dispersal_datetime)
                dep_amt.append(c.airborne_amount)
            else:
                idx = np.searchsorted(
                    self.time_index,
                    np.datetime64(c.dispersal_datetime)
                )
                rain_idx = np.where(
                    self.wh.iloc[idx:]["precip_acc_period_adjusted"].values
                    >= dep["rain_min"]
                )[0]

                if len(rain_idx) == 0:
                    dep_dt.append(pd.NaT)
                    dep_amt.append(np.nan)
                else:
                    dep_dt.append(
                        self.wh.iloc[idx + rain_idx[0]].datetime
                    )
                    dep_amt.append(c.airborne_amount)

        cohorts["deposition_datetime"] = dep_dt
        cohorts["deposited_amount"] = dep_amt

        self.cs.deposition = cohorts.dropna(
            subset=["deposition_datetime"]
        ).reset_index(drop=True)

    # ======================================================
    # S4. DETACHED SURVIVAL
    # ======================================================
    def stage_detached_survival(self):

        p = self._DETACHED_MORTALITY
        th = self.cfg["thresholds"]["min_spores"]

        cohorts = self.cs.deposition.copy()
        alive_end = []

        for _, c in cohorts.iterrows():

            D = c.deposited_amount
            start = np.searchsorted(
                self.time_index,
                np.datetime64(c.dispersal_datetime)
            )

            end_time = self.wh.datetime.iloc[-1]

            for i in range(start, len(self.wh)):
                hr = self.wh.iloc[i]
                if 8 <= hr.datetime.hour <= 18:
                    lam = max(
                        0.0,
                        p["b0"] - p["b1"] * self.VPD[i]
                        + p["b2"] * self.VPD[i] ** 2
                    ) / 24.0
                    D *= math.exp(-lam)
                    if D < th:
                        end_time = hr.datetime
                        break

            alive_end.append(end_time)

        cohorts["detached_alive_end"] = alive_end
        self.cs.detached_survival = cohorts.reset_index(drop=True)

    # ======================================================
    # S5–S7. PRIMARY CHAIN (EARLY STOP)
    # ======================================================
    def stage_primary_chain(self):

        inf = self.cfg["infection"]
        inc = self.cfg["incubation"]
        inc_fix = self._INCUBATION_PARAMS

        results = []
        earliest_infection = None

        for _, c in self.cs.detached_survival.iterrows():

            start = np.searchsorted(
                self.time_index,
                np.datetime64(c.deposition_datetime)
            )
            end = np.searchsorted(
                self.time_index,
                np.datetime64(c.detached_alive_end)
            )
            if start >= end:
                continue

            wet = self.wet[start:end]
            if not wet.any():
                continue

            grp = np.cumsum(wet != np.roll(wet, 1))
            release_time = None

            for g in np.unique(grp[wet]):
                idxs = np.where((grp == g) & wet)[0]
                T = self.wh.iloc[start:end].iloc[idxs]["air_temp"].values
                WD = np.arange(1, len(idxs) + 1)
                TWD = np.cumsum(T) / WD

                with np.errstate(divide="ignore"):
                    WDmin = np.exp((-1.022 + 19.634) / TWD)

                if (WD >= WDmin).any():
                    release_time = self.wh.iloc[
                        start + idxs[np.argmax(WD >= WDmin)]
                    ].datetime
                    break

            if release_time is None:
                continue

            start2 = np.searchsorted(
                self.time_index,
                np.datetime64(release_time)
            )
            wet2 = self.wet[start2:]
            grp2 = np.cumsum(wet2 != np.roll(wet2, 1))

            infection_time = None
            for g in np.unique(grp2[wet2]):
                idxs = np.where((grp2 == g) & wet2)[0]
                T = self.wh.iloc[start2:].iloc[idxs]["air_temp"].values
                WD = np.arange(1, len(idxs) + 1)
                TWD = np.cumsum(T) / WD

                if (WD * TWD >= inf["infection_condition"]).any():
                    infection_time = self.wh.iloc[
                        start2 + idxs[np.argmax(WD * TWD >= inf["infection_condition"])]
                    ].datetime
                    break

            if infection_time is None:
                continue

            if earliest_infection is None:
                earliest_infection = infection_time
            elif infection_time >= earliest_infection:
                break

            sub = self.wh[self.wh.datetime >= infection_time]
            T = sub["air_temp"].values

            INC1 = 1 / (24 * (inc_fix["a1"] - inc_fix["b1"] * T + inc_fix["c1"] * T ** 2))
            INC2 = 1 / (24 * (inc_fix["a2"] - inc_fix["b2"] * T + inc_fix["c2"] * T ** 2))

            c1 = np.cumsum(INC1)
            c2 = np.cumsum(INC2)

            idx1 = np.where(c1 >= inc["spl_end"])[0]
            idx2 = np.where(c2 >= inc["spl_end"])[0]

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            results.append({
                **c.to_dict(),
                "zoospore_release_datetime": release_time,
                "infection_datetime": infection_time,
                "oil_spot_start": sub.iloc[idx1[0]].datetime,
                "oil_spot_end": sub.iloc[idx2[0]].datetime
            })

        self.cs.primary_chain = pd.DataFrame(results)
