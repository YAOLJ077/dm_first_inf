import pandas as pd
import numpy as np
import math


class FirstInfectionSpore:
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

        # ===== TIME INDEX =====
        self.time_index = self.wh["datetime"].to_numpy(dtype="datetime64[ns]")

        # ===== Rossi fixed parameters =====
        self._ATTACHED_MORTALITY = {
            "a0": 9.27, "a1": 1.12, "a2": 0.04,
            "start_hour": 8, "end_hour": 20
        }

        self._DETACHED_MORTALITY = {
            "b0": 5.67, "b1": 0.47, "b2": 0.02
        }

        self._INCUBATION_PARAMS = {
            "a1": 45.1, "b1": 3.45, "c1": 0.073,
            "a2": 59.9, "b2": 4.55, "c2": 0.095
        }

        self._precompute_weather_fields()
        self._build_daily_index()

    # ======================================================
    def _precompute_weather_fields(self):

        wh = self.wh
        hour = wh["datetime"].dt.hour.values
        self.is_night = (hour >= 20) | (hour < 8)

        T = wh["air_temp"].values
        RH = wh["relative_humidity"].values

        self.VPD = (
            6.11 * np.exp((17.47 * T) / (239 + T))
            * (1 - RH / 100)
        )

        self.wet = wh["precip_acc_period_adjusted"].values > 0

    def _build_daily_index(self):
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
                    "airborne_amount": attached_pool,
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
    # S3. DEPOSITION (spore pressure adjustment)
    # ======================================================
    def stage_deposition(self):

        dep = self.cfg["deposition"]
        th = self.cfg["thresholds"]["min_spores"]
        p = self._DETACHED_MORTALITY
        spore_alpha_dep = (
            self.cfg.get("spore", {}).get("alpha_dep", 0.0)
            if isinstance(self.cfg, dict)
            else 0.0
        )
        spore_alpha_decay = (
            self.cfg.get("spore", {}).get("alpha_decay", 0.0)
            if isinstance(self.cfg, dict)
            else 0.0
        )
        cohorts = self.cs.dispersal_cohorts.copy()

        dep_dt, dep_amt = [], []

        for _, c in cohorts.iterrows():

            if c.dispersal_type == "rain":
                dep_dt.append(c.dispersal_datetime + pd.Timedelta(hours=1))
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
                    dep_time = self.wh.iloc[idx + rain_idx[0]].datetime
                    dep_idx = idx + rain_idx[0]

                    D = c.airborne_amount
                    for i in range(idx, dep_idx):
                        hr = self.wh.iloc[i]
                        if 8 <= hr.datetime.hour <= 18:
                            s_day = self._spore_day_value(hr.datetime)
                            denom = 1.0 + spore_alpha_decay * s_day
                            if denom <= 0.0:
                                denom = 0.1
                            lam = max(
                                0.0,
                                p["b0"] - p["b1"] * self.VPD[i]
                                + p["b2"] * self.VPD[i] ** 2
                            ) / 24.0
                            lam = lam / denom
                            D *= math.exp(-lam)
                            if D < th:
                                D = np.nan
                                break

                    if np.isnan(D):
                        dep_dt.append(pd.NaT)
                        dep_amt.append(np.nan)
                    else:
                        s_window = self._spore_window_value(
                            c.dispersal_datetime,
                            dep_time
                        )
                        D = D * (1.0 + spore_alpha_dep * s_window)
                        if D > c.airborne_amount:
                            D = c.airborne_amount
                        if D < th:
                            dep_dt.append(pd.NaT)
                            dep_amt.append(np.nan)
                            continue
                        dep_dt.append(dep_time)
                        dep_amt.append(D)

        cohorts["deposition_datetime"] = dep_dt
        cohorts["deposited_amount"] = dep_amt

        self.cs.deposition = cohorts.dropna(
            subset=["deposition_datetime"]
        ).reset_index(drop=True)

    def _spore_day_value(self, dt) -> float:
        spore_day = getattr(self.cs, "spore_day", None)
        if spore_day is None or spore_day.empty:
            return 0.0
        return float(spore_day.get(pd.to_datetime(dt).normalize(), 0.0))

    # ======================================================
    # Spore pressure window (daily index)
    # ======================================================
    def _spore_window_value(self, start_dt, end_dt) -> float:
        spore_day = getattr(self.cs, "spore_day", None)
        if spore_day is None or spore_day.empty:
            return 0.0
        start = pd.to_datetime(start_dt)
        end = pd.to_datetime(end_dt)
        if end <= start:
            return float(spore_day.get(start.normalize(), 0.0))

        start_date = start.normalize()
        end_date = end.normalize()

        total_hours = 0.0
        weighted = 0.0
        cur = start_date
        while cur <= end_date:
            day_start = cur
            day_end = cur + pd.Timedelta(days=1)
            win_start = max(start, day_start)
            win_end = min(end, day_end)
            hours = max(0.0, (win_end - win_start).total_seconds() / 3600.0)
            if hours > 0:
                val = float(spore_day.get(cur, 0.0))
                weighted += val * hours
                total_hours += hours
            cur = cur + pd.Timedelta(days=1)

        if total_hours <= 0:
            return 0.0
        return weighted / total_hours

    # ======================================================
    # S4. DETACHED SPORANGIA SURVIVAL
    # ======================================================
    def stage_detached_survival(self):

        p = self._DETACHED_MORTALITY
        th = self.cfg["thresholds"]["min_spores"]

        cohorts = self.cs.deposition.copy()
        alive_end = []

        for _, c in cohorts.iterrows():

            D = c.deposited_amount

            # Detached decay continues from deposition time
            start = np.searchsorted(
                self.time_index,
                np.datetime64(c.deposition_datetime)
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

        cohorts["sporangia_alive_end"] = alive_end

        self.cs.detached_survival = cohorts.reset_index(drop=True)



    # ======================================================
    # S5–S7. PRIMARY CHAIN (Rossi-compliant)
    # ======================================================
    def stage_primary_chain(self):

        inf = self.cfg["infection"]
        inc = self.cfg["incubation"]
        inc_fix = self._INCUBATION_PARAMS

        # -------- zoospore survival control (from config) --------
        rain_thr = self.cfg["zoospore_alive"]["rain"]
        vpd_thr = self.cfg["zoospore_alive"]["vpd"]

        # primary-style leaf wetness (USED ONLY FOR SURVIVAL GATES)
        moisture_of_leaf = (
            (self.wh["precip_acc_period_adjusted"].values > rain_thr)
            | (self.VPD <= vpd_thr)
        )

        results = []
        earliest_infection = None

        for _, c in self.cs.detached_survival.iterrows():

            # --------------------------------------------------
            # 1. Sporangia alive window
            # --------------------------------------------------
            start = np.searchsorted(
                self.time_index,
                np.datetime64(c.deposition_datetime)
            )
            end = np.searchsorted(
                self.time_index,
                np.datetime64(c.sporangia_alive_end)
            )

            if start >= end:
                continue

            # --------------------------------------------------
            # 2. Zoospore RELEASE (Rossi WD / TWD condition)
            #    NOTE: moisture is only a prerequisite, not the trigger
            # --------------------------------------------------
            wet = moisture_of_leaf[start:end]
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

                hit = np.where(WD >= WDmin)[0]
                if len(hit) > 0:
                    release_time = self.wh.iloc[start + idxs[hit[0]]].datetime
                    break

            if release_time is None:
                continue

            # --------------------------------------------------
            # 3. Zoospore SURVIVAL window (SUZ)
            #    from release until FIRST dry leaf
            # --------------------------------------------------
            rz = np.searchsorted(
                self.time_index,
                np.datetime64(release_time)
            )

            mol_after = moisture_of_leaf[rz:]

            dry_idx = np.where(~mol_after)[0]
            if len(dry_idx) > 0:
                suz_end = self.wh.iloc[rz + dry_idx[0]].datetime
            else:
                suz_end = self.wh.datetime.iloc[-1]

            if rz >= np.searchsorted(self.time_index, np.datetime64(suz_end)):
                continue

            # --------------------------------------------------
            # 4. INFECTION (Rossi WD × TWD condition, inside SUZ)
            # --------------------------------------------------
            start2 = rz
            end2 = np.searchsorted(
                self.time_index,
                np.datetime64(suz_end)
            )

            wh_window = self.wh.iloc[start2:end2]
            wet2 = moisture_of_leaf[start2:end2]
            if not wet2.any():
                continue

            grp2 = np.cumsum(wet2 != np.roll(wet2, 1))
            infection_time = None

            for g in np.unique(grp2[wet2]):
                idxs = np.where((grp2 == g) & wet2)[0]

                T = wh_window.iloc[idxs]["air_temp"].values
                WD = np.arange(1, len(idxs) + 1)
                TWD = np.cumsum(T) / WD

                hit = np.where(WD * TWD >= inf["infection_condition"])[0]
                if len(hit) > 0:
                    infection_time = self.wh.iloc[start2 + idxs[hit[0]]].datetime
                    break

            if infection_time is None:
                continue

            # --------------------------------------------------
            # 5. EARLY STOP (unchanged)
            # --------------------------------------------------
            if earliest_infection is None:
                earliest_infection = infection_time
            elif infection_time >= earliest_infection:
                break

            # --------------------------------------------------
            # 6. INCUBATION → OIL SPOT
            # --------------------------------------------------
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
                "zoospore_suz_end": suz_end,
                "infection_datetime": infection_time,
                "oil_spot_start": sub.iloc[idx1[0]].datetime,
                "oil_spot_end": sub.iloc[idx2[0]].datetime
            })

        self.cs.primary_chain = pd.DataFrame(results)
