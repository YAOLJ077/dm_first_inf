import pandas as pd
import numpy as np
import math

from model.process_rules import (
    INCUBATION_COMPLETION,
    leaf_moisture_from_vpd,
    validate_incubation_completion,
)

from model.sporulation_timing import (
    SPORULATION_AVAILABILITY_HOUR,
    sporulation_dose_for_day,
    summarize_sporulation_nights,
)


class FirstInfectionSpore:
    """
    FIRST infection model
    (engine-optimized, biology-identical)
    """

    def __init__(self, cs, config):

        self.cs = cs
        self.cfg = config
        validate_incubation_completion(config)

        # ===== weather =====
        self.wh = cs.weather_hourly
        self.wd = cs.weather_daily

        # ===== TIME INDEX =====
        self.time_index = self.wh["datetime"].to_numpy(dtype="datetime64[ns]")

        # ===== Literature-fixed parameters =====
        # Brischetto et al. (2020, 2021), based on Blaeser and Weltzien (1979)
        self._ATTACHED_MORTALITY = {
            "a0": 9.27, "a1": 1.12, "a2": 0.04
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

    @staticmethod
    def _hourly_mortality_rate(mortality_index, coefficients):
        """Brischetto et al. (2020, 2021) hourly sporangium mortality."""
        if "a0" in coefficients:
            c0, c1, c2 = coefficients["a0"], coefficients["a1"], coefficients["a2"]
        else:
            c0, c1, c2 = coefficients["b0"], coefficients["b1"], coefficients["b2"]
        denominator = 24.0 * (
            c0 - c1 * mortality_index + c2 * mortality_index ** 2
        )
        if denominator <= 0:
            raise ValueError("VPD mortality equation produced a non-positive denominator")
        return 1.0 / denominator

    @staticmethod
    def _remaining_amount(initial_amount, cumulative_mortality):
        """Apply Brischetto's cumulative mortality to an initial dose."""
        return initial_amount * max(0.0, 1.0 - cumulative_mortality)

    @staticmethod
    def _release_condition(wet_hours, mean_wet_temperature):
        """Rossi release criterion in log space; non-positive T cannot release."""
        hours, temperature = np.broadcast_arrays(
            np.asarray(wet_hours, dtype=float),
            np.asarray(mean_wet_temperature, dtype=float),
        )
        valid = (
            np.isfinite(hours) & np.isfinite(temperature)
            & (hours > 0) & (temperature > 0)
        )
        result = np.zeros(hours.shape, dtype=bool)
        # Rearrange the log criterion to avoid division by near-zero T.
        log_factor = np.log(hours[valid]) + 1.022
        possible = log_factor > 0
        matches = np.zeros(log_factor.shape, dtype=bool)
        matches[possible] = (
            temperature[valid][possible] >= 19.634 / log_factor[possible]
        )
        result[valid] = matches
        return result

    @classmethod
    def _advance_mortality(
        cls, initial_amount, cumulative_mortality, mortality_index, coefficients
    ):
        cumulative_mortality += cls._hourly_mortality_rate(
            mortality_index, coefficients
        )
        return cumulative_mortality, cls._remaining_amount(
            initial_amount, cumulative_mortality
        )

    # ======================================================
    def _precompute_weather_fields(self):

        wh = self.wh
        T = wh["air_temp"].values
        RH = wh["relative_humidity"].values

        self.VPD = (
            6.11 * np.exp((17.47 * T) / (239 + T))
            * (1 - RH / 100)
        )
        # Predictor paired with the published Brischetto/Blaeser-Weltzien
        # mortality coefficients; it is distinct from physical VPD in hPa.
        self.sporangium_mortality_index = T * (1 - RH / 100)

    def _build_daily_index(self):
        dates = self.wh["date"].values.astype("datetime64[ns]")
        self.day_to_idxs = {
            d: np.where(dates == d)[0] for d in np.unique(dates)
        }

    # ======================================================
    # S1–S2. ATTACHED INVENTORY + DISPERSAL
    # ======================================================
    def stage_attached_and_dispersal(self):

        cfg = self.cfg
        disp = cfg["dispersal"]
        spore_alpha_air = (
            self.cfg.get("spore", {}).get("alpha_air", 0.0)
            if isinstance(self.cfg, dict)
            else 0.0
        )

        attached_cohorts = []
        cohorts = []
        cohort_num = 0

        d = summarize_sporulation_nights(
            self.wh, rh_threshold=cfg["mp_tmp"]["rh_wet_night"]
        )

        wd = self.wd.merge(
            d, left_on="datetime", right_on="production_date", how="left"
        )
        wd[["MP", "TMP"]] = wd[["MP", "TMP"]].fillna(0)

        inventory = []

        for _, day in wd.iterrows():
            date = np.datetime64(day.datetime, "ns")

            new_dose = sporulation_dose_for_day(day)
            dose_available = False

            for i in self.day_to_idxs[date]:
                hr = self.wh.iloc[i]
                if not dose_available and hr.datetime.hour >= SPORULATION_AVAILABILITY_HOUR:
                    if new_dose > 0:
                        attached_cohorts.append({
                            "production_datetime": hr.datetime,
                            "initial_sporangia_amount": new_dose,
                            "cumulative_mortality": 0.0,
                            "amount": new_dose,
                        })
                    dose_available = True

                viable_attached = []
                for attached in attached_cohorts:
                    cumulative, amount = self._advance_mortality(
                        attached["initial_sporangia_amount"],
                        attached["cumulative_mortality"],
                        self.sporangium_mortality_index[i],
                        self._ATTACHED_MORTALITY,
                    )
                    attached["cumulative_mortality"] = cumulative
                    attached["amount"] = amount
                    if amount > 0:
                        viable_attached.append(attached)
                attached_cohorts = viable_attached

                if not attached_cohorts:
                    continue

                if hr.precip_acc_period_adjusted >= disp["rain_min"]:
                    dtype = "rain"
                elif hr.get("wind_speed", 0) >= disp["wind_min"]:
                    dtype = "wind"
                else:
                    continue

                s_day = self._spore_day_value(hr.datetime)
                for attached in attached_cohorts:
                    airborne_amount = max(
                        0.0, attached["amount"] * (1.0 + spore_alpha_air * s_day)
                    )
                    if airborne_amount <= 0:
                        continue
                    cohorts.append({
                        "Num": cohort_num,
                        "production_datetime": attached["production_datetime"],
                        "initial_sporangia_amount": attached["initial_sporangia_amount"],
                        "attached_mortality_cumulative": attached["cumulative_mortality"],
                        "dispersal_datetime": hr.datetime,
                        "dispersal_type": dtype,
                        "airborne_amount": airborne_amount,
                    })
                    cohort_num += 1
                attached_cohorts = []

            inventory.append({
                "date": day.datetime,
                "attached_pool": sum(c["amount"] for c in attached_cohorts),
                "attached_cohort_count": len(attached_cohorts),
            })

        self.cs.attached_inventory = pd.DataFrame(inventory)
        self.cs.dispersal_cohorts = pd.DataFrame(
            cohorts,
            columns=[
                "Num", "production_datetime", "initial_sporangia_amount",
                "attached_mortality_cumulative", "dispersal_datetime",
                "dispersal_type", "airborne_amount",
            ],
        )

    # ======================================================
    # S3. DEPOSITION (spore pressure adjustment)
    # ======================================================
    def stage_deposition(self):

        dep = self.cfg["deposition"]
        p = self._DETACHED_MORTALITY
        spore_alpha_dep = (
            self.cfg.get("spore", {}).get("alpha_dep", 0.0)
            if isinstance(self.cfg, dict)
            else 0.0
        )
        spore_alpha_air = (
            self.cfg.get("spore", {}).get("alpha_air", 0.0)
            if isinstance(self.cfg, dict)
            else 0.0
        )
        cohorts = self.cs.dispersal_cohorts.copy()

        dep_dt, dep_amt, dep_mortality = [], [], []

        for _, c in cohorts.iterrows():

            if c.dispersal_type == "rain":
                dep_time = c.dispersal_datetime + pd.Timedelta(hours=1)
                dep_idx = np.searchsorted(self.time_index, np.datetime64(dep_time))
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
                    dep_mortality.append(np.nan)
                    continue
                else:
                    dep_time = self.wh.iloc[idx + rain_idx[0]].datetime
                    dep_idx = idx + rain_idx[0]

            idx = np.searchsorted(
                self.time_index, np.datetime64(c.dispersal_datetime)
            )
            if dep_idx >= len(self.wh):
                dep_dt.append(pd.NaT)
                dep_amt.append(np.nan)
                dep_mortality.append(np.nan)
                continue

            cumulative = 0.0
            for i in range(idx, dep_idx):
                cumulative += self._hourly_mortality_rate(
                    self.sporangium_mortality_index[i], p
                )
                if cumulative >= 1.0:
                    break

            amount = self._remaining_amount(c.airborne_amount, cumulative)
            s_window = self._spore_window_value(c.dispersal_datetime, dep_time)
            amount *= max(0.0, 1.0 + spore_alpha_dep * s_window)
            amount = min(amount, c.airborne_amount)
            if amount <= 0:
                dep_dt.append(pd.NaT)
                dep_amt.append(np.nan)
                dep_mortality.append(np.nan)
            else:
                dep_dt.append(dep_time)
                dep_amt.append(amount)
                dep_mortality.append(cumulative)

        cohorts["deposition_datetime"] = dep_dt
        cohorts["deposited_amount"] = dep_amt
        cohorts["detached_mortality_cumulative_at_deposition"] = dep_mortality

        self.cs.deposition = cohorts.dropna(
            subset=["deposition_datetime"]
        ).reset_index(drop=True)

    def _spore_day_value(self, dt) -> float:
        spore_day = getattr(self.cs, "spore_day", None)
        if spore_day is None or spore_day.empty:
            return 0.0
        return float(spore_day.get(pd.to_datetime(dt).normalize(), 0.0))

    def _spore_rolling_value(self, dt, window_days: int = 3) -> float:
        spore_roll = getattr(self.cs, "spore_day_roll", None)
        if spore_roll is None or spore_roll.empty:
            spore_day = getattr(self.cs, "spore_day", None)
            if spore_day is None or spore_day.empty:
                return 0.0
            end = pd.to_datetime(dt).normalize()
            start = end - pd.Timedelta(days=window_days - 1)
            idx = pd.date_range(start=start, end=end, freq="D")
            vals = [float(spore_day.get(d, 0.0)) for d in idx]
            return float(np.mean(vals)) if vals else 0.0
        return float(spore_roll.get(pd.to_datetime(dt).normalize(), 0.0))

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
        cohorts = self.cs.deposition.copy()
        alive_end = []
        final_mortality = []

        for _, c in cohorts.iterrows():

            cumulative = c.detached_mortality_cumulative_at_deposition
            start = np.searchsorted(
                self.time_index,
                np.datetime64(c.deposition_datetime)
            )

            end_time = self.wh.datetime.iloc[-1]

            for i in range(start, len(self.wh)):
                hr = self.wh.iloc[i]
                cumulative += self._hourly_mortality_rate(
                    self.sporangium_mortality_index[i], p
                )
                if cumulative >= 1.0:
                    end_time = hr.datetime
                    break

            alive_end.append(end_time)
            final_mortality.append(min(1.0, cumulative))

        cohorts["sporangia_alive_end"] = alive_end
        cohorts["detached_mortality_cumulative_at_end"] = final_mortality

        self.cs.detached_survival = cohorts.reset_index(drop=True)



    # ======================================================
    # S5–S7. PRIMARY CHAIN (Rossi-compliant)
    # ======================================================
    def stage_primary_chain(self):

        inf = self.cfg["infection"]
        inc_fix = self._INCUBATION_PARAMS

        # Same physical-VPD moisture proxy as FSIM.
        moisture_of_leaf = leaf_moisture_from_vpd(
            self.VPD, self.cfg["zoospore_alive"]["vpd"]
        )

        results = []
        spore_alpha_pos = (
            self.cfg.get("spore", {}).get("alpha_pos", 0.0)
            if isinstance(self.cfg, dict)
            else 0.0
        )
        spore_alpha_neg = (
            self.cfg.get("spore", {}).get("alpha_neg", 0.0)
            if isinstance(self.cfg, dict)
            else 0.0
        )

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

                hit = np.where(self._release_condition(WD, TWD))[0]
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

                base_cond = inf["infection_condition"]
                for j, idx in enumerate(idxs):
                    hr_dt = wh_window.iloc[idx]["datetime"]
                    p_day = self._spore_rolling_value(hr_dt, window_days=3)
                    if p_day >= 0:
                        eff_cond = base_cond * math.exp(-spore_alpha_pos * p_day)
                    else:
                        eff_cond = base_cond * math.exp(spore_alpha_neg * (-p_day))
                    if WD[j] * TWD[j] >= eff_cond:
                        infection_time = self.wh.iloc[start2 + idx].datetime
                        break
                if infection_time is not None:
                    break

            if infection_time is None:
                continue

            # --------------------------------------------------
            # 5. INCUBATION → OIL SPOT
            # --------------------------------------------------
            sub = self.wh[self.wh.datetime >= infection_time]
            T = sub["air_temp"].values

            INC1 = 1 / (24 * (inc_fix["a1"] - inc_fix["b1"] * T + inc_fix["c1"] * T ** 2))
            INC2 = 1 / (24 * (inc_fix["a2"] - inc_fix["b2"] * T + inc_fix["c2"] * T ** 2))

            c1 = np.cumsum(INC1)
            c2 = np.cumsum(INC2)

            idx1 = np.where(c1 >= INCUBATION_COMPLETION)[0]
            idx2 = np.where(c2 >= INCUBATION_COMPLETION)[0]

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
