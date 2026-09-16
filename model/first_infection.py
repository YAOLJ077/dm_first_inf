import pandas as pd
import numpy as np
from model.parameter_table import TABLE_CONFIG

from model.process_rules import (
    INCUBATION_COMPLETION,
    leaf_moisture_from_vpd,
    validate_incubation_completion,
)

from model.sporulation_timing import (
    SPORULATION_MIN_MOIST_HOURS,
    SPORULATION_AVAILABILITY_HOUR,
    sporulation_dose_for_day,
    summarize_sporulation_nights,
)


class FirstInfection:
    """FSIM with merged transfer, optionally controlled by one wind gate."""

    def __init__(self, cs, config, *, use_wind_gate=None, external_dose_provider=None):

        self.cs = cs
        self.cfg = config
        self.external_dose_provider = external_dose_provider
        self.use_wind_gate = (
            config.get("dispersal", {}).get("use_wind_gate", True)
            if use_wind_gate is None else use_wind_gate
        )
        if not isinstance(self.use_wind_gate, (bool, np.bool_)):
            raise ValueError("use_wind_gate must be True or False")
        validate_incubation_completion(config)

        # ===== weather =====
        self.wh = cs.weather_hourly
        self.wd = cs.weather_daily

        # ===== TIME INDEX =====
        self.time_index = self.wh["datetime"].to_numpy(dtype="datetime64[ns]")

        # ===== Literature-fixed parameters =====
        # Brischetto et al. (2020, 2021), based on Blaeser and Weltzien (1979)
        self._ATTACHED_MORTALITY = TABLE_CONFIG['fixed']['attached_mortality'].copy()
        self._DETACHED_MORTALITY = TABLE_CONFIG['fixed']['detached_mortality'].copy()
        self._INCUBATION_PARAMS = TABLE_CONFIG['fixed']['incubation'].copy()

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
        relation = TABLE_CONFIG['fixed']['release']
        log_factor = np.log(hours[valid]) + relation['offset']
        possible = log_factor > 0
        matches = np.zeros(log_factor.shape, dtype=bool)
        matches[possible] = (
            temperature[valid][possible] >= relation['temperature_coefficient'] / log_factor[possible]
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

    def infection_condition_at(self, datetime):
        """Return the infection requirement for one hourly time step.

        This hook keeps FSIM-S structurally nested in FSIM: the spore-informed
        model may change the threshold, while all cohort and weather processes
        remain identical.
        """
        return float(self.cfg["infection"]["infection_condition"])

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
        transfer_type = "wind_gated" if self.use_wind_gate else "merged"
        wind_values = None
        wind_min = None
        if self.use_wind_gate:
            wind_min = cfg.get("dispersal", {}).get("wind_min")
            if wind_min is None or not np.isfinite(wind_min) or wind_min < 0:
                raise ValueError("Wind-gated transfer requires a finite non-negative wind_min")
            if "wind_speed" not in self.wh:
                raise ValueError("Wind-gated transfer requires hourly wind_speed")
            wind = pd.to_numeric(self.wh["wind_speed"], errors="coerce")
            if not np.isfinite(wind).all() or (wind < 0).any():
                raise ValueError("Wind-gated transfer requires finite non-negative hourly wind_speed")
            wind_values = wind.to_numpy(dtype=float)
        attached_cohorts = []
        cohorts = []
        cohort_num = 0

        daily_moisture = summarize_sporulation_nights(
            self.wh, rh_threshold=cfg["mp_tmp"]["rh_wet_night"]
        )
        wd = self.wd.merge(
            daily_moisture,
            left_on="datetime",
            right_on="production_date",
            how="left",
        )
        wd[["MP", "TMP"]] = wd[["MP", "TMP"]].fillna(0)
        inventory = []

        for _, day in wd.iterrows():
            date = np.datetime64(day.datetime, "ns")
            new_dose = sporulation_dose_for_day(
                day, min_moist_hours=cfg.get("fixed", {}).get(
                    "min_moist_hours", SPORULATION_MIN_MOIST_HOURS
                )
            )
            dose_available = False

            for i in self.day_to_idxs[date]:
                hr = self.wh.iloc[i]
                if not dose_available and hr.datetime.hour >= SPORULATION_AVAILABILITY_HOUR:
                    if self.external_dose_provider is not None:
                        pressure = self.external_dose_provider.pressure_at(hr.datetime)
                        new_dose *= max(0.0, 1.0 + self.external_dose_provider.beta_sp * pressure)
                    if new_dose > 0:
                        attached_cohorts.append({
                            "production_datetime": hr.datetime,
                            "initial_sporangia_amount": new_dose,
                            "cumulative_mortality": 0.0,
                            "amount": new_dose,
                        })
                    dose_available = True

                if not attached_cohorts:
                    continue

                # At the beginning of each hourly step, transfer the already
                # viable available inventory if the optional gate permits it.
                # Only untransferred inventory ages in the attached state this
                # hour. Detached survival accounts for transferred cohorts from
                # this same timestamp, so each hour belongs to one state only.
                transfer_allowed = not self.use_wind_gate or wind_values[i] >= wind_min
                if transfer_allowed:
                    for attached in attached_cohorts:
                        cohorts.append({
                            "Num": cohort_num,
                            "production_datetime": attached["production_datetime"],
                            "initial_sporangia_amount": attached["initial_sporangia_amount"],
                            "attached_mortality_cumulative": attached["cumulative_mortality"],
                            "dispersal_datetime": hr.datetime,
                            "dispersal_type": transfer_type,
                            "airborne_amount": attached["amount"],
                        })
                        cohort_num += 1
                    attached_cohorts = []
                else:
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
        # External pressure has already adjusted the dose at production time.

    def stage_deposition(self):
        """Expose merged relative leaf availability through the existing API.

        There is no separate airborne residence interval in this abstraction.
        Detached mortality starts at transfer and accumulates downstream; this
        assignment does not reset an already-aged detached cohort. The amount
        is relative availability, not a claim of 100% physical interception.
        """
        cohorts = self.cs.dispersal_cohorts.copy()
        cohorts["deposition_datetime"] = pd.to_datetime(cohorts["dispersal_datetime"])
        cohorts["deposited_amount"] = cohorts["airborne_amount"].astype(float)
        cohorts["detached_mortality_cumulative_at_deposition"] = 0.0
        self.cs.deposition = cohorts.reset_index(drop=True)

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

        # Physical-VPD leaf-moisture proxy shared by release, survival and
        # infection. Source-organ sporulation moisture is a separate rule.
        moisture_of_leaf = leaf_moisture_from_vpd(
            self.VPD, self.cfg["zoospore_alive"]["vpd"]
        )

        results = []
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
            # Only wet hours with an eligible host contribute to infection.
            # A pre-gate zoospore release remains usable only while SUZ lasts.
            host_ready = getattr(self.cs, "bbch_15_datetime", pd.NaT)
            start2 = rz
            if pd.notna(host_ready):
                start2 = max(rz, np.searchsorted(self.time_index, np.datetime64(host_ready)))
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

                thresholds = np.asarray([
                    self.infection_condition_at(self.wh.iloc[start2 + idxs[j]].datetime)
                    for j in range(len(idxs))
                ], dtype=float)
                hit = np.where(WD * TWD >= thresholds)[0]
                if len(hit) > 0:
                    infection_time = self.wh.iloc[start2 + idxs[hit[0]]].datetime
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
