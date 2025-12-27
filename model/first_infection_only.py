import pandas as pd
import math


class FirstInfection:
    """
    FIRST infection model
    """

    def __init__(self, cs, config):

        self.cs = cs
        self.cfg = config

        self.wh = cs.weather_hourly.copy()
        self.wd = cs.weather_daily.copy()

        # ===============================
        # Rossi fixed parameters
        # ===============================
        self._NIGHT_TIME = {"start": 20, "end": 8}

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
        current_sporulation_start = None
        cohorts = []
        cohort_num = 0

        h = self.wh.copy()
        night = self._NIGHT_TIME

        h["is_night"] = (
            (h.datetime.dt.hour >= night["start"]) |
            (h.datetime.dt.hour < night["end"])
        )
        h["wet"] = h.relative_humidity >= cfg["mp_tmp"]["rh_wet_night"]

        night_wet = h[h.is_night & h.wet]
        d = (
            night_wet.groupby("date")
            .agg(MP=("wet", "sum"), TMP=("air_temp", "mean"))
            .reset_index()
        )

        wd = self.wd.merge(d, left_on="datetime", right_on="date", how="left")
        wd[["MP", "TMP"]] = wd[["MP", "TMP"]].fillna(0)

        inventory = []

        for _, day in wd.iterrows():
            date = day.datetime.date()

            if day.MP >= sp["MP_min"] and sp["Tmin"] < day.TMP < sp["Tmax"]:

                if attached_pool == 0:
                    nh = h[h.datetime.dt.date == date].copy()
                    nh = nh[nh.is_night & nh.wet]
                    nh["MP_cum"] = nh.wet.cumsum()
                    hit = nh[nh.MP_cum == sp["MP_min"]]
                    current_sporulation_start = (
                        hit.iloc[0].datetime if not hit.empty else day.datetime
                    )

                fT = ((day.TMP - sp["Tmin"]) / (sp["Topt"] - sp["Tmin"])) * \
                     ((sp["Tmax"] - day.TMP) / (sp["Tmax"] - sp["Topt"]))
                fM = 1 - math.exp(-k * day.MP)
                attached_pool += max(0.0, min(1.0, fT * fM))

            for _, hr in self.wh[self.wh.datetime.dt.date == date].iterrows():

                if self._ATTACHED_MORTALITY["start_hour"] <= hr.datetime.hour <= self._ATTACHED_MORTALITY["end_hour"]:
                    VPD = (
                        0.6108 * math.exp((17.27 * hr.air_temp) / (hr.air_temp + 237.3))
                        * (1 - hr.relative_humidity / 100)
                    )
                    lam = max(
                        0.0,
                        self._ATTACHED_MORTALITY["a0"]
                        - self._ATTACHED_MORTALITY["a1"] * VPD
                        + self._ATTACHED_MORTALITY["a2"] * VPD**2
                    ) / 24.0
                    attached_pool *= math.exp(-lam)

                if attached_pool < th:
                    continue

                dispersal_type = None
                if hr.precip_acc_period_adjusted >= disp["rain_min"]:
                    dispersal_type = "rain"
                elif hr.get("wind_speed", 0) >= disp["wind_min"]:
                    dispersal_type = "wind"

                if dispersal_type:
                    cohorts.append({
                        "Num": cohort_num,
                        "sporulation_start": current_sporulation_start,
                        "sporulation_end": hr.datetime,
                        "dispersal_datetime": hr.datetime,
                        "dispersal_type": dispersal_type,
                        "airborne_amount": attached_pool
                    })
                    cohort_num += 1
                    attached_pool = 0.0
                    current_sporulation_start = None

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

        cohorts["deposition_datetime"] = None
        cohorts["deposited_amount"] = None

        for idx, c in cohorts.iterrows():
            if c.dispersal_type == "rain":
                cohorts.loc[idx, "deposition_datetime"] = c.dispersal_datetime
                cohorts.loc[idx, "deposited_amount"] = c.airborne_amount
            elif c.dispersal_type == "wind":
                wh_after = self.wh[self.wh.datetime > c.dispersal_datetime]
                rain = wh_after[
                    wh_after.precip_acc_period_adjusted >= dep["rain_min"]
                ]
                if not rain.empty:
                    cohorts.loc[idx, "deposition_datetime"] = rain.iloc[0].datetime
                    cohorts.loc[idx, "deposited_amount"] = c.airborne_amount

        self.cs.deposition = cohorts.dropna(subset=["deposition_datetime"]).reset_index(drop=True)

    # ======================================================
    # S4. DETACHED SURVIVAL
    # ======================================================
    def stage_detached_survival(self):
        p = self._DETACHED_MORTALITY
        th = self.cfg["thresholds"]["min_spores"]

        cohorts = self.cs.deposition.copy()
        cohorts["detached_alive_end"] = None

        for idx, c in cohorts.iterrows():
            D = c.deposited_amount
            for _, h in self.wh[self.wh.datetime >= c.deposition_datetime].iterrows():
                if 8 <= h.datetime.hour <= 18:
                    VPD = (
                        0.6108 * math.exp((17.27 * h.air_temp) / (h.air_temp + 237.3))
                        * (1 - h.relative_humidity / 100)
                    )
                    lam = max(
                        0.0,
                        p["b0"] - p["b1"] * VPD + p["b2"] * VPD**2
                    ) / 24.0
                    D *= math.exp(-lam)
                    if D < th:
                        cohorts.loc[idx, "detached_alive_end"] = h.datetime
                        break

            if pd.isna(cohorts.loc[idx, "detached_alive_end"]):
                cohorts.loc[idx, "detached_alive_end"] = self.wh.datetime.iloc[-1]

        self.cs.detached_survival = cohorts.reset_index(drop=True)

    # ======================================================
    # S5–S7. PRIMARY CHAIN (SURVIVOR-ONLY)
    # ======================================================
    def stage_primary_chain(self):

        inf = self.cfg["infection"]
        inc = self.cfg["incubation"]
        mol = self.cfg["zoospore_alive"]
        inc_fix = self._INCUBATION_PARAMS

        cohorts = self.cs.detached_survival.copy()
        successful = []

        for _, c in cohorts.iterrows():

            # ---------- zoospore release ----------
            wh = self.wh[
                (self.wh.datetime >= c.deposition_datetime) &
                (self.wh.datetime <= c.detached_alive_end)
            ].copy()

            wh["wet"] = wh.precip_acc_period_adjusted > 0
            wh["grp"] = (wh.wet != wh.wet.shift()).cumsum()

            release_time = None
            for _, gp in wh[wh.wet].groupby("grp"):
                gp = gp.copy()
                gp["WDh"] = range(1, len(gp) + 1)
                gp["TWDh"] = gp.air_temp.expanding().mean()
                WDmin = gp["TWDh"].apply(
                    lambda T: math.exp((-1.022 + 19.634) / T) if T > 0 else math.inf
                )
                if (gp["WDh"] >= WDmin).any():
                    release_time = gp.loc[(gp["WDh"] >= WDmin).idxmax(), "datetime"]
                    break

            if release_time is None:
                continue

            # ---------- zoospore survival ----------
            ceil_release = release_time.ceil("h")
            wh2 = self.wh[self.wh.datetime > ceil_release].copy()

            wh2["VPD"] = wh2.apply(
                lambda x: (1 - x.relative_humidity / 100)
                * 6.11
                * math.exp((17.47 * x.air_temp) / (239 + x.air_temp)),
                axis=1
            )

            wh2["alive"] = wh2.apply(
                lambda x: 1 if (
                    x.precip_acc_period_adjusted >= mol["rain"]
                    or x.VPD <= mol["vpd"]
                ) else 0,
                axis=1
            )

            dead = wh2[wh2.alive == 0]
            alive_end = dead.iloc[0].datetime if not dead.empty else self.wh.datetime.iloc[-1]

            # ---------- infection ----------
            wh3 = self.wh[
                (self.wh.datetime >= release_time) &
                (self.wh.datetime <= alive_end)
            ].copy()

            wh3["wet"] = wh3.precip_acc_period_adjusted > 0
            wh3["grp"] = (wh3.wet != wh3.wet.shift()).cumsum()

            infection_time = None
            for _, gp in wh3[wh3.wet].groupby("grp"):
                gp = gp.copy()
                gp["WDh"] = range(1, len(gp) + 1)
                gp["TWDh"] = gp.air_temp.expanding().mean()
                if (gp["WDh"] * gp["TWDh"] >= inf["infection_condition"]).any():
                    infection_time = gp.loc[
                        (gp["WDh"] * gp["TWDh"] >= inf["infection_condition"]).idxmax(),
                        "datetime"
                    ]
                    break

            if infection_time is None:
                continue

            # ---------- incubation ----------
            wh4 = self.wh[self.wh.datetime >= infection_time].copy()

            wh4["INC1"] = 1 / (
                24 * (inc_fix["a1"] - inc_fix["b1"] * wh4.air_temp + inc_fix["c1"] * wh4.air_temp**2)
            )
            wh4["INC2"] = 1 / (
                24 * (inc_fix["a2"] - inc_fix["b2"] * wh4.air_temp + inc_fix["c2"] * wh4.air_temp**2)
            )

            wh4["INC1_cum"] = wh4.INC1.cumsum()
            wh4["INC2_cum"] = wh4.INC2.cumsum()

            oil_spot_start = wh4.loc[
                wh4.INC1_cum >= inc["spl_end"], "datetime"
            ].min()

            oil_spot_end = wh4.loc[
                wh4.INC2_cum >= inc["spl_end"], "datetime"
            ].min()

            if pd.isna(oil_spot_start) or pd.isna(oil_spot_end):
                continue

            # ===== SUCCESSFUL COHORT =====
            row = c.copy()
            row["zoospore_release_datetime"] = release_time
            row["infection_datetime"] = infection_time
            row["oil_spot_start"] = oil_spot_start
            row["oil_spot_end"] = oil_spot_end

            successful.append(row)

        self.cs.primary_chain = pd.DataFrame(successful).reset_index(drop=True)
