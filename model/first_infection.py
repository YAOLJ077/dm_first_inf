import pandas as pd
import math
import datetime
import os


class FirstInfection:
    """
    First infection model under subtropical, non-dormant conditions
    (Rossi-consistent, cohort-based, with explicit dispersal & deposition)
    """

    def __init__(self, weather_hourly, weather_daily, config, out_dir):
        self.wh = weather_hourly.copy()
        self.wd = weather_daily.copy()
        self.cfg = config
        self.out_dir = out_dir
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)

        # ===== fixed definitions (Rossi, non-tunable) =====
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

        # ===== state variables =====
        self.attached_pool = 0.0
        self.current_sporulation_start = None
        self.cohorts = []
        self.cohort_num = 0

    # ======================================================
    def _export(self, name, df):
        if self.out_dir is None:
            return
        df.to_excel(os.path.join(self.out_dir, name), index=False)


    # ======================================================
    def _compute_MP_TMP(self):
        h = self.wh.copy()
        night = self._NIGHT_TIME
        rh_thr = self.cfg["mp_tmp"]["rh_wet_night"]

        h["is_night"] = (
            (h.datetime.dt.hour >= night["start"]) |
            (h.datetime.dt.hour < night["end"])
        )
        h["wet"] = h.relative_humidity >= rh_thr

        night_wet = h[h.is_night & h.wet]

        d = (
            night_wet
            .groupby("date")
            .agg(MP=("wet", "sum"), TMP=("air_temp", "mean"))
            .reset_index()
        )

        self.wd = self.wd.merge(
            d, left_on="datetime", right_on="date", how="left"
        )
        self.wd[["MP", "TMP"]] = self.wd[["MP", "TMP"]].fillna(0)
        self.wd.drop(columns="date", inplace=True, errors="ignore")

    # ======================================================
    # S1–S2. SPORULATION + DISPERSAL
    # ======================================================
    def stage_attached_inventory_and_dispersal(self):

        if "MP" not in self.wd:
            self._compute_MP_TMP()

        sp = self.cfg["sporulation"]
        disp = self.cfg["dispersal"]
        th = self.cfg["thresholds"]["min_spores"]
        k = 0.2  # Rossi fixed

        mor = self._ATTACHED_MORTALITY

        inventory_records = []

        for _, day in self.wd.iterrows():
            date = day.datetime.date()
            night_marked = False

            # ---------- night sporulation ----------
            if day.MP >= sp["MP_min"] and sp["Tmin"] < day.TMP < sp["Tmax"]:

                if self.attached_pool == 0 and not night_marked:
                    night = self._NIGHT_TIME
                    nh = self.wh[
                        (self.wh.datetime.dt.date == date) &
                        (
                            (self.wh.datetime.dt.hour >= night["start"]) |
                            (self.wh.datetime.dt.hour < night["end"])
                        )
                    ].copy()
                    nh["wet"] = nh.relative_humidity >= self.cfg["mp_tmp"]["rh_wet_night"]
                    nh["MP_cum"] = nh["wet"].cumsum()

                    hit = nh[nh.MP_cum == sp["MP_min"]]
                    self.current_sporulation_start = (
                        hit.iloc[0]["datetime"]
                        if not hit.empty else day.datetime
                    )
                    night_marked = True

                fT = ((day.TMP - sp["Tmin"]) / (sp["Topt"] - sp["Tmin"])) * \
                     ((sp["Tmax"] - day.TMP) / (sp["Tmax"] - sp["Topt"]))
                fM = 1 - math.exp(-k * day.MP)
                self.attached_pool += max(0.0, min(1.0, fT * fM))

            # ---------- hourly mortality + dispersal ----------
            hours = self.wh[self.wh.datetime.dt.date == date]

            for _, h in hours.iterrows():

                # daytime mortality
                if mor["start_hour"] <= h.datetime.hour <= mor["end_hour"]:
                    VPD = (
                        0.6108 * math.exp((17.27 * h.air_temp) / (h.air_temp + 237.3))
                        * (1 - h.relative_humidity / 100)
                    )
                    lam = max(
                        0.0,
                        mor["a0"] - mor["a1"] * VPD + mor["a2"] * VPD**2
                    ) / 24.0
                    self.attached_pool *= math.exp(-lam)

                # ---------- dispersal trigger ----------
                if self.attached_pool < th:
                    continue

                rain = h.precip_acc_period_adjusted
                wind = h.get("wind_speed", 0)

                dispersal_type = None
                if rain >= disp["rain_min"]:
                    dispersal_type = "rain"
                elif wind >= disp["wind_min"]:
                    dispersal_type = "wind"

                if dispersal_type:
                    self.cohorts.append({
                        "Num": self.cohort_num,

                        "sporulation_start": self.current_sporulation_start,
                        "sporulation_end": h.datetime,

                        "dispersal_datetime": h.datetime,
                        "dispersal_type": dispersal_type,

                        "airborne_amount": self.attached_pool,

                        "deposition_datetime": None,
                        "deposited_amount": None,

                        "detached_alive_end": None,

                        "zoospore_release_datetime": None,
                        "infection_datetime": None,
                        "oil_spot_start": None,
                        "oil_spot_end": None,
                    })

                    self.cohort_num += 1
                    self.attached_pool = 0.0
                    self.current_sporulation_start = None

            inventory_records.append({
                "date": day.datetime,
                "attached_pool": self.attached_pool
            })

        self._export("01_attached_inventory.xlsx", pd.DataFrame(inventory_records))
        self._export("02_dispersal_cohorts.xlsx", pd.DataFrame(self.cohorts))

    # ======================================================
    # S3. DEPOSITION (rain-driven)
    # ======================================================
    def stage_deposition(self):

        dep = self.cfg["deposition"]

        for c in self.cohorts:

            if c["dispersal_type"] == "rain":
                c["deposition_datetime"] = c["dispersal_datetime"]
                c["deposited_amount"] = c["airborne_amount"]

            elif c["dispersal_type"] == "wind":
                wh_after = self.wh[self.wh.datetime > c["dispersal_datetime"]]
                rain = wh_after[
                    wh_after.precip_acc_period_adjusted >= dep["rain_min"]
                ]
                if not rain.empty:
                    c["deposition_datetime"] = rain.iloc[0]["datetime"]
                    c["deposited_amount"] = c["airborne_amount"]

        self._export("03_deposition.xlsx", pd.DataFrame(self.cohorts))

    # ======================================================
    # S4. DETACHED SURVIVAL
    # ======================================================
    def stage_detached_survival(self):

        p = self._DETACHED_MORTALITY
        th = self.cfg["thresholds"]["min_spores"]

        for c in self.cohorts:

            if c["deposition_datetime"] is None:
                continue

            D = c["deposited_amount"]
            start = c["dispersal_datetime"]

            for _, h in self.wh[self.wh.datetime >= start].iterrows():
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
                        c["detached_alive_end"] = h.datetime
                        break

            if c["detached_alive_end"] is None:
                c["detached_alive_end"] = self.wh.datetime.iloc[-1]

        self._export("04_detached_survival.xlsx", pd.DataFrame(self.cohorts))

    # ======================================================
    # S5–S7. PRIMARY INFECTION CHAIN (Rossi)
    # ======================================================
    def stage_primary_chain(self):

        inf = self.cfg["infection"]
        inc = self.cfg["incubation"]
        inc_fix = self._INCUBATION_PARAMS
        mol = self.cfg["zoospore_alive"]   # ← 新增：Rossi 原始参数

        for c in self.cohorts:
            # ==================================================
            # S5. ZOOSPORE RELEASE (unchanged, Rossi Eq. 7 前半)
            # ==================================================

            # ---------- gate: must have entered detached stage ----------
            if c.get("dispersal_datetime") is None:
                continue

            # ---------- gate: deposition must have occurred ----------
            if c.get("deposition_datetime") is None:
                continue

            # ---------- gate: detached must still be alive ----------
            if c.get("detached_alive_end") is None:
                continue

            wh = self.wh[
                (self.wh.datetime >= c["deposition_datetime"]) &
                (self.wh.datetime <= c["detached_alive_end"])
            ].copy()
            wh["wet"] = wh["precip_acc_period_adjusted"] > 0
            wh["grp"] = (wh["wet"] != wh["wet"].shift()).cumsum()

            zoospore_release_time = None

            for _, gp in wh[wh.wet].groupby("grp"):
                gp = gp.copy()
                gp["WDh"] = range(1, len(gp) + 1)
                gp["TWDh"] = gp.air_temp.expanding().mean()

                # Rossi (2008) Eq. 7 release condition
                WDmin = gp["TWDh"].apply(lambda T: math.exp((-1.022 + 19.634) / T) if T > 0 else math.inf)

                if (gp["WDh"] >= WDmin).any():
                    zoospore_release_time = gp.loc[(gp["WDh"] >= WDmin).idxmax(), "datetime"]
                    c["zoospore_release_datetime"] = zoospore_release_time
                    break

            # 若未释放游动孢子 → 队列终止
            if zoospore_release_time is None:
                continue
            # ==================================================
            # S5b. ZOOSPORE SURVIVAL (SUZ) — Rossi 原始定义
            # ==================================================
            ceil_release = zoospore_release_time.ceil("h")

            wh_suz = self.wh[self.wh.datetime > ceil_release].copy()

            # ---- VPD calculation (Rossi original) ----
            wh_suz["VPD"] = wh_suz.apply(lambda x: (1 - x["relative_humidity"] / 100) * 6.11 * math.exp((17.47 * x["air_temp"]) / (239 + x["air_temp"])),axis=1)

            # ---- moisture_of_leaf (Rain OR low VPD) ----
            wh_suz["zoospore_alive"] = wh_suz.apply(lambda x: 1
                if (x["precip_acc_period_adjusted"] >= mol["rain"]
                    or x["VPD"] <= mol["vpd"])
                else 0,
                axis=1
            )

            dry_hours = wh_suz[wh_suz["zoospore_alive"] == 0]

            if not dry_hours.empty:
                zoospore_alive_end = max(dry_hours.iloc[0]["datetime"], ceil_release)
            else:
                zoospore_alive_end = self.wh.datetime.iloc[-1]

            # ==================================================
            # S6. INFECTION (Rossi primary, unchanged)
            # ==================================================
            wh2 = self.wh[(self.wh.datetime >= zoospore_release_time) & (self.wh.datetime <= zoospore_alive_end)].copy()

            wh2["wet"] = wh2["precip_acc_period_adjusted"] > 0
            wh2["grp"] = (wh2["wet"] != wh2["wet"].shift()).cumsum()

            infection_time = None

            for _, gp in wh2[wh2.wet].groupby("grp"):
                gp = gp.copy()
                gp["WDh"] = range(1, len(gp) + 1)
                gp["TWDh"] = gp.air_temp.expanding().mean()

                if (gp["WDh"] * gp["TWDh"] >= inf["infection_condition"]).any():
                    infection_time = gp.loc[
                        (gp["WDh"] * gp["TWDh"] >= inf["infection_condition"]).idxmax(),
                        "datetime"
                    ]
                    c["infection_datetime"] = infection_time
                    break

            if infection_time is None:
                continue

            # ==================================================
            # S7. SYMPTOM (INCUBATION) — Rossi primary
            # ==================================================
            wh3 = self.wh[self.wh.datetime >= infection_time].copy()

            wh3["INC1"] = 1 / (
                24 * (inc_fix["a1"] - inc_fix["b1"] * wh3.air_temp + inc_fix["c1"] * wh3.air_temp**2))

            wh3["INC2"] = 1 / (24 * ( inc_fix["a2"] - inc_fix["b2"] * wh3.air_temp + inc_fix["c2"] * wh3.air_temp**2))

            wh3["INC1_cum"] = wh3.INC1.cumsum()
            wh3["INC2_cum"] = wh3.INC2.cumsum()

            c["oil_spot_start"] = wh3.loc[wh3.INC1_cum >= inc["spl_end"], "datetime"].min()

            c["oil_spot_end"] = wh3.loc[wh3.INC2_cum >= inc["spl_end"], "datetime"].min()

        # ======================================================
        self._export("05_cohort_life_history.xlsx", pd.DataFrame(self.cohorts))

    # ======================================================
    def run_all(self):
        self.stage_attached_inventory_and_dispersal()
        self.stage_deposition()
        self.stage_detached_survival()
        self.stage_primary_chain()
