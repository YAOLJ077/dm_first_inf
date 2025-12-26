import pandas as pd
import math


class FirstInfection:
    """
    FIRST infection biological model
    - Strict biological process
    - State propagation via CropSeasonFirst (cs)
    """

    def __init__(self, cs, config):
        self.cs = cs
        self.config = config

        self.wh = cs.weather_hourly
        self.wd = cs.weather_daily

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

    # ======================================================
    # S1–S2. SPORULATION + DISPERSAL
    # ======================================================
    def sporulation_and_dispersal(self):

        sp = self.config["sporulation"]
        disp = self.config["dispersal"]
        th = self.config["thresholds"]["min_spores"]
        k = 0.2

        attached_pool = 0.0
        cohorts = []
        cohort_num = 0

        for _, day in self.wd.iterrows():

            # ---- night sporulation ----
            if day.MP >= sp["MP_min"] and sp["Tmin"] < day.TMP < sp["Tmax"]:
                fT = ((day.TMP - sp["Tmin"]) / (sp["Topt"] - sp["Tmin"])) * \
                     ((sp["Tmax"] - day.TMP) / (sp["Tmax"] - sp["Topt"]))
                fM = 1 - math.exp(-k * day.MP)
                attached_pool += max(0.0, min(1.0, fT * fM))

            # ---- hourly dispersal ----
            hours = self.wh[self.wh.datetime.dt.date == day.datetime.date()]

            for _, h in hours.iterrows():

                if attached_pool < th:
                    continue

                rain = h.precip_acc_period_adjusted
                wind = h.get("wind_speed", 0)

                dispersal_type = None
                if rain >= disp["rain_min"]:
                    dispersal_type = "rain"
                elif wind >= disp["wind_min"]:
                    dispersal_type = "wind"

                if dispersal_type:
                    cohorts.append({
                        "Num": cohort_num,
                        "sporulation_end": h.datetime,
                        "dispersal_datetime": h.datetime,
                        "dispersal_type": dispersal_type,
                        "airborne_amount": attached_pool,

                        # downstream placeholders
                        "deposition_datetime": None,
                        "deposited_amount": None,
                        "detached_alive_end": None,
                        "zoospore_release_datetime": None,
                        "infection_datetime": None,
                        "oil_spot_start": None,
                        "oil_spot_end": None,
                    })

                    cohort_num += 1
                    attached_pool = 0.0

        self.cs.sporangia_cohorts = pd.DataFrame(cohorts)

    # ======================================================
    # S3. DEPOSITION
    # ======================================================
    def deposition(self):

        dep = self.config["deposition"]
        df = self.cs.sporangia_cohorts

        for i, c in df.iterrows():

            if c.dispersal_type == "rain":
                df.at[i, "deposition_datetime"] = c.dispersal_datetime
                df.at[i, "deposited_amount"] = c.airborne_amount

            elif c.dispersal_type == "wind":
                rain = self.wh[
                    (self.wh.datetime > c.dispersal_datetime) &
                    (self.wh.precip_acc_period_adjusted >= dep["rain_min"])
                ]
                if not rain.empty:
                    df.at[i, "deposition_datetime"] = rain.iloc[0].datetime
                    df.at[i, "deposited_amount"] = c.airborne_amount

        self.cs.sporangia_cohorts = df

    # ======================================================
    # S4–S7. DETACHED → RELEASE → INFECTION → INCUBATION
    # ======================================================
    def primary_chain(self):

        inf = self.config["infection"]
        inc = self.config["incubation"]
        mol = self.config["zoospore_alive"]
        p = self._DETACHED_MORTALITY
        inc_fix = self._INCUBATION_PARAMS
        th = self.config["thresholds"]["min_spores"]

        df = self.cs.sporangia_cohorts.copy()

        for i, c in df.iterrows():

            if pd.isna(c.deposition_datetime):
                continue

            # ---------- detached survival ----------
            D = c.deposited_amount
            start = c.dispersal_datetime

            for _, h in self.wh[self.wh.datetime >= start].iterrows():
                if 8 <= h.datetime.hour <= 18:
                    VPD = (
                        0.6108 * math.exp((17.27 * h.air_temp) /
                        (h.air_temp + 237.3)) *
                        (1 - h.relative_humidity / 100)
                    )
                    lam = max(0.0, p["b0"] - p["b1"] * VPD + p["b2"] * VPD**2) / 24
                    D *= math.exp(-lam)
                    if D < th:
                        df.at[i, "detached_alive_end"] = h.datetime
                        break

            if pd.isna(df.at[i, "detached_alive_end"]):
                df.at[i, "detached_alive_end"] = self.wh.datetime.iloc[-1]

            # ---------- zoospore release ----------
            wh = self.wh[
                (self.wh.datetime >= c.deposition_datetime) &
                (self.wh.datetime <= df.at[i, "detached_alive_end"])
            ].copy()

            wh["wet"] = wh.precip_acc_period_adjusted > 0
            wh["grp"] = (wh.wet != wh.wet.shift()).cumsum()

            release_time = None
            for _, gp in wh[wh.wet].groupby("grp"):
                gp["WDh"] = range(1, len(gp) + 1)
                gp["TWDh"] = gp.air_temp.expanding().mean()
                WDmin = gp.TWDh.apply(
                    lambda T: math.exp((-1.022 + 19.634) / T) if T > 0 else math.inf
                )
                if (gp.WDh >= WDmin).any():
                    release_time = gp.loc[(gp.WDh >= WDmin).idxmax(), "datetime"]
                    break

            if release_time is None:
                continue

            df.at[i, "zoospore_release_datetime"] = release_time

            # ---------- infection ----------
            wh2 = self.wh[self.wh.datetime >= release_time].copy()
            wh2["wet"] = wh2.precip_acc_period_adjusted > 0
            wh2["grp"] = (wh2.wet != wh2.wet.shift()).cumsum()

            for _, gp in wh2[wh2.wet].groupby("grp"):
                gp["WDh"] = range(1, len(gp) + 1)
                gp["TWDh"] = gp.air_temp.expanding().mean()
                if (gp.WDh * gp.TWDh >= inf["infection_condition"]).any():
                    inf_time = gp.loc[
                        (gp.WDh * gp.TWDh >= inf["infection_condition"]).idxmax(),
                        "datetime"
                    ]
                    df.at[i, "infection_datetime"] = inf_time
                    break

            if pd.isna(df.at[i, "infection_datetime"]):
                continue

            # ---------- incubation ----------
            wh3 = self.wh[self.wh.datetime >= df.at[i, "infection_datetime"]].copy()

            wh3["INC1"] = 1 / (
                24 * (inc_fix["a1"] - inc_fix["b1"] * wh3.air_temp +
                      inc_fix["c1"] * wh3.air_temp**2)
            )
            wh3["INC2"] = 1 / (
                24 * (inc_fix["a2"] - inc_fix["b2"] * wh3.air_temp +
                      inc_fix["c2"] * wh3.air_temp**2)
            )

            wh3["INC1_cum"] = wh3.INC1.cumsum()
            wh3["INC2_cum"] = wh3.INC2.cumsum()

            df.at[i, "oil_spot_start"] = wh3.loc[
                wh3.INC1_cum >= inc["spl_end"], "datetime"
            ].min()

            df.at[i, "oil_spot_end"] = wh3.loc[
                wh3.INC2_cum >= inc["spl_end"], "datetime"
            ].min()

        self.cs.oil_spot_on_leaf = df
