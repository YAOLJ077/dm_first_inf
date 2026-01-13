import pandas as pd
from model.first_infection import FirstInfection


class CropSeasonFirst:
    """
    Crop season controller for FIRST infection
    """

    def __init__(self, csrb: dict, config: dict):

        self.config = config
        self.year = csrb.get("year")
        self.place = csrb.get("Place")

        self.bbch_53_datetime = pd.to_datetime(
            csrb.get("HHCH_53"), errors="coerce"
        )
        # Allow incubation period before BBCH53 by shifting the gate earlier.
        self.bbch_53_gate_start = (
            self.bbch_53_datetime - pd.Timedelta(days=7)
            if pd.notna(self.bbch_53_datetime)
            else pd.NaT
        )

        # ===== weather =====
        self.weather_hourly = pd.DataFrame(csrb["weather_hourly"])
        self.weather_hourly["datetime"] = pd.to_datetime(
            self.weather_hourly["datetime"]
        )
        self.weather_hourly["date"] = (
            self.weather_hourly["datetime"].dt.normalize()
        )


        if pd.notna(self.bbch_53_gate_start):
            self.weather_hourly = self.weather_hourly[
                self.weather_hourly["datetime"] >= self.bbch_53_gate_start
            ].reset_index(drop=True)

        self.weather_daily = (
            self.weather_hourly
            .groupby("date", as_index=False)
            .agg(datetime=("date", "first"))
        )

        # ===== shared state =====
        self.attached_inventory = None
        self.dispersal_cohorts = None
        self.deposition = None
        self.detached_survival = None
        self.primary_chain = None
        self.primary_chain_gated = None
        self.primary_chain_unique = None

    # ==================================================
    def simulate(self, use_obs_gate=False, obs_gate=None):

        model = FirstInfection(
            self,
            self.config,
            obs_gate=obs_gate if use_obs_gate else None
        )

        model.stage_attached_and_dispersal()
        model.stage_deposition()
        model.stage_detached_survival()
        model.stage_primary_chain()

        self.apply_bbch_53_gate()
        self.collapse_equivalent_cohorts()

    # ==================================================
    def apply_bbch_53_gate(self):

        if self.primary_chain is None or self.primary_chain.empty:
            self.primary_chain_gated = self.primary_chain
            return

        df = self.primary_chain.copy()

        # ---------- ensure column exists ----------
        if "infection_effective_bbch53" not in df.columns:
            if pd.notna(self.bbch_53_gate_start):
                df["infection_effective_bbch53"] = (
                    df["oil_spot_start"] >= self.bbch_53_gate_start
                )
            else:
                df["infection_effective_bbch53"] = True

        self.primary_chain = df
        self.primary_chain_gated = df[
            df["infection_effective_bbch53"]
        ].reset_index(drop=True)


    # ==================================================
    def collapse_equivalent_cohorts(self):

        if self.primary_chain_gated is None or self.primary_chain_gated.empty:
            self.primary_chain_unique = self.primary_chain_gated
            return

        df = self.primary_chain_gated.sort_values("infection_datetime")
        self.primary_chain_unique = (
            df.drop_duplicates(
                subset=["infection_datetime"],
                keep="first"
            )
            .reset_index(drop=True)
        )
