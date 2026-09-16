import pandas as pd
from model.first_infection import FirstInfection


class CropSeasonFirst:
    """
    Crop season controller for FIRST infection
    """

    def __init__(self, csrb: dict, config: dict | None = None):

        if config is None:
            from model.config import load_config
            config = load_config()

        self.config = config
        self.year = csrb.get("year")
        self.place = csrb.get("Place")
        # BBCH15 gates infection accumulation, not inoculum or symptom output.
        # Missing dates retain the explicit ungated sensitivity option.
        bbch15_value = csrb.get("BBCH_15")
        self.bbch_15_datetime = pd.to_datetime(bbch15_value, errors="coerce")
        self.season_start = pd.NaT

        # ===== weather =====
        self.weather_hourly = pd.DataFrame(csrb["weather_hourly"])
        self.weather_hourly["datetime"] = pd.to_datetime(
            self.weather_hourly["datetime"]
        )
        self.weather_hourly["date"] = (
            self.weather_hourly["datetime"].dt.normalize()
        )


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
    def simulate(self, *, use_wind_gate=None):
        """Run M0 (False) or M1 (True); None uses the configuration setting."""

        model = FirstInfection(self, self.config, use_wind_gate=use_wind_gate)
        self.use_wind_gate = model.use_wind_gate

        model.stage_attached_and_dispersal()
        model.stage_deposition()
        model.stage_detached_survival()
        model.stage_primary_chain()

        self.apply_bbch_15_gate()
        self.collapse_equivalent_cohorts()

    # ==================================================
    def apply_bbch_15_gate(self):

        if self.primary_chain is None or self.primary_chain.empty:
            self.primary_chain_gated = self.primary_chain
            return

        df = self.primary_chain.copy()

        # ---------- ensure column exists ----------
        if "infection_effective_bbch15" not in df.columns:
            if pd.notna(self.bbch_15_datetime):
                df["infection_effective_bbch15"] = (
                    df["infection_datetime"] >= self.bbch_15_datetime
                )
            else:
                df["infection_effective_bbch15"] = True

        self.primary_chain = df
        self.primary_chain_gated = df[
            df["infection_effective_bbch15"]
        ].reset_index(drop=True)


    # ==================================================
    def collapse_equivalent_cohorts(self):

        if self.primary_chain_gated is None or self.primary_chain_gated.empty:
            self.primary_chain_unique = self.primary_chain_gated
            return

        df = self.primary_chain_gated.sort_values(
            ["oil_spot_start", "infection_datetime"]
        )
        self.primary_chain_unique = (
            df.drop_duplicates(
                subset=["infection_datetime"],
                keep="first"
            )
            .reset_index(drop=True)
        )
