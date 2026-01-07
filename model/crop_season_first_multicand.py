import pandas as pd
from model.first_infection_multi import FirstInfectionMultiCand


class CropSeasonFirstMultiCand:
    """
    Crop season controller for FIRST infection
    (multi-candidate baseline version)

    Key principles:
    ----------------
    1. Generate ALL biologically feasible infection candidates
    2. Apply BBCH 53 constraint (unchanged)
    3. Select the EARLIEST infection as baseline result

    This controller is:
    - Structurally different from the original early-break model
    - Behaviorally equivalent to the original baseline
    """

    def __init__(self, csrb: dict, config: dict):

        # ===== basic info =====
        self.config = config
        self.year = csrb.get("year")
        self.place = csrb.get("Place")

        # ===== BBCH 53 =====
        self.bbch_53_datetime = pd.to_datetime(
            csrb.get("HHCH_53"), errors="coerce"
        )

        # ===== weather (hourly) =====
        self.weather_hourly = pd.DataFrame(csrb["weather_hourly"])
        self.weather_hourly["datetime"] = pd.to_datetime(
            self.weather_hourly["datetime"]
        )
        self.weather_hourly["date"] = (
            self.weather_hourly["datetime"].dt.normalize()
        )

        # Apply BBCH 53 gate on weather (same as original model)
        if pd.notna(self.bbch_53_datetime):
            self.weather_hourly = self.weather_hourly[
                self.weather_hourly["datetime"] >= self.bbch_53_datetime
            ].reset_index(drop=True)

        # ===== weather (daily index only) =====
        self.weather_daily = (
            self.weather_hourly
            .groupby("date", as_index=False)
            .agg(datetime=("date", "first"))
        )

        # ===== shared state (populated by model layer) =====
        self.attached_inventory = None
        self.dispersal_cohorts = None
        self.deposition = None
        self.detached_survival = None

        # ===== infection results =====
        self.primary_chain = None           # ALL infection candidates
        self.primary_chain_unique = None    # selected baseline infection

    # ==================================================
    # MAIN SIMULATION PIPELINE
    # ==================================================
    def simulate(self):
        """
        Run full FIRST infection simulation
        using the multi-candidate infection engine
        """

        model = FirstInfectionMultiCand(self, self.config)

        model.stage_attached_and_dispersal()
        model.stage_deposition()
        model.stage_detached_survival()
        model.stage_primary_chain()

        self._apply_bbch53_gate()
        self._select_earliest_infection()

    # ==================================================
    # BBCH 53 FILTER (on infection results)
    # ==================================================
    def _apply_bbch53_gate(self):
        """
        Keep only infection candidates
        whose oil spot start is after BBCH 53
        """

        if self.primary_chain is None or self.primary_chain.empty:
            return

        if pd.notna(self.bbch_53_datetime):
            self.primary_chain = self.primary_chain[
                self.primary_chain["oil_spot_start"] >= self.bbch_53_datetime
            ].reset_index(drop=True)

    # ==================================================
    # BASELINE SELECTION RULE
    # ==================================================
    def _select_earliest_infection(self):
        """
        Baseline rule:
        select the earliest infection event
        """

        if self.primary_chain is None or self.primary_chain.empty:
            self.primary_chain_unique = self.primary_chain
            return

        df = self.primary_chain.sort_values("infection_datetime")

        # Keep DataFrame shape for compatibility
        self.primary_chain_unique = (
            df.iloc[[0]]
            .reset_index(drop=True)
        )
