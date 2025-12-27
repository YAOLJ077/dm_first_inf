import pandas as pd
from model.first_infection_only import FirstInfection


class CropSeasonFirst:
    """
    Crop season controller for FIRST infection
    Acts as:
    - simulation controller
    - shared state bag
    - stage snapshot container
    - phenology gate (BBCH=53) handler
    - equivalent cohort collapse (FIRST infection event abstraction)
    """

    def __init__(self, csrb: dict, config: dict):

        # ===== config & meta =====
        self.config = config
        self.year = csrb.get("year")
        self.place = csrb.get("Place")

        # ===== phenology gate (BBCH=53) =====
        self.bbch_53_datetime = pd.to_datetime(csrb.get("HHCH_53"), errors="coerce")

        # ===== weather =====
        self.weather_hourly = pd.DataFrame(csrb["weather_hourly"])
        self.weather_hourly["datetime"] = pd.to_datetime(self.weather_hourly["datetime"])
        self.weather_hourly["date"] = self.weather_hourly.datetime.dt.normalize()

        self.weather_daily = (self.weather_hourly.groupby("date", as_index=False).agg(datetime=("date", "first")))
        # ==================================================
        # SHARED STATE BAG
        # ==================================================

        # ----- S1–S2 -----
        self.attached_inventory = None      # DataFrame
        self.dispersal_cohorts = None       # DataFrame
        # ----- S3 -----
        self.deposition = None              # DataFrame
        # ----- S4 -----
        self.detached_survival = None       # DataFrame
        # ----- S5–S7 (biological completion) -----
        self.primary_chain = None           # DataFrame
        # ----- derived states -----
        self.primary_chain_gated = None     # BBCH=53 后有效
        self.primary_chain_unique = None    # 等价 cohort 压缩后（FIRST infection event）

    # ==================================================
    # SIMULATION PIPELINE
    # ==================================================
    def simulate(self):
        """
        Run full FIRST infection simulation
        (biological process → phenology gate → event abstraction)
        """

        model = FirstInfection(cs=self, config=self.config)

        model.stage_attached_and_dispersal()
        model.stage_deposition()
        model.stage_detached_survival()
        model.stage_primary_chain()

        # ----- apply BBCH=53 gate -----
        self.apply_bbch_53_gate()

        # ----- collapse equivalent cohorts -----
        self.collapse_equivalent_cohorts()

    # ==================================================
    # PHENOLOGY GATE
    # ==================================================
    def apply_bbch_53_gate(self):
        """
        Define 'effective' FIRST infection cohorts:
        oil spot occurring AFTER BBCH=53.
        """

        if self.primary_chain is None or self.primary_chain.empty:
            self.primary_chain_gated = self.primary_chain
            return

        df_all = self.primary_chain.copy()

        if pd.notna(self.bbch_53_datetime):df_all["infection_effective_bbch53"] = (df_all["oil_spot_start"] >= self.bbch_53_datetime)
        else:
            df_all["infection_effective_bbch53"] = True

        self.primary_chain = df_all
        self.primary_chain_gated = (df_all[df_all["infection_effective_bbch53"]].reset_index(drop=True))

    # ==================================================
    # EQUIVALENT COHORT COLLAPSE
    # ==================================================
    def collapse_equivalent_cohorts(self):
        """
        Collapse cohorts that share identical infection_datetime.
        Keep the FIRST cohort for each infection time.
        This represents FIRST infection EVENTS.
        """

        if self.primary_chain_gated is None or self.primary_chain_gated.empty:
            self.primary_chain_unique = self.primary_chain_gated
            return

        df = self.primary_chain_gated.copy()

        # sort to ensure "first cohort" is deterministic
        df = df.sort_values("infection_datetime")

        # drop equivalent cohorts
        df_unique = df.drop_duplicates(subset=["infection_datetime"],keep="first")
        self.primary_chain_unique = df_unique.reset_index(drop=True)

    # ==================================================
    # STAGE SNAPSHOT GETTERS
    # ==================================================
    def get_attached_inventory(self):
        return self.attached_inventory

    def get_dispersal_cohorts(self):
        return self.dispersal_cohorts

    def get_deposition(self):
        return self.deposition

    def get_detached_survival(self):
        return self.detached_survival

    def get_primary_chain(self):
        """
        Biological completion (no phenology gate)
        """
        return self.primary_chain

    def get_primary_chain_gated(self):
        """
        FIRST EFFECTIVE infection cohorts
        (oil spot after BBCH=53)
        """
        return self.primary_chain_gated

    def get_primary_chain_unique(self):
        """
        FIRST infection EVENTS
        (unique infection time points)
        """
        return self.primary_chain_unique
