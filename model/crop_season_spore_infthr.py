from __future__ import annotations
import pandas as pd
from model.crop_season import CropSeasonFirst
from model.first_infection_spore_infthr import FirstInfectionSpore
from model.observed_background_dose import ObservedBackgroundDose


class CropSeasonFirstSporeInfThr(CropSeasonFirst):
    """FSIM-S controller; inherits the complete FSIM season workflow."""

    def __init__(self, csrb: dict, config: dict, spore_day: pd.Series | None = None):
        super().__init__(csrb, config)
        self.spore_day = spore_day

    def simulate(self, *, debug=False, use_wind_gate=None):
        provider = ObservedBackgroundDose(
            self.spore_day,
            beta_sp=self.config.get("spore", {}).get("beta_sp", 0.0),
        ) if self.spore_day is not None else None
        model = FirstInfectionSpore(self, self.config,
                                    external_dose_provider=provider,
                                    use_wind_gate=use_wind_gate)
        self.use_wind_gate = model.use_wind_gate
        model.stage_attached_and_dispersal()
        model.stage_deposition()
        model.stage_detached_survival()
        model.stage_primary_chain()
        self.apply_bbch_15_gate()
        self.collapse_equivalent_cohorts()
        if debug:
            print(f"[DEBUG] dispersal={len(self.dispersal_cohorts) if self.dispersal_cohorts is not None else 0} "
                  f"deposition={len(self.deposition) if self.deposition is not None else 0} "
                  f"primary={len(self.primary_chain_unique) if self.primary_chain_unique is not None else 0}")
