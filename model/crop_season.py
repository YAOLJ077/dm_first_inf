import os
import pandas as pd
from model.first_infection import FirstInfection


class CropSeasonFirst:
    """
    Crop season controller for FIRST infection
    - Reads weather & phenology
    - Runs pathogen model
    - Applies host phenology gate (HHCH_53)
    """

    def __init__(self, csrb, config):
        self.csrb = csrb
        self.config = config

        self.year = csrb["year"]
        self.place = csrb["Place"]

        # ===== hourly weather =====
        self.wh = pd.DataFrame(csrb["weather_hourly"])
        self.wh["datetime"] = pd.to_datetime(self.wh["datetime"])
        self.wh["date"] = self.wh.datetime.dt.normalize()

        # ===== daily axis =====
        self.wd = self.wh.groupby("date", as_index=False).agg(
            datetime=("date", "first")
        )

        # ===== phenology gate =====
        self.hhch_53 = pd.to_datetime(
            csrb.get("HHCH_53"), errors="coerce"
        )

        # ===== results =====
        self.cohort_results = None

        self.stage_outputs = {
            "S1_attached_inventory": "01_attached_inventory.xlsx",
            "S2_dispersal": "02_dispersal_cohorts.xlsx",
            "S3_deposition": "03_deposition.xlsx",
            "S4_detached_survival": "04_detached_survival.xlsx",
            "S7_full_life": "05_cohort_life_history.xlsx",
            }


    # ======================================================
    def simulate(self, base_output_dir: str = None):

        # ---------- output directory ----------
        if base_output_dir:
            out_dir = os.path.join(
                base_output_dir,
                "first_infection",
                f"{self.place}_{self.year}"
            )
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = None

        # ---------- run model ----------
        model = FirstInfection(
            self.wh,
            self.wd,
            self.config,
            out_dir
        )
        model.run_all()

        # ======================================================
        # GET COHORT RESULTS
        # ======================================================
        if out_dir is None:
            # batch / silent mode → from memory
            df = pd.DataFrame(model.cohorts)
        else:
            cohort_fp = os.path.join(out_dir, "05_cohort_life_history.xlsx")
            if not os.path.exists(cohort_fp):
                print("⚠️ cohort output not found, skip phenology gate")
                return
            df = pd.read_excel(cohort_fp, parse_dates=True)

        # ======================================================
        # APPLY HHCH_53 GATE
        # ======================================================
        if self.hhch_53 is not None and "infection_datetime" in df.columns:
            df["infection_effective"] = (
                df["infection_datetime"] >= self.hhch_53
            )
            df.loc[~df["infection_effective"], [
                "infection_datetime",
                "oil_spot_start",
                "oil_spot_end"
            ]] = pd.NaT
        else:
            df["infection_effective"] = True

        # ======================================================
        # SAVE ONLY IF out_dir EXISTS
        # ======================================================
        if out_dir is not None:
            gated_fp = os.path.join(
                out_dir, "06_cohort_life_history_gated.xlsx"
            )
            df.to_excel(gated_fp, index=False)

        # ---------- store in memory ----------
        self.cohort_results = df


    def export_stage(self, stage_name):
        if stage_name not in self.stage_outputs:
            raise ValueError(f"Unknown stage: {stage_name}")

        if self.base_out_dir is None:
            raise RuntimeError("Stage export is disabled in memory-only mode.")

        fp = os.path.join(
            self.base_out_dir,
            "first_infection",
            f"{self.place}_{self.year}",
            self.stage_outputs[stage_name]
        )
        return pd.read_excel(fp)

