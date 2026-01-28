import os, sys
import json
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{SRC_DIR}/..")

from model.crop_season import CropSeasonFirst
from model.config import config


def run_simulation(rbf: str):

    # ===============================
    # 1. read input
    # ===============================
    with open(rbf, "r") as f:
        rb = json.load(f)

    # ===============================
    # 2. run simulation
    # ===============================
    crop_season = CropSeasonFirst(
        csrb=rb,
        config=config
    )

    crop_season.simulate()

    # ===============================
    # 3. output (optional, explicit)
    # ===============================
    out_dir = f"{SRC_DIR}/../result/model_result"
    os.makedirs(out_dir, exist_ok=True)

    prefix = f"first_infection_{crop_season.place}_{crop_season.year}_"

    # ---- stage summary (single table) ----
    summary_df = None

    if crop_season.dispersal_cohorts is not None:
        summary_df = crop_season.dispersal_cohorts.copy()

    def _merge_on_num(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        if other is None or other.empty:
            return base
        if base is None or base.empty:
            return other.copy()
        cols = [c for c in other.columns if c != "Num"]
        return base.merge(other[["Num"] + cols], on="Num", how="left")

    summary_df = _merge_on_num(summary_df, crop_season.deposition)
    summary_df = _merge_on_num(summary_df, crop_season.detached_survival)
    summary_df = _merge_on_num(summary_df, crop_season.primary_chain_unique)

    # add attached pool on dispersal date (if available)
    if (
        summary_df is not None
        and not summary_df.empty
        and crop_season.attached_inventory is not None
    ):
        inv = crop_season.attached_inventory.copy()
        inv["date"] = pd.to_datetime(inv["date"]).dt.normalize()
        summary_df["dispersal_date"] = (
            pd.to_datetime(summary_df["dispersal_datetime"]).dt.normalize()
        )
        summary_df = summary_df.merge(
            inv.rename(columns={"attached_pool": "attached_pool_on_dispersal_date"}),
            left_on="dispersal_date",
            right_on="date",
            how="left"
        ).drop(columns=["date"])

    if summary_df is not None and not summary_df.empty:
        summary_df.to_excel(
            f"{out_dir}/{prefix}00_summary_all_stages.xlsx", index=False
        )

    # ---- stage snapshots ----
    if crop_season.attached_inventory is not None:
        crop_season.attached_inventory.to_excel(
            f"{out_dir}/{prefix}01_attached_inventory.xlsx", index=False
        )

    if crop_season.dispersal_cohorts is not None:
        crop_season.dispersal_cohorts.to_excel(
            f"{out_dir}/{prefix}02_dispersal_cohorts.xlsx", index=False
        )

    if crop_season.deposition is not None:
        crop_season.deposition.to_excel(
            f"{out_dir}/{prefix}03_deposition.xlsx", index=False
        )

    if crop_season.detached_survival is not None:
        crop_season.detached_survival.to_excel(
            f"{out_dir}/{prefix}04_detached_survival.xlsx", index=False
        )

    if crop_season.primary_chain_unique is not None:
        crop_season.primary_chain_unique.to_excel(
            f"{out_dir}/{prefix}05_primary_chain_unique.xlsx", index=False
        )


if __name__ == "__main__":
    run_simulation(
        rbf=f"{SRC_DIR}/../data/DA_MPT_2023_grape_rb.json"
    )
