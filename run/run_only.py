import os, sys
import json
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{SRC_DIR}/..")

from model.crop_season_only import CropSeasonFirst
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
    out_dir = f"{SRC_DIR}/../result/first_infection_{crop_season.place}_{crop_season.year}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- stage snapshots ----
    if crop_season.attached_inventory is not None:
        crop_season.attached_inventory.to_excel(
            f"{out_dir}/01_attached_inventory.xlsx", index=False
        )

    if crop_season.dispersal_cohorts is not None:
        crop_season.dispersal_cohorts.to_excel(
            f"{out_dir}/02_dispersal_cohorts.xlsx", index=False
        )

    if crop_season.deposition is not None:
        crop_season.deposition.to_excel(
            f"{out_dir}/03_deposition.xlsx", index=False
        )

    if crop_season.detached_survival is not None:
        crop_season.detached_survival.to_excel(
            f"{out_dir}/04_detached_survival.xlsx", index=False
        )

    if crop_season.primary_chain_unique is not None:
        crop_season.primary_chain_unique.to_excel(
            f"{out_dir}/05_primary_chain_unique.xlsx", index=False
        )


if __name__ == "__main__":
    run_simulation(
        rbf=f"{SRC_DIR}/../data/DA_MPT_2023_grape_rb.json"
    )
