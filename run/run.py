import os,sys
src_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'%s/..'%src_dir)

import json
from model.crop_season import CropSeasonFirst
from model.config import config

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def run_simulation(rbf: str = None):

    with open(rbf, "r") as f:
        rb = json.load(f)

    crop_season = CropSeasonFirst(
        csrb=rb,
        config=config
    )

    crop_season.simulate(
        base_output_dir=f"{SRC_DIR}/../result"
    )


if __name__ == "__main__":
    run_simulation(
        rbf=f"{SRC_DIR}/../data/DA_MPT_2023_grape_rb.json"
    )
