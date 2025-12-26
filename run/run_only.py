# -*- coding: utf-8 -*-

import os
import sys
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.crop_season import CropSeasonFirst
from model.config import config


def run_simulation(rb_path: str):
    with open(rb_path, "r") as f:
        rb = json.load(f)

    cs = CropSeasonFirst(csrb=rb, config=config)

    first_window = cs.simulate_first_only()

    if first_window is None:
        print("❌ No effective first infection predicted.")
    else:
        start, end = first_window
        print("✅ First infection window:")
        print(f"   start: {start}")
        print(f"   end  : {end}")


if __name__ == "__main__":
    run_simulation(
        rb_path=os.path.join(
            PROJECT_ROOT, "data", "DA_MPT_2023_grape_rb.json"
        )
    )
