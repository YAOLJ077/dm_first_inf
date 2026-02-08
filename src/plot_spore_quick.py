#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick plot for spore observations.")
    parser.add_argument("--input", default="data/Spore_Overall.xlsx")
    parser.add_argument("--plot", default="", help="Filter by plot, e.g. NN")
    parser.add_argument("--year", type=int, default=0, help="Filter by year, e.g. 2024")
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if args.plot:
        df = df[df["plot"].astype(str) == args.plot]
    if args.year:
        df = df[df["year"].astype(int) == args.year]

    obs = df.dropna(subset=["spore"]).copy()
    if obs.empty:
        raise SystemExit("No non-null spore values found for current filters.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (plot, year), g in obs.groupby(["plot", "year"]):
        g = g.sort_values("date")
        ax.plot(g["date"], g["spore"], marker="o", linewidth=1.4, markersize=3, label=f"{plot}-{year}")

    ax.set_xlabel("Date")
    ax.set_ylabel("Spore")
    ax.set_title("Spore observations")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()

    # out = Path("fig/spore_quick.png")
    # out.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
