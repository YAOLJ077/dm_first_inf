#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple spore plot.")
    parser.add_argument("--input", default="data/Spore_Overall.xlsx")
    parser.add_argument("--plot", default="")
    parser.add_argument("--year", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["spore"] = pd.to_numeric(df["spore"], errors="coerce")
    df = df.dropna(subset=["date", "spore"])

    if args.plot:
        df = df[df["plot"].astype(str) == args.plot]
    if args.year:
        df = df[df["year"].astype(int) == args.year]

    if df.empty:
        raise SystemExit("No data after filter.")

    fig, ax = plt.subplots(figsize=(8, 4))
    for (plot, year), g in df.groupby(["plot", "year"]):
        g = g.sort_values("date")
        ax.plot(g["date"], g["spore"], marker="o", ms=3, lw=1.2, label=f"{plot}-{year}")

    ax.set_xlabel("Date")
    ax.set_ylabel("Spore count")
    ax.grid(ls="--", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    # fig.savefig("fig/spore_data_simple.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
