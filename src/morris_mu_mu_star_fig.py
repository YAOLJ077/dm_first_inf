# -*- coding: utf-8 -*-
"""
Plot mu and mu_star (Morris) as a 1x2 figure for N_TRAJ=256, seed=42.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUMMARY_CSV = os.path.join(
    PROJECT_ROOT, "result", "morris", "morris_first_inf_summary.csv"
)
OUT_DIR = os.path.join(PROJECT_ROOT, "fig")
os.makedirs(OUT_DIR, exist_ok=True)

PARAM_ABBR = {
    "dispersal__rain_min": "DIRM",
    "infection__infection_condition": "IC",
    "zoospore_alive__vpd": "ZVPD",
    "mp_tmp__rh_wet_night": "MTWN",
    "sporulation__Tmax": "STMX",
    "dispersal__wind_min": "DIWM",
    "sporulation__Topt": "STOP",
    "sporulation__MP_min": "SMMI",
    "deposition__rain_min": "DERM",
    "sporulation__Tmin": "STMI",
    "incubation__spl_end": "ISED",
    "zoospore_alive__rain": "ZARA",
    "zoospore_alibe__rain": "ZARA",
}


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_CSV}")

    df_all = pd.read_csv(SUMMARY_CSV)
    df = df_all[
        (df_all["N_TRAJ"] == 256)
        & (df_all["seed"] == 42)
        & (df_all["metric"].str.strip() == "MAE")
    ].copy()

    if df.empty:
        raise ValueError("No rows for N_TRAJ=256, seed=42, metric=MAE.")

    df_std = df_all[
        (df_all["N_TRAJ"] == 256)
        & (df_all["metric"].str.strip() == "MAE")
    ].groupby("Parameter").agg(
        mu_std=("mu", "std"),
        mu_star_std=("mu_star", "std"),
    ).reset_index()

    df = df.merge(df_std, on="Parameter", how="left")
    df["abbr"] = df["Parameter"].map(PARAM_ABBR).fillna(df["Parameter"])
    df_mu_star = df.sort_values("mu_star", ascending=False)
    df_mu = df.sort_values("mu", ascending=False)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
    })

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.2, 6.5),
        sharex=False
    )

    # (a) mu_star
    axes[0].barh(
        df_mu_star["abbr"],
        df_mu_star["mu_star"],
        xerr=df_mu_star["mu_star_std"],
        color="#4C78A8",
        ecolor="#333333",
        capsize=3
    )
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axes[0].set_xlabel("mu_star")
    axes[0].text(
        0.99, 0.98, "(a)",
        transform=axes[0].transAxes,
        ha="right", va="top"
    )

    # (b) mu
    colors = ["#E45756" if v < 0 else "#72B7B2" for v in df_mu["mu"]]
    axes[1].barh(
        df_mu["abbr"],
        df_mu["mu"],
        xerr=df_mu["mu_std"],
        color=colors,
        ecolor="#333333",
        capsize=3
    )
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axes[1].set_xlabel("mu (signed)")
    axes[1].text(
        0.99, 0.98, "(b)",
        transform=axes[1].transAxes,
        ha="right", va="top"
    )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "morris_mu_mu_star_N256_seed42.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
