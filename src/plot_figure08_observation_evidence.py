"""Draw candidate Figure 8 panels from the clarified observation tables."""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OBS_PATH = ROOT / "data" / "grape_experiment_CK_site_years.xlsx"
ENDPOINT_PATH = ROOT / "data" / "site_year_onset_endpoints.csv"
OUTPUT_A = ROOT / "fig" / "Figure_08A_observation_timeline.png"
OUTPUT_B = ROOT / "fig" / "Figure_08B_lead_intervals.png"
OUTPUT_CORE = ROOT / "fig" / "Figure_08_core_timeline.png"

SITE_META = {
    "Guilin": {"label": "Guilin", "color": "#D55E00", "order": 0},
    "Duan": {"label": "Hechi", "color": "#0072B2", "order": 1},
    "Mingyang": {"label": "Nanning", "color": "#009E73", "order": 2},
}

RAW_TICKS = np.array([0, 1, 5, 10, 50, 100, 200])
TRANSFORMED_TICKS = np.log1p(RAW_TICKS)
ONSET_COLOR = "#C62828"
ZERO_COLOR = "#6F6F6F"
GRID_COLOR = "#D9D9D9"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.read_excel(OBS_PATH, sheet_name="CK_Data")
    endpoints = pd.read_csv(ENDPOINT_PATH, encoding="utf-8-sig")

    observations["Date"] = pd.to_datetime(observations["Date"])
    endpoints["L"] = pd.to_datetime(endpoints["L"])
    endpoints["U"] = pd.to_datetime(endpoints["U"])

    keys = endpoints[["Place", "Year"]].drop_duplicates()
    observations = observations.merge(keys, on=["Place", "Year"], how="inner")

    summaries = []
    for endpoint in endpoints.itertuples(index=False):
        group = observations.loc[
            (observations["Place"] == endpoint.Place)
            & (observations["Year"] == endpoint.Year)
        ]
        sampled = group.dropna(subset=["Spore"])
        positive = sampled.loc[sampled["Spore"] > 0]
        if positive.empty:
            raise ValueError(
                f"No spore-positive observation for {endpoint.Place} {endpoint.Year}."
            )

        summaries.append(
            {
                "Place": endpoint.Place,
                "Year": endpoint.Year,
                "first_spore_date": positive["Date"].min(),
                "L": endpoint.L,
                "U": endpoint.U,
                "lead_to_L": (endpoint.L - positive["Date"].min()).days,
                "lead_to_U": (endpoint.U - positive["Date"].min()).days,
            }
        )

    summary = pd.DataFrame(summaries)
    summary["site_order"] = summary["Place"].map(
        lambda place: SITE_META[place]["order"]
    )
    summary = summary.sort_values(["site_order", "Year"]).reset_index(drop=True)
    return observations, endpoints, summary


def panel_a(
    observations: pd.DataFrame,
    endpoints: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        len(summary),
        1,
        figsize=(7.1, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    max_count = pd.to_numeric(observations["Spore"], errors="coerce").max()
    y_max = np.log1p(max(200, max_count)) * 1.08

    for index, (ax, case) in enumerate(zip(axes, summary.itertuples(index=False))):
        meta = SITE_META[case.Place]
        group = observations.loc[
            (observations["Place"] == case.Place)
            & (observations["Year"] == case.Year)
        ].copy()
        sampled = group.dropna(subset=["Spore"]).copy()
        sampled["DOY"] = sampled["Date"].dt.dayofyear
        positive = sampled.loc[sampled["Spore"] > 0]
        zero = sampled.loc[sampled["Spore"] == 0]

        endpoint = endpoints.loc[
            (endpoints["Place"] == case.Place)
            & (endpoints["Year"] == case.Year)
        ].iloc[0]
        l_doy = int(endpoint["L_DOY"])
        u_doy = int(endpoint["U_DOY"])

        ax.axvspan(l_doy, u_doy, color=ONSET_COLOR, alpha=0.16, linewidth=0)
        ax.axvline(l_doy, color=ONSET_COLOR, linewidth=0.9, linestyle=(0, (3, 2)))
        ax.axvline(u_doy, color=ONSET_COLOR, linewidth=1.2)

        heights = np.log1p(positive["Spore"].to_numpy(dtype=float))
        ax.vlines(
            positive["DOY"],
            0,
            heights,
            color=meta["color"],
            linewidth=1.4,
            alpha=0.9,
        )
        ax.scatter(
            positive["DOY"],
            heights,
            s=15,
            color=meta["color"],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            zero["DOY"],
            np.zeros(len(zero)),
            s=17,
            facecolor="white",
            edgecolor=ZERO_COLOR,
            linewidth=0.8,
            zorder=3,
        )

        first_positive = positive.loc[positive["Date"].idxmin()]
        ax.scatter(
            first_positive["DOY"],
            np.log1p(float(first_positive["Spore"])),
            marker="D",
            s=38,
            facecolor="black",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

        disease_obs = group.dropna(subset=["disease"]).copy()
        disease_obs["DOY"] = disease_obs["Date"].dt.dayofyear
        disease_negative = disease_obs.loc[disease_obs["disease"] == 0]
        ax.scatter(
            disease_negative["DOY"],
            np.full(len(disease_negative), -0.48),
            s=10,
            color="#A7A7A7",
            zorder=3,
            clip_on=False,
        )
        ax.scatter(
            [u_doy],
            [-0.48],
            marker="s",
            s=24,
            color=ONSET_COLOR,
            zorder=4,
            clip_on=False,
        )

        ax.text(
            0.01,
            0.88,
            f"{meta['label']} {case.Year}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=meta["color"],
        )
        ax.text(
            0.99,
            0.88,
            f"{case.lead_to_L}–{case.lead_to_U} d",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color=ONSET_COLOR,
        )

        ax.set_ylim(-0.65, y_max)
        ax.set_yticks(TRANSFORMED_TICKS)
        ax.set_yticklabels([str(value) for value in RAW_TICKS])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
        if index != len(summary) - 1:
            ax.spines["bottom"].set_color("#BFBFBF")

    axes[-1].set_xlim(55, 170)
    axes[-1].set_xticks([60, 80, 100, 120, 140, 160])
    axes[-1].set_xlabel("Day of year (DOY)")
    fig.supylabel("Airborne sporangia (count per sample; log1p scale)")

    positive_handle = mlines.Line2D(
        [], [], color="#444444", marker="o", markersize=4, linewidth=1.3,
        label="Spore-positive sample"
    )
    zero_handle = mlines.Line2D(
        [], [], color=ZERO_COLOR, marker="o", markerfacecolor="white",
        markersize=4, linewidth=0, label="Zero-count sample"
    )
    first_handle = mlines.Line2D(
        [], [], color="black", marker="D", markersize=5, linewidth=0,
        label="First spore-positive sample"
    )
    negative_handle = mlines.Line2D(
        [], [], color="#A7A7A7", marker="o", markersize=3, linewidth=0,
        label="Disease-negative assessment"
    )
    positive_disease_handle = mlines.Line2D(
        [], [], color=ONSET_COLOR, marker="s", markersize=4, linewidth=0,
        label="First disease-positive assessment"
    )
    onset_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor=ONSET_COLOR, alpha=0.16,
        label="Observed disease-onset interval (L, U]"
    )
    fig.legend(
        handles=[
            positive_handle,
            zero_handle,
            first_handle,
            negative_handle,
            positive_disease_handle,
            onset_handle,
        ],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.5,
    )

    OUTPUT_A.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_A, dpi=600, facecolor="white")
    plt.close(fig)


def panel_b(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.8), constrained_layout=True)
    y_positions = np.arange(len(summary))[::-1]

    for y, case in zip(y_positions, summary.itertuples(index=False)):
        meta = SITE_META[case.Place]
        ax.hlines(y, 0, case.lead_to_U, color="#B9B9B9", linewidth=1.2)
        ax.scatter(
            0,
            y,
            marker="D",
            s=34,
            facecolor="black",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.hlines(
            y,
            case.lead_to_L,
            case.lead_to_U,
            color=ONSET_COLOR,
            linewidth=7,
            alpha=0.72,
            zorder=2,
        )
        ax.scatter(
            case.lead_to_L,
            y,
            s=42,
            facecolor="white",
            edgecolor=ONSET_COLOR,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            case.lead_to_U,
            y,
            s=42,
            facecolor=ONSET_COLOR,
            edgecolor=ONSET_COLOR,
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            case.lead_to_U + 1.5,
            y,
            f"{case.lead_to_L}–{case.lead_to_U} d",
            ha="left",
            va="center",
            fontsize=9.5,
            color=meta["color"],
        )

    labels = [
        f"{SITE_META[row.Place]['label']} {row.Year}"
        for row in summary.itertuples(index=False)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    for tick, row in zip(ax.get_yticklabels(), summary.itertuples(index=False)):
        tick.set_color(SITE_META[row.Place]["color"])
        tick.set_fontweight("bold")

    ax.set_xlim(-2, max(80, summary["lead_to_U"].max() + 10))
    ax.set_xticks(np.arange(0, 81, 10))
    ax.set_xlabel("Days after the first spore-positive sample")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="out", length=3, width=0.8)

    first_handle = mlines.Line2D(
        [], [], color="black", marker="D", markersize=5, linewidth=0,
        label="First spore-positive sample"
    )
    l_handle = mlines.Line2D(
        [], [], color=ONSET_COLOR, marker="o", markerfacecolor="white",
        markersize=5, linewidth=0, label="L: last disease-negative assessment"
    )
    u_handle = mlines.Line2D(
        [], [], color=ONSET_COLOR, marker="o", markerfacecolor=ONSET_COLOR,
        markersize=5, linewidth=0, label="U: first disease-positive assessment"
    )
    ax.legend(
        handles=[first_handle, l_handle, u_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    OUTPUT_B.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_B, dpi=600, facecolor="white")
    plt.close(fig)


def season_day(value: pd.Timestamp) -> int:
    """Map month and day to a common leap-year calendar axis."""
    return pd.Timestamp(2000, value.month, value.day).dayofyear


def panel_core(
    observations: pd.DataFrame,
    endpoints: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    y_positions = np.arange(len(summary))[::-1]
    max_count = max(1.0, pd.to_numeric(observations["Spore"], errors="coerce").max())

    for y, case in zip(y_positions, summary.itertuples(index=False)):
        meta = SITE_META[case.Place]
        endpoint = endpoints.loc[
            (endpoints["Place"] == case.Place)
            & (endpoints["Year"] == case.Year)
        ].iloc[0]
        l_day = season_day(endpoint["L"])
        u_day = season_day(endpoint["U"])
        first_day = season_day(case.first_spore_date)

        group = observations.loc[
            (observations["Place"] == case.Place)
            & (observations["Year"] == case.Year)
            & observations["Spore"].notna()
            & (observations["Spore"] > 0)
            & (observations["Date"] <= endpoint["U"])
        ].copy()
        group["season_day"] = group["Date"].map(season_day)
        group["height"] = 0.07 + 0.30 * (
            np.log1p(group["Spore"].astype(float)) / np.log1p(max_count)
        )

        ax.hlines(y, first_day, l_day, color="#BEBEBE", linewidth=1.2, zorder=1)
        ax.vlines(
            group["season_day"],
            y,
            y + group["height"],
            color=meta["color"],
            linewidth=1.35,
            alpha=0.9,
            zorder=2,
        )
        ax.scatter(
            group["season_day"],
            y + group["height"],
            s=18,
            facecolor=meta["color"],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

        ax.scatter(
            first_day,
            y,
            marker="D",
            s=38,
            facecolor="black",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.hlines(
            y,
            l_day,
            u_day,
            color=ONSET_COLOR,
            linewidth=7,
            alpha=0.72,
            zorder=3,
        )
        ax.scatter(
            l_day,
            y,
            s=40,
            facecolor="white",
            edgecolor=ONSET_COLOR,
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            u_day,
            y,
            s=40,
            facecolor=ONSET_COLOR,
            edgecolor=ONSET_COLOR,
            linewidth=1.0,
            zorder=4,
        )
        ax.text(
            u_day + 2.0,
            y,
            f"{case.lead_to_L}–{case.lead_to_U} d",
            ha="left",
            va="center",
            fontsize=9.5,
            color=meta["color"],
        )

    labels = [
        f"{SITE_META[row.Place]['label']} {row.Year}"
        for row in summary.itertuples(index=False)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    for tick, row in zip(ax.get_yticklabels(), summary.itertuples(index=False)):
        tick.set_color(SITE_META[row.Place]["color"])
        tick.set_fontweight("bold")

    tick_dates = pd.to_datetime(
        [
            "2000-03-01",
            "2000-03-15",
            "2000-04-01",
            "2000-04-15",
            "2000-05-01",
            "2000-05-15",
            "2000-06-01",
            "2000-06-15",
        ]
    )
    ax.set_xticks([date.dayofyear for date in tick_dates])
    ax.set_xticklabels([date.strftime("%b %d") for date in tick_dates])
    ax.set_xlim(season_day(pd.Timestamp("2000-03-01")) - 2, season_day(pd.Timestamp("2000-06-15")) + 4)
    ax.set_ylim(-0.45, len(summary) - 0.35)
    ax.set_xlabel("Calendar date")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="out", length=3, width=0.8)

    spore_handle = mlines.Line2D(
        [], [], color="#555555", marker="o", markersize=4, linewidth=1.2,
        label="Spore-positive samples (relative count)"
    )
    first_handle = mlines.Line2D(
        [], [], color="black", marker="D", markersize=5, linewidth=0,
        label="First spore-positive sample"
    )
    onset_handle = mlines.Line2D(
        [], [], color=ONSET_COLOR, marker="o", markerfacecolor="white",
        markersize=5, linewidth=5, alpha=0.72,
        label="Observed onset interval (L, U]"
    )
    ax.legend(
        handles=[first_handle, spore_handle, onset_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    OUTPUT_CORE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_CORE, dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_style()
    observations, endpoints, summary = load_data()
    panel_a(observations, endpoints, summary)
    panel_b(summary)
    panel_core(observations, endpoints, summary)
    print(f"Saved: {OUTPUT_A}")
    print(f"Saved: {OUTPUT_B}")
    print(f"Saved: {OUTPUT_CORE}")


if __name__ == "__main__":
    main()
