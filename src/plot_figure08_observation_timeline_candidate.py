"""Plot a common-calendar view of spore observations and disease-onset windows."""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple


ROOT = Path(__file__).resolve().parents[1]
SPORE_PATH = ROOT / "data" / "spore_data" / "guangxi_site_years_spore_data.xlsx"
DISEASE_PATH = ROOT / "data" / "disease_data" / "grape_downy_mildew_data.xlsx"
OUTPUT_PATH = ROOT / "fig" / "Figure_08_observation_timeline_candidate.png"
FINAL_OUTPUT_PATH = ROOT / "fig" / "Figure_08_core_timeline.png"

SITE_META = {
    "Guilin": {"label": "Guilin", "color": "#D55E00", "order": 0},
    "Duan": {"label": "Hechi", "color": "#0072B2", "order": 1},
    "Nanning": {"label": "Nanning", "color": "#009E73", "order": 2},
}

ONSET_COLOR = "#B2182B"
ZERO_COLOR = "#8C8C8C"
GRID_COLOR = "#D9D9D9"
REFERENCE_YEAR = 2001


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
        }
    )


def month_day_axis(date_series: pd.Series) -> pd.Series:
    """Map dates from different years to one non-leap reference calendar."""
    return date_series.map(
        lambda value: pd.Timestamp(
            REFERENCE_YEAR, value.month, value.day
        ).dayofyear
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    spores = pd.read_excel(SPORE_PATH, sheet_name="CK_Data")
    spores["Date"] = pd.to_datetime(spores["Date"], errors="raise")
    spores["Spore"] = pd.to_numeric(spores["Spore"], errors="coerce")

    disease = pd.read_excel(DISEASE_PATH, sheet_name=0)
    disease = disease.dropna(subset=["City", "Year", "Date"]).copy()
    disease["Year"] = disease["Year"].astype(int)
    disease["Date"] = pd.to_datetime(disease["Date"], errors="raise")
    disease = disease.rename(columns={"City": "Place", "Date": "U"})
    disease["L"] = disease["U"] - pd.Timedelta(days=3)

    # Keep the six site-years represented by the current observation files.
    keys = spores[["Place", "Year"]].drop_duplicates()
    endpoints = keys.merge(disease[["Place", "Year", "L", "U"]],
                           on=["Place", "Year"], how="left", validate="one_to_one")
    if endpoints[["L", "U"]].isna().any().any():
        missing = endpoints.loc[endpoints["U"].isna(), ["Place", "Year"]]
        raise ValueError(f"Missing disease-onset dates for {missing.to_dict('records')}")
    endpoints["site_order"] = endpoints["Place"].map(
        lambda place: SITE_META[place]["order"]
    )
    endpoints = endpoints.sort_values(["site_order", "Year"]).reset_index(drop=True)
    return spores, endpoints


def format_month_day(day_of_year: int) -> str:
    value = pd.Timestamp(REFERENCE_YEAR, 1, 1) + pd.Timedelta(days=int(day_of_year) - 1)
    return value.strftime("%b %-d")


def plot() -> None:
    configure_style()
    observations, endpoints = load_data()
    max_count = float(observations["Spore"].max())
    raw_ticks = np.array([0, 1, 5, 10, 50, 100])
    transformed_ticks = np.log1p(raw_ticks)
    y_max = np.log1p(max_count) * 1.08

    fig, axes = plt.subplots(
        len(endpoints),
        1,
        figsize=(7.0, 7.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.16},
    )
    axes = np.atleast_1d(axes)

    for index, (ax, endpoint) in enumerate(
        zip(axes, endpoints.itertuples(index=False))
    ):
        meta = SITE_META[endpoint.Place]
        group = observations.loc[
            (observations["Place"] == endpoint.Place)
            & (observations["Year"] == endpoint.Year)
        ].copy()
        sampled = group.dropna(subset=["Spore"]).copy()
        sampled["x"] = month_day_axis(sampled["Date"])
        positive = sampled.loc[sampled["Spore"] > 0]
        zero = sampled.loc[sampled["Spore"] == 0]
        onset_x = month_day_axis(pd.Series([endpoint.L, endpoint.U]))
        l_x, u_x = onset_x.iloc[0], onset_x.iloc[1]

        # The red bar marks the observed interval [last negative, first positive].
        ax.hlines(-0.48, l_x, u_x, color=ONSET_COLOR, linewidth=5.5,
                  alpha=0.88, zorder=2)
        ax.scatter(l_x, -0.48, s=28, facecolor="white", edgecolor=ONSET_COLOR,
                   linewidth=1.1, zorder=4)
        ax.scatter(u_x, -0.48, s=28, facecolor=ONSET_COLOR, edgecolor=ONSET_COLOR,
                   linewidth=1.0, zorder=4)

        positive_height = np.log1p(positive["Spore"].to_numpy(dtype=float))
        ax.vlines(
            positive["x"],
            0,
            positive_height,
            color=meta["color"],
            linewidth=1.35,
            alpha=0.92,
            zorder=2,
        )
        ax.scatter(
            positive["x"],
            positive_height,
            s=20,
            color=meta["color"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.scatter(
            zero["x"],
            np.zeros(len(zero)),
            s=13,
            facecolor="white",
            edgecolor=ZERO_COLOR,
            linewidth=0.75,
            zorder=3,
        )

        ax.text(
            0.012,
            0.86,
            f"{meta['label']} {endpoint.Year}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=meta["color"],
        )
        ax.set_ylim(-0.72, y_max)
        ax.set_yticks(transformed_ticks)
        ax.set_yticklabels([str(value) for value in raw_ticks])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.75)
        if index != len(axes) - 1:
            ax.tick_params(labelbottom=False)
            ax.spines["bottom"].set_color("#BFBFBF")

    start = pd.Timestamp(REFERENCE_YEAR, 3, 20).dayofyear
    end = pd.Timestamp(REFERENCE_YEAR, 6, 5).dayofyear
    ticks = [
        pd.Timestamp(REFERENCE_YEAR, 3, 20).dayofyear,
        pd.Timestamp(REFERENCE_YEAR, 4, 1).dayofyear,
        pd.Timestamp(REFERENCE_YEAR, 4, 15).dayofyear,
        pd.Timestamp(REFERENCE_YEAR, 5, 1).dayofyear,
        pd.Timestamp(REFERENCE_YEAR, 5, 15).dayofyear,
        pd.Timestamp(REFERENCE_YEAR, 6, 1).dayofyear,
    ]
    axes[-1].set_xlim(start, end)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([format_month_day(value) for value in ticks])
    axes[-1].set_xlabel("Calendar date")
    fig.supylabel("Airborne sporangia per collection (count; log1p scale)", x=0.072)

    positive_handle = mlines.Line2D(
        [], [], color="#444444", marker="o", markersize=3.5, linewidth=1.1,
        label="Spore-positive collection",
    )
    zero_handle = mlines.Line2D(
        [], [], color=ZERO_COLOR, marker="o", markerfacecolor="white",
        markersize=3.5, linewidth=0, label="Zero-count collection",
    )
    onset_handle = (
        mlines.Line2D(
            [], [], color=ONSET_COLOR, marker="o", markerfacecolor="white",
            markeredgecolor=ONSET_COLOR, markersize=3.5, linewidth=0,
        ),
        mlines.Line2D(
            [], [], color=ONSET_COLOR, linewidth=3.8,
        ),
        mlines.Line2D(
            [], [], color=ONSET_COLOR, marker="o", markerfacecolor=ONSET_COLOR,
            markeredgecolor=ONSET_COLOR, markersize=3.5, linewidth=0,
        ),
    )
    fig.legend(
        handles=[positive_handle, zero_handle, onset_handle],
        labels=[
            "Spore-positive collection",
            "Zero-count collection",
            "Observed disease-onset window",
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.85,
        handletextpad=0.4,
        handler_map={tuple: HandlerTuple(ndivide=3, pad=0.08)},
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.935, bottom=0.09)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=600, facecolor="white")
    fig.savefig(FINAL_OUTPUT_PATH, dpi=600, facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    plot()
