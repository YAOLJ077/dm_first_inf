#!/usr/bin/env python3
"""Plot Sobol sensitivity ranking and sample-size stability.

The figure uses the censored mean first-symptom-onset day as the primary
output and compares first-order (S1) and total-order (ST) indices across
the three available Sobol sample sizes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm


OUTPUT_NAME = "mean_first_window_day_censored"
SAMPLE_SIZES = (256, 512, 1024)

PARAMETER_LABELS = {
    "infection_degree_hours": "IDH",
    "wind_min_m_s": "WTH",
    "effective_deposition_min_rain_mm": "EDR",
    "sporulation_rh_min_pct": "SRH",
    "minimum_moist_hours": "MMH",
    "heavy_rain_washoff_mm": "HWR",
}

def _read_indices(project_root: Path) -> pd.DataFrame:
    frames = []
    for sample_size in SAMPLE_SIZES:
        path = project_root / "result" / "sobol_sensitivity" / f"S_N{sample_size}" / "sobol_indices.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["sample_size"] = sample_size
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["output"].eq(OUTPUT_NAME)].copy()
    expected = set(PARAMETER_LABELS)
    observed = set(data["parameter"])
    missing = expected - observed
    if missing:
        raise ValueError(f"Missing Sobol parameters: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"No rows found for output {OUTPUT_NAME!r}")
    return data


def _ordered_parameters(data: pd.DataFrame) -> list[str]:
    latest = data[data["sample_size"].eq(max(SAMPLE_SIZES))]
    return (
        latest.groupby("parameter", as_index=False)["ST"]
        .mean()
        .sort_values("ST", ascending=False)["parameter"]
        .tolist()
    )


def _plot_rank_panel(ax: plt.Axes, data: pd.DataFrame, parameters: list[str]) -> None:
    latest = data[data["sample_size"].eq(max(SAMPLE_SIZES))].set_index("parameter").loc[parameters]
    y = np.arange(len(parameters))
    offset = 0.19

    ax.barh(
        y + offset,
        latest["S1"],
        height=0.32,
        xerr=latest["S1_conf"],
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
        error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 2},
        label=r"First-order $S_1$",
    )
    ax.barh(
        y - offset,
        latest["ST"],
        height=0.32,
        xerr=latest["ST_conf"],
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.5,
        error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 2},
        label=r"Total-order $S_T$",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([PARAMETER_LABELS[p] for p in parameters])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.95, float((latest["ST"] + latest["ST_conf"]).max()) * 1.08))
    ax.set_xlabel("Sobol index")
    ax.set_title(r"$N=1024$ ranking", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, linestyle="--")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5, handlelength=1.4)


def _plot_stability_panel(ax: plt.Axes, data: pd.DataFrame, parameters: list[str]) -> None:
    values = (
        data.pivot(index="parameter", columns="sample_size", values="ST")
        .reindex(index=parameters, columns=SAMPLE_SIZES)
    )
    max_value = float(values.to_numpy().max())

    image = ax.imshow(
        values.to_numpy(),
        cmap="Blues",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=max_value),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(SAMPLE_SIZES)))
    ax.set_xticklabels([str(size) for size in SAMPLE_SIZES])
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_yticklabels([PARAMETER_LABELS[p] for p in parameters])
    ax.set_xlabel("Sobol sample size ($N$)")
    ax.set_title(r"Total-order index $S_T$ across $N$", loc="left", fontweight="bold")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values.iloc[row, col])
            text_color = "white" if value / max_value >= 0.28 else "#222222"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=text_color)

    ax.set_xticks(np.arange(-0.5, len(SAMPLE_SIZES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "fig" / "figure_09.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _read_indices(project_root)
    parameters = _ordered_parameters(data)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.1,
            "axes.linewidth": 0.75,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "savefig.dpi": 600,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 4.25), gridspec_kw={"width_ratios": (1.25, 1.0)})
    _plot_rank_panel(axes[0], data, parameters)
    _plot_stability_panel(axes[1], data, parameters)

    axes[0].text(-0.12, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold", va="bottom")
    axes[1].text(-0.14, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.15, top=0.91, wspace=0.34)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
