#!/usr/bin/env python3
"""Plot aggregate and site-year FSIM/FSIM-S error metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "result" / "cv_spore_pressure_s_dep38_395_rh80_coarse"
DEFAULT_SPORE = ROOT / "result" / "cv_spore_pressure_s_observed_zero_gate"
DEFAULT_OUTPUT = ROOT / "fig" / "Figure_FSIM_vs_FSIM-S_error_metrics.png"
DEFAULT_SITE_ERROR_OUTPUT = ROOT / "fig" / "Figure_FSIM_vs_FSIM-S_site_year_errors.png"

SITE_LABELS = {
    "23_83_108_1": "Duan",
    "22_61_108_23": "Mingyang",
    "25_45_110_8": "Guilin",
}
SITE_ORDER = {"Duan": 0, "Mingyang": 1, "Guilin": 2}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return 0.0
    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _metric_rows(
    baseline_dir: Path,
    spore_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per model, data split, and metric plus SEM values."""

    baseline_fold = _read_csv(baseline_dir / "fold_parameters.csv")
    baseline_predictions = _read_csv(baseline_dir / "predictions.csv")
    spore_predictions = _read_csv(spore_dir / "predictions.csv")

    rows: list[dict[str, float | str]] = []

    # Training values are means of the six outer-fold metrics. Validation
    # values are recomputed from the six held-out site-year errors.
    fsim_train_mae_samples = baseline_fold["s_training_mae_days"]
    fsim_train_rmse_samples = baseline_fold["s_training_rmse_days"]
    fsim_valid_errors = baseline_predictions.loc[
        baseline_predictions["model"].eq("FSIM"), "error_days"
    ].astype(float)
    fsim_valid_error_samples = fsim_valid_errors.abs()
    rows.extend(
        [
            {
                "model": "FSIM",
                "dataset": "Training",
                "metric": "MAE",
                "value": float(fsim_train_mae_samples.mean()),
                "sem": _sem(fsim_train_mae_samples),
            },
            {
                "model": "FSIM",
                "dataset": "Training",
                "metric": "RMSE",
                "value": float(fsim_train_rmse_samples.mean()),
                "sem": _sem(fsim_train_rmse_samples),
            },
            {
                "model": "FSIM",
                "dataset": "Validation",
                "metric": "MAE",
                "value": float(fsim_valid_error_samples.mean()),
                "sem": _sem(fsim_valid_error_samples),
            },
            {
                "model": "FSIM",
                "dataset": "Validation",
                "metric": "RMSE",
                "value": float(np.sqrt(np.mean(np.square(fsim_valid_errors)))),
                "sem": _sem(fsim_valid_error_samples),
            },
        ]
    )

    # The new FSIM-S gate result stores training rows for each outer fold and
    # held-out rows marked by is_held_out. Fold-level summaries provide the
    # independent replicates for the training error bars.
    train = spore_predictions.loc[spore_predictions["is_held_out"].eq(0)].copy()
    valid = spore_predictions.loc[spore_predictions["is_held_out"].eq(1)].copy()
    train_by_fold = train.groupby("held_out_group", sort=True)
    train_mae_samples = train_by_fold["absolute_error_days"].mean()
    train_rmse_samples = train_by_fold["error_days"].apply(
        lambda values: float(np.sqrt(np.mean(np.square(values))))
    )
    valid_errors = valid["error_days"].astype(float)
    valid_error_samples = valid["absolute_error_days"].astype(float)
    rows.extend(
        [
            {
                "model": "FSIM-S",
                "dataset": "Training",
                "metric": "MAE",
                "value": float(train_mae_samples.mean()),
                "sem": _sem(train_mae_samples),
            },
            {
                "model": "FSIM-S",
                "dataset": "Training",
                "metric": "RMSE",
                "value": float(train_rmse_samples.mean()),
                "sem": _sem(train_rmse_samples),
            },
            {
                "model": "FSIM-S",
                "dataset": "Validation",
                "metric": "MAE",
                "value": float(valid_error_samples.mean()),
                "sem": _sem(valid_error_samples),
            },
            {
                "model": "FSIM-S",
                "dataset": "Validation",
                "metric": "RMSE",
                "value": float(np.sqrt(np.mean(np.square(valid_errors)))),
                "sem": _sem(valid_error_samples),
            },
        ]
    )
    return pd.DataFrame(rows), spore_predictions


def _site_year_error_rows(
    baseline_dir: Path,
    spore_dir: Path,
) -> pd.DataFrame:
    """Return paired final errors for every held-out site-year fold."""

    baseline_predictions = _read_csv(baseline_dir / "predictions.csv")
    spore_predictions = _read_csv(spore_dir / "predictions.csv")

    fsim = baseline_predictions.loc[
        baseline_predictions["model"].eq("FSIM"),
        ["held_out_group", "error_days"],
    ].copy()
    fsim["model"] = "FSIM"

    fsim_s = spore_predictions.loc[
        spore_predictions["is_held_out"].eq(1),
        ["held_out_group", "error_days"],
    ].copy()
    fsim_s["model"] = "FSIM-S"

    rows = pd.concat([fsim, fsim_s], ignore_index=True)
    rows["error_days"] = pd.to_numeric(rows["error_days"], errors="raise")
    rows = rows.drop_duplicates(subset=["model", "held_out_group"])

    groups = sorted(rows["held_out_group"].unique())
    expected = {"FSIM", "FSIM-S"}
    for group in groups:
        observed = set(rows.loc[rows["held_out_group"].eq(group), "model"])
        if observed != expected:
            raise ValueError(
                f"site-year {group} does not have exactly one FSIM and one FSIM-S error: "
                f"{sorted(observed)}"
            )

    rows["site_year"] = rows["held_out_group"].str.rsplit("_", n=1).str[-1]
    rows["site_label"] = rows["held_out_group"].map(
        lambda value: (
            f"{SITE_LABELS.get(value.rsplit('_', 1)[0], value.rsplit('_', 1)[0])}\n"
            f"{value.rsplit('_', 1)[1]}"
        )
    )
    order = sorted(
        rows["held_out_group"].unique(),
        key=lambda value: (
            SITE_ORDER.get(SITE_LABELS.get(value.rsplit("_", 1)[0], ""), 99),
            -int(value.rsplit("_", 1)[1]),
        ),
    )
    rows["held_out_group"] = pd.Categorical(
        rows["held_out_group"], categories=order, ordered=True
    )
    return rows.sort_values(["held_out_group", "model"]).reset_index(drop=True)


def _plot(metrics: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "savefig.dpi": 600,
        }
    )

    models = ["FSIM", "FSIM-S"]
    datasets = ["Training", "Validation"]
    colors = {"FSIM": "#4C78A8", "FSIM-S": "#D17A2B"}
    x = np.arange(len(models), dtype=float)
    width = 0.30
    offsets = {"Training": -width / 2, "Validation": width / 2}

    upper = float((metrics["value"] + metrics["sem"]).max())
    y_max = max(10.0, np.ceil(upper * 1.16))
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.45), sharey=True)

    for axis, metric, panel in zip(axes, ("MAE", "RMSE"), ("(a)", "(b)")):
        for dataset in datasets:
            values = []
            errors = []
            for model in models:
                row = metrics.query(
                    "model == @model and dataset == @dataset and metric == @metric"
                ).iloc[0]
                values.append(float(row["value"]))
                errors.append(float(row["sem"]))
            bars = axis.bar(
                x + offsets[dataset],
                values,
                width=width,
                yerr=errors,
                capsize=2.5,
                color=[colors[model] for model in models],
                edgecolor="#263238",
                linewidth=0.55,
                alpha=0.95 if dataset == "Training" else 0.62,
                hatch=None if dataset == "Training" else "///",
                error_kw={"ecolor": "#263238", "elinewidth": 0.8, "capthick": 0.8},
                zorder=3,
            )
            for bar, value, error in zip(bars, values, errors):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + error + 0.18,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#202830",
                )
        axis.text(0.01, 0.98, f"{panel} {metric}", transform=axis.transAxes,
                  ha="left", va="top", fontweight="bold", fontsize=10)
        axis.set_xticks(x)
        axis.set_xticklabels(models)
        axis.set_ylim(0, y_max)
        axis.set_yticks(np.arange(0, y_max + 0.01, 2))
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.6, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.65, length=3)

    axes[0].set_ylabel("Error (days)")
    axes[1].legend(
        handles=[
            Patch(facecolor="#7D8790", edgecolor="#263238", label="Training"),
            Patch(facecolor="#7D8790", edgecolor="#263238", hatch="///", alpha=0.62, label="Validation"),
        ],
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.4,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.98, wspace=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_site_year_errors(site_errors: pd.DataFrame, output: Path) -> None:
    """Plot final held-out errors for each site-year and model."""

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "savefig.dpi": 600,
        }
    )

    models = ["FSIM", "FSIM-S"]
    colors = {"FSIM": "#4C78A8", "FSIM-S": "#D17A2B"}
    groups = list(site_errors["held_out_group"].cat.categories)
    labels = [
        site_errors.loc[site_errors["held_out_group"].eq(group), "site_label"].iloc[0]
        for group in groups
    ]
    y = np.arange(len(groups), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 4.30), sharex=True, sharey=True)
    lower = site_errors["error_days"].min()
    upper = site_errors["error_days"].max()
    x_min = float(np.floor(min(lower, -7.0) - 1.0))
    x_max = float(np.ceil(max(upper, 7.0) + 1.0))
    ticks = np.arange(np.ceil(x_min / 5) * 5, x_max + 0.01, 5.0)
    for axis, model, panel in zip(axes, models, ("(a)", "(b)")):
        values = [
            float(
                site_errors.loc[
                    site_errors["held_out_group"].eq(group)
                    & site_errors["model"].eq(model),
                    "error_days",
                ].iloc[0]
            )
            for group in groups
        ]
        for row_y, value in zip(y, values):
            if value != 0.0:
                axis.plot(
                    [0.0, value],
                    [row_y, row_y],
                    color=colors[model],
                    linewidth=2.2,
                    solid_capstyle="butt",
                    zorder=3,
                )
        for row_y, value in zip(y, values):
            axis.vlines(
                value,
                row_y - 0.13,
                row_y + 0.13,
                color=colors[model],
                linewidth=2.3,
                zorder=5,
            )
            x_text = value + 0.55
            axis.text(
                x_text,
                row_y - 0.16,
                f"{value:+.0f}" if value != 0 else "0",
                ha="left",
                va="bottom",
                fontsize=8,
                color=colors[model],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
            )
        axis.axvspan(-7, 7, color="#DCEAD8", alpha=0.72, zorder=0)
        axis.axvline(0.0, color="#263238", linewidth=0.8, zorder=2)
        axis.text(
            0.01,
            1.03,
            f"{panel} {model}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(labels)
        axis.set_ylim(len(groups) - 0.5, -0.5)
        axis.set_xlim(x_min, x_max)
        axis.set_xticks(ticks)
        axis.grid(axis="x", color="#D9DEE3", linewidth=0.6, zorder=1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.65, length=3)

    axes[0].set_xlabel("Error (days)")
    axes[1].set_xlabel("Error (days)")
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.83, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--spore", type=Path, default=DEFAULT_SPORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--site-error-output",
        type=Path,
        default=DEFAULT_SITE_ERROR_OUTPUT,
        help="Output PNG for paired signed held-out site-year errors.",
    )
    args = parser.parse_args()

    metrics, _ = _metric_rows(args.baseline, args.spore)
    site_errors = _site_year_error_rows(args.baseline, args.spore)
    _plot(metrics, args.output)
    _plot_site_year_errors(site_errors, args.site_error_output)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(site_errors[["held_out_group", "model", "error_days"]].to_string(index=False))
    print(f"Saved {args.output}")
    print(f"Saved {args.site_error_output}")


if __name__ == "__main__":
    main()
