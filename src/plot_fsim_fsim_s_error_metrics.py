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
DEFAULT_BASELINE = ROOT / "result" / "fsim_calibration"
DEFAULT_SPORE = ROOT / "result" / "fsim_s_calibration"
DEFAULT_OUTPUT = ROOT / "fig" / "figure_10.png"
DEFAULT_SITE_ERROR_OUTPUT = ROOT / "fig" / "figure_11.png"

SITE_LABELS = {
    "23_83_108_1": "Hechi",
    "22_61_108_23": "Nanning",
    "25_45_110_8": "Guilin",
}
SITE_ORDER = {"Hechi": 0, "Nanning": 1, "Guilin": 2}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _summarize_errors(
    model: str, training: pd.DataFrame, validation: pd.DataFrame
) -> list[dict[str, float | str]]:
    """Bootstrap site-years with the fitted fold models held fixed.

    Repeated training predictions of a site-year receive the same bootstrap
    weight across folds. The overlapping folds are not independent replicates.
    """
    groups = sorted(validation["site_year"].unique())
    n = len(groups)
    if n < 2 or len(validation) != n:
        raise ValueError(f"{model}: expected one validation error per site-year")
    matrix = training.pivot(
        index="held_out_group", columns="site_year", values="error_days"
    ).reindex(index=groups, columns=groups).to_numpy(dtype=float)
    available = np.isfinite(matrix)
    if len(training) != n * (n - 1) or not np.array_equal(
        available, ~np.eye(n, dtype=bool)
    ):
        raise ValueError(f"{model}: expected complete leave-one-site-year-out training rows")
    valid_errors = validation.set_index("site_year").loc[groups, "error_days"].to_numpy(dtype=float)
    if not np.isfinite(valid_errors).all():
        raise ValueError(f"{model}: validation errors must be finite")

    weights = np.random.default_rng(20261008).multinomial(
        n, np.full(n, 1.0 / n), size=20000
    )
    denominators = weights @ available.T
    # A resample containing only the held-out site-year leaves one fold empty.
    usable = (denominators > 0).all(axis=1)
    rows = []
    for metric in ("MAE", "RMSE"):
        train_values = np.where(available, np.abs(matrix) if metric == "MAE" else matrix**2, 0.0)
        valid_values = np.abs(valid_errors) if metric == "MAE" else valid_errors**2
        train_point = train_values.sum(axis=1) / available.sum(axis=1)
        train_boot = (weights[usable] @ train_values.T) / denominators[usable]
        valid_point = valid_values.mean()
        valid_boot = weights @ valid_values / n
        if metric == "RMSE":
            train_point, train_boot = np.sqrt(train_point), np.sqrt(train_boot)
            valid_point, valid_boot = np.sqrt(valid_point), np.sqrt(valid_boot)
        for dataset, value, samples in (
            ("Training", train_point.mean(), train_boot.mean(axis=1)),
            ("Validation", valid_point, valid_boot),
        ):
            rows.append({
                "model": model, "dataset": dataset, "metric": metric,
                "value": float(value), "se": float(np.std(samples, ddof=1)),
            })
    return rows


def _metric_rows(
    baseline_dir: Path,
    spore_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fold-mean training and pooled validation metrics with bootstrap SE."""

    baseline_fold = _read_csv(baseline_dir / "fold_parameters.csv")
    baseline_predictions = _read_csv(baseline_dir / "predictions.csv")
    baseline_training = _read_csv(baseline_dir / "training_predictions.csv")
    spore_predictions = _read_csv(spore_dir / "predictions.csv")
    baseline_training = baseline_training.loc[baseline_training["model"].eq("FSIM")]
    # Check the recovered training predictions against the saved fold scores.
    for held_out, group in baseline_training.groupby("held_out_group"):
        fold = baseline_fold.set_index("held_out_group").loc[held_out]
        values = group["error_days"].to_numpy(dtype=float)
        if not np.allclose(
            [np.abs(values).mean(), np.sqrt(np.mean(values**2))],
            [fold["s_training_mae_days"], fold["s_training_rmse_days"]],
        ):
            raise ValueError(f"FSIM training predictions disagree with saved scores: {held_out}")
    rows = _summarize_errors(
        "FSIM", baseline_training,
        baseline_predictions.loc[baseline_predictions["model"].eq("FSIM")],
    )
    rows.extend(_summarize_errors(
        "FSIM-S",
        spore_predictions.loc[spore_predictions["is_held_out"].eq(0)],
        spore_predictions.loc[spore_predictions["is_held_out"].eq(1)],
    ))
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

    upper = float((metrics["value"] + metrics["se"]).max())
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
                errors.append(float(row["se"]))
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
