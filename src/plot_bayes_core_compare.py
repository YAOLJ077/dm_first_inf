from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_summary(sum_dir: Path) -> pd.DataFrame:
    path = sum_dir / "bayes_model_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return pd.read_csv(path)


def _plot_panel(
    ax,
    baseline_row: pd.Series,
    spore_row: pd.Series,
    prefix: str,
    panel_tag: str,
) -> None:
    # Grouped bars: baseline vs spore, each with MAE and RMSE
    x = [0.0, 1.0]
    width = 0.288
    mae_vals = [baseline_row[f"{prefix}_mae_mean"], spore_row[f"{prefix}_mae_mean"]]
    rmse_vals = [baseline_row[f"{prefix}_rmse_mean"], spore_row[f"{prefix}_rmse_mean"]]
    mae_err = [baseline_row[f"{prefix}_mae_std"], spore_row[f"{prefix}_mae_std"]]
    rmse_err = [baseline_row[f"{prefix}_rmse_std"], spore_row[f"{prefix}_rmse_std"]]

    bars_mae = ax.bar(
        [xi - width / 2 for xi in x],
        mae_vals,
        yerr=mae_err,
        width=width,
        color="#4C78A8",
        ecolor="#666666",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        alpha=0.9,
        label="MAE",
    )
    bars_rmse = ax.bar(
        [xi + width / 2 for xi in x],
        rmse_vals,
        yerr=rmse_err,
        width=width,
        color="#F28E2B",
        ecolor="#666666",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        alpha=0.9,
        label="RMSE",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["FIM", "FIM-S"])
    ax.tick_params(axis="both", direction="in")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.text(0.05, 0.96, panel_tag, transform=ax.transAxes, ha="left", va="top")
    # value labels in bar center
    for b in list(bars_mae) + list(bars_rmse):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h * 0.5,
            f"{h:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="Times New Roman",
        )

    # Reduction labels (%) above spore bars (relative to baseline)
    mae_pct = (
        (baseline_row[f"{prefix}_mae_mean"] - spore_row[f"{prefix}_mae_mean"])
        / baseline_row[f"{prefix}_mae_mean"]
        * 100.0
        if baseline_row[f"{prefix}_mae_mean"] != 0
        else 0.0
    )
    rmse_pct = (
        (baseline_row[f"{prefix}_rmse_mean"] - spore_row[f"{prefix}_rmse_mean"])
        / baseline_row[f"{prefix}_rmse_mean"]
        * 100.0
        if baseline_row[f"{prefix}_rmse_mean"] != 0
        else 0.0
    )
    spore_mae_bar = bars_mae[1]
    spore_rmse_bar = bars_rmse[1]
    ax.text(
        spore_mae_bar.get_x() + spore_mae_bar.get_width() / 2,
        (spore_row[f"{prefix}_mae_mean"] + spore_row[f"{prefix}_mae_std"] + 0.05) * 1.08,
        f"\u2193 {mae_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontfamily="Times New Roman",
    )
    ax.text(
        spore_rmse_bar.get_x() + spore_rmse_bar.get_width() / 2,
        (spore_row[f"{prefix}_rmse_mean"] + spore_row[f"{prefix}_rmse_std"] + 0.05) * 1.05,
        f"\u2193 {rmse_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontfamily="Times New Roman",
    )


def main() -> None:
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    sum_dir = project_root / "result" / "bayes_sum"
    summary = _load_summary(sum_dir)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})

    b = summary[summary["model_type"] == "baseline"]
    s = summary[summary["model_type"].str.contains("spore", case=False, na=False)]
    if b.empty or s.empty:
        raise ValueError("Could not find baseline/spore rows in bayes_model_summary.csv")

    b_row = b.iloc[0]
    s_row = s.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.2))
    _plot_panel(axes[0], b_row, s_row, "train", "(a)")
    _plot_panel(axes[1], b_row, s_row, "val", "(b)")
    axes[1].legend(frameon=False, fontsize=10, loc="upper right")

    for ax in axes:
        ax.set_ylim(0, 17.5)

    axes[0].set_ylabel("Error(days)")
    fig.tight_layout()
    # out = project_root / "fig" / "bayes_core_bar_1x2.png"
    # fig.savefig(out, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
