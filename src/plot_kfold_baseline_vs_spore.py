import json
import os

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = os.path.join(os.getcwd(), "result", "calibration")
BASELINE_SUMMARY_CSV = os.path.join(RESULT_DIR, "baseline_kfold_summary.csv")
SPORE_SUMMARY_CSV = os.path.join(RESULT_DIR, "spore_kfold_summary.csv")
BASELINE_SUMMARY_JSON = os.path.join(RESULT_DIR, "baseline_kfold_summary.json")
SPORE_SUMMARY_JSON = os.path.join(RESULT_DIR, "spore_kfold_summary.json")

FONT_FAMILY = "Times New Roman"
FONT_SIZE = 12
FIGSIZE = (7.0, 3.5)  # double-column width


def _setup_style():
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary json: {path}")
    with open(path, "r") as f:
        return json.load(f)


def plot_fold_mae_train_val(baseline_df: pd.DataFrame, spore_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = baseline_df["fold"].values
    width = 0.2
    ax.bar(x - 1.5 * width, baseline_df["train_mae"], width, label="Baseline Train")
    ax.bar(x - 0.5 * width, baseline_df["val_mae"], width, label="Baseline Val")
    ax.bar(x + 0.5 * width, spore_df["train_mae"], width, label="Spore Train")
    ax.bar(x + 1.5 * width, spore_df["val_mae"], width, label="Spore Val")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (days)")
    ax.set_title("Fold MAE (Train vs Val)")
    ax.legend(frameon=False)
    fig.tight_layout()
    # fig.savefig(os.path.join(RESULT_DIR, "fig_fold_mae_train_val.png"), dpi=300)
    plt.show()


def plot_val_mae_box(baseline_df: pd.DataFrame, spore_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(
        [baseline_df["val_mae"], spore_df["val_mae"]],
        labels=["Baseline", "Spore"],
        showmeans=True,
    )
    ax.set_ylabel("Val MAE (days)")
    ax.set_title("Validation MAE Distribution")
    fig.tight_layout()
    # fig.savefig(os.path.join(RESULT_DIR, "fig_val_mae_box.png"), dpi=300)
    plt.show()


def plot_avg_mae_rmse(baseline_json: dict, spore_json: dict) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = ["Baseline", "Spore"]
    train_mae_vals = [baseline_json["train_mae_mean"], spore_json["train_mae_mean"]]
    val_mae_vals = [baseline_json["val_mae_mean"], spore_json["val_mae_mean"]]
    train_rmse_vals = [baseline_json["train_rmse"], spore_json["train_rmse"]]
    val_rmse_vals = [baseline_json["val_rmse"], spore_json["val_rmse"]]

    x = range(len(labels))
    width = 0.2
    ax.bar([i - 1.5 * width for i in x], train_mae_vals, width, label="Train MAE")
    ax.bar([i - 0.5 * width for i in x], val_mae_vals, width, label="Val MAE")
    ax.bar([i + 0.5 * width for i in x], train_rmse_vals, width, label="Train RMSE")
    ax.bar([i + 1.5 * width for i in x], val_rmse_vals, width, label="Val RMSE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Days")
    ax.set_title("Average Error (Train vs Val)")
    ax.legend(frameon=False)
    fig.tight_layout()
    # fig.savefig(os.path.join(RESULT_DIR, "fig_avg_mae_rmse.png"), dpi=300)
    plt.show()


def main() -> None:
    _setup_style()

    if not os.path.exists(BASELINE_SUMMARY_CSV):
        raise FileNotFoundError(f"Missing baseline summary csv: {BASELINE_SUMMARY_CSV}")
    if not os.path.exists(SPORE_SUMMARY_CSV):
        raise FileNotFoundError(f"Missing spore summary csv: {SPORE_SUMMARY_CSV}")

    baseline_df = pd.read_csv(BASELINE_SUMMARY_CSV)
    spore_df = pd.read_csv(SPORE_SUMMARY_CSV)
    baseline_json = _load_json(BASELINE_SUMMARY_JSON)
    spore_json = _load_json(SPORE_SUMMARY_JSON)

    plot_fold_mae_train_val(baseline_df, spore_df)
    plot_val_mae_box(baseline_df, spore_df)
    plot_avg_mae_rmse(baseline_json, spore_json)

    if "train_spore_consistency" in baseline_df and "train_spore_consistency" in spore_df:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        x = baseline_df["fold"].values
        width = 0.2
        ax.bar(x - 1.5 * width, baseline_df["train_spore_consistency"], width, label="Baseline Train")
        ax.bar(x - 0.5 * width, baseline_df["val_spore_consistency"], width, label="Baseline Val")
        ax.bar(x + 0.5 * width, spore_df["train_spore_consistency"], width, label="Spore Train")
        ax.bar(x + 1.5 * width, spore_df["val_spore_consistency"], width, label="Spore Val")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Spore Consistency")
        ax.set_title("Spore Consistency (Train vs Val)")
        ax.legend(frameon=False)
        fig.tight_layout()
        # fig.savefig(os.path.join(RESULT_DIR, "fig_spore_consistency.png"), dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
