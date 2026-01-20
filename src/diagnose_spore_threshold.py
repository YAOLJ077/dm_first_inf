import argparse
import os

import pandas as pd


def _load_stage1_fold(result_dir: str, fold: int) -> pd.DataFrame:
    path = os.path.join(result_dir, f"spore_loo_stage1_fold{fold}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing stage1 file: {path}")
    return pd.read_csv(path)


def _load_stage2_fold(result_dir: str, fold: int) -> pd.DataFrame:
    path = os.path.join(result_dir, f"spore_loo_stage2_fold{fold}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing stage2 file: {path}")
    return pd.read_csv(path)


def diagnose(result_dir: str, folds: int, thresholds: list) -> None:
    per_fold = []
    for fold in range(1, folds + 1):
        stage1 = _load_stage1_fold(result_dir, fold)
        stage2 = _load_stage2_fold(result_dir, fold)
        if stage1.empty or stage2.empty:
            continue

        stage1 = stage1.sort_values("train_loss").reset_index(drop=True)
        best_stage1_loss = float(stage1.iloc[0]["train_loss"])
        best_stage2_row = stage2.sort_values("train_mae").iloc[0]
        best_stage2_train = float(best_stage2_row["train_mae"])
        best_stage2_val = float(best_stage2_row["val_mae"])

        for thr in thresholds:
            feasible = stage1[stage1["train_loss"] <= thr]
            feasible_ratio = 0.0 if len(stage1) == 0 else len(feasible) / len(stage1)
            if not feasible.empty:
                feasible_loss_min = float(feasible["train_loss"].min())
                feasible_loss_mean = float(feasible["train_loss"].mean())
            else:
                feasible_loss_min = float("nan")
                feasible_loss_mean = float("nan")

            per_fold.append(
                {
                    "fold": fold,
                    "threshold": thr,
                    "stage1_count": len(stage1),
                    "feasible_count": len(feasible),
                    "feasible_ratio": feasible_ratio,
                    "stage1_best_loss": best_stage1_loss,
                    "feasible_loss_min": feasible_loss_min,
                    "feasible_loss_mean": feasible_loss_mean,
                    "stage2_best_train_mae": best_stage2_train,
                    "stage2_best_val_mae": best_stage2_val,
                }
            )

    fold_df = pd.DataFrame(per_fold)
    fold_out = os.path.join(result_dir, "spore_threshold_fold_summary.csv")
    fold_df.to_csv(fold_out, index=False)

    summary = (
        fold_df.groupby("threshold")
        .agg(
            fold_count=("fold", "count"),
            feasible_ratio_mean=("feasible_ratio", "mean"),
            feasible_ratio_min=("feasible_ratio", "min"),
            feasible_ratio_max=("feasible_ratio", "max"),
            stage1_best_loss_mean=("stage1_best_loss", "mean"),
            feasible_loss_min_mean=("feasible_loss_min", "mean"),
            feasible_loss_mean=("feasible_loss_mean", "mean"),
            stage2_best_train_mae_mean=("stage2_best_train_mae", "mean"),
            stage2_best_val_mae_mean=("stage2_best_val_mae", "mean"),
        )
        .reset_index()
    )
    summary_out = os.path.join(result_dir, "spore_threshold_summary.csv")
    summary.to_csv(summary_out, index=False)

    print(f"Saved: {fold_out}")
    print(f"Saved: {summary_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default=os.path.join(os.getcwd(), "result", "calibration"),
        help="Folder containing spore stage1/stage2 fold CSVs.",
    )
    parser.add_argument("--folds", type=int, default=6, help="Number of folds.")
    parser.add_argument(
        "--thresholds",
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated thresholds for train_loss feasibility.",
    )
    args = parser.parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    diagnose(args.result_dir, args.folds, thresholds)


if __name__ == "__main__":
    main()
