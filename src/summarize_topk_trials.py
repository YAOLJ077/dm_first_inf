import argparse
import math
import os

import pandas as pd


def _parse_topk_list(value: str) -> list:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _load_fold_csv(result_dir: str, mode: str, fold: int) -> pd.DataFrame:
    if mode == "baseline":
        filename = f"baseline_loo_fold_{fold}_trials.csv"
    elif mode == "spore":
        filename = f"spore_loo_stage2_fold{fold}.csv"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    path = os.path.join(result_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing fold file: {path}")
    return pd.read_csv(path)


def summarize_topk(result_dir: str, mode: str, topk_list: list, n_folds: int) -> None:
    per_fold_rows = []
    for fold in range(1, n_folds + 1):
        df = _load_fold_csv(result_dir, mode, fold)
        if df.empty:
            continue
        df = df.sort_values("train_mae").reset_index(drop=True)
        n_total = len(df)
        for pct in topk_list:
            n_top = max(1, int(math.ceil(n_total * pct / 100.0)))
            top = df.iloc[:n_top].copy()
            val_best_idx = top["val_mae"].idxmin()
            val_best_row = top.loc[val_best_idx]
            per_fold_rows.append(
                {
                    "fold": fold,
                    "pct": pct,
                    "n_total": n_total,
                    "n_top": n_top,
                    "train_mae_mean": float(top["train_mae"].mean()),
                    "val_mae_mean": float(top["val_mae"].mean()),
                    "train_at_val_best": float(val_best_row["train_mae"]),
                    "val_best": float(val_best_row["val_mae"]),
                }
            )

    fold_df = pd.DataFrame(per_fold_rows)
    fold_out = os.path.join(result_dir, f"{mode}_topk_fold_summary.csv")
    fold_df.to_csv(fold_out, index=False)

    summary = (
        fold_df.groupby("pct")
        .agg(
            fold_count=("fold", "count"),
            train_mae_mean=("train_mae_mean", "mean"),
            val_mae_mean=("val_mae_mean", "mean"),
            train_at_val_best_mean=("train_at_val_best", "mean"),
            val_best_mean=("val_best", "mean"),
        )
        .reset_index()
    )
    summary_out = os.path.join(result_dir, f"{mode}_topk_summary.csv")
    summary.to_csv(summary_out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "spore"],
        default="baseline",
        help="Which fold trials to summarize.",
    )
    parser.add_argument(
        "--result-dir",
        default=os.path.join(os.getcwd(), "result", "calibration"),
        help="Folder containing fold trial CSVs.",
    )
    parser.add_argument(
        "--topk",
        default="1,2,3,5,10",
        help="Comma-separated top%% list, based on train_mae.",
    )
    parser.add_argument("--folds", type=int, default=6, help="Number of folds.")
    args = parser.parse_args()

    topk_list = _parse_topk_list(args.topk)
    summarize_topk(args.result_dir, args.mode, topk_list, args.folds)
    print(f"Saved: {os.path.join(args.result_dir, f'{args.mode}_topk_fold_summary.csv')}")
    print(f"Saved: {os.path.join(args.result_dir, f'{args.mode}_topk_summary.csv')}")


if __name__ == "__main__":
    main()
