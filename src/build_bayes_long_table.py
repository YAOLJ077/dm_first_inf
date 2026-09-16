# LEGACY / 旧版：不属于当前 FSIM 分析入口。
# Status: legacy_result_processing; see config/project_registry.json and config/README.md.
# Retained for historical traceability; current results require separate validation.
import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd


def _collect_seed_dirs(model_dir: Path) -> List[Path]:
    return sorted([p for p in model_dir.glob("seed_*") if p.is_dir()])


def _read_seed_summary(seed_dir: Path, model_type: str) -> pd.DataFrame:
    summary_path = seed_dir / "baseline_kfold_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(summary_path)
    if df.empty:
        return df

    seed_str = seed_dir.name.replace("seed_", "")
    try:
        seed = int(seed_str)
    except ValueError:
        seed = pd.NA

    df["model_type"] = model_type
    df["seed"] = seed
    df["seed_dir"] = seed_dir.name
    df["source_file"] = str(summary_path)

    # Standardize fold column in case files differ
    if "fold" not in df.columns:
        if "Fold" in df.columns:
            df = df.rename(columns={"Fold": "fold"})
        else:
            df["fold"] = pd.NA

    # Keep a stable preferred column order first, then append others
    preferred = [
        "model_type",
        "seed",
        "fold",
        "train_mae",
        "val_mae",
        "train_rmse",
        "val_rmse",
        "train_std",
        "val_std",
        "train_var",
        "val_var",
        "source_file",
        "seed_dir",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def _collect_model(model_dir: Path, model_type: str) -> pd.DataFrame:
    rows = []
    for seed_dir in _collect_seed_dirs(model_dir):
        sdf = _read_seed_summary(seed_dir, model_type)
        if not sdf.empty:
            rows.append(sdf)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="result/calibration",
        help="Directory containing baseline seed_* folders.",
    )
    parser.add_argument(
        "--spore-dir",
        type=str,
        default="result/calibration_spore_infthr",
        help="Directory containing spore model seed_* folders.",
    )
    parser.add_argument(
        "--spore-label",
        type=str,
        default="spore_infthr",
        help="Label written to model_type for the spore model rows.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="result/bayes_sum",
        help="Output directory for unified table.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    baseline_dir = (project_root / args.baseline_dir).resolve()
    spore_dir = (project_root / args.spore_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = _collect_model(baseline_dir, "baseline")
    spore_df = _collect_model(spore_dir, args.spore_label)

    if baseline_df.empty and spore_df.empty:
        raise FileNotFoundError(
            f"No summary files found under:\n- {baseline_dir}\n- {spore_dir}"
        )

    long_df = pd.concat([baseline_df, spore_df], ignore_index=True)
    long_df = long_df.sort_values(["model_type", "seed", "fold"]).reset_index(drop=True)

    out_csv = out_dir / "bayes_models_fold_long.csv"
    out_xlsx = out_dir / "bayes_models_fold_long.xlsx"
    long_df.to_csv(out_csv, index=False)
    long_df.to_excel(out_xlsx, index=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")
    print(
        f"Rows: {len(long_df)} | baseline: {len(baseline_df)} | {args.spore_label}: {len(spore_df)}"
    )


if __name__ == "__main__":
    main()
