#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_fold_best_rows(root: Path, model_name: str) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except Exception:
            continue
        for fold_dir in sorted(seed_dir.glob("baseline_fold_*")):
            if not fold_dir.is_dir():
                continue
            try:
                fold = int(fold_dir.name.split("_")[-1])
            except Exception:
                continue
            p = fold_dir / "baseline_kfold_best_params.json"
            if not p.exists():
                continue
            with open(p, "r") as f:
                d = json.load(f)
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
                    "train_mae": float(d.get("train_mae", float("nan"))),
                    "val_mae": float(d.get("val_mae", float("nan"))),
                    "train_std": float(d.get("train_std", float("nan"))),
                    "val_std": float(d.get("val_std", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def _load_selected(sum_dir: Path) -> tuple[int, int, int]:
    report_path = sum_dir / "final_selection_report.json"
    b_path = sum_dir / "baseline_seed_best_global.csv"
    s_path = sum_dir / "spore_seed_best_global.csv"
    if not report_path.exists() or not b_path.exists() or not s_path.exists():
        raise FileNotFoundError(
            "Missing selection files. Run summarize_calibration_results.py and select_final_calibration_params.py first."
        )
    with open(report_path, "r") as f:
        selected_seed = int(json.load(f)["selected_seed"])
    b = pd.read_csv(b_path)
    s = pd.read_csv(s_path)
    b_fold = int(b.loc[b["seed"] == selected_seed, "best_from_fold"].iloc[0])
    s_fold = int(s.loc[s["seed"] == selected_seed, "best_from_fold"].iloc[0])
    return selected_seed, b_fold, s_fold


def main() -> None:
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    baseline_dir = project_root / "result" / "calibration"
    spore_dir = project_root / "result" / "calibration_spore_infthr"
    sum_dir = project_root / "result" / "calibration_sum"
    fig_dir = project_root / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bdf = _load_fold_best_rows(baseline_dir, "baseline")
    sdf = _load_fold_best_rows(spore_dir, "spore")
    selected_seed, selected_b_fold, selected_s_fold = _load_selected(sum_dir)

    if bdf.empty or sdf.empty:
        raise ValueError("No fold-best rows found.")

    merged = bdf.merge(
        sdf,
        on=["seed", "fold"],
        suffixes=("_baseline", "_spore"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping seed/fold pairs between baseline and spore.")

    seeds = sorted(merged["seed"].unique().tolist())
    folds = [1, 2, 3]

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, axes = plt.subplots(3, 3, figsize=(7.08, 7.4), sharey=True)

    # unified y-scale
    y_max = float(
        max(
            merged["train_mae_baseline"].max(),
            merged["val_mae_baseline"].max(),
            merged["train_mae_spore"].max(),
            merged["val_mae_spore"].max(),
        )
    )
    y_max *= 1.35

    for r, seed in enumerate(seeds):
        for c, fold in enumerate(folds):
            ax = axes[r, c]
            sub = merged[(merged["seed"] == seed) & (merged["fold"] == fold)]
            if sub.empty:
                ax.axis("off")
                continue
            row = sub.iloc[0]

            x = [0.0, 1.0]  # 0=train, 1=val
            width = 0.34
            b_vals = [row["train_mae_baseline"], row["val_mae_baseline"]]
            s_vals = [row["train_mae_spore"], row["val_mae_spore"]]
            b_err = [row["train_std_baseline"], row["val_std_baseline"]]
            s_err = [row["train_std_spore"], row["val_std_spore"]]

            ax.bar(
                [i - width / 2 for i in x],
                b_vals,
                yerr=b_err,
                width=width,
                color="#4C78A8",
                edgecolor="black",
                linewidth=0.5,
                ecolor="#666666",
                capsize=2,
                alpha=0.9,
                label="FIM" if (r == 0 and c == 0) else None,
            )
            ax.bar(
                [i + width / 2 for i in x],
                s_vals,
                yerr=s_err,
                width=width,
                color="#F28E2B",
                edgecolor="black",
                linewidth=0.5,
                ecolor="#666666",
                capsize=2,
                alpha=0.9,
                label="FIM-S" if (r == 0 and c == 0) else None,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(["Train", "Val"])
            ax.set_ylim(0, y_max)
            ax.tick_params(axis="both", direction="in")
            ax.grid(axis="y", linestyle="--", color="#DDDDDD", linewidth=0.6)
            ax.set_title(f"S{seed}-F{fold}", fontsize=10, pad=3)

            # highlight selected combos with panel border color
            if seed == selected_seed and fold == selected_b_fold:
                for spine in ax.spines.values():
                    spine.set_color("#1F4E79")
                    spine.set_linewidth(1.8)
            if seed == selected_seed and fold == selected_s_fold:
                for spine in ax.spines.values():
                    spine.set_color("#C55A11")
                    spine.set_linewidth(1.8)

            # val reduction label
            if row["val_mae_baseline"] > 0:
                pct = (row["val_mae_baseline"] - row["val_mae_spore"]) / row["val_mae_baseline"] * 100.0
                ax.text(
                    1.0,
                    max(row["val_mae_baseline"], row["val_mae_spore"]) * 1.05,
                    f"↓ {pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontfamily="Times New Roman",
                )

    fig.text(0.5, 0.02, "Metric Split", ha="center", fontsize=12)
    fig.text(0.02, 0.5, "Error (days)", va="center", rotation="vertical", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.95])

    # out = fig_dir / "bayes_candidates_3x3_seed_fold.png"
    # fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

