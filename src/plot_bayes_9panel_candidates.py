#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os    
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "actual_data.xlsx")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "calibration_spore_infthr_v2")
os.makedirs(RESULT_DIR, exist_ok=True)

from model.crop_season import CropSeasonFirst
from model.crop_season_spore_infthr import CropSeasonFirstSporeInfThr
from model.config import config as base_config
from bayes_calibrate_first_kfold_spore_infthr import (
    apply_params,
    build_spore_day_map,
    load_actual,
)


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


def _calc_error(actual_date, event_df: pd.DataFrame) -> float:
    if event_df is None or event_df.empty:
        return np.nan
    first = event_df.iloc[0]
    doy_actual = pd.to_datetime(actual_date).dayofyear
    doy_start = pd.to_datetime(first["oil_spot_start"]).dayofyear
    doy_end = pd.to_datetime(first["oil_spot_end"]).dayofyear
    if doy_end < doy_actual:
        return float(doy_end - doy_actual)
    if doy_start > doy_actual:
        return float(doy_start - doy_actual)
    return 0.0


def _global_mae_for_params(params: dict, rows, spore_day_map: dict | None, use_spore: bool) -> float:
    errors = []
    data_dir = Path(__file__).resolve().parents[1] / "data"
    penalty = 120.0
    cfg = apply_params(base_config, params)
    for row in rows:
        rb_path = data_dir / f"{row.plot}_{row.variety}_{row.first_infection_date.year}_grape_rb.json"
        if not rb_path.exists():
            continue
        try:
            with open(rb_path, "r") as f:
                rb = json.load(f)
            if use_spore:
                spore_day = spore_day_map.get((str(row.plot), int(row.first_infection_date.year)))
                cs = CropSeasonFirstSporeInfThr(csrb=rb, config=cfg, spore_day=spore_day)
            else:
                cs = CropSeasonFirst(csrb=rb, config=cfg)
            cs.simulate()
            err = _calc_error(row.first_infection_date, cs.primary_chain_unique)
        except Exception:
            err = np.nan
        if np.isfinite(err):
            errors.append(float(err))
        else:
            errors.append(penalty)
    if not errors:
        return penalty
    return float(np.abs(np.array(errors, dtype=float)).mean())


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

    # Compute global_mae for all 9 fold-best combinations (both models)
    actual_df = load_actual()
    rows = list(actual_df.itertuples(index=False))
    spore_day_map = build_spore_day_map(actual_df)

    b_params_map = {}
    s_params_map = {}
    for seed_dir in sorted(baseline_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        for fold_dir in sorted(seed_dir.glob("baseline_fold_*")):
            fold = int(fold_dir.name.split("_")[-1])
            p = fold_dir / "baseline_kfold_best_params.json"
            if p.exists():
                with open(p, "r") as f:
                    b_params_map[(seed, fold)] = json.load(f).get("params", {})
    for seed_dir in sorted(spore_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        for fold_dir in sorted(seed_dir.glob("baseline_fold_*")):
            fold = int(fold_dir.name.split("_")[-1])
            p = fold_dir / "baseline_kfold_best_params.json"
            if p.exists():
                with open(p, "r") as f:
                    s_params_map[(seed, fold)] = json.load(f).get("params", {})

    g_b, g_s = [], []
    for _, r0 in merged.iterrows():
        k = (int(r0["seed"]), int(r0["fold"]))
        g_b.append(_global_mae_for_params(b_params_map.get(k, {}), rows, None, use_spore=False))
        g_s.append(_global_mae_for_params(s_params_map.get(k, {}), rows, spore_day_map, use_spore=True))
    merged["global_mae_baseline"] = g_b
    merged["global_mae_spore"] = g_s

    seeds = sorted(merged["seed"].unique().tolist())
    folds = [1, 2, 3]

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, axes = plt.subplots(3, 3, figsize=(7.08, 7.4), sharey=True)

    # unified y-scale
    y_max = float(
        max(
            merged["train_mae_baseline"].max(),
            merged["val_mae_baseline"].max(),
            merged["global_mae_baseline"].max(),
            merged["train_mae_spore"].max(),
            merged["val_mae_spore"].max(),
            merged["global_mae_spore"].max(),
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

            x = [0.0, 1.0, 2.0]  # 0=train, 1=val, 2=global
            width = 0.28
            b_vals = [row["train_mae_baseline"], row["val_mae_baseline"], row["global_mae_baseline"]]
            s_vals = [row["train_mae_spore"], row["val_mae_spore"], row["global_mae_spore"]]
            b_err = [row["train_std_baseline"], row["val_std_baseline"]]
            s_err = [row["train_std_spore"], row["val_std_spore"]]

            ax.bar(
                [i - width / 2 for i in x],
                b_vals,
                yerr=[b_err[0], b_err[1], 0.0],
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
                yerr=[s_err[0], s_err[1], 0.0],
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
            ax.set_xticklabels(["Train", "Val", "Global"])
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
