#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_pair(sum_dir: Path) -> pd.DataFrame:
    b_path = sum_dir / "baseline_seed_best_global.csv"
    s_path = sum_dir / "spore_seed_best_global.csv"
    if not b_path.exists():
        raise FileNotFoundError(f"Missing file: {b_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Missing file: {s_path}")

    b = pd.read_csv(b_path)[["seed", "global_mae"]].rename(columns={"global_mae": "global_mae_baseline"})
    s = pd.read_csv(s_path)[["seed", "global_mae"]].rename(columns={"global_mae": "global_mae_spore"})
    df = b.merge(s, on="seed", how="inner").sort_values("seed").reset_index(drop=True)
    if df.empty:
        raise ValueError("No overlapping seeds between baseline and spore best-global files.")
    return df


def _load_candidates(sum_dir: Path) -> tuple[pd.DataFrame, int]:
    c_path = sum_dir / "final_seed_candidates.csv"
    r_path = sum_dir / "final_selection_report.json"
    if not c_path.exists() or not r_path.exists():
        raise FileNotFoundError("Missing final_seed_candidates.csv or final_selection_report.json. Run select_final_calibration_params.py first.")
    c = pd.read_csv(c_path).sort_values("seed").reset_index(drop=True)
    selected_seed = int(pd.read_json(r_path, typ="series")["selected_seed"])
    return c, selected_seed


def main() -> None:
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    sum_dir = project_root / "result" / "calibration_sum"
    fig_dir = project_root / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_pair(sum_dir)
    cand, selected_seed = _load_candidates(sum_dir)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.0))  # double-column

    # (a) Global MAE comparison across all seed candidates
    ax = axes[0]
    x = list(range(len(df)))
    width = 0.34
    selected_idx = df.index[df["seed"] == selected_seed]
    selected_idx = int(selected_idx[0]) if len(selected_idx) else -1

    colors_b = ["#4C78A8"] * len(df)
    colors_s = ["#F28E2B"] * len(df)
    if selected_idx >= 0:
        colors_b[selected_idx] = "#1F4E79"
        colors_s[selected_idx] = "#C55A11"

    bars_b = ax.bar(
        [i - width / 2 for i in x],
        df["global_mae_baseline"],
        width=width,
        color=colors_b,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        label="FIM",
    )
    bars_s = ax.bar(
        [i + width / 2 for i in x],
        df["global_mae_spore"],
        width=width,
        color=colors_s,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        label="FIM-S",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in df["seed"]])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Global MAE (days)")
    ax.tick_params(axis="both", direction="in")
    ax.grid(axis="y", linestyle="--", color="#DDDDDD", linewidth=0.6)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top")
    y_top = max(df["global_mae_baseline"].max(), df["global_mae_spore"].max())
    ax.set_ylim(0, y_top * 1.35)
    if selected_idx >= 0:
        ax.text(selected_idx, y_top * 1.28, "selected", ha="center", va="top", fontsize=10, fontfamily="Times New Roman")

    for i, (bb, ss) in enumerate(zip(bars_b, bars_s)):
        b_val = float(df.loc[i, "global_mae_baseline"])
        s_val = float(df.loc[i, "global_mae_spore"])
        if b_val > 0:
            pct = (b_val - s_val) / b_val * 100.0
            ax.text(i + width / 2, max(b_val, s_val) * 1.08, f"↓ {pct:.1f}%", ha="center", va="bottom")
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    # (b) Candidate selection basis (spore validation mean ± std and global_mae)
    ax2 = axes[1]
    c = cand.copy()
    c["seed"] = c["seed"].astype(int)
    c = c.sort_values("seed").reset_index(drop=True)
    x2 = list(range(len(c)))
    sel_mask = c["seed"] == selected_seed
    colors = ["#4C78A8" if not m else "#1F4E79" for m in sel_mask]
    ax2.bar(
        x2,
        c["val_mae_mean_spore"],
        yerr=c["val_mae_std_spore"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        alpha=0.9,
        label="val_mae_mean_spore ± std",
    )
    ax2.plot(x2, c["global_mae_spore"], color="#F28E2B", marker="o", linewidth=1.2, label="global_mae_spore")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(v) for v in c["seed"]])
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Error (days)")
    ax2.tick_params(axis="both", direction="in")
    ax2.grid(axis="y", linestyle="--", color="#DDDDDD", linewidth=0.6)
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, ha="left", va="top")
    ax2.legend(frameon=False, loc="upper right", fontsize=9)

    fig.tight_layout()
    # out = fig_dir / "bayes_global_mae_seed_compare_1x2.png"
    # fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
