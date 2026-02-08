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


def main() -> None:
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    sum_dir = project_root / "result" / "calibration_sum"
    fig_dir = project_root / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_pair(sum_dir)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, ax = plt.subplots(figsize=(3.54, 3.0))  # single-column style

    x = list(range(len(df)))
    width = 0.34

    bars_b = ax.bar(
        [i - width / 2 for i in x],
        df["global_mae_baseline"],
        width=width,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        label="FIM",
    )
    bars_s = ax.bar(
        [i + width / 2 for i in x],
        df["global_mae_spore"],
        width=width,
        color="#F28E2B",
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

    y_top = max(df["global_mae_baseline"].max(), df["global_mae_spore"].max())
    ax.set_ylim(0, y_top * 1.35)

    for i, (bb, ss) in enumerate(zip(bars_b, bars_s)):
        b_val = float(df.loc[i, "global_mae_baseline"])
        s_val = float(df.loc[i, "global_mae_spore"])
        ax.text(
            bb.get_x() + bb.get_width() / 2,
            b_val * 0.5,
            f"{b_val:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="Times New Roman",
        )
        ax.text(
            ss.get_x() + ss.get_width() / 2,
            s_val * 0.5,
            f"{s_val:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="Times New Roman",
        )
        if b_val > 0:
            pct = (b_val - s_val) / b_val * 100.0
            y = max(b_val, s_val) * 1.10
            ax.text(
                i + width / 2,
                y,
                f"↓ {pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontfamily="Times New Roman",
            )

    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()

    # out = fig_dir / "bayes_global_mae_seed_compare.png"
    # fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

