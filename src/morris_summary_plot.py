# -*- coding: utf-8 -*-
# LEGACY / 旧版：不属于当前 FSIM 分析入口。
# Status: legacy_result_processing; see config/project_registry.json and config/README.md.
# Retained for historical traceability; current results require separate validation.
"""
Summarize and plot Morris sensitivity stability metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# path setup
# --------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "morris")
SUMMARY_DIR = os.path.join(RESULT_DIR, "summary")
SUMMARY_CSV = os.path.join(RESULT_DIR, "morris_first_inf_summary.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "fig")
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TOP_N_LIST = [12]
BOOTSTRAP_N = 500


def _spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlation between seed-wise rankings.
    Expects columns: seed, Parameter, mu_star
    """
    pivot = df.pivot_table(
        index="Parameter",
        columns="seed",
        values="mu_star",
        aggfunc="mean"
    )
    ranks = pivot.rank(ascending=False, axis=0)
    return ranks.corr(method="spearman")


def _spearman_pairwise_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Spearman correlation with bootstrap CI across seeds.
    """
    pivot = df.pivot_table(
        index="Parameter",
        columns="seed",
        values="mu_star",
        aggfunc="mean"
    )
    ranks = pivot.rank(ascending=False, axis=0)
    seeds = list(ranks.columns)
    params = ranks.index.to_list()

    rows = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            s1, s2 = seeds[i], seeds[j]
            r1 = ranks[s1].values
            r2 = ranks[s2].values
            rho = pd.Series(r1).corr(pd.Series(r2), method="spearman")

            boot = []
            for _ in range(BOOTSTRAP_N):
                idx = np.random.randint(0, len(params), len(params))
                rb1 = r1[idx]
                rb2 = r2[idx]
                boot.append(
                    pd.Series(rb1).corr(pd.Series(rb2), method="spearman")
                )
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            significant = not (ci_low <= 0 <= ci_high)

            rows.append({
                "seed_a": s1,
                "seed_b": s2,
                "rho": rho,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significant": significant
            })
    return pd.DataFrame(rows)


def _topn_overlap(df: pd.DataFrame, n: int) -> float:
    """
    Compute average pairwise Top-N overlap ratio across seeds.
    """
    top_sets = {}
    for seed, g in df.groupby("seed"):
        top_sets[seed] = set(
            g.sort_values("mu_star", ascending=False)
            .head(n)["Parameter"].tolist()
        )
    seeds = list(top_sets.keys())
    if len(seeds) < 2:
        return np.nan

    overlaps = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = top_sets[seeds[i]]
            b = top_sets[seeds[j]]
            overlaps.append(len(a & b) / float(n))
    return float(np.mean(overlaps)) if overlaps else np.nan


def _plot_heatmap(mat: pd.DataFrame, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(mat.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    # fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)


def _plot_mu_star_bar(stats: pd.DataFrame, title: str, out_path: str):
    stats = stats.sort_values("mu_star_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(stats))))
    ax.barh(
        stats["Parameter"],
        stats["mu_star_mean"],
        xerr=stats["mu_star_std"],
        color="#4C78A8",
        ecolor="#333333",
        capsize=3
    )
    ax.set_xlabel("mu_star (mean ± std)")
    ax.set_title(title)
    fig.tight_layout()
    # fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)


def _plot_signed_mu_bar(stats: pd.DataFrame, title: str, out_path: str):
    stats = stats.sort_values("mu_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(stats))))
    colors = ["#E45756" if v < 0 else "#72B7B2" for v in stats["mu_mean"]]
    ax.barh(
        stats["Parameter"],
        stats["mu_mean"],
        xerr=stats["mu_std"],
        color=colors,
        ecolor="#333333",
        capsize=3
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("mu (signed, mean ± std)")
    ax.set_title(title)
    fig.tight_layout()
    # fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    # Normalize columns
    df["metric"] = df["metric"].str.strip()

    summary_rows = []
    pairwise_rows = []

    for (n_traj, metric), g in df.groupby(["N_TRAJ", "metric"]):
        # Stability metrics
        spearman = _spearman_matrix(g)
        topn = {n: _topn_overlap(g, n) for n in TOP_N_LIST}
        pairwise = _spearman_pairwise_bootstrap(g)
        pairwise["N_TRAJ"] = n_traj
        pairwise["metric"] = metric
        pairwise_rows.append(pairwise)

        summary_rows.append({
            "N_TRAJ": n_traj,
            "metric": metric,
            "spearman_mean": float(np.nanmean(spearman.values)),
            "top10_overlap": topn.get(10, np.nan),
            "spearman_ci_low_mean": float(pairwise["ci_low"].mean()),
            "spearman_ci_high_mean": float(pairwise["ci_high"].mean()),
            "spearman_sig_rate": float(pairwise["significant"].mean())
        })

        # Plot heatmap
        heatmap_out = os.path.join(
            PLOT_DIR,
            f"spearman_heatmap_{metric}_N{n_traj}.png"
        )
        _plot_heatmap(
            spearman,
            f"Spearman (seed) - {metric} - N={n_traj}",
            heatmap_out
        )

        # Aggregate parameter stats
        stats = (
            g.groupby("Parameter")
            .agg(
                mu_mean=("mu", "mean"),
                mu_std=("mu", "std"),
                mu_star_mean=("mu_star", "mean"),
                mu_star_std=("mu_star", "std"),
            )
            .reset_index()
        )

        # Plot bars
        mu_star_out = os.path.join(
            PLOT_DIR,
            f"mu_star_{metric}_N{n_traj}.png"
        )
        _plot_mu_star_bar(
            stats,
            f"mu_star (mean ± std) - {metric} - N={n_traj}",
            mu_star_out
        )

        if metric.lower() == "signed":
            mu_out = os.path.join(
                PLOT_DIR,
                f"mu_signed_{metric}_N{n_traj}.png"
            )
            _plot_signed_mu_bar(
                stats,
                f"mu (signed, mean ± std) - {metric} - N={n_traj}",
                mu_out
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(SUMMARY_DIR, "morris_stability_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    if pairwise_rows:
        pairwise_df = pd.concat(pairwise_rows, ignore_index=True)
        pairwise_out = os.path.join(SUMMARY_DIR, "morris_spearman_pairs.csv")
        pairwise_df.to_csv(pairwise_out, index=False)
        print(f"Saved: {pairwise_out}")

    print(f"Saved: {summary_out}")
    # print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
