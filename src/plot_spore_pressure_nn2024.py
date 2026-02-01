import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "fig"


def _normalize_spore_series(spore_vals: pd.Series) -> pd.Series:
    vals = spore_vals.dropna().astype(float)
    if vals.empty:
        return pd.Series(dtype=float)
    p95 = float(vals.quantile(0.95))
    if p95 <= 0:
        return spore_vals.fillna(0.0).astype(float) * 0.0
    norm = (np.log1p(spore_vals.astype(float)) / np.log1p(p95)).clip(lower=0.0, upper=1.0)
    baseline = norm.rolling(window=7, min_periods=1, center=True).median()
    return norm - baseline


def main() -> None:
    actual_path = DATA_DIR / "actual_data.xlsx"
    spore_path = DATA_DIR / "Spore_Overall.xlsx"
    actual = pd.read_excel(actual_path)
    spore = pd.read_excel(spore_path)
    spore["date"] = pd.to_datetime(spore["date"])

    actual["first_infection_date"] = pd.to_datetime(actual["first_infection_date"], errors="coerce")
    site_years = (
        actual[["plot", "year", "first_infection_date"]]
        .dropna(subset=["plot", "year"])
        .sort_values(["plot", "year"])
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for row in site_years.itertuples(index=False):
        plot = row.plot
        year = int(row.year)
        inf_date = row.first_infection_date

        g = spore[(spore["plot"] == plot) & (spore["year"] == year)].copy()
        if g.empty:
            continue
        g = g.sort_values("date")
        obs = g[g["spore"].notna()].copy()
        if obs.empty:
            continue

        start = obs["date"].min().normalize()
        end = obs["date"].max().normalize()
        if pd.notna(inf_date):
            end = min(end, inf_date.normalize())
        if end < start:
            continue

        daily_index = pd.date_range(start=start, end=end, freq="D")
        raw_daily = pd.Series(index=daily_index, dtype=float)
        obs = obs[(obs["date"] >= start) & (obs["date"] <= end)]
        raw_daily.loc[obs["date"].dt.normalize()] = obs["spore"].values
        raw_daily = raw_daily.interpolate(method="time", limit_direction="both")

        norm_daily = (np.log1p(raw_daily) / np.log1p(max(raw_daily.quantile(0.95), 1e-9))).clip(
            lower=0.0, upper=1.0
        )
        obs_dates = obs["date"].dt.normalize().drop_duplicates().sort_values()
        if len(obs_dates) >= 2:
            gaps = obs_dates.diff().dt.days.dropna()
            median_gap = float(gaps.median()) if not gaps.empty else 3.0
        else:
            median_gap = 3.0
        window = int(max(3, min(9, round(median_gap * 2))))
        smooth_norm = norm_daily.rolling(window=window, min_periods=1, center=True).mean()
        if smooth_norm.dropna().size:
            target_pos = 0.4
            candidates = np.linspace(0.45, 0.75, 31)
            pos_fracs = [(smooth_norm > q).mean() for q in candidates]
            q_idx = int(np.argmin([abs(p - target_pos) for p in pos_fracs]))
            global_baseline = float(candidates[q_idx])
        else:
            global_baseline = 0.6
        adj_daily = smooth_norm - global_baseline

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axes[0].bar(
            obs["date"].dt.normalize(), obs["spore"].values, width=1.0, alpha=0.4, label="Observed"
        )
        axes[0].plot(raw_daily.index, raw_daily.values, color="C1", lw=1.5, label="Interpolated")
        axes[0].set_ylabel("Spore count")
        axes[0].set_title(f"{plot}-{year} Raw Spore Observations")
        axes[0].legend(loc="upper right")

        axes[1].plot(norm_daily.index, norm_daily.values, color="C0", lw=1.5)
        axes[1].set_ylabel("Normalized")
        axes[1].set_title("Normalized (log1p / p95)")

        axes[2].plot(adj_daily.index, adj_daily.values, color="C2", lw=1.5)
        axes[2].axhline(0.0, color="0.5", lw=0.8)
        axes[2].set_ylabel("Baseline-corrected")
        axes[2].set_title(f"Smoothed Normalized - Global Baseline (q≈{global_baseline:.2f})")

        if pd.notna(inf_date):
            for ax in axes:
                ax.axvline(inf_date.normalize(), color="k", ls="--", lw=1.0, alpha=0.7)
                ax.scatter(
                    [inf_date.normalize()],
                    [ax.get_ylim()[1]],
                    marker="v",
                    color="k",
                    s=30,
                    zorder=5,
                    clip_on=False,
                )

        axes[2].set_xlabel("Date")
        fig.tight_layout()

        out_path = FIG_DIR / f"spore_pressure_{plot}_{year}.png"
        fig.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
        saved += 1

    if saved == 0:
        raise ValueError("No figures were generated; check input data.")

    plt.show()


if __name__ == "__main__":
    main()
