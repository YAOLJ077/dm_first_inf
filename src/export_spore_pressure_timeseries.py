from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def build_timeseries() -> pd.DataFrame:
    actual_path = DATA_DIR / "actual_data.xlsx"
    spore_path = DATA_DIR / "Spore_Overall.xlsx"
    actual = pd.read_excel(actual_path)
    spore = pd.read_excel(spore_path)
    spore["date"] = pd.to_datetime(spore["date"])
    actual["first_infection_date"] = pd.to_datetime(actual["first_infection_date"], errors="coerce")

    rows = []
    for row in actual.itertuples(index=False):
        plot = str(row.plot)
        year = int(row.year)
        inf_date = row.first_infection_date
        if pd.isna(inf_date):
            continue

        g = spore[(spore["plot"] == plot) & (spore["year"] == year)].copy()
        if g.empty:
            continue
        g = g.sort_values("date")
        obs = g[g["spore"].notna()].copy()
        if obs.empty:
            continue

        start = obs["date"].min().normalize()
        end = obs["date"].max().normalize()
        end = min(end, inf_date.normalize())
        if end < start:
            continue

        daily_index = pd.date_range(start=start, end=end, freq="D")
        raw_daily = pd.Series(index=daily_index, dtype=float)
        obs = obs[(obs["date"] >= start) & (obs["date"] <= end)]
        raw_daily.loc[obs["date"].dt.normalize()] = obs["spore"].values
        raw_daily = raw_daily.interpolate(method="time", limit_direction="both")

        p95 = float(raw_daily.dropna().quantile(0.95)) if raw_daily.dropna().size else 0.0
        if p95 <= 0:
            norm_daily = raw_daily.fillna(0.0) * 0.0
        else:
            norm_daily = (np.log1p(raw_daily) / np.log1p(p95)).clip(lower=0.0, upper=1.0)

        obs_dates = obs["date"].dt.normalize().drop_duplicates().sort_values()
        if len(obs_dates) >= 2:
            gaps = obs_dates.diff().dt.days.dropna()
            median_gap = float(gaps.median()) if not gaps.empty else 3.0
        else:
            median_gap = 3.0
        window = int(max(3, min(9, round(median_gap * 2))))

        rolling_norm = norm_daily.rolling(window=window, min_periods=1, center=True).mean()

        if rolling_norm.dropna().size:
            target_pos = 0.4
            candidates = np.linspace(0.45, 0.75, 31)
            pos_fracs = [(rolling_norm > q).mean() for q in candidates]
            q_idx = int(np.argmin([abs(p - target_pos) for p in pos_fracs]))
            global_baseline = float(candidates[q_idx])
        else:
            global_baseline = 0.6

        s_day = norm_daily - global_baseline
        rolling_s_day = rolling_norm - global_baseline

        for dt in daily_index:
            rows.append(
                {
                    "site": plot,
                    "year": year,
                    "date": dt,
                    "count_spore": float(raw_daily.loc[dt]),
                    "S_day": float(s_day.loc[dt]),
                    "Rolling_S_day": float(rolling_s_day.loc[dt]),
                    "infection": int(dt.normalize() == inf_date.normalize()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    df = build_timeseries()
    if df.empty:
        raise ValueError("No rows generated; check input data.")
    out_xlsx = DATA_DIR / "spore_pressure_timeseries.xlsx"
    out_csv = DATA_DIR / "spore_pressure_timeseries.csv"
    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_xlsx}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
