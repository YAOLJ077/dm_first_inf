import pandas as pd
from pathlib import Path

# =====================================================
# 路径
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEATHER_DIR = DATA_DIR / "weather_gx"
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)

BBCH_FILE = DATA_DIR / "bbch_year.xlsx"

WEATHER_FILES = {
    "guilin": WEATHER_DIR / "N3024_guilin.csv",
    "mingyang": WEATHER_DIR / "N1730_mingyang.csv",
    "duan": WEATHER_DIR / "N1115_duan.csv",
}

# =====================================================
# 参数
# =====================================================
TBASE = 10.0
RGDD_WINDOW = 7
LOOKBACK_DAYS = 14
LAST_DAYS = 7

# =====================================================
# 工具函数
# =====================================================
def load_weather_daily(fp):
    df = pd.read_csv(fp)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["date"] = df["Datetime"].dt.normalize()
    return (
        df.groupby("date", as_index=False)
          .agg(tmean=("TEM", "mean"))
    )

def add_rgdd(df):
    df = df.sort_values("date").copy()
    df["gdd"] = (df["tmean"] - TBASE).clip(lower=0)
    df["rgdd_7"] = df["gdd"].rolling(RGDD_WINDOW, min_periods=RGDD_WINDOW).sum()
    return df

# =====================================================
# 分类规则（核心）
# =====================================================
def classify_pattern(row):
    bbch = row["rgdd_7_bbch"]
    min7 = row["rgdd_7_min_last_7d"]
    delta = abs(row["rgdd_7_delta_prev"])

    # A: 稳定型
    if (min7 >= 0.8 * bbch) and (delta <= 0.2 * bbch):
        return "A_stable"

    # B: 跃迁型
    if (min7 < 0.6 * bbch) and (row["rgdd_7_delta_prev"] > 0.3 * bbch):
        return "B_jump"

    # C: 过渡型
    return "C_intermediate"

# =====================================================
# 主流程
# =====================================================
bbch = pd.read_excel(BBCH_FILE)
bbch["date"] = pd.to_datetime(bbch["date"])
bbch["year"] = bbch["date"].dt.year

records = []

for _, r in bbch.iterrows():
    site = r["plot"]
    year = r["year"]
    bbch_stage = r["bbch"]
    bbch_date = r["date"].normalize()

    weather_fp = WEATHER_FILES.get(site)
    if weather_fp is None:
        continue

    wd = load_weather_daily(weather_fp)
    wd = wd[wd["date"].dt.year == year].copy()
    wd = add_rgdd(wd)

    prev = wd[
        (wd["date"] <= bbch_date) &
        (wd["date"] >= bbch_date - pd.Timedelta(days=LOOKBACK_DAYS))
    ].copy()

    if prev.empty or prev["rgdd_7"].isna().all():
        continue

    last = prev.tail(LAST_DAYS)
    rgdd_bbch = prev.loc[prev["date"] == bbch_date, "rgdd_7"].iloc[0]

    records.append({
        "plot": site,
        "year": year,
        "bbch": bbch_stage,
        "bbch_date": bbch_date.date(),
        "rgdd_7_bbch": rgdd_bbch,
        "rgdd_7_min_last_7d": last["rgdd_7"].min(),
        "rgdd_7_mean_last_7d": last["rgdd_7"].mean(),
        "rgdd_7_min_prev": prev["rgdd_7"].min(),
        "rgdd_7_mean_prev": prev["rgdd_7"].mean(),
        "rgdd_7_std_prev": prev["rgdd_7"].std(),
        "rgdd_7_delta_prev": rgdd_bbch - prev["rgdd_7"].iloc[0],
    })

out = pd.DataFrame(records)

# 应用分类
out["thermal_pattern"] = out.apply(classify_pattern, axis=1)

# 排序输出
out = out[
    [
        "plot", "year", "bbch", "bbch_date",
        "rgdd_7_bbch",
        "rgdd_7_min_last_7d", "rgdd_7_mean_last_7d",
        "rgdd_7_min_prev", "rgdd_7_mean_prev", "rgdd_7_std_prev",
        "rgdd_7_delta_prev",
        "thermal_pattern",
    ]
]

print("\n===== BBCH thermal characterisation with pattern =====")
print(out)

# 保存
out.to_csv(
    RESULT_DIR / "bbch_thermal_characterisation_with_pattern.csv",
    index=False
)
