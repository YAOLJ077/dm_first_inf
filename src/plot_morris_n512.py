import os
import pandas as pd
import matplotlib.pyplot as plt


PARAM_ABBR = {
    "dispersal__rain_min": "DIRM",
    "infection__infection_condition": "IC",
    "zoospore_alive__vpd": "ZVPD",
    "mp_tmp__rh_wet_night": "MTWN",
    "sporulation__Tmax": "STMX",
    "dispersal__wind_min": "DIWM",
    "sporulation__Topt": "STOP",
    "sporulation__MP_min": "SMMI",
    "deposition__rain_min": "DERM",
    "sporulation__Tmin": "STMI",
    "incubation__spl_end": "ISED",
    "zoospore_alive__rain": "ZARA",
    "zoospore_alibe__rain": "ZARA",
    "thresholds__min_spores": "TMSO",
}


def _load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path)


def _apply_abbr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abbr"] = out["Parameter"].map(PARAM_ABBR).fillna(out["Parameter"])
    return out


def _plot_mae(df: pd.DataFrame):
    df = _apply_abbr(df).sort_values("mu_star", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(
        df["abbr"],
        df["mu_star"],
        xerr=df.get("mu_star_std"),
        color="#4C78A8",
        ecolor="#333333",
        capsize=3
    )
    ax.invert_yaxis()
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_xlabel("mu_star (mean)")
    ax.set_title("Morris N=512 MAE (st)")
    ax.tick_params(axis="both", direction="in")
    fig.tight_layout()
    # out_path = os.path.join(fig_dir, "morris_first_inf_st_N512_MAE.png")
    # fig.savefig(out_path, dpi=300)
    return fig


def _plot_signed(df: pd.DataFrame):
    df = _apply_abbr(df).sort_values("mu", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#E45756" if v < 0 else "#72B7B2" for v in df["mu"]]
    ax.barh(
        df["abbr"],
        df["mu"],
        xerr=df.get("mu_std"),
        color=colors,
        ecolor="#333333",
        capsize=3
    )
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_xlabel("mu (signed mean)")
    ax.set_title("Morris N=512 Signed (st)")
    ax.tick_params(axis="both", direction="in")
    fig.tight_layout()
    # out_path = os.path.join(fig_dir, "morris_first_inf_st_N512_Signed.png")
    # fig.savefig(out_path, dpi=300)
    return fig


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    result_dir = os.path.join(project_root, "result", "morris")
    fig_dir = os.path.join(project_root, "fig")
    os.makedirs(fig_dir, exist_ok=True)

    mae_path = os.path.join(
        result_dir,
        "morris_first_inf_st_ONLY_MAE_N512_agg.xlsx"
    )
    signed_path = os.path.join(
        result_dir,
        "morris_first_inf_st_ONLY_Signed_N512_agg.xlsx"
    )

    mae_df = _load(mae_path)
    signed_df = _load(signed_path)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
    })

    fig1 = _plot_mae(mae_df)
    plt.show()
    plt.close(fig1)

    fig2 = _plot_signed(signed_df)
    plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    main()
