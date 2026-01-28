import os
import pandas as pd
import matplotlib.pyplot as plt


def _load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path)


def _plot_one(ax, df: pd.DataFrame, title: str):
    ax.barh(df["Parameter"], df["mu_star"])
    ax.set_title(title)
    ax.set_xlabel("mu_star (mean)")
    ax.invert_yaxis()


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

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    _plot_one(axes[0], mae_df, "Morris N=512 MAE (st)")
    _plot_one(axes[1], signed_df, "Morris N=512 Signed (st)")
    plt.tight_layout()

    # out_path = os.path.join(fig_dir, "morris_first_inf_st_N512_agg.png")
    # plt.savefig(out_path, dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
