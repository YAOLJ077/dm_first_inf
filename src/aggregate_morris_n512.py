import os
import glob
import pandas as pd


def _load_one(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["__source__"] = os.path.basename(path)
    return df


def _aggregate(label: str, result_dir: str) -> pd.DataFrame:
    pattern = os.path.join(
        result_dir,
        f"morris_first_inf_ONLY_{label}_N512_seed*.xlsx"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    dfs = [_load_one(p) for p in files]
    all_df = pd.concat(dfs, ignore_index=True)

    grouped = all_df.groupby("Parameter", as_index=False)[["mu", "mu_star", "sigma"]]
    mean_df = grouped.mean()
    std_df = grouped.std().rename(columns={
        "mu": "mu_std",
        "mu_star": "mu_star_std",
        "sigma": "sigma_std",
    })

    agg = (
        mean_df.merge(std_df, on="Parameter", how="left")
        .sort_values("mu_star", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    result_dir = os.path.join(project_root, "result", "morris")

    os.makedirs(result_dir, exist_ok=True)

    for label in ["MAE", "Signed"]:
        agg = _aggregate(label, result_dir)
        out = os.path.join(
            result_dir,
            f"morris_first_inf_st_ONLY_{label}_N512_agg.xlsx"
        )
        agg.to_excel(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
