# LEGACY / 旧版：不属于当前 FSIM 分析入口。
# Status: legacy_result_processing; see config/project_registry.json and config/README.md.
# Retained for historical traceability; current results require separate validation.
import glob
import os
import re
from typing import Dict, List

import pandas as pd


def _load_one(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["__source__"] = os.path.basename(path)
    return df


def _aggregate(files: List[str]) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError("No Morris files found for aggregation.")
    all_df = pd.concat([_load_one(p) for p in files], ignore_index=True)
    grouped = all_df.groupby("Parameter", as_index=False)[["mu", "mu_star", "sigma"]]
    mean_df = grouped.mean()
    std_df = grouped.std(ddof=0).rename(
        columns={"mu": "mu_std", "mu_star": "mu_star_std", "sigma": "sigma_std"}
    )
    return (
        mean_df.merge(std_df, on="Parameter", how="left")
        .sort_values("mu_star", ascending=False)
        .reset_index(drop=True)
    )


def _strict_seed_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    # Keep only canonical seed files, exclude variants like *_st.xlsx
    return [f for f in files if re.search(r"_seed\d+\.xlsx$", os.path.basename(f))]


def _collect_patterns(project_root: str) -> Dict[str, Dict[str, str]]:
    return {
        "baseline": {
            "MAE": os.path.join(
                project_root, "result", "morris", "morris_first_inf_ONLY_MAE_N512_seed*.xlsx"
            ),
            "Signed": os.path.join(
                project_root, "result", "morris", "morris_first_inf_ONLY_Signed_N512_seed*.xlsx"
            ),
        },
        "spore_infthr_v2": {
            "MAE": os.path.join(
                project_root,
                "result",
                "morris_spore_infthr_v2",
                "morris_first_inf_spore_infthr_ONLY_MAE_N512_seed*.xlsx",
            ),
            "Signed": os.path.join(
                project_root,
                "result",
                "morris_spore_infthr_v2",
                "morris_first_inf_spore_infthr_ONLY_Signed_N512_seed*.xlsx",
            ),
        },
    }


def main() -> None:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    out_dir = os.path.join(project_root, "result", "morris_sum")
    os.makedirs(out_dir, exist_ok=True)

    patterns = _collect_patterns(project_root)
    all_rows = []

    for model_type, by_metric in patterns.items():
        for metric, pattern in by_metric.items():
            files = _strict_seed_files(pattern)
            if not files:
                raise FileNotFoundError(f"No files matched: {pattern}")
            agg = _aggregate(files)
            out_xlsx = os.path.join(
                out_dir, f"morris_{model_type}_{metric}_N512_agg.xlsx"
            )
            agg.to_excel(out_xlsx, index=False)
            print(f"Saved: {out_xlsx}")

            tmp = agg.copy()
            tmp["model_type"] = model_type
            tmp["metric"] = metric
            all_rows.append(tmp)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_out = os.path.join(out_dir, "morris_models_N512_agg_long.csv")
    combined.to_csv(combined_out, index=False)
    print(f"Saved: {combined_out}")


if __name__ == "__main__":
    main()
