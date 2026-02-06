import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


METRICS = ["train_mae", "val_mae", "train_rmse", "val_rmse"]


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for model_type, g in df.groupby("model_type"):
        row: Dict[str, float | str] = {"model_type": model_type, "n_rows": int(len(g))}
        for m in METRICS:
            if m in g.columns:
                s = pd.to_numeric(g[m], errors="coerce")
                row[f"{m}_mean"] = float(s.mean())
                row[f"{m}_median"] = float(s.median())
                row[f"{m}_std"] = float(s.std(ddof=0))
                row[f"{m}_min"] = float(s.min())
                row[f"{m}_max"] = float(s.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_delta(df: pd.DataFrame, baseline_label: str, spore_label: str) -> pd.DataFrame:
    key_cols = ["seed", "fold"]
    base = df[df["model_type"] == baseline_label].copy()
    spo = df[df["model_type"] == spore_label].copy()
    merged = base.merge(spo, on=key_cols, how="inner", suffixes=("_base", "_spore"))
    if merged.empty:
        raise ValueError("No paired rows found. Check model_type labels and fold/seed columns.")

    for m in METRICS:
        b = f"{m}_base"
        s = f"{m}_spore"
        if b in merged.columns and s in merged.columns:
            merged[f"delta_{m}"] = merged[s] - merged[b]
            merged[f"improve_abs_{m}"] = merged[b] - merged[s]
            merged[f"improve_pct_{m}"] = np.where(
                merged[b] != 0, (merged[b] - merged[s]) / merged[b] * 100.0, np.nan
            )
    return merged


def _improvement_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        col_abs = f"improve_abs_{m}"
        col_pct = f"improve_pct_{m}"
        if col_abs not in pair_df.columns:
            continue
        a = pd.to_numeric(pair_df[col_abs], errors="coerce")
        p = pd.to_numeric(pair_df[col_pct], errors="coerce")
        rows.append(
            {
                "metric": m,
                "n_pairs": int(a.notna().sum()),
                "improve_abs_mean": float(a.mean()),
                "improve_abs_median": float(a.median()),
                "improve_abs_std": float(a.std(ddof=0)),
                "improve_pct_mean": float(p.mean()),
                "improve_pct_median": float(p.median()),
                "improve_pct_std": float(p.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _significance(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        col = f"delta_{m}"  # spore - baseline
        if col not in pair_df.columns:
            continue
        d = pd.to_numeric(pair_df[col], errors="coerce").dropna()
        out = {
            "metric": m,
            "n_pairs": int(len(d)),
            "delta_mean_spore_minus_base": float(d.mean()),
            "delta_median_spore_minus_base": float(d.median()),
        }
        if wilcoxon is not None and len(d) > 0:
            try:
                w = wilcoxon(d)
                out["wilcoxon_stat"] = float(w.statistic)
                out["wilcoxon_p"] = float(w.pvalue)
            except Exception:
                out["wilcoxon_stat"] = np.nan
                out["wilcoxon_p"] = np.nan
        else:
            out["wilcoxon_stat"] = np.nan
            out["wilcoxon_p"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _effect_label(abs_d: float) -> str:
    if np.isnan(abs_d):
        return ""
    if abs_d > 1.2:
        return "** very strong"
    if abs_d >= 0.8:
        return "* strong"
    if abs_d >= 0.5:
        return "† moderate"
    return "small/none"


def _effect_sizes(
    df: pd.DataFrame, pair_df: pd.DataFrame, baseline_label: str, spore_label: str
) -> pd.DataFrame:
    rows = []
    base = df[df["model_type"] == baseline_label].copy()
    spo = df[df["model_type"] == spore_label].copy()

    for m in METRICS:
        bcol = f"{m}_base"
        scol = f"{m}_spore"
        dcol = f"improve_abs_{m}"  # baseline - spore

        # Independent-sample Cohen's d (pooled SD), for consistency with prior framework
        b = pd.to_numeric(base[m], errors="coerce").dropna() if m in base.columns else pd.Series(dtype=float)
        s = pd.to_numeric(spo[m], errors="coerce").dropna() if m in spo.columns else pd.Series(dtype=float)
        if len(b) > 1 and len(s) > 1:
            mean_diff = float(b.mean() - s.mean())
            pooled_sd = float(np.sqrt((b.std(ddof=1) ** 2 + s.std(ddof=1) ** 2) / 2.0))
            d_pooled = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
        else:
            mean_diff = np.nan
            pooled_sd = np.nan
            d_pooled = np.nan

        # Paired-sample Cohen's dz (preferred for paired folds)
        if dcol in pair_df.columns:
            d = pd.to_numeric(pair_df[dcol], errors="coerce").dropna()
            dz_sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
            d_z = float(d.mean() / dz_sd) if (len(d) > 1 and dz_sd > 0) else np.nan
            n_pairs = int(len(d))
        else:
            d_z = np.nan
            n_pairs = 0

        rows.append(
            {
                "metric": m,
                "n_baseline": int(len(b)),
                "n_spore": int(len(s)),
                "n_pairs": n_pairs,
                "mean_diff_base_minus_spore": mean_diff,
                "pooled_sd": pooled_sd,
                "cohens_d_pooled": d_pooled,
                "cohens_d_pooled_label": _effect_label(abs(d_pooled) if not np.isnan(d_pooled) else np.nan),
                "cohens_dz_paired": d_z,
                "cohens_dz_paired_label": _effect_label(abs(d_z) if not np.isnan(d_z) else np.nan),
            }
        )

    return pd.DataFrame(rows)


def _save(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    csv_path = out_dir / f"{stem}.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--long-table",
        type=str,
        default="result/bayes_sum/bayes_models_fold_long.csv",
    )
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--spore-label", type=str, default="spore_infthr")
    parser.add_argument("--out-dir", type=str, default="result/bayes_sum")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    long_table = (project_root / args.long_table).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _safe_read(long_table)
    for col in ["seed", "fold", "model_type"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {long_table}")

    # Core outputs
    model_sum = _model_summary(df)
    pair_df = _pairwise_delta(df, args.baseline_label, args.spore_label)
    improve_sum = _improvement_summary(pair_df)
    signif = _significance(pair_df)
    effect = _effect_sizes(df, pair_df, args.baseline_label, args.spore_label)

    _save(model_sum, out_dir, "bayes_model_summary")
    _save(pair_df, out_dir, "bayes_pairwise_delta")
    _save(improve_sum, out_dir, "bayes_improvement_summary")
    _save(signif, out_dir, "bayes_significance")
    _save(effect, out_dir, "bayes_effect_size")


if __name__ == "__main__":
    main()
