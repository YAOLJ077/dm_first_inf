#!/usr/bin/env python3
"""Plot final 2p/3p/4p TPE held-out bias in three rows and two columns."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_data(results, seed, arm):
    metadata = json.loads((results / "run_metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Calibration run is not marked complete")
    folds = pd.read_csv(results / "fold_summary.csv", dtype={"held_out": str})
    oof = pd.read_csv(results / "oof_predictions.csv", dtype={"held_out": str})
    folds = folds.loc[(folds.seed == seed) & folds.view.eq("final") & folds.arm.eq(arm)].copy()
    oof = oof.loc[(oof.seed == seed) & oof.view.eq("final") & oof.arm.eq(arm)].copy()
    if len(folds) != 6 or len(oof) != 12:
        raise ValueError(f"Expected six fold rows and twelve site-year predictions; got {len(folds)}, {len(oof)}")
    rows = []
    for scheme in ("site", "year"):
        subset = folds.loc[folds.scheme.eq(scheme)]
        if len(subset) != 3:
            raise ValueError(f"Expected three {scheme} folds")
        expected = {"DA", "GL", "NN"} if scheme == "site" else {"2022", "2023", "2024"}
        if set(subset.held_out) != expected:
            raise ValueError(f"Unexpected folds for {arm}/{scheme}")
        held = oof.loc[oof.scheme.eq(scheme)]
        if len(held) != 6 or held.input_key.nunique() != 6:
            raise ValueError(f"Expected six unique cases for {arm}/{scheme}")
        for row in subset.itertuples(index=False):
            cases = oof.loc[oof.scheme.eq(scheme) & oof.held_out.eq(row.held_out)]
            predicted = cases.loc[cases.status.eq("predicted")]
            errors = pd.to_numeric(predicted.signed_interval_error_days, errors="coerce")
            if not np.isfinite(errors).all():
                raise ValueError(f"Nonfinite predicted error: {arm}/{scheme}/{row.held_out}")
            rows.append({
                "arm": arm,
                "scheme": scheme, "fold": row.held_out, "trials": int(row.n_trials),
                "training_MAE_days": float(row.training_MAE_days),
                "n_predictions": int(len(predicted)), "n_test": int(len(cases)),
                "bias_days": float(errors.mean()) if len(errors) else np.nan,
                "absolute_error_days": float(errors.abs().mean()) if len(errors) else np.nan,
                "case_biases": "; ".join(f"{x.input_key}={x.signed_interval_error_days:+.2f} d"
                                           for x in predicted.itertuples(index=False)),
                "missing_cases": "; ".join(cases.loc[~cases.status.eq("predicted"), "input_key"]),
            })
    return pd.DataFrame(rows)


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 12))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2, sharey=True)
    colors = {"site": "#397A9E", "year": "#D88739"}
    finite = data.loc[np.isfinite(data.bias_days), "bias_days"]
    lower = min(-75, float(finite.min()) - 15) if len(finite) else -75
    upper = max(105, float(finite.max()) + 15) if len(finite) else 105
    for panel, ax in enumerate(axes.flat):
        row_index, col_index = divmod(panel, 2)
        arm = ("2p_tpe", "3p_tpe", "4p_tpe")[row_index]
        scheme = ("site", "year")[col_index]
        d = data.loc[data.scheme.eq(scheme) & data.arm.eq(arm)].sort_values("fold").reset_index(drop=True)
        x = np.arange(len(d))
        values = d.bias_days.to_numpy(dtype=float)
        valid = np.isfinite(values)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.bar(x[valid], values[valid], color=colors[scheme], width=0.62, zorder=3)
        for i, value in zip(x[valid], values[valid]):
            ax.text(i, value + (2 if value >= 0 else -2), f"{value:+.1f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
        for i in x[~valid]:
            ax.scatter(i, 0, marker="x", s=70, color="#A13D32", zorder=4)
            ax.text(i, 2, "NA\nno window", ha="center", va="bottom", color="#A13D32", fontsize=9)
        labels = [f"{fold}\n({n}/{total})" for fold, n, total in zip(d.fold, d.n_predictions, d.n_test)]
        ax.set(xticks=x, xticklabels=labels,
               title=f"({chr(97 + panel)}) {arm[:2]} TPE — {'Site' if scheme == 'site' else 'Year'} leave-one-out",
               xlabel="Held-out fold" if row_index == 2 else "", ylim=(lower, upper))
        if col_index == 0:
            ax.set_ylabel("Signed bias (days)")
        ax.grid(axis="y", color="#E3E3E3", zorder=0)
    fig.suptitle("TPE: per-fold held-out bias", y=0.98, fontsize=16)
    budgets = sorted(data.trials.unique())
    budget_text = str(budgets[0]) if len(budgets) == 1 else f"{budgets[0]}–{budgets[-1]}"
    fig.text(0.08, 0.025,
             "Bars: mean signed interval error among predicted site-years in each fold.\n"
             "Positive = predicted later; negative = predicted earlier. Parentheses: predicted/test cases.\n"
             f"Missing predictions are excluded, never assigned zero bias. Final selections; {budget_text} trials per fold.\n"
             "2p: VPD/I; 3p: VPD/I/wind (RH=80, MP=3); 4p: VPD/I/RH/MP.", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.15, wspace=0.12, hspace=0.39)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    fig.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "result/fsim_unified_joblib_local")
    parser.add_argument("--results-3p", type=Path, default=ROOT / "result/fsim_m0_3p_joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "fig/fsim_2p_tpe_fold_bias")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    data = pd.concat([load_data(args.results_3p if arm == "3p_tpe" else args.results, args.seed, arm)
                      for arm in ("2p_tpe", "3p_tpe", "4p_tpe")], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(data.to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="NA"), flush=True)
    data.to_csv(args.output.with_name(args.output.name + "_data.csv"), index=False)
    plot(data, args.output)
    print(f"\n已输出: {args.output}.png / {args.output}.pdf / {args.output.name}_data.csv")


if __name__ == "__main__":
    main()
