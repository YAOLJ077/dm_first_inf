#!/usr/bin/env python3
"""Plot training and held-out FSIM results without modifying calibration outputs."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("2p_grid", "2p_tpe", "3p_grid", "3p_tpe", "4p_grid", "4p_tpe")
FOLDS = {"site": ("DA", "GL", "NN"), "year": ("2022", "2023", "2024")}


def error_summary(rows):
    errors = pd.to_numeric(rows.signed_interval_error_days, errors="coerce").abs()
    valid = rows.status.eq("predicted") & np.isfinite(errors)
    n = int(valid.sum())
    conditional = float(errors[valid].mean()) if n else float("nan")
    return dict(n_predictions=n, n_test=len(rows),
                MAE_days=conditional if n == len(rows) else float("nan"),
                conditional_MAE_days=conditional)


def load_results(directory, seed, arms):
    metadata = json.loads((directory / "run_metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Calibration is not marked complete; wait for completion.")
    frames = []
    for name in ("fold_summary", "oof_predictions", "method_summary"):
        frame = pd.read_csv(directory / f"{name}.csv", dtype={"held_out": str})
        frames.append(frame.loc[frame.seed.eq(seed) & frame.view.eq("final")].copy())
    folds, predictions, methods = frames
    summaries, reports = [], []
    for scheme, expected_folds in FOLDS.items():
        for arm in arms:
            selected = folds.loc[folds.scheme.eq(scheme) & folds.arm.eq(arm)]
            oof = predictions.loc[predictions.scheme.eq(scheme) & predictions.arm.eq(arm)]
            if len(selected) != 3 or set(selected.held_out) != set(expected_folds):
                raise ValueError(f"Missing or duplicate folds: {scheme}/{arm}")
            if len(oof) != 6 or oof.input_key.nunique() != 6:
                raise ValueError(f"Expected six unique held-out site-years: {scheme}/{arm}")
            training_values, training_weights = [], []
            for fold in expected_folds:
                row = selected.loc[selected.held_out.eq(fold)].iloc[0]
                job_id = f"{scheme}/{arm}/seed_{seed}/folds/{fold}"
                job = metadata["jobs"][job_id]
                count = int(row.n_trials)
                trials = pd.read_csv(directory / job_id / "trials.csv")
                if (job["status"] != "complete" or job["n_trials"] != count
                        or len(trials) != count or trials.trial.tolist() != list(range(count))):
                    raise ValueError(f"Incomplete trial sequence: {job_id}")
                held = oof.loc[oof.held_out.eq(fold)]
                test_keys = set(ast.literal_eval(row.test_keys))
                training_keys = set(ast.literal_eval(row.training_keys))
                if set(held.input_key) != test_keys or training_keys & test_keys:
                    raise ValueError(f"Invalid held-out partition: {job_id}")
                if row.calibration_status == "selected":
                    chosen = trials.loc[trials.trial.eq(row.selected_trial)]
                    if len(chosen) != 1 or not np.isclose(
                            chosen.iloc[0].training_MAE_days, row.training_MAE_days):
                        raise ValueError(f"Selected training loss differs: {job_id}")
                    training_values.append(float(row.training_MAE_days))
                else:
                    training_values.append(float("nan"))
                training_weights.append(len(training_keys))
                report = dict(scheme=scheme, arm=arm, fold=fold, trials=count,
                              calibration_status=row.calibration_status,
                              n_training=len(training_keys),
                              training_MAE_days=row.training_MAE_days,
                              **error_summary(held),
                              missing_cases=";".join(held.loc[~held.status.eq("predicted"), "input_key"]),
                              VPD_w=row.VPD_w, I_req=row.I_req,
                              RH_sp=row.RH_sp, MP_sp=row.MP_sp,
                              u_tr=row.get("u_tr", float("nan")),
                              source_directory=str(directory))
                reports.append(report)
            summary = dict(scheme=scheme, arm=arm, **error_summary(oof),
                           training_MAE_days=float(np.average(training_values, weights=training_weights)),
                           min_trials=int(selected.n_trials.min()),
                           max_trials=int(selected.n_trials.max()))
            saved = methods.loc[methods.scheme.eq(scheme) & methods.arm.eq(arm)]
            if len(saved) != 1:
                raise ValueError(f"Missing or duplicate method summary: {scheme}/{arm}")
            for metric in ("MAE_days", "conditional_MAE_days", "n_predictions"):
                if not np.isclose(summary[metric], saved.iloc[0][metric], equal_nan=True):
                    raise ValueError(f"Recomputed {metric} differs: {scheme}/{arm}")
            summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(reports)


def make_report(summary, folds, directory, seed):
    lines = [f"结果目录: {directory}", f"seed={seed}; final 最终选择; {len(folds)} 折的 trial 序列与结果汇总已核对。",
             "误差为留出样本的绝对区间距离（天），按 site-year 等权汇总。",
             "训练 MAE 按各折训练样本数加权汇总，采用各折最终选定参数；训练集相互重叠，不是独立验证。",
             "Grid 为局部加密，实际预算不同；本图不是严格等预算比较。",
             "2p: VPD/I；3p: VPD/I/风速阈值（固定 RH=80、MP=3）；4p: VPD/I/RH/MP。3p 与 4p 不是嵌套参数空间。",
             "NA 表示存在无预测样本，不能报告完整测试集 MAE；conditional_MAE 仅统计有预测样本。"]
    for scheme in FOLDS:
        lines += [f"\n{scheme.upper()} 各方法汇总:",
                  summary.loc[summary.scheme.eq(scheme)].to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="NA"),
                  f"\n{scheme.upper()} 各折（training_MAE 为训练误差，MAE 为留出误差）:",
                  folds.loc[folds.scheme.eq(scheme)].to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="NA")]
    return "\n".join(lines)


def plot_comparison(summary, stem, seed):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch

    colors = {"training": "#397A9E", "test": "#D88739"}
    with matplotlib.rc_context({"font.size": 11, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig = Figure(figsize=(16, 7))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2, sharey=True)
        maximum = max(10, float(summary[["MAE_days", "training_MAE_days"]].max().max()) * 1.25)
        for ax, scheme, panel in zip(axes, FOLDS, ("a", "b")):
            for i, arm in enumerate(ARMS):
                row = summary.loc[summary.scheme.eq(scheme) & summary.arm.eq(arm)].iloc[0]
                for metric, category, offset in (("training_MAE_days", "training", -0.19),
                                                  ("MAE_days", "test", 0.19)):
                    x = i + offset
                    value = row[metric]
                    if np.isfinite(value):
                        ax.bar(x, value, width=0.34, color=colors[category], zorder=3)
                        ax.text(x, value + 0.8, f"{value:.2f}", ha="center", fontsize=8)
                    else:
                        label = f"NA\n{int(row.n_predictions)}/6" if category == "test" else "NA"
                        ax.text(x, 1.5, label, ha="center", color="#A13D32", fontsize=8)
                budget = str(int(row.min_trials)) if row.min_trials == row.max_trials else f"{int(row.min_trials)}–{int(row.max_trials)}"
                ax.text(i, -0.16, f"n={budget}", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=8, color="#555555")
            ax.set(title=f"({panel}) Leave-one-{scheme}-out", xticks=range(len(ARMS)),
                   xticklabels=[a.replace("_grid", "\nLocal grid").replace("_tpe", "\nTPE") for a in ARMS],
                   ylim=(0, maximum), xlim=(-0.6, len(ARMS) - 0.4))
            ax.grid(axis="y", color="#E3E3E3", zorder=0)
        axes[0].set_ylabel("MAE (days; lower is better)")
        fig.legend(handles=[Patch(color=colors[k], label=v) for k, v in
                            (("training", "Training"), ("test", "Held-out test"))],
                   loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
        footnote = (f"Final selections; seed {seed}. Six site-years per method; n = trials per fold (unequal budgets).\n"
                    "Training: fold MAEs weighted by training-set size; overlapping training sets are not independent validation.\n"
                    "Held-out test: six site-years, equally weighted; a complete MAE requires all six predictions.\n"
                    "2p: VPD/I; 3p: VPD/I/wind (RH=80, MP=3); 4p: VPD/I/RH/MP. 3p and 4p are not nested.")
        missing = summary.loc[summary.n_predictions.lt(summary.n_test)]
        for row in missing.itertuples():
            footnote += (f"\n{row.scheme}/{row.arm}: conditional MAE = {row.conditional_MAE_days:.2f} days"
                         f" for {row.n_predictions}/{row.n_test} predicted cases only (not plotted).")
        fig.text(0.07, 0.025, footnote, fontsize=8.5, va="bottom", linespacing=1.5)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.34, wspace=0.14)
        for extension in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{extension}"), dpi=300)
        fig.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "result/fsim_unified_joblib_local")
    parser.add_argument("--results-3p", type=Path, default=ROOT / "result/fsim_m0_3p_joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "fig/fsim_joblib_local_comparison",
                        help="Output path stem, without extension")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    settings = [json.loads((p / "effective_settings.json").read_text())
                for p in (args.results, args.results_3p)]
    for key in ("manifest", "parameter_table", "seeds", "tie_atol_days", "check_duplicate_policy"):
        if settings[0][key] != settings[1][key]:
            raise ValueError(f"Incompatible calibration settings: {key}")
    for name in ("VPD_w", "I_req"):
        if settings[0]["search_space"][name] != settings[1]["search_space"][name]:
            raise ValueError(f"Incompatible shared parameter range: {name}")
    loaded = [load_results(args.results, args.seed, [a for a in ARMS if not a.startswith("3p")]),
              load_results(args.results_3p, args.seed, ["3p_grid", "3p_tpe"])]
    summary, folds = [pd.concat([pair[i] for pair in loaded], ignore_index=True) for i in range(2)]
    for frame in (summary, folds):
        frame["arm"] = pd.Categorical(frame.arm, categories=ARMS, ordered=True)
        frame.sort_values(["scheme", "arm"], inplace=True)
    report = make_report(summary, folds, f"{args.results}; {args.results_3p}", args.seed)
    print(report, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(summary, args.output, args.seed)
    folds.to_csv(args.output.with_name(args.output.name + "_folds.csv"), index=False)
    summary.to_csv(args.output.with_name(args.output.name + "_summary.csv"), index=False)
    args.output.with_suffix(".txt").write_text(report + "\n", encoding="utf-8")
    print(f"\n已输出图与逐折报告: {args.output}.{{png,pdf,txt}} / {args.output.name}_folds.csv")


if __name__ == "__main__":
    main()
