#!/usr/bin/env python3
"""Calibrate the final Table/Vine phenology profiles.

This is the production copy of the validated Guangxi profile calibration.
Guilin is excluded, Nanning BBCH09 uses the agreed assumed dates, and the
candidate search keeps the forcing base temperature at or below 10 C.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model.phenology.profile_calibration_core import (  # noqa: E402
    decode, fit_model, load_data, metric_rows, predictions,
)


ASSUMED_NANNING_09 = {
    "22.610000_108.230000_2023": "2023-03-06",
    "22.610000_108.230000_2024": "2024-03-18",
}
GROUP_PREFIX = {
    "table": "22.610000_108.230000_",
    "vine": "23.830000_108.100000_",
}
SPECS = [
    ("linear_fixed_base_0", "free_start"),
    ("linear_fixed_base_3", "free_start"),
    ("linear_fixed_base_5", "free_start"),
    ("linear_fixed_base_8", "free_start"),
    ("linear_fixed_base_10", "free_start"),
    ("linear_free_base", "free_start"),
    ("linear_fixed_base_0", "start_1"),
    ("beta_fixed_cardinals", "free_start"),
    ("beta_fixed_cardinals", "start_1"),
    ("beta_free_cardinals", "free_start"),
]


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/phenology_data/calibration_data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "result/phenology_calibration")
    p.add_argument("--maxiter", type=int, default=75)
    p.add_argument("--popsize", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261008, 20261009])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def build_targets(data_dir: Path, output_dir: Path):
    targets = pd.read_csv(data_dir / "phenology_calibration_targets.csv", encoding="utf-8-sig")
    targets = targets.loc[~targets.site_year.str.startswith("25.450000_110.800000_")].copy()
    for site, date_text in ASSUMED_NANNING_09.items():
        mask = targets.site_year.eq(site) & targets.target_stage.eq(9)
        if mask.sum() != 1:
            raise ValueError(f"Expected one Nanning BBCH09 target for {site}")
        ordinal = pd.Timestamp(date_text).date().toordinal()
        targets.loc[mask, "censor_type"] = "exact_assumed"
        targets.loc[mask, "observed_start"] = date_text
        targets.loc[mask, "observed_end"] = date_text
        targets.loc[mask, "observed_start_ordinal"] = ordinal
        targets.loc[mask, "observed_end_ordinal"] = ordinal
        targets.loc[mask, "calibration_lower"] = date_text
        targets.loc[mask, "calibration_lower_ordinal"] = ordinal
        targets.loc[mask, "target_source"] = targets.loc[mask, "target_source"].astype(str) + "+assumed_BBCH09_0306_0318"
    path = output_dir / "phenology_calibration_targets.csv"
    targets.to_csv(path, index=False)
    records = pd.read_csv(data_dir / "phenology_calibration_records.csv", encoding="utf-8-sig")
    records = records.loc[~records.site_year.str.startswith("25.450000_110.800000_")].copy()
    records.to_csv(output_dir / "phenology_calibration_records.csv", index=False)
    return path


def summary(rows):
    x = pd.DataFrame(rows)
    x = x.loc[x.target_stage.isin([9, 65])]
    if x.empty:
        return np.nan, np.nan, np.nan
    exact = x.loc[x.censor_type.isin(["exact", "exact_assumed"]), "exact_abs_error_days"]
    return float(x.interval_error_days.mean()), float(exact.mean()) if len(exact) else np.nan, float((x.interval_error_days <= 7).mean())


def main(args):
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_path = build_targets(args.data_dir, args.output_dir)
    targets, weather, local_sites = load_data(args.data_dir, targets_path=target_path)
    groups = {name: sorted(s for s in local_sites if s.startswith(prefix)) for name, prefix in GROUP_PREFIX.items()}
    if any(len(v) < 2 for v in groups.values()):
        raise ValueError(f"Expected at least two local years per profile: {groups}")
    fit_args = SimpleNamespace(seeds=args.seeds, maxiter=args.maxiter, popsize=args.popsize)
    all_rows, comparisons, selected = [], [], {}
    for group, sites in groups.items():
        for family, start_mode in SPECS:
            name = f"{group}__{family}_{start_mode}"
            best, restarts = fit_model(targets, weather, sites, family, start_mode, fit_args)
            full_pred = predictions(weather, sites, best.x, family, start_mode)
            apparent = metric_rows(targets, full_pred, sites, "apparent", name)
            all_rows.extend(apparent)
            validation = []
            for test_site in sites:
                train = [s for s in sites if s != test_site]
                fitted, _ = fit_model(targets, weather, train, family, start_mode, fit_args)
                pred = predictions(weather, [test_site], fitted.x, family, start_mode)
                validation.extend(metric_rows(targets, pred, [test_site], "leave_one_year_out", name))
            all_rows.extend(validation)
            av, ae, aw = summary(apparent); vv, ve, vw = summary(validation)
            params = decode(best.x, family, start_mode)
            comparisons.append({"group": group, "model": name, "family": family, "start_mode": start_mode,
                                "apparent_primary_mae": av, "apparent_exact_mae": ae,
                                "validation_primary_mae": vv, "validation_exact_mae": ve, "validation_within_7d": vw,
                                "objective_days": float(best.fun), "parameters": json.dumps({
                                    "base_c": params["base_c"], "cardinals": params["cardinals"], "start_doy": params["start_doy"],
                                    "thresholds": [float(v) for v in params["thresholds"]]}, ensure_ascii=False),
                                "restarts": json.dumps(restarts)})
        options = pd.DataFrame([r for r in comparisons if r["group"] == group]).sort_values(
            ["validation_primary_mae", "validation_exact_mae"])
        chosen = options.iloc[0].to_dict()
        selected[group] = chosen
        best, restarts = fit_model(targets, weather, sites, chosen["family"], chosen["start_mode"], fit_args)
        params = decode(best.x, chosen["family"], chosen["start_mode"])
        (args.output_dir / f"{group}_final_parameters.json").write_text(json.dumps({
            "group": group, "site_years": sites, "family": chosen["family"], "start_mode": chosen["start_mode"],
            "parameters": {"base_c": params["base_c"], "cardinals": params["cardinals"], "start_doy": params["start_doy"],
                           "H09": float(params["thresholds"][0]), "H09_65": float(params["thresholds"][1]),
                           "H65_81": float(params["thresholds"][2])}, "restarts": restarts,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pred = pd.DataFrame(all_rows)
    comp = pd.DataFrame(comparisons)
    pred.to_csv(args.output_dir / "table_vine_predictions.csv", index=False)
    comp.to_csv(args.output_dir / "table_vine_model_comparison.csv", index=False)
    (args.output_dir / "selected_profiles.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["Table/Wine phenology profile calibration", "", comp[["group", "model", "validation_primary_mae", "validation_exact_mae", "validation_within_7d"]].to_string(index=False), "", "Selected:", json.dumps(selected, ensure_ascii=False, indent=2)]
    chosen_rows = []
    for group, chosen in selected.items():
        chosen_rows.extend(r for r in all_rows if r["model"] == chosen["model"] and r["split"] == "leave_one_year_out")
    report.extend(["", "Selected model stage metrics:", pd.DataFrame(chosen_rows).groupby("target_stage").agg(
        n=("target_stage", "size"), mae_days=("interval_error_days", "mean"), exact_mae_days=("exact_abs_error_days", "mean"),
        within_7d=("interval_error_days", lambda x: float((x <= 7).mean()))).to_string()])
    (args.output_dir / "calibration_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main(cli())
