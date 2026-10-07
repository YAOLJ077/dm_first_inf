"""Select one stable FSIM/FSIM-S parameter set for deployment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FSIM_FOLDS = ROOT / "result/cv_spore_pressure_s_dep38_395_rh80_coarse/fold_parameters.csv"
DEFAULT_SPORE_FOLDS = ROOT / "result/cv_spore_pressure_s_observed_zero_gate/fold_parameters.csv"
DEFAULT_FSIM_CONFIG = ROOT / "model/FSIM/config.json"
DEFAULT_SPORE_CONFIG = ROOT / "model/FSIM-S/config.json"
DEFAULT_OUTPUT = ROOT / "result/deployment_parameters"

FSIM_PARAMETERS = {
    "wind_min_m_s": "s_wind_min_m_s",
    "heavy_rain_washoff_mm": "s_heavy_rain_washoff_mm",
    "minimum_moist_hours": "s_minimum_moist_hours",
    "infection_degree_hours": "s_infection_degree_hours",
    "effective_deposition_min_rain_mm": "s_effective_deposition_min_rain_mm",
    "sporulation_rh_min_pct": "s_sporulation_rh_min_pct",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _consensus(values: pd.Series) -> tuple[float, str]:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    counts = pd.Series(numeric).value_counts()
    maximum = int(counts.max())
    modes = sorted(float(value) for value in counts[counts.eq(maximum)].index)
    if len(modes) == 1:
        return modes[0], "unique_mode"
    return float(np.median(numeric)), "median_tie_break"


def _validate_fold_inputs(fsim: pd.DataFrame, spore: pd.DataFrame) -> None:
    for frame, label, required in (
        (
            fsim,
            "FSIM",
            {"cv_scheme", "held_out_group", *FSIM_PARAMETERS.values()},
        ),
        (spore, "FSIM-S", {"cv_scheme", "held_out_group", "selected_beta_sp"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} fold file is missing columns: {sorted(missing)}")
        if not frame["cv_scheme"].eq("site_year").all():
            raise ValueError(f"{label} fold file must contain only site_year folds")
        if frame["held_out_group"].duplicated().any():
            raise ValueError(f"{label} fold file contains duplicate held-out groups")
    if set(fsim["held_out_group"]) != set(spore["held_out_group"]):
        raise ValueError("FSIM and FSIM-S fold files do not contain the same held-out groups")
    if len(fsim) != 6 or len(spore) != 6:
        raise ValueError("Expected six site-year folds for deployment selection")


def select_deployment_parameters(fsim: pd.DataFrame, spore: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Apply the fixed-value, mode, and median tie-break rules."""

    _validate_fold_inputs(fsim, spore)
    fsim = fsim.sort_values("held_out_group").reset_index(drop=True)
    spore = spore.set_index("held_out_group").loc[fsim["held_out_group"]].reset_index()

    selected = {}
    audit_rows = []
    for name, column in FSIM_PARAMETERS.items():
        values = pd.to_numeric(fsim[column], errors="raise")
        if values.nunique(dropna=False) == 1:
            value = float(values.iloc[0])
            rule = "fixed_across_folds"
        else:
            value, rule = _consensus(values)
        selected[name] = value
        audit_rows.append({
            "model": "FSIM",
            "parameter": name,
            "source_column": column,
            "fold_values": ",".join(str(float(value)) for value in values),
            "selected_value": value,
            "selection_rule": rule,
        })

    beta_values = pd.to_numeric(spore["selected_beta_sp"], errors="raise")
    beta, beta_rule = _consensus(beta_values)
    selected_s = {
        **selected,
        "beta_sp": beta,
        "observed_zero_weather_gate": True,
        "pressure_field": {
            "shape": "interval_rate",
            "normalization": "zscore",
            "rolling_days": 3,
            "outside_fill": "zero",
            "reference_level": 0.5,
        },
    }
    audit_rows.append({
        "model": "FSIM-S",
        "parameter": "beta_sp",
        "source_column": "selected_beta_sp",
        "fold_values": ",".join(str(float(value)) for value in beta_values),
        "selected_value": beta,
        "selection_rule": beta_rule,
    })
    audit_rows.extend([
        {
            "model": "FSIM-S",
            "parameter": "observed_zero_weather_gate",
            "source_column": "accepted structural rule",
            "fold_values": "true",
            "selected_value": True,
            "selection_rule": "prespecified observation rule",
        },
        {
            "model": "FSIM-S",
            "parameter": "pressure_field",
            "source_column": "accepted structural configuration",
            "fold_values": "interval_rate/zscore/3-day/zero/0.5",
            "selected_value": "interval_rate/zscore/3-day/zero/0.5",
            "selection_rule": "prespecified observation rule",
        },
    ])
    return {"FSIM": selected, "FSIM-S": selected_s}, pd.DataFrame(audit_rows)


def _write_deployment_configs(output: Path, parameters: dict, fsim_path: Path, spore_path: Path) -> None:
    fsim_config = deepcopy(_read_json(fsim_path))
    fsim = parameters["FSIM"]
    fsim_config["structure_parameters"]["wind_min_m_s"] = fsim["wind_min_m_s"]
    fsim_config["deposition_parameters"]["effective_deposition_min_rain_mm"] = fsim["effective_deposition_min_rain_mm"]
    fsim_config["deposition_parameters"]["heavy_rain_washoff_mm"] = fsim["heavy_rain_washoff_mm"]
    fsim_config["sporulation_parameters"]["minimum_moist_hours"] = fsim["minimum_moist_hours"]
    fsim_config["sporulation_parameters"]["sporulation_rh_min_pct"] = fsim["sporulation_rh_min_pct"]
    fsim_config["biological_parameters"]["infection_degree_hours"] = fsim["infection_degree_hours"]

    spore_config = deepcopy(_read_json(spore_path))
    pressure = spore_config["pressure_adjustment"]
    pressure["enabled"] = True
    pressure["beta_sp"] = parameters["FSIM-S"]["beta_sp"]
    pressure["rolling_days"] = parameters["FSIM-S"]["pressure_field"]["rolling_days"]
    pressure["reference_level"] = parameters["FSIM-S"]["pressure_field"]["reference_level"]
    pressure["observed_zero_weather_gate"] = True

    (output / "fsim_deployment_config.json").write_text(
        json.dumps(fsim_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "fsim_s_deployment_config.json").write_text(
        json.dumps(spore_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsim-folds", type=Path, default=DEFAULT_FSIM_FOLDS)
    parser.add_argument("--spore-folds", type=Path, default=DEFAULT_SPORE_FOLDS)
    parser.add_argument("--fsim-config", type=Path, default=DEFAULT_FSIM_CONFIG)
    parser.add_argument("--spore-config", type=Path, default=DEFAULT_SPORE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError(f"Output directory is not empty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    fsim = pd.read_csv(args.fsim_folds, encoding="utf-8-sig")
    spore = pd.read_csv(args.spore_folds, encoding="utf-8-sig")
    parameters, audit = select_deployment_parameters(fsim, spore)
    _write_deployment_configs(output, parameters, args.fsim_config, args.spore_config)
    audit.to_csv(output / "selection_audit.csv", index=False, encoding="utf-8-sig")

    result = {
        "schema_version": 1,
        "model": "FSIM/FSIM-S deployment parameters",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": {
            "outer_cv": "six site-year leave-one-out folds",
            "fixed_parameters": "retain values identical across all folds",
            "fold_consensus": "unique mode; median when the mode is tied",
            "spore_gate": "numeric zero-spore collection intervals suppress weather-generated cohorts; missing observations do not",
        },
        "parameters": parameters,
        "source_files": {
            "fsim_fold_parameters": str(args.fsim_folds.resolve()),
            "spore_fold_parameters": str(args.spore_folds.resolve()),
        },
    }
    (output / "deployment_parameters.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(parameters, ensure_ascii=False, indent=2))
    print(f"Deployment files: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
