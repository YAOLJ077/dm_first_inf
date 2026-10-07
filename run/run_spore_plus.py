"""Run FSIM and the optional CK-derived observed-dose extension."""

from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

run_spore_plus = importlib.import_module("model.FSIM-S").run_spore_plus  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", required=True, type=Path)
    parser.add_argument(
        "--spore-file",
        type=Path,
        default=PROJECT_ROOT / "data/spore_data/guangxi_site_years_spore_data.xlsx",
    )
    parser.add_argument(
        "--pressure-config",
        type=Path,
        default=None,
        help="Optional FSIM-S deployment configuration JSON",
    )
    parser.add_argument("--beta-sp", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args(argv)
    pressure_config = (
        args.pressure_config
        if args.pressure_config is not None
        else {"enabled": args.enable, "beta_sp": args.beta_sp}
    )
    result = run_spore_plus(
        args.weather,
        pressure_enabled=args.enable,
        spore_file=args.spore_file,
        pressure_config=pressure_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    process = result.get("spore_process") or result["process"]
    diagnostics = process["diagnostics"]
    print(
        f"{result['season']['site_id']}_{result['season']['year']}: "
        f"infection_events={diagnostics['infection_events']}, "
        f"first_symptom_start={diagnostics['first_symptom_start']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
