"""Run both regional scenarios, including phenology, disease, and long tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for profile in ("table", "vine"):
        for script in ("run_guangxi_phenology", "run_guangxi_disease_windows",
                       "build_guangxi_phenology_long_table", "build_guangxi_onset_long_table",
                       "build_guangxi_fungicide_savings"):
            command = [sys.executable, str(ROOT / "src" / f"{script}.py"), "--profile", profile]
            if script.startswith("run_"):
                command += ["--workers", str(args.workers)]
                if args.overwrite:
                    command.append("--overwrite")
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
