"""Run the two merged-transfer diagnostics with BBCH15 gating infection.

Both entry points share the same scenario definitions and output directory.
This fixed-parameter audit is not model selection or validation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audit_fsim_dispersal_scenarios import main


if __name__ == "__main__":
    main()
