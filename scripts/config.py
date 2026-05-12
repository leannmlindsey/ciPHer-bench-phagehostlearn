"""Resolve env-sourced paths for ciPHer-bench-phagehostlearn wrappers.

Usage in any wrapper:

    from config import PHL_REPO, CIPHER_REPO, CIPHER_VAL_GENOMES, PHL_OUTPUT_ROOT

Before running the wrapper, source the env file:

    cp config/phagehostlearn.env.template phagehostlearn.env   # or _delta / _biowulf
    pico phagehostlearn.env                                    # edit paths
    source phagehostlearn.env
    python scripts/run_<dataset>_inference.py
"""
import os
import sys
from pathlib import Path


def _require_env(name: str, hint: str = "") -> Path:
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"ERROR: env var {name} is not set.\n"
            f"  source <repo_root>/phagehostlearn.env first.\n"
            f"  See SETUP.md. {hint}"
        )
    p = Path(val)
    if not p.exists():
        sys.exit(
            f"ERROR: env var {name} = {val} does not exist on disk.\n"
            f"  Edit phagehostlearn.env to fix. {hint}"
        )
    return p


PHL_REPO            = _require_env("PHL_REPO", "(point at your PhageHostLearn clone)")
CIPHER_REPO         = _require_env("CIPHER_REPO", "(point at your cipher checkout)")
CIPHER_VAL_GENOMES  = _require_env("CIPHER_VAL_GENOMES", "(point at cipher_data/validation_genomes)")

# Output root may not exist yet — create it if needed (env var must still be set)
_phl_out = os.environ.get("PHL_OUTPUT_ROOT")
if not _phl_out:
    sys.exit("ERROR: env var PHL_OUTPUT_ROOT is not set. source phagehostlearn.env first.")
PHL_OUTPUT_ROOT = Path(_phl_out)
PHL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

XGB_MODEL = PHL_REPO / "code" / "phagehostlearn_esm2_xgb.json"
if not XGB_MODEL.exists():
    sys.exit(
        f"ERROR: XGBoost model not found at {XGB_MODEL}.\n"
        f"  Confirm $PHL_REPO points at a complete upstream PhageHostLearn clone.\n"
        f"  See SETUP.md."
    )
