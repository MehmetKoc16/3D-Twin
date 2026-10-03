"""Local FLAME transplant; model data and personal outputs are never package assets."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LAB = HERE.parents[1]
REPO = LAB.parents[1]
sys.path.insert(0, str(LAB / "head/recon"))
# Import the existing package to establish its shared refine/texture/rig imports.
import headrecon  # noqa: E402, F401
