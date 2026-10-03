"""Template-character twin: the MakeHuman head is deformed to a private FLAME fit (no graft, no seam).

The package reuses the sibling stages without editing them (all of them run in the refine environment):
``flamehead`` (FLAME cameras, photo baking, colour tools), ``twinrefine`` / ``twintex`` (renderer, rasteriser) and the
rig stage's MakeHuman model and GLB writer. Nothing here is shipped: inputs and outputs contain FLAME-derived data
about one person and stay in ``user-data/``.
"""

import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
REPO = LAB.parents[1]
BODY_ASSETS = REPO / "apps/web/public/assets/body"
PARTS_ASSETS = REPO / "apps/web/public/assets/parts"

for _folder in ("refine", "texture", "rig", "head/recon", "head/flame"):
    if str(LAB / _folder) not in sys.path:
        sys.path.insert(0, str(LAB / _folder))


def log(message: str) -> None:
    print(f"[hybrid {time.strftime('%H:%M:%S')}] {message}", flush=True)
