"""twin-lab / refine: face relief + armpit separation for a textured scan, between the texture and rig stages.

The package reuses the sibling stages without editing them: ``twintex`` (texture stage: cameras, flow alignment,
rasterisers) and the rig stage's MakeHuman model / fitter / GLB reader. Both folders are put on ``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REFINE_DIR = PKG_DIR.parent
LAB_DIR = REFINE_DIR.parent  # tools/twin-lab


def _find_repo_root() -> Path:
    for p in [PKG_DIR, *PKG_DIR.parents]:
        if (p / "AGENTS.md").exists():
            return p
    return LAB_DIR.parents[1]


REPO = _find_repo_root()
BODY_ASSETS = REPO / "apps" / "web" / "public" / "assets" / "body"
FACE_MAP_JSON = BODY_ASSETS / "face-map.json"
LANDMARKER_TASK = REPO / "apps" / "web" / "public" / "models" / "face_landmarker.task"

for _d in (LAB_DIR / "rig", LAB_DIR / "texture"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


def log(msg: str) -> None:
    import time

    print(f"[refine {time.strftime('%H:%M:%S')}] {msg}", flush=True)
