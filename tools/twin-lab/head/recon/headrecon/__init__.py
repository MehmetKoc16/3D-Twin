"""twin-lab / head: reshape and re-texture the twin's head from real photos (between the refine and rig stages).

Reuses the refine stage (``twinrefine``: scan IO, renderer, landmark detector) and, through it, the texture stage
(``twintex``) and the rig stage's GLB reader. Run it with the refine stage's Python environment.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
RECON_DIR = PKG_DIR.parent
LAB_DIR = RECON_DIR.parents[1]  # tools/twin-lab
REPO = LAB_DIR.parents[1]

for _d in (LAB_DIR / "refine", LAB_DIR / "rig", LAB_DIR / "texture"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


def log(msg: str) -> None:
    print(f"[head {time.strftime('%H:%M:%S')}] {msg}", flush=True)
