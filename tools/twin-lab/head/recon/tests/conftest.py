import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import headrecon  # noqa: E402,F401  (puts the refine, rig and texture stage folders on sys.path)
