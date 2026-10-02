"""Stand-alone person segmentation worker (rembg u2net_human_seg). Run with the shape stage's Python (it has rembg).

    python segment_worker.py <out_dir> <img> [<img> ...]   ->  <out_dir>/mask_<stem>.png (0/255, same size as the image)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[2]
os.environ.setdefault("U2NET_HOME", str(LAB / "shape" / "weights" / "rembg"))


def main() -> int:
    from rembg import new_session, remove

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    session = new_session("u2net_human_seg")
    for p in sys.argv[2:]:
        img = Image.open(p).convert("RGB")
        res = remove(img, session=session, only_mask=True)
        m = (np.array(res) > 127).astype(np.uint8) * 255
        Image.fromarray(m).save(out / f"mask_{Path(p).stem}.png")
        print("masked", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
