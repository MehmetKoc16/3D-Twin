#!/usr/bin/env python
"""head.py - reshape and re-texture the twin's head from four real photos (between the refine and rig stages).

    python head.py --in user-data/twin/out/refine/refined.glb --photos user-data/twin/head \\
        --out user-data/twin/out/head/head.glb

Photos in ``--photos``: front.jpg, back.jpg, profile_nose_right.jpg, profile_nose_left.jpg (with --use-clean: de-glassed copies in
``clean/<name>.jpg`` with ``clean/<name>_mask.png`` marking the frame pixels). Everything runs locally. Run it with the
refine stage's Python environment (it imports the refine, texture and rig stages); person masks use the shape stage's
environment (rembg) when present, GrabCut otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from headrecon import REPO, log  # noqa: E402
from headrecon.pipeline import HeadConfig, run  # noqa: E402
from headrecon.reshape import ReshapeParams  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="refined.glb of the refine stage")
    ap.add_argument("--photos", type=Path, default=REPO / "user-data" / "twin" / "head", help="folder with the 4 head photos")
    ap.add_argument("--out", type=Path, required=True, help="head.glb to write")
    ap.add_argument("--use-clean", action="store_true", help="texture from the de-glassed photos in <photos>/clean/ (+ their frame masks) instead of the originals")
    ap.add_argument("--no-reshape", action="store_true", help="texture only, keep the geometry")
    ap.add_argument("--no-texture", action="store_true", help="reshape only, keep the atlas")
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--preview-dir", type=Path, default=None, help="default: <out dir>/previews")
    ap.add_argument("--rounds", type=int, default=6, help="silhouette matching rounds")
    ap.add_argument("--smoothing", type=float, default=2e-3, help="stiffness of the deformation field (thin-plate smoothing)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args(argv)
    if not args.inp.exists():
        print(f"input not found: {args.inp}", file=sys.stderr)
        return 2
    cfg = HeadConfig(
        inp=args.inp, photos=args.photos, out=args.out, use_clean=args.use_clean, reshape=not args.no_reshape,
        texture=not args.no_texture, previews=not args.no_previews, preview_dir=args.preview_dir,
        reshape_params=ReshapeParams(rounds=args.rounds, smoothing=args.smoothing), jpeg_quality=args.jpeg_quality,
    )
    run(cfg)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
