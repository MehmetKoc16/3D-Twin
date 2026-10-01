#!/usr/bin/env python
"""refine.py - refine a textured scan between the texture and rig stages of the twin lab.

    python refine.py --in user-data/twin/out/texture/textured.glb --out user-data/twin/out/refine/refined.glb \\
        [--front user-data/twin/front.png] [--face user-data/twin/face.png]

1. Face relief: MediaPipe landmarks of the front photo -> fit of our MakeHuman face (face-fit modifiers + thin-plate
   warp) -> the MakeHuman face depth replaces the scan's face depth (rays along the front camera axis, eyelid / lip
   openings closed, joined to the scan at the face-oval boundary) -> the front photo (and ``face.png`` when present)
   is projected again onto the face region.
2. Armpits: the arms are cut from the torso along the fitted body's arm / torso interface (below the armpit apex),
   both lips are capped and the arm side is nudged outward a few millimetres, so a T-pose shows no webbing.

Everything runs locally; outputs next to ``--out`` (refined.glb, refine_report.json, previews/, cache/fit.pkl).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from twinrefine import log  # noqa: E402
from twinrefine.armpit import ArmpitParams  # noqa: E402
from twinrefine.pipeline import RefineConfig, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="textured.glb of the texture stage")
    ap.add_argument("--out", type=Path, required=True, help="refined.glb to write")
    ap.add_argument("--front", type=Path, default=None, help="front photo (default: user-data/twin/front.png)")
    ap.add_argument("--face", type=Path, default=None, help="head-and-shoulders portrait (default: user-data/twin/face.png if present)")
    ap.add_argument("--fit-cache", type=Path, default=None, help="body-fit pickle to reuse / write (default: <out dir>/cache/fit.pkl)")
    ap.add_argument("--no-armpits", action="store_true", help="skip the armpit separation")
    ap.add_argument("--no-face", action="store_true", help="skip the face relief (and the face texture re-projection)")
    ap.add_argument("--no-rebake", action="store_true", help="relief only, keep the old atlas")
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--preview-dir", type=Path, default=None, help="default: <out dir>/previews")
    ap.add_argument("--gap-mm", type=float, default=4.0, help="outward nudge of the arm side at the armpit (mm)")
    ap.add_argument("--y-top", type=float, default=None, help="armpit cut zone top (m); default: apex of the fitted body + 3 cm")
    ap.add_argument("--face-edge-mm", type=float, default=2.2, help="target edge length in the face region (mm)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args(argv)

    if not args.inp.exists():
        print(f"input not found: {args.inp}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = RefineConfig(
        inp=args.inp, out=args.out, front=args.front, face=args.face, fit_cache=args.fit_cache,
        armpits=not args.no_armpits, face_relief=not args.no_face, rebake=not args.no_rebake,
        previews=not args.no_previews, preview_dir=args.preview_dir, gap_mm=args.gap_mm,
        face_edge_mm=args.face_edge_mm, jpeg_quality=args.jpeg_quality,
        armpit=ArmpitParams(y_top=args.y_top),
    )
    run(cfg)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
