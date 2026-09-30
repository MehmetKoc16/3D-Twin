#!/usr/bin/env python
"""project.py - texture an untextured body mesh with the user's real view photos.

Example:
    python project.py                 # user-data/twin/out/shape/mesh.glb + user-data/twin/{front,back,left,right}.png
    python project.py --stand-in      # validate end-to-end on the generic MakeHuman stand-in body
    python project.py --size 2048 --no-previews
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from twintex.pipeline import PipelineConfig, run  # noqa: E402


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "AGENTS.md").exists():
            return p
    return HERE.parents[2]


def find_shape_mesh(shape_dir: Path) -> Path:
    direct = shape_dir / "mesh.glb"
    if direct.exists():
        return direct
    found = sorted(shape_dir.glob("*/mesh.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else direct


def main(argv: list[str] | None = None) -> int:
    root = repo_root()
    twin = root / "user-data" / "twin"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", type=Path, default=None,
                    help="default: user-data/twin/out/shape/mesh.glb, else the newest */mesh.glb below it")
    ap.add_argument("--own-matting", action="store_true",
                    help="ignore the shape agent's cutout_<view>.png masks and use our own background removal")
    ap.add_argument("--views-dir", type=Path, default=twin, help="folder with front/back/left/right .png")
    ap.add_argument(
        "--view", action="append", default=[], metavar="NAME=PATH",
        help="explicit view image, e.g. --view back=some/photo.jpg (NAME in front/back/left/right)",
    )
    ap.add_argument("--out", type=Path, default=twin / "out" / "texture")
    ap.add_argument("--stand-in", action="store_true", help="use the generic stand-in body (validation only)")
    ap.add_argument("--size", type=int, default=4096, help="texture size (square)")
    ap.add_argument("--padding", type=int, default=8, help="atlas gutter in texels")
    ap.add_argument("--max-faces", type=int, default=150_000, help="decimate above this face count (0 = never)")
    ap.add_argument("--up", default="+y", help="mesh up axis (default +y)")
    ap.add_argument("--front", default="+z", help="mesh front axis (default +z)")
    ap.add_argument("--no-flow", action="store_true", help="disable the non-rigid silhouette warp")
    ap.add_argument("--aspect", action="store_true", help="also fit an anisotropic scale per view")
    ap.add_argument("--no-harmonize", action="store_true", help="skip per-view colour gain matching")
    ap.add_argument("--delight", type=float, default=0.0, help="0..1 strength of illumination-gradient removal")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror-fill unseen side texels")
    ap.add_argument("--no-hair-prior", action="store_true", help="do not paint unseen head areas with hair colour")
    ap.add_argument("--sharpness", type=float, default=4.0, help="blend weight exponent k of cos^k")
    ap.add_argument("--face-boost", type=float, default=8.0, help="extra weight of the front view on the face")
    ap.add_argument("--png", action="store_true", help="embed a PNG instead of a JPEG in the GLB")
    ap.add_argument("--jpeg-quality", type=int, default=93)
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--no-debug", action="store_true")
    args = ap.parse_args(argv)

    mesh = args.mesh if args.mesh is not None else find_shape_mesh(twin / "out" / "shape")
    out = args.out
    if args.stand_in:
        from make_standin import make

        mesh = HERE / ".cache" / "standin_mesh.glb"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        make().export(mesh)
        if out == twin / "out" / "texture":
            out = out.parent / "texture_standin"
    if not mesh.exists():
        print(f"mesh not found: {mesh} (use --stand-in to validate on the generic body)", file=sys.stderr)
        return 2
    overrides = {}
    for item in args.view:
        name, _, path = item.partition("=")
        overrides[name.strip()] = Path(path.strip())
    cfg = PipelineConfig(
        mesh_path=mesh,
        views_dir=args.views_dir,
        out_dir=out,
        view_files=overrides,
        size=args.size,
        padding=args.padding,
        max_faces=args.max_faces,
        up=args.up,
        front=args.front,
        use_flow=not args.no_flow,
        allow_aspect=args.aspect,
        harmonize=not args.no_harmonize,
        delight=args.delight,
        mirror=not args.no_mirror,
        hair_prior=not args.no_hair_prior,
        shape_cutouts=not args.own_matting,
        sharpness=args.sharpness,
        face_boost=args.face_boost,
        texture_format="png" if args.png else "jpeg",
        jpeg_quality=args.jpeg_quality,
        previews=not args.no_previews,
        debug_maps=not args.no_debug,
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
