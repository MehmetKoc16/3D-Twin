#!/usr/bin/env python
"""Template-character twin: deform the MakeHuman head to a private FLAME fit and bake the photos into its UV.

Runs in the refine environment. Everything it reads or writes about the person stays under ``user-data/``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The package directory is this script's folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybridbody.hair import PROCEDURAL  # noqa: E402
from hybridbody.hairgen import HairStyle  # noqa: E402
from hybridbody.pipeline import DEFAULT_HAIR, run, verify_report  # noqa: E402


def hair_style_from(args, parser) -> HairStyle | None:
    """The procedural hair style from the CLI flags (None = defaults); errors for flags given with a MakeHuman hair."""
    overrides = {}
    if args.hairline_mm is not None:
        overrides["hairline_front"] = args.hairline_mm
    if (
        args.hair_top_mm is not None
    ):  # the top is longest at the front, a little shorter in the middle, short at the crown
        overrides.update(
            length_front=args.hair_top_mm, length_mid=args.hair_top_mm * 0.9, length_crown=args.hair_top_mm * 0.55
        )
    if args.hair_side_mm is not None:
        overrides["length_side"] = args.hair_side_mm
    if args.hair_seed is not None:
        overrides["seed"] = args.hair_seed
    for item in args.hair_param or []:
        key, _, value = item.partition("=")
        if not value:
            parser.error(f"--hair-param expects NAME=VALUE, got {item!r}")
        try:
            overrides[key.strip()] = float(value)
        except ValueError:
            parser.error(f"--hair-param {key}: {value!r} is not a number")
    if overrides and args.hair != PROCEDURAL:
        parser.error("the hair style flags only apply to --hair procedural")
    if args.hair_hex and args.hair != PROCEDURAL:
        parser.error("--hair-hex only applies to --hair procedural")
    try:
        return HairStyle.with_overrides(overrides) if overrides else None
    except ValueError as error:
        parser.error(str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodyfix", type=Path, required=True, help="bodyfix directory or bodyfixed.glb (dtBodyfix)")
    parser.add_argument(
        "--measurements",
        type=Path,
        help="tape measurements: re-solve the skin body natively (no clothing allowance), starting from --bodyfix",
    )
    parser.add_argument("--fit", type=Path, help="FLAME fit folder (default user-data/twin/head/flame/fit)")
    parser.add_argument("--photos", type=Path, help="folder with front.jpg and right.jpg (default colab_upload)")
    parser.add_argument("--flame-assets", type=Path, help="FLAME masks / MediaPipe embedding (default user-data/flame)")
    parser.add_argument("--out", type=Path, required=True, help="hybrid.glb path (under user-data/)")
    parser.add_argument(
        "--hair",
        default=DEFAULT_HAIR,
        help="'procedural' (default: cards grown on the deformed head, short sides and back, volume on top swept up "
        "and back, no fringe) or a MakeHuman CC0 hair part id from parts/index.json (hair-short, hair-tousled, ...)",
    )
    parser.add_argument(
        "--hair-hex",
        help="procedural hair colour #rrggbb (default: the photo colour made a natural dark brown if it is too grey)",
    )
    parser.add_argument("--hairline-mm", type=float, help="procedural hair: front hairline height above the eye line")
    parser.add_argument("--hair-top-mm", type=float, help="procedural hair: length of the top at the front hairline")
    parser.add_argument("--hair-side-mm", type=float, help="procedural hair: length of the sides and back")
    parser.add_argument("--hair-seed", type=int, help="procedural hair: random seed of the strands")
    parser.add_argument(
        "--hair-param",
        action="append",
        metavar="NAME=VALUE",
        help="procedural hair: any HairStyle field in millimetres (repeatable), e.g. temple_recession=8, "
        "sideburn_drop=5, nape_above_crease=25, taper_side=48, triangle_target=25000, lift_front=0.4",
    )
    parser.add_argument(
        "--iris-hex",
        help="iris colour #rrggbb (default: measured from the photo; a near-grey measurement becomes dark brown)",
    )
    parser.add_argument(
        "--brows",
        action="store_true",
        help="also add the MakeHuman eyebrow cards (off by default: the photographed brows are baked into the face)",
    )
    parser.add_argument("--texture-size", type=int, default=4096, help="square UV size; the part strip adds S/4 rows")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args(argv)
    if not 512 <= args.texture_size <= 4096 or args.texture_size % 256:
        parser.error("--texture-size must be a multiple of 256 between 512 and 4096")
    style = hair_style_from(args, parser)
    try:
        report = run(
            args.bodyfix,
            args.out,
            measurements=args.measurements,
            fit=args.fit,
            photos=args.photos,
            flame_assets=args.flame_assets,
            hair=args.hair,
            hair_hex=args.hair_hex,
            hair_style=style,
            brows=args.brows,
            iris_hex=args.iris_hex,
            texture_size=args.texture_size,
            previews=not args.no_previews,
            preview_dir=args.preview_dir,
        )
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    problems = verify_report(report)
    print(
        json.dumps(
            {"status": "ok" if not problems else "check", "problems": problems, **report}, indent=2, default=float
        )
    )
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
