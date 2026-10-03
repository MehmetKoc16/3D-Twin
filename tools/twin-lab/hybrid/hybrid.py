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

from hybridbody.pipeline import DEFAULT_HAIR, run, verify_report  # noqa: E402


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
        help="MakeHuman CC0 hair part id from parts/index.json (default: short sides, volume on top, swept back)",
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
    try:
        report = run(
            args.bodyfix,
            args.out,
            measurements=args.measurements,
            fit=args.fit,
            photos=args.photos,
            flame_assets=args.flame_assets,
            hair=args.hair,
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
