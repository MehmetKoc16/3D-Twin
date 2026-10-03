#!/usr/bin/env python
"""Transplant a locally fitted FLAME face onto the textured scan, using the refine venv."""

import argparse
import json
import sys
from pathlib import Path

from flamehead.pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--flame-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-ears", action="store_true")
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args(argv)
    if args.texture_size < 512 or args.texture_size > 4096:
        parser.error("--texture-size must be between 512 and 4096")
    try:
        report = run(
            args.inp,
            args.fit,
            args.photos,
            args.flame_assets,
            args.out,
            include_ears=args.include_ears,
            texture_size=args.texture_size,
            preview_dir=args.preview_dir,
            previews=not args.no_previews,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
