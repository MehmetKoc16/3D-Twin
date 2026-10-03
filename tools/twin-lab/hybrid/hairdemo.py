#!/usr/bin/env python
"""Procedural hair on the generic MakeHuman head: head previews (front, side, back, 3/4, top) and hair numbers.

Runs in the refine environment. Nothing personal is read: the head is the plain MakeHuman body, the skin a flat tone.

    tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/hybrid/hairdemo.py \
        --out user-data/twin/out/hybrid/previews_generic_hair [--gender 1.0] [--hair-hex #3a281c] [--hair-param key=value]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybridbody.hairdemo import build_generic_twin, dump_summary, summary, write_generic_previews  # noqa: E402
from hybridbody.hairgen import HairStyle  # noqa: E402
from hybridbody.partstex import plausible_hair  # noqa: E402


def parse_params(items):
    values = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--hair-param expects name=value, got {item!r}")
        values[key.strip()] = float(value)
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="folder for the previews and generic_hair_report.json")
    parser.add_argument("--gender", type=float, default=1.0, help="MakeHuman gender macro 0..1 (default 1 = male)")
    parser.add_argument("--hair-hex", help="strand colour #rrggbb (default: natural dark brown)")
    parser.add_argument(
        "--hair-param", action="append", metavar="NAME=VALUE", help="hair style parameter (millimetres)"
    )
    parser.add_argument("--texture-size", type=int, default=4096, help="square UV size (the strip adds S/4 rows)")
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument("--no-clay", action="store_true", help="skip the clay renders (faster)")
    args = parser.parse_args(argv)
    style = HairStyle.with_overrides(parse_params(args.hair_param))
    colour = plausible_hair(None, args.hair_hex)["srgb"]
    twin = build_generic_twin(
        gender=args.gender, colour_srgb=colour, style=style, size=args.texture_size, coverage=not args.no_coverage
    )
    paths = write_generic_previews(args.out, twin, variants=(False,) if args.no_clay else (False, True))
    report = dump_summary(args.out, twin)
    print(json.dumps({"previews": paths, "report": str(report), **summary(twin)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
