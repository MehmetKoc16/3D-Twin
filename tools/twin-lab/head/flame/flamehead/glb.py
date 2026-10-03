"""Shared GLB I/O plus extras preservation and atomic validation."""

import copy
import json
import struct
from pathlib import Path

import numpy as np
from glbio import _split
from twinrefine.scan import load_scan, save_scan


def restore_extras(source, destination, report):
    for key in ("extras",):
        if key in source:
            destination[key] = copy.deepcopy(source[key])
    destination["asset"]["extras"] = copy.deepcopy(source.get("asset", {}).get("extras", {}))
    destination["asset"]["extras"]["dtFlameHead"] = report
    for key in ("meshes", "nodes", "scenes", "materials", "textures", "samplers", "images"):
        for old, new in zip(source.get(key, []), destination.get(key, [])):
            if "extras" in old:
                new["extras"] = copy.deepcopy(old["extras"])
            if key == "meshes":
                for op, np_ in zip(old.get("primitives", []), new.get("primitives", [])):
                    if "extras" in op:
                        np_["extras"] = copy.deepcopy(op["extras"])


def save(path: Path, scan, original: Path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.glb")
    try:
        save_scan(temporary, scan)
        source, _ = _split(original.read_bytes())
        destination, binary = _split(temporary.read_bytes())
        restore_extras(source, destination, report)
        encoded = json.dumps(destination, separators=(",", ":"), allow_nan=False).encode()
        encoded += b" " * (-len(encoded) % 4)
        binary += b"\0" * (-len(binary) % 4)
        temporary.write_bytes(
            struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
            + struct.pack("<II", len(encoded), 0x4E4F534A)
            + encoded
            + struct.pack("<II", len(binary), 0x004E4942)
            + binary
        )
        checked = load_scan(temporary)
        if len(checked.faces) != len(scan.faces) or not np.isfinite(checked.verts).all():
            raise ValueError("GLB round-trip validation failed")
        if checked.atlas is None or np.any(checked.uv < 0) or np.any(checked.uv > 1):
            raise ValueError("Invalid packed texture or UVs")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
