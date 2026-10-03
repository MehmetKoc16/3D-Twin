"""The portable "face asset": head offsets (morph target), the baked face texture window and part choices.

Format (``dt-face-asset/1``) is documented in ``tools/twin-lab/hybrid/README.md``. It targets the app's standard
MakeHuman model: the offsets use the exact binary layout of ``morphs.bin`` (so they load as one more target), the
texture is the head UV island's rectangle of the fixed MakeHuman UV, and everything personal stays in ``user-data/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "dt-face-asset/1"
LICENSE = (
    "Derived from a private FLAME/Pixel3DMM fit (non-commercial, personal use) and CC0 MakeHuman assets. "
    "Contains data about one person: keep it in user-data/, never commit or redistribute."
)


def pack_offsets(offsets: np.ndarray, threshold: float = 1e-7) -> tuple[bytes, int]:
    """Sparse target entries of ``morphs.bin``: uint32 index, float32 dx, dy, dz (little endian), ascending index."""
    index = np.flatnonzero(np.abs(offsets).max(axis=1) > threshold)
    record = np.zeros(len(index), np.dtype([("i", "<u4"), ("d", "<f4", 3)]))
    record["i"] = index
    record["d"] = offsets[index]
    return record.tobytes(), int(len(index))


def read_offsets(blob: bytes, vertex_count: int) -> np.ndarray:
    record = np.frombuffer(blob, np.dtype([("i", "<u4"), ("d", "<f4", 3)]))
    out = np.zeros((vertex_count, 3))
    out[record["i"]] = record["d"]
    return out


def write_face_asset(
    folder: Path,
    *,
    offsets: np.ndarray,
    manifest_sha256: str,
    solved_body: dict,
    head_texture: np.ndarray,
    window_px: tuple[int, int, int, int],
    atlas_size: tuple[int, int],
    skin: dict,
    parts: dict,
    metrics: dict,
) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    blob, count = pack_offsets(offsets)
    (folder / "face-offsets.bin").write_bytes(blob)
    Image.fromarray(head_texture).save(folder / "face-texture.png", optimize=True)
    x, y, w, h = window_px
    aw, ah = atlas_size
    document = {
        "schema": SCHEMA,
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "license": LICENSE,
        "template": {
            "manifestSha256": manifest_sha256,
            "renderVertexCount": int(len(offsets)),
            "uvLayout": "apps/web/public/assets/body/base.glb TEXCOORD_0 (fixed MakeHuman UV, glTF v down)",
        },
        "solvedBody": solved_body,
        "headOffsets": {
            "file": "face-offsets.bin",
            "format": "morphs.bin entries: uint32 vertex index, float32 dx, dy, dz; little endian; 16 bytes each",
            "indexSpace": "render vertices of base.glb (index < renderVertexCount); seam copies carry equal offsets",
            "unit": "metre",
            "frame": "grounded MakeHuman A-pose of solvedBody; add to the solved body positions with weight 1",
            "count": count,
            "maxMm": float(np.linalg.norm(offsets, axis=1).max() * 1000),
            "note": "valid on top of the solved body; on another body they act as a head shape delta",
        },
        "texture": {
            "file": "face-texture.png",
            "colourSpace": "sRGB",
            "pixelWindow": {"x": x, "y": y, "width": w, "height": h},
            "atlasSizeAtBake": {"width": aw, "height": ah},
            "uvWindow": {"u0": x / aw, "v0": y / aw, "u1": (x + w) / aw, "v1": (y + h) / aw},
            "note": "UV rectangle of the head island; texels outside the island are padding. v is measured over the "
            "square MakeHuman UV space (the atlas strip below it holds part tiles and is not part of this window).",
        },
        "skin": skin,
        "parts": parts,
        "metrics": metrics,
    }
    (folder / "face-asset.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return document
