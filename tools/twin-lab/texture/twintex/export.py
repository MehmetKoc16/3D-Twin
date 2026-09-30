"""Binary glTF (GLB) writer for one textured mesh (baseColor texture embedded, sRGB)."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def encode_texture(rgb_u8: np.ndarray, fmt: str = "jpeg", quality: int = 93) -> tuple[bytes, str]:
    im = Image.fromarray(rgb_u8, "RGB")
    buf = io.BytesIO()
    if fmt.lower() in ("jpg", "jpeg"):
        im.save(buf, format="JPEG", quality=quality, subsampling=0, optimize=True)
        return buf.getvalue(), "image/jpeg"
    im.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue(), "image/png"


def _pad4(b: bytes, pad: bytes = b"\x00") -> bytes:
    return b + pad * ((4 - len(b) % 4) % 4)


def write_glb(
    path: str | Path,
    positions: np.ndarray,
    normals: np.ndarray,
    uv: np.ndarray,
    faces: np.ndarray,
    image_bytes: bytes,
    mime: str,
    name: str = "twin",
    roughness: float = 0.9,
    extras: dict | None = None,
) -> None:
    pos = np.ascontiguousarray(positions, dtype="<f4")
    nrm = np.ascontiguousarray(normals, dtype="<f4")
    tex = np.ascontiguousarray(uv, dtype="<f4")  # glTF: (0, 0) = top-left of the image, v down: same as ours
    idx = np.ascontiguousarray(faces.reshape(-1), dtype="<u4")
    chunks: list[bytes] = []
    views: list[dict] = []

    def add(data: bytes, target: int | None) -> int:
        offset = sum(len(c) for c in chunks)
        chunks.append(_pad4(data))
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    bv_pos = add(pos.tobytes(), 34962)
    bv_nrm = add(nrm.tobytes(), 34962)
    bv_uv = add(tex.tobytes(), 34962)
    bv_idx = add(idx.tobytes(), 34963)
    bv_img = add(image_bytes, None)
    accessors = [
        {"bufferView": bv_pos, "componentType": 5126, "count": len(pos), "type": "VEC3",
         "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
        {"bufferView": bv_nrm, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
        {"bufferView": bv_uv, "componentType": 5126, "count": len(tex), "type": "VEC2"},
        {"bufferView": bv_idx, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
    ]
    gltf = {
        "asset": {"version": "2.0", "generator": "twintex (Dijital Ikiz twin lab)"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0, "mode": 4}]}],
        "materials": [{
            "name": "twin_skin_cloth",
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                     "roughnessFactor": roughness},
        }],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": bv_img, "mimeType": mime, "name": "baseColor"}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(c) for c in chunks)}],
    }
    if extras:
        gltf["extras"] = extras
    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_bytes = b"".join(chunks)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        f.write(json_bytes)
        f.write(struct.pack("<I4s", len(bin_bytes), b"BIN\x00"))
        f.write(bin_bytes)
