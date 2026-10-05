"""Dependency-free accessory GLB parsing and validation (numpy only)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

DTYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: Path) -> tuple[dict, bytes]:
    """Keep unknown extensions and extras intact, including nested vendor fields."""
    return read_glb_bytes(path.read_bytes())


def read_glb_bytes(raw: bytes) -> tuple[dict, bytes]:
    if len(raw) < 20 or struct.unpack_from("<4sII", raw) != (b"glTF", 2, len(raw)):
        raise ValueError("Invalid GLB header")
    chunks = []
    offset = 12
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError("Truncated GLB chunk header")
        size, kind = struct.unpack_from("<I4s", raw, offset)
        offset += 8
        if size % 4 or offset + size > len(raw):
            raise ValueError("Invalid GLB chunk length")
        chunks.append((kind, raw[offset : offset + size]))
        offset += size
    if [kind for kind, _ in chunks] != [b"JSON", b"BIN\0"]:
        raise ValueError("Expected one JSON chunk and one embedded BIN chunk")
    document = json.loads(chunks[0][1])
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or buffers[0].get("uri"):
        raise ValueError("Expected a single embedded buffer")
    length = buffers[0]["byteLength"]
    if not 0 <= len(chunks[1][1]) - length <= 3:
        raise ValueError("GLB buffer length does not match BIN chunk")
    return document, chunks[1][1][:length]


def validate_glasses(raw: bytes) -> dict:
    """Validate the head-local, self-contained accessory before embedding it."""
    document, blob = read_glb_bytes(raw)
    metadata = document.get("asset", {}).get("extras", {}).get("dtAccessory", {})
    if (
        metadata.get("id") != "glasses"
        or metadata.get("bone") != "head"
        or metadata.get("coordinateSpace") != "head-local"
        or not isinstance(metadata.get("params"), dict)
    ):
        raise ValueError("Glasses require head-local dtAccessory metadata and params")
    nodes = document.get("nodes", [])
    meshes = [node for node in nodes if "mesh" in node]
    if len(meshes) != 1 or "skin" not in meshes[0]:
        raise ValueError("Glasses require one rigidly skinned mesh")
    # The web mounts local vertices directly under its own head, ignoring this standalone skin.
    for node in nodes:
        if (
            node.get("translation", [0, 0, 0]) != [0, 0, 0]
            or node.get("rotation", [0, 0, 0, 1]) != [0, 0, 0, 1]
            or node.get("scale", [1, 1, 1]) != [1, 1, 1]
            or ("matrix" in node and node["matrix"] != np.eye(4).ravel().tolist())
        ):
            raise ValueError("Glasses nodes must have identity transforms")
    skin = document["skins"][meshes[0]["skin"]]
    if len(skin["joints"]) != 1 or nodes[skin["joints"][0]].get("name") != "head":
        raise ValueError("Glasses must bind only to head")
    inverses = accessor_array(document, blob, skin["inverseBindMatrices"])
    if inverses.shape != (1, 16) or not np.allclose(inverses[0], np.eye(4).ravel()):
        raise ValueError("Glasses require an identity head-local inverse bind")
    primitives = document["meshes"][meshes[0]["mesh"]]["primitives"]
    if not primitives:
        raise ValueError("Glasses mesh is empty")
    for primitive in primitives:
        attrs = primitive["attributes"]
        positions = accessor_array(document, blob, attrs["POSITION"])
        normals = accessor_array(document, blob, attrs["NORMAL"])
        joints = accessor_array(document, blob, attrs["JOINTS_0"])
        weights = accessor_array(document, blob, attrs["WEIGHTS_0"])
        indices = accessor_array(document, blob, primitive["indices"]).ravel()
        if (
            primitive.get("mode", 4) != 4
            or positions.shape[1] != 3
            or not np.isfinite(positions).all()
            or normals.shape != positions.shape
            or not np.isfinite(normals).all()
            or joints.shape != (len(positions), 4)
            or np.any(joints != 0)
            or weights.shape != joints.shape
            or not np.all(weights == [1, 0, 0, 0])
            or len(indices) % 3
            or indices.dtype.kind not in "iu"
            or np.any(indices >= len(positions))
        ):
            raise ValueError("Invalid glasses geometry or rigid head weights")
    for collection in ("buffers", "images"):
        if any("uri" in item for item in document.get(collection, [])):
            raise ValueError("Glasses must embed all resources")
    if document.get("extensionsRequired"):
        raise ValueError("Glasses must not require external decoders")
    return metadata["params"]


def view_bytes(document: dict, blob: bytes, index: int) -> bytes:
    view = document["bufferViews"][index]
    start, size = view.get("byteOffset", 0), view["byteLength"]
    if view.get("buffer", 0) != 0 or start < 0 or size < 0 or start + size > len(blob):
        raise ValueError("Buffer view is outside the embedded buffer")
    return blob[start : start + size]


def accessor_array(document: dict, blob: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    if accessor.get("sparse") or "bufferView" not in accessor:
        raise ValueError("Sparse or bufferless accessors are unsupported")
    dtype = np.dtype(DTYPES[accessor["componentType"]])
    width = WIDTHS[accessor["type"]]
    view = document["bufferViews"][accessor["bufferView"]]
    data = view_bytes(document, blob, accessor["bufferView"])
    offset, count = accessor.get("byteOffset", 0), accessor["count"]
    stride = view.get("byteStride", dtype.itemsize * width)
    if count <= 0 or offset < 0 or stride < dtype.itemsize * width:
        raise ValueError("Invalid accessor layout")
    if offset + (count - 1) * stride + dtype.itemsize * width > len(data):
        raise ValueError("Accessor is outside its buffer view")
    values = np.ndarray(
        (count, width),
        dtype=dtype,
        buffer=data,
        offset=offset,
        strides=(stride, dtype.itemsize),
    ).copy()
    if accessor.get("normalized") and dtype.kind in "iu":
        values = values.astype(np.float64) / np.iinfo(dtype).max
        if dtype.kind == "i":
            values = np.maximum(values, -1)
    return values
