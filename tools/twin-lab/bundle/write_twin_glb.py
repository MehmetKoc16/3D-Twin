"""Embed a twin package in a GLB without rewriting its existing JSON/BIN data."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from pygltflib import GLTF2

REPO = Path(__file__).resolve().parents[3]
RIG = REPO / "apps/web/public/assets/body/rig.json"
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


def mesh_info(
    document: dict, twin: dict, rig_path: Path
) -> tuple[dict, list[str], int]:
    nodes = [node for node in document["nodes"] if "mesh" in node]
    if len(nodes) != 1 or "skin" not in nodes[0]:
        raise ValueError("The twin loader requires exactly one skinned mesh instance")
    node = nodes[0]
    primitives = document["meshes"][node["mesh"]]["primitives"]
    if len(primitives) != 1:
        raise ValueError("The twin loader requires a single mesh primitive")
    primitive = primitives[0]
    attrs = primitive["attributes"]
    if not {"POSITION", "JOINTS_0", "WEIGHTS_0"} <= attrs.keys():
        raise ValueError("Mesh must contain positions and skin weights")
    names = [
        document["nodes"][i].get("name")
        for i in document["skins"][node["skin"]]["joints"]
    ]
    expected = [
        bone["name"]
        for bone in json.loads(rig_path.read_text(encoding="utf-8"))["bones"]
    ]
    if names != expected or twin.get("boneOrder") != expected:
        raise ValueError(
            "Skin joints / twin boneOrder do not match rig.json bone names and order"
        )
    count = document["accessors"][attrs["POSITION"]]["count"]
    if count <= 0 or twin.get("version") != 1:
        raise ValueError("Expected nonempty mesh and twin.json version 1")
    mapping = twin.get("mapping", {})
    if mapping.get("twinVertexCount") != count or mapping.get("format") != "uint32le":
        raise ValueError(
            "twin.json mapping must be uint32le with the mesh vertex count"
        )
    for key in ("JOINTS_0", "WEIGHTS_0"):
        if document["accessors"][attrs[key]]["count"] != count:
            raise ValueError("Skin attribute count differs from vertex count")
    return primitive, names, count


def base_color_info(document: dict, primitive: dict) -> tuple[dict, dict]:
    material = document["materials"][primitive["material"]]
    info = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if info is None:
        raise ValueError("Mesh requires an embedded baseColor texture")
    texture = document["textures"][info["index"]]
    return info, texture


def embedded_image(document: dict, blob: bytes, texture: dict) -> bytes:
    image = document["images"][texture["source"]]
    if "bufferView" in image:
        return view_bytes(document, blob, image["bufferView"])
    uri = image.get("uri", "")
    if uri.startswith("data:") and ";base64," in uri:
        return base64.b64decode(uri.split(",", 1)[1], validate=True)
    raise ValueError(
        "Texture must be embedded; external image references are unsupported"
    )


def wrap_uv(values: np.ndarray, mode: int) -> np.ndarray:
    if mode == 33071:  # CLAMP_TO_EDGE
        return np.clip(values, 0, 1)
    if mode == 33648:  # MIRRORED_REPEAT
        return 1 - np.abs(np.mod(values, 2) - 1)
    if mode != 10497:
        raise ValueError("Unsupported texture wrap mode")
    return np.mod(values, 1)


def sample_skin_tone(
    document: dict,
    blob: bytes,
    primitive: dict,
    names: list[str],
    texture_path: Path | None = None,
) -> str:
    info, texture = base_color_info(document, primitive)
    image_source = (
        texture_path
        if texture_path is not None
        else io.BytesIO(embedded_image(document, blob, texture))
    )
    with Image.open(image_source) as image:
        pixels = np.asarray(image.convert("RGBA"))
    attrs = primitive["attributes"]
    joints = accessor_array(document, blob, attrs["JOINTS_0"])
    weights = accessor_array(document, blob, attrs["WEIGHTS_0"])
    if joints.shape[1] != 4 or weights.shape != joints.shape:
        raise ValueError("Expected VEC4 joint indices and weights")
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Invalid skin weights")
    if (
        (joints < 0).any()
        or (joints >= len(names)).any()
        or joints.dtype.kind not in "iu"
    ):
        raise ValueError("Invalid joint indices")
    selected = np.zeros(len(joints), dtype=bool)
    for name in ("lowerarm_l", "lowerarm_r"):
        selected |= (
            np.sum(np.where(joints == names.index(name), weights, 0), axis=1) > 0.5
        )
    transform = info.get("extensions", {}).get("KHR_texture_transform", {})
    coord = transform.get("texCoord", info.get("texCoord", 0))
    key = f"TEXCOORD_{coord}"
    if key not in attrs:
        raise ValueError("BaseColor texture requires matching UVs")
    uv = accessor_array(document, blob, attrs[key])
    if uv.shape != (len(joints), 2) or not np.isfinite(uv).all():
        raise ValueError("Invalid UV coordinates")
    uv = uv[selected].astype(np.float64)
    uv *= transform.get("scale", [1, 1])
    angle = transform.get("rotation", 0)
    cosine, sine = np.cos(angle), np.sin(angle)
    uv = uv @ np.array([[cosine, sine], [-sine, cosine]])
    uv += transform.get("offset", [0, 0])
    sampler = (
        document.get("samplers", [])[texture["sampler"]] if "sampler" in texture else {}
    )
    height, width = pixels.shape[:2]
    # glTF UVs use a top-left origin (v grows down); do not flip the image.
    x = np.minimum(
        (wrap_uv(uv[:, 0], sampler.get("wrapS", 10497)) * width).astype(int), width - 1
    )
    y = np.minimum(
        (wrap_uv(uv[:, 1], sampler.get("wrapT", 10497)) * height).astype(int),
        height - 1,
    )
    texels = pixels.reshape(-1, 4)[np.unique(y * width + x)]
    rgb = texels[:, :3].astype(np.float64)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    valid = (texels[:, 3] >= 128) & (luminance > 20) & (luminance < 235)
    rgb, luminance = rgb[valid], luminance[valid]
    if not len(rgb):
        raise ValueError(
            "No usable forearm texels (weight > 0.5, opaque, luminance 20..235)"
        )
    low, high = np.percentile(luminance, [5, 95], method="nearest")
    rgb = rgb[(luminance >= low) & (luminance <= high)]
    median = np.floor(np.median(rgb, axis=0) + 0.5).astype(int)
    return "#" + "".join(f"{channel:02x}" for channel in median)


def validate_bundle(
    path: Path,
    *,
    rig_path: Path = RIG,
    expected_twin: dict | None = None,
    expected_mapping: bytes | None = None,
) -> dict:
    """Re-read with two readers and validate the single-file contract, returning only counts."""
    parsed = GLTF2().load_binary(str(path))
    document, blob = read_glb(path)
    if parsed.binary_blob()[: len(blob)] != blob:
        raise ValueError("GLB readers disagree on the binary buffer")
    extras = document.get("asset", {}).get("extras", {}).get("dtTwin")
    if not isinstance(extras, dict) or extras.get("version") != 1:
        raise ValueError("Missing asset.extras.dtTwin version 1")
    twin = extras.get("twin")
    if not isinstance(twin, dict) or (
        expected_twin is not None and twin != expected_twin
    ):
        raise ValueError("Embedded twin.json differs from the input")
    primitive, names, count = mesh_info(document, twin, rig_path)
    if not re.fullmatch(r"#[0-9a-f]{6}", extras.get("skinToneHex", "")):
        raise ValueError("Invalid skinToneHex")
    provenance = extras.get("provenance", {})
    if any(
        not isinstance(provenance.get(key), str) or not provenance[key].strip()
        for key in ("shape", "license", "createdAt")
    ):
        raise ValueError("Missing provenance")
    timestamp = datetime.fromisoformat(provenance["createdAt"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("createdAt must include an ISO-8601 timezone")
    mapping = extras.get("mh2twin", {})
    if mapping.get("componentType") != "uint32" or mapping.get("count") != count:
        raise ValueError("Mapping component type / count mismatch")
    index = mapping.get("bufferView")
    if type(index) is not int or not 0 <= index < len(document["bufferViews"]):
        raise ValueError("Invalid mapping buffer view index")
    view = document["bufferViews"][index]
    if view.get("byteOffset", 0) % 4 or view.get("byteStride") is not None:
        raise ValueError("Mapping must be aligned and tightly packed")
    data = view_bytes(document, blob, index)
    if len(data) != count * 4:
        raise ValueError("Mapping buffer view length must equal vertex count * 4 bytes")
    if expected_mapping is not None and data != expected_mapping:
        raise ValueError("Embedded mapping differs from mh2twin.bin")
    render_count = twin["mapping"].get("renderVertexCount", 0)
    if render_count <= 0 or np.any(np.frombuffer(data, dtype="<u4") >= render_count):
        raise ValueError("Mapping contains an out-of-range MakeHuman render vertex")
    for image in document.get("images", []):
        if "uri" in image and not image["uri"].startswith("data:"):
            raise ValueError("Single-file bundle contains an external image")
    for i in range(len(document["bufferViews"])):
        view_bytes(document, blob, i)
    _, texture = base_color_info(document, primitive)
    embedded_image(document, blob, texture)
    accessories = extras.get("accessories", [])
    if not isinstance(accessories, list) or len(accessories) > 1:
        raise ValueError("Invalid accessories section")
    for accessory in accessories:
        if accessory.get("id") != "glasses" or accessory.get("bone") != "head":
            raise ValueError("Unsupported accessory")
        index = accessory.get("mesh", {}).get("bufferView")
        if type(index) is not int or not 0 <= index < len(document["bufferViews"]):
            raise ValueError("Invalid accessory buffer view")
        params = validate_glasses(view_bytes(document, blob, index))
        if params != accessory.get("params"):
            raise ValueError("Accessory params differ from the embedded GLB")
    return {"vertices": count, "bones": len(names), "mappingBytes": len(data)}


def write_bundle(
    rigged: Path,
    twin_path: Path,
    mapping_path: Path,
    out: Path,
    *,
    texture: Path | None = None,
    shape: str = "unspecified",
    license_name: str = "unspecified",
    rig_path: Path = RIG,
    glasses: Path | None = None,
) -> dict:
    inputs = [rigged, twin_path, mapping_path] + ([texture] if texture else [])
    if glasses is not None:
        inputs.append(glasses)
    if out.resolve() in [path.resolve() for path in inputs]:
        raise ValueError("Output must not overwrite an input")
    # Personal artifacts can never be copied into tracked paths, even accidentally.
    private_root = REPO / "user-data"
    if any(
        path.resolve().is_relative_to(private_root) for path in inputs
    ) and not out.resolve().is_relative_to(private_root):
        raise ValueError("Personal bundle output must stay under user-data/")
    document, blob = read_glb(rigged)
    twin = json.loads(twin_path.read_text(encoding="utf-8"))
    primitive, names, count = mesh_info(document, twin, rig_path)
    data = mapping_path.read_bytes()
    if len(data) != count * 4:
        raise ValueError("mh2twin.bin length must equal vertex count * 4 bytes")
    tone = sample_skin_tone(document, blob, primitive, names, texture)
    blob += b"\0" * (-len(blob) % 4)
    views = document.setdefault("bufferViews", [])
    mapping_index = len(views)
    views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
    blob += data
    document["buffers"][0]["byteLength"] = len(blob)
    document["asset"].setdefault("extras", {})["dtTwin"] = {
        "version": 1,
        "twin": twin,
        "skinToneHex": tone,
        "mh2twin": {
            "bufferView": mapping_index,
            "count": count,
            "componentType": "uint32",
        },
        "provenance": {
            "shape": shape,
            "license": license_name,
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }
    if glasses is not None:
        accessory = glasses.read_bytes()
        params = validate_glasses(accessory)
        blob += b"\0" * (-len(blob) % 4)
        index = len(views)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(accessory)})
        blob += accessory
        document["buffers"][0]["byteLength"] = len(blob)
        document["asset"]["extras"]["dtTwin"]["accessories"] = [
            {"id": "glasses", "bone": "head", "mesh": {"bufferView": index}, "params": params}
        ]
    encoded = json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    padded = blob + b"\0" * (-len(blob) % 4)
    raw = (
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(padded))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(padded), b"BIN\0")
        + padded
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=out.parent, suffix=".glb", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
        result = validate_bundle(
            temporary, rig_path=rig_path, expected_twin=twin, expected_mapping=data
        )
        temporary.replace(out)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigged", required=True, type=Path)
    parser.add_argument("--twin", required=True, type=Path)
    parser.add_argument("--mh2twin", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--glasses", type=Path, help="optional head-local glasses GLB")
    parser.add_argument(
        "--texture",
        type=Path,
        help="optional sampling image; the embedded GLB texture stays unchanged",
    )
    parser.add_argument(
        "--shape", default="unspecified", help="shape source for provenance"
    )
    parser.add_argument(
        "--license", default="unspecified", help="shape license for provenance"
    )
    args = parser.parse_args(argv)
    try:
        write_bundle(
            args.rigged,
            args.twin,
            args.mh2twin,
            args.out,
            texture=args.texture,
            shape=args.shape,
            license_name=args.license,
            glasses=args.glasses,
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        # Do not dump private JSON, texture samples, or geometry in error logs.
        message = (
            str(exc)
            if isinstance(exc, (OSError, ValueError))
            else "Invalid GLB / twin contract structure"
        )
        print(f"Bundle failed ({type(exc).__name__}): {message}")
        return 1
    print(f"Bundle validation: PASS; size: {args.out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
