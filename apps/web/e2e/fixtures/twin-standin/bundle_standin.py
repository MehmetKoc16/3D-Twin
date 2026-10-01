"""Bundle only the adjacent NON-personal CC0 fixture; never accepts a user-data path.

Run with tools/twin-lab/rig/.venv/Scripts/python (Pillow is already installed there).
"""
import io
import json
from pathlib import Path
import statistics
import struct

from PIL import Image


def main():
    folder = Path(__file__).resolve().parent
    source = (folder / "rigged.glb").read_bytes()
    json_length = struct.unpack_from("<I", source, 12)[0]
    gltf = json.loads(source[20:20 + json_length])
    bin_start = 28 + json_length
    bin_length = struct.unpack_from("<I", source, 20 + json_length)[0]
    binary = source[bin_start:bin_start + bin_length]

    def attribute(name, formats, components):
        primitive = gltf["meshes"][0]["primitives"][0]
        accessor = gltf["accessors"][primitive["attributes"][name]]
        view = gltf["bufferViews"][accessor["bufferView"]]
        fmt = "<" + formats[accessor["componentType"]] * components
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride", struct.calcsize(fmt))
        return [struct.unpack_from(fmt, binary, offset + i * stride) for i in range(accessor["count"])]

    image_view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    start = image_view.get("byteOffset", 0)
    image = Image.open(io.BytesIO(binary[start:start + image_view["byteLength"]])).convert("RGB")
    joint_names = [gltf["nodes"][node]["name"] for node in gltf["skins"][0]["joints"]]
    joints = attribute("JOINTS_0", {5121: "B", 5123: "H"}, 4)
    weights = attribute("WEIGHTS_0", {5126: "f"}, 4)
    uvs = attribute("TEXCOORD_0", {5126: "f"}, 2)
    samples = []
    for indices, skin, uv in zip(joints, weights, uvs):
        dominant = max(range(4), key=lambda i: skin[i])
        if joint_names[indices[dominant]] not in ("lowerarm_l", "lowerarm_r"):
            continue
        x = min(image.width - 1, max(0, round(uv[0] * (image.width - 1))))
        y = min(image.height - 1, max(0, round(uv[1] * (image.height - 1))))
        samples.append(image.getpixel((x, y)))
    assert samples, "stand-in must contain forearm texture samples"
    skin_hex = "#" + "".join(f"{int(statistics.median(pixel[c] for pixel in samples)):02x}" for c in range(3))
    mapping = (folder / "mh2twin.bin").read_bytes()
    binary += b"\0" * (-len(binary) % 4)
    mapping_view = len(gltf["bufferViews"])
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(mapping)})
    binary += mapping
    gltf["buffers"][0]["byteLength"] = len(binary)
    gltf["asset"].setdefault("extras", {})["dtTwin"] = {
        "version": 1,
        "twin": json.loads((folder / "twin.json").read_text(encoding="utf-8")),
        "skinToneHex": skin_hex,
        "mh2twin": {"bufferView": mapping_view, "count": len(mapping) // 4, "componentType": "uint32"},
        "provenance": {"shape": "non-personal MakeHuman stand-in", "license": "CC0", "createdAt": "2026-10-01T00:00:00Z"},
    }
    encoded = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 28 + len(encoded) + len(binary)
    (folder / "twin.glb").write_bytes(struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        + struct.pack("<II", len(binary), 0x004E4942) + binary)
    print(f"Built CC0 stand-in twin.glb: {total} bytes, {len(mapping) // 4} mappings, synthetic forearm median {skin_hex}")


if __name__ == "__main__":
    main()
