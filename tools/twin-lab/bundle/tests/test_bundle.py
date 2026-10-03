import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import bundle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "head/glasses"))
import make_glasses


def save_glb(path, document, blob):
    document["buffers"] = [{"byteLength": len(blob)}]
    encoded = json.dumps(document).encode()
    encoded += b" " * (-len(encoded) % 4)
    blob += b"\0" * (-len(blob) % 4)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(blob))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + blob
    )


@pytest.fixture
def synthetic(tmp_path):
    names = [bone["name"] for bone in json.loads(bundle.RIG.read_text())["bones"]]
    n = 8
    blob = bytearray()
    views, accessors = [], []

    def add_view(data, stride=None):
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append(
            {
                "buffer": 0,
                "byteOffset": len(blob),
                "byteLength": len(data),
                **({"byteStride": stride} if stride else {}),
            }
        )
        blob.extend(data)
        return len(views) - 1

    def add_accessor(view, component, kind, count=n, offset=0, normalized=False):
        accessors.append(
            {
                "bufferView": view,
                "byteOffset": offset,
                "componentType": component,
                "count": count,
                "type": kind,
                "normalized": normalized,
            }
        )
        return len(accessors) - 1

    position = add_accessor(
        add_view(np.zeros((n, 3), dtype="<f4").tobytes()), 5126, "VEC3"
    )
    interleaved = bytearray()
    for i in range(n):
        bone = names.index("lowerarm_l" if i < 3 else "lowerarm_r")
        # Last two vertices are just below 0.5 and must be excluded.
        weight = 153 if i < 6 else 127
        interleaved.extend(
            struct.pack("<4H4B", bone, 0, 0, 0, weight, 255 - weight, 0, 0)
        )
    skin = add_view(interleaved, stride=12)
    joints = add_accessor(skin, 5123, "VEC4")
    weights = add_accessor(skin, 5121, "VEC4", offset=8, normalized=True)
    # Interleaved UVs with both a bufferView offset and an accessor byteOffset.
    uv_bytes = b"".join(
        struct.pack("<I2f", 0, (i % 4 + 0.5) / 4, (i // 4 + 0.5) / 2) for i in range(n)
    )
    uv = add_accessor(add_view(uv_bytes, stride=12), 5126, "VEC2", offset=4)
    ibm = add_accessor(
        add_view(np.tile(np.eye(4, dtype="<f4"), (len(names), 1, 1)).tobytes()),
        5126,
        "MAT4",
        count=len(names),
    )
    pixels = np.array(
        [
            [[0, 0, 0], [255, 255, 255], [180, 120, 80], [180, 120, 80]],
            [[180, 120, 80], [180, 120, 80], [250, 0, 0], [0, 0, 250]],
        ],
        dtype=np.uint8,
    )
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    image_view = add_view(output.getvalue())
    document = {
        "asset": {"version": "2.0", "extras": {"other": {"keep": True}}},
        "extensions": {"VENDOR_preserve": {"payload": [1, 2, 3]}},
        "bufferViews": views,
        "accessors": accessors,
        "nodes": [{"name": name, "translation": [0, 0, 0]} for name in names]
        + [{"name": "Twin", "mesh": 0, "skin": 0}],
        "skins": [{"joints": list(range(len(names))), "inverseBindMatrices": ibm}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position,
                            "JOINTS_0": joints,
                            "WEIGHTS_0": weights,
                            "TEXCOORD_0": uv,
                        },
                        "material": 0,
                        "mode": 0,
                    }
                ]
            }
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "scenes": [{"nodes": list(range(len(names) + 1))}],
        "scene": 0,
    }
    twin = {
        "version": 1,
        "boneOrder": names,
        "mapping": {"format": "uint32le", "twinVertexCount": n, "renderVertexCount": n},
        "additional": {"fullObject": True},
    }
    rigged, twin_path, mapping = [
        tmp_path / name for name in ("rigged.glb", "twin.json", "mh2twin.bin")
    ]
    save_glb(rigged, document, bytes(blob))
    twin_path.write_text(json.dumps(twin))
    mapping.write_bytes(np.arange(n, dtype="<u4").tobytes())
    return rigged, twin_path, mapping, tmp_path / "twin.glb"


def test_synthetic_round_trip_preserves_data_and_forearm_median(synthetic):
    rigged, twin, mapping, out = synthetic
    original, old_blob = bundle.read_glb(rigged)
    result = bundle.write_bundle(
        rigged, twin, mapping, out, shape="synthetic", license_name="CC0"
    )
    document, blob = bundle.read_glb(out)
    extras = document["asset"]["extras"]["dtTwin"]
    assert extras["skinToneHex"] == "#b47850"
    assert extras["twin"] == json.loads(twin.read_text())
    assert document["asset"]["extras"]["other"] == {"keep": True}
    assert document["extensions"] == original["extensions"]
    assert blob[: len(old_blob)] == old_blob
    for key in (
        "meshes",
        "skins",
        "nodes",
        "images",
        "materials",
        "accessors",
        "textures",
    ):
        assert document[key] == original[key]
    assert document["bufferViews"][:-1] == original["bufferViews"]
    assert (
        bundle.view_bytes(document, blob, extras["mh2twin"]["bufferView"])
        == mapping.read_bytes()
    )
    assert result == {"vertices": 8, "bones": 53, "mappingBytes": 32}
    assert bundle.validate_bundle(out) == result


def test_bodyfix_solution_and_achieved_measurements_survive_bundle(synthetic):
    rigged, twin_path, mapping, out = synthetic
    twin = json.loads(twin_path.read_text())
    twin.update(fittedMacros={"gender": 0.61, "muscle": 0.43, "weight": 0.52, "height": 0.6123456789},
                fittedModifiers={"measure/measure-upperarm-length": 0.234567891},
                measurementsCm={"height": 181.08, "armLength": 64.87})
    twin["bodyfix"] = {"version": 1, "fittedMacros": twin["fittedMacros"],
                       "fittedModifiers": twin["fittedModifiers"],
                       "targetsCm": {"height": 181.0, "armLength": 65.0},
                       "achievedCm": twin["measurementsCm"],
                       "residualsCm": {"height": 0.08, "armLength": -0.13}}
    twin_path.write_text(json.dumps(twin))
    bundle.write_bundle(rigged, twin_path, mapping, out, shape="synthetic", license_name="CC0")
    document, _ = bundle.read_glb(out)
    assert document["asset"]["extras"]["dtTwin"]["twin"] == twin


def test_flame_and_unknown_extras_survive_bundle_without_stale_dt_twin(synthetic):
    rigged, twin_path, mapping, out = synthetic
    document, blob = bundle.read_glb(rigged)
    extras = document["asset"]["extras"]
    extras.update(dtFlameHead={"version": 1, "synthetic": {"residual": .002}},
                  dtScanHandsRemoved=True, dtTwin={"stale": True})
    document["extras"] = {"rootVendor": [1, {"keep": True}]}
    document["scenes"][0]["extras"] = {"sceneVendor": "keep"}
    document["nodes"][0]["extras"] = {"nodeVendor": "keep"}
    document["meshes"][0]["extras"] = {"meshVendor": "keep"}
    document["meshes"][0]["primitives"][0]["extras"] = {"primitiveVendor": "keep"}
    document["images"][0]["extras"] = {"imageVendor": "keep"}
    save_glb(rigged, document, blob)
    bundle.write_bundle(rigged, twin_path, mapping, out, shape="synthetic", license_name="CC0")
    packed, _ = bundle.read_glb(out)
    assert packed["asset"]["extras"]["dtFlameHead"] == extras["dtFlameHead"]
    assert packed["asset"]["extras"]["dtScanHandsRemoved"] is True
    assert packed["asset"]["extras"]["dtTwin"]["twin"] == json.loads(twin_path.read_text())
    assert "stale" not in packed["asset"]["extras"]["dtTwin"]
    assert packed["extras"] == document["extras"]
    assert packed["scenes"][0]["extras"] == document["scenes"][0]["extras"]
    assert packed["nodes"][0]["extras"] == document["nodes"][0]["extras"]
    assert packed["meshes"][0]["extras"] == document["meshes"][0]["extras"]
    assert packed["meshes"][0]["primitives"][0]["extras"] == document["meshes"][0]["primitives"][0]["extras"]
    assert packed["images"][0]["extras"] == document["images"][0]["extras"]


def test_optional_glasses_round_trip_preserves_body_and_embeds_valid_rigid_mesh(synthetic):
    rigged, twin, mapping, out = synthetic
    accessory = out.parent / "glasses.glb"
    accessory.write_bytes(make_glasses.encode_glasses(make_glasses.DEFAULTS, np.array([0, 0.12, 0.11]), "synthetic"))
    original, old_blob = bundle.read_glb(rigged)
    bundle.write_bundle(rigged, twin, mapping, out, glasses=accessory)
    document, blob = bundle.read_glb(out)
    extras = document["asset"]["extras"]["dtTwin"]
    glasses = extras["accessories"][0]
    assert glasses["id"] == "glasses" and glasses["bone"] == "head"
    assert glasses["params"] == make_glasses.DEFAULTS
    assert bundle.view_bytes(document, blob, glasses["mesh"]["bufferView"]) == accessory.read_bytes()
    assert blob[:len(old_blob)] == old_blob
    for key in ("meshes", "nodes", "skins", "accessors", "images", "materials"):
        assert document[key] == original[key]
    assert bundle.validate_bundle(out)["vertices"] == 8
    glasses["params"]["thickness"] = 0.003
    save_glb(out, document, blob)
    with pytest.raises(ValueError, match="params differ"):
        bundle.validate_bundle(out)


def test_invalid_accessory_does_not_replace_destination(synthetic):
    rigged, twin, mapping, out = synthetic
    out.write_bytes(b"keep existing output")
    accessory = out.parent / "bad.glb"
    accessory.write_bytes(b"invalid")
    with pytest.raises(ValueError):
        bundle.write_bundle(rigged, twin, mapping, out, glasses=accessory)
    assert out.read_bytes() == b"keep existing output"


def test_optional_sampling_texture_does_not_replace_embedded_image(synthetic):
    rigged, twin, mapping, out = synthetic
    texture = out.parent / "sample.png"
    Image.new("RGB", (4, 2), (80, 140, 180)).save(texture)
    bundle.write_bundle(rigged, twin, mapping, out, texture=texture)
    document, blob = bundle.read_glb(out)
    assert document["asset"]["extras"]["dtTwin"]["skinToneHex"] == "#508cb4"
    original, old_blob = bundle.read_glb(rigged)
    assert bundle.embedded_image(
        document, blob, document["textures"][0]
    ) == bundle.embedded_image(original, old_blob, original["textures"][0])


def test_forearm_threshold_excludes_exactly_half_weight(synthetic):
    rigged, twin, mapping, out = synthetic
    document, blob = bundle.read_glb(rigged)
    primitive = document["meshes"][0]["primitives"][0]
    weights = np.zeros((8, 4), dtype="<f4")
    weights[:, 0] = [0.6] * 6 + [0.5, 0.5]
    weights[:, 1] = 1 - weights[:, 0]
    blob += b"\0" * (-len(blob) % 4)
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": len(blob), "byteLength": weights.nbytes}
    )
    document["accessors"][primitive["attributes"]["WEIGHTS_0"]] = {
        "bufferView": len(document["bufferViews"]) - 1,
        "componentType": 5126,
        "count": 8,
        "type": "VEC4",
    }
    save_glb(rigged, document, blob + weights.tobytes())
    bundle.write_bundle(rigged, twin, mapping, out)
    document, _ = bundle.read_glb(out)
    assert document["asset"]["extras"]["dtTwin"]["skinToneHex"] == "#b47850"


def test_cc0_standin_round_trip(tmp_path):
    fixture = bundle.REPO / "apps/web/e2e/fixtures/twin-standin"
    out = tmp_path / "twin.glb"
    mapping = (fixture / "mh2twin.bin").read_bytes()
    result = bundle.write_bundle(
        fixture / "rigged.glb",
        fixture / "twin.json",
        fixture / "mh2twin.bin",
        out,
        shape="MakeHuman stand-in",
        license_name="CC0",
    )
    assert result == {"vertices": 14517, "bones": 53, "mappingBytes": len(mapping)}
    assert (
        bundle.validate_bundle(
            out,
            expected_twin=json.loads((fixture / "twin.json").read_text()),
            expected_mapping=mapping,
        )
        == result
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("length", "length"),
        ("range", "out-of-range"),
        ("bone", "bone names"),
        ("count", "vertex count"),
        ("external", "external"),
        ("dark", "No usable"),
    ],
)
def test_reject_invalid_input_without_replacing_existing_output(
    synthetic, mutation, match
):
    rigged, twin_path, mapping, out = synthetic
    document, blob = bundle.read_glb(rigged)
    if mutation == "length":
        mapping.write_bytes(b"\0")
    elif mutation == "range":
        mapping.write_bytes(np.full(8, 99, dtype="<u4").tobytes())
    elif mutation == "bone":
        document["nodes"][0]["name"] = "Wrong"
    elif mutation == "count":
        twin = json.loads(twin_path.read_text())
        twin["mapping"]["twinVertexCount"] = 9
        twin_path.write_text(json.dumps(twin))
    elif mutation == "external":
        document["images"][0] = {"uri": "external.png"}
    elif mutation == "dark":
        texture = out.parent / "dark.png"
        Image.new("RGB", (4, 2), (0, 0, 0)).save(texture)
    save_glb(rigged, document, blob)
    out.write_bytes(b"existing")
    with pytest.raises(ValueError, match=match):
        bundle.write_bundle(
            rigged,
            twin_path,
            mapping,
            out,
            texture=texture if mutation == "dark" else None,
        )
    assert out.read_bytes() == b"existing"
    assert not list(out.parent.glob("tmp*.glb"))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("version", 2, "version"),
        ("skinToneHex", "bad", "skinTone"),
        (
            "mh2twin",
            {"bufferView": 999, "count": 8, "componentType": "uint32"},
            "buffer view",
        ),
        ("mh2twin", {"bufferView": 0, "count": 8, "componentType": "uint32"}, "length"),
        (
            "provenance",
            {"shape": "synthetic", "license": "CC0", "createdAt": "2026-01-01"},
            "timezone",
        ),
    ],
)
def test_validator_rejects_corrupt_contract(synthetic, field, value, match):
    rigged, twin, mapping, out = synthetic
    bundle.write_bundle(rigged, twin, mapping, out)
    document, blob = bundle.read_glb(out)
    document["asset"]["extras"]["dtTwin"][field] = value
    save_glb(out, document, blob)
    with pytest.raises(ValueError, match=match):
        bundle.validate_bundle(out)


def test_refuse_overwriting_input(synthetic):
    rigged, twin, mapping, _ = synthetic
    with pytest.raises(ValueError, match="overwrite"):
        bundle.write_bundle(rigged, twin, mapping, rigged)


def test_private_output_guard_without_reading_personal_data(tmp_path):
    private = bundle.REPO / "user-data/nonexistent-synthetic.glb"
    with pytest.raises(ValueError, match="user-data"):
        bundle.write_bundle(
            private, tmp_path / "twin.json", tmp_path / "map.bin", tmp_path / "out.glb"
        )
