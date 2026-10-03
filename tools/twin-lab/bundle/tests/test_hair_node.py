"""The separate ``dtHair`` node of a twin bundle (synthetic data only): contract and rejection of broken hair."""

import io
import json

import numpy as np
import pytest
from conftest import bundle
from PIL import Image
from test_bundle import save_glb, synthetic  # noqa: F401  (fixture)

HAIR_EXTRAS = {
    "format": "rcov-groot-bvar/1",
    "colorHex": "#2a1e18",
    "rootHex": "#1f1611",
    "tipHex": "#3a2a20",
    "cardCount": 2,
}


def strand_atlas(size=64):
    """A synthetic strand data atlas: R coverage stripes, G root to tip (rows), B variation, A = min(1, 2.5 R)."""
    x = np.arange(size)[None, :].repeat(size, 0)
    r = np.clip(0.45 + 0.5 * np.sin(x * 0.9), 0, 1)
    g = np.linspace(0, 1, size)[:, None].repeat(size, 1)
    b = np.full((size, size), 0.5)
    a = np.minimum(1.0, 2.5 * r)
    return np.rint(np.dstack((r, g, b, a)) * 255).astype(np.uint8)


def add_hair(document, blob, **options):
    """Append a hair node (two cards: 8 vertices, 4 triangles) sharing the body's skin; ``options`` break one thing."""
    blob = bytearray(blob)
    views, accessors = document["bufferViews"], document["accessors"]

    def view(data):
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        return len(views) - 1

    def accessor(array, component, kind):
        accessors.append(
            {
                "bufferView": view(np.ascontiguousarray(array).tobytes()),
                "componentType": component,
                "count": len(array),
                "type": kind,
            }
        )
        return len(accessors) - 1

    quad = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    positions = np.vstack((quad * 0.01, quad * 0.01 + [0, 0.02, 0.0])) + [0.0, 1.6, 0.05]
    normals = np.tile(np.array([0, 0, 1], np.float32), (8, 1))
    uv = np.vstack((quad[:, :2] * 0.5, quad[:, :2] * 0.5 + 0.5)).astype(np.float32)
    names = [bone["name"] for bone in json.loads(bundle.RIG.read_text())["bones"]]
    head = names.index("head")
    joints = np.zeros((8, 4), np.uint16)
    joints[:, 0] = head
    weights = np.zeros((8, 4), np.float32)
    weights[:, 0] = 1.0
    indices = np.array([0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7], np.uint16)
    if options.get("uv") == "range":
        uv[0] = [1.5, 0.2]
    if options.get("weights") == "sum":
        weights[:, 0] = 0.5
    attrs = {
        "POSITION": accessor(positions, 5126, "VEC3"),
        "NORMAL": accessor(normals, 5126, "VEC3"),
        "TEXCOORD_0": accessor(uv, 5126, "VEC2"),
        "JOINTS_0": accessor(joints, 5123, "VEC4"),
        "WEIGHTS_0": accessor(weights, 5126, "VEC4"),
    }
    if options.get("attributes") == "no-normal":
        del attrs["NORMAL"]
    atlas = strand_atlas()
    if options.get("atlas") == "black":
        atlas[..., :3] = 0
    png = io.BytesIO()
    Image.fromarray(atlas, "RGBA").save(png, format="PNG")
    document["images"].append({"bufferView": view(png.getvalue()), "mimeType": "image/png"})
    document["textures"].append({"source": len(document["images"]) - 1, "sampler": 0})
    extras = {**HAIR_EXTRAS, **options.get("extras", {})}
    material = {
        "name": options.get("material_name", "dtHair"),
        "pbrMetallicRoughness": {"baseColorTexture": {"index": len(document["textures"]) - 1}},
        "alphaMode": options.get("alpha_mode", "MASK"),
        "alphaCutoff": 0.5,
        "doubleSided": True,
        "extras": {"dtHair": extras},
    }
    if options.get("drop_extras"):
        del material["extras"]
    document["materials"].append(material)
    primitive = {"attributes": attrs, "indices": accessor(indices, 5123, "SCALAR"), "mode": 4}
    primitive["material"] = 0 if options.get("material") == "body" else len(document["materials"]) - 1
    document["meshes"].append({"name": "dtHair", "primitives": [primitive]})
    skin = 0
    if options.get("skin") == "other":
        document["skins"].append(dict(document["skins"][0]))
        skin = 1
    document["nodes"].append({"name": "dtHair", "mesh": len(document["meshes"]) - 1, "skin": skin})
    document["scenes"][0]["nodes"].append(len(document["nodes"]) - 1)
    document["asset"].setdefault("extras", {})["dtHairNode"] = options.get("node", "dtHair")
    return document, bytes(blob)


def with_hair(paths, **options):
    rigged, twin, mapping, out = paths
    document, blob = bundle.read_glb(rigged)
    document, blob = add_hair(document, blob, **options)
    save_glb(rigged, document, blob)
    return rigged, twin, mapping, out


def test_hair_node_round_trips_and_is_validated(synthetic):  # noqa: F811
    rigged, twin, mapping, out = with_hair(synthetic)
    original, old_blob = bundle.read_glb(rigged)
    result = bundle.write_bundle(rigged, twin, mapping, out, shape="synthetic", license_name="CC0")
    assert result["vertices"] == 8 and result["bones"] == 53
    assert result["hair"] == {"node": "dtHair", "vertices": 8, "triangles": 4, "cardCount": 2, "atlas": [64, 64]}
    document, blob = bundle.read_glb(out)
    assert document["asset"]["extras"]["dtHairNode"] == "dtHair"
    for key in ("meshes", "skins", "nodes", "images", "materials", "accessors", "textures"):
        assert document[key] == original[key]  # the hair node and its material pass through unchanged
    assert blob[: len(old_blob)] == old_blob
    assert bundle.validate_bundle(out) == result
    # the body is still found next to the hair: its tone, mapping and vertex count are those of the body mesh
    assert document["asset"]["extras"]["dtTwin"]["skinToneHex"] == "#b47850"
    assert bundle.validate_bundle(out)["vertices"] == json.loads(twin.read_text())["mapping"]["twinVertexCount"]


@pytest.mark.parametrize(
    "options,match",
    [
        ({"skin": "other"}, "same skin"),
        ({"material": "body"}, "own material"),
        ({"material_name": "hairy"}, "named dtHair"),
        ({"alpha_mode": "OPAQUE"}, "alphaMode MASK"),
        ({"drop_extras": True}, "extras.dtHair"),
        ({"extras": {"format": "other/1"}}, "extras.dtHair"),
        ({"extras": {"colorHex": "red"}}, "colorHex"),
        ({"extras": {"cardCount": 0}}, "cardCount"),
        ({"uv": "range"}, "geometry"),
        ({"weights": "sum"}, "geometry"),
        ({"attributes": "no-normal"}, "NORMAL"),
        ({"atlas": "black"}, "coverage"),
        ({"node": "missing"}, "exactly one"),
    ],
)
def test_broken_hair_is_rejected_without_replacing_the_output(synthetic, options, match):  # noqa: F811
    rigged, twin, mapping, out = with_hair(synthetic, **options)
    out.write_bytes(b"existing")
    with pytest.raises(ValueError, match=match):
        bundle.write_bundle(rigged, twin, mapping, out)
    assert out.read_bytes() == b"existing"


def test_a_bundle_without_the_marker_has_no_hair_section(synthetic):  # noqa: F811
    rigged, twin, mapping, out = synthetic
    result = bundle.write_bundle(rigged, twin, mapping, out)
    assert "hair" not in result and bundle.hair_node_name(bundle.read_glb(out)[0]) is None


def test_hair_node_name_must_be_a_string(synthetic):  # noqa: F811
    rigged, *_ = synthetic
    document, _ = bundle.read_glb(rigged)
    document["asset"].setdefault("extras", {})["dtHairNode"] = 7
    with pytest.raises(ValueError, match="node name"):
        bundle.hair_node_name(document)
