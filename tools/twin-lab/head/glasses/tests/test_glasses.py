"""Analytic stand-in head only: no personal meshes or images."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_glasses as glasses
from write_twin_glb import accessor_array, read_glb, read_glb_bytes, validate_glasses


@pytest.fixture
def head(tmp_path):
    angles = np.arange(80)*2*np.pi/80
    heights = np.linspace(-0.99, 0.99, 50)
    points = np.array([
        [0.08*np.sqrt(1-y*y)*np.cos(a), 0.10+0.12*y, 0.015+0.095*np.sqrt(1-y*y)*np.sin(a)]
        for y in heights for a in angles
    ])
    document = {"asset": {"version": "2.0"}, "bufferViews": [], "accessors": []}
    blob = bytearray()
    def add(values, dtype, component, kind):
        values = np.asarray(values, dtype=dtype)
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": values.nbytes})
        blob.extend(values.tobytes())
        document["accessors"].append({"bufferView": len(document["bufferViews"])-1, "componentType": component, "type": kind, "count": len(values)})
        return len(document["accessors"])-1
    attrs = {
        "POSITION": add(points, "<f4", 5126, "VEC3"),
        "JOINTS_0": add(np.zeros((len(points), 4)), "<u2", 5123, "VEC4"),
        "WEIGHTS_0": add(np.tile([1, 0, 0, 0], (len(points), 1)), "<f4", 5126, "VEC4"),
    }
    document.update(
        buffers=[{"byteLength": len(blob)}],
        nodes=[{"name": "Root", "translation": [0.03, 1.5, -0.01], "children": [1]},
               {"name": "head", "translation": [0, 0.1, 0.02]},
               {"name": "scan", "mesh": 0, "skin": 0, "translation": [0.03, 1.6, 0.01]}],
        skins=[{"joints": [1]}], meshes=[{"primitives": [{"attributes": attrs}]}],
    )
    raw = json.dumps(document).encode()
    raw += b" " * (-len(raw)%4)
    path = tmp_path / "twin.glb"
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28+len(raw)+len(blob)) + struct.pack("<I4s", len(raw), b"JSON") + raw + struct.pack("<I4s", len(blob), b"BIN\0") + blob)
    return path


def test_geometry_rigid_binding_and_smooth_normals():
    for lenses in (False, True):
        params = {**glasses.DEFAULTS, "clearLenses": lenses}
        raw = glasses.encode_glasses(params, np.array([0, 0.12, 0.11]), "synthetic")
        assert validate_glasses(raw) == params
        document, blob = read_glb_bytes(raw)
        assert len(document["meshes"]) == 1
        for primitive in document["meshes"][0]["primitives"]:
            attrs = primitive["attributes"]
            p = accessor_array(document, blob, attrs["POSITION"])
            n = accessor_array(document, blob, attrs["NORMAL"])
            f = accessor_array(document, blob, primitive["indices"]).reshape(-1, 3)
            cross = np.cross(p[f[:, 1]]-p[f[:, 0]], p[f[:, 2]]-p[f[:, 0]])
            assert np.linalg.norm(cross, axis=1).min() > 1e-10
            assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-6)
            # All triangle winding agrees with the supplied smooth outward normals.
            assert np.all(np.sum(cross*n[f].mean(axis=1), axis=1) > 0)
        factor = document["materials"][0]["pbrMetallicRoughness"]
        assert factor["metallicFactor"] == 1
        assert factor["roughnessFactor"] == 0.3


def test_geometry_placement_relative_to_synthetic_head_and_parent_transforms(head):
    centre, method = glasses.estimate_placement(head, [])
    assert method == "head-geometry"
    assert abs(centre[0]) < 0.002
    assert 0.115 < centre[1] < 0.13
    assert 0.10 < centre[2] < 0.12
    out = head.parent / "glasses.glb"
    glasses.make_glasses(head, out, head.parent / "not-yet-written.json")
    document, blob = read_glb(out)
    primitive = document["meshes"][0]["primitives"][0]
    p = accessor_array(document, blob, primitive["attributes"]["POSITION"])
    assert p[:, 0].min() < -0.067
    assert p[:, 0].max() > 0.067
    assert p[:, 2].min() < centre[2]-0.138
    assert p[:, 1].min() < centre[1]-0.025
    assert p[:, 1].max() < centre[1]+0.026


def test_3d_eye_and_nose_landmarks_take_precedence_and_offset_applies(head):
    report = head.parent / "refine/refine_report.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"landmarks3d": {
        "leftEye": [0.06, 1.73, 0.11], "rightEye": [0, 1.73, 0.11],
        "noseBridge": [0.031, 1.725, 0.12],
    }}))
    centre, method = glasses.estimate_placement(head, [report])
    assert method == "landmarks3d"
    assert np.allclose(centre, [0.001, 0.13, 0.114])
    params = head.parent / "params.json"
    params.write_text(json.dumps({"vertical_offset": 0.007}))
    out = head.parent / "glasses.glb"
    glasses.make_glasses(head, out, params)
    document, blob = read_glb(out)
    p = accessor_array(document, blob, document["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
    assert p[:, 1].max() == pytest.approx(0.137+0.0246, abs=2e-6)


def test_2d_landmarks_are_ignored_and_local_landmarks_supported(head):
    report = head.parent / "report.json"
    report.write_text(json.dumps({"landmarks": {"leftEye": [400, 300], "rightEye": [500, 300]}}))
    assert glasses.estimate_placement(head, [report])[1] == "head-geometry"
    report.write_text(json.dumps({"faceLandmarksM": {"coordinateSpace": "head-local", "left_eye": [-0.03, 0.14, 0.1], "right_eye": [0.03, 0.14, 0.1]}}))
    assert np.allclose(glasses.estimate_placement(head, [report])[0], [0, 0.14, 0.104])


@pytest.mark.parametrize("raw", [{"thickness": 0}, {"outerRadius": float("nan")}, {"colour": "red"}, {"lensShape": "square"}, {"frameWidth": 0.08}, {"clearLenses": 1}])
def test_invalid_params_rejected(tmp_path, raw):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        glasses.read_params(path)


def test_output_cannot_overwrite_twin(head):
    with pytest.raises(ValueError, match="overwrite"):
        glasses.make_glasses(head, head)
