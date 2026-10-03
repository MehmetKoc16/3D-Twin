"""Synthetic shapes and metadata only; never read personal scans or tape data."""

import copy
import json
import pickle
import struct
import sys

import numpy as np
import pytest

from bodyfix_solution import read_solution
from rigfit import Fitter
from twin_export import canonicalize_fit, measure_body, write_twin_package


def solution(model):
    macro = {"gender": 0.7123456789, "muscle": 0.4, "weight": 0.6, "height": 0.623456789}
    mods = {"measure/measure-upperarm-length": 0.234567891,
            "head/head-scale-horiz": 0.0000004}
    raw = measure_body(model, model.shape(macro, mods, ground=False))
    allowance = {key: 0.75 for key in raw}
    # Deliberately differ from proxy measurements: bodyfix samples the scan surface.
    raw = {key: value + 0.2 for key, value in raw.items()}
    achieved = {key: value - allowance[key] for key, value in raw.items()}
    targets = {"height": achieved["height"] - 0.12, "armLength": achieved["armLength"] + 0.08}
    return {"version": 1, "fittedMacros": macro, "fittedModifiers": mods,
            "achievedRawCm": raw, "achievedCm": achieved, "targetsCm": targets,
            "residualsCm": {key: achieved[key] - value for key, value in targets.items()},
            "clothingAllowanceCm": allowance, "measurementBasis": "synthetic scan landmarks"}


def test_absent_solution_keeps_legacy_fit(model):
    assert read_solution({}, model) is None
    assert read_solution({"unrelated": True}, model) is None
    body = model.shape({})[:model.nr]
    fitter = Fitter(model, body, None)
    fitter.init_alignment()
    assert fitter.fixed_shape is None
    assert fitter.macro["height"] != model.macro_vars["height"]["default"]


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("version", True), ("fittedMacros", {}),
    ("fittedModifiers", {"unknown": 0.1}), ("fittedModifiers", {"head/head-scale-horiz": 99}),
    ("achievedCm", {"height": float("nan")}), ("achievedRawCm", {}),
    ("residualsCm", {}), ("measurementBasis", None),
])
def test_invalid_solution_fails_instead_of_refitting(model, field, value):
    data = solution(model)
    data[field] = value
    with pytest.raises(ValueError, match="dtBodyfix"):
        read_solution({"dtBodyfix": data}, model)


def test_residuals_are_checked(model):
    data = solution(model)
    data["residualsCm"]["height"] += 1
    with pytest.raises(ValueError, match="residual"):
        read_solution({"dtBodyfix": data}, model)


def test_fixed_shape_keeps_parameters_and_alignment(model, monkeypatch):
    data = solution(model)
    macro, mods = data["fittedMacros"], data["fittedModifiers"]
    rest = model.shape(macro, mods, ground=False)
    translation = np.array([0.11, 0.24, -0.07])
    scan = rest[:model.nr] + translation
    fitter = Fitter(model, scan, None, fixed_shape=(macro, mods))
    pose_calls = []
    monkeypatch.setattr(fitter, "limb_search", lambda: None)

    def fit_pose(pos, gate, **kwargs):
        np.testing.assert_array_equal(pos, rest)
        pose_calls.append(gate)
        fitter.root_t = translation.copy()

    monkeypatch.setattr(fitter, "update_pose", fit_pose)
    monkeypatch.setattr(fitter, "update_shape", lambda *args, **kwargs: pytest.fail("shape must stay fixed"))
    result = fitter.run()
    canonicalize_fit(model, result, quantize=False)
    assert len(pose_calls) == 8
    assert result.macro == macro and result.mods == mods
    np.testing.assert_array_equal(result.rest_positions, rest)
    np.testing.assert_allclose(result.posed_vertices, scan, atol=1e-9)
    np.testing.assert_allclose(result.root_t, translation)


def test_export_keeps_scan_measurements_and_exact_body(model, tmp_path):
    data = read_solution({"dtBodyfix": solution(model)}, model)
    rest = model.shape(data["fittedMacros"], data["fittedModifiers"], ground=False)
    from rigfit import FitResult

    result = FitResult(copy.deepcopy(data["fittedMacros"]), copy.deepcopy(data["fittedModifiers"]),
                       rest, {}, np.zeros(3), rest[:model.nr])
    canonicalize_fit(model, result, quantize=False)
    shift = 0.023
    twin = write_twin_package(str(tmp_path), model, result, rest[:model.nr] - [0, shift, 0],
                              None, shift, {"scan": "synthetic.glb"}, bodyfix=data)
    disk = json.loads((tmp_path / "twin.json").read_text())
    assert disk == twin
    assert twin["fittedMacros"] == data["fittedMacros"]
    assert twin["fittedModifiers"] == data["fittedModifiers"]
    assert twin["measurementsCm"] == data["achievedCm"]
    assert twin["measurementsRawCm"] == data["achievedRawCm"]
    assert twin["bodyfix"] == data
    assert abs(twin["measurementsCm"]["height"] - data["targetsCm"]["height"]) == pytest.approx(0.12)
    heads = model.rest_heads(model.shape(twin["fittedMacros"], twin["fittedModifiers"], ground=False)) - [0, shift, 0]
    np.testing.assert_allclose(list(twin["restHeadsM"].values()), heads, atol=5e-7)


def test_rig_entry_point_discovers_moved_glb_and_ignores_stale_fit(model, tmp_path, monkeypatch):
    from glbio import GlbScene, Prim, _accessor, _split, write_static_glb
    import rig_scan

    data = solution(model)
    rest = model.shape(data["fittedMacros"], data["fittedModifiers"], ground=False)
    # No sidecar and a different name: the solution travels inside the GLB.
    source = tmp_path / "renamed.glb"
    scan = rest[:model.nr] + [0.08, 0.21, -0.06]
    write_static_glb(str(source), GlbScene([Prim(scan, model.faces)]))
    document, binary = _split(source.read_bytes())
    document["asset"]["extras"] = {"dtBodyfix": data}
    encoded = json.dumps(document).encode()
    encoded += b" " * (-len(encoded) % 4)
    source.write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
                       + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                       + struct.pack("<II", len(binary), 0x004E4942) + binary)
    out = tmp_path / "rig"
    out.mkdir()
    (out / "fit.pkl").write_bytes(pickle.dumps(None))  # must not be loaded
    monkeypatch.setattr(sys, "argv", ["rig_scan.py", str(source), str(out), "--reuse-fit",
                                      "--samples", "15000", "--smooth", "0"])
    rig_scan.main()
    twin = json.loads((out / "twin.json").read_text())
    assert twin["fittedMacros"] == data["fittedMacros"]
    assert twin["fittedModifiers"] == data["fittedModifiers"]
    assert twin["measurementsCm"] == data["achievedCm"]
    assert twin["bodyfix"] == data
    rigged, blob = _split((out / "rigged.glb").read_bytes())
    primitive = rigged["meshes"][0]["primitives"][0]
    weights = _accessor(rigged, blob, primitive["attributes"]["WEIGHTS_0"])
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
    positions = _accessor(rigged, blob, primitive["attributes"]["POSITION"])
    assert np.isfinite(positions).all()
    joints = rigged["skins"][0]["joints"]
    world_heads = []
    for index, bone in zip(joints, model.bones):
        local = np.array(rigged["nodes"][index]["translation"])
        world_heads.append(local + (world_heads[bone.parent] if bone.parent >= 0 else 0))
    np.testing.assert_allclose(world_heads, list(twin["restHeadsM"].values()), atol=1e-6)
    # Same shape as the hidden browser body, with only the grounding frame different.
    delta = np.array(world_heads) - model.rest_heads(model.shape(twin["fittedMacros"], twin["fittedModifiers"]))
    np.testing.assert_allclose(delta, np.broadcast_to(delta.mean(axis=0), delta.shape), atol=1e-6)
