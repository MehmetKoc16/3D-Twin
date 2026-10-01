"""twin_export.py: the app-facing package (twin.json + mh2twin.bin) written next to rigged.glb."""

from __future__ import annotations

import json
import os
import types

import numpy as np

from twin_export import (
    CLOTHING_ALLOWANCE_CM,
    MACRO_KEYS,
    canonicalize_fit,
    evaluate_measure,
    measure_body,
    nearest_render_vertices,
    write_twin_package,
)

MACRO = {"gender": 0.7, "muscle": 0.4, "weight": 0.62, "height": 0.55}
MODS = {"measure/measure-waist-circ": 0.4, "torso/torso-scale-horiz": -0.2, "armslegs/upperleg-fat": 0.31}


def fake_fit(model, rest_positions):
    """A FitResult stand-in: only what canonicalize_fit / write_twin_package read."""
    return types.SimpleNamespace(
        macro=dict(MACRO),
        mods={**MODS, "hip/hip-scale-horiz": 0.0000004},
        rest_positions=rest_positions,
        pose_rotvec={"upperarm_l": np.array([0.0, 0.0, -0.3]), "thigh_r": np.array([0.1, 0.0, 0.0])},
        root_t=np.array([0.0, 0.02, 0.0]),
        posed_vertices=rest_positions[: model.nr],
        stats={},
    )


def test_circumference_and_distance_ports_are_tape_measure_semantics():
    theta = np.linspace(0, 2 * np.pi, 41)[:-1]
    ring = np.stack([0.15 * np.cos(theta), np.full_like(theta, 1.2), 0.1 * np.sin(theta)], axis=1)
    got = evaluate_measure({"type": "circumference", "verts": list(range(40))}, ring, 40)
    a, b = 0.15, 0.1  # ellipse perimeter (Ramanujan) ~ the convex hull of 40 points on it
    expected = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    assert abs(got - expected) < 0.003
    pts = np.array([[0, 0, 0], [0.3, 1.0, 0.4], [0.3, 0.0, 0.0]])
    assert abs(evaluate_measure({"type": "distance", "verts": [0, 1]}, pts, 3) - np.linalg.norm(pts[1])) < 1e-12
    assert abs(evaluate_measure({"type": "distance", "verts": [0, 1], "axis": "z"}, pts, 3) - 0.4) < 1e-12
    assert abs(evaluate_measure({"type": "height"}, pts, 3) - 1.0) < 1e-12


def test_neutral_body_measures_match_the_documented_body(model):
    """docs/ARCHITECTURE.md: the neutral body is 1.659 m tall; every measure of measures.json is produced in cm."""
    body = model.shape({})
    cm = measure_body(model, body)
    assert set(cm) == {"height", "neck", "shoulder", "chest", "waist", "hip", "thigh", "upperArm", "armLength", "inseam", "footLength"}
    assert abs(cm["height"] - 165.9) < 0.2
    assert 70 < cm["chest"] < 130 and 55 < cm["waist"] < 120 and 70 < cm["hip"] < 130
    assert 30 < cm["thigh"] < 80 and 55 < cm["inseam"] < 95 and 15 < cm["footLength"] < 32


def test_canonicalize_fit_rebuilds_the_body_from_the_exported_values(model):
    res = fake_fit(model, model.shape({"gender": 0.2}, ground=False))  # a fit whose shape differs from its own values
    canonicalize_fit(model, res)
    assert set(res.macro) == set(MACRO_KEYS)
    assert all(round(v, 6) == v for v in res.macro.values())
    assert "hip/hip-scale-horiz" not in res.mods  # below 1e-6: dropped
    assert all(round(v, 6) == v for v in res.mods.values())
    # exactly what the browser rebuilds (avatar-core: macro tents + net modifier values)
    assert np.array_equal(res.rest_positions, model.shape(res.macro, res.mods, ground=False))
    assert res.posed_vertices.shape == (model.nr, 3)
    assert "canonical_shape_dev_mm" in res.stats
    # posing the canonical body with an all-zero pose only translates it by root_t
    res.pose_rotvec = {b: np.zeros(3) for b in res.pose_rotvec}
    canonicalize_fit(model, res)
    assert np.allclose(res.posed_vertices, res.rest_positions[: model.nr] + res.root_t, atol=1e-9)


def test_nearest_render_vertices_maps_a_body_onto_itself_and_respects_normals(model):
    body = model.shape({}, ground=True)[: model.nr]
    idx, stats = nearest_render_vertices(body, model.faces, body, None)
    assert idx.dtype == np.dtype("<u4")
    assert np.allclose(body[idx], body, atol=1e-9)  # UV-seam copies may swap, positions are identical
    assert stats["max_cm"] < 1e-6
    # a vertex 2 cm outside the surface, facing outwards, still lands on the vertex it came from
    import trimesh

    normals = np.asarray(trimesh.Trimesh(body, model.faces, process=False).vertex_normals)
    outside = body + normals * 0.02
    idx2, stats2 = nearest_render_vertices(body, model.faces, outside[::50], normals[::50])
    assert stats2["median_cm"] < 3.0
    assert np.mean(np.linalg.norm(body[idx2] - body[::50], axis=1) < 0.03) > 0.9


def test_write_twin_package_roundtrip(model, tmp_path):
    res = fake_fit(model, model.shape(MACRO, MODS, ground=False))
    canonicalize_fit(model, res)
    shift = 0.013
    twin_verts = (res.rest_positions[: model.nr] + np.array([0.0, -shift, 0.0]))
    out = write_twin_package(str(tmp_path), model, res, twin_verts, None, shift, {"scan": "unit-test.glb", "options": {}})
    disk = json.load(open(os.path.join(tmp_path, "twin.json"), encoding="utf8"))
    assert disk["version"] == 1 and disk["glb"] == "rigged.glb"
    assert disk["boneOrder"] == list(model.bone_names) and len(disk["boneOrder"]) == 53
    assert disk["fittedMacros"] == {k: res.macro[k] for k in MACRO_KEYS}
    assert disk["fittedModifiers"] == dict(sorted(res.mods.items()))
    raw, allowance, person = disk["measurementsRawCm"], disk["clothingAllowanceCm"], disk["measurementsCm"]
    assert set(raw) == set(person) == set(allowance)
    for k in raw:
        assert abs(person[k] - (raw[k] - allowance[k])) < 0.011  # rounded to 0.01 cm
        assert allowance[k] == CLOTHING_ALLOWANCE_CM[k]
    assert disk["measurementsCm"]["height"] < disk["measurementsRawCm"]["height"]  # hair and soles are not the person
    # rest heads in the frame of rigged.glb = fit frame - ground shift
    heads = model.rest_heads(res.rest_positions)
    for i, name in enumerate(model.bone_names):
        assert np.allclose(disk["restHeadsM"][name], heads[i] - np.array([0.0, shift, 0.0]), atol=1e-6)
    mapping = np.fromfile(os.path.join(tmp_path, "mh2twin.bin"), dtype="<u4")
    assert disk["mapping"] == {"file": "mh2twin.bin", "format": "uint32le", "twinVertexCount": len(twin_verts), "renderVertexCount": model.nr}
    assert len(mapping) == len(twin_verts) and mapping.max() < model.nr
    assert out["fit"]["mappingCm"]["max_cm"] < 0.01  # the twin here IS the body
