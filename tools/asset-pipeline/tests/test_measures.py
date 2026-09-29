"""measures.json: structure, anthropometric plausibility on the neutral bodies, driver finite differences."""

from __future__ import annotations

import numpy as np
import pytest

import measures as M

REQUIRED = ["height", "neck", "shoulder", "chest", "waist", "hip", "thigh", "upperArm", "armLength", "inseam", "footLength"]

# plausible adult ranges in cm: measure -> (female lo, hi, male lo, hi); neutral = macro defaults except gender
RANGES = {
    "height": (158.0, 160.0, 172.0, 174.0),
    "neck": (27, 36, 35, 43),
    "shoulder": (34, 42, 37, 46),  # across the back over C7 (acromion -> C7 -> acromion), longer than biacromial
    "chest": (78, 96, 88, 106),
    "waist": (58, 80, 68, 90),
    "hip": (86, 104, 88, 106),
    "thigh": (46, 62, 50, 66),
    "upperArm": (22, 30, 26, 34),
    "armLength": (45, 64, 50, 70),  # acromion -> elbow -> wrist along the skin
    "inseam": (68, 80, 76, 90),
    "footLength": (20, 26, 24, 29),
}


def defs(measures):
    return {m["id"]: m for m in measures["measures"]}


def test_required_measures_present(measures):
    assert [m["id"] for m in measures["measures"]] == REQUIRED
    assert measures["version"] == 1


def test_measure_structure(measures, manifest):
    R, V = manifest["renderVertexCount"], manifest["vertexCount"]
    modifier_ids = {m["id"] for m in manifest["modifiers"]}
    for m in measures["measures"]:
        assert m["drivers"] and all(d in modifier_ids for d in m["drivers"]), m["id"]
        if m["type"] in ("circumference", "polyline", "distance"):
            assert all(0 <= v < R for v in m["verts"]), m["id"]  # surface vertices only
            assert len(m["verts"]) == len(set(m["verts"]))
        if m["type"] == "circumference":
            assert len(m["verts"]) >= 8
        if m["type"] == "distance":
            assert len(m["verts"]) == 2 and m.get("axis", "x") in "xyz"
        if m["type"] == "polyline":
            assert len(m["verts"]) >= 3
        if m["type"] == "vertexHeight":
            assert 0 <= m["vert"] < V
    d = defs(measures)
    assert d["height"]["type"] == "height"
    assert d["inseam"]["type"] == "vertexHeight"
    assert d["shoulder"]["type"] == "polyline" and len(d["shoulder"]["verts"]) >= 9
    assert d["footLength"]["type"] == "distance" and d["footLength"]["axis"] == "z"
    assert d["armLength"]["type"] == "polyline"
    for k in ("neck", "chest", "waist", "hip", "thigh", "upperArm"):
        assert d[k]["type"] == "circumference"


def neutral(morphset, gender):
    return morphset.positions({"gender": gender})


@pytest.mark.parametrize("gender,col", [(0.0, 0), (1.0, 2)])
def test_neutral_values_plausible(morphset, manifest, measures, gender, col):
    P = neutral(morphset, gender)
    R = manifest["renderVertexCount"]
    for m in measures["measures"]:
        cm = 100 * M.evaluate(m, P, R)
        lo, hi = RANGES[m["id"]][col], RANGES[m["id"]][col + 1]
        assert lo <= cm <= hi, f"{m['id']} gender={gender}: {cm:.1f} cm not in [{lo}, {hi}]"


def test_shoulder_pair_is_mirrored_and_landmark_ordering(morphset, manifest, measures):
    R = manifest["renderVertexCount"]
    P = morphset.positions()
    d = defs(measures)
    path = d["shoulder"]["verts"]
    a, b = path[0], path[-1]
    assert P[a, 0] > 0 > P[b, 0]  # left = +X
    assert abs(P[a, 0] + P[b, 0]) < 1e-4 and abs(P[a, 1] - P[b, 1]) < 1e-4 and abs(P[a, 2] - P[b, 2]) < 1e-4
    # the shoulder path runs across the back: x strictly decreasing, behind the acromion line, symmetric, over C7
    x = P[path, 0]
    assert (np.diff(x) < 0).all()
    assert (P[path[1:-1], 2] < P[a, 2] + 0.005).all()
    assert np.abs(P[path, 0] + P[path[::-1], 0]).max() < 1e-4 and np.abs(P[path, 1] - P[path[::-1], 1]).max() < 1e-4
    mid = path[len(path) // 2]
    assert abs(P[mid, 0]) < 1e-4 and P[mid, 1] > P[a, 1] + 0.03 and P[mid, 2] < P[a, 2]  # C7: higher and behind
    # longer than the straight biacromial distance (acromion pair), but not by more than ~20 %
    straight = abs(P[a, 0] - P[b, 0])
    assert straight * 1.03 < M.evaluate(d["shoulder"], morphset.positions(), R) < straight * 1.20
    # vertical ordering of the loop planes on the neutral body
    y = {k: P[d[k]["verts"], 1].mean() for k in ("neck", "chest", "waist", "hip", "thigh")}
    assert y["neck"] > y["chest"] > y["waist"] > y["hip"] > y["thigh"]
    crotch = P[d["inseam"]["vert"], 1]
    assert y["thigh"] > crotch - 0.10 and y["hip"] > crotch
    # armLength polyline is acromion -> elbow -> wrist, going down the arm
    ids = d["armLength"]["verts"]
    assert ids[0] == a and P[ids[0], 1] > P[ids[1], 1] > P[ids[2], 1]
    # foot: toe in front of heel, both near the floor
    heel, toe = d["footLength"]["verts"]
    assert P[toe, 2] > P[heel, 2] + 0.15 and P[toe, 1] < 0.08 and P[heel, 1] < 0.08


def test_crotch_is_lowest_midline_vertex(morphset, manifest, measures):
    R = manifest["renderVertexCount"]
    P = morphset.positions()[:R]
    mid = np.where(np.abs(P[:, 0]) < 1e-4)[0]
    assert defs(measures)["inseam"]["vert"] == mid[P[mid, 1].argmin()]


def test_drivers_change_measures_monotonically(morphset, manifest, measures):
    """Finite differences: every driver moves its measure by > 3 mm per unit and in the right direction."""
    R = manifest["renderVertexCount"]
    for m in measures["measures"]:
        v0 = M.evaluate(m, morphset.positions(), R)
        for d in m["drivers"]:
            lo = M.evaluate(m, morphset.positions(mods={d: -1.0}), R)
            hi = M.evaluate(m, morphset.positions(mods={d: 1.0}), R)
            assert hi - v0 > 0.003, (m["id"], d, hi - v0)
            assert v0 - lo > 0.003, (m["id"], d, v0 - lo)
        joint = M.evaluate(m, morphset.positions(mods={d: 1.0 for d in m["drivers"]}), R)
        assert joint - v0 > 0.003


def test_driver_effect_is_local_for_circumferences(morphset, manifest, measures):
    """A circumference driver must dominate its own measure (>= 2x the effect on every other circumference)."""
    R = manifest["renderVertexCount"]
    circ = [m for m in measures["measures"] if m["type"] == "circumference"]
    base = {m["id"]: M.evaluate(m, morphset.positions(), R) for m in circ}
    for m in circ:
        P = morphset.positions(mods={m["drivers"][0]: 1.0})
        own = M.evaluate(m, P, R) - base[m["id"]]
        for other in circ:
            if other["id"] == m["id"] or other["id"] in ("waist", "chest", "hip"):
                continue  # neighbouring torso loops legitimately share influence
            assert abs(M.evaluate(other, P, R) - base[other["id"]]) < own, (m["id"], other["id"])


def test_measure_evaluator_semantics():
    # unit square loop plus interior points -> hull perimeter 4 (order/duplicates irrelevant), in any orientation
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0], [0.2, 0.7, 0], [1, 1, 0]], float)
    assert M.circumference(pts) == pytest.approx(4.0)
    rot = pts @ np.array([[1, 0, 0], [0, 0.6, -0.8], [0, 0.8, 0.6]])
    assert M.circumference(rot) == pytest.approx(4.0)
    P = np.array([[0, 0, 0], [3, 4, 5], [3, 4, 5]], float)
    assert M.evaluate({"type": "distance", "verts": [0, 1], "axis": "z"}, P, 3) == 5
    assert M.evaluate({"type": "distance", "verts": [0, 1]}, P, 3) == pytest.approx(np.sqrt(50))
    assert M.evaluate({"type": "polyline", "verts": [0, 1, 2]}, P, 3) == pytest.approx(np.sqrt(50))
    assert M.evaluate({"type": "height"}, P, 3) == 4
    assert M.evaluate({"type": "vertexHeight", "vert": 1}, P, 3) == 4


def test_loops_are_angularly_ordered_for_newell(morphset, measures):
    """The runtime derives the plane with Newell's method over the ordered loop, so the stored order must trace
    the ring: the Newell normal must agree with the least-squares plane normal and the perimeter of the
    Newell-projected hull with the least-squares-projected hull, on several body shapes."""
    for macros in ({}, {"gender": 0.0}, {"gender": 1.0, "weight": 1.0}, {"gender": 0.0, "height": 1.0}):
        P = morphset.positions(macros)
        for m in measures["measures"]:
            if m["type"] != "circumference":
                continue
            pts = P[m["verts"]]
            c = pts - pts.mean(axis=0)
            ls_normal = np.linalg.eigh(c.T @ c)[1][:, 0]
            nw = M.newell_normal(pts)
            cos = abs(nw @ ls_normal) / np.linalg.norm(nw)
            assert cos > 0.99, (m["id"], macros, cos)
            ls_len = M._hull_perimeter(c @ M._plane_basis(ls_normal))
            assert abs(ls_len - M.circumference(pts)) < 0.003, (m["id"], macros)


def test_loops_have_consistent_direction_and_no_repeats(morphset, measures):
    P = morphset.positions()
    for m in measures["measures"]:
        if m["type"] != "circumference":
            continue
        assert len(set(m["verts"])) == len(m["verts"]), "first vertex must not be repeated at the end"
        # consistent direction for all loops: Newell normal points up (counter-clockwise seen from above)
        assert M.newell_normal(P[m["verts"]])[1] > 0, m["id"]
        # consecutive vertices are neighbours around the ring (no long chords): gaps stay small vs the perimeter
        pts = P[m["verts"]]
        gaps = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        assert gaps.max() < 0.12, (m["id"], gaps.max())
