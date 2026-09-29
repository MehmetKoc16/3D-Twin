"""Morph model correctness against an independent evaluation of the raw MakeHuman formulas."""

from __future__ import annotations

import numpy as np
import pytest

import macro
import targets as T
from config import DM_TO_M

TARGETS = T.TARGET_DIR


def raw_delta(rel: str) -> np.ndarray:
    idx, val = T.parse_target(TARGETS / (rel + ".target"))
    d = np.zeros((T.MH_VERTEX_COUNT, 3))
    np.add.at(d, idx, val)
    return d


def mh_lin(t: float):
    """MakeHuman's min/average/max split of a 0..1 slider."""
    mx = max(0.0, 2 * t - 1)
    mn = max(0.0, 1 - 2 * t)
    return mn, 1 - mn - mx, mx


def mh_reference_delta(gender, muscle, weight, height, cup=0.5, firm=0.5) -> np.ndarray:
    """Sum of raw MakeHuman target deltas (decimeters) for age 25, races 1/3 each. Independent of macro.py."""
    total = np.zeros((T.MH_VERTEX_COUNT, 3))
    mu, wt, ht = mh_lin(muscle), mh_lin(weight), (max(0.0, 1 - 2 * height), max(0.0, 2 * height - 1))
    cu, fi = mh_lin(cup), mh_lin(firm)
    for g, gv in (("female", 1 - gender), ("male", gender)):
        if gv == 0:
            continue
        for race in ("caucasian", "asian", "african"):
            total += raw_delta(f"macrodetails/{race}-{g}-young") * gv / 3.0
        for mn, mv in zip(("minmuscle", "averagemuscle", "maxmuscle"), mu):
            for wn, wv in zip(("minweight", "averageweight", "maxweight"), wt):
                f = gv * mv * wv
                if f == 0:
                    continue
                total += raw_delta(f"macrodetails/universal-{g}-young-{mn}-{wn}") * f
                for hn, hv in zip(("minheight", "maxheight"), ht):
                    if hv:
                        total += raw_delta(f"macrodetails/height/{g}-young-{mn}-{wn}-{hn}") * f * hv
                if g == "female":
                    for cn, cv in zip(("mincup", "averagecup", "maxcup"), cu):
                        for fn, fv in zip(("minfirmness", "averagefirmness", "maxfirmness"), fi):
                            p = TARGETS / "breast" / f"female-young-{mn}-{wn}-{cn}-{fn}.target"
                            if p.exists() and f * cv * fv:
                                total += raw_delta(f"breast/female-young-{mn}-{wn}-{cn}-{fn}") * f * cv * fv
    return total


def test_tent_semantics_match_makehuman():
    vars_ = {v["id"]: v for v in macro.MACRO_VARIABLES}
    for t in (0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 1.0):
        for var, names in (("weight", ("minweight", "averageweight", "maxweight")),
                           ("muscle", ("minmuscle", "averagemuscle", "maxmuscle")),
                           ("height", ("minheight", "averageheight", "maxheight"))):
            w = macro.tent_weights(vars_[var]["buckets"], t)
            assert [w[n] for n in names] == pytest.approx(mh_lin(t))
    w = macro.tent_weights(vars_["gender"]["buckets"], 0.3)
    assert w == pytest.approx({"female": 0.7, "male": 0.3})
    assert macro.tent_weights(vars_["age"]["buckets"], 0.5) == {"young": 1.0}


@pytest.mark.parametrize(
    "macros",
    [
        {},
        {"gender": 0.0},
        {"gender": 1.0, "height": 0.9},
        {"gender": 0.3, "muscle": 0.7, "weight": 0.2, "height": 0.8},
        {"gender": 0.0, "weight": 1.0, "cupsize": 0.9, "firmness": 0.2},
        {"gender": 0.6, "muscle": 0.1, "weight": 0.9, "height": 0.1},
    ],
)
def test_macros_match_raw_makehuman_formula(morphset, manifest, mesh, macros):
    offset = manifest["provenance"]["groundOffsetY"]
    R = manifest["renderVertexCount"]
    delta = mh_reference_delta(
        macros.get("gender", 0.5), macros.get("muscle", 0.5), macros.get("weight", 0.5),
        macros.get("height", 0.5), macros.get("cupsize", 0.5), macros.get("firmness", 0.5),
    )
    expected = (mesh.verts + delta)[mesh.render_mh] * DM_TO_M
    expected[:, 1] += offset
    got = morphset.positions(macros)[:R]
    err = np.abs(got - expected)
    # base.glb stores float32 and deltas < 1e-5 m are dropped -> sub-0.3 mm agreement
    assert err.max() < 3e-4, err.max()
    assert err.mean() < 2e-5


def test_modifier_matches_raw_target_and_joint_points_are_cube_means(morphset, manifest, mesh):
    R = manifest["renderVertexCount"]
    names = [j["name"] for j in manifest["jointPoints"]]
    base = morphset.positions()
    for mod, rel in (("torso/torso-scale-vert", "torso/torso-scale-vert-incr"),
                     ("armslegs/upperlegs-height", "armslegs/upperlegs-height-incr"),
                     ("measure/measure-waist-circ", "measure/measure-waist-circ-decr")):
        v = 1.0 if rel.endswith("incr") else -1.0
        moved = morphset.positions(mods={mod: v}) - base
        d = raw_delta(rel) * DM_TO_M
        # body vertices: same delta on every UV-split copy
        assert np.abs(moved[:R] - d[mesh.render_mh]).max() < 2e-5
        # joint points: exact mean of the 8 cube vertex deltas
        for j, n in enumerate(names):
            cube = mesh.groups[n]
            assert np.abs(moved[R + j] - d[cube].mean(axis=0)).max() < 2e-5, (mod, n)


def test_uv_split_copies_share_deltas(manifest, morph_bytes, mesh):
    entries = np.frombuffer(morph_bytes, dtype=T.ENTRY_DTYPE)
    R = manifest["renderVertexCount"]
    t = next(t for t in manifest["targets"] if t["id"] == "torso/torso-scale-vert-incr")
    s = t["byteOffset"] // 16
    run = entries[s : s + t["count"]]
    dense = np.zeros((R, 3), dtype=np.float32)
    has = np.zeros(R, dtype=bool)
    body = run["i"] < R
    dense[run["i"][body]] = run["d"][body]
    has[run["i"][body]] = True
    checked = 0
    for mh in np.nonzero(np.diff(mesh.mh_to_render_start[:13381]) > 1)[0]:  # seam vertices
        ids = mesh.render_ids_of(int(mh))
        assert np.array_equal(has[ids], np.full(len(ids), has[ids[0]]))
        assert (dense[ids] == dense[ids[0]]).all()
        checked += 1
    assert checked > 0
    extra = np.diff(mesh.mh_to_render_start[:13381]) - 1
    assert extra.sum() == R - 13380


def test_neutral_body_and_floor(morphset, manifest):
    R = manifest["renderVertexCount"]
    p = morphset.positions()
    assert abs(p[:R, 1].min()) < 1e-5  # neutral body stands on y = 0
    assert abs(p[:R, 0].mean()) < 1e-3  # centered on x
    # +Z is the front: the nose/toes point forward (max z of the mesh is in front of the origin)
    assert p[:R, 2].max() > abs(p[:R, 2].min())


@pytest.mark.parametrize("gender,lo,hi", [(0.0, 1.58, 1.60), (1.0, 1.72, 1.74), (0.5, 1.65, 1.67)])
def test_neutral_heights(morphset, manifest, gender, lo, hi):
    R = manifest["renderVertexCount"]
    p = morphset.positions({"gender": gender})[:R, 1]
    assert lo <= p.max() - p.min() <= hi


def test_height_macro_range(morphset, manifest):
    R = manifest["renderVertexCount"]

    def h(**kw):
        p = morphset.positions(kw)[:R, 1]
        return p.max() - p.min()

    assert h(gender=0.0, height=0.0) < h(gender=0.0) < h(gender=0.0, height=1.0)
    assert h(gender=1.0, height=0.0) < h(gender=1.0) < h(gender=1.0, height=1.0)
    assert 1.1 < h(gender=0.0, height=0.0) and h(gender=1.0, height=1.0) < 2.6


# ---------------------------------------------------------------------------------------------------------------
# Extended measure-modifier ranges (targets.EXTENDED_RANGE): linear extrapolation must stay geometrically sane
# ---------------------------------------------------------------------------------------------------------------

EXT_BASES = ({}, {"gender": 1.0, "weight": 1.0}, {"gender": 0.0, "weight": 0.0})


def _tri_normals(P, tris):
    a, b, c = (P[tris[:, i]] for i in range(3))
    return np.cross(b - a, c - a)


def test_extended_ranges_are_geometrically_sane(morphset, mesh, manifest, measures):
    """At every extended end (|value| > 1): no folded triangle (normal turned by more than ~78 deg), no collapsed
    triangle (area >= 20 % of the base area) on neutral, heavy male and thin female bodies; and the driven
    circumference is monotonic over the whole range with its loop convexity intact."""
    import measures as M

    R = manifest["renderVertexCount"]
    tris = mesh.tris
    bases = [(m, morphset.positions(m)) for m in EXT_BASES]
    base_n = [_tri_normals(P[:R], tris) for _, P in bases]
    defs = {m["id"]: m for m in manifest["modifiers"]}
    checked = 0
    for mid, (lo, hi) in T.EXTENDED_RANGE.items():
        assert defs[mid]["min"] == lo and defs[mid]["max"] == hi
        for val in (lo, hi):
            if abs(val) <= 1.0:
                continue
            for (macros, P0), n0 in zip(bases, base_n):
                P = morphset.positions(macros, {mid: val})
                n = _tri_normals(P[:R], tris)
                na, nb = np.linalg.norm(n, axis=1), np.maximum(np.linalg.norm(n0, axis=1), 1e-30)
                cos = (n * n0).sum(axis=1) / np.maximum(na * nb, 1e-30)
                moved = (np.abs(P[:R] - P0[:R]).max(axis=1) > 1e-6)[tris].any(axis=1)
                assert (cos[moved] >= 0.2).all(), (mid, val, macros, int((cos < 0.2).sum()))
                assert (na[moved] / nb[moved]).min() >= 0.2, (mid, val, macros)
            checked += 1
    assert checked >= 15

    # circumference drivers: monotonic perimeter and convexity kept over the full extended range
    for meas in measures["measures"]:
        if meas["type"] != "circumference":
            continue
        drv = meas["drivers"][0]
        lo, hi = T.EXTENDED_RANGE.get(drv, (-1.0, 1.0))
        vals = np.linspace(lo, hi, 7)
        for macros in EXT_BASES[1:]:
            per, conv = [], []
            for v in vals:
                pts = morphset.positions(macros, {drv: float(v)})[meas["verts"]]
                per.append(M.circumference(pts))
                conv.append(_polygon_area(pts) / _hull_area(pts))
            assert (np.diff(per) > 0).all(), (meas["id"], macros)
            assert min(conv) > conv[list(vals).index(0.0) if 0.0 in vals else 3] - 0.05, (meas["id"], macros, conv)


def _project(pts):
    import measures as M

    n = M.newell_normal(pts)
    return (pts - pts.mean(axis=0)) @ M._plane_basis(n)


def _polygon_area(pts):
    q = _project(pts)
    return 0.5 * abs((q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]).sum())


def _hull_area(pts):
    q = _project(pts)
    p = sorted(set(map(tuple, np.round(q, 9).tolist())))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for r in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], r) <= 0:
            lower.pop()
        lower.append(r)
    for r in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], r) <= 0:
            upper.pop()
        upper.append(r)
    h = np.array(lower[:-1] + upper[:-1])
    return 0.5 * abs((h[:, 0] * np.roll(h[:, 1], -1) - np.roll(h[:, 0], -1) * h[:, 1]).sum())
