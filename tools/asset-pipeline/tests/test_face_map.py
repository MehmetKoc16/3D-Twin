"""face-map.json: binding of the 468 MediaPipe canonical landmarks to the MakeHuman head."""

from __future__ import annotations

import json

import numpy as np
import pytest

import face_map as fm
from config import MEDIAPIPE

R_EXPECTED = 14517
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


@pytest.fixture(scope="module")
def fmap(out_a) -> dict:
    return json.loads((out_a / "face-map.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def canonical(raw_data):
    return fm.load_canonical()


@pytest.fixture(scope="module")
def conn(raw_data):
    return fm.load_connections()


@pytest.fixture(scope="module")
def islands(mesh):
    return fm.uv_islands(mesh.tris, mesh.render_count)


@pytest.fixture(scope="module")
def result(mesh, morphset, manifest):
    """A fresh in-process build of the face map (report + objects), independent of the JSON file."""
    return fm.build_face_map(mesh, morphset.positions(), morphset, manifest)


@pytest.fixture(scope="module")
def bound(fmap, morphset):
    """Arrays over the landmarks: tri, bary, uv and the bound positions on the neutral body."""
    L = fmap["landmarks"]
    tri = np.array([l["tri"] for l in L])
    bary = np.array([l["bary"] for l in L])
    uv = np.array([l["uv"] for l in L])
    P = morphset.positions()
    pts = (bary[:, :, None] * P[tri]).sum(axis=1)
    return {"tri": tri, "bary": bary, "uv": uv, "pts": pts, "P": P}


def test_header_and_topology(fmap, canonical):
    assert fmap["version"] == 1
    assert fmap["source"] == {"mediapipeCommit": MEDIAPIPE["sha"], "license": "Apache-2.0"}
    assert len(MEDIAPIPE["sha"]) == 40
    assert fmap["triangles"] == canonical.faces.tolist()
    assert len(fmap["triangles"]) == 898
    assert np.array(fmap["triangles"]).max() == 467


def test_all_468_landmarks_bound(fmap, manifest):
    L = fmap["landmarks"]
    assert [l["index"] for l in L] == list(range(468))
    R = manifest["renderVertexCount"]
    assert R == R_EXPECTED
    for l in L:
        assert len(set(l["tri"])) == 3 and all(0 <= t < R for t in l["tri"]), l["index"]


def test_bary_weights(bound):
    b = bound["bary"]
    assert (b >= 0).all()
    assert np.abs(b.sum(axis=1) - 1.0).max() < 1e-5


def test_uv_matches_interpolated_base_uv(bound, mesh):
    uvg = fm.gltf_uv(mesh)
    expect = (bound["bary"][:, :, None] * uvg[bound["tri"]]).sum(axis=1)
    assert np.abs(expect - bound["uv"]).max() < 2e-6
    assert bound["uv"].min() >= 0 and bound["uv"].max() <= 1


def test_uv_bounds(fmap, bound):
    assert fmap["uvBounds"]["min"] == pytest.approx(bound["uv"].min(axis=0).tolist(), abs=2e-6)
    assert fmap["uvBounds"]["max"] == pytest.approx(bound["uv"].max(axis=0).tolist(), abs=2e-6)


def test_single_uv_island(bound, islands, mesh):
    """All 3 vertices of every bound triangle are on one UV island, the same island for all 468 landmarks: the
    head island (it holds the nose tip)."""
    lab = islands[bound["tri"]]
    assert (lab == lab[:, :1]).all()
    assert len(np.unique(lab)) == 1
    ren = bound["P"][: mesh.render_count]
    nose_island = islands[np.argmax(np.where(ren[:, 1] > 1.46, ren[:, 2], -9))]
    assert lab[0, 0] == nose_island


def test_no_seam_straddling(result):
    """The face never crosses a UV cut of the head island: faces whose UV samples fall outside the island's UV
    area may only touch an eye or lip contour (eyelid openings and the lip slit are holes of the island)."""
    s = result.report["seams"]
    assert s["single_island"] is True
    assert s["faces_outside_not_on_contour"] == 0


def test_uv_orientation_consistent_and_not_flipped(fmap, bound):
    faces = np.array(fmap["triangles"])
    a = fm.uv_triangle_areas(bound["uv"], faces)
    sign = 1.0 if (a > 0).sum() > (a < 0).sum() else -1.0
    # glTF UV convention (v down): canonical faces (CCW in the frontal view) map clockwise, the face is rotated
    assert sign == -1.0
    flipped = a * sign < 0
    # only collapsed slivers at the eyelid / lip holes may flip; no real triangle is inverted
    assert flipped.sum() <= 20
    assert np.abs(a[flipped]).max() < 1.5e-5
    assert np.abs(a[flipped]).sum() < 2e-3 * np.abs(a).sum()
    assert (np.abs(a) > 0).all()  # no exactly degenerate triangle
    contour = set(fmap["regions"]["leftEye"]) | set(fmap["regions"]["rightEye"]) | set(fmap["regions"]["lips"])
    off = [f for f in np.where(flipped)[0] if not (set(faces[f].tolist()) & contour)]
    assert len(off) <= 4 and all(np.abs(a[f]) < 1e-6 for f in off)
    assert (np.abs(a) < fm.SLIVER_AREA).sum() <= 40


def test_uv_layout_is_rotated_face(bound, canonical):
    """The face occupies the head island rotated by 90 degrees: u grows from the forehead to the chin, v decreases
    towards the subject's left (+x). Documented for the browser baker in ADR 0006."""
    x, y = canonical.verts[:, 0], canonical.verts[:, 1]
    A = np.c_[x, -y, np.ones(468)]
    sol, *_ = np.linalg.lstsq(A, bound["uv"], rcond=None)
    assert sol[0, 1] < -0.005 and abs(sol[0, 0]) < 0.002  # dv/dx < 0, du/dx ~ 0
    assert sol[1, 0] > 0.005 and abs(sol[1, 1]) < 0.002  # du/dy_down > 0, dv/dy_down ~ 0
    assert bound["uv"][152, 0] > bound["uv"][10, 0]  # chin right of forehead in u


def test_symmetric_pairs_mirror(bound, canonical):
    v = canonical.verts
    mirror = np.array([np.argmin(np.linalg.norm(v - np.array([-x, y, z]), axis=1)) for x, y, z in v])
    assert np.abs(v[mirror] * np.array([-1, 1, 1]) - v).max() < 1e-4  # the canonical model is exactly symmetric
    pairs = np.where(mirror != np.arange(468))[0]
    assert len(pairs) >= 440
    pts, uv = bound["pts"], bound["uv"]
    assert np.linalg.norm(pts[pairs] * np.array([-1, 1, 1]) - pts[mirror[pairs]], axis=1).max() < 0.002
    mid_v = uv[10, 1]
    assert np.abs(uv[pairs, 0] - uv[mirror[pairs], 0]).max() < 0.002
    assert np.abs(uv[pairs, 1] + uv[mirror[pairs], 1] - 2 * mid_v).max() < 0.002
    # the midline landmarks sit on the mid-sagittal plane
    mid = np.where(np.abs(v[:, 0]) < 1e-6)[0]
    assert np.abs(pts[mid, 0]).max() < 0.001


def test_anchors_land_where_expected(bound, fmap, canonical, mesh):
    pts = bound["pts"]
    ren = bound["P"][: mesh.render_count]
    nose = int(canonical.verts[:, 2].argmax())
    head = ren[:, 1] > 1.46
    assert pts[nose, 2] > ren[head, 2].max() - 0.008  # nose tip: max-z region of the head
    assert int(pts[:, 2].argmax()) in (nose, 1, 4, 5, 195, 197, 2)  # ... and the max-z bound point is on the nose
    assert int(pts[:, 1].argmin()) in fmap["faceOval"]  # lowest bound point is on the jaw line
    assert pts[152, 1] < pts[[61, 291], 1].min() - 0.03  # chin below the mouth corners
    assert pts[10, 1] > pts[[33, 263], 1].max() + 0.04  # forehead point above the eyes
    # left/right: the subject's left is +x
    assert pts[fmap["regions"]["leftEye"], 0].mean() > 0.02
    assert pts[fmap["regions"]["rightEye"], 0].mean() < -0.02
    assert pts[fmap["regions"]["leftCheek"], 0].min() > 0 > pts[fmap["regions"]["rightCheek"], 0].max()
    assert pts[263, 0] > 0 > pts[33, 0] and pts[291, 0] > 0 > pts[61, 0]
    # vertical ordering along the face
    y = lambda name: pts[fmap["regions"][name], 1].mean()  # noqa: E731
    assert y("forehead") > y("leftEye") > y("leftCheek") > y("lips")
    # eye and mouth centres near the eyelid openings / lip slit of the mesh (centres of the hole rims)
    head_obj = fm.analyse_head(mesh, bound["P"], 1.41)
    anchors = fm.anchors_mh(mesh, bound["P"], head_obj)
    eye_l = pts[fmap["regions"]["leftEye"]].mean(axis=0)
    eye_r = pts[fmap["regions"]["rightEye"]].mean(axis=0)
    mouth = pts[INNER_LIPS].mean(axis=0)
    assert np.linalg.norm(eye_l - anchors["eye_l"]) < 0.006
    assert np.linalg.norm(eye_r - anchors["eye_r"]) < 0.006
    assert np.linalg.norm(mouth - anchors["mouth"]) < 0.007
    assert np.linalg.norm(pts[291] - anchors["mouth_l"]) < 0.008


def test_binding_distances_are_small(result):
    r = result.report
    assert r["bound"] == 468
    assert r["max_xy_distance_mm"] < 7.0 and r["mean_xy_distance_mm"] < 1.0
    assert r["mean_distance_mm"] < 5.0 and r["max_distance_mm"] < 20.0
    assert abs(r["alignment"]["pitchDeg"]) < 5


def test_regions_and_oval(fmap, conn):
    reg = fmap["regions"]
    assert set(reg) == {"leftEye", "rightEye", "lips", "leftCheek", "rightCheek", "forehead"}
    assert len(reg["leftEye"]) == len(reg["rightEye"]) == 16
    assert len(reg["leftCheek"]) == len(reg["rightCheek"]) >= 8
    assert len(reg["forehead"]) >= 10
    for name, ids in reg.items():
        assert ids == sorted(set(ids)) and all(0 <= i < 468 for i in ids), name
    assert 263 in reg["leftEye"] and 33 in reg["rightEye"]  # MediaPipe left eye = subject's left = +x
    assert not set(reg["leftCheek"]) & set(reg["lips"]) and not set(reg["leftCheek"]) & set(reg["leftEye"])
    oval = fmap["faceOval"]
    assert len(oval) == 36 and len(set(oval)) == 36 and oval[0] == 10
    pairs = {frozenset(e) for e in conn["FACEMESH_FACE_OVAL"]}
    for a, b in zip(oval, oval[1:] + oval[:1]):
        assert frozenset((a, b)) in pairs


def test_fit_modifiers(fmap, manifest, morphset, canonical, conn):
    ids = {m["id"] for m in manifest["modifiers"]}
    fit = fmap["fitModifiers"]
    assert len(fit) >= 8 and len(fit) == len(set(fit))
    assert all(m in ids for m in fit)
    assert "forehead/forehead-scale-vert" not in fit  # moves only hair-covered surface: no frontal landmark effect
    # finite differences on the bound points of the file itself: every fit modifier changes the layout of the
    # stable landmarks by at least MIN_EFFECT_MM on each side
    L = fmap["landmarks"]
    b = fm.Binding(
        np.array([l["tri"] for l in L]),
        np.array([l["bary"] for l in L]),
        np.array([l["uv"] for l in L]),
        np.zeros((468, 3)),
        np.zeros((468, 3)),
    )
    stable = fm.stable_landmarks(canonical, conn)
    eff = fm.modifier_effects(morphset, manifest, b, stable, fit)
    for m, row in eff.items():
        assert all(e >= fm.MIN_EFFECT_MM for e in row if e is not None), (m, row)
    # the stable set excludes eye contours, eyebrows and lips (except the corners)
    assert not set(fmap["regions"]["leftEye"]) & set(stable)
    assert {61, 291} <= set(stable) and 13 not in stable and 152 in stable


def test_closest_point_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 400
    A = rng.normal(size=(n, 3))
    B = A + rng.normal(scale=0.3, size=(n, 3))
    C = A + rng.normal(scale=0.3, size=(n, 3))
    pts = rng.normal(size=(60, 3))
    t1, b1, d1 = fm.closest_on_triangles(pts, A, B, C, candidates=n)
    _, _, d2 = fm.closest_on_triangles(pts, A, B, C, candidates=120)
    assert np.allclose(d1, d2, atol=1e-12)
    for q in range(60):
        tri = np.stack([A[t1[q]], B[t1[q]], C[t1[q]]])
        cp = (b1[q][:, None] * tri).sum(0)
        assert abs(np.linalg.norm(pts[q] - cp) ** 2 - d1[q]) < 1e-9
        assert (b1[q] >= -1e-12).all() and abs(b1[q].sum() - 1) < 1e-9
    # never farther than any vertex of any triangle
    allv = np.concatenate([A, B, C])
    assert (d1 <= ((pts[:, None, :] - allv[None]) ** 2).sum(-1).min(axis=1) + 1e-12).all()


def test_alignment_recovers_a_known_transform():
    rng = np.random.default_rng(5)
    q = rng.normal(size=(200, 3)) * 5
    truth = fm.Align(0.009, np.deg2rad(4.0), 1.5, 0.09)
    got = fm.fit_alignment(q, truth.apply(q))
    assert got.s == pytest.approx(truth.s, rel=1e-6)
    assert got.theta == pytest.approx(truth.theta, abs=np.deg2rad(0.03))
    assert got.ty == pytest.approx(truth.ty, abs=1e-5) and got.tz == pytest.approx(truth.tz, abs=1e-5)


def test_chain_loop_and_connection_parser(conn):
    loop = fm.chain_loop(conn["FACEMESH_FACE_OVAL"], 10)
    assert loop[:3] == [10, 338, 297] and len(loop) == 36
    assert len(conn["FACEMESH_LEFT_EYE"]) == 16 and len(conn["FACEMESH_LIPS"]) == 40


def test_debug_png_writer(tmp_path):
    from debug_png import write_png

    img = np.zeros((4, 5, 3), dtype=np.uint8)
    write_png(tmp_path / "x.png", img)
    data = (tmp_path / "x.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and b"IEND" in data[-16:]


def test_fit_modifiers_are_identifiable(fmap, manifest, morphset, canonical, conn):
    """No fit modifier is (nearly) a linear combination of the others once the similarity is removed: head-scale-vert
    was pruned because it only differs from head-scale-horiz by the removed uniform scale."""
    L = fmap["landmarks"]
    b = fm.Binding(
        np.array([l["tri"] for l in L]),
        np.array([l["bary"] for l in L]),
        np.array([l["uv"] for l in L]),
        np.zeros((468, 3)),
        np.zeros((468, 3)),
    )
    stable = fm.stable_landmarks(canonical, conn)
    fit = fmap["fitModifiers"]
    assert fm.prune_unidentifiable(morphset, manifest, b, stable, fit) == fit
    assert "head/head-scale-horiz" in fit and "head/head-scale-vert" not in fit


def test_typescript_stable_subset_matches_python(fmap, canonical, conn):
    """packages/avatar-core/src/face/fitFace.ts hard-codes the eyebrow and lip-corner indices of its stable landmark
    subset; they must equal the rule used here."""
    import re

    from config import REPO_ROOT

    src = (REPO_ROOT / "packages" / "avatar-core" / "src" / "face" / "fitFace.ts").read_text(encoding="utf-8")
    brows = re.search(r"MEDIAPIPE_EYEBROWS: readonly number\[\] = \[(.*?)\];", src, re.S).group(1)
    ts_brows = {int(x) for x in re.findall(r"\d+", re.sub(r"//.*", "", brows))}
    corners = re.search(r"LIP_CORNERS: readonly number\[\] = \[(.*?)\];", src, re.S).group(1)
    ts_corners = {int(x) for x in re.findall(r"\d+", corners)}
    py_brows = set(fm._members(conn, "FACEMESH_LEFT_EYEBROW")) | set(fm._members(conn, "FACEMESH_RIGHT_EYEBROW"))
    assert ts_brows == py_brows
    reg = fmap["regions"]
    ts_stable = set(range(468)) - set(reg["leftEye"]) - set(reg["rightEye"]) - ts_brows - (set(reg["lips"]) - ts_corners)
    assert ts_stable == set(fm.stable_landmarks(canonical, conn))
