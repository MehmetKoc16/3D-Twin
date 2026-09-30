import numpy as np
import pytest
import trimesh
from synthetic import color_fn, make_capsule, manual_view

from twintex import meshio
from twintex.bake import (
    BakeConfig,
    blend_views,
    face_prior,
    run_bake,
    sample_view,
    texel_attributes,
    view_weights,
)
from twintex.compose import compose_atlas
from twintex.estimate import estimate_gains
from twintex.unwrap import unwrap

SIZE = 512


@pytest.fixture(scope="module")
def capsule():
    mesh = make_capsule()
    unw = unwrap(mesh, size=SIZE, padding=4)
    vn = meshio.smooth_vertex_normals(mesh, iterations=1)[unw.vmapping]
    return mesh, unw, vn


def sym_color_fn(P):
    """Mirror-symmetric ground truth (x -> |x|)."""
    Q = np.array(P, dtype=np.float64)
    Q[:, 0] = np.abs(Q[:, 0])
    return color_fn(Q)


def _bake(capsule, names, mirror=True, fn=color_fn, **cfg_kw):
    mesh, unw, vn = capsule
    views = [manual_view(mesh, n, fn=fn) for n in names]
    cfg = BakeConfig(size=SIZE, band_rows=256, **cfg_kw)
    res = run_bake(unw, vn, views, cfg, log=lambda *_: None)
    atlas, _ = compose_atlas(res, cfg, x_mid=0.0, mirror=mirror, padding=4, log=lambda *_: None)
    rr, cc, P, N = texel_attributes(unw, vn, SIZE, 0, SIZE)
    return res, atlas, rr, cc, P, N


# ------------------------------------------------------------------------------------------ weights
def test_view_weights_zero_for_backfaces_and_occluded_and_monotonic():
    cfg = BakeConfig()
    cos = np.array([-0.5, 0.0, 0.1, 0.3, 0.6, 1.0])
    vis = np.ones(6, bool)
    q, w = view_weights(cos, vis, np.ones(6, np.float32), cfg)
    assert q[0] == 0 and q[1] == 0 and w[0] == 0 and w[1] == 0
    assert np.all(np.diff(q) >= 0) and np.all(np.diff(w) >= 0)
    assert q[-1] == pytest.approx(1.0) and w[-1] == pytest.approx(1.0)
    # winner-take-most: weight ratio between cos=0.6 and 1.0 is 0.6^k
    assert w[4] / w[5] == pytest.approx(0.6**cfg.sharpness, rel=1e-3)
    # occluded texels get nothing regardless of the angle
    q2, w2 = view_weights(cos, np.zeros(6, bool), np.ones(6, np.float32), cfg)
    assert q2.sum() == 0 and w2.sum() == 0
    # feathering scales the reliability
    q3, _ = view_weights(cos, vis, np.full(6, 0.5, np.float32), cfg)
    assert np.allclose(q3, q * 0.5)


def test_face_prior_only_boosts_front_view_in_the_head_zone():
    cfg = BakeConfig(face_boost=8.0, head_frac=0.135)
    P = np.array([[0, 1.6, 0.1], [0, 0.9, 0.1], [0, 1.6, -0.1]], np.float32)
    N = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1]], np.float32)
    pf = face_prior("front", P, N, cfg, 0.0, 1.7)
    assert pf[0] == pytest.approx(8.0) and pf[1] == pytest.approx(1.0) and pf[2] == pytest.approx(1.0)
    assert face_prior("back", P, N, cfg, 0.0, 1.7) == 1.0


# ---------------------------------------------------------------------------------------- visibility
def test_occluded_texels_are_not_visible_and_seen_from_the_other_side():
    # ball A far behind ball B (same x,y), seen from the front: A is hidden by B, but visible from the back.
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
    b.vertices += [0, 0.5, 0.45]
    a = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
    a.vertices += [0, 0.5, -0.45]
    mesh = trimesh.util.concatenate([a, b])
    front = manual_view(mesh, "front")
    back = manual_view(mesh, "back")
    cfg = BakeConfig()
    # texels on A's front pole (facing the front camera, but behind B) and on B's front pole
    P = np.array([[0, 0.5, -0.45 + 0.3], [0, 0.5, 0.45 + 0.3], [0, 0.5, -0.45 - 0.3]], np.float64)
    N = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1]], np.float32)
    q_f, _, _, vis_f = sample_view(front, P, N, cfg, 1.0)
    q_b, _, _, vis_b = sample_view(back, P, N, cfg, 1.0)
    assert not vis_f[0] and q_f[0] == 0  # A's front pole is hidden behind B (as seen from the front)
    assert vis_f[1] and q_f[1] > 0.9  # B's front pole is seen by the front camera
    assert not vis_b[1]  # B's front pole is hidden from the back camera (it sits behind A from there)
    assert vis_b[2] and q_b[2] > 0.9  # A's back pole is seen from the back camera


def test_back_facing_texels_have_no_weight_in_a_view():
    mesh = make_capsule()
    front = manual_view(mesh, "front")
    cfg = BakeConfig()
    P = np.array([[0.0, 0.6, -0.22], [0.0, 0.6, 0.22]])
    N = np.array([[0, 0, -1.0], [0, 0, 1.0]], np.float32)
    q, cos, _, _ = sample_view(front, P, N, cfg, 1.2)
    assert q[0] == 0 and cos[0] < 0
    assert q[1] > 0.9


# ------------------------------------------------------------------------------------------ round trip
def test_four_view_roundtrip_reproduces_ground_truth(capsule):
    res, atlas, rr, cc, P, N = _bake(capsule, ["front", "back", "left", "right"])
    conf = res.conf[rr, cc] / 255.0
    truth = color_fn(P.astype(np.float64))
    got = atlas[rr, cc].astype(np.float32) / 255.0
    good = conf > 0.9
    assert good.mean() > 0.6  # most of the capsule is reliably seen by 4 axis views
    err = np.abs(got[good] - truth[good])
    assert err.mean() < 0.02, err.mean()
    assert np.percentile(err, 99) < 0.12
    # nothing invalid: every covered texel carries a colour close to the analytic one (fill is plausible too)
    all_err = np.abs(got - truth).mean(1)
    assert np.isfinite(atlas).all() and all_err.mean() < 0.05


def test_front_only_still_gives_a_complete_plausible_atlas(capsule):
    res, atlas, rr, cc, P, N = _bake(capsule, ["front"])
    conf = res.conf[rr, cc] / 255.0
    truth = color_fn(P.astype(np.float64))
    got = atlas[rr, cc].astype(np.float32) / 255.0
    front_facing = (N[:, 2] > 0.8) & (np.abs(P[:, 1] - 0.6) < 0.4)
    assert (conf[front_facing] > 0.9).all()
    assert np.abs(got[front_facing] - truth[front_facing]).mean() < 0.02
    back_facing = N[:, 2] < -0.5
    assert conf[back_facing].max() == 0
    # back is filled (not black / not NaN) and inside the colour range of what the front saw
    back = got[back_facing]
    assert back.min() > 0.03 and np.isfinite(back).all()
    seen = conf > 0.6  # everything the front view saw reliably (incl. the flanks)
    lo = got[seen].min(0) - 0.08
    hi = got[seen].max(0) + 0.08
    assert (back >= lo).all() and (back <= hi).all()


def test_mirror_fill_recovers_the_unseen_side(capsule):
    # front + back + right cameras: the character's left flank (+X, normal almost along +X) is seen by nobody
    kw = dict(fn=sym_color_fn)
    res, atlas, rr, cc, P, N = _bake(capsule, ["front", "back", "right"], mirror=True, **kw)
    _, atlas_nm, *_ = _bake(capsule, ["front", "back", "right"], mirror=False, **kw)
    conf = res.conf[rr, cc] / 255.0
    truth = sym_color_fn(P)
    left_flank = (N[:, 0] > 0.985) & (np.abs(P[:, 1] - 0.6) < 0.3)
    assert left_flank.sum() > 50 and conf[left_flank].max() < 0.3
    err_m = np.abs(atlas[rr, cc][left_flank] / 255.0 - truth[left_flank]).mean()
    err_n = np.abs(atlas_nm[rr, cc][left_flank] / 255.0 - truth[left_flank]).mean()
    assert err_m < 0.05, err_m
    assert err_m < err_n


def test_blend_prefers_the_front_view_on_the_face_and_is_gain_consistent(capsule):
    mesh, unw, vn = capsule
    front = manual_view(mesh, "front")
    right = manual_view(mesh, "right")
    cfg = BakeConfig(face_boost=8.0)
    # texel at 45 deg between front and right, in the head zone: front should dominate the blend
    ang = np.radians(45)
    P = np.array([[0.22 * np.sin(ang), 1.05, 0.22 * np.cos(ang)]], np.float64)
    N = np.array([[np.sin(ang), 0, np.cos(ang)]], np.float32)
    _, _, dom, _ = blend_views([front, right], P, N, cfg, 0.0, 1.42)
    assert dom[0] == 1  # face zone at the top of the capsule -> front wins (y=1.05 is inside head_frac of 1.42)
    P2 = P.copy()
    P2[0, 1] = 0.3
    _, _, dom2, _ = blend_views([front, right], P2, N, cfg, 0.0, 1.42)
    assert dom2[0] in (1, 2)


def test_colour_gain_estimation_matches_views(capsule):
    mesh, unw, vn = capsule
    front = manual_view(mesh, "front")
    right = manual_view(mesh, "right", gain=0.8)
    cfg = BakeConfig()
    _, _, P, N = texel_attributes(unw, vn, SIZE // 2, 0, SIZE // 2)
    g = estimate_gains([front, right], P, N, cfg, 0.0, 1.42)
    assert np.allclose(g["front"], 1.0)
    # image gain 0.8 in (roughly) sRGB space -> a linear gain of about 0.8^2.2 ~ 0.61: the estimate must bring the
    # right view closer to the front one (ratio > 1) and by a sensible amount
    assert (g["right"] > 1.2).all() and (g["right"] < 1.9).all()


def test_orient_maps_axes():
    # a mesh whose "up" is +Z and "front" is -Y ends up with up = +Y and front = +Z ...
    # ... and the right-handed frame is preserved (no mirroring): up x front = z x (-y) = +x -> the character's left = +X
    e = trimesh.Trimesh(vertices=[[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], faces=[[0, 1, 2]], process=False)
    oe = meshio.orient(e, up="+z", front="-y")
    assert np.allclose(oe.vertices, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_surfaces_in_front_counts_layers_along_the_ray():
    # two balls in a row along the view axis: B (near the front camera) and A (far)
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
    b.vertices += [0, 0.5, 0.45]
    a = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
    a.vertices += [0, 0.5, -0.45]
    mesh = trimesh.util.concatenate([a, b])
    front = manual_view(mesh, "front")
    tol = 0.01
    pts = np.array([
        [0, 0.5, 0.75],  # B front pole: nothing in front
        [0, 0.5, 0.15],  # B back pole: B's front surface in front -> 1
        [0, 0.5, -0.15],  # A front pole: B front + B back in front -> 2
        [0, 0.5, -0.75],  # A back pole: 3 surfaces in front
    ])
    p = front.cam.project(pts)
    n = front.surfaces_in_front(p[:, 0], p[:, 1], p[:, 2], tol)
    assert n.tolist() == [0, 1, 2, 3]


def test_opposite_prior_shines_the_front_colour_through_onto_the_back_but_not_behind_other_parts():
    from twintex.prior import opposite_prior

    mesh = make_capsule()
    front = manual_view(mesh, "front")  # smooth colour function: low-pass ~ identity
    cfg = BakeConfig()
    P = np.array([[0.0, 0.6, -0.22], [0.0, 0.6, 0.22]])
    N = np.array([[0, 0, -1.0], [0, 0, 1.0]], np.float32)
    col, w = opposite_prior([front], P, N, cfg, 1.2)
    assert w[0] > 0.9 and w[1] == 0  # only the back-facing texel gets the prior
    front_pt = np.array([[0.0, 0.6, 0.22]])
    truth = color_fn(front_pt)[0]
    from twintex.colorspace import linear_to_srgb

    assert np.abs(linear_to_srgb(col[0]) - truth).max() < 0.08
    # with a back view present the prior is off
    back = manual_view(mesh, "back")
    _, w2 = opposite_prior([front, back], P, N, cfg, 1.2)
    assert w2.sum() == 0
