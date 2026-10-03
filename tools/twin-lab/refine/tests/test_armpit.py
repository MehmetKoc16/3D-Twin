import numpy as np
import trimesh
from helpers import peanut

from twinrefine import armpit, meshops


def _setup(subdivisions=4):
    m, x_pinch = peanut(subdivisions=subdivisions)
    uv = np.stack([m.vertices[:, 0] * 3, m.vertices[:, 1] * 3 + 0.5], axis=1).astype(np.float32) % 1.0
    mc = meshops.to_corners(m.vertices, m.faces, uv)
    aw_l = (mc.P[:, 0] > x_pinch).astype(float)  # the "arm" is the smaller sphere (x > pinch plane)
    aw_r = np.zeros(len(mc.P))
    atlas = np.full((256, 256, 3), 120, np.uint8)
    return mc, x_pinch, aw_l, aw_r, atlas


def _volume(P, F):
    return trimesh.Trimesh(P, F, process=False).volume


def test_separate_arms_closed_loop_gives_two_closed_parts():
    mc, x_pinch, aw_l, aw_r, atlas = _setup()
    prm = armpit.ArmpitParams(y_top=10.0, y_low=-10.0, gap_mm=4.0, chain_smooth=0)
    out, atlas2, rep = armpit.separate_arms(mc, atlas.copy(), aw_l, aw_r, y_apex=1.0, params=prm)
    assert rep.sides["cut_edges"] > 20
    et = meshops.edge_table(out.F)
    assert (et.count == 2).all()  # both parts are closed (caps close the lips)
    n, _lab = meshops.face_components(out.F, np.ones(len(out.F), bool))
    assert n == 2
    # the caps are planar and the arm part is only translated: the enclosed volume is preserved
    assert abs(_volume(out.P, out.F) - _volume(mc.P, mc.F)) < 0.02 * _volume(mc.P, mc.F)
    # no vertex of the two parts coincides any more (the rig stage welds by position)
    n_welded = len(np.unique(np.round(out.P * 1e5).astype(np.int64), axis=0))
    assert n_welded == len(out.P)


def test_separate_arms_gap_between_the_two_caps():
    mc, x_pinch, aw_l, aw_r, atlas = _setup()
    prm = armpit.ArmpitParams(y_top=10.0, y_low=-10.0, gap_mm=6.0, chain_smooth=0)
    out, _a, _rep = armpit.separate_arms(mc, atlas.copy(), aw_l, aw_r, y_apex=1.0, params=prm)
    _n, comp = meshops.face_components(out.F, np.ones(len(out.F), bool))
    verts_by_comp = [np.unique(out.F[comp == c].reshape(-1)) for c in (0, 1)]
    xs = [out.P[v, 0] for v in verts_by_comp]
    arm = 0 if xs[0].mean() > xs[1].mean() else 1
    # at the pinch plane the arm part starts gap_mm beyond the torso part's cap
    assert xs[arm].min() > xs[1 - arm][xs[1 - arm] > x_pinch - 0.02].max() - 1e-9
    assert xs[arm].min() - x_pinch > 0.004  # nudged outward by (nearly) the gap


def test_separate_arms_cap_uses_patch_colour_and_leaves_the_rest():
    mc, x_pinch, aw_l, aw_r, atlas = _setup()
    atlas[:] = (30, 90, 160)
    prm = armpit.ArmpitParams(y_top=10.0, y_low=-10.0, chain_smooth=0)
    out, atlas2, _rep = armpit.separate_arms(mc, atlas.copy(), aw_l, aw_r, y_apex=1.0, params=prm)
    # cap triangles carry one constant UV; its atlas block is painted with the lips' mean colour (here: the atlas colour)
    same = (np.abs(out.C[:, 0] - out.C[:, 1]).max(1) < 1e-9) & (np.abs(out.C[:, 0] - out.C[:, 2]).max(1) < 1e-9)
    assert same.sum() > 20
    uv = out.C[same][0, 0]
    px = int(uv[0] * 256), int(uv[1] * 256)
    np.testing.assert_allclose(atlas2[px[1], px[0]], (30, 90, 160), atol=2)


def test_separate_arms_open_arc_stays_one_closed_surface():
    mc, x_pinch, aw_l, aw_r, atlas = _setup()
    # zone = upper half of the pinch ring only: the cut is an open arc, the surface stays connected and closed
    prm = armpit.ArmpitParams(y_top=10.0, y_low=0.0, gap_mm=3.0, chain_smooth=2)
    out, _a, rep = armpit.separate_arms(mc, atlas.copy(), aw_l, aw_r, y_apex=1.0, params=prm)
    assert rep.sides["cut_edges"] > 8
    et = meshops.edge_table(out.F)
    assert (et.count == 2).all()
    n, _lab = meshops.face_components(out.F, np.ones(len(out.F), bool))
    assert n == 1


def test_separate_arms_noop_when_nothing_is_fused():
    mc, x_pinch, aw_l, aw_r, atlas = _setup()
    prm = armpit.ArmpitParams(y_top=-5.0, y_low=-10.0)  # zone without interface edges
    out, _a, rep = armpit.separate_arms(mc, atlas.copy(), aw_l, aw_r, y_apex=1.0, params=prm)
    assert out is mc
    assert rep.sides["cut_edges"] < prm.min_cut_edges


def test_clean_face_labels_drops_islands():
    mc, x_pinch, aw_l, aw_r, _atlas = _setup(3)
    et = meshops.edge_table(mc.F)
    vlab = armpit.vertex_labels(aw_l, aw_r)
    flab = armpit.face_labels(mc.F, vlab)
    # add a one-face arm island on the torso side and a one-face torso hole inside the arm
    torso_faces = np.flatnonzero(flab == 0)
    arm_faces = np.flatnonzero(flab == armpit.LEFT)
    dirty = flab.copy()
    dirty[torso_faces[len(torso_faces) // 2]] = armpit.LEFT
    dirty[arm_faces[len(arm_faces) // 2]] = 0
    clean = armpit.clean_face_labels(mc.F, dirty, et)
    np.testing.assert_array_equal(clean, flab)


def test_smooth_field_is_a_contraction_towards_the_mean():
    mc, *_ = _setup(2)
    phi = np.where(mc.P[:, 0] > 0.05, 1.0, -1.0)
    out = armpit.smooth_field(mc.F, phi, 5)
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert (np.abs(out) < 0.9).sum() > 10  # the interface got soft
    assert np.corrcoef(out, phi)[0, 1] > 0.9


def test_apex_height_of_the_makehuman_body(model, mh_fit):
    y = armpit.apex_height(model, mh_fit)
    assert 1.15 < y < 1.5


def test_separate_arms_cap_colour_ignores_the_shadowed_lip():
    """Photos see the contact crease in shadow: the cap takes the brighter surface around the lip, not the lip itself."""
    mc, x_pinch, aw_l, aw_r, atlas = _setup()
    size = atlas.shape[0]
    atlas[:] = (200, 190, 180)
    u0 = (3.0 * x_pinch) % 1.0
    for dx in range(-6, 7):  # dark vertical band: every texel at the pinch plane's u coordinate
        atlas[:, int(u0 * size) + dx] = (25, 20, 20)
    prm = armpit.ArmpitParams(y_top=10.0, y_low=-10.0, chain_smooth=0)
    out, atlas2, _rep = armpit.separate_arms(mc, atlas.copy(), aw_l, aw_r, y_apex=1.0, params=prm)
    same = (np.abs(out.C[:, 0] - out.C[:, 1]).max(1) < 1e-9) & (np.abs(out.C[:, 0] - out.C[:, 2]).max(1) < 1e-9)
    uv = out.C[same][0, 0]
    cap_rgb = atlas2[int(uv[1] * size), int(uv[0] * size)].astype(float)
    assert cap_rgb.min() > 150  # the lip's own colour would be ~(25, 20, 20)


def test_separate_arms_cap_colour_falls_back_to_the_lip_without_neighbours():
    mc, x_pinch, aw_l, aw_r, atlas = _setup()
    atlas[:] = (90, 100, 110)
    prm = armpit.ArmpitParams(y_top=10.0, y_low=-10.0, chain_smooth=0, colour_radius=1e-6)
    out, atlas2, _rep = armpit.separate_arms(mc, atlas.copy(), aw_l, aw_r, y_apex=1.0, params=prm)
    same = (np.abs(out.C[:, 0] - out.C[:, 1]).max(1) < 1e-9) & (np.abs(out.C[:, 0] - out.C[:, 2]).max(1) < 1e-9)
    uv = out.C[same][0, 0]
    np.testing.assert_allclose(atlas2[int(uv[1] * 256), int(uv[0] * 256)], (90, 100, 110), atol=2)
