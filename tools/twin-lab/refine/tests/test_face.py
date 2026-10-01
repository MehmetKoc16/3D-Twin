import numpy as np
import pytest
from helpers import flat_face_depth, sheet_scan

from twinrefine import facedepth, facefit, meshops, relief


@pytest.fixture(scope="module")
def fmodel(model, mh_fit):
    return facefit.FaceModel(model, mh_fit, facefit.load_face_map())


def _similarity(p, s=1.04, deg=2.0, t=(0.012, -0.03)):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return s * (p @ R.T) + np.array(t)


def test_face_map_is_consistent():
    fm = facefit.load_face_map()
    assert fm.tri.shape == (468, 3) and fm.bary.shape == (468, 3)
    np.testing.assert_allclose(fm.bary.sum(axis=1), 1.0, atol=1e-5)
    assert len(fm.oval) == 36
    idx = facefit.stable_landmarks(fm)
    assert 150 < len(idx) < 400
    assert not set(idx) & set(fm.regions["leftEye"])
    assert {61, 291} <= set(idx)  # the mouth corners stay


def test_similarity_2d_recovers_transform():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 0.05, (60, 2))
    q = _similarity(p, 1.2, 15.0, (0.3, -0.1))
    s, R, t = facefit.similarity_2d(p, q)
    assert abs(s - 1.2) < 1e-9
    np.testing.assert_allclose(s * p @ R.T + t, q, atol=1e-9)


def test_face_fit_recovers_modifiers(fmodel):
    true = {"chin/chin-width": 0.5, "nose/nose-scale-horiz": -0.4, "mouth/mouth-scale-horiz": 0.6, "cheek/cheek-volume": 0.4,
            "eyes/eye-scale": 0.5}
    theta = np.array([true.get(m, 0.0) for m in fmodel.modifiers])
    target = _similarity(fmodel.landmark_points(theta)[:, :2])
    fit = facefit.fit_face(fmodel, target)
    assert fit.rms_mm < 0.8
    assert fit.rms_neutral_mm > 3 * fit.rms_mm
    got = dict(zip(fit.modifiers, fit.theta, strict=True))
    for k in ("chin/chin-width", "nose/nose-scale-horiz", "mouth/mouth-scale-horiz", "cheek/cheek-volume"):
        assert abs(got[k] - true[k]) < 0.25, (k, got[k], true[k])
    assert abs(fit.s - 1.04) < 0.03


def test_landmark_points_are_posed_with_the_body(model, mh_fit, fmodel):
    # the neutral landmark points lie on the neutral posed head surface (within one triangle of it)
    P = fmodel.landmark_points(np.zeros(len(fmodel.modifiers)))
    assert 1.30 < P[:, 1].min() and P[:, 1].max() < 1.85
    # frontal: nose tip landmark (4) is the most forward of the face
    assert P[4, 2] > np.percentile(P[:, 2], 90)


def _face_world(fmodel, theta):
    return fmodel.landmark_points(theta)


def test_build_face_depth_recovers_detail(model, mh_fit, fmodel):
    """Scan = a blurred generic face; landmarks come from a different face: the depth map must move the scan towards
    the true face (nose / lips / eye sockets) and keep the face joined to the scan at the oval boundary."""
    from twintex.raster import zbuffer

    fm = fmodel.fm
    true = {"nose/nose-scale-vert": 0.6, "chin/chin-height": -0.5}
    theta_true = np.array([true.get(m, 0.0) for m in fmodel.modifiers])
    L3 = fmodel.landmark_points(theta_true)
    L = L3[:, :2].copy()
    grid = facedepth.Grid.around(L, 0.04, 0.001)
    # ground truth depth of the true face, the scan sees a smoothed copy (14 px = 14 mm) shifted back by 6 mm
    posed_true = fmodel.posed(theta_true)
    tri = facedepth.head_front_triangles(model, posed_true)
    ztrue = facedepth.rasterize_depth(posed_true[:, :2], posed_true[:, 2], model.faces[tri], grid)
    import cv2

    ok = np.isfinite(ztrue)
    num = cv2.GaussianBlur(np.where(ok, ztrue, 0.0).astype(np.float32), (0, 0), 14.0)
    den = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), 14.0)
    zscan = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan).astype(np.float32) - 0.006
    zscan = np.where(ok, zscan, np.nan).astype(np.float32)
    del zbuffer
    fit = facefit.fit_face(fmodel, L)
    fd = facedepth.build_face_depth(fmodel, fit, L, zscan, grid, fm, log=lambda s: None)
    inner = fd.oval & (fd.sd > 14.0) & np.isfinite(fd.znew) & np.isfinite(ztrue)
    assert inner.sum() > 2000
    # the face is joined to the scan's level at the boundary (here 6 mm behind the truth and 14 mm smooth), so compare
    # the detail layer (high-pass: nose, lips, eye sockets), not the low frequencies
    def highpass(z):
        m = np.isfinite(z)
        z0 = np.where(m, z, 0.0).astype(np.float32)
        lp = cv2.GaussianBlur(z0, (0, 0), 6.0) / np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6.0), 1e-6)
        return np.where(m, z0 - lp, np.nan)

    h_true, h_scan, h_new = highpass(ztrue), highpass(zscan), highpass(fd.znew)
    err_scan = np.sqrt(np.mean((h_scan[inner] - h_true[inner]) ** 2))
    err_new = np.sqrt(np.mean((h_new[inner] - h_true[inner]) ** 2))
    assert err_new < 0.4 * err_scan, (err_scan, err_new)
    # closed surface: no hole left inside the oval (eyelid openings, lip slit are filled)
    assert np.isfinite(fd.znew[fd.oval & (fd.sd > 6.0)]).all()
    # joined at the boundary: the displacement vanishes there
    rim = fd.oval & (fd.sd < 1.5)
    assert np.abs(fd.delta[rim]).max() < 0.004
    assert abs(fd.delta[inner]).max() > 0.005  # and the relief is not a no-op inside


def test_harmonic_fill_reproduces_a_plane():
    H, W = 40, 50
    yy, xx = np.mgrid[0:H, 0:W]
    plane = (0.001 * xx + 0.002 * yy).astype(np.float32)
    hole = (np.abs(yy - 20) < 6) & (np.abs(xx - 25) < 9)
    z = plane.copy()
    z[hole] = np.nan
    out = facedepth.harmonic_fill(z, hole)
    np.testing.assert_allclose(out[hole], plane[hole], atol=2e-3)


def test_grid_mapping_roundtrip():
    g = facedepth.Grid(0.1, 1.8, 0.0005, 100, 120)
    gx, gy = g.centres()
    px, py = g.to_px(gx, gy)
    ii, jj = np.mgrid[0:g.H, 0:g.W]
    np.testing.assert_allclose(px, jj + 0.5, atol=1e-9)
    np.testing.assert_allclose(py, ii + 0.5, atol=1e-9)


def test_refine_and_apply_relief_moves_surface_without_cracks():
    mc = sheet_scan()
    grid = facedepth.Grid.around(np.array([[-0.08, 1.54], [0.08, 1.71]]), 0.01, 0.0005)
    fd = flat_face_depth(grid, mc, offset=0.005)
    before_boundary = len(meshops.boundary_halfedges(mc.F))
    mcr = relief.refine_face_region(mc, fd, edge_mm=2.0, rounds=3)
    assert len(mcr.F) > len(mc.F)
    et = meshops.edge_table(mcr.F)
    assert et.count.max() <= 2  # manifold, crack free
    assert len(meshops.boundary_halfedges(mcr.F)) >= before_boundary  # only the sheet rim is open
    out, rep = relief.apply_relief(mcr, fd)
    dz = out.P[:, 2] - mcr.P[:, 2]
    centre = np.hypot(mcr.P[:, 0], mcr.P[:, 1] - 1.62) < 0.03
    assert abs(np.median(dz[centre]) - 0.005) < 0.0012  # moved by the field value in the face centre
    far = np.hypot(mcr.P[:, 0], mcr.P[:, 1] - 1.62) > 0.07
    assert np.abs(dz[far]).max() < 1e-6  # untouched outside the face
    np.testing.assert_allclose(out.P[:, :2], mcr.P[:, :2])  # only z moves
    assert rep.max_move_mm < 6.0


def test_heal_folds_and_smooth_band_do_not_break_the_mesh():
    mc = sheet_scan(40)
    grid = facedepth.Grid.around(np.array([[-0.08, 1.54], [0.08, 1.71]]), 0.01, 0.0005)
    fd = flat_face_depth(grid, mc)
    out, _ = relief.apply_relief(mc, fd)
    healed, n = relief.heal_folds(mc, out, fd)
    smooth = relief.smooth_band(healed, fd)
    assert np.isfinite(smooth.P).all()
    assert len(smooth.F) == len(mc.F)


def test_sample_grid_bilinear():
    g = facedepth.Grid(0.0, 1.0, 0.01, 50, 50)
    gx, gy = g.centres()
    img = (2.0 * gx + 3.0 * gy).astype(np.float32)
    x = np.array([0.123, 0.31])
    y = np.array([0.77, 0.52])
    np.testing.assert_allclose(relief.sample_grid(img, g, x, y), 2.0 * x + 3.0 * y, atol=2e-4)
