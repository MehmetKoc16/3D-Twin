import numpy as np
from helpers import peanut
from twintex.camera import OrthoCamera

from twinrefine import meshops, posecheck, render
from twinrefine.scan import welded_vertex_normals


def test_render_shades_a_sphere_and_background():
    m, _ = peanut(subdivisions=3)
    n = welded_vertex_normals(m.vertices, m.faces)
    cam = OrthoCamera.axis("front").fit_bounds(m.vertices, 120, 120, margin=0.1)
    img = render.render(m.vertices, m.faces, n, cam, 120, 120, clay=True, ss=1)
    assert img.shape == (120, 120, 3) and img.dtype == np.uint8
    bg = img[2, 2]
    inside = img[60, 60]
    assert (inside != bg).any()
    # key light from the upper left: the upper-left of the body is brighter than the lower right
    assert img[40:50, 40:50].mean() > img[75:85, 75:85].mean()


def test_render_face_albedo_override():
    m, _ = peanut(subdivisions=2)
    n = welded_vertex_normals(m.vertices, m.faces)
    cam = OrthoCamera.axis("front").fit_bounds(m.vertices, 80, 80, margin=0.1)
    red = np.tile(np.array([0.9, 0.05, 0.05], np.float32), (len(m.faces), 1))
    img = render.render(m.vertices, m.faces, n, cam, 80, 80, face_albedo=red, ss=1)
    c = img[40, 40].astype(int)
    assert c[0] > c[1] + 60 and c[0] > c[2] + 60


def test_head_sheet_writes_files(tmp_path):
    m, _ = peanut(subdivisions=3)
    n = welded_vertex_normals(m.vertices, m.faces)
    out = render.render_head_sheet(m.vertices, m.faces, n, None, None, tmp_path, prefix="t_", size=64)
    assert len(out) == 6
    for p in out.values():
        assert (tmp_path / p.split("\\")[-1].split("/")[-1]).exists()


def test_pose_mesh_tpose_raises_the_arms(model, mh_fit):
    verts = mh_fit.rest_positions[: model.nr]
    mc = meshops.to_corners(verts, model.faces, np.zeros((model.nr, 2)))
    pm = posecheck.pose_mesh(model, mh_fit, mc.P, mc.F, "t-pose")
    assert pm.rest.shape == mc.P.shape and np.isfinite(pm.posed).all()
    # unposing a body that is at the template pose changes nothing
    np.testing.assert_allclose(pm.rest, mc.P, atol=1e-6)
    # the T-pose arm span is wider than the A-pose one and the arms are level with the shoulders
    assert np.ptp(pm.posed[:, 0]) > np.ptp(mc.P[:, 0]) + 0.05
    hands = pm.posed[:, 0] > 0.55
    assert hands.any() and abs(pm.posed[hands, 1].mean() - 1.37) < 0.2
    stats = posecheck.webbing_stats(pm)
    assert stats["edges"] > 1000 and stats["max"] < 6.0
