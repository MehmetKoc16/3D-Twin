import numpy as np
from synth import CENTER, F, H, W, ellipsoid, make_cam

from headrecon.viewcam import iou, sample_surface, silhouette


def test_pivot_projects_to_image_centre_plus_translation():
    cam = make_cam(0.0, tx=10.0, ty=-5.0)
    p = cam.project(CENTER[None])[0]
    assert abs(p[0] - (W / 2 + 10)) < 1e-6 and abs(p[1] - (H / 2 - 5)) < 1e-6
    assert abs(p[2] - cam.dist) < 1e-9


def test_yaw_plus_90_sees_the_left_side():
    cam = make_cam(90.0)
    left, right = CENTER + [0.1, 0, 0], CENTER - [0.1, 0, 0]
    assert cam.project(left[None])[0, 2] < cam.project(right[None])[0, 2]  # +X is nearer to the camera
    assert cam.project((CENTER + [0, 0, 0.1])[None])[0, 0] < W / 2  # the nose (+Z) points to the image left
    assert make_cam(-90.0).project((CENTER + [0, 0, 0.1])[None])[0, 0] > W / 2  # -90: right side, nose to the right


def test_plane_offset_to_world_inverts_projection():
    cam = make_cam(30.0, pitch=10.0)
    p = CENTER + np.array([0.02, 0.03, -0.01])
    a = cam.project(p[None])[0]
    d_px = np.array([[5.0, -7.0]])
    dw = cam.plane_offset_to_world(d_px, a[2:3])[0]
    b = cam.project((p + dw)[None])[0]
    assert np.allclose(b[:2] - a[:2], d_px[0], atol=0.2)


def test_silhouette_of_a_sphere_has_the_expected_area():
    v, f, _ = ellipsoid((0.1, 0.1, 0.1), 3, soup=False)
    cam = make_cam(0.0)
    m = silhouette(cam, sample_surface(v, f), W, H) > 0
    r_px = F * 0.1 / np.sqrt(0.5**2 - 0.1**2)  # tangent cone of a sphere seen by a pinhole camera
    assert abs(m.sum() / (np.pi * r_px**2) - 1.0) < 0.08
    assert iou(m, m) == 1.0
