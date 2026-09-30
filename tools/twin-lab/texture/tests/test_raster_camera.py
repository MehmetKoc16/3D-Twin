import numpy as np

from twintex.camera import OrthoCamera
from twintex.raster import barycentric_at, rasterize_uv, zbuffer


def test_single_triangle_coverage_and_barycentrics():
    xy = np.array([[1.0, 1.0], [9.0, 1.0], [1.0, 9.0]])
    faces = np.array([[0, 1, 2]])
    tri, bary = rasterize_uv(xy, faces, 12, 12)
    covered = tri >= 0
    # pixel centres on the hypotenuse (x+y=10) count as covered (shared-edge tolerance)
    for py in range(12):
        for px in range(12):
            inside = (px + 0.5 > 1) and (py + 0.5 > 1) and (px + 0.5 + py + 0.5 <= 10)
            assert covered[py, px] == inside, (px, py)
    assert np.allclose(bary[covered].sum(-1), 1.0, atol=1e-5)
    # barycentric reproduces the pixel centre
    ys, xs = np.nonzero(covered)
    rec = bary[ys, xs] @ xy[faces[0]]
    assert np.allclose(rec, np.stack([xs + 0.5, ys + 0.5], 1), atol=1e-4)


def test_adjacent_triangles_leave_no_gaps():
    # a quad split along its diagonal, with a non-integer position: every centre inside is covered
    xy = np.array([[2.3, 2.7], [17.6, 2.7], [17.6, 15.2], [2.3, 15.2]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    tri, _ = rasterize_uv(xy, faces, 20, 20)
    for py in range(20):
        for px in range(20):
            inside = 2.3 < px + 0.5 < 17.6 and 2.7 < py + 0.5 < 15.2
            assert (tri[py, px] >= 0) == inside


def test_many_random_triangles_match_bruteforce():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 40, size=(30, 2))
    faces = rng.integers(0, 30, size=(25, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    tri, bary = rasterize_uv(xy, faces, 40, 40)
    cov = np.zeros((40, 40), bool)
    for f in faces:
        a, b, c = xy[f]
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(det) < 1e-9:
            continue
        for py in range(40):
            for px in range(40):
                p = np.array([px + 0.5, py + 0.5])
                lb = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / det
                lc = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / det
                if min(1 - lb - lc, lb, lc) >= -1e-7:
                    cov[py, px] = True
    assert ((tri >= 0) == cov).all()


def test_zbuffer_nearest_wins_and_barycentric_at():
    # two overlapping triangles at different depths; smaller depth = nearer
    xyz = np.array(
        [[0, 0, 5.0], [10, 0, 5.0], [0, 10, 5.0],  # far
         [0, 0, 1.0], [10, 0, 1.0], [0, 10, 1.0]],  # near
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    depth, fid = zbuffer(xyz, faces, 10, 10)
    assert (fid[depth < np.inf] == 1).all()
    assert np.isclose(depth[2, 2], 1.0)
    # order of the faces must not matter
    depth2, fid2 = zbuffer(xyz, faces[::-1], 10, 10)
    assert (fid2[depth2 < np.inf] == 0).all() and np.allclose(depth, depth2)
    lam = barycentric_at(xyz[:, :2], faces, np.array([1]), np.array([2]), np.array([3]))
    assert np.allclose(lam.sum(), 1.0)


def test_axis_cameras_are_right_handed_and_mirror_correctly():
    p_left_of_char = np.array([[0.3, 1.0, 0.0]])  # character's left = +X
    p_front = np.array([[0.0, 1.0, 0.2]])  # in front of the character = +Z
    front = OrthoCamera.axis("front", scale=100, tx=50, ty=50)
    back = OrthoCamera.axis("back", scale=100, tx=50, ty=50)
    left = OrthoCamera.axis("left", scale=100, tx=50, ty=50)
    right = OrthoCamera.axis("right", scale=100, tx=50, ty=50)
    # seen from the front the character's left is on the image right; from the back it is on the image left
    assert front.project(p_left_of_char)[0, 0] > 50 > back.project(p_left_of_char)[0, 0]
    # in the left view (camera on +X) the front of the character points to the image left
    assert left.project(p_front)[0, 0] < 50
    assert right.project(p_front)[0, 0] > 50
    # y grows downwards, +Y up
    assert front.project(np.array([[0, 2.0, 0]]))[0, 1] < front.project(np.array([[0, 1.0, 0]]))[0, 1]
    # depth: things nearer to the camera have smaller depth
    assert front.project(np.array([[0, 0, 1.0]]))[0, 2] < front.project(np.array([[0, 0, -1.0]]))[0, 2]
    assert back.project(np.array([[0, 0, -1.0]]))[0, 2] < back.project(np.array([[0, 0, 1.0]]))[0, 2]


def test_azimuth_camera_matches_axis_cameras():
    for name, az in (("front", 0), ("left", 90), ("back", 180), ("right", 270)):
        a = OrthoCamera.azimuth(name, az)
        b = OrthoCamera.axis(name)
        assert np.allclose(a.forward, b.forward, atol=1e-9) and np.allclose(a.right, b.right, atol=1e-9)
