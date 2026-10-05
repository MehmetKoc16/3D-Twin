"""Synthetic checks for the lower hair outline and ear-protected stubble fringe."""

from types import SimpleNamespace

import numpy as np

from hybridbody.hairhy3d import ShellField
from hybridbody.hairseg import SegmentParams, VertexGraph, azimuth_of, smooth_lower_boundary


def test_temple_fringe_closes_nearby_skin_without_tinting_ears_face_or_cheeks():
    frame = SimpleNamespace(x0=0., zc=0., eye_y=1., ear_points=np.array([[.09, 1.03, 0.]]))
    points = np.array([[.08, 1.02, 0.], [.09, 1.03, 0.], [0., 1.02, .08], [.08, .98, 0.]])
    field = ShellField(None, points + [0., .008, 0.], np.full((4, 3), 30, np.uint8), [0., 1., 0.], frame)
    cover, band = field.tint_cover(points, np.zeros_like(points), np.zeros(4))
    assert cover.tolist() == [1., 0., 0., 0.]
    assert band.tolist() == [True, False, False, False]
    far, _ = field.tint_cover(points + [0., 0., .03], np.zeros_like(points), np.zeros(4))
    assert not far.any()


def test_boundary_smoothing_reduces_side_and_nape_jaggedness_and_preserves_front():
    angles = np.deg2rad(np.arange(-180, 180, 2))
    heights = np.arange(0., .10, .002)
    a, y = np.meshgrid(angles, heights)
    points = np.column_stack((.08*np.sin(a.ravel()), 1.+y.ravel(), .08*np.cos(a.ravel())))
    n = len(angles)
    faces = []
    for row in range(len(heights)-1):
        for col in range(n):
            aa, bb = row*n+col, row*n+(col+1) % n
            faces.extend([[aa, bb, aa+n], [bb, bb+n, aa+n]])
    graph = VertexGraph.from_faces(np.asarray(faces), len(points))
    frame = SimpleNamespace(x0=0., zc=0.)
    mask = points[:, 1] >= 1.04 + .004*np.sin(a.ravel()*20)
    result, report = smooth_lower_boundary(points, graph, mask, np.ones(len(points), bool), frame, SegmentParams())
    front = np.abs(azimuth_of(points, frame)) < 30
    assert np.array_equal(result[front], mask[front])
    assert report["changed_vertices"] > 0
    assert report["profile_second_difference_mm_after"] < report["profile_second_difference_mm_before"]
    assert result[points[:, 1] > 1.06].all()


def test_hairline_smoothing_rounds_a_corner_and_leaves_straight_edges():
    from hybridbody.hairseg import smooth_hairline

    n = 41
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    points = np.column_stack((0.001 * (x.ravel() - 20), 1.0 + 0.001 * y.ravel(), np.full(n * n, 0.09)))
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            faces += [[a, b, d], [a, d, c]]
    graph = VertexGraph.from_faces(np.asarray(faces), len(points))
    frame = SimpleNamespace(x0=0.0, zc=0.0)
    mask = (x.ravel() >= 20) & (y.ravel() >= 20)  # a convex corner at (20, 20)
    result, report = smooth_hairline(points, graph, mask, frame, SegmentParams.with_overrides({"hairline_smooth": 4}))
    assert report["applied"] and not result[20 * n + 20]  # the corner vertex recedes
    assert result[35 * n + 35] and not result[5 * n + 35]
    assert result[30 * n + 30] and result[(30 * n) + 20 + 5]
    off, none = smooth_hairline(points, graph, mask, frame, SegmentParams.with_overrides({"hairline_smooth": 0}))
    assert not none["applied"] and np.array_equal(off, mask)


def test_graded_tint_fades_over_a_few_millimetres_and_never_touches_the_brows():
    frame = SimpleNamespace(x0=0.0, zc=0.0, eye_y=1.0, ear_points=np.empty((0, 3)))
    shell = np.array([[0.0, 1.07, 0.09]])
    field = ShellField(None, shell, np.full((1, 3), 30, np.uint8), [0.0, 1.0, 0.0], frame)
    d = np.array([0.0, 0.002, 0.005, 0.010, 0.020])
    profile = field.front_profile(d)
    assert profile[0] == 1.0 and (np.diff(profile) <= 1e-9).all() and profile[-1] == 0.0
    assert (field.front_profile(d, legacy=True) <= profile + 1e-9).all()
    brow = np.array([[0.0, 1.0, 0.09]])
    assert field.tint_weight(brow, np.array([0.009]))[0] == 0.0 < field.tint_weight(brow, np.array([0.003]))[0]
    forehead = np.array([[0.0, 1.07, 0.09]])
    assert 0.0 < field.tint_weight(forehead, np.array([0.006]))[0] < 1.0


def test_hairline_profile_is_straight_for_a_straight_edge_and_tight_for_a_notch():
    from hybridbody.hairhy3d import hairline_profile

    frame = SimpleNamespace(x0=0.0, zc=0.0)
    rng = np.random.default_rng(1)
    a = np.radians(rng.uniform(0, 100, 60000))
    y = 1.0 + rng.uniform(0.02, 0.12, 60000)
    points = np.column_stack((0.09 * np.sin(a), y, 0.09 * np.cos(a)))
    straight = hairline_profile(points, (y > 1.07).astype(float), frame, 1.0)
    notched = hairline_profile(points, (y > 1.07 + 0.02 * (np.degrees(a) > 50)).astype(float), frame, 1.0)
    assert straight["left"]["p50_radius_mm"] > notched["left"]["p50_radius_mm"]
    assert notched["left"]["min_radius_mm"] < straight["left"]["min_radius_mm"]
