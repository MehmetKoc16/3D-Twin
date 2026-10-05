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
