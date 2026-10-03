import numpy as np
import pytest
from hybridbody import PARTS_ASSETS
from hybridbody.template import Part, load_part, triangle_quality, uv_islands, weld, welded_edges
from synth import sphere_template


def test_weld_merges_exact_seam_copies_only():
    points = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1e-6], [0, 1, 0]], float)
    inverse, first = weld(points)
    assert inverse[0] == inverse[1] and len(first) == 4 and inverse[2] != inverse[3]
    np.testing.assert_array_equal(points[first][inverse], points)


def test_islands_and_welded_edges_on_a_uv_split_sphere():
    positions, faces, uv, _ = sphere_template(subdivisions=2)
    islands = uv_islands(faces, len(positions))
    assert islands.max() == 0  # one island: the seam copies are joined by faces
    inverse, first = weld(positions)
    assert len(first) < len(positions)  # the seam duplicated vertices
    edges = welded_edges(faces, inverse)
    # Euler for a closed genus-0 surface: V - E + F = 2
    assert len(first) - len(edges) + len(faces) == 2
    assert (edges[:, 0] < edges[:, 1]).all()


def test_triangle_quality_flags_flips_and_stretch():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    ok = triangle_quality(reference * 1.5, faces, reference)
    assert ok["flipped_triangles"] == 0 and ok["edge_stretch_min"] == pytest.approx(1.5)
    flipped = reference.copy()
    flipped[2] = [0, -1, 0]
    assert triangle_quality(flipped, faces, reference)["flipped_triangles"] >= 1


def synthetic_part():
    positions = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.0]])
    return Part(
        "p",
        "hair",
        positions,
        np.array([[0, 1, 2]]),
        np.array([[0, 0], [1, 0], [0, 1.0]]),
        np.zeros((2, 2, 4), np.uint8),
        np.ones(3),
        "MASK",
        True,
        {"x": [0, 1, 0.1], "y": [0, 2, 0.1], "z": [0, 3, 0.1]},
        np.array([[0, 1, 2]] * 3),
        np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1.0]]),
        np.array([[0, 0, 0.01], [0, 0.01, 0], [0.01, 0, 0]]),
    )


def test_bind_follows_body_and_scales_offsets_per_axis():
    part = synthetic_part()
    body = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.4, 0], [0, 0, 0.6]])
    bound = part.bind(body)
    scale = np.array([0.2 / 0.1, 0.4 / 0.1, 0.6 / 0.1])
    anchors = np.einsum("nk,nkj->nj", part.weights, body[part.indices])
    np.testing.assert_allclose(bound, anchors + part.offsets * scale)
    moved = body + [0.0, 1.0, 0.0]
    np.testing.assert_allclose(part.bind(moved), bound + [0, 1, 0])  # rigid body motion keeps scales


def test_compact_renumbers_vertices():
    part = synthetic_part()
    part.faces = np.array([[0, 1, 2], [0, 1, 2]])
    smaller = part.compact(np.array([True, False]))
    assert len(smaller.faces) == 1 and len(smaller.positions) == 3
    assert smaller.faces.max() == 2


def test_real_cc0_parts_reproduce_on_the_neutral_body(model):
    neutral = model.shape({}, {}, ground=True)
    for part_id in ("eyes-default", "hair-short", "eyebrows-default"):
        part = load_part(PARTS_ASSETS, part_id)
        np.testing.assert_allclose(part.bind(neutral), part.positions, atol=3e-4)
        assert part.texture.shape[2] == 4 and len(part.faces) > 0
    assert len(load_part(PARTS_ASSETS, "eyes-default").delete_verts) > 0
