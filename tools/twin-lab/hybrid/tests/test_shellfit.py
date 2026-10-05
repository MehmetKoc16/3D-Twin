"""Fitting the hair shell to a head: the warp onto the scalp where the hair is short, and the clearance pass."""

import numpy as np
import pytest
import trimesh

from hybridbody import shellfit
from hybridbody.hairgen import HeadSurface
from hybridbody.register import Surface
from hybridbody.shellfit import FitParams

CENTRE = np.array([0.0, 1.65, 0.05])
MM = 1e-3


def sphere(radius, subdivisions):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(mesh.vertices) + CENTRE, np.asarray(mesh.faces)


@pytest.fixture(scope="module")
def head():
    vertices, faces = sphere(0.1, 4)
    return HeadSurface(vertices, faces), Surface(vertices, faces)


def cap_shell(radius, subdivisions=3):
    """The upper hemisphere of a sphere as a shell (vertices, faces)."""
    vertices, faces = sphere(radius, subdivisions)
    keep = vertices[faces][:, :, 1].min(1) > CENTRE[1] - 0.01
    used = np.unique(faces[keep])
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces[keep]]


def test_fit_params_overrides_are_validated():
    assert FitParams.with_overrides({"clearance": 2.5, "iterations": 4}).iterations == 4
    assert isinstance(FitParams.with_overrides({"iterations": 4}).iterations, int)
    with pytest.raises(ValueError, match="Unknown shell fit parameter"):
        FitParams.with_overrides({"thickness": 1})
    with pytest.raises(ValueError, match="non-negative"):
        FitParams.with_overrides({"clearance": -1})
    assert FitParams().to_dict()["clearance"] == 1.8


def test_graph_helpers_laplacian_rows_sum_to_zero_and_blocks_are_block_diagonal():
    faces = np.array([[0, 1, 2], [2, 1, 3]])
    edges = shellfit.unique_edges(faces)
    assert len(edges) == 5 and (edges[:, 0] < edges[:, 1]).all()
    laplacian = shellfit.laplacian(edges, 4)
    assert np.allclose(np.asarray(laplacian.sum(1)).ravel(), 0.0) and laplacian[1, 2] == -1
    blocks = np.stack([np.eye(3) * (k + 1) for k in range(2)])
    matrix = shellfit.block_diagonal(blocks).toarray()
    assert matrix.shape == (6, 6) and matrix[0, 0] == 1 and matrix[3, 3] == 2 and matrix[0, 3] == 0


def test_warp_pulls_short_hair_to_the_scalp_keeps_the_long_volume_and_ignores_what_is_out_of_reach(head):
    _, scalp = head
    positions, faces = cap_shell(0.108)  # 8 mm above the scalp everywhere
    height = positions[:, 1] - CENTRE[1]
    short = (height < 0.045).astype(float)  # short hair below, long volume on top
    depth = np.full(len(positions), 0.03)
    fp = FitParams()
    moved, report = shellfit.warp_to_scalp(positions, faces, depth, short, scalp, np.arange(len(scalp.faces)), fp)
    signed = np.linalg.norm(moved - CENTRE, axis=1) - 0.1
    low = height < 0.03
    assert np.abs(signed[low] - fp.thickness_short * MM).max() < 1.0 * MM  # pulled to the target thickness
    top = height > 0.085
    assert np.abs(moved[top] - positions[top]).max() < 2.0 * MM  # the long hair on top keeps its volume
    assert report["displacement_mm"]["max"] < 8.5 and report["iterations"][-1]["max_change_mm"] < 0.5
    # the correction decays smoothly into the volume: no step between neighbouring vertices
    edges = shellfit.unique_edges(faces)
    step = np.linalg.norm((moved - positions)[edges[:, 0]] - (moved - positions)[edges[:, 1]], axis=1)
    assert step.max() < 3.0 * MM
    # a shell far from the scalp is out of reach: nothing pulls it
    far_positions, far_faces = cap_shell(0.16)
    far, _ = shellfit.warp_to_scalp(
        far_positions, far_faces, np.full(len(far_positions), 0.03), np.ones(len(far_positions)), scalp,
        np.arange(len(scalp.faces)), fp,
    )  # fmt: skip
    assert np.abs(far - far_positions).max() < 1.5 * MM


def test_the_hairline_band_and_the_margin_sit_on_the_scalp_at_the_edge_thickness(head):
    _, scalp = head
    positions, faces = cap_shell(0.109)
    height = positions[:, 1] - CENTRE[1]
    depth = np.where(height < 0.02, -0.005 + height * 0.25, 0.03)  # a margin below the edge, hair above
    moved, _ = shellfit.warp_to_scalp(
        positions, faces, depth, np.zeros(len(positions)), scalp, np.arange(len(scalp.faces)), FitParams()
    )
    signed = np.linalg.norm(moved - CENTRE, axis=1) - 0.1
    edge = depth <= 0.002
    assert abs(np.median(signed[edge]) - 2.0 * MM) < 0.8 * MM  # thickness_edge, even though short = 0 there
    assert signed[depth > 0.025].mean() > 4 * MM  # the interior is left higher


def test_enforce_clearance_lifts_vertices_and_triangle_interiors_out_of_the_head(head):
    surface, _ = head
    inside, faces = cap_shell(0.097)  # 3 mm INSIDE the head
    fp = FitParams()
    fixed, report = shellfit.enforce_clearance(
        inside, faces, surface, fp.clearance * MM, fp.clearance_iterations, tolerance=2e-4
    )
    assert report["vertices_inside_head"] == 0 and report["samples_inside_head"] == 0
    assert report["vertices_below_clearance"] == 0 and report["samples_below_clearance"] == 0
    assert report["min_vertex_mm"] >= fp.clearance - 0.2
    pushed = np.linalg.norm(fixed - inside, axis=1)
    assert pushed.max() < 8 * MM and pushed.std() < 1.5 * MM  # a smooth lift (about 5 mm), no spikes
    edges = shellfit.unique_edges(faces)
    before = np.linalg.norm(inside[edges[:, 0]] - inside[edges[:, 1]], axis=1)
    after = np.linalg.norm(fixed[edges[:, 0]] - fixed[edges[:, 1]], axis=1)
    assert (after / before).max() < 1.25
    # a shell that is already clear is left alone
    clear, _ = cap_shell(0.11)
    untouched, ok = shellfit.enforce_clearance(
        clear, faces, surface, fp.clearance * MM, fp.clearance_iterations, tolerance=2e-4
    )
    assert np.array_equal(untouched, clear) and ok["passes"] == 1


def test_a_big_triangle_whose_middle_dips_into_the_head_is_lifted_until_its_samples_are_clear(head):
    surface, _ = head
    angles = np.radians([0, 40, 80])
    corners = np.array([CENTRE + 0.1005 * np.array([np.sin(a), np.cos(a), 0.0]) for a in angles])
    faces = np.array([[0, 1, 2]])
    # the chord between the corners cuts through the head: the triangle centre is inside it
    assert surface.signed_distance(corners.mean(0, keepdims=True))[0][0] < 0
    fixed, report = shellfit.enforce_clearance(corners, faces, surface, 1.8 * MM, 60, tolerance=2e-4)
    assert report["samples_inside_head"] == 0 and report["samples_below_clearance"] == 0
    assert report["min_sample_mm"] > 1.5 and np.linalg.norm(fixed - CENTRE, axis=1).min() > 0.1005
