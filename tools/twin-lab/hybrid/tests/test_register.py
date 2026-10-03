import numpy as np
import scipy.sparse as sp
import trimesh
from hybridbody.headfit import repair_foldovers
from hybridbody.register import Landmarks, Params, Surface, laplacian, register, smoothstep
from hybridbody.template import triangle_quality, weld, welded_edges
from synth import sphere_template


def welded_sphere():
    positions, faces, _, _ = sphere_template(radius=0.1, subdivisions=3)
    inverse, first = weld(positions)
    return positions[first], inverse[faces], welded_edges(faces, inverse)


def test_surface_closest_returns_bary_and_unit_normals():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    surface = Surface(mesh.vertices, mesh.faces)
    probe = np.array([[0.0, 0.0, 0.13], [0.09, 0.0, 0.0]])
    point, face, bary, normal = surface.closest(probe)
    assert np.allclose(np.linalg.norm(point, axis=1), 0.1, atol=2e-3)
    np.testing.assert_allclose(bary.sum(1), 1)
    assert (normal @ probe.T).diagonal().min() > 0
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1)
    subset = np.array([face[0]])
    far, _, _, _ = surface.closest(probe, subset)
    assert np.linalg.norm(far[1] - probe[1]) > np.linalg.norm(point[1] - probe[1])


def test_laplacian_rows_sum_to_zero():
    _, _, edges = welded_sphere()
    lap = laplacian(edges, edges.max() + 1)
    assert abs(lap.sum(1)).max() < 1e-9


def test_register_moves_free_vertices_to_the_target_and_keeps_anchors():
    base, faces, edges = welded_sphere()
    centre = base.mean(0)
    target = Surface((base - centre) * [1.08, 1.04, 1.0] + centre, faces)
    free = base[:, 1] > centre[1] - 0.04
    weight = np.where(free, 1.0, 0.0)
    displacement, info = register(
        base, faces, edges, free, target, weight, params=Params(iterations=14, stiffness_start=40, stiffness_end=0.5)
    )
    assert np.abs(displacement[~free]).max() == 0.0  # anchored vertices never move
    moved = base + displacement
    before = np.linalg.norm(target.closest(base[free])[0] - base[free], axis=1).mean()
    after = np.linalg.norm(target.closest(moved[free])[0] - moved[free], axis=1)
    assert before > 2e-3 and after.mean() < 0.35 * before and np.percentile(after, 95) < 3e-3
    assert triangle_quality(moved, faces, base)["flipped_triangles"] == 0
    assert len(info["history"]) == 14


def test_gated_vertices_follow_the_smooth_field_not_the_target():
    base, faces, edges = welded_sphere()
    centre = base.mean(0)
    target = Surface(base + [0.0, 0.0, 0.01], faces)  # everything should shift +z by 1 cm
    free = np.ones(len(base), bool)
    weight = np.where(base[:, 1] > centre[1], 1.0, 0.0)  # lower half has no data of its own
    d, _ = register(
        base, faces, edges, free, target, weight, params=Params(iterations=10, stiffness_start=40, stiffness_end=5)
    )
    assert d[base[:, 1] > centre[1] + 0.05, 2].mean() > 0.007
    lower = d[base[:, 1] < centre[1] - 0.05, 2].mean()
    assert 0.0 < lower  # pulled along by the smooth field


def test_landmark_constraints_pull_their_vertices():
    base, faces, edges = welded_sphere()
    centre = base.mean(0)
    free = base[:, 1] > centre[1] - 0.05
    weight = np.zeros(len(base))  # no surface term at all: only landmarks and smoothness
    ids = np.flatnonzero((base[:, 2] > 0.08) & free)[:5]
    bary = sp.csr_matrix((np.ones(len(ids)), (np.arange(len(ids)), ids)), shape=(len(ids), len(base)))
    goal = base[ids] + [0.0, 0.01, 0.0]
    lm = Landmarks(bary, goal, np.ones(len(ids)))
    d, _ = register(
        base,
        faces,
        edges,
        free,
        Surface(base, faces),
        weight,
        landmarks=lm,
        params=Params(iterations=4, stiffness_start=2, stiffness_end=0.5),
    )
    moved = base + d
    assert np.linalg.norm(moved[ids] - goal, axis=1).mean() < 0.35 * 0.01


def test_stiff_ear_edges_keep_a_patch_rigid():
    base, faces, edges = welded_sphere()
    centre = base.mean(0)
    patch = np.linalg.norm(base - [0.1, centre[1], 0.0], axis=1) < 0.045
    stiff = np.where(patch[edges[:, 0]] & patch[edges[:, 1]], 50.0, 1.0)
    target = Surface(base + [0.0, 0.0, 0.01], faces)
    weight = np.where(patch, 1.0, 0.0)
    free = np.ones(len(base), bool)
    free[base[:, 1] < centre[1] - 0.07] = False
    d, _ = register(
        base,
        faces,
        edges,
        free,
        target,
        weight,
        edge_stiffness=stiff,
        params=Params(iterations=8, stiffness_start=4, stiffness_end=1),
    )
    spread = np.ptp(d[patch], axis=0)
    assert spread.max() < 0.5 * np.abs(d[patch]).max() + 1e-4


def test_repair_foldovers_removes_flipped_slivers():
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    d = np.zeros_like(base)
    d[2] = [0.0, -1.5, 0.0]  # drags vertex 2 through the opposite edge: both triangles flip
    free = np.ones(4, bool)
    assert triangle_quality(base + d, faces, base)["flipped_triangles"] >= 1
    fixed, remaining = repair_foldovers(base, d, faces, free)
    assert remaining == 0 and triangle_quality(base + fixed, faces, base)["flipped_triangles"] == 0
    assert smoothstep(np.array([-1.0, 0.5, 2.0])).tolist() == [0.0, 0.5, 1.0]
