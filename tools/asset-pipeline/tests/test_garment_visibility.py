"""Synthetic collision, mask and binding regressions; no catalogue or personal inputs."""
import numpy as np
import trimesh

from garmentimport.geometry import binding, fit, prism_coverage, reconstruct, surface
from garmentimport.pose_safety import pose_delete_mask, protected_vertices, regional_clearance, stable_binding
from garmentimport.visibility import diagnostics, hidden_triangles, triangle_pairs


def synthetic_body():
    from garmentimport.body import Body
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.3)
    names = ["spine_01", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "neck_01", "head", "hand_l"]
    centers = np.array([[0, 0, 0], [.2, .2, 0], [-.2, .2, 0], [.28, 0, 0], [-.28, 0, 0],
                        [0, .28, 0], [0, .35, 0], [.38, -.1, 0]])
    bones = [{"name": n, "parent": None, "head": {"strategy": "VERTEX", "vert": len(mesh.vertices)+i},
              "tail": {"strategy": "VERTEX", "vert": len(mesh.vertices)+i}} for i, n in enumerate(names)]
    joints = np.zeros((len(mesh.vertices), 4), int)
    weights = np.tile([1., 0, 0, 0], (len(mesh.vertices), 1))
    body = Body(np.vstack([mesh.vertices, centers]), mesh.faces, joints, weights, names,
                {"bones": bones}, {}, None, None)
    body.pose = lambda name: {} if name != "walk" else {"spine_01": [0, 0, np.sin(.15), np.cos(.15)]}
    return body


def test_triangle_intersections_include_interior_and_coplanar_overlap():
    a = np.array([[-1., -1, 0], [1, -1, 0], [0, 1, 0]])
    b = np.array([[0., 0, -1], [0, 0, 1], [0, 2, 0]])
    f = np.array([[0, 1, 2]])
    assert triangle_pairs(a, f, b, f).tolist() == [[0, 0]]
    assert triangle_pairs(a, f, a+[.1, 0, 0], f).tolist() == [[0, 0]]
    assert not len(triangle_pairs(a, f, a+[3, 0, 0], f))
    assert not len(triangle_pairs(a, f, a+[0, 0, .001], f))


def test_uv_welded_adjacency_is_not_self_collision():
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    assert not len(triangle_pairs(points, faces))
    points = np.vstack([points, [[.2, .2, -1], [.2, .2, 1], [.3, .3, 0]]])
    assert len(triangle_pairs(points, np.vstack([faces, [6, 7, 8]]))) > 0


def test_visible_body_collisions_apply_all_three_corners_rule():
    body = trimesh.creation.box(extents=[.6, .6, .6])
    garment = trimesh.creation.box(extents=[.59, .59, .59])
    exposed = diagnostics(garment.vertices, garment.faces, body.vertices, body.faces, [])
    hidden = diagnostics(garment.vertices, garment.faces, body.vertices, body.faces, np.arange(len(body.vertices)))
    assert exposed['visiblePokeThroughTriangles'] > 0
    assert hidden['visiblePokeThroughTriangles'] == 0
    assert hidden['insideDeletedRegionVertices'] == len(garment.vertices)
    assert not hidden_triangles(np.array([[0, 1, 2]]), 3, [0, 1])[0]
    assert hidden_triangles(np.array([[0, 1, 2]]), 3, [0, 1, 2])[0]


def test_prism_coverage_both_sides_slack_and_depth():
    triangle = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    body = np.array([[.2, .2, .019], [.2, .2, -.019], [.2, .2, .021], [-.1, .2, 0], [2, 2, 0]])
    assert prism_coverage(body, triangle, np.array([[0, 1, 2]])).tolist() == [True, True, False, False, False]


def test_regional_clearance_smooth_and_per_vertex_fit():
    body = synthetic_body()
    points = np.array([[.2, .13, 0], [.2, .130001, 0], [0, -.8, 0], [.2, .21, 0]])
    baseline = regional_clearance(points, body, .003)
    assert baseline[0] > .006 and baseline[2] == .003
    assert baseline[3] == .003  # Collar/shoulder-cap fabric must not be inflated by the armpit field.
    assert abs(baseline[0]-baseline[1]) < 1e-6
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.3)
    garment = mesh.vertices*1.001
    clearance = np.linspace(.003, .015, len(garment))
    _, report = fit(garment, mesh.faces, mesh.vertices, mesh.faces, clearance)
    assert report['remainingInsidePercent'] == 0
    assert report['minRegionalClearanceResidualMm'] >= -.001


def test_stable_triangle_binding_round_trip_and_uv_copies():
    body = synthetic_body()
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.31)
    points = np.vstack([mesh.vertices, mesh.vertices])
    faces = np.vstack([mesh.faces, mesh.faces+len(mesh.vertices)])
    bind = binding(points, body.positions[:body.render_count], body.faces)
    refs = {k: [0, 1, abs(body.positions[0, i]-body.positions[1, i]) or 1.] for i, k in enumerate('xyz')}
    # Use well-separated real scale references on each axis.
    from garmentimport.geometry import scale_refs
    refs = scale_refs(body.positions[:body.render_count])
    stable, report = stable_binding(points, faces, bind, body, refs)
    assert report['reboundVertices'] > 0
    assert np.linalg.norm(reconstruct(stable, body.positions, refs)-points, axis=1).max() < 1e-9
    assert np.array_equal(stable[0][:len(mesh.vertices)], stable[0][len(mesh.vertices):])
    assert np.allclose(stable[1][:len(mesh.vertices)], stable[1][len(mesh.vertices):])
    assert all(tuple(row) in set(map(tuple, body.faces)) for row in stable[0])


def test_pose_union_protects_neck_hands_and_openings():
    body = synthetic_body()
    body.joints[0, 0] = body.names.index('hand_l')
    garment = trimesh.creation.icosphere(subdivisions=1, radius=.31)
    bind = binding(garment.vertices, body.positions[:body.render_count], body.faces)
    deleted, report = pose_delete_mask(body, garment.vertices, garment.faces, bind, np.arange(body.render_count))
    assert not np.intersect1d(deleted, np.flatnonzero(protected_vertices(body))).size
    assert 0 not in deleted and set(report['poseCoverageVertices']) == {'A', 'T', 'relaxed', 'walk'}
    # An open sheet's boundary must not acquire pose contact deletes.
    sheet = np.array([[-.3, 0, .31], [.3, 0, .31], [0, .3, .31]])
    mask, _ = pose_delete_mask(body, sheet, np.array([[0, 1, 2]]),
                               binding(sheet, body.positions[:body.render_count], body.faces), [])
    assert not len(mask)


def test_bounded_pose_separation_reduces_crossings_without_opening_uv_seams():
    from garmentimport.untangle import untangle
    body = synthetic_body()
    points = np.array([[0., 0, 0], [.05, 0, 0], [0, .05, 0],
                       [.01, .01, -.001], [.01, .01, .001], [.03, .03, 0]])
    points = np.vstack([points, points])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    bind = binding(points, body.positions[:body.render_count], body.faces)
    fixed, report = untangle(points, faces, bind, body, iterations=8, max_displacement=.004)
    assert sum(report['afterPairs'].values()) < sum(report['beforePairs'].values())
    assert np.allclose(fixed[:6], fixed[6:])
    assert report['maxDisplacementMm'] <= 4.000001
    assert np.isfinite(fixed).all()


def test_distant_bare_body_is_not_visible_poke_through():
    body = trimesh.creation.box(extents=[.6, .6, .6])
    garment = trimesh.creation.box(extents=[.5, .5, .5])
    report = diagnostics(garment.vertices+[0, 2, 0], garment.faces, body.vertices, body.faces, [])
    assert report['visiblePokeThroughTriangles'] == 0


def test_binding_avoids_nearby_triangle_that_collapses_in_validation_pose():
    from garmentimport.geometry import scale_refs
    body = synthetic_body()
    triangle = np.array([[.18, .12, 0], [.23, .12, 0], [.18, .17, 0]])
    vertices = np.vstack([triangle, triangle+[0, 0, -.006]])
    centers = body.positions[body.render_count:].copy()
    body.positions = np.vstack([vertices, centers])
    body.faces = np.array([[0, 1, 2], [3, 4, 5]])
    body.joints = np.zeros((6, 4), int)
    body.joints[1, 0] = 1
    body.weights = np.tile([1., 0, 0, 0], (6, 1))
    for i, bone in enumerate(body.rig['bones']):
        bone['head']['vert'] = bone['tail']['vert'] = 6+i
    def transforms(pose):
        result = np.tile(np.eye(4), (len(body.names), 1, 1))
        if pose:
            result[1, 0, 3] = -.05
        return result
    body.transforms = transforms
    points = np.array([[.181, .121, .004], [.183, .121, .004], [.181, .123, .004]])
    initial = binding(points, vertices, body.faces)
    assert np.all(initial[0] == body.faces[0])
    refs = scale_refs(vertices)
    stable, _ = stable_binding(points, np.array([[0, 1, 2]]), initial, body, refs)
    assert np.all(stable[0] == body.faces[1])
    assert np.allclose(reconstruct(stable, body.positions, refs), points)


def test_remaining_contacts_identify_protected_hand_influences():
    from garmentimport.visibility import contact_anatomy
    body = synthetic_body()
    vertices = body.faces[0]
    body.joints[vertices, 0] = body.names.index('hand_l')
    result = contact_anatomy(body, {'visibleTriangleIds': [0]})
    assert result['dominantBoneVertices'] == {'hand_l': 3}
    assert result['protectedVertices'] == 3
    assert np.asarray(result['neutralRestBoundsM']).shape == (2, 3)


def test_projected_separation_keeps_body_clearance():
    from garmentimport.untangle import untangle
    body = synthetic_body()
    points = np.array([[0., 0, .31], [.05, 0, .31], [0, .05, .31],
                       [.01, .01, .309], [.01, .01, .311], [.03, .03, .31]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    body_mesh = surface(body.positions[:body.render_count], body.faces, weld=True)
    fixed, report = untangle(points, faces, binding(points, body.positions[:body.render_count], body.faces),
                             body, iterations=4, max_displacement=.004, clearance=np.full(len(points), .003))
    assert report['clearanceProjectedEachIteration']
    assert sum(report['afterPairs'].values()) < sum(report['beforePairs'].values())
    assert not body_mesh.contains(fixed).any()
    assert body_mesh.nearest.on_surface(fixed)[1].min() >= .003-1e-6
