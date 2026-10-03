"""Synthetic capped surfaces and metadata; no personal geometry or images."""

import numpy as np

from bridges import cut_bridges
from glbio import GlbScene, Prim, _split, read_glb, write_skinned_glb, write_static_glb
from surface import cap_weights, constant_uv_faces, seam_average
from mesh_qa import boundary_stats


def test_seam_average_shares_transforms_without_joining_nearby_lips():
    vertices = np.array([[0., 0, 0], [0., 0, 0], [0.002, 0, 0], [0.002, 0, 0]])
    average = seam_average(vertices)
    transforms = np.arange(36).reshape(4, 9)
    shared = average @ transforms
    np.testing.assert_array_equal(shared[0], shared[1])
    np.testing.assert_array_equal(shared[2], shared[3])
    assert not np.array_equal(shared[0], shared[2])
    assert average[0, 2] == 0


def test_subdivided_cap_interiors_inherit_their_own_lip_weights():
    n = 5
    tri = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            tri.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]])
    rim = [x for x in range(n)] + [y * n + n - 1 for y in range(1, n)]
    rim += [n * (n - 1) + x for x in range(n - 2, -1, -1)]
    rim += [y * n for y in range(n - 2, 0, -1)]
    side = [[a, b, n * n] for a, b in zip(rim, rim[1:] + rim[:1])]
    local = np.array(tri + side)
    # Two independent caps: even neighbouring surfaces must never exchange rows.
    faces = np.vstack([local, local + n * n + 1])
    cap = np.tile(np.arange(len(local)) < len(tri), 2)
    weights = np.zeros((2 * (n * n + 1), 3))
    weights[:n * n + 1, 0] = 1  # arm-side lip
    weights[n * n + 1:, 1] = 1  # torso-side lip
    inner = [y * n + x for y in range(1, n - 1) for x in range(1, n - 1)]
    unknown = np.array(inner + [i + n * n + 1 for i in inner])
    weights[unknown] = [0, 0, 1]  # arbitrary closest-point bone, must disappear
    result, stats = cap_weights(weights, faces, cap, np.arange(len(weights)))
    assert stats["capOnlyVertices"] == stats["inpaintedVertices"] == 18
    assert stats["unanchoredVertices"] == 0
    np.testing.assert_allclose(result[inner], np.tile([1, 0, 0], (len(inner), 1)), atol=1e-8)
    np.testing.assert_allclose(result[np.array(inner) + n * n + 1],
                               np.tile([0, 1, 0], (len(inner), 1)), atol=1e-8)
    np.testing.assert_allclose(result.sum(axis=1), 1)
    np.testing.assert_array_equal(result[np.setdiff1d(np.arange(len(weights)), unknown)],
                                  weights[np.setdiff1d(np.arange(len(weights)), unknown)])


def test_cap_lip_uv_copies_are_anchors_after_position_welding():
    faces = np.array([[0, 1, 2], [3, 4, 5], [3, 5, 6], [3, 6, 4]])
    # Raw cap vertices 3,4,5 duplicate textured lip vertices 0,1,2; only 6 is interior.
    inverse = np.array([0, 1, 2, 0, 1, 2, 3])
    uv = np.array([[0, 0], [1, 0], [0, 1], [.4, .7], [.4, .7], [.4, .7], [.4, .7]])
    weights = np.array([[1., 0], [1., 0], [1., 0], [0., 1]])
    result, stats = cap_weights(weights, faces, constant_uv_faces(faces, uv), inverse)
    assert stats["capOnlyVertices"] == 1
    np.testing.assert_allclose(result[:, 0], 1)
    assert not constant_uv_faces(faces, None).any()


def test_bridge_repair_keeps_closed_surface_triangles(model):
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.uint32)
    joints = np.zeros((4, 4), dtype=np.uint16)
    joints[:, 0] = model.bone_index["spine_01"]
    joints[0, 0] = model.bone_index["lowerarm_l"]
    weights = np.tile([1., 0, 0, 0], (4, 1))
    dropped, *_ = cut_bridges(faces, joints, weights, model.bone_names, list(model.parent))
    assert len(dropped) < len(faces)  # historical path introduced an opening
    kept, source, _, _, stats = cut_bridges(faces, joints, weights, model.bone_names,
                                           list(model.parent), mode="preserve")
    np.testing.assert_array_equal(kept, faces)
    np.testing.assert_array_equal(source, np.arange(4))
    assert stats["preserved_triangles"] > 0
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    assert boundary_stats(vertices, kept, 1e-5)["boundaryEdges"] == 0
    assert boundary_stats(vertices, dropped, 1e-5)["boundaryEdges"] > 0


def test_extras_round_trip_through_static_reader_and_rig_writer(tmp_path):
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    prim = Prim(vertices, np.array([[0, 1, 2]]), extras={"primitiveVendor": [1, {"ok": True}]},
                mesh_extras={"meshVendor": "keep"}, node_extras={"nodeVendor": 12})
    scene = GlbScene([prim], images=[{"data": b"synthetic image bytes", "mimeType": "image/png",
                                      "extras": {"imageVendor": "keep"}}],
                     extras={"dtFlameHead": {"version": 1, "fit": {"synthetic": True}},
                             "dtScanHandsRemoved": True, "unknown": {"nested": [1, 2]}},
                     root_extras={"rootVendor": True}, scene_extras={"sceneVendor": ["keep"]})
    source, destination = tmp_path / "static.glb", tmp_path / "rigged.glb"
    write_static_glb(str(source), scene)
    loaded = read_glb(str(source))
    joints = [{"name": "Root", "parent": None, "head": [0, 0, 0]}]
    write_skinned_glb(str(destination), loaded, joints, [np.zeros((3, 4), dtype=np.uint16)],
                      [np.tile([1., 0, 0, 0], (3, 1))])
    before, _ = _split(source.read_bytes())
    after, _ = _split(destination.read_bytes())
    assert after["asset"]["extras"] == before["asset"]["extras"]
    assert after["extras"] == before["extras"]
    assert after["scenes"][0]["extras"] == before["scenes"][0]["extras"]
    assert after["nodes"][0]["extras"] == before["nodes"][0]["extras"]
    assert after["meshes"][0]["extras"] == before["meshes"][0]["extras"]
    assert after["meshes"][0]["primitives"][0]["extras"] == before["meshes"][0]["primitives"][0]["extras"]
    assert after["images"][0]["extras"] == before["images"][0]["extras"]
