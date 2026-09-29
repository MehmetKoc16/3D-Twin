"""Rig JSON, skin weights and joint index validity."""

from __future__ import annotations

import numpy as np
import pygltflib


def test_rig_structure(rig, manifest):
    R, V = manifest["renderVertexCount"], manifest["vertexCount"]
    bones = rig["bones"]
    assert len(bones) == 53
    names = [b["name"] for b in bones]
    assert len(set(names)) == len(names)
    seen = set()
    roots = 0
    for b in bones:
        assert b["roll"] == 0
        if b["parent"] is None:
            roots += 1
        else:
            assert b["parent"] in seen, "parents must precede children"
        seen.add(b["name"])
        for end in ("head", "tail"):
            j = b[end]
            if j["strategy"] == "VERTEX":
                assert R <= j["vert"] < V, (b["name"], end, j)
            elif j["strategy"] == "MEAN":
                assert all(R <= v < V for v in j["verts"])
            else:
                assert j["strategy"] == "FIXED" and len(j["position"]) == 3
    assert roots == 1


def test_all_joint_points_used_by_rig(rig, manifest):
    R = manifest["renderVertexCount"]
    used = set()
    for b in rig["bones"]:
        for end in ("head", "tail"):
            if b[end]["strategy"] == "VERTEX":
                used.add(b[end]["vert"] - R)
    assert used == set(range(len(manifest["jointPoints"])))


def read_accessor(g, blob, index):
    acc = g.accessors[index]
    view = g.bufferViews[acc.bufferView]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc.type]
    dtype = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[acc.componentType]
    offset = view.byteOffset + (acc.byteOffset or 0)
    return np.frombuffer(blob, dtype=dtype, count=acc.count * comps, offset=offset).reshape(acc.count, comps)


def test_skin_weights(out_a, rig):
    g = pygltflib.GLTF2().load_binary(str(out_a / "base.glb"))
    blob = g.binary_blob()
    at = g.meshes[0].primitives[0].attributes
    joints = read_accessor(g, blob, at.JOINTS_0).astype(np.int64)
    weights = read_accessor(g, blob, at.WEIGHTS_0).astype(np.float64)
    assert joints.shape == weights.shape == (14517, 4)
    assert joints.max() < len(rig["bones"])
    assert (weights >= 0).all()
    assert np.abs(weights.sum(axis=1) - 1.0).max() < 1e-6
    assert ((weights > 0).sum(axis=1) <= 4).all()
    assert ((weights > 0).sum(axis=1) >= 1).all()
    # zero-weight slots point at joint 0 (glTF convention); active slots use distinct joints
    assert (joints[weights == 0] == 0).all()
    for row_j, row_w in zip(joints[::7], weights[::7]):
        active = row_j[row_w > 0]
        assert len(set(active.tolist())) == len(active)


def test_glb_skeleton_matches_rig(out_a, rig):
    g = pygltflib.GLTF2().load_binary(str(out_a / "base.glb"))
    skin = g.skins[0]
    assert [g.nodes[j].name for j in skin.joints] == [b["name"] for b in rig["bones"]]
    ibm = read_accessor(g, g.binary_blob(), skin.inverseBindMatrices).reshape(-1, 4, 4)
    assert ibm.shape[0] == len(rig["bones"])
    # bind pose = world-aligned translations: inverse bind matrix is translation-only (column-major storage)
    assert np.array_equal(ibm[:, :3, :3], np.broadcast_to(np.eye(3), (ibm.shape[0], 3, 3)))
    assert np.array_equal(ibm[:, 3, 3], np.ones(ibm.shape[0]))
    # world head position from the node hierarchy == -IBM translation
    parent_of = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}

    def world(i):
        t = np.array(g.nodes[i].translation or [0.0, 0.0, 0.0])
        return t + (world(parent_of[i]) if i in parent_of else 0.0)

    for k, j in enumerate(skin.joints):
        assert np.allclose(world(j), -ibm[k, 3, :3], atol=1e-6), g.nodes[j].name
    # rig head positions (VERTEX joint points) agree with the glTF bind pose
    # (checked in test_morphs via joint point positions)


def test_bind_pose_matches_joint_points(out_a, rig, manifest):
    g = pygltflib.GLTF2().load_binary(str(out_a / "base.glb"))
    R = manifest["renderVertexCount"]
    pts = {R + j: p["position"] for j, p in enumerate(manifest["jointPoints"])}
    ibm = read_accessor(g, g.binary_blob(), g.skins[0].inverseBindMatrices).reshape(-1, 4, 4)
    for k, b in enumerate(rig["bones"]):
        assert np.allclose(pts[b["head"]["vert"]], -ibm[k, 3, :3], atol=1e-5), b["name"]


def test_root_bone_spans_ground_to_pelvis(rig, manifest):
    R = manifest["renderVertexCount"]
    names = [j["name"] for j in manifest["jointPoints"]]
    bones = {b["name"]: b for b in rig["bones"]}
    assert bones["Root"]["head"] == {"strategy": "VERTEX", "vert": R + names.index("joint-ground")}
    assert bones["Root"]["tail"] == bones["pelvis"]["head"]
    assert all(b[e]["strategy"] == "VERTEX" for b in rig["bones"] for e in ("head", "tail"))
