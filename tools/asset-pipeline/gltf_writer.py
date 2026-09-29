"""base.glb writer (pygltflib): one skinned mesh, one material, skeleton nodes. No glTF morph targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib as gl

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def write_glb(
    path: Path,
    positions: np.ndarray,  # (R, 3) meters
    normals: np.ndarray,  # (R, 3)
    uvs: np.ndarray,  # (R, 2) OBJ convention (v up); flipped here for glTF
    joints: np.ndarray,  # (R, 4) uint16
    weights: np.ndarray,  # (R, 4) float32
    tris: np.ndarray,  # (T, 3)
    bone_names: list[str],
    bone_parents: list[str | None],
    bone_heads: np.ndarray,  # (B, 3) meters, world rest positions
) -> None:
    R = positions.shape[0]
    pos = np.ascontiguousarray(positions, dtype="<f4")
    nor = np.ascontiguousarray(normals, dtype="<f4")
    tex = np.ascontiguousarray(np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=1), dtype="<f4")
    jnt = np.ascontiguousarray(joints, dtype="<u2")
    wgt = np.ascontiguousarray(weights, dtype="<f4")
    idx = np.ascontiguousarray(tris.reshape(-1), dtype="<u2" if R < 65535 else "<u4")

    bone_index = {n: i for i, n in enumerate(bone_names)}
    B = len(bone_names)
    ibm = np.zeros((B, 4, 4), dtype="<f4")
    for i in range(B):
        ibm[i] = np.eye(4, dtype="<f4")
        ibm[i, 3, :3] = -bone_heads[i]  # column-major translation (row 3 of the C-order array)
    ibm = np.ascontiguousarray(ibm)

    blobs = [
        ("POSITION", pos, ARRAY_BUFFER, gl.FLOAT, gl.VEC3),
        ("NORMAL", nor, ARRAY_BUFFER, gl.FLOAT, gl.VEC3),
        ("TEXCOORD_0", tex, ARRAY_BUFFER, gl.FLOAT, gl.VEC2),
        ("JOINTS_0", jnt, ARRAY_BUFFER, gl.UNSIGNED_SHORT, gl.VEC4),
        ("WEIGHTS_0", wgt, ARRAY_BUFFER, gl.FLOAT, gl.VEC4),
        ("indices", idx, ELEMENT_ARRAY_BUFFER, gl.UNSIGNED_SHORT if idx.dtype == np.dtype("<u2") else gl.UNSIGNED_INT, gl.SCALAR),
        ("inverseBindMatrices", ibm, None, gl.FLOAT, gl.MAT4),
    ]

    g = gl.GLTF2()
    g.asset = gl.Asset(version="2.0", generator="dijital-ikiz asset-pipeline")
    data = bytearray()
    acc_index: dict[str, int] = {}
    for name, arr, target, ctype, atype in blobs:
        while len(data) % 4:
            data.append(0)
        offset = len(data)
        raw = arr.tobytes()
        data += raw
        g.bufferViews.append(gl.BufferView(buffer=0, byteOffset=offset, byteLength=len(raw), target=target))
        count = arr.shape[0]
        acc = gl.Accessor(bufferView=len(g.bufferViews) - 1, componentType=ctype, count=int(count), type=atype)
        if name == "POSITION":
            acc.min = [float(x) for x in pos.min(axis=0)]
            acc.max = [float(x) for x in pos.max(axis=0)]
        g.accessors.append(acc)
        acc_index[name] = len(g.accessors) - 1
    while len(data) % 4:
        data.append(0)
    g.buffers.append(gl.Buffer(byteLength=len(data)))

    g.materials.append(
        gl.Material(
            name="skin",
            pbrMetallicRoughness=gl.PbrMetallicRoughness(
                baseColorFactor=[0.80, 0.62, 0.52, 1.0], metallicFactor=0.0, roughnessFactor=0.75
            ),
        )
    )
    g.meshes.append(
        gl.Mesh(
            name="Body",
            primitives=[
                gl.Primitive(
                    attributes=gl.Attributes(
                        POSITION=acc_index["POSITION"],
                        NORMAL=acc_index["NORMAL"],
                        TEXCOORD_0=acc_index["TEXCOORD_0"],
                        JOINTS_0=acc_index["JOINTS_0"],
                        WEIGHTS_0=acc_index["WEIGHTS_0"],
                    ),
                    indices=acc_index["indices"],
                    material=0,
                    mode=gl.TRIANGLES,
                )
            ],
        )
    )

    # node 0 = skinned mesh node; nodes 1..B = bones (world-aligned, translation only)
    g.nodes.append(gl.Node(name="Body", mesh=0, skin=0))
    for i, name in enumerate(bone_names):
        parent = bone_parents[i]
        head = bone_heads[i]
        local = head - bone_heads[bone_index[parent]] if parent else head
        g.nodes.append(gl.Node(name=name, translation=[round(float(x), 7) for x in local], children=[]))
    root_node = None
    for i, parent in enumerate(bone_parents):
        if parent:
            g.nodes[1 + bone_index[parent]].children.append(1 + i)
        else:
            root_node = 1 + i
    for n in g.nodes:
        if not n.children:
            n.children = None  # glTF forbids empty children arrays
    g.skins.append(
        gl.Skin(
            name="Armature",
            joints=[1 + i for i in range(B)],
            skeleton=root_node,
            inverseBindMatrices=acc_index["inverseBindMatrices"],
        )
    )
    g.scenes.append(gl.Scene(name="Scene", nodes=[root_node, 0]))
    g.scene = 0
    g.set_binary_blob(bytes(data))
    g.save_binary(str(path))
