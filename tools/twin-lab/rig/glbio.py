"""Minimal GLB reader/writer (numpy only) for the twin rigging lab.

Reads every triangle primitive of a GLB (world transforms baked in), keeps the original materials,
textures and images verbatim, and writes a skinned GLB with an arbitrary skeleton.
"""

from __future__ import annotations

import json
import struct
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class Prim:
    positions: np.ndarray  # (n,3) float32, world space
    indices: np.ndarray  # (m,3) uint32
    normals: np.ndarray | None = None
    uv: np.ndarray | None = None
    material: int | None = None
    name: str = "mesh"
    extras: dict = field(default_factory=dict)
    mesh_extras: dict = field(default_factory=dict)
    node_extras: dict = field(default_factory=dict)


@dataclass
class GlbScene:
    prims: list[Prim]
    materials: list[dict] = field(default_factory=list)
    textures: list[dict] = field(default_factory=list)
    samplers: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)  # each: {"data": bytes, "mimeType": str} or {"uri": str}
    extras: dict = field(default_factory=dict)  # asset extras, including unknown stage provenance
    root_extras: dict = field(default_factory=dict)
    scene_extras: dict = field(default_factory=dict)


def _split(data: bytes) -> tuple[dict, bytes]:
    magic, _ver, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("not a GLB file")
    off = 12
    js: dict | None = None
    binary = b""
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8 : off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk)
        elif ctype == 0x004E4942:
            binary = bytes(chunk)
        off += 8 + clen
    assert js is not None
    return js, binary


def _accessor(js: dict, binary: bytes, idx: int) -> np.ndarray:
    acc = js["accessors"][idx]
    n = _NCOMP[acc["type"]]
    dt = np.dtype(_COMP[acc["componentType"]]).newbyteorder("<")
    count = acc["count"]
    if "bufferView" in acc:
        bv = js["bufferViews"][acc["bufferView"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride")
        item = dt.itemsize * n
        if stride and stride != item:
            raw = np.frombuffer(binary, dtype=np.uint8, count=stride * count, offset=start)
            arr = np.frombuffer(raw.reshape(count, stride)[:, :item].tobytes(), dtype=dt).reshape(count, n)
        else:
            arr = np.frombuffer(binary, dtype=dt, count=count * n, offset=start).reshape(count, n)
    else:
        arr = np.zeros((count, n), dtype=dt)
    arr = arr.astype(np.float32 if dt.kind == "f" else arr.dtype)
    if acc.get("normalized") and dt.kind in "iu":
        arr = arr / np.float32(np.iinfo(dt).max)
    return arr


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def read_glb(path: str) -> GlbScene:
    with open(path, "rb") as fh:
        js, binary = _split(fh.read())
    nodes = js.get("nodes", [])
    prims: list[Prim] = []

    def visit(ni: int, parent: np.ndarray) -> None:
        node = nodes[ni]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            for p in js["meshes"][node["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                pos = _accessor(js, binary, p["attributes"]["POSITION"]).astype(np.float64)
                pos = (pos @ world[:3, :3].T + world[:3, 3]).astype(np.float32)
                nrm = None
                if "NORMAL" in p["attributes"]:
                    nrm = _accessor(js, binary, p["attributes"]["NORMAL"])
                    nrm = nrm @ np.linalg.inv(world[:3, :3]).astype(np.float32)
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
                uv = _accessor(js, binary, p["attributes"]["TEXCOORD_0"]) if "TEXCOORD_0" in p["attributes"] else None
                if "indices" in p:
                    idx = _accessor(js, binary, p["indices"]).astype(np.uint32).reshape(-1, 3)
                else:
                    idx = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
                prims.append(Prim(pos, idx, nrm, uv, p.get("material"), node.get("name", "mesh"),
                                  deepcopy(p.get("extras", {})),
                                  deepcopy(js["meshes"][node["mesh"]].get("extras", {})),
                                  deepcopy(node.get("extras", {}))))
        for c in node.get("children", []):
            visit(c, world)

    scene = js.get("scenes", [{"nodes": list(range(len(nodes)))}])[js.get("scene", 0)]
    for r in scene["nodes"]:
        visit(r, np.eye(4))

    images = []
    for im in js.get("images", []):
        if "bufferView" in im:
            bv = js["bufferViews"][im["bufferView"]]
            off = bv.get("byteOffset", 0)
            image = deepcopy(im)
            image.pop("bufferView")
            image.update(data=binary[off : off + bv["byteLength"]], mimeType=im.get("mimeType", "image/png"))
            images.append(image)
        else:
            images.append(dict(im))
    return GlbScene(
        prims=prims,
        materials=js.get("materials", []),
        textures=js.get("textures", []),
        samplers=js.get("samplers", []),
        images=images,
        extras=deepcopy(js.get("asset", {}).get("extras", {})),
        root_extras=deepcopy(js.get("extras", {})),
        scene_extras=deepcopy(scene.get("extras", {})),
    )


class _Bin:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, data: bytes, target: int | None = None) -> int:
        while len(self.buf) % 4:
            self.buf.append(0)
        v: dict = {"buffer": 0, "byteOffset": len(self.buf), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.buf += data
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, ctype: int, atype: str, target: int | None = None, minmax: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        bv = self.view(arr.tobytes(), target)
        acc: dict = {"bufferView": bv, "componentType": ctype, "count": int(arr.shape[0]), "type": atype}
        if minmax:
            acc["min"] = arr.min(axis=0).astype(float).tolist()
            acc["max"] = arr.max(axis=0).astype(float).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


def write_skinned_glb(
    path: str,
    scene: GlbScene,
    joints: list[dict],
    skin_joints: list[np.ndarray],
    skin_weights: list[np.ndarray],
    generator: str = "dijital-ikiz twin-lab rig",
) -> None:
    """Write a skinned GLB.

    joints: parents-first list of {"name", "parent" (name|None), "head": (3,)}. Rest rotations are identity, node
    translation = head - parent head, inverse bind matrices are pure translations of -head (same convention as
    apps/web/public/assets/body/base.glb).
    skin_joints[i]: (n_i,4) uint16 indices into `joints`; skin_weights[i]: (n_i,4) float32 rows summing to 1.
    """
    b = _Bin()
    index = {j["name"]: i for i, j in enumerate(joints)}
    nodes: list[dict] = []
    mesh_prims = []
    # material/texture/image remapping: keep everything verbatim
    images_js = []
    for im in scene.images:
        if "data" in im:
            bv = b.view(im["data"])
            images_js.append({**{k: deepcopy(v) for k, v in im.items() if k != "data"}, "bufferView": bv})
        else:
            images_js.append(im)
    for pi, p in enumerate(scene.prims):
        attrs = {"POSITION": b.accessor(p.positions.astype(np.float32), 5126, "VEC3", 34962, True)}
        if p.normals is not None:
            attrs["NORMAL"] = b.accessor(p.normals.astype(np.float32), 5126, "VEC3", 34962)
        if p.uv is not None:
            attrs["TEXCOORD_0"] = b.accessor(p.uv.astype(np.float32), 5126, "VEC2", 34962)
        attrs["JOINTS_0"] = b.accessor(skin_joints[pi].astype(np.uint16), 5123, "VEC4", 34962)
        attrs["WEIGHTS_0"] = b.accessor(skin_weights[pi].astype(np.float32), 5126, "VEC4", 34962)
        ia = b.accessor(p.indices.astype(np.uint32).reshape(-1), 5125, "SCALAR", 34963)
        prim = {"attributes": attrs, "indices": ia, "mode": 4}
        if p.material is not None:
            prim["material"] = p.material
        if p.extras:
            prim["extras"] = deepcopy(p.extras)
        mesh_prims.append(prim)

    n_mesh_nodes = len(scene.prims)
    # node layout: mesh nodes first, then joints (joint i at node n_mesh_nodes + i)
    for pi, p in enumerate(scene.prims):
        nodes.append({"mesh": pi, "skin": 0, "name": p.name,
                      **({"extras": deepcopy(p.node_extras)} if p.node_extras else {})})
    for i, j in enumerate(joints):
        parent = joints[index[j["parent"]]] if j["parent"] else None
        head = np.asarray(j["head"], dtype=np.float64)
        t = head - (np.asarray(parent["head"], dtype=np.float64) if parent else 0.0)
        nodes.append({"name": j["name"], "translation": [float(x) for x in t]})
    for i, j in enumerate(joints):
        if j["parent"]:
            nodes[n_mesh_nodes + index[j["parent"]]].setdefault("children", []).append(n_mesh_nodes + i)

    ibm = np.zeros((len(joints), 16), dtype=np.float32)
    for i, j in enumerate(joints):
        m = np.eye(4)
        m[:3, 3] = -np.asarray(j["head"])
        ibm[i] = m.T.reshape(-1)  # column-major
    ibm_acc = b.accessor(ibm, 5126, "MAT4")
    root_node = n_mesh_nodes + next(i for i, j in enumerate(joints) if j["parent"] is None)

    js = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [root_node, *range(n_mesh_nodes)]}],
        "nodes": nodes,
        "meshes": [{"primitives": [mp], "name": scene.prims[i].name} for i, mp in enumerate(mesh_prims)],
        "skins": [
            {
                "inverseBindMatrices": ibm_acc,
                "skeleton": root_node,
                "joints": [n_mesh_nodes + i for i in range(len(joints))],
                "name": "Armature",
            }
        ],
        "materials": scene.materials,
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.buf)}],
    }
    if scene.textures:
        js["textures"] = scene.textures
    if scene.extras:
        js["asset"]["extras"] = deepcopy(scene.extras)
    if scene.root_extras:
        js["extras"] = deepcopy(scene.root_extras)
    if scene.scene_extras:
        js["scenes"][0]["extras"] = deepcopy(scene.scene_extras)
    for mesh, p in zip(js["meshes"], scene.prims):
        if p.mesh_extras:
            mesh["extras"] = deepcopy(p.mesh_extras)
    if scene.samplers:
        js["samplers"] = scene.samplers
    if images_js:
        js["images"] = images_js
    if not js["materials"]:
        js["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.62, 0.52, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.75}}]
        for mp in js["meshes"]:
            mp["primitives"][0].setdefault("material", 0)

    jb = json.dumps(js, separators=(",", ":")).encode()
    jb += b" " * (-len(jb) % 4)
    bb = bytes(b.buf) + b"\0" * (-len(b.buf) % 4)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        fh.write(struct.pack("<II", len(bb), 0x004E4942) + bb)


def write_static_glb(path: str, scene: GlbScene) -> None:
    """Write the scene's primitives without any skin (used to make a raw stand-in mesh)."""
    b = _Bin()
    images_js = []
    for im in scene.images:
        if "data" in im:
            images_js.append({**{k: deepcopy(v) for k, v in im.items() if k != "data"},
                              "bufferView": b.view(im["data"])})
        else:
            images_js.append(im)
    meshes, nodes = [], []
    for pi, p in enumerate(scene.prims):
        attrs = {"POSITION": b.accessor(p.positions.astype(np.float32), 5126, "VEC3", 34962, True)}
        if p.normals is not None:
            attrs["NORMAL"] = b.accessor(p.normals.astype(np.float32), 5126, "VEC3", 34962)
        if p.uv is not None:
            attrs["TEXCOORD_0"] = b.accessor(p.uv.astype(np.float32), 5126, "VEC2", 34962)
        ia = b.accessor(p.indices.astype(np.uint32).reshape(-1), 5125, "SCALAR", 34963)
        prim = {"attributes": attrs, "indices": ia, "mode": 4, "material": p.material if p.material is not None else 0}
        if p.extras:
            prim["extras"] = deepcopy(p.extras)
        meshes.append({"primitives": [prim], "name": p.name})
        nodes.append({"mesh": pi, "name": p.name})
        if p.mesh_extras:
            meshes[-1]["extras"] = deepcopy(p.mesh_extras)
        if p.node_extras:
            nodes[-1]["extras"] = deepcopy(p.node_extras)
    js = {
        "asset": {"version": "2.0", "generator": "dijital-ikiz twin-lab rig"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": scene.materials
        or [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.62, 0.52, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.75}}],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.buf)}],
    }
    if scene.textures:
        js["textures"] = scene.textures
    if scene.extras:
        js["asset"]["extras"] = deepcopy(scene.extras)
    if scene.root_extras:
        js["extras"] = deepcopy(scene.root_extras)
    if scene.scene_extras:
        js["scenes"][0]["extras"] = deepcopy(scene.scene_extras)
    if scene.samplers:
        js["samplers"] = scene.samplers
    if images_js:
        js["images"] = images_js
    jb = json.dumps(js, separators=(",", ":")).encode()
    jb += b" " * (-len(jb) % 4)
    bb = bytes(b.buf) + b"\0" * (-len(b.buf) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb) + 8 + len(bb)))
        fh.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        fh.write(struct.pack("<II", len(bb), 0x004E4942) + bb)
