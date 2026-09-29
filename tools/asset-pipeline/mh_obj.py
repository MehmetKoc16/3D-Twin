"""MakeHuman base.obj parsing and the render-vertex (UV-split) index space.

The OBJ (CC0 data) holds 19158 vertices: the body (0..13379), helpers and 125 joint cubes of 8 vertices each.
We keep only the body faces; UV seams split some vertices, so a glTF "render vertex" is a unique (v, vt) pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BODY_GROUP = "body"
BODY_VERTEX_COUNT = 13380  # body vertices are exactly 0..13379 in hm08


@dataclass
class BaseMesh:
    verts: np.ndarray  # (19158, 3) float64, decimeters (MakeHuman axes: +Y up, +Z front)
    uvs: np.ndarray  # (n_vt, 2) float64, OBJ convention (v up)
    body_faces: np.ndarray  # (n_faces, 4, 2) int64: (vertex, uv) per corner, 0-based
    groups: dict[str, np.ndarray]  # group name -> sorted vertex ids (face membership)
    # render space
    render_mh: np.ndarray  # (R,) int64: MakeHuman vertex id of each render vertex
    render_uv: np.ndarray  # (R, 2) float64 (OBJ convention)
    quads: np.ndarray  # (Q, 4) int64 render-vertex ids of each body quad (CCW)
    tris: np.ndarray  # (T, 3) int64 indices into render vertices
    mh_to_render_start: np.ndarray  # (19158+1,) CSR offsets into mh_to_render_ids
    mh_to_render_ids: np.ndarray  # render ids sorted by mh vertex id

    @property
    def render_count(self) -> int:
        return int(self.render_mh.shape[0])

    def render_ids_of(self, mh_vertex: int) -> np.ndarray:
        s, e = self.mh_to_render_start[mh_vertex], self.mh_to_render_start[mh_vertex + 1]
        return self.mh_to_render_ids[s:e]


def _parse_obj(path: Path):
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[str, list[tuple[int, int]]]] = []
    group = ""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            p = line.split()
            if not p:
                continue
            tag = p[0]
            if tag == "v":
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif tag == "vt":
                uvs.append((float(p[1]), float(p[2])))
            elif tag == "g":
                group = p[1]
            elif tag == "f":
                corners = []
                for tok in p[1:]:
                    a = tok.split("/")
                    corners.append((int(a[0]) - 1, int(a[1]) - 1))
                faces.append((group, corners))
    return np.array(verts, dtype=np.float64), np.array(uvs, dtype=np.float64), faces


def load_base_mesh(obj_path: Path) -> BaseMesh:
    verts, uvs, faces = _parse_obj(obj_path)

    membership: dict[str, set[int]] = {}
    body_quads: list[list[tuple[int, int]]] = []
    for group, corners in faces:
        membership.setdefault(group, set()).update(c[0] for c in corners)
        if group == BODY_GROUP:
            if len(corners) != 4:
                raise ValueError("body face is not a quad")
            body_quads.append(corners)
    groups = {k: np.array(sorted(v), dtype=np.int64) for k, v in sorted(membership.items())}
    body_faces = np.array(body_quads, dtype=np.int64)

    body_v = groups[BODY_GROUP]
    if not (body_v[0] == 0 and body_v[-1] == BODY_VERTEX_COUNT - 1 and len(body_v) == BODY_VERTEX_COUNT):
        raise ValueError("unexpected body vertex range; wrong base.obj?")

    # Render vertices = unique (v, vt) pairs, ordered by (v, vt) for determinism.
    flat = body_faces.reshape(-1, 2)
    keys = flat[:, 0] * (len(uvs) + 1) + flat[:, 1]
    uniq_keys, inverse = np.unique(keys, return_inverse=True)
    render_mh = uniq_keys // (len(uvs) + 1)
    render_vt = uniq_keys % (len(uvs) + 1)
    corner_render = inverse.reshape(-1, 4)

    # Split each quad along its shorter diagonal (on the raw base mesh), keeping the winding.
    a, b, c, d = (corner_render[:, i] for i in range(4))
    pa, pb, pc, pd = (verts[render_mh[x]] for x in (a, b, c, d))
    diag_ac = np.linalg.norm(pa - pc, axis=1)
    diag_bd = np.linalg.norm(pb - pd, axis=1)
    use_ac = diag_ac <= diag_bd
    t1 = np.where(use_ac[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
    t2 = np.where(use_ac[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
    tris = np.empty((len(a) * 2, 3), dtype=np.int64)
    tris[0::2] = t1
    tris[1::2] = t2

    order = np.argsort(render_mh, kind="stable")  # render ids grouped by mh vertex
    counts = np.bincount(render_mh, minlength=len(verts))
    start = np.zeros(len(verts) + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])

    return BaseMesh(
        verts=verts,
        uvs=uvs,
        body_faces=body_faces,
        groups=groups,
        render_mh=render_mh.astype(np.int64),
        render_uv=uvs[render_vt],
        quads=corner_render.astype(np.int64),
        tris=tris,
        mh_to_render_start=start,
        mh_to_render_ids=order.astype(np.int64),
    )


def joint_cube_names(mesh: BaseMesh) -> list[str]:
    return sorted(k for k in mesh.groups if k.startswith("joint-"))


def vertex_normals(positions: np.ndarray, tris: np.ndarray, render_mh: np.ndarray) -> np.ndarray:
    """Area-weighted normals per MakeHuman vertex (shared across UV-split copies), returned per render vertex."""
    p0, p1, p2 = (positions[tris[:, i]] for i in range(3))
    fn = np.cross(p1 - p0, p2 - p0)  # magnitude = 2 * area
    acc = np.zeros((int(render_mh.max()) + 1, 3), dtype=np.float64)
    for i in range(3):
        np.add.at(acc, render_mh[tris[:, i]], fn)
    n = acc[render_mh]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln
