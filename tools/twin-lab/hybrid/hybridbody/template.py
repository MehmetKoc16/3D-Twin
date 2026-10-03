"""MakeHuman template helpers: welding, UV islands, the head region and MHCLO-style part binding.

The web app's render mesh duplicates a vertex per UV seam, so every geometric operation that must keep the surface
closed (smoothing, registration) works on a position-welded copy and is expanded back onto the render vertices.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from PIL import Image
from scipy.sparse.csgraph import connected_components


def weld(points: np.ndarray, decimals: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Exact-duplicate welding. Returns ``inv`` (welded id per vertex) and ``first`` (one vertex per welded id).

    Seam copies of a MakeHuman vertex carry bit-identical morph deltas, so a 10 nm grid never merges distinct ones.
    """
    key = np.round(np.asarray(points, np.float64) * 10.0**decimals).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return inv.ravel().astype(np.int64), first.astype(np.int64)


def uv_islands(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    """Island label per render vertex (render vertices are already split at UV seams)."""
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(vertex_count, vertex_count))
    return connected_components(graph, directed=False)[1]


def welded_edges(faces: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Unique undirected edges between welded ids (no self loops)."""
    f = inverse[faces]
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(np.sort(edges, axis=1), axis=0)


def triangle_quality(points: np.ndarray, faces: np.ndarray, reference: np.ndarray) -> dict:
    """Foldover / stretch statistics of ``points`` against the undeformed ``reference`` positions."""
    a, b = points[faces], reference[faces]
    na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    area_a, area_b = np.linalg.norm(na, axis=1) / 2, np.linalg.norm(nb, axis=1) / 2
    valid = area_b > 1e-12
    dot = np.einsum("ij,ij->i", na, nb)
    ratio = area_a[valid] / area_b[valid]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    la = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    lb = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    keep = lb > 1e-9
    stretch = la[keep] / lb[keep]
    return {
        "flipped_triangles": int(np.sum(valid & (dot < 0))),
        "degenerate_triangles": int(np.sum(area_a < 1e-12)),
        "area_ratio_min": float(ratio.min()) if len(ratio) else 1.0,
        "area_ratio_max": float(ratio.max()) if len(ratio) else 1.0,
        "edge_stretch_min": float(stretch.min()) if len(stretch) else 1.0,
        "edge_stretch_max": float(stretch.max()) if len(stretch) else 1.0,
    }


# --------------------------------------------------------------------------------------------------- parts
@dataclass
class Part:
    """One MakeHuman body part (eyes, brows, lashes, hair) with its MHCLO-style binding to the body."""

    id: str
    category: str
    positions: np.ndarray  # neutral-body positions of the shipped asset (metres)
    faces: np.ndarray
    uv: np.ndarray
    texture: np.ndarray  # (h, w, 4) uint8 RGBA, straight alpha
    base_color: np.ndarray  # linear baseColorFactor rgb
    alpha_mode: str
    tintable: bool
    scale_refs: dict
    indices: np.ndarray  # (n, 3) body render vertex ids
    weights: np.ndarray  # (n, 3)
    offsets: np.ndarray  # (n, 3) metres on the neutral body, scaled per axis at bind time
    delete_verts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    meta: dict = field(default_factory=dict)

    def compact(self, keep_faces: np.ndarray) -> Part:
        """The part restricted to ``keep_faces`` (vertices renumbered, everything else untouched)."""
        faces = self.faces[keep_faces]
        used = np.zeros(len(self.positions), bool)
        used[faces] = True
        remap = np.cumsum(used) - 1
        from dataclasses import replace

        return replace(
            self,
            positions=self.positions[used],
            faces=remap[faces],
            uv=self.uv[used],
            indices=self.indices[used],
            weights=self.weights[used],
            offsets=self.offsets[used],
        )

    def bind(self, body: np.ndarray) -> np.ndarray:
        """``bindGarment`` of avatar-core: barycentric body positions plus the axis-scaled offset."""
        scale = np.ones(3)
        for axis, key in enumerate("xyz"):
            a, b, ref = self.scale_refs[key]
            scale[axis] = abs(body[int(a), axis] - body[int(b), axis]) / ref
        anchors = np.einsum("nk,nkj->nj", self.weights, body[self.indices])
        return anchors + self.offsets * scale


def load_part(folder: Path, part_id: str) -> Part:
    from glbio import read_glb

    index = json.loads((folder / "index.json").read_text(encoding="utf8"))
    meta = next(p for p in index["parts"] if p["id"] == part_id)
    scene = read_glb(str(folder / meta["mesh"]))
    if len(scene.prims) != 1:
        raise ValueError(f"Part {part_id} must have one primitive")
    prim = scene.prims[0]
    image = Image.open(io.BytesIO(scene.images[0]["data"])).convert("RGBA")
    material = scene.materials[0]
    factor = np.asarray(material.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1])[:3], float)
    records = np.frombuffer(
        (folder / meta["binding"]).read_bytes(), np.dtype([("i", "<u4", 3), ("w", "<f4", 3), ("o", "<f4", 3)])
    )
    if len(records) != len(prim.positions):
        raise ValueError(f"Binding of {part_id} does not match its mesh")
    delete = np.zeros(0, np.int64)
    if meta.get("deleteVerts"):
        delete = np.frombuffer((folder / meta["deleteVerts"]).read_bytes(), "<u4").astype(np.int64)
    return Part(
        id=part_id,
        category=meta["category"],
        positions=prim.positions.astype(np.float64),
        faces=prim.indices.astype(np.int64),
        uv=prim.uv.astype(np.float64),
        texture=np.asarray(image),
        base_color=factor,
        alpha_mode=meta["material"]["alphaMode"],
        tintable=bool(meta["material"].get("tintable")),
        scale_refs=meta["scaleRefs"],
        indices=records["i"].astype(np.int64),
        weights=records["w"].astype(np.float64),
        offsets=records["o"].astype(np.float64),
        delete_verts=delete,
        meta=meta,
    )
