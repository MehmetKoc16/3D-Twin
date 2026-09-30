"""Mesh loading, cleaning, orientation and (optional) decimation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a GLB/OBJ/PLY as one geometry-only mesh (all node transforms applied, visuals dropped)."""
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    return trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )


def orient(mesh: trimesh.Trimesh, up: str = "+y", front: str = "+z") -> trimesh.Trimesh:
    """Rotate so that the mesh's ``up`` axis becomes +Y and its ``front`` axis becomes +Z."""
    u = np.array(_AXES[up.lower()], dtype=float)
    f = np.array(_AXES[front.lower()], dtype=float)
    if abs(u @ f) > 1e-6:
        raise ValueError("up and front axes must be orthogonal")
    x = np.cross(u, f)  # character's left = +X when up=+Y, front=+Z  (Y x Z = X)
    src = np.stack([x, u, f])  # rows: new X, Y, Z expressed in the source frame
    if np.allclose(src, np.eye(3)):
        return mesh
    out = mesh.copy()
    out.vertices = np.asarray(mesh.vertices) @ src.T
    if np.linalg.det(src) < 0:  # pragma: no cover - guarded by construction (right-handed triple)
        out.invert()
    return out


def clean_mesh(mesh: trimesh.Trimesh, min_component_frac: float = 0.01) -> trimesh.Trimesh:
    """Weld, drop degenerate faces and tiny floating components (keeps the body)."""
    m = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    m.merge_vertices(merge_tex=False, merge_norm=False)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    if len(m.faces) == 0:
        raise ValueError("mesh has no faces")
    comps = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1)
    if len(comps) > 1:
        sizes = np.array([len(c) for c in comps])
        keep = np.concatenate([c for c, s in zip(comps, sizes, strict=True) if s >= min_component_frac * len(m.faces)])
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[np.sort(keep)], process=False)
        m.remove_unreferenced_vertices()
    return m


def decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Quadric decimation down to ``max_faces`` (no-op when the mesh is already smaller)."""
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    import fast_simplification

    v, f = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_count=max_faces,
        agg=5,
    )
    return trimesh.Trimesh(vertices=v.astype(np.float64), faces=f.astype(np.int64), process=False)


def smooth_vertex_normals(mesh: trimesh.Trimesh, iterations: int = 2) -> np.ndarray:
    """Area-weighted vertex normals, lightly diffused over the 1-ring so noisy meshes give stable weights."""
    n = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    if iterations <= 0:
        return n
    e = mesh.edges_unique
    nv = len(mesh.vertices)
    deg = np.bincount(e.ravel(), minlength=nv).astype(np.float64)
    for _ in range(iterations):
        acc = n.copy()
        np.add.at(acc, e[:, 0], n[e[:, 1]])
        np.add.at(acc, e[:, 1], n[e[:, 0]])
        n = acc / (deg + 1.0)[:, None]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n


def load_shape_meta(shape_dir: str | Path) -> dict:
    """Best-effort read of the shape agent's normalisation JSON (all keys optional, see README)."""
    d = Path(shape_dir)
    for name in ("mesh.json", "shape.json", "normalization.json", "meta.json", "mesh_meta.json"):
        p = d / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def ensure_outward(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Flip all faces if the majority of the surface points toward the mesh centre (inverted winding)."""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted normals
    c = tri.mean(1) - v.mean(0)
    score = float(np.einsum("ij,ij->i", n, c).sum())
    if score >= 0:
        return mesh
    return trimesh.Trimesh(vertices=v, faces=f[:, ::-1], process=False)
