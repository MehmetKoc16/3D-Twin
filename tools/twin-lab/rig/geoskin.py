"""Geometry-only skin binding (Pinocchio / Maya "geodesic voxel binding" style) used as the comparison baseline.

Voxelises the (unposed) scan, computes the interior geodesic distance from every bone segment with multi-source
Dijkstra and turns the distances into softmax weights. It uses only our fitted joint positions, not the MakeHuman
weights, so it shows what a generic skeleton-driven auto-binder achieves.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def geodesic_weights(
    verts: np.ndarray,
    faces: np.ndarray,
    heads: np.ndarray,
    tails: np.ndarray,
    pitch: float = 0.008,
    sigma: float = 0.035,
) -> np.ndarray:
    """Returns dense per-vertex weights (n, nbones)."""
    mesh = trimesh.Trimesh(verts, faces, process=False)
    vox = mesh.voxelized(pitch).fill()
    idx = np.argwhere(vox.matrix)
    centers = vox.indices_to_points(idx)
    n = len(idx)
    key = {tuple(i): k for k, i in enumerate(idx)}
    rows, cols, vals = [], [], []
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
    for dx, dy, dz in offs:
        nb = idx + np.array([dx, dy, dz])
        for k, t in enumerate(map(tuple, nb)):
            j = key.get(t)
            if j is not None:
                rows.append(k)
                cols.append(j)
                vals.append(pitch * float(np.sqrt(dx * dx + dy * dy + dz * dz)))
    g = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    g = g + g.T
    tree = cKDTree(centers)
    nb = len(heads)
    dist = np.full((len(verts), nb), np.inf)
    vt = cKDTree(centers)
    _, vnn = vt.query(verts)
    for b in range(nb):
        seg = np.linspace(heads[b], tails[b], max(3, int(np.linalg.norm(tails[b] - heads[b]) / pitch) + 2))
        d0, near = tree.query(seg)
        src = np.unique(near[d0 < 2.5 * pitch])
        if len(src) == 0:
            src = np.unique(near[np.argsort(d0)[:2]])
        d = dijkstra(g, directed=False, indices=src, min_only=True)
        dist[:, b] = d[vnn] + np.linalg.norm(verts - centers[vnn], axis=1)
    dist[~np.isfinite(dist)] = 10.0
    dmin = dist.min(axis=1, keepdims=True)
    w = np.exp(-((dist - dmin) ** 2) / (2 * sigma**2))
    w /= w.sum(axis=1, keepdims=True)
    return w
