"""Non-rigid registration of the template head onto a fitted surface (Laplacian-regularised NICP).

The unknown is one displacement per free (head) vertex of the position-welded template; vertices of the neck and
body are anchored at exactly zero, so their positions (and every measurement taken there) are unchanged. Every
iteration finds closest points on the target, then solves one sparse linear system

    min  sum_v w_v ( a (n_v . (b_v + D_v - c_v))^2 + b |b_v + D_v - c_v|^2 )          surface (plane + point)
       + sum_l w_l | sum_k bary_lk (b_k + D_k) - l_l |^2                               landmarks
       + lambda sum_edges s_e |D_i - D_j|^2                                           smooth displacement field

so the template keeps its own detail (ears, lips, nostrils) and only the smooth difference to the target is applied.
``lambda`` is annealed from stiff to loose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import trimesh
from flamehead.geometry import closest_surface
from scipy.sparse.linalg import spsolve


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Surface:
    """A triangle mesh that can answer closest-point queries with barycentrics and interpolated normals."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray = field(init=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, np.float64)
        self.faces = np.asarray(self.faces, np.int64)
        self.normals = trimesh.Trimesh(self.vertices, self.faces, process=False).vertex_normals.copy()

    def closest(self, points: np.ndarray, face_subset: np.ndarray | None = None):
        """(closest points, face ids, barycentrics, unit normals); ``face_subset`` restricts the faces searched."""
        subset = np.arange(len(self.faces)) if face_subset is None else np.asarray(face_subset, np.int64)
        point, local = closest_surface(self.vertices, self.faces[subset], points)
        face = subset[local]
        bary = trimesh.triangles.points_to_barycentric(self.vertices[self.faces[face]], point)
        bary = np.clip(bary, 0.0, 1.0)
        bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
        normal = np.einsum("nk,nkj->nj", bary, self.normals[self.faces[face]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        return point, face, bary, normal


@dataclass
class Landmarks:
    """Point constraints: ``bary`` (J, W) sparse over welded template vertices, ``targets`` (J, 3), ``weight`` (J,)."""

    bary: sp.csr_matrix
    targets: np.ndarray
    weight: np.ndarray


@dataclass
class Params:
    iterations: int = 16
    stiffness_start: float = 400.0
    stiffness_end: float = 6.0
    max_distance_start: float = 0.030
    max_distance_end: float = 0.012
    min_normal_dot: float = 0.55
    plane_weight: float = 1.0
    point_weight: float = 0.08
    landmark_weight: float = 6.0
    ridge: float = 1e-7


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return trimesh.Trimesh(points, faces, process=False).vertex_normals


def laplacian(edges: np.ndarray, count: int, edge_weight: np.ndarray | None = None) -> sp.csr_matrix:
    w = np.ones(len(edges)) if edge_weight is None else edge_weight
    a = sp.coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(count, count))
    adjacency = (a + a.T).tocsr()
    return sp.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency


def _block_diag3(blocks: np.ndarray) -> sp.csr_matrix:
    """(M, 3, 3) -> sparse (3M, 3M) block diagonal."""
    m = len(blocks)
    base = np.repeat(np.arange(m), 9) * 3
    rows = base + np.tile(np.repeat(np.arange(3), 3), m)
    cols = base + np.tile(np.arange(3), 3 * m)
    return sp.csr_matrix((blocks.reshape(-1), (rows, cols)), shape=(3 * m, 3 * m))


def register(
    base: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    free: np.ndarray,
    surface: Surface,
    data_weight: np.ndarray,
    *,
    landmarks: Landmarks | None = None,
    edge_stiffness: np.ndarray | None = None,
    alt_surface: Surface | None = None,
    alt_mask: np.ndarray | None = None,
    alt_plane_weight: float = 0.3,
    alt_min_normal_dot: float = 0.2,
    params: Params | None = None,
    log=None,
) -> tuple[np.ndarray, dict]:
    """Displacements (W, 3) of the welded template ``base`` toward ``surface``; vertices with ``free`` False stay put.

    ``data_weight`` (W,) gates the surface term per vertex (0 disables it, e.g. cavities and the neck).
    ``alt_surface`` / ``alt_mask`` give a second target for a vertex subset (ears match only the FLAME ear patch).
    ``edge_stiffness`` (E,) multiplies the smoothness weight of single edges (ears are kept stiff).
    """
    params = params or Params()
    count = len(base)
    free_ids = np.flatnonzero(free)
    local = np.full(count, -1, np.int64)
    local[free_ids] = np.arange(len(free_ids))
    m = len(free_ids)
    lap = laplacian(edges, count, edge_stiffness)[free_ids][:, free_ids]
    lap3 = sp.kron(lap, sp.eye(3), format="csr")
    displacement = np.zeros((count, 3))
    history = []
    if landmarks is not None:
        bary_free = landmarks.bary[:, free_ids].tocsr()
        residual_fixed = landmarks.targets - landmarks.bary @ base
        lm_weight = sp.diags(landmarks.weight * params.landmark_weight)
        lm_matrix = (bary_free.T @ lm_weight @ bary_free).tocsr()
        lm_matrix3 = sp.kron(lm_matrix, sp.eye(3), format="csr")
        lm_rhs = np.asarray(bary_free.T @ (lm_weight @ residual_fixed))
    alt = np.zeros(count, bool) if alt_mask is None else np.asarray(alt_mask, bool)
    for step in range(params.iterations):
        t = step / max(params.iterations - 1, 1)
        stiffness = params.stiffness_start * (params.stiffness_end / params.stiffness_start) ** t
        limit = params.max_distance_start + (params.max_distance_end - params.max_distance_start) * t
        position = base + displacement
        normal = vertex_normals(position, faces)
        point, _, _, target_normal = surface.closest(position[free_ids])
        valid_normal = np.einsum("ij,ij->i", normal[free_ids], target_normal) >= params.min_normal_dot
        plane = np.full(m, params.plane_weight)
        if alt_surface is not None and alt[free_ids].any():
            sel = alt[free_ids]
            a_point, _, _, a_normal = alt_surface.closest(position[free_ids][sel])
            point[sel], target_normal[sel] = a_point, a_normal
            valid_normal[sel] = np.einsum("ij,ij->i", normal[free_ids][sel], a_normal) >= alt_min_normal_dot
            plane[sel] = alt_plane_weight
        distance = np.linalg.norm(point - position[free_ids], axis=1)
        weight = data_weight[free_ids] * (distance <= limit) * valid_normal
        # Late iterations lean on the plane term (sliding along the target), early ones on the point term.
        point_weight = params.point_weight * (1.0 + 4.0 * (1.0 - t))
        outer = np.einsum("ni,nj->nij", target_normal, target_normal)
        blocks = weight[:, None, None] * (plane[:, None, None] * outer + point_weight * np.eye(3)[None])
        matrix = _block_diag3(blocks) + stiffness * lap3 + params.ridge * sp.eye(3 * m)
        rhs = np.einsum("nij,nj->ni", blocks, point - base[free_ids]).reshape(-1)
        if landmarks is not None:
            matrix = matrix + lm_matrix3
            rhs = rhs + lm_rhs.reshape(-1)
        solution = spsolve(matrix.tocsc(), rhs).reshape(m, 3)
        change = float(np.abs(solution - displacement[free_ids]).max())
        displacement[free_ids] = solution
        history.append(
            {
                "step": step,
                "stiffness": stiffness,
                "limit_mm": limit * 1000,
                "constrained": int((weight > 0).sum()),
                "max_change_mm": change * 1000,
                "mean_distance_mm": float(distance[weight > 0].mean() * 1000) if (weight > 0).any() else 0.0,
            }
        )
        if log:
            log(
                f"register step {step}: stiffness {stiffness:.1f}, {history[-1]['constrained']} constrained, "
                f"mean distance {history[-1]['mean_distance_mm']:.2f} mm, change {change * 1000:.2f} mm"
            )
    return displacement, {"history": history}
