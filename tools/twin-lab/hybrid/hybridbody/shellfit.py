"""Fit the hair shell to the twin head: a smooth warp onto the scalp where the hair is short, the volume kept on top.

The shell comes from the bust, whose skull differs from the twin's by up to a centimetre (the twin head is a statistical
FLAME fit of photos; the back of both heads is a guess). So after the rigid alignment the shell is warped:

* **where the hair is short** (below the taper line of the cut: the faded sides, the back, and a band along the whole
  hairline including the margin that the alpha channel hides) the shell is pulled to a target ``thickness`` above the
  scalp (``2 mm`` at the hairline, a few mm inside), so no skin shows between the shell and the scalp at its edge;
* **where it is long** (the top, the swept-up front) nothing pulls: the volume is the person's hair, not the twin's skull;
* between the two the warp is a Laplacian-regularised displacement field (a membrane on the regular shell grid) with a
  weak anchor, so the correction decays smoothly into the volume instead of making a step.

The solve is the same linear system as ``register.py`` (point-to-plane + weak point term to the closest scalp point,
membrane smoothness, one sparse solve per iteration, closest points refreshed). A last pass pushes every vertex (and the
interior of every triangle: centroids and edge midpoints) at least ``clearance`` outside the head surface, spreading the
push over the neighbours, so the shell has zero penetration with the whole head (ears and face included).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .register import smoothstep

MM = 1e-3
INT_PARAMS = ("iterations", "clearance_iterations")
CELL_AREA = (3.4 * MM) ** 2  # surface per vertex of the shell grid the weights are tuned for (3.4 mm edges)


@dataclass(frozen=True)
class FitParams:
    """Tunables of the warp (millimetres)."""

    thickness_edge: float = 2.0  # target height above the scalp at the hairline (and in the margin)
    thickness_short: float = 3.5  # ... inside the short hair, from ``short_ramp`` mm inside the hairline
    short_ramp: float = 25.0
    edge_band: float = 7.0  # the pull is full this far inside the hairline and fades beyond
    reach_full: float = 24.0  # a vertex farther than this from the scalp is out of reach (full weight below it) ...
    reach_zero: float = 40.0  # ... and ignored beyond this (the long hair on top)
    clearance: float = 1.8  # minimum distance of every vertex and triangle sample from the head surface
    stiffness: float = (
        8.0  # membrane weight (data weight 1 per 3.4 x 3.4 mm of surface: independent of the mesh density)
    )
    anchor: float = 0.06  # pull of the displacement towards zero where nothing constrains the shell
    point_weight: float = 0.08  # weak point-to-point term next to the point-to-plane one
    iterations: int = 10
    clearance_iterations: int = 40

    @classmethod
    def with_overrides(cls, overrides: dict) -> FitParams:
        names = {f.name for f in fields(cls)}
        values = {}
        for key, value in overrides.items():
            if key not in names:
                raise ValueError(f"Unknown shell fit parameter {key!r}; known: {', '.join(sorted(names))}")
            number = float(value)
            if not np.isfinite(number) or number < 0:
                raise ValueError(f"Shell fit parameter {key} must be a finite, non-negative number")
            values[key] = int(number) if key in INT_PARAMS else number
        return replace(cls(), **values)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    return np.unique(edges, axis=0)


def laplacian(edges: np.ndarray, count: int) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    adjacency = (a + a.T).tocsr()
    return sp.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency


def block_diagonal(blocks: np.ndarray) -> sp.csr_matrix:
    """(M, 3, 3) -> sparse (3M, 3M) block diagonal."""
    m = len(blocks)
    base = np.repeat(np.arange(m), 9) * 3
    rows = base + np.tile(np.repeat(np.arange(3), 3), m)
    cols = base + np.tile(np.arange(3), 3 * m)
    return sp.csr_matrix((blocks.reshape(-1), (rows, cols)), shape=(3 * m, 3 * m))


@dataclass
class Fit:
    positions: np.ndarray
    report: dict


def warp_to_scalp(
    positions: np.ndarray,
    faces: np.ndarray,
    depth: np.ndarray,
    short: np.ndarray,
    scalp,
    scalp_faces: np.ndarray,
    params: FitParams,
) -> tuple[np.ndarray, dict]:
    """Displaced vertex positions.

    ``depth`` (metres) is the distance of every vertex inside the hairline (negative in the margin), ``short`` (0..1)
    how short the hair is there (1 below the taper line), ``scalp`` a ``register.Surface`` of the head without the ears and
    ``scalp_faces`` the faces to search. Returns the positions and a report.
    """
    count = len(positions)
    laplace = sp.kron(laplacian(unique_edges(faces), count), sp.eye(3), format="csr")
    # data and anchor weights per surface area, so a denser or coarser mesh behaves the same
    tri = positions[faces]
    face_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    vertex_area = np.zeros(count)
    for k in range(3):
        np.add.at(vertex_area, faces[:, k], face_area / 3.0)
    area_weight = vertex_area / CELL_AREA
    edge_weight = 1.0 - smoothstep((depth - params.edge_band * MM) / (params.edge_band * MM))  # 1 at / outside the edge
    edge_weight = np.where(depth <= params.edge_band * MM, 1.0, edge_weight)
    thickness = (
        params.thickness_edge
        + (params.thickness_short - params.thickness_edge)
        * smoothstep(np.maximum(depth, 0.0) / (params.short_ramp * MM))
    ) * MM
    displacement = np.zeros_like(positions)
    history = []
    for step in range(params.iterations):
        current = positions + displacement
        point, _, _, normal = scalp.closest(current, scalp_faces)
        offset = current - point
        signed = np.einsum("ij,ij->i", offset, normal)
        reach = 1.0 - smoothstep(
            (np.abs(signed) - params.reach_full * MM) / ((params.reach_zero - params.reach_full) * MM)
        )
        weight = np.clip(np.maximum(short * reach, edge_weight * reach), 0.0, 1.0)
        target = point + normal * thickness[:, None]
        outer = np.einsum("ni,nj->nij", normal, normal)
        blocks = (weight * area_weight)[:, None, None] * (outer + params.point_weight * np.eye(3)[None])
        anchor = params.anchor * (1.0 - weight) * area_weight
        matrix = block_diagonal(blocks) + params.stiffness * laplace + sp.diags(np.repeat(anchor, 3) + 1e-9)
        rhs = np.einsum("nij,nj->ni", blocks, target - positions).reshape(-1)
        solution = spsolve(matrix.tocsc(), rhs).reshape(count, 3)
        change = float(np.abs(solution - displacement).max())
        displacement = solution
        history.append(
            {
                "step": step,
                "max_change_mm": change * 1000,
                "pulled_vertices": int((weight > 0.5).sum()),
                "mean_signed_mm_of_pulled": float(signed[weight > 0.5].mean() * 1000) if (weight > 0.5).any() else 0.0,
            }
        )
    moved = positions + displacement
    norm = np.linalg.norm(displacement, axis=1)
    return moved, {
        "iterations": history,
        "displacement_mm": {
            "max": float(norm.max() * 1000),
            "mean": float(norm.mean() * 1000),
            "p95": float(np.percentile(norm, 95) * 1000),
            "pulled_mean": float(norm[weight > 0.5].mean() * 1000) if (weight > 0.5).any() else 0.0,
        },
    }


def triangle_samples(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = positions[faces]
    return np.concatenate(
        (tri.mean(1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 2] + tri[:, 0]) / 2)
    )


def enforce_clearance(
    positions: np.ndarray, faces: np.ndarray, head, clearance: float, iterations: int, tolerance: float = 1e-4
) -> tuple[np.ndarray, dict]:
    """Push the vertices out of the head until every vertex and triangle sample is ``clearance`` away from its surface.

    ``head`` is a ``hairgen.HeadSurface`` (signed distance, negative inside). The push of a violating vertex is spread to
    its neighbours (decaying by 25 percent per ring) so the surface rises as a whole instead of spiking.
    """
    edges = unique_edges(faces)
    out = positions.copy()
    passes = 0
    for _ in range(iterations):
        passes += 1
        signed, _, normal = head.signed_distance(out, candidates=24)
        need = np.maximum(clearance - signed, 0.0)
        samples = triangle_samples(out, faces)
        s_signed, _, s_normal = head.signed_distance(samples, candidates=24)
        s_need = np.maximum(clearance - s_signed, 0.0)
        # a violating triangle sample lifts its three corners along the sample's push direction
        per_corner = np.zeros(len(out))
        push_dir = np.zeros_like(out)
        corner_count = len(faces)
        for block in range(4):
            lo, hi = block * corner_count, (block + 1) * corner_count
            need_b = s_need[lo:hi]
            sel = need_b > tolerance
            if not sel.any():
                continue
            for k in range(3):
                np.maximum.at(per_corner, faces[sel, k], need_b[sel])
                push_dir[faces[sel, k]] = s_normal[lo:hi][sel]
        use_sample = per_corner > need
        magnitude = np.maximum(need, per_corner)
        direction = np.where(use_sample[:, None], push_dir, normal)
        if magnitude.max() <= tolerance:
            break
        # spread to the neighbours, decaying
        spread = magnitude.copy()
        for _ in range(3):
            neighbour = np.zeros_like(spread)
            np.maximum.at(neighbour, edges[:, 0], spread[edges[:, 1]])
            np.maximum.at(neighbour, edges[:, 1], spread[edges[:, 0]])
            spread = np.maximum(spread, 0.75 * neighbour)
        spread_dir = direction.copy()
        weak = magnitude < 1e-12
        # neighbours without a direction of their own take the head normal at their position
        spread_dir[weak] = normal[weak]
        out = out + spread_dir * spread[:, None]
    signed, _, _ = head.signed_distance(out, candidates=24)
    s_signed, _, _ = head.signed_distance(triangle_samples(out, faces), candidates=24)
    return out, {
        "passes": passes,
        "clearance_mm": clearance * 1000,
        "min_vertex_mm": float(signed.min() * 1000),
        "min_sample_mm": float(s_signed.min() * 1000),
        "vertices_below_clearance": int((signed < clearance - tolerance).sum()),
        "samples_below_clearance": int((s_signed < clearance - tolerance).sum()),
        "vertices_inside_head": int((signed < 0).sum()),
        "samples_inside_head": int((s_signed < 0).sum()),
    }
