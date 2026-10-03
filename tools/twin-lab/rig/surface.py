"""Position-seam groups and topology-based weights for solid-colour caps.

Only the existing 10 micrometre position weld is used. Nearby arm and torso
lips remain separate; no proximity repair joins the intentionally cut gap.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from bridges import inpaint_weights


def seam_average(vertices: np.ndarray) -> sp.csr_matrix:
    """Average attributes within input position-weld groups, retaining raw vertex order."""
    _, inverse = np.unique(np.round(vertices * 1e5).astype(np.int64), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = len(vertices)
    count = int(inverse.max()) + 1
    pool = sp.coo_matrix((np.ones(n), (inverse, np.arange(n))), shape=(count, n)).tocsr()
    mean = sp.diags(1 / np.bincount(inverse)) @ pool
    return (pool.T @ mean).tocsr()


def constant_uv_faces(faces: np.ndarray, uv: np.ndarray | None) -> np.ndarray:
    """Solid atlas patches have identical UV coordinates at all three corners."""
    if uv is None:
        return np.zeros(len(faces), dtype=bool)
    return np.max(np.ptp(uv[faces], axis=1), axis=1) <= 1e-8


def cap_weights(weights: np.ndarray, faces: np.ndarray, cap_faces: np.ndarray,
                inverse: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Inpaint cap-only welded vertices from the same cap's non-cap lip vertices.

    The graph contains cap triangles only, so no nearest-point search or path
    through the opposite lip decides the weights. Interior vertices of domed or
    subdivided caps use the harmonic extension of the boundary rows. Seam copies
    of a lip already share its row through `inverse`.
    """
    tri = inverse[np.asarray(faces, dtype=np.int64)]
    n = len(weights)
    inside, outside = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    inside[tri[cap_faces].ravel()] = True
    outside[tri[~cap_faces].ravel()] = True
    unknown = inside & ~outside
    result, count = inpaint_weights(weights, tri[cap_faces], unknown)
    result /= np.maximum(result.sum(axis=1, keepdims=True), 1e-12)
    return result, {"capFaces": int(cap_faces.sum()), "capOnlyVertices": int(unknown.sum()),
                    "inpaintedVertices": count, "unanchoredVertices": int(unknown.sum()) - count}
