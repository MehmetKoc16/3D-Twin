"""Topology operations on a welded triangle mesh with per-corner UVs: edge table, wedge cut, edge refinement,
boundary extraction, polygon triangulation.

Representation (``Corners``): welded vertex positions ``P`` (nw, 3), faces ``F`` (m, 3) of welded ids and per-corner
UVs ``C`` (m, 3, 2). UV seams are not vertices here: a welded vertex simply carries a different UV in each face
corner, which makes every topological edit independent of the atlas layout. ``to_corners`` / ``from_corners``
convert from / to the index-space arrays (vertices duplicated per distinct UV) that the GLB stores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .scan import weld_ids


@dataclass
class Corners:
    P: np.ndarray  # (nw, 3) float64
    F: np.ndarray  # (m, 3) int64
    C: np.ndarray  # (m, 3, 2) float32

    def copy(self) -> Corners:
        return Corners(self.P.copy(), self.F.copy(), self.C.copy())


def to_corners(verts: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> Corners:
    inv, first = weld_ids(verts)
    P = np.asarray(verts, dtype=np.float64)[first]
    F = inv[faces]
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return Corners(P, F[keep], np.asarray(uv, dtype=np.float32)[faces[keep]])


def from_corners(mc: Corners, return_welded: bool = False):
    """Index-space arrays (verts, faces, uv): one vertex per distinct (welded id, uv) pair. With ``return_welded``
    also the welded id of every index vertex (to carry per-vertex data such as a pose along)."""
    m = len(mc.F)
    vid = mc.F.reshape(-1)
    uvc = mc.C.reshape(-1, 2)
    key = np.concatenate([vid[:, None].astype(np.float64), uvc.astype(np.float64)], axis=1)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    verts = mc.P[vid[first]]
    uv = uvc[first].astype(np.float32)
    faces = inv.reshape(m, 3).astype(np.int64)
    if return_welded:
        return verts, faces, uv, vid[first]
    return verts, faces, uv


# ------------------------------------------------------------------------------------------------ edge table
@dataclass
class EdgeTable:
    he_a: np.ndarray  # (3m,) start vertex of every half-edge (half-edge h = 3 * face + corner)
    he_b: np.ndarray
    edge_of: np.ndarray  # (3m,) unique-edge id of every half-edge
    n_edges: int
    count: np.ndarray  # (E,) faces per unique edge
    he_pair: np.ndarray  # (3m,) opposite half-edge of manifold interior edges, -1 otherwise

    @property
    def interior(self) -> np.ndarray:
        return self.count == 2


def edge_table(F: np.ndarray) -> EdgeTable:
    m = len(F)
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    nw = int(F.max()) + 1
    key = lo * nw + hi
    _u, edge_of, count = np.unique(key, return_inverse=True, return_counts=True)
    edge_of = edge_of.ravel()
    order = np.argsort(edge_of, kind="stable")
    pair = np.full(3 * m, -1, dtype=np.int64)
    first = np.cumsum(np.concatenate([[0], count[:-1]]))
    two = np.flatnonzero(count == 2)
    h1 = order[first[two]]
    h2 = order[first[two] + 1]
    pair[h1] = h2
    pair[h2] = h1
    return EdgeTable(a, b, edge_of.reshape(-1), len(count), count, pair)


def face_components(F: np.ndarray, face_mask: np.ndarray, et: EdgeTable | None = None) -> tuple[int, np.ndarray]:
    """Connected components (through shared edges) of the faces selected by ``face_mask``; label -1 elsewhere."""
    et = et or edge_table(F)
    m = len(F)
    h = np.flatnonzero(et.he_pair >= 0)
    f1, f2 = h // 3, et.he_pair[h] // 3
    ok = face_mask[f1] & face_mask[f2]
    g = coo_matrix((np.ones(int(ok.sum())), (f1[ok], f2[ok])), shape=(m, m))
    n, lab = connected_components(g, directed=False)
    out = np.where(face_mask, lab, -1)
    # relabel to 0..k-1 over the selected faces
    sel = np.unique(out[out >= 0])
    remap = -np.ones(lab.max() + 1, dtype=np.int64)
    remap[sel] = np.arange(len(sel))
    out = np.where(out >= 0, remap[np.maximum(out, 0)], -1)
    return len(sel), out


# ------------------------------------------------------------------------------------------------ wedge cut
def cut_along_edges(mc: Corners, cut_edge: np.ndarray, et: EdgeTable | None = None) -> tuple[Corners, np.ndarray]:
    """Split the mesh along the unique edges flagged in ``cut_edge`` (bool per unique edge of ``edge_table(mc.F)``).

    Every vertex is duplicated once per fan of faces around it that stays connected through uncut edges, so a cut
    arc with free ends opens a slit (the end vertices stay single), a closed cut separates two surfaces.
    Returns the new mesh (faces keep their order and UVs) and ``src`` (new vertex -> old vertex).
    """
    et = et or edge_table(mc.F)
    F = mc.F
    m = len(F)
    # union-find through scipy: nodes are corners (3 * face + k)
    rows, cols = [], []
    he = np.arange(3 * m)
    pair = et.he_pair
    ok = (pair >= 0) & ~cut_edge[et.edge_of]
    h1 = he[ok]
    h2 = pair[ok]
    f1, k1 = h1 // 3, h1 % 3
    f2, k2 = h2 // 3, h2 % 3
    # vertex a (start of h1) lives at corner k1 of f1 and corner (k2 + 1) % 3 of f2; vertex b at (k1 + 1) % 3 / k2
    rows += [3 * f1 + k1, 3 * f1 + (k1 + 1) % 3]
    cols += [3 * f2 + (k2 + 1) % 3, 3 * f2 + k2]
    # non-manifold edges stay connected: chain their half-edges in order
    nm = np.flatnonzero(et.count > 2)
    if len(nm):
        order = np.argsort(et.edge_of, kind="stable")
        eo = et.edge_of[order]
        for e in nm:
            hs = order[np.searchsorted(eo, e) : np.searchsorted(eo, e, side="right")]
            for i in range(len(hs) - 1):
                ha, hb = int(hs[i]), int(hs[i + 1])
                fa, ka = divmod(ha, 3)
                fb, kb = divmod(hb, 3)
                va0, va1 = F[fa, ka], F[fa, (ka + 1) % 3]
                for kk in range(3):
                    if F[fb, kk] == va0:
                        rows.append(np.array([3 * fa + ka]))
                        cols.append(np.array([3 * fb + kk]))
                    if F[fb, kk] == va1:
                        rows.append(np.array([3 * fa + (ka + 1) % 3]))
                        cols.append(np.array([3 * fb + kk]))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(3 * m, 3 * m))
    ncomp, lab = connected_components(g, directed=False)
    corner_vertex = F.reshape(-1)
    # one new vertex per component (a component never mixes old vertices)
    src = np.zeros(ncomp, dtype=np.int64)
    src[lab] = corner_vertex
    newF = lab.reshape(m, 3)
    return Corners(mc.P[src], newF, mc.C.copy()), src


# ------------------------------------------------------------------------------------------------ boundary
def boundary_halfedges(F: np.ndarray, et: EdgeTable | None = None) -> np.ndarray:
    """Half-edges (3 * face + corner) of edges used by exactly one face."""
    et = et or edge_table(F)
    return np.flatnonzero(et.count[et.edge_of] == 1)


def chain_halfedges(F: np.ndarray, hes: np.ndarray) -> list[tuple[list[int], bool]]:
    """Chain boundary half-edges (a -> b) into vertex paths. Returns [(vertex list, closed)], where a closed loop
    lists every vertex once (the last connects to the first) and an open path runs start -> end."""
    a = F.reshape(-1)[hes]
    b = F[:, [1, 2, 0]].reshape(-1)[hes]
    nxt: dict[int, list[int]] = {}
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        nxt.setdefault(x, []).append(y)
    indeg: dict[int, int] = {}
    for y in b.tolist():
        indeg[y] = indeg.get(y, 0) + 1
    used: set[tuple[int, int]] = set()
    out: list[tuple[list[int], bool]] = []
    starts = [v for v in nxt if indeg.get(v, 0) == 0]
    for s in starts + [v for v in nxt if v not in starts]:
        while nxt.get(s):
            path = [s]
            cur = s
            closed = False
            while True:
                cand = [y for y in nxt.get(cur, []) if (cur, y) not in used]
                if not cand:
                    break
                y = cand[0]
                used.add((cur, y))
                nxt[cur].remove(y)
                if y == s:
                    closed = True
                    break
                path.append(y)
                cur = y
            if len(path) > 1 or closed:
                out.append((path, closed))
            if not closed and path[-1] == s:
                break
    return out


# ------------------------------------------------------------------------------------------------ polygons
def _area2(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def ear_clip(poly: np.ndarray) -> np.ndarray | None:
    """Triangulate a simple polygon given as (n, 2) points in order (either orientation). Returns (n - 2, 3) vertex
    indices with the orientation of the input order, or None when no ear can be found (self-intersecting input)."""
    n = len(poly)
    if n < 3:
        return None
    idx = list(range(n))
    sign = 1.0 if _area2(poly) > 0 else -1.0
    tris: list[tuple[int, int, int]] = []

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        found = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if sign * cross(a, b, c) <= 1e-18:
                continue  # reflex or degenerate
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = poly[j]
                if sign * cross(a, b, p) >= 0 and sign * cross(b, c, p) >= 0 and sign * cross(c, a, p) >= 0:
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            found = True
            break
        if not found:
            # drop a degenerate (collinear) vertex if any, otherwise give up
            drop = None
            for k in range(len(idx)):
                if abs(cross(poly[idx[k - 1]], poly[idx[k]], poly[idx[(k + 1) % len(idx)]])) <= 1e-18:
                    drop = k
                    break
            if drop is None:
                return None
            idx.pop(drop)
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return np.array(tris, dtype=np.int64)


def zipper_triangulate(path: list[int], P: np.ndarray, min_side: int = 3) -> np.ndarray | None:
    """Triangulate the lens between the two halves of an open U-shaped lip path (down one crease, round the bottom tip,
    up the other crease) as a ladder: the two halves are paired by arc length, every rung is short (the width of the
    lens) instead of the long fan edges of an ear-clipped sliver, so the cap deforms like a strip when the lips move.

    ``path`` runs along the lip's half-edge direction; the triangles contain the REVERSED lip edges (the orientation a cap
    needs). Returns (k, 3) vertex ids or None when the path is not U shaped."""
    n = len(path)
    if n < 2 * min_side:
        return None
    y = P[np.asarray(path), 1]
    m = int(np.argmin(y))
    if m < min_side - 1 or m > n - min_side:
        return None
    s1 = list(path[: m + 1])  # first tip ... bottom tip
    s2 = list(reversed(path[m:]))  # second tip ... bottom tip (path[m] is shared)
    if len(s2) < min_side:
        return None

    def arclen(ids: list[int]) -> np.ndarray:
        q = P[np.asarray(ids)]
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        return d / max(d[-1], 1e-12)

    t1, t2 = arclen(s1), arclen(s2)
    i = j = 0
    tris = []
    while i < len(s1) - 1 or j < len(s2) - 1:
        adv1 = j >= len(s2) - 1 or (i < len(s1) - 1 and t1[i + 1] <= t2[j + 1])
        if adv1:
            tri = (s1[i], s2[j], s1[i + 1])
            i += 1
        else:
            tri = (s1[i], s2[j], s2[j + 1])
            j += 1
        if len(set(tri)) == 3:
            tris.append(tri)
    return np.array(tris, dtype=np.int64) if tris else None


def triangulate_loop_3d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Triangulate a closed 3-D vertex loop (n, 3) that is roughly planar. Returns (tris (k, 3) loop indices keeping
    the loop's orientation, extra_points (None unless a centre vertex had to be added; then index n is the centre))."""
    n = len(pts)
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axes = vt  # principal axes; vt[2] is the plane normal
    p2 = (pts - c) @ axes[:2].T
    # make the 2-D orientation agree with the 3-D loop orientation (Newell normal vs vt[2])
    nn = np.cross(pts - c, np.roll(pts, -1, axis=0) - c).sum(axis=0)
    if nn @ axes[2] < 0:
        p2[:, 1] *= -1.0
    t = ear_clip(p2)
    if t is not None and len(t) == n - 2:
        return t, None
    # fallback: fan from an added centre vertex (always topologically valid)
    tris = np.array([(i, (i + 1) % n, n) for i in range(n)], dtype=np.int64)
    return tris, c[None, :]


# ------------------------------------------------------------------------------------------------ refinement
def edge_endpoints(et: EdgeTable) -> tuple[np.ndarray, np.ndarray]:
    """Canonical (lo, hi) vertex pair of every unique edge."""
    lo = np.zeros(et.n_edges, dtype=np.int64)
    hi = np.zeros(et.n_edges, dtype=np.int64)
    lo[et.edge_of] = np.minimum(et.he_a, et.he_b)
    hi[et.edge_of] = np.maximum(et.he_a, et.he_b)
    return lo, hi


def refine_marked_edges(
    mc: Corners, edge_marked: np.ndarray, et: EdgeTable | None = None, edge_t: np.ndarray | None = None
) -> tuple[Corners, np.ndarray]:
    """Split every marked unique edge and re-triangulate the faces around (1 -> 2 / 3 / 4), so the result stays
    crack free. UVs are interpolated per face corner. ``edge_t`` (per unique edge, default 0.5) is the split position
    from the edge's lower to its higher vertex id. Returns the new mesh and ``is_new`` (bool per vertex of the
    result: True for the inserted vertices)."""
    et = et or edge_table(mc.F)
    F, C, P = mc.F, mc.C, mc.P
    m = len(F)
    if not edge_marked.any():
        return mc, np.zeros(len(P), dtype=bool)
    lo, hi = edge_endpoints(et)
    t_e = np.full(et.n_edges, 0.5) if edge_t is None else np.asarray(edge_t, dtype=np.float64)
    hm = edge_marked[et.edge_of].reshape(m, 3)  # half-edge k of face f runs corner k -> k + 1
    nm = hm.sum(axis=1)
    marked_ids = np.flatnonzero(edge_marked)
    mid_of_edge = -np.ones(et.n_edges, dtype=np.int64)
    mid_of_edge[marked_ids] = len(P) + np.arange(len(marked_ids))
    newP = np.vstack([P, P[lo[marked_ids]] + t_e[marked_ids, None] * (P[hi[marked_ids]] - P[lo[marked_ids]])])
    mid = mid_of_edge[et.edge_of].reshape(m, 3)  # midpoint vertex id per half-edge (valid where hm)
    # fraction of the way from corner k to corner k + 1 at which half-edge k is split
    frac = np.where(F == lo[et.edge_of].reshape(m, 3), t_e[et.edge_of].reshape(m, 3), 1.0 - t_e[et.edge_of].reshape(m, 3))

    out_F: list[np.ndarray] = []
    out_C: list[np.ndarray] = []

    def emit(fi: np.ndarray, corners: list[tuple[str, int]]) -> None:
        """corners: list of 3 items ('v', k) = corner k of the face, ('m', k) = split point of half-edge k."""
        idx = []
        uvs = []
        for kind, k in corners:
            if kind == "v":
                idx.append(F[fi, k])
                uvs.append(C[fi, k])
            else:
                idx.append(mid[fi, k])
                a = frac[fi, k][:, None]
                uvs.append((1.0 - a) * C[fi, k] + a * C[fi, (k + 1) % 3])
        out_F.append(np.stack(idx, axis=1))
        out_C.append(np.stack(uvs, axis=1))

    f0 = np.flatnonzero(nm == 0)
    out_F.append(F[f0])
    out_C.append(C[f0])
    f1 = np.flatnonzero(nm == 3)
    if len(f1):
        emit(f1, [("v", 0), ("m", 0), ("m", 2)])
        emit(f1, [("m", 0), ("v", 1), ("m", 1)])
        emit(f1, [("m", 2), ("m", 1), ("v", 2)])
        emit(f1, [("m", 0), ("m", 1), ("m", 2)])
    # exactly one marked edge k: split into two triangles through the opposite corner
    for k in range(3):
        fk = np.flatnonzero((nm == 1) & hm[:, k])
        if len(fk):
            k1, k2 = (k + 1) % 3, (k + 2) % 3
            emit(fk, [("v", k), ("m", k), ("v", k2)])
            emit(fk, [("m", k), ("v", k1), ("v", k2)])
    # exactly two marked edges: the unmarked edge is k; the corner between the marked edges (k2) is cut off by the
    # segment m_k1 - m_k2 (the isoline when splitting by a scalar field); the rest becomes two triangles
    for k in range(3):
        fk = np.flatnonzero((nm == 2) & ~hm[:, k])
        if len(fk):
            k1, k2 = (k + 1) % 3, (k + 2) % 3
            emit(fk, [("v", k), ("v", k1), ("m", k2)])
            emit(fk, [("v", k1), ("m", k1), ("m", k2)])
            emit(fk, [("m", k1), ("v", k2), ("m", k2)])
    is_new = np.zeros(len(newP), dtype=bool)
    is_new[len(P):] = True
    return Corners(newP, np.vstack(out_F), np.vstack(out_C)), is_new


# ------------------------------------------------------------------------------------------------ edge flips
def flip_delaunay_xy(
    mc: Corners,
    face_ok: np.ndarray | None = None,
    max_iters: int = 40,
    uv_tol: float = 1e-6,
    min_dot: float = 0.82,
    min_nz: float = 0.3,
) -> tuple[Corners, int]:
    """Flip interior edges to the Delaunay triangulation of the (x, y) projection (front view).

    Thin sliver triangles of a decimated scan become well shaped in the projection the relief is defined on. Only
    edges whose two faces are both front facing (3-D normal z >= ``min_nz``) and nearly coplanar (normals dot >=
    ``min_dot``, so a flip never connects across a fold), whose quad is convex in xy, that are not UV seams
    (the corner UVs of both faces agree at the edge) and, with ``face_ok``, whose faces are both selected, are flipped.
    Returns the new mesh and the number of flips."""
    F = mc.F.copy()
    C = mc.C.copy()
    P = mc.P
    xy = P[:, :2]
    total = 0
    for _ in range(max_iters):
        et = edge_table(F)
        m = len(F)
        t = xy[F]
        area = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
        n3 = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
        n3 = n3 / np.maximum(np.linalg.norm(n3, axis=1, keepdims=True), 1e-18)
        front = (area > 1e-12) & (n3[:, 2] >= min_nz)
        h1 = np.flatnonzero(et.he_pair >= 0)
        h2 = et.he_pair[h1]
        keep = h1 < h2
        h1, h2 = h1[keep], h2[keep]
        f1, k1 = h1 // 3, h1 % 3
        f2, k2 = h2 // 3, h2 % 3
        ok = front[f1] & front[f2] & (np.einsum("ij,ij->i", n3[f1], n3[f2]) >= min_dot)
        if face_ok is not None:
            ok &= face_ok[f1] & face_ok[f2]
        a = F[f1, k1]
        b = F[f1, (k1 + 1) % 3]
        c1 = F[f1, (k1 + 2) % 3]
        c2 = F[f2, (k2 + 2) % 3]
        # seam test: UV of a and b must agree in both faces
        ua1, ub1 = C[f1, k1], C[f1, (k1 + 1) % 3]
        ub2, ua2 = C[f2, k2], C[f2, (k2 + 1) % 3]
        ok &= (np.abs(ua1 - ua2).max(axis=1) < uv_tol) & (np.abs(ub1 - ub2).max(axis=1) < uv_tol)
        # in-circle: is c2 inside the circumcircle of (a, b, c1)? (a, b, c1 counter-clockwise)
        pa, pb, pc, pd = xy[a], xy[b], xy[c1], xy[c2]
        ax, ay = pa[:, 0] - pd[:, 0], pa[:, 1] - pd[:, 1]
        bx, by = pb[:, 0] - pd[:, 0], pb[:, 1] - pd[:, 1]
        cx, cy = pc[:, 0] - pd[:, 0], pc[:, 1] - pd[:, 1]
        det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) + (cx * cx + cy * cy) * (ax * by - bx * ay)
        # new triangles (c1, a, c2) and (c2, b, c1) must both be counter-clockwise
        def tri_area(p, q, r):
            return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (r[:, 0] - p[:, 0]) * (q[:, 1] - p[:, 1])

        convex = (tri_area(pc, pa, pd) > 1e-12) & (tri_area(pd, pb, pc) > 1e-12)
        cand = ok & convex & (det > 1e-14 * 0 + 1e-18)
        if not cand.any():
            break
        idx = np.flatnonzero(cand)
        # independent set: every face is used by at most one flip (largest in-circle violation wins)
        order = idx[np.argsort(-det[idx], kind="stable")]
        rank = np.arange(len(order))
        # a candidate wins when it is the best (lowest rank) candidate of both of its faces
        best = np.full(m, len(order), dtype=np.int64)
        np.minimum.at(best, f1[order], rank)
        np.minimum.at(best, f2[order], rank)
        win = (best[f1[order]] == rank) & (best[f2[order]] == rank)
        sel = order[win]
        if len(sel) == 0:
            break
        s1, s2 = f1[sel], f2[sel]
        A, B, C1, C2 = a[sel], b[sel], c1[sel], c2[sel]
        uvA, uvB = ua1[sel], ub1[sel]
        uvC1 = C[f1[sel], (k1[sel] + 2) % 3]
        uvC2 = C[f2[sel], (k2[sel] + 2) % 3]
        F[s1] = np.stack([C1, A, C2], axis=1)
        C[s1] = np.stack([uvC1, uvA, uvC2], axis=1)
        F[s2] = np.stack([C2, B, C1], axis=1)
        C[s2] = np.stack([uvC2, uvB, uvC1], axis=1)
        total += len(sel)
    return Corners(mc.P, F, C), total
