"""Apply the face displacement field to the scan: local refinement, then z displacement of the front-facing surface."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .facedepth import FaceDepth, Grid, smoothstep01
from .meshops import Corners, edge_endpoints, edge_table, flip_delaunay_xy, refine_marked_edges
from .scan import welded_vertex_normals


def sample_grid(img: np.ndarray, grid: Grid, x: np.ndarray, y: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Bilinear lookup of a grid image at world (x, y); NaN pixels contaminate their neighbours (kept as NaN)."""
    px, py = grid.to_px(x, y)
    mx = (px - 0.5).astype(np.float32)
    my = (py - 0.5).astype(np.float32)
    cols = 2048
    n = len(mx)
    rows = -(-n // cols)
    a = np.full(rows * cols, -1.0, np.float32)
    b = np.full(rows * cols, -1.0, np.float32)
    a[:n], b[:n] = mx, my
    src = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
    ok = np.isfinite(img).astype(np.float32)
    v = cv2.remap(src, a.reshape(rows, cols), b.reshape(rows, cols), cv2.INTER_LINEAR, borderValue=0).reshape(-1)[:n]
    o = cv2.remap(ok, a.reshape(rows, cols), b.reshape(rows, cols), cv2.INTER_LINEAR, borderValue=0).reshape(-1)[:n]
    out = np.where(o > 0.999, v / np.maximum(o, 1e-6), np.nan)
    return np.where(np.isfinite(out), out, fill) if not np.isnan(fill) else out


@dataclass
class ReliefReport:
    faces_before: int
    faces_after: int
    moved_vertices: int
    max_move_mm: float
    mean_move_mm: float


def vertex_fields(P: np.ndarray, fd: FaceDepth, gate_lo_mm: float = 4.0, gate_hi_mm: float = 12.0):
    """(delta, gate*weight-ish) per vertex: the z displacement and the front-surface gate."""
    g = fd.grid
    d = sample_grid(fd.delta, g, P[:, 0], P[:, 1], fill=0.0)
    zs = sample_grid(fd.zscan, g, P[:, 0], P[:, 1], fill=np.nan)
    behind = (zs - P[:, 2]) * 1000.0  # mm behind the nearest surface seen from the front
    gate = 1.0 - smoothstep01((behind - gate_lo_mm) / (gate_hi_mm - gate_lo_mm))
    gate = np.where(np.isfinite(behind), gate, 0.0)
    return d, gate, sample_grid(fd.weight, g, P[:, 0], P[:, 1], fill=0.0)


def refine_face_region(mc: Corners, fd: FaceDepth, edge_mm: float = 2.2, rounds: int = 7) -> Corners:
    """Split the edges of the front-facing faces under the face mask until they are shorter than ``edge_mm``."""
    out = mc
    for _ in range(rounds):
        out, _nflip = flip_delaunay_xy(out, _face_region(out, fd), max_iters=15)
        et = edge_table(out.F)
        d, gate, w = vertex_fields(out.P, fd)
        active_v = (w > 1e-3) & (gate > 0.5)
        lo, hi = edge_endpoints(et)
        length = np.linalg.norm(out.P[hi] - out.P[lo], axis=1)
        mark = active_v[lo] & active_v[hi] & (length > edge_mm * 1e-3)
        if not mark.any():
            break
        out, _ = refine_marked_edges(out, mark, et)
    out, _nflip = flip_delaunay_xy(out, _face_region(out, fd), max_iters=40)
    return out


def _face_region(mc: Corners, fd: FaceDepth) -> np.ndarray:
    """Faces under the face mask whose vertices all lie on the front surface."""
    d, gate, w = vertex_fields(mc.P, fd)
    v_ok = (w > 1e-3) & (gate > 0.5)
    return v_ok[mc.F].all(axis=1)


def vertex_graph(mc: Corners):
    """Sparse row-normalised adjacency of the welded mesh (for diffusing per-vertex fields)."""
    from scipy.sparse import coo_matrix

    et = edge_table(mc.F)
    lo, hi = edge_endpoints(et)
    n = len(mc.P)
    A = coo_matrix((np.ones(2 * len(lo)), (np.r_[lo, hi], np.r_[hi, lo])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
    return A, deg


def apply_relief(
    mc: Corners, fd: FaceDepth, diffuse_iters: int = 8, smooth_iters: int = 2, snap_lo_mm: float = 3.0, snap_hi_mm: float = 9.0,
    nz_lo: float = -1.0, nz_hi: float = -0.5,
) -> tuple[Corners, ReliefReport]:
    """z displacement of the front-facing surface under the face mask.

    * A per-vertex gate (is the vertex on the surface the front camera sees?) selects what moves by its field value;
      where the gate switches (folds: under the chin, ear, hairline) the displacement is diffused over the mesh.
    * Vertices that lie within a few millimetres of the front surface under the face mask are snapped onto the NEW
      surface, which removes the scan's own spikes and dents there instead of carrying them along.
    * A light Laplacian smoothing of z finishes the face region."""
    P0 = mc.P
    d, gate, w = vertex_fields(P0, fd)
    # surfaces turning away from the camera (cheek sides, jaw, hairline folds) are left to the scan
    nz = welded_vertex_normals(P0, mc.F, np.arange(len(P0)))[:, 2]
    face_on = smoothstep01((nz - nz_lo) / (nz_hi - nz_lo))
    gate = gate * face_on
    w = w * face_on
    A, deg = vertex_graph(mc)
    d0 = d.copy()
    dz = d * gate
    for _ in range(diffuse_iters):
        avg = (A @ dz) / deg
        dz = gate * d0 + (1.0 - gate) * avg
    z = P0[:, 2] + dz
    # snap near-surface vertices onto the new front surface Zscan + delta
    g = fd.grid
    zs = sample_grid(fd.zscan, g, P0[:, 0], P0[:, 1], fill=np.nan)
    front_new = zs + d
    behind = (zs - P0[:, 2]) * 1000.0
    kappa = np.where(np.isfinite(behind), 1.0 - smoothstep01((behind - snap_lo_mm) / (snap_hi_mm - snap_lo_mm)), 0.0) * w
    z = np.where(np.isfinite(front_new), (1.0 - kappa) * z + kappa * front_new, z)
    region = (w > 1e-3) | (np.abs(z - P0[:, 2]) > 1e-5)
    for _ in range(smooth_iters):
        avg = (A @ z) / deg
        z = np.where(region, 0.5 * z + 0.5 * avg, z)
    P = P0.copy()
    P[:, 2] = z
    dzf = z - P0[:, 2]
    mv = np.abs(dzf) > 1e-5
    rep = ReliefReport(0, len(mc.F), int(mv.sum()), float(np.abs(dzf).max() * 1000), float(np.abs(dzf[mv]).mean() * 1000) if mv.any() else 0.0)
    return Corners(P, mc.F, mc.C), rep


def smooth_band(mc: Corners, fd: FaceDepth, iters: int = 4, alpha: float = 0.5, out_mm: float = 22.0, in_mm: float = 14.0) -> Corners:
    """Light Laplacian smoothing (x, y, z) of the scan around the face-oval boundary (jaw line, hairline, ears' front):
    the scan's own jaggies there would otherwise stand out next to the clean face."""
    sd = sample_grid(fd.sd, fd.grid, mc.P[:, 0], mc.P[:, 1], fill=-1e3) if fd.sd is not None else np.full(len(mc.P), -1e3)
    band = smoothstep01((sd + out_mm) / 10.0) * (1.0 - smoothstep01((sd - 4.0) / max(in_mm - 4.0, 1e-3)))
    A, deg = vertex_graph(mc)
    P = mc.P.copy()
    for _ in range(iters):
        avg = (A @ P) / deg[:, None]
        P = P + (alpha * band)[:, None] * (avg - P)
    return Corners(P, mc.F, mc.C)


def face_nz(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    t = P[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-18)


def heal_folds(
    before: Corners, after: Corners, fd: FaceDepth, iters: int = 6, alpha: float = 0.7, min_nz: float = 0.0, w_min: float = 0.5
) -> tuple[Corners, int]:
    """Faces that were front facing before the relief but are folded over (or edge-on) after it, inside the face mask,
    are the thin scratches seen under raking light: relax their vertices (and one ring around) toward their neighbours."""
    P = after.P.copy()
    A, deg = vertex_graph(after)
    nz0 = face_nz(before.P, before.F)
    cen = after.P[after.F].mean(axis=1)
    w = sample_grid(fd.weight, fd.grid, cen[:, 0], cen[:, 1], fill=0.0)
    total = 0
    for _ in range(iters):
        nz1 = face_nz(P, after.F)
        bad = (nz1 < min_nz) & (w > w_min) & ((nz0 > 0.25) | (nz1 < -0.02))
        if not bad.any():
            break
        total = max(total, int(bad.sum()))
        v = np.zeros(len(P), dtype=bool)
        v[after.F[bad].reshape(-1)] = True
        avg = (A @ P) / deg[:, None]
        P = np.where(v[:, None], (1 - alpha) * P + alpha * avg, P)
    return Corners(P, after.F, after.C), total
