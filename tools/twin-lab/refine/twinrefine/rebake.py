"""Re-project the front photo (and an optional higher resolution face photo) onto the relief face.

The relief only moves vertices along the camera axis, so a vertex keeps its pixel in the front photo and the texture
stays registered. Still the old atlas was baked on the lumpy scan: surfaces that now face the camera differently
(nostril sides, lips, eyelids, cheeks) were weighted by the old incidence angle and sometimes filled. Here the texels
of the face region are sampled again with the texture stage's own camera (similarity + flow of the front view): colour
= the photo pixel at the texel's position, weight = visibility x incidence x a feather at the oval boundary.

With ``face.png`` (a head-and-shoulders portrait, any resolution) the face colours come from it instead: its landmarks
are mapped onto the front photo's landmarks by a thin-plate warp, per-channel gains match its colours to the front photo
over the face, and the front photo only fills where the portrait has no coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from twintex.bake import remap_points  # noqa: E402  (texture stage)
from twintex.colorspace import linear_to_u8, u8_to_linear  # noqa: E402
from twintex.raster import raster_pairs  # noqa: E402

from . import log
from .facedepth import FaceDepth, rasterize_depth, smoothstep01
from .frontview import FrontView
from .meshops import Corners
from .relief import sample_grid
from .scan import welded_vertex_normals


@dataclass
class PhotoSource:
    name: str
    lin: np.ndarray  # (H, W, 3) float32 linear light
    to_pixel: Callable[[np.ndarray], np.ndarray]  # world (n, 3) -> pixel (n, 2) in cv2 index convention
    reliability: Callable[[np.ndarray], np.ndarray] | None = None  # world (n, 3) -> weight in [0, 1]


def front_source(fv: FrontView) -> PhotoSource:
    return PhotoSource("front", fv.view.image_lin, fv.world_to_photo)


def portrait_source(
    rgb: np.ndarray, lm_xy: np.ndarray, L_world: np.ndarray, rms_px: float = 1.0
) -> PhotoSource:
    """A portrait registered to the world frame by its landmarks (468 pairs, thin-plate spline, smoothed)."""
    lam = 1e-1
    rbf = None
    for lam in (3.0, 1.0, 0.3, 0.1, 3e-2, 1e-2, 1e-3):  # smoothest warp within ``rms_px`` of the landmarks
        rbf = RBFInterpolator(L_world, lm_xy[:468], kernel="thin_plate_spline", smoothing=lam, degree=1)
        if np.sqrt(((rbf(L_world) - lm_xy[:468]) ** 2).sum(axis=1).mean()) <= rms_px:
            break
    H, W = rgb.shape[:2]

    def to_pixel(P: np.ndarray) -> np.ndarray:
        return rbf(P[:, :2])

    def reliability(P: np.ndarray) -> np.ndarray:
        s = rbf(P[:, :2])
        m = 6.0
        inside = (s[:, 0] >= m) & (s[:, 0] <= W - 1 - m) & (s[:, 1] >= m) & (s[:, 1] <= H - 1 - m)
        return inside.astype(np.float32)

    return PhotoSource("portrait", u8_to_linear(rgb), to_pixel, reliability)


def sample_source(src: PhotoSource, P: np.ndarray) -> np.ndarray:
    s = src.to_pixel(P)
    return np.clip(remap_points(src.lin, s[:, 0].astype(np.float32), s[:, 1].astype(np.float32), cv2.INTER_CUBIC), 0.0, None)


@dataclass
class RebakeReport:
    texels: int
    mean_weight: float
    gains: list[float] | None
    sources: list[str]


def face_texels(mc: Corners, face_sel: np.ndarray, size: int):
    """Atlas texels of the selected faces: rows, cols, world positions (n, 3), face index into ``mc.F`` (n,),
    barycentric weights (n, 3)."""
    uv = mc.C[face_sel].reshape(-1, 2).astype(np.float64) * size
    Fs = np.arange(len(face_sel) * 3).reshape(-1, 3)
    rows, cols, fid, lam = [], [], [], []
    for f, px, py, bary in raster_pairs(uv, Fs, size, size):
        rows.append(py)
        cols.append(px)
        fid.append(face_sel[f])
        lam.append(bary)
    if not rows:
        return None
    rows, cols, fid, lam = (np.concatenate(a) for a in (rows, cols, fid, lam))
    P = np.einsum("ij,ijk->ik", lam, mc.P[mc.F[fid]])
    return rows, cols, P, fid, lam


def rebake_face(
    mc: Corners,
    atlas: np.ndarray,
    fd: FaceDepth,
    sources: list[PhotoSource],
    feather_mm: float = 12.0,
    outside_mm: float = 8.0,
    cos_lo: float = 0.0,
    cos_hi: float = 0.30,
    vis_tol_mm: float = 2.5,
) -> tuple[np.ndarray, RebakeReport]:
    """Update ``atlas`` (uint8, modified copy returned) on the face region from ``sources`` (first = preferred)."""
    S = atlas.shape[0]
    grid = fd.grid
    sd_face = sample_grid(fd.sd, grid, mc.P[:, 0], mc.P[:, 1], fill=-1e3)
    in_band = (sd_face > -outside_mm - 2.0)[mc.F].any(axis=1)
    # faces facing the camera
    t = mc.P[mc.F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)
    front = (n[:, 2] > 0.02) & (t[:, :, 2].mean(axis=1) > 0.0)
    face_sel = np.flatnonzero(in_band & front)
    out = atlas.copy()
    res = face_texels(mc, face_sel, S)
    if res is None:
        return out, RebakeReport(0, 0.0, None, [])
    rows, cols, P, fid, lam = res
    vn = welded_vertex_normals(mc.P, mc.F, np.arange(len(mc.P)))
    N = np.einsum("ij,ijk->ik", lam, vn[mc.F[fid]])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    # visibility against the refined surface seen from the front
    tri = mc.F
    zfront = rasterize_depth(mc.P[:, :2], mc.P[:, 2], tri, grid)
    zs = sample_grid(zfront, grid, P[:, 0], P[:, 1], fill=np.nan)
    slope_tol = 0.5 * grid.res * 1000.0 * np.sqrt(np.maximum(1.0 / np.maximum(N[:, 2], 0.2) ** 2 - 1.0, 0.0))
    visible = np.isfinite(zs) & ((zs - P[:, 2]) * 1000.0 <= vis_tol_mm + slope_tol)
    cosw = smoothstep01((N[:, 2] - cos_lo) / (cos_hi - cos_lo))
    sd = sample_grid(fd.sd, grid, P[:, 0], P[:, 1], fill=-1e3)
    feather = smoothstep01((sd + outside_mm) / (feather_mm + outside_mm))
    base_rel = (visible * cosw).astype(np.float32)  # reliability of a photo at the texel, before the boundary feather

    old_lin = u8_to_linear(out[rows, cols])
    used: list[str] = []
    gains = None
    colour: list[tuple[np.ndarray, np.ndarray]] = []
    for src in sources:
        c = sample_source(src, P)
        rel = base_rel * (src.reliability(P) if src.reliability is not None else 1.0)
        colour.append((c, rel))
    if len(colour) > 1:
        # per-channel gains that bring every other source (the portrait) to the colours of the front photo (the last
        # source, the reference the rest of the atlas was baked from) over the inner face, in linear light
        ref_c, ref_w = colour[-1]
        for i in range(len(colour) - 1):
            c, w = colour[i]
            inner = (w > 0.9) & (ref_w > 0.9) & (sd > 12.0)
            if inner.sum() > 500:
                g = np.clip(ref_c[inner].mean(axis=0) / np.maximum(c[inner].mean(axis=0), 1e-6), 0.6, 1.6).astype(np.float32)
                colour[i] = (c * g[None, :], w)
                if gains is None:
                    gains = g
                log(f"rebake: {sources[i].name} colour gains {np.round(g, 3).tolist()}")
    # preferred source first; later sources only fill what the earlier ones do not cover
    acc = np.zeros_like(old_lin)
    cover = np.zeros(len(rows), np.float32)
    for (c, rel), src in zip(colour, sources, strict=True):
        take = np.clip(rel, 0.0, 1.0) * (1.0 - cover)
        acc += take[:, None] * c
        cover += take
        used.append(src.name)
    photo = acc / np.maximum(cover, 1e-6)[:, None]
    a = (np.clip(cover, 0.0, 1.0) * feather).astype(np.float32)  # one feather towards the boundary for all sources
    final = np.where(cover[:, None] > 1e-6, a[:, None] * photo + (1.0 - a)[:, None] * old_lin, old_lin)
    out[rows, cols] = linear_to_u8(final)
    return out, RebakeReport(int(len(rows)), float(a.mean()), None if gains is None else gains.tolist(), used)
