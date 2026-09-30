"""Silhouette-based alignment of an orthographic view image to the mesh.

Stage 1: similarity fit (uniform scale + translation, optional aspect) maximising silhouette IoU.
Stage 2: smooth boundary-driven 2-D warp field ("flow") so that the image silhouette matches the mesh silhouette
         (compensates the small proportion differences between an AI generated image and the extracted mesh).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from .camera import OrthoCamera
from .raster import raster_pairs


def fill_silhouette(xy_px: np.ndarray, faces: np.ndarray, width: int, height: int) -> np.ndarray:
    """Binary silhouette (float32 0/1) of a projected mesh, rasterised with the vectorised triangle filler."""
    img = np.zeros((height, width), np.uint8)
    for _, px, py, _ in raster_pairs(xy_px, faces, width, height, eps=1e-3):
        img[py, px] = 1
    return img.astype(np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    return inter / union if union > 0 else 0.0


@dataclass
class AlignResult:
    cam: OrthoCamera
    iou_initial: float
    iou_similarity: float
    iou_final: float
    flow: np.ndarray | None  # (H, W, 2) float32, added to projected pixel coordinates


def _canvas(uv_metric: np.ndarray, faces: np.ndarray, target_px: int = 700, pad: int = 40):
    lo, hi = uv_metric.min(0), uv_metric.max(0)
    s_c = target_px / max(hi[1] - lo[1], 1e-9)
    w = int(np.ceil((hi[0] - lo[0]) * s_c)) + 2 * pad
    h = int(np.ceil((hi[1] - lo[1]) * s_c)) + 2 * pad
    xy = np.stack([(uv_metric[:, 0] - lo[0]) * s_c + pad, (hi[1] - uv_metric[:, 1]) * s_c + pad], axis=1)
    sil = fill_silhouette(xy, faces, w, h)
    sil = cv2.GaussianBlur(sil, (0, 0), 1.2)
    return sil, s_c, lo, hi, pad


def _simplex(p0: np.ndarray, steps: list[float]) -> np.ndarray:
    n = p0.size
    sx = np.tile(p0, (n + 1, 1))
    for i in range(n):
        sx[i + 1, i] += steps[i]
    return sx


def fit_similarity(
    cam0: OrthoCamera,
    verts: np.ndarray,
    faces: np.ndarray,
    alpha: np.ndarray,
    analysis_height: int = 520,
    allow_aspect: bool = False,
    init: tuple[float, float, float] | None = None,
) -> tuple[OrthoCamera, float, float]:
    """Fit scale/translation of ``cam0`` so the mesh silhouette matches ``alpha``. Returns (cam, iou0, iou).

    ``init = (px_per_m, tx, ty)`` (full-resolution pixels) seeds the search instead of the silhouette bbox.
    """
    H, W = alpha.shape
    r = analysis_height / H
    Ha, Wa = analysis_height, int(round(W * r))
    A = cv2.resize(alpha, (Wa, Ha), interpolation=cv2.INTER_AREA)
    A = (A > 0.5).astype(np.float32)
    A = cv2.GaussianBlur(A, (0, 0), 1.2)
    uvm = cam0.metric_uv(verts)
    sil, s_c, lo, hi, pad = _canvas(uvm, faces)

    ys, xs = np.nonzero(A > 0.5)
    top, bottom = ys.min(), ys.max() + 1
    cx_img = xs.mean()
    ext_v = hi[1] - lo[1]
    s0 = (bottom - top) / ext_v
    cy_c, cx_c = ndi.center_of_mass(sil > 0.5)
    u_cen = (cx_c - pad) / s_c + lo[0]
    tx0 = cx_img - s0 * u_cen
    ty0 = bottom + s0 * lo[1]
    if init is not None:
        s0, tx0, ty0 = init[0] * r, init[1] * r, init[2] * r

    def matrix(params: np.ndarray) -> np.ndarray:
        s = s0 * np.exp(params[0])
        asp = np.exp(params[3]) if allow_aspect else 1.0
        tx = tx0 + params[1] * Ha
        ty = ty0 + params[2] * Ha
        kx = s * asp / s_c
        ky = s / s_c
        return np.array(
            [[kx, 0, tx + s * asp * lo[0] - kx * pad], [0, ky, ty - s * hi[1] - ky * pad]], dtype=np.float64
        )

    def cost(p: np.ndarray) -> float:
        warped = cv2.warpAffine(sil, matrix(p), (Wa, Ha), flags=cv2.INTER_LINEAR, borderValue=0)
        return 1.0 - iou(warped, A)

    n = 4 if allow_aspect else 3
    p0 = np.zeros(n)
    c0 = cost(p0)
    best = minimize(
        cost, p0, method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400, "initial_simplex": _simplex(p0, [0.03] * n)},
    )
    best = minimize(
        cost, best.x, method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 300, "initial_simplex": _simplex(best.x, [0.006] * n)},
    )
    p = best.x
    s = s0 * np.exp(p[0])
    cam = cam0.copy()
    cam.scale = float(s / r)
    cam.aspect = float(np.exp(p[3])) if allow_aspect else 1.0
    cam.tx = float((tx0 + p[1] * Ha) / r)
    cam.ty = float((ty0 + p[2] * Ha) / r)
    cam.width, cam.height = W, H
    return cam, 1.0 - c0, 1.0 - float(best.fun)


def _edges(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    return (m > 0) & (er == 0)


def refine_flow(
    cam: OrthoCamera,
    verts: np.ndarray,
    faces: np.ndarray,
    alpha: np.ndarray,
    sigmas: tuple[float, ...] = (0.035, 0.022, 0.014),
    max_shift_frac: float = 0.035,
) -> tuple[np.ndarray, float]:
    """Estimate a smooth flow ``f(q)`` such that ``alpha(q + f(q))`` matches the mesh silhouette M(q)."""
    H, W = alpha.shape
    xy = cam.project(verts)[:, :2]
    M = fill_silhouette(xy, faces, W, H) > 0.5
    edge_m = _edges(M)
    ey, ex = np.nonzero(edge_m)
    flow = np.zeros((H, W, 2), np.float32)
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    max_shift = max_shift_frac * H
    for sig in sigmas:
        warped = cv2.remap(alpha, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderValue=0)
        edge_a = _edges(warped > 0.5)
        idx = ndi.distance_transform_edt(~edge_a, return_distances=False, return_indices=True)
        ny = idx[0][ey, ex]
        nx = idx[1][ey, ex]
        d = np.stack([nx - ex, ny - ey], axis=1).astype(np.float32)
        ok = np.linalg.norm(d, axis=1) <= max_shift
        acc = np.zeros((H, W, 3), np.float32)
        acc[ey[ok], ex[ok], 0] = d[ok, 0]
        acc[ey[ok], ex[ok], 1] = d[ok, 1]
        acc[ey[ok], ex[ok], 2] = 1.0
        blurred = cv2.GaussianBlur(acc, (0, 0), sig * H)
        wgt = blurred[..., 2:3]
        delta = blurred[..., :2] / np.maximum(wgt, 1e-6)
        conf = np.clip(wgt / (wgt.max() * 0.05 + 1e-9), 0, 1)
        flow = np.clip(flow + (delta * conf).astype(np.float32), -max_shift, max_shift)
    warped = cv2.remap(alpha, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderValue=0)
    return flow, iou((warped > 0.5).astype(np.float32), M.astype(np.float32))
