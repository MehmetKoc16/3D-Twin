"""Hole filling: UV-space push-pull, gutter dilation and 3-D nearest-visible / mirror / diffusion fill."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------------------------- UV space
def push_pull(img: np.ndarray, conf: np.ndarray, min_size: int = 1) -> np.ndarray:
    """Multi-resolution ("push-pull") fill of an image with per-pixel confidence in [0, 1].

    ``out = conf * img + (1 - conf) * coarser_estimate`` recursively; pixels with conf == 1 are returned
    unchanged, pixels with conf == 0 are interpolated from the coarser levels. ``img`` (H, W, C) float32.
    """
    img = np.asarray(img, dtype=np.float32)
    conf = np.clip(np.asarray(conf, dtype=np.float32), 0.0, 1.0)
    num = img * conf[..., None]
    levels: list[tuple[np.ndarray, np.ndarray]] = [(num, conf)]
    while min(levels[-1][1].shape) > min_size:
        n, w = levels[-1]
        h2, w2 = max(w.shape[0] // 2, 1), max(w.shape[1] // 2, 1)
        levels.append((cv2.resize(n, (w2, h2), interpolation=cv2.INTER_AREA).reshape(h2, w2, -1),
                       cv2.resize(w, (w2, h2), interpolation=cv2.INTER_AREA)))
        if h2 == 1 and w2 == 1:
            break
    n, w = levels[-1]
    result = n / np.maximum(w, 1e-8)[..., None]
    for n, w in reversed(levels[:-1]):
        up = cv2.resize(result, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(*w.shape, -1)
        obs = n / np.maximum(w, 1e-8)[..., None]
        a = np.clip(w * 1.0, 0.0, 1.0)[..., None]
        result = a * obs + (1 - a) * up
    return result.astype(np.float32)


def dilate_fill(img: np.ndarray, mask: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Grow valid pixels outwards by ``iterations`` px (gutter / padding fill). Returns (img, new_mask)."""
    img = img.astype(np.float32, copy=True)
    m = mask.astype(np.float32)
    k = (3, 3)
    for _ in range(iterations):
        num = cv2.blur(img * m[..., None], k)
        den = cv2.blur(m, k)
        grow = (m < 0.5) & (den > 1e-6)
        if not grow.any():
            break
        img[grow] = num[grow] / den[grow][:, None]
        m[grow] = 1.0
    return img, m > 0.5


# --------------------------------------------------------------------------------------------- 3-D fill
def mirror_fill(
    pos: np.ndarray,
    nrm: np.ndarray,
    col: np.ndarray,
    conf: np.ndarray,
    x_mid: float,
    known_thr: float = 0.5,
    unknown_thr: float = 0.15,
    max_dist: float = 0.02,
    min_normal_dot: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy colours across the sagittal plane ``x = x_mid`` into texels no view has seen.

    Returns ``(conf_out, col_out)`` where mirrored texels get confidence ``0.9 * source confidence``.
    """
    known = conf >= known_thr
    unknown = conf < unknown_thr
    conf2, col2 = conf.copy(), col.copy()
    if known.sum() < 8 or not unknown.any():
        return conf2, col2
    tree = cKDTree(pos[known])
    q = pos[unknown].copy()
    q[:, 0] = 2 * x_mid - q[:, 0]
    d, j = tree.query(q, k=1)
    nm = nrm[unknown].copy()
    nm[:, 0] *= -1
    kn = nrm[known][j]
    ok = (d <= max_dist) & (np.einsum("ij,ij->i", nm, kn) >= min_normal_dot)
    idx = np.flatnonzero(unknown)[ok]
    conf2[idx] = 0.9 * conf[known][j[ok]]
    col2[idx] = col[known][j[ok]]
    return conf2, col2


def nearest_visible_fill(
    pos: np.ndarray,
    nrm: np.ndarray,
    col: np.ndarray,
    conf: np.ndarray,
    known_thr: float = 0.6,
    k: int = 24,
    d0: float = 0.01,
    normal_floor: float = 0.1,
    smooth_iters: int = 8,
    smooth_k: int = 10,
    aniso: tuple[float, float, float] = (1.0, 2.0, 1.0),
) -> np.ndarray:
    """Colour estimate for every point from the confident ("visible") points, respecting 3-D proximity.

    IDW over the k nearest known points (weights ``conf / (d^2 + d0^2) * (floor + max(0, n.n'))^2``), followed by a
    few Jacobi diffusion iterations on the kNN graph with the known points held fixed. Distances are measured with
    ``aniso`` axis scales (default y x2): colour then propagates *around* the body (garment bands, sleeves)
    rather than along it. Returns (M, 3).
    """
    pos = pos * np.asarray(aniso, dtype=np.float32)
    known = conf >= known_thr
    n_known = int(known.sum())
    if n_known == 0:
        return np.tile(col.mean(0, keepdims=True) if len(col) else np.zeros((1, 3), np.float32), (len(col), 1)).astype(np.float32)
    kp, kn, kc, kw = pos[known], nrm[known], col[known], conf[known]
    kk = min(k, n_known)
    tree = cKDTree(kp)
    d, j = tree.query(pos, k=kk)
    d = d.reshape(len(pos), kk)
    j = j.reshape(len(pos), kk)
    aff = np.clip(np.einsum("ij,ikj->ik", nrm, kn[j]), 0.0, 1.0) + normal_floor
    w = kw[j] * aff**2 / (d * d + d0 * d0)
    w[d < 1e-9] = 0.0  # a known point does not vote for itself
    ws = w.sum(1, keepdims=True)
    fill = np.einsum("ik,ikc->ic", w, kc[j]) / np.maximum(ws, 1e-20)
    fill = np.where(ws > 0, fill, kc.mean(0, keepdims=True)).astype(np.float32)
    # known points keep their own colour as the fixed boundary of the diffusion
    cur = np.where(known[:, None], col, fill).astype(np.float32)
    free = ~known
    if smooth_iters > 0 and free.any():
        tree_all = cKDTree(pos)
        _, nb = tree_all.query(pos[free], k=min(smooth_k, len(pos)))
        for _ in range(smooth_iters):
            cur_free = cur[nb].mean(axis=1)
            cur[free] = cur_free
    out = np.where(known[:, None], fill, cur)
    # known points: use the IDW estimate of their neighbours as "fill" (used only when conf < 1)
    return out.astype(np.float32)
