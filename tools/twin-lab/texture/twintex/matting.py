"""Background removal for evenly lit studio views (no ML weights: colour model + GrabCut)."""

from __future__ import annotations

import cv2
import numpy as np


def _poly_terms(h: int, w: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    x = (xs / max(w - 1, 1)).ravel() * 2 - 1
    y = (ys / max(h - 1, 1)).ravel() * 2 - 1
    A = np.stack([np.ones_like(x), x, y, x * y, x * x, y * y, x * x * y, x * y * y, x**3, y**3], axis=1)
    return A, xs.ravel(), ys.ravel()


def fit_background(lab: np.ndarray, border_frac: float = 0.05) -> np.ndarray:
    """Fit a smooth cubic surface per Lab channel to the image border band (robust, iterated)."""
    h, w = lab.shape[:2]
    step = 4
    A, xs, ys = _poly_terms(h, w, step)
    b = max(int(border_frac * min(h, w)), 4)
    band = (xs < b) | (xs >= w - b) | (ys < b) | (ys >= h - b)
    # the feet may touch the bottom band: prefer top and side bands for the first fit
    band_top = band & (ys < h * 0.9)
    samples = lab[ys, xs].astype(np.float64)
    coef = np.zeros((A.shape[1], 3))
    sel = band_top.copy()
    for _ in range(4):
        for c in range(3):
            coef[:, c], *_ = np.linalg.lstsq(A[sel], samples[sel, c], rcond=None)
        res = np.linalg.norm(A @ coef - samples, axis=1)
        thr = max(3.0, 2.5 * np.median(res[band_top]))
        sel = band_top & (res < thr)
    A_full, xs2, ys2 = _poly_terms(h, w, 1)
    out = (A_full @ coef).reshape(h, w, 3).astype(np.float32)
    return out


def matte(img_u8: np.ndarray, grabcut_iters: int = 5) -> np.ndarray:
    """Return a float32 alpha (H, W) in [0, 1] for the largest foreground object of a studio image."""
    h, w = img_u8.shape[:2]
    # work at reduced resolution for speed, upsample the mask afterwards
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(img_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img_u8
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = fit_background(lab)
    de = np.linalg.norm(lab - bg, axis=2)
    de = cv2.GaussianBlur(de, (0, 0), 1.0)
    thr_hi = max(14.0, float(np.percentile(de, 97)) * 0.35)
    fg_sure = (de > thr_hi).astype(np.uint8)
    fg_sure = cv2.morphologyEx(fg_sure, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg_sure, connectivity=8)
    if n <= 1:
        raise RuntimeError("matting failed: no foreground found")
    # keep components that are big (person + separate parts like hands) but drop specks
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas > 0.02 * areas.max()) + 1
    fg_sure = np.isin(lbl, keep).astype(np.uint8)
    thr_lo = max(5.0, thr_hi * 0.4)
    prob_fg = (de > thr_lo).astype(np.uint8)
    prob_fg = cv2.dilate(prob_fg, np.ones((5, 5), np.uint8))
    mask = np.full(de.shape, cv2.GC_BGD, np.uint8)
    mask[prob_fg > 0] = cv2.GC_PR_BGD
    mask[(de > thr_lo)] = cv2.GC_PR_FGD
    er = cv2.erode(fg_sure, np.ones((7, 7), np.uint8))
    mask[er > 0] = cv2.GC_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, grabcut_iters, cv2.GC_INIT_WITH_MASK)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        fg = (de > thr_lo).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas > 0.02 * areas.max()) + 1
    fg = np.isin(lbl, keep).astype(np.uint8)
    # fill pin-holes
    inv = (1 - fg).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.002 * fg.size:
            fg[lbl == i] = 1
    alpha = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 0.8)
    if scale < 1:
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0, 1).astype(np.float32)
