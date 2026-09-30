"""Light-touch de-lighting: remove only a smooth directional illumination gradient from a studio view."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_illumination_field(img_lin: np.ndarray, alpha: np.ndarray, max_log: float = 0.35) -> np.ndarray:
    """Smooth log-illumination field (H, W) with zero median over the foreground.

    A quadratic field ``f = a x + b y + c xy + d x^2 + e y^2`` (x, y in [-1, 1]) is fitted to the *derivatives* of
    the log-luminance on flat (low-gradient, interior) foreground pixels. Using derivatives on flat areas means
    garment / skin contrast edges do not enter the fit, so only large-scale shading is captured.
    """
    H, W = alpha.shape
    step = max(1, min(H, W) // 300)
    small = img_lin[::step, ::step]
    a = cv2.erode((alpha[::step, ::step] > 0.7).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    h, w = a.shape
    if a.sum() < 500:
        return np.zeros((H, W), np.float32)
    lum = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ll = cv2.GaussianBlur(np.log(np.maximum(lum, 1e-4)).astype(np.float32), (0, 0), 1.5)
    gy, gx = np.gradient(ll)
    gx *= (w - 1) / 2.0  # derivative w.r.t. the normalised coordinate
    gy *= (h - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    xn = xs / max(w - 1, 1) * 2 - 1
    yn = ys / max(h - 1, 1) * 2 - 1
    mag = np.hypot(gx, gy)
    interior = a & cv2.erode(a.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    thr = np.percentile(mag[interior], 60)
    sel = interior & (mag <= thr)
    x, y, dgx, dgy = xn[sel], yn[sel], gx[sel], gy[sel]
    # unknowns: a b c d e ;  df/dx = a + c y + 2 d x ;  df/dy = b + c x + 2 e y
    zero, one = np.zeros_like(x), np.ones_like(x)
    A = np.concatenate([np.stack([one, zero, y, 2 * x, zero], 1), np.stack([zero, one, x, zero, 2 * y], 1)])
    rhs = np.concatenate([dgx, dgy])
    keep = np.ones(len(rhs), bool)
    coef = np.zeros(5)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[keep], rhs[keep], rcond=None)
        r = rhs - A @ coef
        keep = np.abs(r) < 2.5 * (np.median(np.abs(r)) * 1.4826 + 1e-6)
    gyy, gxx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = gxx / max(W - 1, 1) * 2 - 1
    Y = gyy / max(H - 1, 1) * 2 - 1
    field = coef[0] * X + coef[1] * Y + coef[2] * X * Y + coef[3] * X * X + coef[4] * Y * Y
    field -= np.median(field[alpha > 0.7])
    return np.clip(field, -max_log, max_log).astype(np.float32)


def remove_shading(img_lin: np.ndarray, alpha: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Divide the image by the estimated smooth illumination field (``strength`` in [0, 1])."""
    if strength <= 0:
        return img_lin
    field = estimate_illumination_field(img_lin, alpha)
    return (img_lin * np.exp(-strength * field)[..., None]).astype(np.float32)
