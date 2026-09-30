"""Colour priors for surfaces no view can see.

* ``opposite_prior``: when a view has no opposite partner (front without back, left without right, ...), its
  low-pass filtered image is "shone through" the body onto the surfaces that face away from it. Region colours
  (skin / shirt / trousers) end up in the right places; all detail is removed, so no ghost of the front (pockets,
  face) appears on the back.
* ``hair_prior``: the back / crown of the head is painted with the colour of the top of the head (hair), instead
  of the skin colour that a plain nearest-neighbour fill would drag around from the face.

Both are only fallbacks: they apply to the *fill* colour, which is blended with real observations by confidence.
"""

from __future__ import annotations

import numpy as np

from .bake import BakeConfig, remap_points, smoothstep
from .views import View

OPPOSITE = {"front": "back", "back": "front", "left": "right", "right": "left"}


def opposite_prior(
    views: list[View], P: np.ndarray, N: np.ndarray, cfg: BakeConfig, body_h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (colour_lin (n, 3), weight (n,)) of the "shine-through" prior."""
    n = len(P)
    acc = np.zeros((n, 3), np.float32)
    wsum = np.zeros(n, np.float32)
    have = {v.name for v in views}
    for v in views:
        if OPPOSITE.get(v.name) in have:
            continue
        cam = v.cam
        W, H = v.size
        cos_t = N @ cam.to_camera
        away = smoothstep(-cos_t, 0.05, 0.45)
        idx = np.flatnonzero(away > 0)
        if idx.size == 0:
            continue
        p = cam.project(P[idx])
        x, y, depth = p[:, 0], p[:, 1], p[:, 2]
        # the texel must be the back side of the surface the photo shows (exactly one surface in front of it);
        # 0 or >= 2 surfaces in front means noise or another body part (arm, hand) hides the "opposite" pixel
        n_front = v.surfaces_in_front(x, y, depth, tol=0.004 * body_h)
        same_part = (n_front == 1).astype(np.float32)
        qx = (x - 0.5).astype(np.float32)
        qy = (y - 0.5).astype(np.float32)
        if v.flow is not None:
            f = remap_points(v.flow, qx, qy)
            qx, qy = qx + f[:, 0], qy + f[:, 1]
        a = remap_points(v.alpha, qx, qy)
        col = remap_points(v.lowpass(), qx, qy)
        w = (away[idx] * same_part * smoothstep(a, 0.3, 0.9)).astype(np.float32)
        acc[idx] += w[:, None] * col
        wsum[idx] += w
    col = acc / np.maximum(wsum, 1e-9)[:, None]
    return col.astype(np.float32), np.clip(wsum, 0.0, 1.0)


def hair_prior(
    pos: np.ndarray,
    nrm: np.ndarray,
    col: np.ndarray,
    conf: np.ndarray,
    ymin: float,
    ymax: float,
    head_frac: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (hair colour (3,), per-point weight (n,)) or None when the crown was not observed."""
    h_head = head_frac * (ymax - ymin)
    top = (pos[:, 1] > ymax - 0.25 * h_head) & (conf > 0.25)
    if top.sum() < 20:
        return None
    hair = np.median(col[top], axis=0).astype(np.float32)
    w_y = smoothstep(pos[:, 1], ymax - 0.90 * h_head, ymax - 0.55 * h_head)
    away = 1.0 - smoothstep(nrm[:, 2], 0.0, 0.45)
    return hair, (w_y * away).astype(np.float32)
