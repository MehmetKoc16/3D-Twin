"""Multi-view projection baking: UV-space rasterisation, per-view visibility / weight / colour, blending."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .colorspace import linear_to_u8
from .raster import rasterize_uv
from .unwrap import Unwrapped
from .views import View


@dataclass
class BakeConfig:
    size: int = 4096
    band_rows: int = 512
    sharpness: float = 4.0  # exponent k of cos^k in the blend weight (higher = sharper / less ghosting)
    cos_lo: float = 0.12  # below this cosine a view has no reliability
    cos_hi: float = 0.55  # above this cosine a view is fully reliable
    feather_px: float = 6.0  # silhouette feathering, in pixels of a 1448 px tall view
    face_boost: float = 8.0  # extra weight of the face view on the face
    face_view: str = "front"
    head_frac: float = 0.135  # top fraction of the body height treated as "head"
    vis_eps: float = 0.004  # depth tolerance as fraction of the body height
    coarse: int = 4  # block size for the fill cloud
    chunk: int = 800_000


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def view_weights(
    cos_t: np.ndarray, visible: np.ndarray, feather: np.ndarray, cfg: BakeConfig, prior: np.ndarray | float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (reliability q in [0,1], blend weight w).

    q = visible * smoothstep(cos) * feather      (used for the confidence)
    w = q * cos^k * prior                         (used for the colour blend: winner-take-most)
    """
    c = np.clip(cos_t, 0.0, 1.0)
    q = visible.astype(np.float32) * smoothstep(c, cfg.cos_lo, cfg.cos_hi) * feather
    w = q * np.power(c, cfg.sharpness) * prior
    return q.astype(np.float32), w.astype(np.float32)


def remap_points(src: np.ndarray, x: np.ndarray, y: np.ndarray, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Sample ``src`` (H, W[, C]) at pixel-index coordinates (cv2 convention: integer = pixel centre)."""
    n = x.size
    cols = 2048
    rows = -(-n // cols)
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(src, mx.reshape(rows, cols), my.reshape(rows, cols), interp, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n] if out.ndim == 3 else out.reshape(-1)[:n]


def sample_view(
    view: View, P: np.ndarray, N: np.ndarray, cfg: BakeConfig, body_h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project texels into a view.

    Returns ``(q, cos, colour_lin, visible)`` where ``q`` is the reliability in [0, 1] (visibility x incidence
    angle x silhouette feather), ``cos`` the incidence cosine and ``colour_lin`` (n, 3) the bicubic colour sample.
    """
    cam = view.cam
    W, H = view.size
    p = cam.project(P)
    x, y, depth = p[:, 0], p[:, 1], p[:, 2]
    cos_t = N @ cam.to_camera
    # ---- visibility against the (conservative) z-buffer
    z = view.zsc
    zi = np.clip(np.floor(x * z).astype(np.int64), 0, W * z - 1)
    zj = np.clip(np.floor(y * z).astype(np.int64), 0, H * z - 1)
    zm = view.zmin[zj, zi]
    eps = cfg.vis_eps * body_h + 2.0 / (cam.scale * z) / np.maximum(cos_t, 0.25)
    visible = (depth <= zm + eps) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    # ---- image coordinates incl. the alignment flow
    qx = (x - 0.5).astype(np.float32)
    qy = (y - 0.5).astype(np.float32)
    if view.flow is not None:
        f = remap_points(view.flow, qx, qy)
        sx, sy = qx + f[:, 0], qy + f[:, 1]
    else:
        sx, sy = qx, qy
    inside = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
    visible &= inside
    dist = remap_points(view.inner_dist, sx, sy)
    fpx = cfg.feather_px * H / 1448.0
    feather = smoothstep(dist, 1.0, 1.0 + fpx)
    col = remap_points(view.image_lin, sx, sy, cv2.INTER_CUBIC)
    col = np.clip(col, 0.0, None)
    q, _ = view_weights(cos_t, visible, feather, cfg)
    return q, cos_t.astype(np.float32), col.astype(np.float32), visible


def face_prior(view_name: str, P: np.ndarray, N: np.ndarray, cfg: BakeConfig, ymin: float, ymax: float) -> np.ndarray | float:
    """Prefer the face view on the face (head zone, front hemisphere)."""
    if view_name != cfg.face_view:
        return 1.0
    h = ymax - ymin
    y0 = ymax - cfg.head_frac * h
    zone = smoothstep(P[:, 1], y0 - 0.03 * h, y0)
    front = smoothstep(N[:, 2], 0.1, 0.6)
    return (1.0 + (cfg.face_boost - 1.0) * zone * front).astype(np.float32)


@dataclass
class BakeResult:
    color_u8: np.ndarray  # (S, S, 3) uint8 sRGB observed colour (blend of views), 0 where unobserved
    conf: np.ndarray  # (S, S) uint8 confidence * 255
    valid: np.ndarray  # (S, S) bool, texel is covered by the UV layout
    dominant: np.ndarray  # (S, S) uint8: index+1 of the highest-weight view (0 = none)
    cloud: dict = field(default_factory=dict)  # coarse cloud of (pos, nrm, col_lin, conf, cell)
    view_names: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


def texel_attributes(
    unw: Unwrapped, vnormals: np.ndarray, size: int, r0: int, r1: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise rows [r0, r1) of the UV layout; return (rows, cols, positions, normals) of covered texels."""
    uv_px = unw.uv.astype(np.float64) * size
    tri_id, bary = rasterize_uv(uv_px, unw.faces, size, size, row_range=(r0, r1))
    rr, cc = np.nonzero(tri_id >= 0)
    tid = tri_id[rr, cc]
    lam = bary[rr, cc].astype(np.float64)
    f = unw.faces[tid]
    P = np.einsum("ij,ijk->ik", lam, unw.vertices[f].astype(np.float64))
    N = np.einsum("ij,ijk->ik", lam, vnormals[f])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return rr + r0, cc, P.astype(np.float32), N.astype(np.float32)


def blend_views(
    views: list[View], P: np.ndarray, N: np.ndarray, cfg: BakeConfig, ymin: float, ymax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend all views at the given texels. Returns (colour_lin (n,3), conf (n,), dominant (n,), sum_w (n,))."""
    n = len(P)
    body_h = ymax - ymin
    sumw = np.zeros(n, np.float32)
    sumc = np.zeros((n, 3), np.float32)
    keep = np.ones(n, np.float32)  # prod(1 - q)
    best = np.zeros(n, np.float32)
    dom = np.zeros(n, np.uint8)
    for vi, view in enumerate(views):
        q, cos_t, col, _ = sample_view(view, P, N, cfg, body_h)
        prior = face_prior(view.name, P, N, cfg, ymin, ymax)
        w = q * np.power(np.clip(cos_t, 0, 1), cfg.sharpness) * prior
        sumw += w
        sumc += w[:, None] * col
        keep *= 1.0 - q
        upd = w > best
        best[upd] = w[upd]
        dom[upd] = vi + 1
    conf = 1.0 - keep
    color = sumc / np.maximum(sumw, 1e-12)[:, None]
    color[sumw <= 1e-12] = 0.0
    return color.astype(np.float32), conf.astype(np.float32), dom, sumw


def run_bake(
    unw: Unwrapped,
    vnormals: np.ndarray,
    views: list[View],
    cfg: BakeConfig,
    log=print,
) -> BakeResult:
    S = cfg.size
    ymin, ymax = float(unw.vertices[:, 1].min()), float(unw.vertices[:, 1].max())
    color_u8 = np.zeros((S, S, 3), np.uint8)
    conf_u8 = np.zeros((S, S), np.uint8)
    valid = np.zeros((S, S), bool)
    dominant = np.zeros((S, S), np.uint8)
    B = cfg.coarse
    Sc = S // B
    n_cells = Sc * Sc
    cell_n = np.zeros(n_cells, np.float64)
    cell_pos = np.zeros((n_cells, 3), np.float64)
    cell_nrm = np.zeros((n_cells, 3), np.float64)
    cell_col = np.zeros((n_cells, 3), np.float64)
    cell_conf = np.zeros(n_cells, np.float64)
    for r0 in range(0, S, cfg.band_rows):
        r1 = min(S, r0 + cfg.band_rows)
        rr, cc, P, N = texel_attributes(unw, vnormals, S, r0, r1)
        if len(rr) == 0:
            continue
        for s in range(0, len(rr), cfg.chunk):
            sl = slice(s, s + cfg.chunk)
            col, conf, dom, _ = blend_views(views, P[sl], N[sl], cfg, ymin, ymax)
            r, c = rr[sl], cc[sl]
            color_u8[r, c] = linear_to_u8(col)
            conf_u8[r, c] = np.rint(conf * 255).astype(np.uint8)
            valid[r, c] = True
            dominant[r, c] = dom
            cell = (r // B) * Sc + (c // B)
            cell_n += np.bincount(cell, minlength=n_cells)
            cell_conf += np.bincount(cell, weights=conf, minlength=n_cells)
            for a in range(3):
                cell_pos[:, a] += np.bincount(cell, weights=P[sl][:, a], minlength=n_cells)
                cell_nrm[:, a] += np.bincount(cell, weights=N[sl][:, a], minlength=n_cells)
                cell_col[:, a] += np.bincount(cell, weights=(col[:, a] * conf), minlength=n_cells)
        log(f"  band {r0}-{r1}: {len(rr)} texels")
    used = cell_n > 0
    cc_ = cell_conf[used]
    nrm = cell_nrm[used] / cell_n[used, None]
    cloud = {
        "cell": np.flatnonzero(used),
        "pos": (cell_pos[used] / cell_n[used, None]).astype(np.float32),
        "nrm": (nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)).astype(np.float32),
        # confidence-weighted mean colour of the cell (only meaningful where conf > 0)
        "col": (cell_col[used] / np.maximum(cc_, 1e-9)[:, None]).astype(np.float32),
        "conf": (cc_ / cell_n[used]).astype(np.float32),
        "grid": Sc,
    }
    cov = {
        "texels": int(valid.sum()),
        "conf_ge_0.5": float(((conf_u8 >= 128) & valid).sum() / max(valid.sum(), 1)),
        "conf_ge_0.9": float(((conf_u8 >= 230) & valid).sum() / max(valid.sum(), 1)),
        "conf_lt_0.1": float(((conf_u8 < 26) & valid).sum() / max(valid.sum(), 1)),
    }
    return BakeResult(color_u8, conf_u8, valid, dominant, cloud, [v.name for v in views], cov)
