"""Texture the reshaped head from the four photos.

Every head texel of the atlas is projected into each photo (pinhole cameras of ``calibrate``), tested for visibility with a
z-buffer of the deformed mesh, and weighted by view angle, distance to the silhouette edge and "no glasses". The views are
colour matched against the front photo on their overlaps and blended with a sharpened weight (the best view wins, so
slightly misregistered views do not blur each other). Texels no photo sees are filled from the nearest seen texels in 3-D
(this removes the dark band under the chin that the single-view texture had). A neck-band gain matches the new head colour
to the old body texture; below the neck the old atlas is kept unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from twintex.bake import remap_points
from twintex.colorspace import linear_to_srgb, srgb_to_linear
from twintex.raster import rasterize_uv, zbuffer

from . import log
from .calibrate import Calib
from .photos import Photo
from .twinhead import TwinHead

VIEW_BASE = {"front": 1.2, "back": 1.0, "profile_nose_right": 1.0, "profile_nose_left": 1.0}


@dataclass
class BakeParams:
    sharpness: float = 8.0  # exponent on the per-view weights before normalisation
    edge_px: float = 14.0  # weight ramps up over this many pixels inside the person mask
    min_cos: float = 0.12
    vis_tol_m: float = 0.010
    zshrink: float = 0.5
    min_conf: float = 0.04
    gutter: int = 6
    jpeg_quality: int = 95
    extra_weight: dict = field(default_factory=dict)  # per-view multipliers (e.g. low weight for extra photos)


@dataclass
class BakeReport:
    texels: int = 0
    covered_frac: float = 0.0
    view_share: dict = field(default_factory=dict)
    gains: dict = field(default_factory=dict)
    seam_gain: list = field(default_factory=list)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def view_samples(
    photo: Photo, cal: Calib, P: np.ndarray, N: np.ndarray, zb: tuple[np.ndarray, float], prm: BakeParams, lin: np.ndarray,
    dist_in: np.ndarray, glass_free: np.ndarray, chin_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear colours (k, 3) and weights (k,) of texel positions ``P`` seen by one photo."""
    cam = cal.cam
    H, W = photo.mask.shape
    p = cam.project(P)
    x, y, zc = p[:, 0], p[:, 1], p[:, 2]
    inb = (x > 2) & (x < W - 3) & (y > 2) & (y < H - 3)
    cam_pos = cam.pivot + cam.to_camera() * cam.dist
    to_cam = cam_pos[None] - P
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
    cos = np.einsum("ij,ij->i", N, to_cam)
    w = _smoothstep(cos, prm.min_cos, 0.75)
    depth, zs = zb
    xi = np.clip(np.rint(x * zs).astype(np.int64), 0, depth.shape[1] - 1)
    yi = np.clip(np.rint(y * zs).astype(np.int64), 0, depth.shape[0] - 1)
    # nearest depth in a 3x3 window (the z-buffer is coarse): visible when not behind the front surface
    vis = zc <= depth[yi, xi] + prm.vis_tol_m * (1.0 + 2.0 * (1.0 - np.clip(cos, 0, 1)))
    qx, qy = np.clip(x, 0, W - 1).astype(np.float32), np.clip(y, 0, H - 1).astype(np.float32)
    d_in = remap_points(dist_in, qx, qy)
    w = w * vis * inb * np.clip(d_in / prm.edge_px, 0.0, 1.0)
    w = w * _smoothstep(P[:, 1], chin_y - 0.045, chin_y - 0.005)  # the photos show clothes below the jaw, not the neck skin
    w = w * remap_points(glass_free, qx, qy)
    col = remap_points(lin, qx, qy)
    return col, w.astype(np.float32)


def bake_head(
    verts: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    normals: np.ndarray,
    atlas: np.ndarray,
    th: TwinHead,
    photos: dict[str, Photo],
    calibs: dict[str, Calib],
    prm: BakeParams | None = None,
) -> tuple[np.ndarray, BakeReport]:
    prm = prm or BakeParams()
    rep = BakeReport()
    S = atlas.shape[0]
    # ---- head faces and their texels
    gate_v = th.falloff(verts[:, 1], below_chin=0.14)
    hf = np.flatnonzero((gate_v[faces] > 0).any(axis=1))
    tri_id, bary = rasterize_uv(uv.astype(np.float64) * S, faces[hf], S, S)
    ty, tx = np.nonzero(tri_id >= 0)
    ft = hf[tri_id[ty, tx]]
    lam = bary[ty, tx].astype(np.float64)
    tv = faces[ft]
    P = np.einsum("ij,ijk->ik", lam, verts[tv])
    N = np.einsum("ij,ijk->ik", lam, normals[tv])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    g = _smoothstep(P[:, 1], th.chin_y - 0.055, th.chin_y - 0.005)  # texture weight: below the jaw the photos show clothes
    rep.texels = int(len(P))
    log(f"bake: {len(P)} head texels")

    # ---- colour samples per view
    keep_occ = verts[:, 1] > th.ymax - 0.55
    occ_faces = faces[keep_occ[faces].all(axis=1)]
    cols, ws, names = [], [], []
    for name, ph in photos.items():
        cal = calibs[name]
        H, W = ph.mask.shape
        zs = prm.zshrink
        pr = cal.cam.project(verts)
        depth, _ = zbuffer(np.stack([pr[:, 0] * zs, pr[:, 1] * zs, pr[:, 2]], axis=1), occ_faces, int(W * zs), int(H * zs))
        lin = srgb_to_linear(ph.tex_rgb.astype(np.float32) / 255.0)
        dist_in = ndimage.distance_transform_edt(ph.mask).astype(np.float32)
        gf = np.ones(ph.mask.shape, np.float32)
        if ph.glass.any():
            strength = 1.0
            gf = 1.0 - strength * cv2.GaussianBlur(cv2.dilate(ph.glass.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 3)
        c, w = view_samples(ph, cal, P, N, (depth, zs), prm, lin, dist_in, gf, th.chin_y)
        w = w * VIEW_BASE.get(name, 1.0) * prm.extra_weight.get(name, 1.0)
        cols.append(c)
        ws.append(w)
        names.append(name)
        log(f"bake: {name}: {int((w > 0.05).sum())} texels seen")

    # ---- colour match every view to the front photo on the overlap (median log-ratio per channel)
    ref = names.index("front") if "front" in names else int(np.argmax([w.sum() for w in ws]))
    for i, nm in enumerate(names):
        if i == ref:
            continue
        both = (ws[ref] > 0.25) & (ws[i] > 0.25)
        if both.sum() > 500:
            r = np.log(np.maximum(cols[ref][both], 1e-4)) - np.log(np.maximum(cols[i][both], 1e-4))
            gain = np.exp(np.clip(np.median(r, axis=0), np.log(0.7), np.log(1.4)))
            cols[i] = cols[i] * gain[None]
            rep.gains[nm] = [float(v) for v in gain]
    # ---- blend: sharpened weights
    Wk = np.stack([w**prm.sharpness for w in ws], axis=1)  # (k, views)
    tot = Wk.sum(axis=1)
    C = np.zeros((len(P), 3), np.float32)
    for i in range(len(names)):
        C += cols[i] * Wk[:, i : i + 1]
    seen = tot > 1e-6
    C[seen] /= tot[seen, None]
    conf = np.clip(np.stack(ws, 1).max(axis=1) / 0.35, 0.0, 1.0)
    rep.covered_frac = float((conf > 0.5).mean())
    for i, nm in enumerate(names):
        rep.view_share[nm] = float((Wk[:, i] / np.maximum(tot, 1e-9))[seen].mean())

    # ---- fill what no view sees from the nearest well-seen texels in 3-D
    good = conf > 0.5
    if good.sum() > 100 and (~good).any():
        sub = np.flatnonzero(good)[:: max(1, int(good.sum() // 60000))]
        tree = cKDTree(P[sub])
        bad = np.flatnonzero(~good)
        _d, nn = tree.query(P[bad], k=6)
        fill = C[sub][nn].mean(axis=1)
        a = conf[bad][:, None]
        C[bad] = a * C[bad] + (1 - a) * fill

    # ---- neck seam: match the new colours to the old atlas inside the falloff band, then blend by the falloff weight
    old = srgb_to_linear(atlas[ty, tx].astype(np.float32) / 255.0)
    band = (g > 0.02) & (g < 0.35) & good
    gain_seam = np.ones(3, np.float32)
    if band.sum() > 300:
        r = np.log(np.maximum(old[band], 1e-4)) - np.log(np.maximum(C[band], 1e-4))
        gain_seam = np.exp(np.clip(np.median(r, axis=0), np.log(0.6), np.log(1.6))).astype(np.float32)
    rep.seam_gain = [float(v) for v in gain_seam]
    Cn = C * np.power(gain_seam[None], (1.0 - g)[:, None])
    blend = g[:, None]
    out_lin = blend * Cn + (1.0 - blend) * old
    out = atlas.copy()
    out[ty, tx] = np.clip(np.rint(linear_to_srgb(out_lin) * 255.0), 0, 255).astype(np.uint8)

    # ---- gutter: grow the new head texels a few pixels into empty atlas space (filtering / mip-map bleeding)
    occ_all, _ = rasterize_uv(uv.astype(np.float64) * S, faces, S, S)
    valid = np.zeros((S, S), np.float32)
    valid[ty, tx] = 1.0
    img = out.astype(np.float32)
    free = occ_all < 0
    for _ in range(prm.gutter):
        num = cv2.blur(img * valid[..., None], (3, 3))
        den = cv2.blur(valid, (3, 3))
        grow = (valid < 0.5) & free & (den > 1e-6)
        if not grow.any():
            break
        img[grow] = num[grow] / den[grow][:, None]
        valid[grow] = 1.0
    out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return out, rep
