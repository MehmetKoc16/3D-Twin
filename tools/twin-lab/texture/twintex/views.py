"""View images: loading, matting, alignment and per-view visibility buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import align, matting
from .camera import OrthoCamera
from .colorspace import u8_to_linear
from .raster import raster_pairs, zbuffer


@dataclass
class View:
    name: str
    cam: OrthoCamera
    image_lin: np.ndarray  # (H, W, 3) float32 linear light
    alpha: np.ndarray  # (H, W) float32
    inner_dist: np.ndarray  # (H, W) float32, distance in pixels to the silhouette border (inside)
    flow: np.ndarray | None = None
    zsc: int = 2  # depth buffer supersampling
    zmin: np.ndarray | None = None  # neighbourhood-min depth buffer (conservative visibility)
    stats: dict = field(default_factory=dict)
    _lp: np.ndarray | None = None
    _layers: tuple | None = None  # (sorted keys, cumulative distinct count, depth min, depth scale)

    def lowpass(self, sigma_frac: float = 0.012) -> np.ndarray:
        """Foreground-only low-pass filtered colour image (linear), extended beyond the silhouette.

        Used as a *prior* for surfaces that no view can see (e.g. the back of the body when only the front
        exists): it keeps the large colour regions (skin / shirt / trousers) but drops all detail.
        """
        if self._lp is None:
            H, W = self.alpha.shape
            a = (self.alpha > 0.5).astype(np.float32)
            a = cv2.erode(a, np.ones((5, 5), np.uint8))  # keep matte-edge pixels out
            s = max(sigma_frac * H, 1.0)
            out = np.zeros_like(self.image_lin)
            filled = np.zeros((H, W), bool)
            for k in (1.0, 3.0, 9.0):  # widen the kernel where the foreground support is too thin
                num = cv2.GaussianBlur(self.image_lin * a[..., None], (0, 0), s * k)
                den = cv2.GaussianBlur(a, (0, 0), s * k)
                ok = (den > 0.08) & ~filled
                out[ok] = num[ok] / den[ok][:, None]
                filled |= ok
            if not filled.all():
                out[~filled] = self.image_lin[a > 0.5].mean(0) if (a > 0.5).any() else 0.5
            self._lp = out.astype(np.float32)
        return self._lp

    def surfaces_in_front(self, x: np.ndarray, y: np.ndarray, depth: np.ndarray, tol: float) -> np.ndarray:
        """Number of distinct mesh surfaces the camera ray passes through *before* reaching ``depth`` at pixel (x, y).

        0 = the texel is on the surface the camera sees, 1 = it is the back side of that surface (the "opposite"
        of what the photo shows), >= 2 = other body parts are in between (e.g. an arm in front of the torso).
        """
        assert self._layers is not None, "build_depth() must run first"
        keys, cum, d0, dscale = self._layers
        W, H = self.size
        pix = np.clip(np.floor(y).astype(np.int64), 0, H - 1) * W + np.clip(np.floor(x).astype(np.int64), 0, W - 1)
        dn = np.clip((depth - d0) / dscale, 0.0, 1.0) * 0.98
        lo = np.searchsorted(keys, pix.astype(np.float64), side="left")
        hi = np.searchsorted(keys, pix + dn - tol / dscale * 0.98, side="left")
        cum0 = np.concatenate([[0], cum])
        return (cum0[hi] - cum0[lo]).astype(np.int32)

    @property
    def size(self) -> tuple[int, int]:
        return self.image_lin.shape[1], self.image_lin.shape[0]


def load_view_image(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8, alpha float32). Uses the file's own alpha if it carries a real cut-out."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = np.array(im.convert("RGBA"))
        a = rgba[..., 3].astype(np.float32) / 255.0
        if (a < 0.98).mean() > 0.02:
            return rgba[..., :3].copy(), a
        rgb = rgba[..., :3].copy()
    else:
        rgb = np.array(im.convert("RGB"))
    return rgb, matting.matte(rgb)


def build_view(
    name: str,
    rgb: np.ndarray,
    alpha: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    cam_template: OrthoCamera | None = None,
    use_flow: bool = True,
    allow_aspect: bool = False,
    zsc: int = 2,
    init: tuple[float, float, float] | None = None,
) -> View:
    H, W = alpha.shape
    cam0 = cam_template.copy() if cam_template is not None else OrthoCamera.axis(name)
    cam0.width, cam0.height = W, H
    cam, iou0, iou1 = align.fit_similarity(cam0, verts, faces, alpha, allow_aspect=allow_aspect)
    if init is not None:  # a second search seeded by the shape agent's camera; keep whichever fits better
        cam_b, iou0_b, iou1_b = align.fit_similarity(cam0, verts, faces, alpha, allow_aspect=allow_aspect, init=init)
        if iou1_b > iou1:
            cam, iou0, iou1 = cam_b, iou0_b, iou1_b
    flow, iou2 = None, iou1
    if use_flow:
        flow, iou2 = align.refine_flow(cam, verts, faces, alpha)
    lin = u8_to_linear(rgb)
    inner = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    v = View(name=name, cam=cam, image_lin=lin, alpha=alpha, inner_dist=inner, flow=flow, zsc=zsc)
    v.stats = {
        "iou_initial": iou0,
        "iou_similarity": iou1,
        "iou_final": iou2,
        "scale_px_per_m": cam.scale,
        "tx": cam.tx,
        "ty": cam.ty,
    }
    build_depth(v, verts, faces)
    return v


def build_depth(view: View, verts: np.ndarray, faces: np.ndarray) -> None:
    """Render the conservative (3x3 min-filtered) depth map at ``zsc`` x image resolution."""
    W, H = view.size
    z = view.zsc
    p = view.cam.project(verts)
    p[:, 0] *= z
    p[:, 1] *= z
    depth, _ = zbuffer(p, faces, W * z, H * z)
    finite = np.where(np.isfinite(depth), depth, np.float32(1e9)).astype(np.float32)
    view.zmin = cv2.erode(finite, np.ones((3, 3), np.uint8))
    build_layers(view, verts, faces)


def build_layers(view: View, verts: np.ndarray, faces: np.ndarray, merge_tol: float = 0.002) -> None:
    """Per-pixel sorted list of all surface fragments (used to tell "the back of what the photo shows")."""
    W, H = view.size
    p = view.cam.project(verts)
    pix_all, d_all = [], []
    for f, px, py, lam in raster_pairs(p[:, :2], faces, W, H, eps=0.0):
        pix_all.append(py * W + px)
        d_all.append(np.einsum("kj,kj->k", lam, p[faces[f], 2]))
    if not pix_all:
        view._layers = (np.zeros(0), np.zeros(0, np.int64), 0.0, 1.0)
        return
    pix = np.concatenate(pix_all)
    d = np.concatenate(d_all)
    d0, d1 = float(d.min()), float(d.max())
    scale = max(d1 - d0, 1e-9)
    key = pix + (d - d0) / scale * 0.98
    order = np.argsort(key)
    key, pix, d = key[order], pix[order], d[order]
    distinct = np.ones(len(key), np.int64)
    same_pix = pix[1:] == pix[:-1]
    close = (d[1:] - d[:-1]) < merge_tol
    distinct[1:] = np.where(same_pix & close, 0, 1)
    view._layers = (key, np.cumsum(distinct), d0, scale)
