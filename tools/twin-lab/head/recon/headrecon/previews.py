"""Preview renders: the head from each photo's own camera next to the photo, plus standard views (lit and clay)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from twinrefine.render import DEFAULT_LIGHTS, RAKING_LIGHTS, render, shade
from twintex.bake import remap_points
from twintex.camera import OrthoCamera
from twintex.colorspace import linear_to_srgb, srgb_to_linear
from twintex.raster import zbuffer

from .calibrate import Calib
from .photos import Photo
from .viewcam import ViewCam

BG = np.array([0.50, 0.51, 0.53], np.float32)


def render_cam(
    cam: ViewCam, verts, faces, normals, width: int, height: int, uv=None, atlas=None, clay: bool = False,
    lights=DEFAULT_LIGHTS, ss: int = 1, y_min: float | None = None,
) -> np.ndarray:
    """Lit render through a pinhole photo camera (head only above ``y_min``)."""
    if y_min is not None:
        faces = faces[(verts[faces][:, :, 1] > y_min).all(axis=1)]
    W, H = width * ss, height * ss
    p = cam.project(verts)
    depth, fid = zbuffer(np.stack([p[:, 0] * ss, p[:, 1] * ss, p[:, 2]], axis=1), faces, W, H)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    tri = faces[f]
    from twintex.raster import barycentric_at

    lam = barycentric_at(p[:, :2] * ss, faces, f, xs, ys)
    n = np.einsum("ij,ijk->ik", lam, normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    right, up, fwd = cam.axes()
    ncam = np.stack([n @ right, n @ up, -(n @ fwd)], axis=1)
    if atlas is not None and uv is not None and not clay:
        S = atlas.shape[0]
        t = np.einsum("ij,ijk->ik", lam, uv[tri].astype(np.float64))
        tex = srgb_to_linear(atlas.astype(np.float32) / 255.0)
        alb = np.clip(remap_points(tex, (t[:, 0] * S - 0.5).astype(np.float32), (t[:, 1] * S - 0.5).astype(np.float32)), 0, None)
    else:
        alb = np.full((len(f), 3), srgb_to_linear(np.float32(0.72)), np.float32)
    col = shade(ncam, alb, lights)
    img = np.tile(srgb_to_linear(BG)[None, None, :], (H, W, 1)).astype(np.float32)
    img[ys, xs] = col
    if ss > 1:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(linear_to_srgb(img) * 255), 0, 255).astype(np.uint8)


def _crop_box(cam: ViewCam, center: np.ndarray, size_m: float, shape: tuple[int, int]) -> tuple[int, int, int]:
    c = cam.project(center[None])[0]
    s = int(size_m * cam.f / cam.dist)
    x0 = int(np.clip(c[0] - s / 2, 0, max(shape[1] - s, 0)))
    y0 = int(np.clip(c[1] - s / 2, 0, max(shape[0] - s, 0)))
    return x0, y0, s


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def photo_comparisons(
    out_dir: Path, verts, faces, normals, uv, atlas, photos: dict[str, Photo], calibs: dict[str, Calib], center: np.ndarray,
    y_min: float, size: int = 640, size_m: float = 0.40,
) -> dict[str, str]:
    """One sheet per photo: [photo | twin lit from the photo's camera | twin clay]. Written to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, ph in photos.items():
        cam = calibs[name].cam
        H, W = ph.mask.shape
        x0, y0, s = _crop_box(cam, center, size_m, ph.mask.shape)
        full_lit = render_cam(cam, verts, faces, normals, W, H, uv, atlas, y_min=y_min)
        full_clay = render_cam(cam, verts, faces, normals, W, H, clay=True, lights=RAKING_LIGHTS, y_min=y_min)
        tiles = [ph.rgb, full_lit, full_clay]
        row = []
        for t, lab in zip(tiles, ("photo", "twin", "twin clay"), strict=True):
            crop = cv2.resize(t[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA)
            row.append(_label(crop, f"{name}: {lab}"))
        p = out_dir / f"compare_{name}.png"
        Image.fromarray(np.concatenate(row, axis=1)).save(p)
        paths[name] = str(p)
    return paths


def standard_views(
    out_dir: Path, items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]], uv, ymax: float,
    size: int = 560,
) -> str:
    """Rows (label, verts, faces, normals, atlas|None) x columns front / 3-4 / side / back (orthographic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, verts, faces, normals, atlas in items:
        sel = verts[:, 1] > ymax - 0.30
        row = []
        for az, nm in ((0, "front"), (35, "3/4"), (90, "side"), (180, "back")):
            cam = OrthoCamera.azimuth(nm, az).fit_bounds(verts[sel], size, size, margin=0.04)
            lights = RAKING_LIGHTS if atlas is None else DEFAULT_LIGHTS
            img = render(verts, faces, normals, cam, size, size, uv, atlas, lights=lights, clay=atlas is None)
            row.append(_label(img, f"{label} {nm}"))
        rows.append(np.concatenate(row, axis=1))
    p = out_dir / "head_views.png"
    Image.fromarray(np.concatenate(rows, axis=0)).save(p)
    return str(p)
