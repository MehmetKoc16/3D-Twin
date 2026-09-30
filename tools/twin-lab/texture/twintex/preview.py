"""Software renderer for previews (numpy z-buffer + texture lookup)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .bake import remap_points
from .camera import OrthoCamera
from .colorspace import linear_to_srgb, srgb_to_linear
from .raster import barycentric_at, zbuffer

BG = np.array([0.62, 0.62, 0.64], np.float32)  # neutral grey (sRGB)


def render_textured(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    normals: np.ndarray,
    texture_u8: np.ndarray | None,
    cam: OrthoCamera,
    width: int,
    height: int,
    ss: int = 2,
    shading: float = 0.28,
    flat_color: tuple[float, float, float] = (0.7, 0.7, 0.72),
) -> np.ndarray:
    """Render one view; returns uint8 (height, width, 3) sRGB."""
    W, H = width * ss, height * ss
    p = cam.project(vertices)
    p[:, 0] *= ss
    p[:, 1] *= ss
    depth, fid = zbuffer(p, faces, W, H)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    lam = barycentric_at(p[:, :2], faces, f, xs, ys)
    tri = faces[f]
    n = np.einsum("ij,ijk->ik", lam, normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if texture_u8 is not None:
        S = texture_u8.shape[0]
        t = np.einsum("ij,ijk->ik", lam, uv[tri].astype(np.float64))
        tex_lin = srgb_to_linear(texture_u8.astype(np.float32) / 255.0)
        mx = (t[:, 0] * S - 0.5).astype(np.float32)
        my = (t[:, 1] * S - 0.5).astype(np.float32)
        col = remap_points(tex_lin, mx, my)
    else:
        col = np.tile(srgb_to_linear(np.array(flat_color, np.float32)), (len(f), 1))
    ndl = np.clip(n @ cam.to_camera, 0, 1)
    shade = (1.0 - shading) + shading * ndl
    col = col * shade[:, None]
    img = np.tile(srgb_to_linear(BG)[None, None, :], (H, W, 1)).astype(np.float32)
    img[ys, xs] = col
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) if ss > 1 else img
    return np.clip(np.rint(linear_to_srgb(img) * 255), 0, 255).astype(np.uint8)


def render_sheet(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    normals: np.ndarray,
    texture_u8: np.ndarray | None,
    out_dir: str | Path,
    tile: tuple[int, int] = (720, 1080),
    prefix: str = "",
) -> dict[str, str]:
    """Write front / 3-4 / side / back renders, a head close-up and a contact sheet. Returns {name: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tw, th = tile
    views = {"front": 0.0, "threequarter": 35.0, "side": 90.0, "back": 180.0}
    paths: dict[str, str] = {}
    tiles = []
    for name, az in views.items():
        cam = OrthoCamera.azimuth(name, az).fit_bounds(vertices, tw, th, margin=0.04)
        img = render_textured(vertices, faces, uv, normals, texture_u8, cam, tw, th)
        p = out / f"{prefix}{name}.png"
        Image.fromarray(img).save(p)
        paths[name] = str(p)
        tiles.append(img)
    # head close-up (front)
    ymax = vertices[:, 1].max()
    ymin = vertices[:, 1].min()
    hh = 0.16 * (ymax - ymin)
    sel = vertices[:, 1] > ymax - hh
    for hname, az in (("head_front", 0.0), ("head_threequarter", 35.0), ("head_side", 90.0), ("head_back", 180.0)):
        cam = OrthoCamera.azimuth(hname, az).fit_bounds(vertices[sel], 900, 900, margin=0.05)
        img = render_textured(vertices, faces, uv, normals, texture_u8, cam, 900, 900, ss=2)
        p = out / f"{prefix}{hname}.png"
        Image.fromarray(img).save(p)
        paths[hname] = str(p)
    sheet = np.concatenate(tiles, axis=1)
    p = out / f"{prefix}sheet.png"
    Image.fromarray(sheet).save(p)
    paths["sheet"] = str(p)
    return paths
