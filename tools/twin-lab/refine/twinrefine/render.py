"""Small lit software renderer for previews (numpy z-buffer; textured or clay; key + fill + rim + hemisphere light).

Orthographic like the texture stage's cameras, so lighting differences between before / after renders are due to
the geometry only.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from twintex.bake import remap_points  # noqa: E402  (texture stage)
from twintex.camera import OrthoCamera  # noqa: E402
from twintex.colorspace import linear_to_srgb, srgb_to_linear  # noqa: E402
from twintex.raster import barycentric_at, zbuffer  # noqa: E402

from . import log  # noqa: F401  (sys.path setup)

BG = np.array([0.50, 0.51, 0.53], np.float32)  # sRGB


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# lights are given as directions TOWARDS the light in camera space (x right, y up, z toward the viewer)
DEFAULT_LIGHTS = (
    (_unit([-0.55, 0.60, 0.58]), np.array([1.00, 0.96, 0.92]) * 1.15),  # key: upper left, warm
    (_unit([0.65, -0.05, 0.50]), np.array([0.62, 0.70, 0.80]) * 0.45),  # fill: right, cool
    (_unit([0.10, 0.50, -0.85]), np.array([1.0, 1.0, 1.0]) * 0.30),  # rim from behind / above
)
RAKING_LIGHTS = (
    (_unit([-0.92, 0.28, 0.28]), np.array([1.0, 0.97, 0.93]) * 1.35),  # strong grazing light from the side
    (_unit([0.30, 0.20, 0.93]), np.array([0.60, 0.68, 0.78]) * 0.28),
)


def shade(
    n_cam: np.ndarray, albedo_lin: np.ndarray, lights=DEFAULT_LIGHTS, ambient: float = 0.20, spec: float = 0.10
) -> np.ndarray:
    """Lambert + soft Blinn-Phong highlight + hemisphere ambient. ``n_cam``: (k, 3) unit normals in camera space."""
    out = np.zeros_like(albedo_lin)
    view = np.array([0.0, 0.0, 1.0])
    up_amb = 0.5 + 0.5 * n_cam[:, 1:2]
    out += albedo_lin * (ambient * (0.6 + 0.8 * up_amb))
    for d, col in lights:
        ndl = np.clip(n_cam @ d, 0.0, 1.0)[:, None]
        out += albedo_lin * col[None, :] * ndl
        if spec > 0:
            h = _unit(d + view)
            ndh = np.clip(n_cam @ h, 0.0, 1.0)
            out += (spec * np.power(ndh, 28.0) * ndl[:, 0])[:, None] * col[None, :]
    return out


def render(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    cam: OrthoCamera,
    width: int,
    height: int,
    uv: np.ndarray | None = None,
    atlas: np.ndarray | None = None,
    lights=DEFAULT_LIGHTS,
    ss: int = 2,
    clay: bool = False,
    clay_color: float = 0.72,
    spec: float = 0.10,
    face_albedo: np.ndarray | None = None,
) -> np.ndarray:
    """``face_albedo``: optional (m, 3) linear albedo per face (debug colouring); overrides texture and clay."""
    W, H = width * ss, height * ss
    p = cam.project(verts)
    p[:, 0] *= ss
    p[:, 1] *= ss
    depth, fid = zbuffer(p, faces, W, H)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    lam = barycentric_at(p[:, :2], faces, f, xs, ys)
    tri = faces[f]
    n = np.einsum("ij,ijk->ik", lam, normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    # camera-space normals: x = right, y = up, z = toward the viewer
    ncam = np.stack([n @ cam.right, n @ cam.up, n @ cam.to_camera], axis=1)
    if face_albedo is not None:
        alb = face_albedo[f].astype(np.float32)
    elif atlas is not None and uv is not None and not clay:
        S = atlas.shape[0]
        t = np.einsum("ij,ijk->ik", lam, uv[tri].astype(np.float64))
        tex_lin = srgb_to_linear(atlas.astype(np.float32) / 255.0)
        alb = remap_points(tex_lin, (t[:, 0] * S - 0.5).astype(np.float32), (t[:, 1] * S - 0.5).astype(np.float32))
        alb = np.clip(alb, 0, None)
    else:
        alb = np.full((len(f), 3), srgb_to_linear(np.float32(clay_color)), dtype=np.float32)
    col = shade(ncam, alb, lights, spec=spec)
    img = np.tile(srgb_to_linear(BG)[None, None, :], (H, W, 1)).astype(np.float32)
    img[ys, xs] = col
    if ss > 1:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(linear_to_srgb(img) * 255), 0, 255).astype(np.uint8)


def head_box(verts: np.ndarray, frac: float = 0.15) -> np.ndarray:
    ymax = verts[:, 1].max()
    h = frac * (ymax - verts[:, 1].min())
    return verts[:, 1] > ymax - h


def render_head_sheet(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uv: np.ndarray | None,
    atlas: np.ndarray | None,
    out_dir: str | Path,
    prefix: str = "",
    size: int = 900,
) -> dict[str, str]:
    """Lit head renders: textured front / 3-4 / side, plus clay under a raking light (shows the relief)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = head_box(verts)
    paths: dict[str, str] = {}
    specs = [
        ("head_front", 0.0, False, DEFAULT_LIGHTS),
        ("head_threequarter", 35.0, False, DEFAULT_LIGHTS),
        ("head_side", 90.0, False, DEFAULT_LIGHTS),
        ("head_clay_front", 0.0, True, RAKING_LIGHTS),
        ("head_clay_threequarter", 35.0, True, RAKING_LIGHTS),
        ("head_clay_side", 90.0, True, DEFAULT_LIGHTS),
    ]
    for name, az, clay, lights in specs:
        cam = OrthoCamera.azimuth(name, az).fit_bounds(verts[sel], size, size, margin=0.05)
        img = render(verts, faces, normals, cam, size, size, uv, atlas, lights=lights, clay=clay)
        p = out / f"{prefix}{name}.png"
        Image.fromarray(img).save(p)
        paths[name] = str(p)
    return paths


def montage(images: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = images[0].shape[:2]
    rows = []
    for i in range(0, len(images), cols):
        row = images[i : i + cols]
        row = row + [np.zeros_like(images[0])] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
