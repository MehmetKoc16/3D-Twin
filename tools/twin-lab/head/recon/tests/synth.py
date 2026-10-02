"""Synthetic fixtures (no personal data): an ellipsoid 'head', pinhole cameras, silhouette photos."""

from __future__ import annotations

import numpy as np
import trimesh

from headrecon.photos import Photo
from headrecon.twinhead import TwinHead
from headrecon.viewcam import ViewCam, sample_surface, silhouette

W, H = 640, 800
F = 0.71 * H
CENTER = np.array([0.0, 1.65, 0.0])


def ellipsoid(radii=(0.09, 0.115, 0.10), subdivisions: int = 3, soup: bool = True):
    """Closed ellipsoid around ``CENTER``. With ``soup`` every triangle gets its own UV cell (seam free atlas)."""
    ico = trimesh.creation.icosphere(subdivisions=subdivisions)
    v = ico.vertices * np.array(radii) + CENTER
    f = ico.faces.astype(np.int64)
    if not soup:
        return v, f, None
    verts = v[f].reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    n = len(f)
    g = int(np.ceil(np.sqrt(n)))
    uv = np.zeros((len(verts), 2))
    for i in range(n):
        cx, cy = (i % g), (i // g)
        a = np.array([[0.1, 0.1], [0.85, 0.1], [0.1, 0.85]])
        uv[3 * i : 3 * i + 3] = (np.array([cx, cy]) + a) / g
    return verts, faces, uv


def make_cam(yaw: float, dist: float = 0.5, tx: float = 0.0, ty: float = 0.0, pitch: float = 0.0) -> ViewCam:
    return ViewCam(yaw, pitch, 0.0, dist, tx, ty, F, W / 2, H / 2, CENTER.copy())


VIEW_YAW = {"front": 0.0, "back": 180.0, "profile_nose_right": -90.0, "profile_nose_left": 90.0}


def photo_from(verts, faces, cam: ViewCam, name: str, color_fn=None) -> Photo:
    m = silhouette(cam, sample_surface(verts, faces), W, H, 1.0) > 0
    yy, xx = np.mgrid[0:H, 0:W]
    rgb = np.zeros((H, W, 3), np.uint8)
    if color_fn is None:
        rgb[..., 0] = np.clip(xx / W * 255, 0, 255)
        rgb[..., 1] = np.clip(yy / H * 255, 0, 255)
        rgb[..., 2] = 120
    else:
        rgb[:] = color_fn
    return Photo(name, rgb, rgb, m, np.zeros((H, W), bool))


def twin_head(ymax: float = 1.765) -> TwinHead:
    return TwinHead(ymax=ymax, chin_y=1.545, eye_y=1.66, pivot=CENTER.copy(), lm3d=None)
