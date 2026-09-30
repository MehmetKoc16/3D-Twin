"""Synthetic fixtures: a procedurally coloured capsule "body" rendered to orthographic views. No user images."""

from __future__ import annotations

import cv2
import numpy as np
import trimesh

from twintex.camera import OrthoCamera
from twintex.colorspace import u8_to_linear
from twintex.raster import barycentric_at, zbuffer
from twintex.views import View, build_depth


def color_fn(P: np.ndarray) -> np.ndarray:
    """Smooth ground-truth sRGB colour (0..1) as a function of the 3-D position."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = 0.55 + 0.35 * np.sin(5.0 * x + 1.0) * np.cos(2.5 * y)
    g = 0.50 + 0.35 * np.sin(4.0 * y + 2.0 * z)
    b = 0.50 + 0.35 * np.cos(4.0 * z - 3.0 * x + y)
    return np.clip(np.stack([r, g, b], axis=1), 0.05, 0.95)


def make_capsule(height: float = 1.2, radius: float = 0.22, count: int = 40) -> trimesh.Trimesh:
    """Upright capsule, feet at y = 0, centred on x = z = 0, welded, outward normals."""
    m = trimesh.creation.capsule(height=height, radius=radius, count=[count, count])
    # trimesh capsules are along Z: rotate to Y
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.vertices[:, 1] -= m.vertices[:, 1].min()
    return trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=True)


def render_view(
    mesh: trimesh.Trimesh,
    cam: OrthoCamera,
    width: int,
    height: int,
    fn=color_fn,
    gain: float = 1.0,
    bg: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Render (rgb uint8, alpha float32) of the mesh with analytic colours."""
    p = cam.project(mesh.vertices)
    depth, fid = zbuffer(p, mesh.faces, width, height)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    lam = barycentric_at(p[:, :2], mesh.faces, f, xs, ys)
    P = np.einsum("ij,ijk->ik", lam, mesh.vertices[mesh.faces[f]])
    img = np.full((height, width, 3), bg, np.float32)
    img[ys, xs] = np.clip(fn(P) * gain, 0, 1)
    alpha = (fid >= 0).astype(np.float32)
    return np.rint(img * 255).astype(np.uint8), alpha


def framed_camera(mesh: trimesh.Trimesh, name: str, width: int, height: int, margin: float = 0.08) -> OrthoCamera:
    return OrthoCamera.axis(name).fit_bounds(mesh.vertices, width, height, margin=margin)


def manual_view(mesh: trimesh.Trimesh, name: str, width: int = 400, height: int = 560, gain: float = 1.0,
                fn=color_fn, zsc: int = 2) -> View:
    """A View with an exact camera (no alignment), for testing projection / blending on their own."""
    cam = framed_camera(mesh, name, width, height)
    rgb, alpha = render_view(mesh, cam, width, height, fn=fn, gain=gain)
    inner = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    v = View(name=name, cam=cam, image_lin=u8_to_linear(rgb), alpha=alpha, inner_dist=inner, flow=None, zsc=zsc)
    build_depth(v, np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64))
    return v
