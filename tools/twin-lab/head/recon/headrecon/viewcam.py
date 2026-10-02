"""Weak-perspective (orthographic) photo cameras around the head, and cheap silhouette rasterisation."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass
class ViewCam:
    """Pinhole camera looking at a pivot (a phone photo: strong perspective at arm's length).

    World: metres, +Y up, the face looks to +Z, the character's left = +X. ``yaw``: azimuth of the camera around the head
    (0 = in front, +90 = on the character's left, 180 = behind), ``pitch``: elevation of the camera (positive = above,
    looking down), ``roll``: rotation about the viewing axis (degrees); ``dist``: camera-to-pivot distance (m), ``f``:
    focal length (px), (``cx`` + ``tx``, ``cy`` + ``ty``) the image position of the pivot.
    ``x = cx + tx + f * u / Zc``, ``y = cy + ty - f * v / Zc`` with ``Zc = dist + depth``.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    dist: float = 0.5
    tx: float = 0.0
    ty: float = 0.0
    f: float = 1450.0
    cx: float = 0.0
    cy: float = 0.0
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, e = np.radians(self.yaw), np.radians(self.pitch)
        to_cam = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        fwd = -to_cam
        right = np.cross(fwd, UP)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        if self.roll:
            R = rot_about(fwd, np.radians(self.roll))
            right, up = R @ right, R @ up
        return right, up, fwd

    def project(self, p: np.ndarray) -> np.ndarray:
        """(n, 3) world -> (x_px, y_px, Zc); Zc = distance along the viewing axis (grows away from the camera)."""
        right, up, fwd = self.axes()
        q = np.asarray(p, dtype=np.float64) - self.pivot
        zc = np.maximum(self.dist + q @ fwd, 1e-3)
        return np.stack([self.cx + self.tx + self.f * (q @ right) / zc, self.cy + self.ty - self.f * (q @ up) / zc, zc], axis=1)

    def to_camera(self) -> np.ndarray:
        return -self.axes()[2]

    def params(self) -> np.ndarray:
        return np.array([self.yaw, self.pitch, self.roll, self.dist, self.tx, self.ty], dtype=np.float64)

    def with_params(self, x: np.ndarray) -> ViewCam:
        return ViewCam(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), self.f, self.cx,
                       self.cy, self.pivot)

    def plane_offset_to_world(self, d_px: np.ndarray, zc: np.ndarray) -> np.ndarray:
        """Pixel offsets (n, 2) at depths ``zc`` (n,) -> world displacement (n, 3) in the plane parallel to the image."""
        right, up, _ = self.axes()
        k = (zc / self.f)[:, None]
        return (d_px[:, 0:1] * right[None] - d_px[:, 1:2] * up[None]) * k


def sample_surface(verts: np.ndarray, faces: np.ndarray, max_edge: float = 0.004) -> np.ndarray:
    """Points on the triangles (vertices + edge midpoints + centroids, long triangles subdivided) for splat silhouettes."""
    t = verts[faces]
    pts = [verts[np.unique(faces)], t.mean(1), 0.5 * (t[:, 0] + t[:, 1]), 0.5 * (t[:, 1] + t[:, 2]), 0.5 * (t[:, 2] + t[:, 0])]
    e = np.linalg.norm(t[:, 1] - t[:, 0], axis=1)
    big = e > max_edge * 2
    if big.any():
        tb = t[big]
        for a_, b_ in ((0.25, 0.25), (0.5, 0.25), (0.25, 0.5), (0.7, 0.15), (0.15, 0.7)):
            pts.append(tb[:, 0] * (1 - a_ - b_) + tb[:, 1] * a_ + tb[:, 2] * b_)
    return np.concatenate(pts, axis=0)


def silhouette(cam: ViewCam, points: np.ndarray, width: int, height: int, shrink: float = 1.0) -> np.ndarray:
    """Binary mask (uint8 0/1) of the projected surface points at ``shrink`` x the photo resolution: splat, close, fill."""
    p = cam.project(points)[:, :2] * shrink
    h, w = int(height * shrink), int(width * shrink)
    xi = np.rint(p[:, 0]).astype(np.int64)
    yi = np.rint(p[:, 1]).astype(np.int64)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    m = np.zeros((h, w), np.uint8)
    m[yi[ok], xi[ok]] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i) / max(float(u), 1.0)
