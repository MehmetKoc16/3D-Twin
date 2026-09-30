"""Orthographic view cameras aligned to the body axes.

World convention (same as the rest of the repo): metres, +Y up, character faces +Z, character's left = +X.
Every view image is an orthographic projection; image x grows to the camera's ``right`` vector, image y grows
downwards (``-up``). ``depth`` grows away from the camera (smaller = nearer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# (forward = direction the camera looks along, right vector). "left"/"right" name the character's side that
# the camera sees (left view = camera on the character's left = +X).
AXIS_VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    "back": ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class OrthoCamera:
    """Orthographic camera: ``x_px = s_x * (p . right) + tx``, ``y_px = -s_y * (p . up) + ty``."""

    name: str
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray = field(default_factory=lambda: UP.copy())
    scale: float = 1.0  # pixels per metre
    aspect: float = 1.0  # s_x = scale * aspect
    tx: float = 0.0
    ty: float = 0.0
    width: int = 0
    height: int = 0

    @classmethod
    def axis(cls, name: str, **kw) -> OrthoCamera:
        fwd, right = AXIS_VIEWS[name]
        return cls(name=name, forward=np.array(fwd), right=np.array(right), **kw)

    @classmethod
    def azimuth(cls, name: str, azimuth_deg: float, **kw) -> OrthoCamera:
        """Camera orbiting the vertical axis. azimuth 0 = front, +90 = character's left (+X)."""
        a = np.radians(azimuth_deg)
        pos_dir = np.array([np.sin(a), 0.0, np.cos(a)])  # from the body toward the camera
        forward = -pos_dir
        right = np.cross(forward, UP)
        return cls(name=name, forward=forward, right=right / np.linalg.norm(right), **kw)

    @property
    def to_camera(self) -> np.ndarray:
        """Unit vector from the surface toward the camera."""
        return -self.forward

    def project(self, p: np.ndarray) -> np.ndarray:
        """World points (n, 3) -> (x_px, y_px, depth), all float64."""
        p = np.asarray(p, dtype=np.float64)
        u = p @ self.right
        v = p @ self.up
        d = p @ self.forward
        return np.stack([self.scale * self.aspect * u + self.tx, -self.scale * v + self.ty, d], axis=1)

    def metric_uv(self, p: np.ndarray) -> np.ndarray:
        """World points -> (u, v) metric image-plane coordinates (right, up)."""
        p = np.asarray(p, dtype=np.float64)
        return np.stack([p @ self.right, p @ self.up], axis=1)

    def copy(self) -> OrthoCamera:
        return OrthoCamera(
            self.name, self.forward.copy(), self.right.copy(), self.up.copy(),
            self.scale, self.aspect, self.tx, self.ty, self.width, self.height,
        )

    def fit_bounds(self, verts: np.ndarray, width: int, height: int, margin: float = 0.06) -> OrthoCamera:
        """Frame the vertices centred in a ``width`` x ``height`` image (used for previews / initial guess)."""
        uv = self.metric_uv(verts)
        lo, hi = uv.min(0), uv.max(0)
        ext = hi - lo
        s = min(width * (1 - 2 * margin) / max(ext[0], 1e-9), height * (1 - 2 * margin) / max(ext[1], 1e-9))
        c = (lo + hi) / 2
        cam = self.copy()
        cam.scale, cam.aspect = float(s), 1.0
        cam.tx = width / 2 - s * c[0]
        cam.ty = height / 2 + s * c[1]
        cam.width, cam.height = width, height
        return cam
