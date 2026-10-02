"""The twin's head: which vertices belong to it, where its face landmarks sit in 3D, the neck falloff weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from twintex.camera import OrthoCamera
from twintex.raster import zbuffer

from . import log

LM_CHIN = 152
LM_NOSE_TIP = 1
LM_FOREHEAD = 10


@dataclass
class TwinHead:
    ymax: float
    chin_y: float
    eye_y: float
    pivot: np.ndarray  # head centre (ear height, mid depth)
    lm3d: np.ndarray | None  # (468, 3) world positions of the twin's face landmarks (None when not detected)

    def falloff(self, y: np.ndarray, below_chin: float = 0.11, above_chin: float = 0.015) -> np.ndarray:
        """Deformation weight: 1 above the chin, smooth ramp to 0 ``below_chin`` metres lower (into the neck)."""
        t = np.clip((y - (self.chin_y - below_chin)) / (below_chin + above_chin), 0.0, 1.0)
        return t * t * (3 - 2 * t)


def head_verts_mask(verts: np.ndarray, ymax: float | None = None, extent: float = 0.34) -> np.ndarray:
    ymax = verts[:, 1].max() if ymax is None else ymax
    return verts[:, 1] > ymax - extent


def detect_twin_landmarks(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, uv, atlas, detector=None):
    """Render the twin's head from the front and run the face landmarker on that render; lift the 2-D landmarks to
    3-D with the render's depth map."""
    from twinrefine.landmarks import detect_landmarks
    from twinrefine.render import render

    ymax = verts[:, 1].max()
    sel = verts[:, 1] > ymax - 0.30
    size = 900
    cam = OrthoCamera.azimuth("f", 0.0).fit_bounds(verts[sel], size, size, margin=0.12)
    img = render(verts, faces, normals, cam, size, size, uv, atlas)
    detect = detector or (lambda rgb: detect_landmarks(rgb, None, full_frame=True))
    lm = detect(img)
    if lm is None:
        log("twin head: no face found in its front render (landmark-driven steps skipped)")
        return None
    p = cam.project(verts)
    depth, _fid = zbuffer(np.stack([p[:, 0], p[:, 1], p[:, 2]], axis=1), faces, size, size)
    xy = lm.xy[:468]
    xi = np.clip(np.rint(xy[:, 0] - 0.5).astype(int), 0, size - 1)
    yi = np.clip(np.rint(xy[:, 1] - 0.5).astype(int), 0, size - 1)
    d = depth[yi, xi]
    if not np.isfinite(d).all():
        # landmarks outside the silhouette: take the nearest finite depth
        fin = np.isfinite(depth)
        from scipy import ndimage

        idx = ndimage.distance_transform_edt(~fin, return_distances=False, return_indices=True)
        d = depth[idx[0][yi, xi], idx[1][yi, xi]]
    x = (xy[:, 0] - cam.tx) / (cam.scale * cam.aspect)
    y = (cam.ty - xy[:, 1]) / cam.scale
    z = -d  # front camera: depth = -z
    return np.stack([x, y, z], axis=1)


def analyse_head(verts, faces, normals, uv, atlas, detector=None) -> TwinHead:
    ymax = float(verts[:, 1].max())
    lm = detect_twin_landmarks(verts, faces, normals, uv, atlas, detector)
    if lm is not None:
        chin_y = float(lm[LM_CHIN, 1])
        eye_y = float(lm[[33, 133, 362, 263], 1].mean())
    else:
        chin_y, eye_y = ymax - 0.235, ymax - 0.12
    sel = (verts[:, 1] > chin_y) & (verts[:, 1] < ymax)
    c = verts[sel].mean(0)
    pivot = np.array([0.0, eye_y, c[2]])
    log(f"twin head: top {ymax:.3f} m, chin {chin_y:.3f} m, eyes {eye_y:.3f} m, landmarks {'yes' if lm is not None else 'no'}")
    return TwinHead(ymax, chin_y, eye_y, pivot, lm)
