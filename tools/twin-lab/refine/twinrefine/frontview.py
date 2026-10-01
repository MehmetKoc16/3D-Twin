"""The front photo as the texture stage sees it: same camera, same alignment flow, so that landmarks detected in the
photo, the relief and the re-projected pixels all live in one frame.

The texture stage fits an orthographic camera (similarity from silhouette IoU) and a smooth 2-D warp ("flow") per
view. ``FrontView`` rebuilds exactly that (same alpha cut-out when the shape stage's ``cutout_front.png`` can be
found, same camera seed) and exposes the world <-> photo-pixel mappings, including the inverse of the flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from twintex import meshio  # noqa: E402  (texture stage)
from twintex.bake import remap_points  # noqa: E402
from twintex.camera import OrthoCamera  # noqa: E402
from twintex.views import View, build_view, load_view_image  # noqa: E402

from . import log


@dataclass
class FrontView:
    rgb: np.ndarray  # (H, W, 3) uint8
    alpha: np.ndarray  # (H, W) float32
    view: View

    @property
    def cam(self) -> OrthoCamera:
        return self.view.cam

    def world_to_photo(self, P: np.ndarray) -> np.ndarray:
        """World points (n, 3) -> photo pixel coordinates (n, 2) in cv2 index convention (pixel centre = integer)."""
        p = self.cam.project(P)
        qx = (p[:, 0] - 0.5).astype(np.float32)
        qy = (p[:, 1] - 0.5).astype(np.float32)
        if self.view.flow is not None:
            f = remap_points(self.view.flow, qx, qy)
            return np.stack([qx + f[:, 0], qy + f[:, 1]], axis=1).astype(np.float64)
        return np.stack([qx, qy], axis=1).astype(np.float64)

    def photo_to_world_xy(self, s: np.ndarray, iters: int = 8) -> np.ndarray:
        """Photo pixel coordinates (n, 2) -> world (x, y) on the camera's image plane (inverse of ``world_to_photo``)."""
        s = np.asarray(s, dtype=np.float64)
        q = s.copy()
        if self.view.flow is not None:
            for _ in range(iters):
                f = remap_points(self.view.flow, q[:, 0].astype(np.float32), q[:, 1].astype(np.float32))
                q = s - f.astype(np.float64)
        cam = self.cam
        xc, yc = q[:, 0] + 0.5, q[:, 1] + 0.5
        x = (xc - cam.tx) / (cam.scale * cam.aspect)
        y = (cam.ty - yc) / cam.scale
        return np.stack([x, y], axis=1)


def find_shape_dir(textured_glb: Path) -> Path | None:
    """The shape stage folder the texture stage was run on (its report.json names the mesh), if it still exists."""
    rep = textured_glb.parent / "report.json"
    if rep.exists():
        try:
            mesh = Path(json.loads(rep.read_text(encoding="utf-8")).get("mesh", ""))
            if mesh.exists():
                return mesh.parent
        except (OSError, ValueError):
            pass
    sib = textured_glb.parent.parent / "shape"
    return sib if sib.exists() else None


def build_front(
    front_path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    textured_glb: Path | None = None,
    use_flow: bool = True,
) -> FrontView:
    """Align the front photo to the mesh like ``twintex.pipeline.run`` does for the front view."""
    rgb, alpha = load_view_image(front_path)
    init = None
    shape_dir = find_shape_dir(textured_glb) if textured_glb is not None else None
    if shape_dir is not None:
        cut = shape_dir / "cutout_front.png"
        if cut.exists():
            ca = np.array(Image.open(cut).convert("RGBA"))
            if ca.shape[:2] == alpha.shape:
                alpha = ca[..., 3].astype(np.float32) / 255.0
                log(f"front view: alpha from {cut}")
        meta = meshio.load_shape_meta(shape_dir)
        mc = ((meta.get("cameras") or {}).get("views") or {}).get("front")
        if isinstance(mc, dict) and "pxPerMeter" in mc and "originPx" in mc:
            init = (float(mc["pxPerMeter"]), float(mc["originPx"][0]), float(mc["originPx"][1]))
    cam0 = OrthoCamera.axis("front")
    view = build_view("front", rgb, alpha, verts, faces, cam0, use_flow=use_flow, init=init)
    log(f"front view: IoU {view.stats['iou_initial']:.3f} -> {view.stats['iou_similarity']:.3f} -> "
        f"{view.stats['iou_final']:.3f}, {view.stats['scale_px_per_m']:.1f} px/m")
    return FrontView(rgb, alpha, view)
