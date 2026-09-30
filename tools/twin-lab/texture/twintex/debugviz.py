"""Debug visualisations (atlas, confidence, dominant view, alignment overlay). Outputs stay in user-data/."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .align import _edges, fill_silhouette
from .bake import BakeResult
from .colorspace import linear_to_u8
from .views import View

PALETTE = np.array(
    [[0, 0, 0], [230, 60, 60], [60, 120, 230], [60, 200, 90], [240, 200, 40], [190, 90, 220]], np.uint8
)


def save_debug_maps(prev: Path, atlas: np.ndarray, bake: BakeResult, names: list[str]) -> None:
    Image.fromarray(cv2.resize(atlas, (1024, 1024), interpolation=cv2.INTER_AREA)).save(prev / "debug_atlas_1024.png")
    conf = cv2.resize(bake.conf, (1024, 1024), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(bake.valid.astype(np.uint8) * 255, (1024, 1024), interpolation=cv2.INTER_AREA)
    heat = cv2.applyColorMap(conf, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    heat[valid < 128] = 0
    Image.fromarray(heat).save(prev / "debug_confidence_1024.png")
    dom = cv2.resize(bake.dominant, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(PALETTE[np.minimum(dom, len(PALETTE) - 1)]).save(prev / "debug_dominant_view_1024.png")
    (prev / "debug_dominant_legend.txt").write_text(
        "\n".join(f"{i + 1}: {n} -> RGB {PALETTE[i + 1].tolist()}" for i, n in enumerate(names)), encoding="utf-8"
    )


def save_alignment_overlay(path: Path, view: View, verts: np.ndarray, faces: np.ndarray) -> None:
    """Warp the view image into the mesh frame (flow applied) and draw the mesh silhouette outline on it."""
    W, H = view.size
    img = linear_to_u8(view.image_lin)
    if view.flow is not None:
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        img = cv2.remap(img, gx + view.flow[..., 0], gy + view.flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    xy = view.cam.project(verts)[:, :2]
    m = fill_silhouette(xy, faces, W, H) > 0.5
    edge = cv2.dilate(_edges(m).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img = img.copy()
    img[edge] = (0, 255, 80)
    Image.fromarray(img).resize((W // 2, H // 2), Image.LANCZOS).save(path)
