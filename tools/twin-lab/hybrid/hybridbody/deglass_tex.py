"""Remove only painted frame lines in head UV space, never load source photos.

``FrameProjection`` consumes FaceBake's exact texel coordinates and deformed
template positions (the same barycentric mapping used for baking). Rims/bridge
project along +Z, arms project laterally onto the temple skin. A narrow line mask plus
texture black/white top-hat detections in a wider corridor captures displaced
thin dark metal and highlights. Eye interiors and brows are protected explicitly.
OpenCV Telea/NS extends neighbouring texture across these small strips. This
does not reconstruct hidden anatomy or remove lens reflections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class FrameProjection:
    texel_points: np.ndarray
    texel_y: np.ndarray
    texel_x: np.ndarray
    front_curves: np.ndarray
    temple_curves: np.ndarray
    eyes: np.ndarray
    radii: np.ndarray
    covered: np.ndarray | None = None
    protected: np.ndarray | None = None


class BakedTexels(Protocol):
    texel_points: np.ndarray
    texel_y: np.ndarray
    texel_x: np.ndarray
    covered: np.ndarray


def projection_from_report(face: BakedTexels, report: dict) -> FrameProjection:
    """Pipeline hook using the private fitted report's ORIGINAL removal guides."""
    eyes = np.asarray(report["placement"]["eyes_m"], float)
    protected = np.zeros(face.covered.shape, bool)
    high = face.texel_points[:, 1] > eyes[:, 1].mean() + 0.022
    protected[face.texel_y[high], face.texel_x[high]] = True
    return FrameProjection(
        face.texel_points,
        face.texel_y,
        face.texel_x,
        np.asarray(report["removal_curves_m"]["front"], float),
        np.asarray(report["removal_curves_m"]["temples"], float),
        eyes,
        np.asarray(report["placement"]["rim_radii_mm"], float) / 1000,
        face.covered,
        protected,
    )


def projected_frame_mask(
    texture: np.ndarray,
    geometry: FrameProjection,
    *,
    line_width_m: float = 0.0016,
    search_width_m: float = 0.005,
    contrast: float = 9.0,
) -> tuple[np.ndarray, dict]:
    """Project the curves onto ONLY mapped front skin texels; v is glTF v-down."""
    h, w = texture.shape[:2]
    points = np.asarray(geometry.texel_points, float)
    y, x = np.asarray(geometry.texel_y), np.asarray(geometry.texel_x)
    if points.shape != (len(y), 3) or len(x) != len(y) or not np.isfinite(points).all():
        raise ValueError("Invalid baked texel mapping")
    if np.any((y < 0) | (y >= h) | (x < 0) | (x >= w)):
        raise ValueError("Texel mapping outside texture")
    front = np.asarray(geometry.front_curves, float)
    if front.ndim != 2 or front.shape[1] != 3 or len(front) < 2 or not np.isfinite(front).all():
        raise ValueError("At least two finite frame curve samples required")
    distance, near = cKDTree(front[:, :2]).query(points[:, :2])
    eye_z = np.asarray(geometry.eyes)[:, 2].mean()
    # Prevent the frame's orthographic projection appearing on the back of the head.
    visible = (points[:, 2] > eye_z - 0.004) & (points[:, 2] < front[near, 2] + 0.012)
    if len(geometry.temple_curves):
        arm = cKDTree(geometry.temple_curves[:, 1:]).query(points[:, 1:])[0]
        side = np.abs(points[:, 0] - np.asarray(geometry.eyes)[:, 0].mean()) > 0.052
        arm = np.where(side, arm, np.inf)
        distance = np.minimum(np.where(visible, distance, np.inf), arm)
    else:
        distance = np.where(visible, distance, np.inf)
    valid = np.ones((h, w), bool) if geometry.covered is None else np.asarray(geometry.covered, bool).copy()
    if valid.shape != (h, w):
        raise ValueError("Coverage must match texture")
    protected = np.zeros((h, w), bool)
    for eye in np.asarray(geometry.eyes):
        # Preserve the actual eyelid/eye patch, not the entire inside of a lens.
        inner = ((points[:, 0] - eye[0]) / 0.016) ** 2 + ((points[:, 1] - eye[1]) / 0.007) ** 2 < 1
        protected[y[inner], x[inner]] = True
    if geometry.protected is not None:
        if geometry.protected.shape != (h, w):
            raise ValueError("Protection mask must match texture")
        protected |= geometry.protected
    corridor, core = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    corridor[y[distance < search_width_m], x[distance < search_width_m]] = 255
    core[y[distance < line_width_m / 2], x[distance < line_width_m / 2]] = 255
    gray = cv2.cvtColor(texture[..., :3], cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    light = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    upper = np.zeros((h, w), bool)
    above = points[:, 1] > np.asarray(geometry.eyes)[:, 1].mean() + 0.004
    upper[y[above], x[above]] = True
    rgb = texture[..., :3].astype(np.int16)
    neutral = rgb.max(2) - rgb.min(2) < 40
    # Over the brows, only a bright neutral metal highlight is safe to erase.
    # A mandatory geometric stripe or black-hat hairs here can destroy eyebrows.
    detected = (np.maximum(dark, light) >= contrast) & ~upper
    protected |= upper & (gray < 145)
    detected |= upper & (light >= max(12, contrast)) & neutral & (gray >= 145)
    detected &= corridor > 0
    mask = (((core > 0) & ~upper) | detected) & valid & ~protected
    return mask, {
        "method": "bake-texel frame projection + corridor black/white top-hat",
        "projected_core_texels": int(((core > 0) & valid & ~protected).sum()),
        "detected_line_texels": int((detected & valid & ~protected).sum()),
        "corridor_texels": int(((corridor > 0) & valid & ~protected).sum()),
        "protected_texels": int(protected.sum()),
    }


def remove_glasses_frames(
    texture: np.ndarray,
    uv_mask_or_geometry: np.ndarray | FrameProjection,
    *,
    dilate_texels: int = 2,
    radius: float = 3.0,
    method: str = "telea",
    return_report: bool = False,
):
    """Return a copy with frame pixels inpainted; preserve alpha and every other pixel.

    Accepts an explicit bool/uint8 UV mask or ``FrameProjection``. With a geometry
    mapping, dilation is clipped to head coverage and the protected eye pixels.
    ``return_report=True`` returns ``(texture, report, final_mask)`` for QA.
    """
    if texture.dtype != np.uint8 or texture.ndim != 3 or texture.shape[2] not in (3, 4):
        raise ValueError("Texture must be uint8 RGB or RGBA")
    if not isinstance(dilate_texels, int) or not 0 <= dilate_texels <= 16:
        raise ValueError("Dilation must be 0..16 texels")
    if not np.isfinite(radius) or not 0 < radius <= 20 or method not in ("telea", "ns"):
        raise ValueError("Invalid inpainting method/radius")
    report = {"method": "explicit UV mask"}
    if isinstance(uv_mask_or_geometry, FrameProjection):
        mask, report = projected_frame_mask(texture, uv_mask_or_geometry)
    else:
        mask = np.asarray(uv_mask_or_geometry)
        if mask.shape != texture.shape[:2] or mask.dtype.kind not in "bu":
            raise ValueError("UV mask must be bool/uint8 and match texture")
        mask = mask > 0
    initial = mask.copy()
    if dilate_texels:
        n = 2 * dilate_texels + 1
        mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))) > 0
    if isinstance(uv_mask_or_geometry, FrameProjection):
        g = uv_mask_or_geometry
        if g.covered is not None:
            mask &= g.covered
        # Re-evaluate eye protection after dilation.
        for eye in g.eyes:
            inner = ((g.texel_points[:, 0] - eye[0]) / 0.016) ** 2 + ((g.texel_points[:, 1] - eye[1]) / 0.007) ** 2 < 1
            mask[g.texel_y[inner], g.texel_x[inner]] = False
        if g.protected is not None:
            mask &= ~g.protected
        above = g.texel_points[:, 1] > g.eyes[:, 1].mean()+.004
        by, bx = g.texel_y[above], g.texel_x[above]
        gray = cv2.cvtColor(texture[..., :3], cv2.COLOR_RGB2GRAY)
        mask[by, bx] &= gray[by, bx] >= 145
    out = texture.copy()
    if mask.any():
        out[..., :3] = cv2.inpaint(
            np.ascontiguousarray(texture[..., :3]),
            mask.astype(np.uint8) * 255,
            radius,
            cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS,
        )
    report.update(
        inpaint=method,
        radius_texels=radius,
        dilation_texels=dilate_texels,
        initial_mask_texels=int(initial.sum()),
        mask_texels=int(mask.sum()),
        fraction_of_texture=float(mask.mean()),
        changed_texels=int(np.any(out[..., :3] != texture[..., :3], axis=2).sum()),
        outside_mask_changed=bool(np.any(out[~mask] != texture[~mask])),
        limitation="frame strips only; hidden anatomy and lens reflections are not reconstructed",
    )
    return (out, report, mask) if return_report else out
