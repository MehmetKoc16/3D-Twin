"""Procedural body skin at the photographed tone, painted underwear, and the head/neck tone hand-over.

The MakeHuman distribution used here ships no skin texture, so the body albedo is a flat Lab tone (the measured
photo skin tone) with fine 3-D grain. It is painted straight into the template's fixed UV layout from the texel's
own 3-D position, so islands never disagree at their seams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from flamehead.colour import from_lab, to_lab
from scipy import ndimage
from scipy.spatial import cKDTree
from twintex.raster import rasterize_uv

from .register import smoothstep


@dataclass
class Underwear:
    bottom_y: float
    top_y: float
    colour_lab: tuple = (36.0, 1.5, -7.0)  # dark slate cotton
    band_height: float = 0.022


def rasterize_bands(uv: np.ndarray, faces: np.ndarray, size: int, rows: int = 512):
    """Yield ``(row0, y, x, face_ids, bary)`` for all covered texels of a square atlas, band by band."""
    for r0 in range(0, size, rows):
        r1 = min(size, r0 + rows)
        fid, bary = rasterize_uv(uv * size, faces, size, size, row_range=(r0, r1))
        y, x = np.nonzero(fid >= 0)
        yield r0, y, x, fid[y, x], bary[y, x]


def _grain(points: np.ndarray, seed: int = 31):
    rng = np.random.default_rng(seed)
    grain = np.zeros(len(points))
    freckles = np.zeros(len(points))
    for frequency, amplitude in ((22, 0.55), (80, 0.25), (240, 0.12), (900, 0.07)):
        direction = rng.normal(size=(3, 3)) * frequency
        noise = np.sin(points @ direction + rng.uniform(0, 2 * np.pi, 3)).mean(1)
        grain += noise * amplitude
        freckles += np.maximum(noise - 0.5, 0) * 0.08
    return grain, freckles


def skin_lab(points: np.ndarray, tone_lab, seed: int = 31) -> np.ndarray:
    grain, freckles = _grain(points, seed)
    lab = np.tile(np.asarray(tone_lab, np.float32), (len(points), 1))
    lab[:, 0] += grain - freckles
    lab[:, 1] += grain * 0.13
    lab[:, 2] += grain * 0.17
    return lab


def underwear_mask(points: np.ndarray, gate: np.ndarray, spec: Underwear) -> np.ndarray:
    y = points[:, 1]
    inside = smoothstep((y - spec.bottom_y) / 0.003) * (1 - smoothstep((y - spec.top_y) / 0.003))
    return inside * smoothstep((gate - 0.35) / 0.3)


def underwear_lab(points: np.ndarray, mask: np.ndarray, spec: Underwear) -> np.ndarray:
    """Fabric colour with a fine weave, a darker waistband and a darker hem line."""
    lab = np.tile(np.asarray(spec.colour_lab, np.float32), (len(points), 1))
    weave = np.sin(points[:, 0] * 2300) * np.sin(points[:, 1] * 2300) + np.sin(points[:, 2] * 2300) * 0.5
    lab[:, 0] += weave * 0.9
    y = points[:, 1]
    lab[:, 0] -= 7.0 * smoothstep((y - (spec.top_y - spec.band_height)) / 0.002)
    lab[:, 0] -= 5.0 * (1 - smoothstep((y - (spec.bottom_y + 0.003)) / 0.0015))
    return lab


def paint_body(
    canvas: np.ndarray,
    covered: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    gate: np.ndarray,
    tone_lab,
    underwear: Underwear | None,
    size: int,
) -> dict:
    """Write skin (and underwear) into ``canvas`` (size x size x 3, sRGB uint8) at every covered texel."""
    boxer_texels = 0
    for r0, y, x, face, bary in rasterize_bands(uv, faces, size):
        if len(y) == 0:
            continue
        tri = faces[face]
        points = np.einsum("ij,ijk->ik", bary, positions[tri])
        lab = skin_lab(points, tone_lab)
        if underwear is not None:
            g = np.einsum("ij,ij->i", bary, gate[tri])
            mask = underwear_mask(points, g, underwear)
            cloth = underwear_lab(points, mask, underwear)
            lab = lab * (1 - mask[:, None]) + cloth * mask[:, None]
            boxer_texels += int((mask > 0.5).sum())
        canvas[r0 + y, x] = np.clip(np.rint(from_lab(lab) * 255), 1, 255).astype(np.uint8)
        covered[r0 + y, x] = True
    return {"skin_lab": [float(v) for v in tone_lab], "underwear_texels": boxer_texels}


def pad_texture(canvas: np.ndarray, covered: np.ndarray, pixels: int = 12) -> np.ndarray:
    """Bleed chart colours outward so mipmaps and bilinear taps never mix in unrelated colours."""
    out = canvas.copy()
    distance, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    ring = ~covered & (distance <= pixels)
    out[ring] = canvas[iy[ring], ix[ring]]
    far = ~covered & ~ring
    if far.any():
        # Unused atlas space: the mean covered colour keeps mip levels clean.
        out[far] = np.rint(canvas[covered].mean(0)).astype(np.uint8)
    return out


def match_mean(texture: np.ndarray, covered: np.ndarray, samples: np.ndarray, target_lab, limit: float = 3.0):
    """Shift the texture's Lab so the mean of ``samples`` (boolean mask) lands on ``target_lab`` (|shift| <= limit)."""
    lab = to_lab(texture)
    current = lab[samples].mean(0, dtype=np.float64)
    shift = np.clip(np.asarray(target_lab, np.float64) - current, -limit, limit).astype(np.float32)
    protect = np.clip(lab[..., 0] / 30.0, 0, 1) ** 2  # keep hair, brows and glasses black
    lab[..., 0] += shift[0] * protect
    lab[..., 1:] += shift[1:] * protect[..., None]
    out = np.clip(np.rint(from_lab(lab) * 255), 1, 255).astype(np.uint8)
    out[~covered] = texture[~covered]
    return out, {"before": current.tolist(), "shift": shift.tolist()}


def seam_blend(
    texture: np.ndarray,
    points: np.ndarray,
    texel_y: np.ndarray,
    texel_x: np.ndarray,
    seam_points: np.ndarray,
    tone_lab,
    radius: float = 0.018,
) -> tuple[np.ndarray, dict]:
    """Fade the head texture to the body tone within ``radius`` of the head/torso seam (the neck)."""
    distance = cKDTree(seam_points).query(points)[0]
    alpha = smoothstep(distance / radius)
    out = texture.copy()
    lab = to_lab(texture[texel_y, texel_x])
    mixed = lab * alpha[:, None] + np.asarray(tone_lab, np.float32) * (1 - alpha[:, None])
    out[texel_y, texel_x] = np.clip(np.rint(from_lab(mixed) * 255), 1, 255).astype(np.uint8)
    return out, {"radius_mm": radius * 1000, "blended_texels": int((alpha < 1).sum())}
