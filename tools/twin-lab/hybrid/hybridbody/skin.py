"""Procedural body skin at the photographed tone, painted underwear, and the head/neck tone hand-over.

The optional pinned CC0 MakeHuman skin supplies anatomy and colour variation,
retinted to the measured photo Lab. Fine world-space grain and pores also work
without the asset. UV borders are repaired before their final colours are padded.
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
    *,
    source=None,
    normal_canvas: np.ndarray | None = None,
    neck_y: float | None = None,
) -> dict:
    """Write skin (and underwear) into ``canvas`` (size x size x 3, sRGB uint8) at every covered texel."""
    boxer_texels = 0
    source_gradients = None
    if source is not None:
        source_lab, reference = source
        height = source_lab[..., 0] - ndimage.gaussian_filter(source_lab[..., 0], 5.0)
        source_gradients = np.gradient(height)
    # Per-triangle UV tangent frames, including mirrored charts. glTF V runs
    # downward: the normal map's green axis follows increasing V.
    tri_p, tri_uv = positions[faces], uv[faces]
    duv1, duv2 = tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]
    edge1, edge2 = tri_p[:, 1] - tri_p[:, 0], tri_p[:, 2] - tri_p[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    safe = np.where(np.abs(det) > 1e-12, det, 1.0)
    tangent = (edge1 * duv2[:, 1, None] - edge2 * duv1[:, 1, None]) / safe[:, None]
    bitangent = (edge2 * duv1[:, 0, None] - edge1 * duv2[:, 0, None]) / safe[:, None]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent -= tangent * np.einsum("ij,ij->i", tangent, bitangent)[:, None]
    bitangent /= np.maximum(np.linalg.norm(bitangent, axis=1, keepdims=True), 1e-12)
    directions = np.random.default_rng(71).normal(size=(12, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions *= np.linspace(5000, 11000, 12)[:, None]
    for r0, y, x, face, bary in rasterize_bands(uv, faces, size):
        if len(y) == 0:
            continue
        tri = faces[face]
        points = np.einsum("ij,ijk->ik", bary, positions[tri])
        lab = skin_lab(points, tone_lab)
        detail_weight = np.ones(len(points)) if neck_y is None else smoothstep((neck_y - points[:, 1]) / 0.018)
        coords = None
        if source is not None:
            coords = [(r0 + y + 0.5) * source_lab.shape[0] / size - 0.5,
                      (x + 0.5) * source_lab.shape[1] / size - 0.5]
            sampled = np.column_stack([ndimage.map_coordinates(source_lab[..., k], coords, order=1, mode="nearest")
                                       for k in range(3)])
            # Preserve anatomy and fine variation, without transferring the
            # source's illumination or the colour of unrelated chart gutters.
            detail = np.clip(sampled - reference, [-9, -3, -3], [9, 3, 3])
            lab += detail * detail_weight[:, None]
        mask = np.zeros(len(points))
        if underwear is not None:
            g = np.einsum("ij,ij->i", bary, gate[tri])
            mask = underwear_mask(points, g, underwear)
            cloth = underwear_lab(points, mask, underwear)
            lab = lab * (1 - mask[:, None]) + cloth * mask[:, None]
            boxer_texels += int((mask > 0.5).sum())
        canvas[r0 + y, x] = np.clip(np.rint(from_lab(lab) * 255), 1, 255).astype(np.uint8)
        if normal_canvas is not None:
            gradient = (np.cos(points @ directions.T) @ directions) * (0.000006 / len(directions))
            tilt = np.column_stack((-np.einsum("ij,ij->i", gradient, tangent[face]),
                                    -np.einsum("ij,ij->i", gradient, bitangent[face])))
            if source_gradients is not None:
                gy, gx = [ndimage.map_coordinates(g, coords, order=1, mode="nearest") for g in source_gradients]
                tilt += np.column_stack((-gx, -gy)) * 0.018 * detail_weight[:, None]
            tilt *= (1 - mask[:, None])
            normals = np.column_stack((np.clip(tilt, -0.12, 0.12), np.ones(len(points))))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            normal_canvas[r0 + y, x] = np.rint((normals * 0.5 + 0.5) * 255).astype(np.uint8)
        covered[r0 + y, x] = True
    return {"skin_lab": [float(v) for v in tone_lab], "underwear_texels": boxer_texels}


def pad_texture(canvas: np.ndarray, covered: np.ndarray, pixels: int = 12) -> np.ndarray:
    """Bleed chart colours outward so mipmaps and bilinear taps never mix in unrelated colours."""
    out = canvas.copy()
    if not covered.any():
        return out
    distance, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    ring = ~covered & (distance <= pixels)
    out[ring] = canvas[iy[ring], ix[ring]]
    far = ~covered & ~ring
    if far.any():
        # Unused atlas space: the mean covered colour keeps mip levels clean.
        out[far] = np.rint(canvas[covered].mean(0)).astype(np.uint8)
    return out


def border_audit(canvas: np.ndarray, covered: np.ndarray, pixels: int = 12) -> dict:
    """Count anomalous chart-border texels against nearby interior colour.

    Report inside light outliers separately from stale outside gutters; only
    padding defects are repaired automatically, preserving photographed detail.
    Thresholds are Lab dE76 > 6 and dL > 4 for inside light outliers, dE > 2
    for gutters. Interior references are at least three pixels inside a chart.
    """
    if not covered.any():
        return {"inside_light_outliers": 0, "gutter_outliers": 0, "border_texels": 0}
    inside_distance = ndimage.distance_transform_edt(covered)
    interior = inside_distance >= 3
    border = covered & (inside_distance < 2)
    _, (iy, ix) = ndimage.distance_transform_edt(~interior, return_indices=True)
    y, x = np.nonzero(border)
    lab = to_lab(canvas[y, x, :3])
    reference = to_lab(canvas[iy[y, x], ix[y, x], :3])
    delta = lab - reference
    inside_bad = (np.linalg.norm(delta, axis=1) > 6) & (delta[:, 0] > 4)
    distance, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    y, x = np.nonzero(~covered & (distance <= pixels))
    delta = to_lab(canvas[y, x, :3]) - to_lab(canvas[iy[y, x], ix[y, x], :3])
    gutter_bad = np.linalg.norm(delta, axis=1) > 2
    return {"inside_light_outliers": int(inside_bad.sum()), "gutter_outliers": int(gutter_bad.sum()),
            "border_texels": int(border.sum()), "gutter_texels": int(len(y)), "padding_px": pixels,
            "thresholds": {"inside_de76": 6, "inside_dL": 4, "gutter_de76": 2}}


def repair_border_colours(canvas: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, dict]:
    """Replace isolated light border contamination and black fill gaps.

    Only the first chart texel ring is eligible; colour more than three pixels
    into the island supplies the reference. Photo features within the island
    remain untouched. This fixes propagation errors that outward padding alone
    faithfully copies into the gutter.
    """
    out = canvas.copy()
    distance = ndimage.distance_transform_edt(covered)
    interior = distance >= 3
    if not interior.any():
        return out, {"light_texels_repaired": 0, "fill_gaps_repaired": 0}
    _, (iy, ix) = ndimage.distance_transform_edt(~interior, return_indices=True)
    y, x = np.nonzero(covered & (distance < 2))
    lab = to_lab(canvas[y, x, :3])
    reference = to_lab(canvas[iy[y, x], ix[y, x], :3])
    delta = lab - reference
    light = (np.linalg.norm(delta, axis=1) > 6) & (delta[:, 0] > 4)
    gaps = (lab[:, 0] < 5) & (reference[:, 0] > 25)
    bad = light | gaps
    out[y[bad], x[bad]] = canvas[iy[y[bad], x[bad]], ix[y[bad], x[bad]]]
    return out, {"light_texels_repaired": int(light.sum()), "fill_gaps_repaired": int(gaps.sum())}


def finish_texture(canvas: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Repair the skin chart boundary, then dilate its final colours outward."""
    repaired, _ = repair_border_colours(canvas, covered)
    return pad_texture(repaired, covered)


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


def fade_to_tone(
    texture: np.ndarray, texel_y: np.ndarray, texel_x: np.ndarray, weight: np.ndarray, tone_lab
) -> np.ndarray:
    """Mix the listed texels toward the flat body tone: ``weight`` 1 = body tone, 0 = unchanged."""
    out = texture.copy()
    lab = to_lab(texture[texel_y, texel_x])
    mixed = lab * (1 - weight[:, None]) + np.asarray(tone_lab, np.float32) * weight[:, None]
    out[texel_y, texel_x] = np.clip(np.rint(from_lab(mixed) * 255), 1, 255).astype(np.uint8)
    return out


def neck_luminance(texture, texel_y, texel_x, points, selection: np.ndarray, tone_lab) -> dict:
    """Mean Lab of the selected texels versus the body skin tone (numbers only)."""
    if not selection.any():
        return {"texels": 0}
    lab = to_lab(texture[texel_y[selection], texel_x[selection]]).astype(np.float64)
    mean = lab.mean(0)
    return {
        "texels": int(selection.sum()),
        "mean_lab": mean.tolist(),
        "delta_l_vs_body": float(mean[0] - tone_lab[0]),
        "delta_e76_vs_body": float(np.linalg.norm(mean - np.asarray(tone_lab, np.float64))),
        "min_l": float(lab[:, 0].min()),
        "p05_l": float(np.percentile(lab[:, 0], 5)),
        "y_range_m": [float(points[selection, 1].min()), float(points[selection, 1].max())],
    }


def sample_hair_surface(positions: np.ndarray, faces: np.ndarray, spacing: float = 0.003) -> np.ndarray:
    """Points spread over the triangles of a part (about ``spacing`` metres apart), vertices included."""
    tri = positions[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    counts = np.clip(np.ceil(area / (spacing * spacing * 0.5)).astype(int), 1, 64)
    rng = np.random.default_rng(5)
    owner = np.repeat(np.arange(len(faces)), counts)
    r = rng.random((len(owner), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    samples = tri[owner, 0] + r[:, :1] * (tri[owner, 1] - tri[owner, 0]) + r[:, 1:] * (tri[owner, 2] - tri[owner, 0])
    return np.vstack((samples, positions))


def hair_cover(
    points: np.ndarray,
    normals: np.ndarray,
    hair_samples: np.ndarray,
    *,
    reach: tuple = (0.002, 0.006, 0.012, 0.02, 0.03, 0.04),
    near: float = 0.004,
) -> np.ndarray:
    """0..1 per texel: how much hair stands above the texel, looking outward along its normal (a short ray march)."""
    tree = cKDTree(hair_samples)
    nearest = np.full(len(points), np.inf)
    for t in reach:
        nearest = np.minimum(nearest, tree.query(points + normals * t, workers=-1)[0])
    return 1.0 - smoothstep((nearest - near) / near)


def tint_scalp(
    texture: np.ndarray,
    texel_y: np.ndarray,
    texel_x: np.ndarray,
    cover: np.ndarray,
    tint_lab,
    blur_px: float = 2.0,
) -> np.ndarray:
    """Darken the texels under the hair toward ``tint_lab`` (gaps between cut-out cards then read as hair)."""
    h, w = texture.shape[:2]
    weight = np.zeros((h, w), np.float32)
    weight[texel_y, texel_x] = cover
    support = np.zeros((h, w), np.float32)
    support[texel_y, texel_x] = 1.0
    soft = ndimage.gaussian_filter(weight, blur_px) / np.maximum(ndimage.gaussian_filter(support, blur_px), 1e-3)
    alpha = np.clip(soft[texel_y, texel_x], 0, 1)
    return fade_to_tone(texture, texel_y, texel_x, alpha, tint_lab)
