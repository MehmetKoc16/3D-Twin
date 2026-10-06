"""Remove only painted frame lines in head UV space, never load source photos.

``FrameProjection`` consumes FaceBake's exact texel coordinates and deformed
template positions (the same barycentric mapping used for baking). Rims/bridge
project along +Z, arms project laterally onto the temple skin. A narrow line mask plus
texture black/white top-hat detections in a wider corridor captures displaced
thin dark metal and highlights. Eye interiors and brows are protected explicitly.
OpenCV Telea/NS extends neighbouring texture across these small strips. This
does not reconstruct hidden anatomy.

``remove_lens_residue`` handles what the thin-line pass cannot: the photographed lens AREA (tint, reflections,
brightness shift inside each rim; corrected as a smooth Lab gain/offset field measured across the rim, never inpainted)
and the temple-arm streak (NS inpainting along a thin band to the ear). Eyes, eyelids, lashes and brows are protected
and verified unchanged.
"""

from __future__ import annotations

import warnings
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


# ---------------------------------------------------------------------------------------------------------------------
# Lens area and temple arm residue
# ---------------------------------------------------------------------------------------------------------------------

_SECTORS = 24


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0), 0, 255).astype(np.uint8)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _dense_positions(g: FrameProjection, shape: tuple[int, int]) -> np.ndarray:
    dense = np.full((*shape, 3), np.nan, np.float32)
    dense[g.texel_y, g.texel_x] = g.texel_points
    return dense


def texel_size_mm(dense: np.ndarray) -> float:
    """Median 3D spacing of adjacent mapped texels, in mm."""
    steps = []
    for diff in (dense[:, 1:] - dense[:, :-1], dense[1:] - dense[:-1]):
        d = np.linalg.norm(diff, axis=2)
        d = d[np.isfinite(d)]
        if len(d):
            steps.append(np.median(d))
    return float(np.mean(steps) * 1000.0) if steps else 1.0


def front_facing(dense: np.ndarray, valid: np.ndarray, cell: float = 0.002, depth: float = 0.008) -> np.ndarray:
    """Texels seen along -Z: within ``depth`` of the front-most surface of their 2 mm XY cell (drops the back of the head)."""
    X, Y, Z = dense[..., 0], dense[..., 1], dense[..., 2]
    out = np.zeros(valid.shape, bool)
    gx = np.floor(X[valid] / cell).astype(np.int64)
    gy = np.floor(Y[valid] / cell).astype(np.int64)
    key = (gx - gx.min()) * (gy.max() - gy.min() + 1) + (gy - gy.min())
    z = Z[valid]
    top = np.full(int(key.max()) + 1, -np.inf)
    np.maximum.at(top, key, z)
    out[valid] = z >= top[key] - depth
    return out


def rim_ellipses(g: FrameProjection) -> np.ndarray:
    """Per eye ``(cx, cy, rx, ry)`` in metres, sorted by x. Uses the measured rim outlines when the report has them."""
    front = np.asarray(g.front_curves, float)
    if len(front) >= 720:
        rows = []
        for block in (front[:360], front[360:720]):
            lo, hi = block[:, :2].min(0), block[:, :2].max(0)
            rows.append([*(lo + hi) / 2, *(hi - lo) / 2])
        out = np.array(rows)
    else:
        radii = np.asarray(g.radii, float).ravel()
        rx, ry = (radii[0], radii[-1]) if len(radii) else (0.025, 0.025)
        out = np.array([[e[0], e[1], rx, ry] for e in np.asarray(g.eyes, float)])
    return out[np.argsort(out[:, 0])]


def _sector_values(sector: np.ndarray, values: np.ndarray, count: int = _SECTORS) -> tuple[np.ndarray, np.ndarray]:
    med = np.full(count, np.nan)
    num = np.zeros(count, int)
    for k in range(count):
        v = values[sector == k]
        num[k] = len(v)
        if len(v):
            med[k] = np.median(v)
    return med, num


def _circular_fill(values: np.ndarray, fallback: float) -> np.ndarray:
    ok = np.isfinite(values)
    if not ok.any():
        return np.full_like(values, fallback)
    if ok.sum() == 1:
        return np.full_like(values, values[ok][0])
    idx = np.arange(len(values))
    return np.interp(idx, np.r_[idx[ok] - len(values), idx[ok], idx[ok] + len(values)], np.r_[values[ok], values[ok], values[ok]])


def _circular_smooth(values: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        values = 0.25 * np.roll(values, 1) + 0.5 * values + 0.25 * np.roll(values, -1)
    return values


def _periodic_interp(theta: np.ndarray, values: np.ndarray) -> np.ndarray:
    n = len(values)
    pos = np.nan_to_num((theta + np.pi) / (2 * np.pi) * n - 0.5)
    lo = np.floor(pos).astype(int)
    f = pos - lo
    return values[lo % n] * (1 - f) + values[(lo + 1) % n] * f


def protection_masks(
    texture: np.ndarray,
    g: FrameProjection,
    *,
    eye_region: np.ndarray | None = None,
    dense: np.ndarray | None = None,
    texel_mm: float | None = None,
) -> dict:
    """Eyes / eyelids / lashes (FLAME eye regions inside an ellipse) and detected brows.

    Returns bool masks ``eyes``, ``brows``, ``protected`` (union, dilated ~1.5-2 mm) and a float ``keep`` weight that is
    0 inside ``protected`` and ramps to 1 over ~2 mm (a feather for corrections).
    """
    h, w = texture.shape[:2]
    dense = _dense_positions(g, (h, w)) if dense is None else dense
    texel_mm = texel_size_mm(dense) if texel_mm is None else texel_mm
    X, Y = dense[..., 0], dense[..., 1]
    lab = _lab(texture)
    valid = np.isfinite(X)
    if g.covered is not None:
        valid &= g.covered
    eyes = np.zeros((h, w), bool)
    loose = np.zeros((h, w), bool)
    eye_y = float(np.asarray(g.eyes)[:, 1].mean())
    for e in np.asarray(g.eyes, float):
        with np.errstate(invalid="ignore"):
            core = (((X - e[0]) / 0.016) ** 2 + ((Y - e[1]) / 0.007) ** 2) < 1
            wide = (((X - e[0]) / 0.018) ** 2 + ((Y - e[1]) / 0.0095) ** 2) < 1
            disc = (((X - e[0]) / 0.034) ** 2 + ((Y - e[1] - 0.022) / 0.016) ** 2) < 1  # brow zone, not the sideburn
        eyes |= core | (wide & (eye_region if eye_region is not None else True))  # lids and lashes, nothing wider
        loose |= disc
    eyes &= valid
    # Brows: dark texels (hair) above the eye line, against the skin level of the same neighbourhood.
    with np.errstate(invalid="ignore"):
        upper = valid & loose & (Y > eye_y + 0.008)
    skin_l = float(np.median(lab[..., 0][upper])) if upper.any() else 0.0
    dark = upper & (lab[..., 0] < skin_l - 10.0)
    if g.protected is not None:
        dark |= g.protected & valid & loose & (lab[..., 0] < skin_l - 6.0)
    # a thin dark wire is not a brow: only dark MASSES (wider than ~2 mm) count, so rim traces stay removable
    n_open = 2 * max(1, int(round(1.2 / texel_mm))) + 1
    brows = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n_open, n_open))) > 0

    def grow(mask, mm):
        n = 2 * max(1, int(round(mm / texel_mm))) + 1
        return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))) > 0

    protected = grow(eyes, 1.5) | grow(brows, 2.5)
    soft = cv2.GaussianBlur(protected.astype(np.float32), (0, 0), max(0.6, 1.5 / texel_mm))
    keep = np.where(protected, 0.0, 1.0 - np.clip(soft * 1.6, 0.0, 1.0)).astype(np.float32)
    return {"eyes": eyes, "brows": brows, "protected": protected, "keep": keep}


def _inliers(lab: np.ndarray, mask: np.ndarray, k: float = 3.0, floor: float = 4.0) -> np.ndarray:
    """Texels of ``mask`` whose Lab lies within k robust sigmas of the median (drops lashes, hair, glints, shadows)."""
    out = mask.copy()
    values = lab[mask]
    med = np.median(values, axis=0)
    sigma = np.maximum(1.4826 * np.median(np.abs(values - med), axis=0), [floor, floor / 2, floor / 2])
    ok = np.all(np.abs(lab - med) <= k * sigma, axis=2)
    return out & ok


_ANGLE_BINS = 360


def _polar(g_dense: np.ndarray, ellipse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance (mm) from the fitted outline along the radius, and the polar angle, per texel."""
    cx, cy, rx, ry = ellipse
    with np.errstate(invalid="ignore"):
        u, v = (g_dense[..., 0] - cx) / rx, (g_dense[..., 1] - cy) / ry
        return (np.hypot(u, v) - 1.0) * np.sqrt(rx * ry) * 1000.0, np.arctan2(v, u)


def track_rims(
    reference: np.ndarray,
    g: FrameProjection,
    ellipses: np.ndarray,
    *,
    visible: np.ndarray,
    reach_mm: float = 5.0,
    min_ridge: float = 3.0,
    min_good: float = 0.35,
) -> list[dict]:
    """Snap each fitted rim outline onto the photographed wire (thin line) of ``reference`` (the texture BEFORE the frame
    strips were inpainted), by a Viterbi ridge track around the outline. Returns per eye ``offset_mm`` (360 angle bins,
    radial shift of the true wire against the fitted outline), ``good_fraction`` and ``tracked`` (False = nominal kept)."""
    h, w = reference.shape[:2]
    dense = _dense_positions(g, (h, w))
    lab = _lab(reference)
    out = []
    for e in ellipses:
        rho, theta = _polar(dense, e)
        abin = np.where(visible & np.isfinite(rho), np.floor((np.nan_to_num(theta) + np.pi) / (2 * np.pi) * _ANGLE_BINS), -1)
        abin = abin.astype(int).clip(-1, _ANGLE_BINS - 1)
        offset, ridge = _track_ridge(
            np.nan_to_num(rho, nan=1e9), lab, visible, abin, _ANGLE_BINS,
            reach_mm=reach_mm, step_mm=0.25, core_mm=0.7, flank_mm=(1.6, 3.5), jump_cost=1.6,
        )
        good = np.isfinite(offset) & (ridge >= min_ridge)
        frac = float(good.mean())
        if frac < min_good:
            out.append({"offset_mm": np.zeros(_ANGLE_BINS), "good_fraction": frac, "tracked": False, "good": good})
            continue
        idx = np.flatnonzero(good)
        track = np.interp(np.arange(_ANGLE_BINS), np.r_[idx - _ANGLE_BINS, idx, idx + _ANGLE_BINS], np.tile(offset[idx], 3))
        pad = np.r_[track[-8:], track, track[:8]]
        smooth = np.array([np.median(pad[i : i + 17]) for i in range(_ANGLE_BINS)])
        out.append({"offset_mm": smooth, "good_fraction": frac, "tracked": True, "good": good})
    return out


def _sector_inliers(lab: np.ndarray, mask: np.ndarray, sector: np.ndarray, k: float = 3.0) -> np.ndarray:
    """``_inliers`` per angular sector (a lens perimeter spans bright and dark skin; a global filter would drop one)."""
    out = np.zeros_like(mask)
    for sec in np.unique(sector[mask]):
        m = mask & (sector == sec)
        if m.sum() >= 10:
            out |= _inliers(lab, m, k=k)
        else:
            out |= m
    return out


def correct_lens_area(
    texture: np.ndarray,
    g: FrameProjection,
    prot: dict,
    *,
    ellipses: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    rims: list[dict] | None = None,
    skin_mask: np.ndarray | None = None,
    avoid: np.ndarray | None = None,
    inner_mm: tuple[float, float] = (2.5, 4.8),
    outer_mm: tuple[float, float] = (2.5, 4.8),
    feather_mm: tuple[float, float] = (-1.5, 1.5),
    decay_mm: float = 4.5,
    bridge_mm: float = 11.0,
    max_gain: float = 0.08,
    max_offset: float = 5.0,
    min_samples: int = 40,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Remove the lens shift as a smooth Lab gain (L) / offset (a, b) field. Returns texture, report, field weight.

    Per eye and per angular sector the shift is the step between a skin band just inside the rim (``inner_mm``) and a
    ring just outside it (``outer_mm``), both excluding eyes, brows, the frame strip (``avoid``) and dark outliers.
    The field equals the measured step within the inner band and decays inward with ``decay_mm`` (the pale band the
    lens leaves along the rim is a rim-hugging feature, not a lens-wide shift), and is feathered over ``feather_mm``
    around the rim outline, so no new edge appears. Only the part below the eye line is corrected (above it the rim
    runs over the brow bone, whose shading is anatomy, not lens). The ring skips the nose bridge (``bridge_mm`` either side of the
    midline), whose anatomy differs from the cheek.
    """
    h, w = texture.shape[:2]
    dense = _dense_positions(g, (h, w))
    X, Y, Z = dense[..., 0], dense[..., 1], dense[..., 2]
    valid = np.isfinite(X)
    if g.covered is not None:
        valid &= g.covered
    lab = _lab(texture)
    ell = rim_ellipses(g) if ellipses is None else ellipses
    visible = front_facing(dense, valid)
    bad = prot["protected"].copy()
    if avoid is not None:
        bad |= avoid
    mid_x = float(np.asarray(g.eyes)[:, 0].mean())
    eye_y = np.asarray(g.eyes, float)[np.argsort(np.asarray(g.eyes, float)[:, 0]), 1]  # per eye, same order as ``ell``
    sd, theta = [], []
    for cx, cy, rx, ry in ell:
        with np.errstate(invalid="ignore"):
            u, v = (X - cx) / rx, (Y - cy) / ry
            sd.append((np.hypot(u, v) - 1.0) * np.sqrt(rx * ry) * 1000.0)
            theta.append(np.arctan2(v, u))
    sd, theta = np.stack(sd), np.stack(theta)
    if rims is None:
        rims = track_rims(reference, g, ell, visible=visible) if reference is not None else []
    for i, rim in enumerate(rims):
        # distance from the SNAPPED wire instead of the fitted outline
        abin = np.floor((np.nan_to_num(theta[i]) + np.pi) / (2 * np.pi) * _ANGLE_BINS).astype(int).clip(0, _ANGLE_BINS - 1)
        sd[i] = sd[i] - rim["offset_mm"][abin]
    gain_field = np.ones((h, w), np.float32)
    off_a = np.zeros((h, w), np.float32)
    off_b = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    channels = {"L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2]}
    per_eye = []
    for i in range(len(ell)):
        others = np.delete(sd, i, axis=0)
        with np.errstate(invalid="ignore"):
            near_other = (others < 4.0).any(0) if len(others) else np.zeros((h, w), bool)
            base = visible & ~bad & np.isfinite(sd[i])
            interior = base & (sd[i] < -inner_mm[0])
            off_bridge = np.abs(X - mid_x) > bridge_mm / 1000.0
            ring_wide = base & (sd[i] >= 2.5) & (sd[i] < 8.0) & ~near_other & off_bridge  # "outer ring" for the totals
            ring = base & (sd[i] >= outer_mm[0]) & (sd[i] < outer_mm[1]) & ~near_other & off_bridge  # rim step band
        if skin_mask is not None:
            ring &= skin_mask
            ring_wide &= skin_mask
        if not interior.any() or not ring.any():
            per_eye.append({"applied": False, "reason": "no skin samples"})
            continue
        # drop dark outliers (lashes, iris edge, hair, shadows) from both sets
        interior &= _inliers(lab, interior)
        ring_wide &= _inliers(lab, ring_wide)
        with np.errstate(invalid="ignore"):
            band_in = interior & (sd[i] > -inner_mm[1])
            band_out = ring & (sd[i] <= outer_mm[1])
        sector = (np.nan_to_num((theta[i] + np.pi) / (2 * np.pi)) * _SECTORS).astype(int).clip(0, _SECTORS - 1)
        # the ring must lie on the same surface as the inner band (not on the nose wing or a crease): within 8 mm in depth
        z_in, _ = _sector_values(np.where(band_in, sector, -1), Z)
        with np.errstate(invalid="ignore"):
            same_surface = np.abs(Z - np.nan_to_num(z_in, nan=1e9)[sector]) < 0.008
        band_out &= same_surface
        band_in &= _sector_inliers(lab, band_in, sector)
        band_out &= _sector_inliers(lab, band_out, sector)
        sector_in, sector_out = np.where(band_in, sector, -1), np.where(band_out, sector, -1)
        steps = {}
        for key, c in channels.items():
            vin, nin = _sector_values(sector_in, c)
            vout, nout = _sector_values(sector_out, c)
            if key == "L":
                diag = {"inner_L": vin, "outer_L": vout, "inner_n": nin, "outer_n": nout}
            enough = (nin >= min_samples) & (nout >= min_samples)
            if key == "L":
                s = np.where(enough, vout / np.maximum(vin, 1e-3), np.nan)
                fallback = 1.0
            else:
                s = np.where(enough, vout - vin, np.nan)
                fallback = 0.0
            found = s[np.isfinite(s)]
            s = _circular_smooth(_circular_fill(s, float(np.median(found)) if len(found) else fallback))
            steps[key] = np.clip(s, 1 - max_gain, 1 + max_gain) if key == "L" else np.clip(s, -max_offset, max_offset)
        mm_rim = np.where(np.isfinite(sd[i]), sd[i], 99.0)
        feather = 1.0 - _smoothstep((mm_rim - feather_mm[0]) / (feather_mm[1] - feather_mm[0]))
        depth = np.maximum(0.0, -mm_rim)
        inward = np.exp(-np.maximum(0.0, depth - 0.5 * (inner_mm[0] + inner_mm[1])) / decay_mm)
        below_eye = 1.0 - _smoothstep((np.nan_to_num(Y, nan=9.0) - (eye_y[i] - 0.002)) / 0.008)
        w_eye = (feather * inward * below_eye * prot["keep"] * visible).astype(np.float32)
        w_eye = np.where(w_eye > 1e-4, w_eye, 0).astype(np.float32)
        use = w_eye > weight
        gain_field = np.where(use, 1.0 + (_periodic_interp(theta[i], steps["L"]) - 1.0) * w_eye, gain_field)
        off_a = np.where(use, _periodic_interp(theta[i], steps["a"]) * w_eye, off_a)
        off_b = np.where(use, _periodic_interp(theta[i], steps["b"]) * w_eye, off_b)
        weight = np.where(use, w_eye, weight)
        per_eye.append(
            {
                "applied": True,
                "center_mm": [float(ell[i][0] * 1000), float(ell[i][1] * 1000)],
                "radii_mm": [float(ell[i][2] * 1000), float(ell[i][3] * 1000)],
                "rim_snap": (
                    {
                        "tracked": rims[i]["tracked"],
                        "good_fraction": rims[i]["good_fraction"],
                        "offset_mm_median_abs": float(np.median(np.abs(rims[i]["offset_mm"]))),
                        "offset_mm_range": [float(rims[i]["offset_mm"].min()), float(rims[i]["offset_mm"].max())],
                    }
                    if rims
                    else None
                ),
                "interior_texels": int(interior.sum()),
                "ring_texels": int(ring_wide.sum()),
                "gain_L_by_sector": [float(v) for v in steps["L"]],
                "inner_L_by_sector": [None if not np.isfinite(v) else float(v) for v in diag["inner_L"]],
                "outer_L_by_sector": [None if not np.isfinite(v) else float(v) for v in diag["outer_L"]],
                "inner_n_by_sector": [int(v) for v in diag["inner_n"]],
                "outer_n_by_sector": [int(v) for v in diag["outer_n"]],
                "offset_a_by_sector": [float(v) for v in steps["a"]],
                "offset_b_by_sector": [float(v) for v in steps["b"]],
                "_masks": (interior, ring_wide, band_in, band_out),
            }
        )
    touched = weight > 1e-4
    new = lab.copy()
    new[..., 0] = np.where(touched, lab[..., 0] * gain_field, lab[..., 0])
    new[..., 1] = np.where(touched, lab[..., 1] + off_a, lab[..., 1])
    new[..., 2] = np.where(touched, lab[..., 2] + off_b, lab[..., 2])
    out = texture.copy()
    out[..., :3] = np.where(touched[..., None], _from_lab(new), texture[..., :3])
    after = _lab(out)
    for entry in per_eye:
        if not entry.get("applied"):
            continue
        interior, ring, band_in, band_out = entry.pop("_masks")

        def delta(img, a, b):
            return {k: float(np.median(img[..., j][a]) - np.median(img[..., j][b])) for j, k in enumerate("Lab")}

        entry["interior_vs_ring_before"] = delta(lab, interior, ring)
        entry["interior_vs_ring_after"] = delta(after, interior, ring)
        entry["rim_step_before"] = delta(lab, band_out, band_in)
        entry["rim_step_after"] = delta(after, band_out, band_in)
    report = {
        "method": "lens-area Lab gain (L) / offset (a, b) field from the rim step, feathered at the rim",
        "eyes": per_eye,
        "changed_texels": int(np.any(out[..., :3] != texture[..., :3], axis=2).sum()),
    }
    return out, report, weight


def _arm_paths(g: FrameProjection) -> list[tuple[int, np.ndarray]]:
    arms = np.asarray(g.temple_curves, float)
    if arms.ndim != 2 or len(arms) < 2:
        return []
    mid = float(np.asarray(g.eyes)[:, 0].mean())
    paths = []
    for sign in (-1, 1):
        part = arms[np.sign(arms[:, 0] - mid) == sign]
        if len(part) >= 2:
            paths.append((sign, part[np.argsort(part[:, 2])][:, 1:]))  # (y, z) sorted by z
    return paths


def _track_ridge(
    d: np.ndarray,
    lab: np.ndarray,
    ok: np.ndarray,
    zbin: np.ndarray,
    n_stations: int,
    *,
    reach_mm: float = 24.0,
    step_mm: float = 0.5,
    core_mm: float = 1.0,
    flank_mm: tuple[float, float] = (2.2, 5.0),
    jump_cost: float = 0.9,
    viterbi_from: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi ridge tracker across depth stations: per station the offset ``d`` (mm) of a thin strip that differs from
    BOTH of its flanks in Lab (so a step edge such as the hairline scores ~0), with a penalty for jumping between
    stations. Stations below ``viterbi_from`` are followed greedily. Returns (offset per station, ridge score per station);
    NaN offset where a station has no data."""
    nb = int(round(2 * reach_mm / step_mm))
    centres = -reach_mm + (np.arange(nb) + 0.5) * step_mm
    sel = ok & (np.abs(d) < reach_mm) & (zbin >= 0)
    ib = np.clip(((d[sel] + reach_mm) / step_mm).astype(int), 0, nb - 1)
    iz = zbin[sel]
    flat = iz * nb + ib
    count = np.bincount(flat, minlength=n_stations * nb).reshape(n_stations, nb).astype(float)
    mean = np.stack(
        [np.bincount(flat, weights=lab[sel][:, j], minlength=n_stations * nb).reshape(n_stations, nb) for j in range(3)], -1
    ) / np.maximum(count, 1)[..., None]
    valid_bin = count >= 2
    mean[~valid_bin] = np.nan

    def window(lo, hi, k):
        """Bin means whose centre lies ``lo..hi`` mm from candidate bin ``k`` (signed)."""
        idx = k + np.arange(int(round(lo / step_mm)), int(round(hi / step_mm)) + 1)
        return mean[:, idx[(idx >= 0) & (idx < nb)]]

    score = np.zeros((n_stations, nb))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are expected where a station has no data
        for k in range(nb):
            core = np.nanmean(window(-core_mm, core_mm, k), axis=1)
            left = np.nanmean(window(-flank_mm[1], -flank_mm[0], k), axis=1)
            right = np.nanmean(window(flank_mm[0], flank_mm[1], k), axis=1)
            s = np.minimum(np.linalg.norm(core - left, axis=1), np.linalg.norm(core - right, axis=1))
            score[:, k] = np.where(np.isfinite(s), s, 0.0)
    jump = jump_cost * np.abs(np.arange(nb)[:, None] - np.arange(nb)[None, :])
    t0 = int(np.clip(viterbi_from, 0, n_stations - 1))
    best = np.zeros_like(score)
    back = np.zeros(score.shape, int)
    best[t0] = score[t0]
    for t in range(t0 + 1, n_stations):
        cand = best[t - 1][None, :] - jump  # [to, from]
        back[t] = cand.argmax(1)
        best[t] = score[t] + cand.max(1)
    path = np.zeros(n_stations, int)
    path[-1] = int(best[-1].argmax())
    for t in range(n_stations - 1, t0, -1):
        path[t - 1] = back[t, path[t]]
    # below the Viterbi part (toward the ear) the strip is followed greedily within +-1.5 mm of the previous station
    reach = int(round(1.5 / step_mm))
    for t in range(t0 - 1, -1, -1):
        lo, hi = max(0, path[t + 1] - reach), min(nb, path[t + 1] + reach + 1)
        path[t] = lo + int(score[t, lo:hi].argmax())
    ridge = score[np.arange(n_stations), path]
    has_data = valid_bin.sum(1) >= 20
    return np.where(has_data, centres[path], np.nan), np.where(has_data, ridge, 0.0)


def regrain(filled: np.ndarray, band: np.ndarray, lab: np.ndarray, *, max_sigma: float = 2.5) -> np.ndarray:
    """Give NS-inpainted texels the fine luminance grain of the skin / hair just around the band (a smooth fill reads as
    a blur): zero-mean Gaussian noise with the (robust, capped) std of the high-pass lightness in a ring around the band.
    ``lab`` is the Lab of the texture the fill was computed from."""
    ring = (cv2.dilate(band.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=10) > 0) & ~band
    high = lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 2.0)
    if ring.sum() < 50:
        return filled
    sigma = min(max_sigma, 1.4826 * float(np.median(np.abs(high[ring] - np.median(high[ring])))))
    rng = np.random.default_rng(7)
    noise = cv2.GaussianBlur(rng.normal(0.0, 1.0, band.shape).astype(np.float32), (0, 0), 0.8)
    noise *= sigma / max(float(noise[band].std()), 1e-6)
    lab_f = _lab(filled)
    lab_f[..., 0] = np.where(band, lab_f[..., 0] + noise, lab_f[..., 0])
    result = filled.copy()
    result[..., :3] = np.where(band[..., None], _from_lab(lab_f), filled[..., :3])
    return result


def _streak_contrast(
    lab: np.ndarray, signed: np.ndarray, ok: np.ndarray, zbin: np.ndarray, n_stations: int
) -> dict:
    """Contrast of the 2 mm strip on the track against its flanks 3.4-6.4 mm away (outside the cleaned band), per depth
    station: ``ridge_dE`` is min(dE to the left flank, dE to the right flank) (a step edge scores ~0), ``dL`` the signed
    lightness of the strip minus the mean of the flanks. Medians over the stations that have enough texels."""
    sets = {
        "core": ok & (np.abs(signed) < 1.0),
        "left": ok & (signed > -6.4) & (signed < -3.4),
        "right": ok & (signed > 3.4) & (signed < 6.4),
    }
    means, counts = {}, {}
    for key, m in sets.items():
        idx = zbin[m]
        counts[key] = np.bincount(idx, minlength=n_stations)
        means[key] = np.stack(
            [np.bincount(idx, weights=lab[..., j][m], minlength=n_stations) for j in range(3)], -1
        ) / np.maximum(counts[key], 1)[:, None]
    good = (counts["core"] >= 5) & (counts["left"] >= 10) & (counts["right"] >= 10)
    if not good.any():
        return {"ridge_dE": None, "dL": None, "stations": 0}
    dl = np.linalg.norm(means["core"] - means["left"], axis=1)
    dr = np.linalg.norm(means["core"] - means["right"], axis=1)
    flank_l = (means["left"][:, 0] + means["right"][:, 0]) / 2
    return {
        "ridge_dE": float(np.median(np.minimum(dl, dr)[good])),
        "dL": float(np.median((means["core"][:, 0] - flank_l)[good])),
        "stations": int(good.sum()),
    }


def temple_streak(
    texture: np.ndarray,
    g: FrameProjection,
    prot: dict,
    *,
    ear: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    half_width_mm: float = 3.2,
    station_mm: float = 1.5,
    min_ridge: float = 2.0,
    inpaint: bool = True,
    radius: float = 5.0,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Find the photographed temple arm in the texture and NS-inpaint a thin band along it.

    The arm lies near (not on) the nominal line from the glasses report, so the strip is TRACKED: per ``station_mm`` of
    depth z a Viterbi ridge search over +-10 mm of the nominal line finds the thin strip that differs from both flanks.
    Only stations with a clear ridge (``min_ridge`` Lab units) are cleaned, short gaps are bridged. Ear, eyes, lashes and
    brows are excluded; the band is ``half_width_mm`` around the track. ``reference`` (the texture before any frame
    inpainting) is what the ridge is searched in and the "before" contrast measured on; the fill is applied to ``texture``.
    """
    h, w = texture.shape[:2]
    dense = _dense_positions(g, (h, w))
    X, Y, Z = dense[..., 0], dense[..., 1], dense[..., 2]
    valid = np.isfinite(X)
    if g.covered is not None:
        valid &= g.covered
    lab = _lab(texture if reference is None else reference)
    mid = float(np.asarray(g.eyes)[:, 0].mean())
    exclude = prot["protected"].copy()
    if ear is not None:
        exclude |= cv2.dilate(ear.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0
    ok_pixel = valid & ~exclude
    band = np.zeros((h, w), bool)
    cores = []
    sides = []
    for sign, path in _arm_paths(g):
        # the nominal line ends at the glasses' reach; the photographed arm continues toward the ear
        z_lo, z_hi = float(path[:, 1].min()) - 0.06, float(path[:, 1].max())
        with np.errstate(invalid="ignore"):
            d = (Y - np.interp(Z, path[:, 1], path[:, 0])) * 1000.0
            side = (sign * (X - mid) > 0.040) & (Z > z_lo) & (Z < z_hi)
        n_st = int(np.ceil((z_hi - z_lo) * 1000.0 / station_mm))
        zbin = np.where(side, np.floor((Z - z_lo) * 1000.0 / station_mm), -1).astype(int).clip(-1, n_st - 1)
        nominal_from = int(round(0.06 * 1000.0 / station_mm))  # stations below the nominal line's own reach: greedy
        offset, ridge = _track_ridge(
            np.nan_to_num(d, nan=1e9), lab, ok_pixel & side, zbin, n_st, viterbi_from=nominal_from,
            reach_mm=10.0, jump_cost=1.8,
        )
        good = np.isfinite(offset) & (ridge >= min_ridge)
        if good.sum() < 5:
            sides.append({"sign": sign, "applied": False, "reason": "no clear ridge", "ridge_median": float(np.median(ridge))})
            continue
        # Walk from the hinge toward the ear over the clear stations: a step of more than 1.5 mm per station or more than
        # 6 missing stations ends the track (beyond it the ridge would latch onto the hairline or the ear rim).
        track = np.full(n_st, np.nan)
        prev, gap, last = None, 0, -1
        for t in range(n_st - 1, -1, -1):
            fine = bool(good[t]) and (prev is None or abs(offset[t] - prev) <= 1.5 * (gap + 1))
            if not fine:
                if prev is not None:
                    gap += 1
                    if gap > 6:
                        break
                continue
            if last >= 0 and last - t > 1:
                track[t : last + 1] = np.linspace(offset[t], prev, last - t + 1)
            track[t] = offset[t]
            prev, gap, last = offset[t], 0, t
        sm = track.copy()
        for t in range(n_st):
            win = track[max(0, t - 2) : t + 3]
            if np.isfinite(track[t]) and np.isfinite(win).sum() >= 3:
                sm[t] = np.nanmedian(win)
        keep = np.isfinite(track)
        if keep.sum() >= 8:
            # the wire continues toward the eye past the last clear station (brow and lid zones hide it from the
            # tracker): extend 10 stations (15 mm) along the trend of the last 8; protected texels are never part of a band
            top = np.flatnonzero(np.isfinite(sm))
            top = top[top >= top.max() - 7]
            slope = float(np.clip(np.polyfit(top, sm[top], 1)[0], -0.6, 0.6))
            for t in range(int(top.max()) + 1, min(n_st, int(top.max()) + 11)):
                sm[t] = sm[top.max()] + slope * (t - top.max())
        if keep.sum() < 5:
            sides.append({"sign": sign, "applied": False, "reason": "no continuous ridge track"})
            continue
        centre = np.where(zbin >= 0, sm[np.clip(zbin, 0, n_st - 1)], np.nan)
        with np.errstate(invalid="ignore"):
            signed = d - centre
            side_band = ok_pixel & side & (np.abs(signed) < half_width_mm)
        band |= side_band
        cores.append((np.where(np.isfinite(signed), signed, 1e9), ok_pixel & side, zbin, n_st))
        sides.append(
            {
                "sign": sign,
                "applied": True,
                "stations": n_st,
                "stations_with_ridge": int(good.sum()),
                "stations_cleaned": int(np.isfinite(sm).sum()),
                "ridge_median": float(np.median(ridge[good])),
                "track_offset_mm_range": [float(np.nanmin(sm)), float(np.nanmax(sm))],
                "band_texels": int(side_band.sum()),
                "track_z_offset_mm": [
                    [round(z_lo * 1000 + (t + 0.5) * station_mm, 1), round(float(sm[t]), 2)]
                    for t in range(0, n_st, 3)
                    if np.isfinite(sm[t])
                ],
            }
        )
    out = texture.copy()
    if inpaint and band.any():
        out[..., :3] = cv2.inpaint(np.ascontiguousarray(texture[..., :3]), band.astype(np.uint8) * 255, radius, cv2.INPAINT_NS)
        out = regrain(out, band, _lab(texture))
    after = _lab(out)
    applied = [s for s in sides if s.get("applied")]
    for s, (signed, ok_side, zbin_s, n_s) in zip(applied, cores):
        s["streak_contrast_before"] = _streak_contrast(lab, signed, ok_side, zbin_s, n_s)
        s["streak_contrast_after"] = _streak_contrast(after, signed, ok_side, zbin_s, n_s)
    report = {"sides": sides, "changed_texels": int(np.any(out[..., :3] != texture[..., :3], axis=2).sum())}
    return out, report, band


def region_stats(texture: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"texels": 0}
    lab = _lab(texture)[mask]
    return {"texels": int(mask.sum()), "mean_Lab": [float(v) for v in lab.mean(0)], "std_L": float(lab[:, 0].std())}


def correct_lens_stage(
    texture: np.ndarray,
    g: FrameProjection,
    *,
    reference: np.ndarray | None = None,
    skin_mask: np.ndarray | None = None,
    eye_region: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Protection masks plus the lens-area gain/offset correction. Returns texture, report and a context for later stages.

    Run it BEFORE the frame strips are inpainted: the wire is then filled from already corrected neighbours on both
    sides, so no bump appears along the rim. ``reference`` (the texture before any frame inpainting) lets the fitted rim
    outlines snap onto the photographed wire, so the lens edge is located from the data."""
    h, w = texture.shape[:2]
    dense = _dense_positions(g, (h, w))
    texel_mm = texel_size_mm(dense)
    prot = protection_masks(texture, g, eye_region=eye_region, dense=dense, texel_mm=texel_mm)
    ell = rim_ellipses(g)
    rims = None
    if reference is not None:
        rims = track_rims(reference, g, ell, visible=front_facing(dense, np.isfinite(dense[..., 0]) if g.covered is None else g.covered))
    out, report, weight = correct_lens_area(texture, g, prot, ellipses=ell, reference=reference, rims=rims, skin_mask=skin_mask)
    report["texel_mm"] = texel_mm
    return out, report, {"prot": prot, "weight": weight, "rims": rims, "ellipses": ell}


def remove_rim_traces(
    texture: np.ndarray,
    g: FrameProjection,
    ctx: dict,
    *,
    half_width_mm: float = 1.2,
    radius: float = 4.0,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """NS-inpaint what is left of the rim wire along the SNAPPED outline, including where it runs between eye and brow.

    Only angles where the wire was found (ridge track) are touched, within ``half_width_mm`` of the snapped wire; tight eye
    regions (lids, lashes) and brow masses stay protected (a thin wire does not count as a brow, see ``protection_masks``).
    Returns texture, report and the mask."""
    rims = ctx.get("rims")
    if not rims:
        return texture, {"applied": False, "reason": "no reference track"}, np.zeros(texture.shape[:2], bool)
    h, w = texture.shape[:2]
    dense = _dense_positions(g, (h, w))
    valid = np.isfinite(dense[..., 0]) if g.covered is None else g.covered & np.isfinite(dense[..., 0])
    visible = front_facing(dense, valid)
    prot = ctx["prot"]
    keep_out = prot["eyes"] | prot["brows"]
    mask = np.zeros((h, w), bool)
    for e, rim in zip(ctx["ellipses"], rims):
        if not rim["tracked"]:
            continue
        rho, theta = _polar(dense, e)
        abin = np.floor((np.nan_to_num(theta) + np.pi) / (2 * np.pi) * _ANGLE_BINS).astype(int).clip(0, _ANGLE_BINS - 1)
        near_good = np.convolve(np.r_[rim["good"], rim["good"], rim["good"]].astype(float), np.ones(7), "same")[
            _ANGLE_BINS : 2 * _ANGLE_BINS
        ] > 0
        with np.errstate(invalid="ignore"):
            mask |= visible & near_good[abin] & (np.abs(rho - rim["offset_mm"][abin]) < half_width_mm)
    mask &= ~keep_out
    out = texture.copy()
    if mask.any():
        out[..., :3] = cv2.inpaint(np.ascontiguousarray(texture[..., :3]), mask.astype(np.uint8) * 255, radius, cv2.INPAINT_NS)
        out = regrain(out, mask, _lab(texture))
    return out, {"applied": True, "texels": int(mask.sum())}, mask


def protected_report(before: np.ndarray, after: np.ndarray, prot: dict) -> dict:
    """Eyes (eyelids, lashes) and brows: Lab statistics and the largest RGB change between two textures."""
    diff = np.abs(after[..., :3].astype(int) - before[..., :3].astype(int)).max(2)
    return {
        key: {
            "before": region_stats(before, m),
            "after": region_stats(after, m),
            "max_abs_rgb_diff": int(diff[m].max()) if m.any() else 0,
        }
        for key, m in (("eyes", prot["eyes"]), ("brows", prot["brows"]))
    }


def remove_lens_residue(
    texture: np.ndarray,
    g: FrameProjection,
    *,
    reference: np.ndarray | None = None,
    skin_mask: np.ndarray | None = None,
    eye_region: np.ndarray | None = None,
    ear: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Lens-area gain correction, then temple-arm streak inpainting (no thin-line pass in between; see
    ``correct_lens_stage``). Returns texture, report and QA masks."""
    stage1, lens_report, ctx = correct_lens_stage(
        texture, g, reference=reference, skin_mask=skin_mask, eye_region=eye_region
    )
    stage2, arm_report, arm_mask = temple_streak(stage1, g, ctx["prot"], ear=ear, reference=reference)
    report = {
        "texel_mm": lens_report["texel_mm"],
        "lens": lens_report,
        "temple": arm_report,
        "protected_regions": protected_report(texture, stage2, ctx["prot"]),
    }
    return stage2, report, {"lens_weight": ctx["weight"], "temple_band": arm_mask, "protected": ctx["prot"]["protected"]}
