"""Colours and atlas tiles for the MakeHuman parts (eyes, brows, lashes, hair) of the single-material twin.

The web app draws the twin with one opaque material, so alpha-masked cards cannot rely on cut-outs: every card is
composited over a flat base colour (skin for brows, dark scalp for hair) and baked into one RGB atlas strip. The
parts are tintable neutral-grey maps, exactly as in the app's part pipeline (``irisRecolor.ts`` is ported here).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from flamehead.colour import to_lab
from twintex.colorspace import linear_to_srgb, srgb_to_linear

from .register import smoothstep
from .template import Part

DEFAULT_HAIR_LINEAR = np.array([0.05, 0.034, 0.025])  # dark brown
DEFAULT_IRIS_SRGB = np.array([76, 52, 34], float)  # dark brown


def srgb_hex(rgb) -> str:
    c = np.clip(np.rint(np.asarray(rgb, float)), 0, 255).astype(int)
    return "#" + "".join(f"{int(v):02x}" for v in c)


def lab_to_srgb255(lab) -> np.ndarray:
    from flamehead.colour import from_lab

    return np.clip(from_lab(np.asarray(lab, np.float32).reshape(1, 3))[0] * 255, 0, 255)


def recolor_iris(texture: np.ndarray, center, radius, colour_srgb) -> np.ndarray:
    """Port of the app's ``recolorIris``: iris pixels become ``colour * luma / reference`` (detail survives)."""
    h, w = texture.shape[:2]
    cx, cy, r = center[0] * w, center[1] * h, radius * w
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / r
    src = texture[..., :3].astype(np.float32)
    luma = (0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]) / 255.0
    ring = (d >= 0.4) & (d <= 0.9)
    reference = max(0.05, float(luma[ring].mean())) if ring.any() else 0.5
    weight = 1 - smoothstep((d - 0.97) / (1.12 - 0.97))
    ratio = np.minimum(1.8, luma / reference)
    target = np.minimum(255.0, np.asarray(colour_srgb, np.float32)[None, None, :] * ratio[..., None])
    out = src + (target - src) * weight[..., None]
    result = texture.copy()
    result[..., :3] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return result


def covered_linear_mean(texture: np.ndarray, cutoff: float) -> float:
    alpha = texture[..., 3] / 255.0
    covered = alpha >= cutoff
    if not covered.any():
        return 1.0
    value = texture[..., :3].max(axis=2)[covered] / 255.0
    return float(max(0.02, srgb_to_linear(value.astype(np.float32)).mean()))


def card_tile(part: Part, colour_linear: np.ndarray | None, base_srgb, cutoff: float | None = None) -> np.ndarray:
    """RGB tile of an alpha-masked part: tinted strands over ``base_srgb`` (soft threshold around the cutoff)."""
    texture = part.texture
    cutoff = part.meta["material"].get("alphaCutoff", 0.5) if cutoff is None else cutoff
    alpha = texture[..., 3] / 255.0
    cover = smoothstep((alpha - (cutoff - 0.12)) / 0.24)
    if colour_linear is not None and part.tintable:
        factor = covered_linear_mean(texture, cutoff)
        grey = srgb_to_linear(texture[..., :3].astype(np.float32) / 255.0)
        strand = np.clip(grey * (np.asarray(colour_linear, np.float32) / factor), 0, 1)
    else:
        strand = srgb_to_linear(texture[..., :3].astype(np.float32) / 255.0)
        strand = np.clip(strand * np.asarray(part.base_color, np.float32), 0, 1) if part.tintable else strand
    base = srgb_to_linear(np.asarray(base_srgb, np.float32).reshape(1, 1, 3) / 255.0)
    mixed = strand * cover[..., None] + base * (1 - cover[..., None])
    return np.clip(np.rint(linear_to_srgb(mixed) * 255), 0, 255).astype(np.uint8)


def triangle_coverage(part: Part, cutoff: float | None = None) -> np.ndarray:
    """Fraction of each card triangle that carries strands (alpha above the cutoff), from 7 samples per triangle."""
    from flamehead.texture import sample

    cutoff = part.meta["material"].get("alphaCutoff", 0.5) if cutoff is None else cutoff
    alpha = (part.texture[..., 3] / 255.0).astype(np.float32)
    h, w = alpha.shape
    corners = part.uv[part.faces]  # (m, 3, 2)
    mids = (corners + np.roll(corners, -1, axis=1)) / 2
    centre = corners.mean(1, keepdims=True)
    points = np.concatenate((corners, mids, centre), axis=1).reshape(-1, 2)
    values = sample(alpha[..., None], points * np.array([w, h]))
    return (values.reshape(len(part.faces), 7) >= cutoff).mean(1)


def eye_tile(part: Part, iris_srgb) -> np.ndarray:
    iris = part.meta["irisUv"]
    return recolor_iris(part.texture, iris["center"], iris["radius"], iris_srgb)[..., :3]


@dataclass
class Tile:
    name: str
    image: np.ndarray  # (h, w, 3) uint8
    uv_min: np.ndarray  # source UV box covered by the part's triangles
    uv_max: np.ndarray
    origin: tuple[int, int] = (0, 0)  # filled by pack_strip: (x, y) in the atlas

    def remap(self, uv: np.ndarray, atlas_width: int, atlas_height: int) -> np.ndarray:
        """Source part UVs -> atlas UVs (glTF convention)."""
        span = np.maximum(self.uv_max - self.uv_min, 1e-9)
        local = (uv - self.uv_min) / span
        pixel = np.array(self.origin, float) + local * np.array([self.image.shape[1], self.image.shape[0]], float)
        return pixel / np.array([atlas_width, atlas_height], float)


def make_tile(name: str, image: np.ndarray, uv: np.ndarray, width: int, max_height: int, pad: float = 0.01) -> Tile:
    """Crop ``image`` to the UV box of the part, resize to ``width`` (height from the aspect, capped)."""
    h, w = image.shape[:2]
    low = np.clip(uv.min(0) - pad, 0, 1)
    high = np.clip(uv.max(0) + pad, 0, 1)
    x0, x1 = int(np.floor(low[0] * w)), max(int(np.ceil(high[0] * w)), int(np.floor(low[0] * w)) + 1)
    y0, y1 = int(np.floor(low[1] * h)), max(int(np.ceil(high[1] * h)), int(np.floor(low[1] * h)) + 1)
    crop = image[y0:y1, x0:x1]
    height = int(np.clip(round(width * crop.shape[0] / crop.shape[1]), 8, max_height))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    return Tile(name, resized, np.array([x0 / w, y0 / h]), np.array([x1 / w, y1 / h]))


def pack_strip(tiles: list[Tile], atlas_width: int, strip_height: int, y_offset: int = 0) -> np.ndarray:
    """Place tiles left to right in a strip of ``strip_height`` rows and return the strip image.

    ``origin`` of every tile becomes its position in the full atlas, whose strip starts at row ``y_offset``.
    """
    strip = np.zeros((strip_height, atlas_width, 3), np.uint8)
    x = 0
    for tile in tiles:
        h, w = tile.image.shape[:2]
        if x + w + 4 > atlas_width:
            raise ValueError("Part tiles do not fit into the atlas strip")
        strip[:h, x : x + w] = tile.image
        # replicate edges into a 2 px gutter so bilinear taps at tile borders stay clean
        strip[:h, x + w : x + w + 2] = tile.image[:, -1:]
        if h < strip_height:
            strip[h : h + 2, x : x + w] = tile.image[-1:]
        tile.origin = (x, y_offset)
        x += w + 4
    return strip


def sample_ring(photo: np.ndarray, centre_px, radius_px, inner=0.4, outer=0.85) -> np.ndarray:
    """sRGB pixels of the ring ``inner..outer`` x radius around a photo position (original photo pixels)."""
    h, w = photo.shape[:2]
    reach = int(np.ceil(radius_px * outer)) + 1
    cx, cy = int(round(centre_px[0])), int(round(centre_px[1]))
    x0, x1 = max(cx - reach, 0), min(cx + reach + 1, w)
    y0, y1 = max(cy - reach, 0), min(cy + reach + 1, h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - centre_px[0], yy - centre_px[1]) / max(radius_px, 1e-6)
    return photo[y0:y1, x0:x1][(d >= inner) & (d <= outer)].reshape(-1, 3).astype(np.float64)


def robust_colour(pixels: np.ndarray, lightness=(8, 80)):
    """Median sRGB of the pixels whose Lab lightness lies in ``lightness`` (glints and pure black rejected)."""
    if len(pixels) == 0:
        return None, 0
    lab = to_lab(pixels.astype(np.float32) / 255.0)
    keep = (lab[:, 0] >= lightness[0]) & (lab[:, 0] <= lightness[1])
    if keep.sum() < 12:
        return None, int(keep.sum())
    return np.median(pixels[keep], axis=0), int(keep.sum())
