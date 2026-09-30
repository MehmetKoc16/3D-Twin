"""Texture preparation for the body-part proxies (Pillow + numpy). Deterministic.

Hair / eyebrow / eyelash textures are RGBA atlases with soft alpha. What is shipped:

  * downscaled with premultiplied alpha (no dark halos), colours bled into the transparent area (no fringes when the
    GPU filters / mip-maps the texture), 8-bit PNG;
  * tintable parts (hair, eyebrows) as a *neutral* grey (value) + alpha map, normalised so strand highlights reach ~0.92.
    The runtime colour is `material.color` (glTF baseColorFactor) x this texture, so every hair colour works (a coloured
    texture could only be darkened). The glb factor carries the colour of the original texture as the default tint, so
    an untouched load looks like the MakeHuman original.
  * the alpha cutoff (alphaMode MASK) is chosen so that the covered area is the same as the soft alpha coverage.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

LUMA = np.array([0.299, 0.587, 0.114])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_of(srgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(round(float(np.clip(v, 0, 1)) * 255)):02x}" for v in srgb[:3])


def load_rgba(path) -> np.ndarray:
    img = Image.open(path)
    img.load()
    return np.asarray(img.convert("RGBA"))


def resize_rgba(rgba: np.ndarray, max_px: int) -> np.ndarray:
    """Lanczos downscale of straight-alpha RGBA to at most max_px (premultiplied, so transparent texels do not leak)."""
    h, w, _ = rgba.shape
    if max(h, w) <= max_px:
        return rgba
    k = max_px / max(h, w)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    a = rgba[..., 3].astype(np.float32) / 255.0
    chans = []
    for c in range(3):
        pm = rgba[..., c].astype(np.float32) * a
        chans.append(np.asarray(Image.fromarray(pm).resize((nw, nh), Image.LANCZOS)))
    alpha = np.asarray(Image.fromarray(a).resize((nw, nh), Image.LANCZOS)).clip(0.0, 1.0)
    out = np.zeros((nh, nw, 4), dtype=np.float32)
    ok = alpha > 1e-4
    for c in range(3):
        out[..., c] = np.where(ok, chans[c] / np.maximum(alpha, 1e-4), 0.0)
    out[..., 3] = alpha * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def bleed_colors(rgba: np.ndarray, threshold: int = 8, iterations: int = 16) -> np.ndarray:
    """Fill the colour of (nearly) transparent texels from the nearest opaque ones; alpha is untouched."""
    rgb = rgba[..., :3].astype(np.float32)
    valid = rgba[..., 3] > threshold
    if valid.all() or not valid.any():
        return rgba
    for _ in range(iterations):
        acc = np.zeros_like(rgb)
        cnt = np.zeros(valid.shape, dtype=np.float32)
        vf = valid.astype(np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ys = slice(max(dy, 0), valid.shape[0] + min(dy, 0))
                yd = slice(max(-dy, 0), valid.shape[0] + min(-dy, 0))
                xs = slice(max(dx, 0), valid.shape[1] + min(dx, 0))
                xd = slice(max(-dx, 0), valid.shape[1] + min(-dx, 0))
                acc[yd, xd] += rgb[ys, xs] * vf[ys, xs, None]
                cnt[yd, xd] += vf[ys, xs]
        fill = (~valid) & (cnt > 0)
        if not fill.any():
            break
        rgb[fill] = acc[fill] / cnt[fill, None]
        valid = valid | fill
    if not valid.all():  # far away from anything opaque: the mean opaque colour
        rgb[~valid] = rgb[valid].mean(axis=0)
    out = rgba.copy()
    out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out


def coverage_cutoff(alpha: np.ndarray, lo: float = 0.3, hi: float = 0.6) -> float:
    """Alpha-test cutoff whose covered area equals the soft coverage sum(alpha) (rounded to 0.05, clamped)."""
    a = alpha.reshape(-1).astype(np.float64) / 255.0
    target = float(a.mean())
    cut = float(np.quantile(a, 1.0 - target)) if 0.0 < target < 1.0 else 0.5
    return float(np.clip(round(cut / 0.05) * 0.05, lo, hi))


def neutralise(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(LA uint8 (H, W, 2), default tint as LINEAR rgb (3,)) of an RGBA texture: tint x LA ~ the original colour.

    The grey level is the per-pixel value (max of r, g, b), so that `orig_c <= value` and the tint stays inside [0, 1]
    (a luma map could not reproduce saturated colours by multiplication). It is stretched so the strand highlights
    (95th percentile) reach 0.92; a flat or black texture has no detail to keep and becomes a flat 0.9."""
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    a = rgba[..., 3].astype(np.float64) / 255.0
    val = rgb.max(axis=2)
    solid = a > 0.5
    ref = float(np.quantile(val[solid], 0.95)) if solid.any() else 1.0
    if ref < 4.0 / 255.0 or float(val[solid].std()) < 1.0 / 255.0:
        lum = np.full_like(val, 0.9)
    else:
        lum = np.clip(val * min(0.92 / ref, 30.0), 0.0, 1.0)
    w = a / max(a.sum(), 1e-9)
    orig_lin = (srgb_to_linear(rgb) * w[..., None]).reshape(-1, 3).sum(axis=0)
    neutral_lin = float((srgb_to_linear(lum) * w).sum())
    tint = np.clip(orig_lin / max(neutral_lin, 1e-6), 0.0, 1.0)
    la = np.stack([np.rint(lum * 255.0), rgba[..., 3]], axis=-1).clip(0, 255).astype(np.uint8)
    return la, tint


def encode_png(arr: np.ndarray) -> bytes:
    """8-bit PNG of an (H, W) L, (H, W, 2) LA, (H, W, 3) RGB or (H, W, 4) RGBA array."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=False, progressive=False)
    return buf.getvalue()
