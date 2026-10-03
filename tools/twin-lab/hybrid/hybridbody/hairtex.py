"""Procedural hair strip texture for the procedural hair cards.

The tile is an RGBA image (straight alpha, ``alphaMode MASK`` cut-outs) that is packed into the twin's single atlas.
It holds many *strips*: each strip is one hair-card texture with a few dozen fine strands that run along the strip
(root at the bottom, tip at the top), taper to a point at their ends, wiggle slightly and differ in brightness. Two
strip classes share the tile: *long* strips (aspect about 1:8) for the cards of the top of the head and *short* strips
for the cropped sides and back. Every card picks one slot, so neighbouring cards differ subtly in colour and strand
pattern.

Nothing here needs data: the texture is generated from a base colour and a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from twintex.colorspace import linear_to_srgb, srgb_to_linear

from .register import smoothstep


@dataclass(frozen=True)
class Slot:
    """One strip rectangle in tile pixels (``x1``/``y1`` exclusive) and the usable inner rectangle."""

    kind: str  # "long" | "short"
    x0: int
    y0: int
    x1: int
    y1: int
    pad: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def uv_box(self, tile_width: int, tile_height: int) -> tuple[float, float, float, float]:
        """Inner ``(u0, v_tip, u1, v_root)`` of the slot in tile UV (glTF convention: v grows downwards)."""
        return (
            (self.x0 + self.pad) / tile_width,
            (self.y0 + self.pad) / tile_height,
            (self.x1 - self.pad) / tile_width,
            (self.y1 - self.pad) / tile_height,
        )


@dataclass(frozen=True)
class StripLayout:
    width: int
    height: int
    long: tuple[Slot, ...]
    short: tuple[Slot, ...]

    def slots(self, kind: str) -> tuple[Slot, ...]:
        return self.long if kind == "long" else self.short


def make_layout(width: int, height: int, long_share: float = 0.75) -> StripLayout:
    """Slots of the tile: tall strips on the left, small strips on the right (both with a transparent gutter)."""
    long_columns = int(np.clip((width * long_share) // 8, 3, 12))
    long_width = int(width * long_share) // long_columns
    pad = max(1, min(4, long_width // 24))
    long = tuple(Slot("long", i * long_width, 0, (i + 1) * long_width, height, pad) for i in range(long_columns))
    x_start = long_columns * long_width
    short_columns = int(np.clip((width - x_start) // 8, 2, 4))
    short_width = (width - x_start) // short_columns
    rows = 5
    short_height = height // rows
    short = tuple(
        Slot(
            "short",
            x_start + c * short_width,
            r * short_height,
            x_start + (c + 1) * short_width,
            (r + 1) * short_height,
            pad,
        )
        for r in range(rows)
        for c in range(short_columns)
    )
    return StripLayout(width, height, long, short)


def _strand_strip(
    width: int,
    height: int,
    pad: int,
    rng: np.random.Generator,
    base_linear: np.ndarray,
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """One strip: ``(linear rgb (h, w, 3), alpha (h, w))``. Root at the bottom row, tip at the top row."""
    inner_w = max(width - 2 * pad, 4)
    inner_h = max(height - 2 * pad, 4)
    strands = int(np.clip(inner_w / 4.2, 4, 40))
    spacing = inner_w / strands
    # strand centres: jittered grid across the strip
    x0 = pad + (np.arange(strands) + rng.uniform(0.1, 0.9, strands)) * spacing
    half = spacing * rng.uniform(0.62, 0.95, strands)  # overlapping strands: the strip is dense near the root
    amplitude = spacing * rng.uniform(0.15, 0.9, strands) * (1.0 if kind == "long" else 0.5)
    wavelength = rng.uniform(0.7, 1.6, strands)
    phase = rng.uniform(0, 2 * np.pi, strands)
    # strand length as a fraction of the strip: cut tips at different heights, the longest reach the top row
    reach = rng.uniform(0.55, 1.0, strands) if kind == "long" else rng.uniform(0.6, 1.0, strands)
    reach[rng.integers(0, strands, max(1, strands // 5))] = 1.0
    root_start = rng.uniform(0.0, 0.22, strands)  # ragged root edge: strands begin at different heights
    brightness = np.exp(rng.normal(0.0, 0.16, strands))
    warmth = rng.normal(0.0, 0.05, strands)
    strip_gain = float(np.exp(rng.normal(0.0, 0.05)))

    y = np.arange(height, dtype=np.float32) + 0.5
    s = np.clip((pad + inner_h - y) / inner_h, -0.5, 1.5)[:, None]  # 0 at the root row, 1 at the tip row
    x = np.arange(width, dtype=np.float32) + 0.5
    best = np.full((height, width), -1e9, np.float32)  # best signed edge coverage over the strands
    colour = np.zeros((height, width, 3), np.float32)
    feather = 0.9
    tilt = rng.normal(0.0, 0.12, strands)  # lateral drift towards the tip
    for i in range(strands):
        centre = x0[i] + (
            amplitude[i] * np.sin(2 * np.pi * (s / wavelength[i]) + phase[i]) + tilt[i] * inner_w * 0.1 * s
        )
        taper = np.clip((reach[i] - s) / 0.24, 0.0, 1.0) ** 0.8
        root_fade = smoothstep((s - root_start[i]) / 0.06)
        hw = np.maximum(half[i] * taper, 0.0) * root_fade
        d = np.abs(x[None, :] - centre) - hw  # (h, w): < 0 inside the strand
        cover = 0.5 - d / feather
        # across the strand: darker edges, brighter core; along it: darker at the root, a bit lighter at the tip
        radial = np.clip((d + hw) / np.maximum(hw, 0.5), 0.0, 1.0)  # 0 on the strand axis, 1 at its edge
        across = 1.0 - 0.26 * radial**2
        along = 0.62 + 0.38 * smoothstep(s / 0.4) + 0.10 * smoothstep((s - 0.55) / 0.45)
        light = (brightness[i] * across * along * strip_gain).astype(np.float32)
        take = cover > best
        best = np.where(take, cover, best)
        tint = np.array([1.0 + warmth[i], 1.0, 1.0 - 0.8 * warmth[i]], np.float32)
        colour = np.where(take[..., None], (base_linear * tint)[None, None, :] * light[..., None], colour)
    alpha = np.clip(best, 0.0, 1.0)
    return colour, alpha


def hair_tile(width: int, height: int, colour_srgb, seed: int = 3) -> tuple[np.ndarray, StripLayout]:
    """The RGBA strip tile (uint8, straight alpha) and its slot layout; colours are strands of ``colour_srgb``."""
    layout = make_layout(width, height)
    base = srgb_to_linear(np.clip(np.asarray(colour_srgb, np.float32) / 255.0, 0, 1))
    rng = np.random.default_rng(seed)
    linear = np.broadcast_to(base, (height, width, 3)).copy()
    alpha = np.zeros((height, width), np.float32)
    for slot in (*layout.long, *layout.short):
        rgb, a = _strand_strip(slot.width, slot.height, slot.pad, rng, base, kind=slot.kind)
        linear[slot.y0 : slot.y1, slot.x0 : slot.x1] = rgb
        alpha[slot.y0 : slot.y1, slot.x0 : slot.x1] = a
    srgb = np.clip(np.rint(linear_to_srgb(np.clip(linear, 0, 1)) * 255), 0, 255).astype(np.uint8)
    return np.dstack((srgb, np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8))), layout


def strip_statistics(tile: np.ndarray, layout: StripLayout, cutoff: float = 0.5) -> dict:
    """Coverage numbers of the slots (alpha above ``cutoff``): near the root, mid-way and at the tip rows."""
    alpha = tile[..., 3] / 255.0 >= cutoff
    result = {}
    for kind in ("long", "short"):
        bands = np.zeros(3)
        for slot in layout.slots(kind):
            inner = alpha[slot.y0 + slot.pad : slot.y1 - slot.pad, slot.x0 + slot.pad : slot.x1 - slot.pad]
            h = inner.shape[0]
            for k, (a, b) in enumerate(((0.8, 1.0), (0.4, 0.6), (0.0, 0.15))):  # root, middle, tip (top rows)
                rows = slice(int(a * h), max(int(b * h), int(a * h) + 1))
                bands[k] += inner[rows].mean()
        bands /= max(len(layout.slots(kind)), 1)
        result[kind] = {"root": float(bands[0]), "middle": float(bands[1]), "tip": float(bands[2])}
    result["slots"] = {"long": len(layout.long), "short": len(layout.short)}
    return result
