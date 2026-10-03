"""Strand data atlas for the procedural hair cards (format ``rcov-groot-bvar/1``).

The atlas is *data*, not colour: it is read by the MIT three.js hair-card shader of creategamecharacters
(https://github.com/creategamecharacters/threejs-hair-shader, ``hair-shader.js``, compact atlas) with
``THREE.NoColorSpace``:

* **R** strand coverage (0..1). The shader multiplies it by ``density`` (2.5 with MSAA) *after* filtering and clamps,
  so a strip body whose mean coverage is about 0.5 renders as a solid card from a distance, while the single strands
  with their soft edges show close up. The no-MSAA inner core clips the raw R at 0.5 (the strand cores).
* **G** root to tip position along the card: 0 at the root, 1 at the tip (root colour, root darkening, roughness).
* **B** per-strand variation, centred on 0.5 (the shader's ``seedVariation`` scales ``(B - 0.5) * 2``).
* **A** ``min(1, 2.5 * R)``: only for plain glTF viewers (material ``MASK`` at 0.5). The shader ignores it.

The atlas holds many *strips* in slots of three classes (long, mid, short). Every card samples one whole slot: U runs
across the card, V from the root row (``v = 0`` side, top of the image, G = 0) to the tip row (G = 1). A strip is built
from a few *locks* (bundles of fine strands that converge towards the tip, clump together and end at different
heights), a soft root veil (the root end is dense and hidden under the next layer) and a lateral feather (cards
blend into their neighbours; nothing hard at the card edges).

Nothing here needs data: the atlas is generated from a seed. No colour enters it (the colours travel in the material
extras of the GLB).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .register import smoothstep

FORMAT = "rcov-groot-bvar/1"
GAIN = 2.5  # the shader's coverage gain with MSAA; the PNG alpha channel is min(1, GAIN * R) for plain glTF viewers


@dataclass(frozen=True)
class Slot:
    """One strip rectangle in atlas pixels (``x1`` / ``y1`` exclusive) and its guard bands."""

    kind: str  # "long" | "mid" | "short"
    x0: int
    y0: int
    x1: int
    y1: int
    pad_x: int
    pad_y: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def uv_box(self, tile_width: int, tile_height: int) -> tuple[float, float, float, float]:
        """Inner ``(u0, v_root, u1, v_tip)`` of the slot in atlas UV (glTF: v grows downwards, the root is on top)."""
        return (
            (self.x0 + self.pad_x) / tile_width,
            (self.y0 + self.pad_y) / tile_height,
            (self.x1 - self.pad_x) / tile_width,
            (self.y1 - self.pad_y) / tile_height,
        )


@dataclass(frozen=True)
class StripLayout:
    width: int
    height: int
    long: tuple[Slot, ...]
    mid: tuple[Slot, ...]
    short: tuple[Slot, ...]

    def slots(self, kind: str) -> tuple[Slot, ...]:
        return {"long": self.long, "mid": self.mid, "short": self.short}[kind]

    @property
    def all(self) -> tuple[Slot, ...]:
        return (*self.long, *self.mid, *self.short)


def make_layout(width: int, height: int) -> StripLayout:
    """Slots of the atlas (2048 x 1024: 20 long strips of 64 x 1024, 16 mid of 64 x 512, 32 short of 64 x 128).

    Every slot keeps empty guard bands (3 px at the sides, up to 8 px on the root and tip side) that cards never sample:
    mip filtering then does not bleed the dense root end of the strip below into the tip of the strip above.
    """
    column = 64 if width >= 2048 else max(16, width // 32)
    columns = max(width // column, 4)
    column = width // columns
    n_long = max(2, round(0.625 * columns))
    n_mid = max(1, round(0.25 * columns))
    n_short = max(1, columns - n_long - n_mid)

    def pads(slot_width: int, slot_height: int) -> tuple[int, int]:
        return max(1, min(3, slot_width // 16)), max(1, min(8, slot_height // 16))

    long_pad = pads(column, height)
    long = tuple(Slot("long", i * column, 0, (i + 1) * column, height, *long_pad) for i in range(n_long))
    x = n_long * column
    mid_height = height // 2
    mid_pad = pads(column, mid_height)
    mid = tuple(
        Slot("mid", x + c * column, r * mid_height, x + (c + 1) * column, (r + 1) * mid_height, *mid_pad)
        for r in range(2)
        for c in range(n_mid)
    )
    x += n_mid * column
    rows = 8
    short_height = height // rows
    short_pad = pads(column, short_height)
    short = tuple(
        Slot("short", x + c * column, r * short_height, x + (c + 1) * column, (r + 1) * short_height, *short_pad)
        for r in range(rows)
        for c in range(n_short)
    )
    return StripLayout(width, height, long, mid, short)


# (strands per lock, lock reach range, lock pitch in px, strand wiggle scale)
KINDS = {
    "long": (8, (0.80, 1.0), 14.0, 1.0),
    "mid": (8, (0.84, 1.0), 14.0, 0.8),
    "short": (6, (0.88, 1.0), 19.0, 0.5),
}


def _strip(width: int, height: int, pad_x: int, pad_y: int, rng: np.random.Generator, kind: str):
    """One strip: ``(R, G, B)`` float32 ``(height, width)``. Root on the first row, tip on the last."""
    per_lock, reach_range, pitch_px, wiggle = KINDS[kind]
    inner_w = max(width - 2 * pad_x, 4)
    inner_h = max(height - 2 * pad_y, 4)
    scale = float(np.clip(inner_w / 58.0, 0.45, 1.25))  # strand widths follow the slot width (small test atlases)
    locks = int(max(2, round(inner_w / (pitch_px * scale))))
    pitch = inner_w / locks
    y = np.arange(height, dtype=np.float32) + 0.5
    s = np.clip((y - pad_y) / inner_h, 0.0, 1.0)  # per row: 0 at the root, 1 at the tip
    x = (np.arange(width, dtype=np.float32) + 0.5)[None, :]

    inv = np.ones((height, width), np.float32)  # product of (1 - coverage): union of the strands
    b_sum = np.zeros((height, width), np.float32)
    w_sum = np.zeros((height, width), np.float32)

    reach_lock = rng.uniform(*reach_range, locks)
    reach_lock[rng.integers(0, locks)] = 1.0  # the longest lock reaches the tip row
    for k in range(locks):
        c_k = pad_x + (k + 0.5 + rng.uniform(-0.3, 0.3)) * pitch
        omega = pitch * rng.uniform(1.0, 1.5)
        drift = rng.normal(0.0, 0.30) * pitch
        wig_a = pitch * rng.uniform(0.04, 0.2) * wiggle
        wig_f = rng.uniform(0.6, 1.5)
        wig_p = rng.uniform(0, 2 * np.pi)
        r_k = float(reach_lock[k])
        lock_b = rng.normal(0.0, 0.10)
        converge = 1.0 - 0.45 * smoothstep(s / max(r_k, 0.3))  # the lock narrows towards its tip
        axis = c_k + drift * s + wig_a * np.sin(2 * np.pi * wig_f * s + wig_p)
        for j in range(per_lock):
            o = float(np.clip(rng.normal(0.0, 0.34), -0.6, 0.6))
            reach = r_k * (1.0 - 0.14 * abs(o) / 0.6 - 0.08 * rng.random()) if j else r_k
            start = rng.uniform(0.0, 0.05)
            half = rng.uniform(0.85, 1.45) * scale
            amp = rng.uniform(0.72, 0.97)
            own_a = rng.uniform(0.2, 0.9) * scale * wiggle
            own_f = rng.uniform(1.5, 4.0)
            own_p = rng.uniform(0, 2 * np.pi)
            centre = axis + o * omega * converge + own_a * np.sin(2 * np.pi * own_f * s + own_p)
            width_s = half * (1.0 - 0.8 * smoothstep((s - (reach - 0.24)) / 0.24))
            amp_s = amp * (1.0 - smoothstep((s - (reach - 0.10)) / 0.10)) * smoothstep((s - start) / 0.03 + 0.5)
            u = (x - centre[:, None]) / (1.8 * np.maximum(width_s, 0.05)[:, None])
            c = (np.clip(1.0 - u * u, 0.0, None) ** 2) * amp_s[:, None]
            b = float(np.clip(0.5 + lock_b + 0.20 * rng.normal(), 0.03, 0.97))
            inv *= 1.0 - c
            b_sum += c * b
            w_sum += c
        # a faint lock body: the gaps between strands are not completely empty close to the root
        halo_u = (x - axis[:, None]) / (0.62 * omega * np.maximum(converge, 0.2)[:, None])
        halo = (np.clip(1.0 - halo_u**2, 0.0, None) ** 2) * (0.30 * (1.0 - smoothstep(s / max(r_k, 0.3))))[:, None]
        inv *= 1.0 - halo
        b_sum += halo * (0.5 + lock_b)
        w_sum += halo
    # root veil: the root end is dense (it is covered by the next layer anyway) and fades into the strands
    root = (0.9 * (1.0 - smoothstep((s - 0.06) / 0.30)) * smoothstep(s / 0.07))[:, None] * np.ones_like(x)
    inv *= 1.0 - root
    b_sum += root * 0.5
    w_sum += root
    coverage = 1.0 - inv
    feather = max(2.5, 0.12 * inner_w)
    window = smoothstep((x - pad_x) / feather) * smoothstep((pad_x + inner_w - x) / feather)
    coverage = coverage * window
    coverage[:, :pad_x] = 0.0
    coverage[:, pad_x + inner_w :] = 0.0
    strip_b = np.where(w_sum > 1e-4, b_sum / np.maximum(w_sum, 1e-4), 0.5) + rng.normal(0.0, 0.03)
    g = np.broadcast_to(s[:, None], (height, width))
    return coverage.astype(np.float32), g.astype(np.float32), np.clip(strip_b, 0.0, 1.0).astype(np.float32)


def hair_atlas(width: int, height: int, seed: int = 3) -> tuple[np.ndarray, StripLayout]:
    """The strand data atlas ``(height, width, 4)`` uint8 (R coverage, G root to tip, B variation, A = min(1, 2.5 R))."""
    layout = make_layout(width, height)
    rng = np.random.default_rng(seed)
    r = np.zeros((height, width), np.float32)
    g = np.zeros((height, width), np.float32)
    b = np.full((height, width), 0.5, np.float32)
    for slot in layout.all:
        cov, root, var = _strip(slot.width, slot.height, slot.pad_x, slot.pad_y, rng, slot.kind)
        r[slot.y0 : slot.y1, slot.x0 : slot.x1] = cov
        g[slot.y0 : slot.y1, slot.x0 : slot.x1] = root
        b[slot.y0 : slot.y1, slot.x0 : slot.x1] = var
    to_byte = lambda a: np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)  # noqa: E731
    alpha = to_byte(np.minimum(1.0, GAIN * r))
    return np.dstack((to_byte(r), to_byte(g), to_byte(b), alpha)), layout


def strip_statistics(atlas: np.ndarray, layout: StripLayout) -> dict:
    """Coverage numbers of the slots: mean R and share of gained coverage above 0.5 near the root, mid-way and at the tip."""
    r = atlas[..., 0] / 255.0
    solid = np.minimum(1.0, GAIN * r) >= 0.5
    result = {}
    for kind in ("long", "mid", "short"):
        mean = np.zeros(3)
        share = np.zeros(3)
        for slot in layout.slots(kind):
            rows = slice(slot.y0 + slot.pad_y, slot.y1 - slot.pad_y)
            cols = slice(slot.x0 + slot.pad_x, slot.x1 - slot.pad_x)
            inner_r, inner_s = r[rows, cols], solid[rows, cols]
            h = inner_r.shape[0]
            for k, (a, b) in enumerate(((0.0, 0.15), (0.4, 0.6), (0.85, 1.0))):  # root, middle, tip
                band = slice(int(a * h), max(int(b * h), int(a * h) + 1))
                mean[k] += inner_r[band].mean()
                share[k] += inner_s[band].mean()
        count = max(len(layout.slots(kind)), 1)
        result[kind] = {
            "mean_r": {"root": float(mean[0] / count), "middle": float(mean[1] / count), "tip": float(mean[2] / count)},
            "gained_ge_0_5": {
                "root": float(share[0] / count),
                "middle": float(share[1] / count),
                "tip": float(share[2] / count),
            },
        }
    result["slots"] = {kind: len(layout.slots(kind)) for kind in ("long", "mid", "short")}
    result["atlas_mean_r"] = float(r.mean())
    return result


def mip_levels(atlas: np.ndarray, levels: int = 5) -> list[np.ndarray]:
    """Box-filtered pyramid of the atlas as float arrays in 0..1 (R, G, B), for the coverage metric and the previews."""
    import cv2

    current = atlas[..., :3].astype(np.float32) / 255.0
    out = [current]
    for _ in range(levels - 1):
        h, w = current.shape[:2]
        current = cv2.resize(current, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
        out.append(current)
    return out


def prefiltered_alpha(atlas: np.ndarray, level: int = 3) -> np.ndarray:
    """The shader's outer-pass alpha after mip filtering: min(1, 2.5 * R) of the box-filtered coverage (``level``)."""
    r = mip_levels(atlas, level + 1)[level][..., 0]
    return np.minimum(1.0, GAIN * r).astype(np.float32)
