"""sRGB <-> linear helpers (float32)."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


_LUT = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)


def u8_to_linear(img: np.ndarray) -> np.ndarray:
    """uint8 sRGB image -> float32 linear via a lookup table."""
    return _LUT[img]


def linear_to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(linear_to_srgb(x) * 255.0), 0, 255).astype(np.uint8)
