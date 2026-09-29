"""Tiny dependency-free raster helpers (numpy + zlib) for pipeline debug images. Not used by the runtime."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array as an 8-bit RGB PNG."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def draw_line(img: np.ndarray, p0, p1, color) -> None:
    """Draw a 1-pixel line between two (x, y) pixel positions (float), clipped to the image."""
    h, w, _ = img.shape
    x0, y0 = p0
    x1, y1 = p1
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n > 4 * max(h, w):
        return
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[ok], xs[ok]] = color


def draw_dot(img: np.ndarray, p, color, r: int = 1) -> None:
    h, w, _ = img.shape
    x, y = int(round(p[0])), int(round(p[1]))
    img[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)] = color
