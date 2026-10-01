"""Tiny solid-colour patches in unused atlas space (for the cap triangles added by the armpit separation)."""

from __future__ import annotations

import cv2
import numpy as np


def uv_coverage(uv_c: np.ndarray, size: int = 512) -> np.ndarray:
    """Coverage mask of the UV layout at ``size`` x ``size`` (uv_c: (m, 3, 2) corner UVs, v down)."""
    img = np.zeros((size, size), np.uint8)
    pts = np.rint(uv_c * size).astype(np.int32)
    for tri in pts:
        cv2.fillConvexPoly(img, tri, 1)
    return img


class PatchAllocator:
    """Hands out free ``cell`` x ``cell`` texel blocks of an atlas; a block is painted with one colour and addressed by
    its centre UV, so every filtered lookup inside it returns exactly that colour."""

    def __init__(self, atlas: np.ndarray, uv_c: np.ndarray, grid: int = 256) -> None:
        self.atlas = atlas
        self.size = atlas.shape[0]
        self.grid = grid
        used = uv_coverage(uv_c, grid)
        used = cv2.dilate(used, np.ones((5, 5), np.uint8))
        self.free = np.argwhere(used == 0)  # (row, col) of free grid cells
        self.next = 0

    def paint(self, rgb_u8: tuple[int, int, int]) -> np.ndarray | None:
        """Paint the next free block; returns its centre UV (2,) or None when the atlas has no free block."""
        if self.next >= len(self.free):
            return None
        r, c = self.free[self.next]
        self.next += 1
        cell = self.size // self.grid
        r0, c0 = int(r) * cell, int(c) * cell
        self.atlas[r0 : r0 + cell, c0 : c0 + cell] = np.array(rgb_u8, dtype=np.uint8)
        return np.array([(c0 + cell / 2) / self.size, (r0 + cell / 2) / self.size], dtype=np.float32)
