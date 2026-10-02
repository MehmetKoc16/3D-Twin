"""Face landmarks of the front photo (MediaPipe through the refine stage's detector, on a head crop)."""

from __future__ import annotations

import numpy as np
from twinrefine.landmarks import detect_landmarks

from .photos import Photo, head_blob


def head_crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Square crop (x0, y0, x1, y1) around the hair-and-face blob of a close-up portrait."""
    x0, y0, x1, y1 = head_blob(mask)
    row = mask[min(y0 + 200, mask.shape[0] - 1)]
    cols = np.flatnonzero(row)
    w = int(cols.max() - cols.min()) if len(cols) else (x1 - x0)
    cx = int((cols.max() + cols.min()) // 2) if len(cols) else (x0 + x1) // 2
    s = int(2.0 * w)
    cy0 = max(y0 - int(0.1 * w), 0)
    cx0 = int(np.clip(cx - s // 2, 0, max(mask.shape[1] - s, 0)))
    return cx0, cy0, min(cx0 + s, mask.shape[1]), min(cy0 + s, mask.shape[0])


def front_landmarks(photo: Photo, detector=None) -> np.ndarray | None:
    """(468, 2) landmark pixels in the photo (original image coordinates), or None when no face is found."""
    x0, y0, x1, y1 = head_crop(photo.mask)
    crop = photo.rgb[y0:y1, x0:x1]
    detect = detector or (lambda rgb: detect_landmarks(rgb, None, full_frame=True))
    lm = detect(crop)
    if lm is None:
        return None
    return lm.xy[:468] + np.array([x0, y0], dtype=np.float64)
