"""Hair mask on the face grid: the landmark oval reaches into the hair at the temples and above the forehead, but the
MakeHuman face is bare skin, so the relief must stop at the real hairline."""

from __future__ import annotations

import cv2
import numpy as np

from .facedepth import Grid
from .facefit import MEDIAPIPE_EYEBROWS, FaceMap
from .frontview import FrontView


def luminance(lin: np.ndarray) -> np.ndarray:
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _skin_reference(fv: FrontView, L_world: np.ndarray, lum_img: np.ndarray) -> float:
    from twintex.bake import remap_points  # noqa: E402  (texture stage)

    cheeks = np.array([50, 101, 118, 117, 205, 280, 330, 347, 346, 425])  # MediaPipe cheek landmarks
    cs = fv.world_to_photo(np.concatenate([L_world[cheeks], np.zeros((len(cheeks), 1))], axis=1))
    return float(np.median(remap_points(lum_img, cs[:, 0].astype(np.float32), cs[:, 1].astype(np.float32))))


def photo_luminance_on_grid(fv: FrontView, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    from twintex.bake import remap_points  # noqa: E402

    gx, gy = grid.centres()
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    s = fv.world_to_photo(P)
    lum_img = luminance(fv.view.image_lin).astype(np.float32)
    lum = remap_points(lum_img, s[:, 0].astype(np.float32), s[:, 1].astype(np.float32)).reshape(grid.H, grid.W)
    return lum, lum_img


def refine_oval(
    fv: FrontView,
    grid: Grid,
    L_world: np.ndarray,
    fm: FaceMap,
    oval: np.ndarray,
    dark_frac: float = 0.55,
    bright_frac: float = 0.62,
    brow_margin_mm: float = 4.0,
    max_up_mm: float = 36.0,
) -> tuple[np.ndarray, dict]:
    """Adjust the landmark oval to the photo: the face region ends at the real hairline.

    * dark (hair) pixels above the eyebrows are cut out of the oval;
    * the oval is grown upward over the bright forehead skin until the hair starts (the landmark oval stops at the
      middle of the forehead), within ``max_up_mm`` and the horizontal extent of the oval's top.
    """
    lum, lum_img = photo_luminance_on_grid(fv, grid)
    ref = _skin_reference(fv, L_world, lum_img)
    gx, gy = grid.centres()
    brow_top = float(L_world[list(MEDIAPIPE_EYEBROWS), 1].max()) + brow_margin_mm * 1e-3
    eye_y = float(L_world[[33, 133, 263, 362], 1].mean())  # eye corners
    sd0 = cv2.distanceTransform(oval.astype(np.uint8), cv2.DIST_L2, 5) * grid.res * 1000.0
    # hair: dark pixels above the eyes that are near the oval boundary (temples) or above the brows (forehead)
    hair_zone = ((gy > eye_y + 3e-3) & (sd0 < 24.0)) | (gy > brow_top)
    above = gy > brow_top
    # 1) hair inside the oval
    dark = (lum < dark_frac * ref) & hair_zone & oval
    k = max(int(round(1.5e-3 / grid.res)) | 1, 3)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    kc = max(int(round(6e-3 / grid.res)) | 1, 3)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
    ring = oval & ~cv2.erode(oval.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    n, lab = cv2.connectedComponents(dark)
    hair = np.zeros_like(oval)
    for c in range(1, n):
        m = lab == c
        if (m & ring).any() and m.sum() * grid.res**2 > 1.0e-4:  # > 1 cm^2
            hair |= m
    out = oval & ~cv2.dilate(hair.astype(np.uint8), np.ones((kc, kc), np.uint8)).astype(bool)
    # 2) grow upward over bright skin
    bright = cv2.morphologyEx((lum > bright_frac * ref).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool)
    top_rows = np.flatnonzero(oval.any(axis=1))
    xs = np.flatnonzero(oval[top_rows[0] : top_rows[0] + int(15e-3 / grid.res)].any(axis=0))
    x_lo, x_hi = (xs.min() - int(5e-3 / grid.res), xs.max() + int(5e-3 / grid.res)) if len(xs) else (0, grid.W)
    allowed = bright & above
    allowed[:, :max(x_lo, 0)] = False
    allowed[:, x_hi:] = False
    ext = out.copy()
    steps = int(max_up_mm * 1e-3 / grid.res)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(steps):
        grown = cv2.dilate(ext.astype(np.uint8), cross).astype(bool) & (allowed | out)
        if grown.sum() == ext.sum():
            break
        ext = grown
    ext = cv2.morphologyEx(ext.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8)).astype(bool)
    # keep the connected piece that holds the face
    n2, lab2 = cv2.connectedComponents(ext.astype(np.uint8))
    if n2 > 2:
        sizes = np.bincount(lab2.ravel())[1:]
        ext = lab2 == (1 + int(np.argmax(sizes)))
    ext = cv2.morphologyEx(ext.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    return ext, {"hair_px": int(hair.sum()), "grown_px": int(ext.sum() - out.sum()), "skin_ref_lum": ref}
