"""Compose the final atlas: observed colour + confidence-weighted 3-D fill + gutter padding."""

from __future__ import annotations

import cv2
import numpy as np

from .bake import BakeConfig, BakeResult
from .colorspace import linear_to_u8, u8_to_linear
from .fill import dilate_fill, mirror_fill, nearest_visible_fill, push_pull
from .prior import hair_prior, opposite_prior
from .views import View


def compute_fill_grid(
    bake: BakeResult,
    cfg: BakeConfig,
    x_mid: float,
    mirror: bool = True,
    views: list[View] | None = None,
    hair: bool = True,
    log=print,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fill colours for the coarse cloud, scattered to a (Sc, Sc, 3) grid + validity (Sc, Sc)."""
    cl = bake.cloud
    pos, nrm, col, conf = cl["pos"], cl["nrm"], cl["col"], cl["conf"]
    Sc = cl["grid"]
    info: dict = {"cells": int(len(pos))}
    conf2, col2 = conf, col
    mirrored = np.zeros(len(pos), bool)
    if mirror:
        conf2, col2 = mirror_fill(pos, nrm, col, conf, x_mid)
        mirrored = conf2 != conf
    info["mirrored_cells"] = int(mirrored.sum())
    fill = nearest_visible_fill(pos, nrm, col2, conf2)
    ymin, ymax = float(pos[:, 1].min()), float(pos[:, 1].max())
    if views:
        # surfaces facing away from a view without an opposite partner: shine the low-pass image through
        pcol, pw = opposite_prior(views, pos, nrm, cfg, ymax - ymin)
        fill = pw[:, None] * pcol + (1 - pw[:, None]) * fill
        info["prior_cells"] = int((pw > 0.5).sum())
        if hair and any(v.name == cfg.face_view for v in views):
            hp = hair_prior(pos, nrm, col, conf, ymin, ymax, cfg.head_frac)
            if hp is not None:
                hair_col, hw = hp
                # the hair prior only applies to what the face view did not see (weights vanish for seen cells)
                hw = hw * (1.0 - np.clip((conf2 - 0.3) / 0.5, 0, 1))
                fill = hw[:, None] * hair_col[None, :] + (1 - hw[:, None]) * fill
                info["hair_colour_srgb"] = [round(float(x), 3) for x in linear_to_u8(hair_col[None])[0] / 255.0]
    fill[mirrored] = col2[mirrored]
    log(f"  fill: {len(pos)} cells, {int(mirrored.sum())} mirrored, {int((conf2 < 0.6).sum())} extrapolated")
    grid = np.zeros((Sc * Sc, 3), np.float32)
    val = np.zeros(Sc * Sc, np.float32)
    grid[cl["cell"]] = fill
    val[cl["cell"]] = 1.0
    return grid.reshape(Sc, Sc, 3), val.reshape(Sc, Sc), info


def compose_atlas(
    bake: BakeResult,
    cfg: BakeConfig,
    x_mid: float,
    mirror: bool = True,
    padding: int = 8,
    views: list[View] | None = None,
    hair: bool = True,
    log=print,
) -> tuple[np.ndarray, dict]:
    """Return the final sRGB uint8 atlas (S, S, 3) and an info dict."""
    S, B = cfg.size, cfg.coarse
    grid, gval, info = compute_fill_grid(bake, cfg, x_mid, mirror, views, hair, log)
    Sc = grid.shape[0]
    out = np.zeros((S, S, 3), np.uint8)
    conf_all = bake.conf
    # per-band: upsample fill (normalised, so empty cells do not leak) and blend with the observation
    for r0 in range(0, S, cfg.band_rows):
        r1 = min(S, r0 + cfg.band_rows)
        c0 = max(r0 // B - 2, 0)
        c1 = min(-(-r1 // B) + 2, Sc)
        num = cv2.resize(grid[c0:c1] * gval[c0:c1, :, None], (S, (c1 - c0) * B), interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(gval[c0:c1], (S, (c1 - c0) * B), interpolation=cv2.INTER_LINEAR)
        off = r0 - c0 * B
        num = num[off : off + (r1 - r0)]
        den = den[off : off + (r1 - r0)]
        fill_up = num / np.maximum(den, 1e-6)[..., None]
        valid = bake.valid[r0:r1]
        conf = conf_all[r0:r1].astype(np.float32) / 255.0
        obs = u8_to_linear(bake.color_u8[r0:r1])
        # where the coarse fill has no support (thin slivers), fall back to the observation
        fill_ok = den > 1e-3
        a = np.where(fill_ok, conf, 1.0)[..., None]
        lin = a * obs + (1 - a) * fill_up
        res = linear_to_u8(lin)
        res[~valid] = 0
        out[r0:r1] = res
    # ---- gutters: grow charts outwards, then fill the unused rest with a smooth push-pull background
    valid = bake.valid
    out = _pad_atlas(out, valid, padding, cfg.band_rows)
    info["padding"] = padding
    return out, info


def _pad_atlas(atlas: np.ndarray, valid: np.ndarray, padding: int, band: int) -> np.ndarray:
    S = atlas.shape[0]
    out = atlas.copy()
    margin = padding + 1
    for r0 in range(0, S, band):
        r1 = min(S, r0 + band)
        a0, a1 = max(r0 - margin, 0), min(r1 + margin, S)
        img = u8_to_linear(atlas[a0:a1])
        img, m = dilate_fill(img, valid[a0:a1], padding)
        res = linear_to_u8(img)
        res[~m] = 0
        out[r0:r1] = res[r0 - a0 : r0 - a0 + (r1 - r0)]
    # unused space: low-res push-pull so mip levels do not pick up black
    small = 1024
    lin_small = u8_to_linear(cv2.resize(out, (small, small), interpolation=cv2.INTER_AREA))
    v_small = cv2.resize(valid.astype(np.float32), (small, small), interpolation=cv2.INTER_AREA)
    filled = push_pull(lin_small, np.clip(v_small * 4.0, 0, 1))
    bg = cv2.resize(linear_to_u8(filled), (S, S), interpolation=cv2.INTER_LINEAR)
    padded_mask = cv2.dilate(valid.astype(np.uint8), np.ones((2 * padding + 1, 2 * padding + 1), np.uint8)) > 0
    out[~padded_mask] = bg[~padded_mask]
    return out
