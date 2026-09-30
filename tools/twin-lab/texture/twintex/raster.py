"""Vectorised triangle rasterisers (numpy only).

Pixel (px, py) has its centre at (px + 0.5, py + 0.5) in the coordinate system of the projected
vertices, i.e. coordinates are in *pixel units* with the origin at the top-left corner and y pointing down.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

MAX_PAIRS = 1_500_000


def _next_pow2(a: np.ndarray) -> np.ndarray:
    a = np.maximum(a, 1)
    return (1 << np.ceil(np.log2(a)).astype(np.int64)).astype(np.int64)


def raster_pairs(
    xy: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    row_range: tuple[int, int] | None = None,
    eps: float = 1e-7,
    max_pairs: int = MAX_PAIRS,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield batches ``(face_idx, px, py, bary)`` of pixel centres covered by triangles.

    Triangles are bucketed by bounding-box size so that each bucket is evaluated with dense numpy
    broadcasting. Winding is ignored. ``bary`` has shape (K, 3) and weights the triangle's 3 vertices.
    ``row_range`` restricts the output to rows ``[r0, r1)`` (used for banded processing).
    """
    xy = np.asarray(xy, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = xy[faces]  # (F, 3, 2)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    mn = tri.min(axis=1)
    mx = tri.max(axis=1)
    r0, r1 = (0, height) if row_range is None else row_range
    j0 = np.maximum(np.ceil(mn[:, 0] - 0.5 - 1e-6), 0).astype(np.int64)
    j1 = np.minimum(np.floor(mx[:, 0] - 0.5 + 1e-6), width - 1).astype(np.int64)
    i0 = np.maximum(np.ceil(mn[:, 1] - 0.5 - 1e-6), r0).astype(np.int64)
    i1 = np.minimum(np.floor(mx[:, 1] - 0.5 + 1e-6), r1 - 1).astype(np.int64)
    ok = (np.abs(det) > 1e-12) & (j1 >= j0) & (i1 >= i0)
    fidx = np.flatnonzero(ok)
    if fidx.size == 0:
        return
    bw = j1[fidx] - j0[fidx] + 1
    bh = i1[fidx] - i0[fidx] + 1
    # exact buckets for small boxes, power-of-two buckets for big ones
    kw = np.where(bw <= 16, bw, _next_pow2(bw))
    kh = np.where(bh <= 16, bh, _next_pow2(bh))
    key = kw * 65536 + kh
    order = np.argsort(key, kind="stable")
    fidx, kw, kh, key = fidx[order], kw[order], kh[order], key[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [fidx.size]])
    for s, e in zip(starts, ends, strict=True):
        BW, BH = int(kw[s]), int(kh[s])
        per = max(1, max_pairs // (BW * BH))
        for cs in range(s, e, per):
            f = fidx[cs : min(e, cs + per)]
            jj = j0[f][:, None, None] + np.arange(BW)[None, None, :]
            ii = i0[f][:, None, None] + np.arange(BH)[None, :, None]
            inb = (jj <= j1[f][:, None, None]) & (ii <= i1[f][:, None, None])
            px = jj + 0.5
            py = ii + 0.5
            ax = a[f, 0][:, None, None]
            ay = a[f, 1][:, None, None]
            bx = b[f, 0][:, None, None]
            by = b[f, 1][:, None, None]
            cx = c[f, 0][:, None, None]
            cy = c[f, 1][:, None, None]
            d = det[f][:, None, None]
            lb = ((px - ax) * (cy - ay) - (cx - ax) * (py - ay)) / d
            lc = ((bx - ax) * (py - ay) - (px - ax) * (by - ay)) / d
            la = 1.0 - lb - lc
            inside = inb & (la >= -eps) & (lb >= -eps) & (lc >= -eps)
            ti, yi, xi = np.nonzero(inside)
            if ti.size == 0:
                continue
            bary = np.stack([la[ti, yi, xi], lb[ti, yi, xi], lc[ti, yi, xi]], axis=1)
            yield f[ti], jj[ti, 0, xi], ii[ti, yi, 0], bary


def rasterize_uv(
    uv_px: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    row_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise a UV layout (texel units) into a triangle-id map and a barycentric map.

    Returns ``tri_id`` (rows, width) int32 with -1 for empty texels and ``bary`` (rows, width, 3) float32.
    Only rows ``[r0, r1)`` are produced when ``row_range`` is given (output row 0 = ``r0``).
    """
    r0, r1 = (0, height) if row_range is None else row_range
    rows = r1 - r0
    tri_id = np.full((rows, width), -1, dtype=np.int32)
    bary = np.zeros((rows, width, 3), dtype=np.float32)
    for f, px, py, lam in raster_pairs(uv_px, faces, width, height, row_range=(r0, r1)):
        tri_id[py - r0, px] = f
        bary[py - r0, px] = lam
    return tri_id, bary


def zbuffer(
    xyz: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a depth map and a triangle-id map. ``xyz`` = (pixel x, pixel y, depth), smaller depth = nearer.

    Returns ``depth`` (H, W) float32 (``inf`` where empty) and ``fid`` (H, W) int32 (-1 where empty).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    zbuf = np.full(width * height, np.inf, dtype=np.float32)
    fid = np.full(width * height, -1, dtype=np.int32)
    z = xyz[:, 2]
    for f, px, py, lam in raster_pairs(xyz[:, :2], faces, width, height):
        d = np.einsum("kj,kj->k", lam, z[faces[f]]).astype(np.float32)
        pix = py * width + px
        # write far-to-near so that (last write wins) leaves the nearest fragment of the batch
        order = np.argsort(-d, kind="stable")
        d, pix, f = d[order], pix[order], f[order]
        keep = d < zbuf[pix]
        zbuf[pix[keep]] = d[keep]
        fid[pix[keep]] = f[keep]
    return zbuf.reshape(height, width), fid.reshape(height, width)


def barycentric_at(
    xy: np.ndarray, faces: np.ndarray, fid: np.ndarray, px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    """Barycentric coordinates of the pixel centres (px+.5, py+.5) inside triangles ``fid``."""
    tri = np.asarray(xy, dtype=np.float64)[faces[fid]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    p = np.stack([px + 0.5, py + 0.5], axis=1)
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    lb = ((p[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1])) / det
    lc = ((b[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1]) - (p[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
    return np.stack([1.0 - lb - lc, lb, lc], axis=1)
