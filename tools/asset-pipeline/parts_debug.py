"""Debug renders for the body-part proxies (not shipped): orthographic, z-buffered, textured with an alpha cutoff.

`Scene` draws a base (the body) once and then each set of part layers on top of a copy, from the front (looking along -Z)
or the side (from +X); `parts.write_debug_images` writes `.cache/debug/parts_*.png`.
"""

from __future__ import annotations

import numpy as np


def sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, 4) uint8 texture at glTF-convention uv (v down). Returns (n, 4) float 0..255."""
    h, w, _ = tex.shape
    x = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
    y = np.clip(uv[:, 1], 0.0, 1.0) * (h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = tex[y0, x0].astype(np.float64) * (1 - fx) + tex[y0, x1].astype(np.float64) * fx
    bot = tex[y1, x0].astype(np.float64) * (1 - fx) + tex[y1, x1].astype(np.float64) * fx
    return top * (1 - fy) + bot * fy


def _raster_layer(img, zbuf, pts, depth, tris, shade, color, uv=None, tex=None, cutoff=0.5, tint=None) -> None:
    h, w, _ = img.shape
    color = np.asarray(color, dtype=np.float64)
    for t, tri in enumerate(tris):
        p = pts[tri]
        x0, x1 = int(max(np.floor(p[:, 0].min()), 0)), int(min(np.ceil(p[:, 0].max()), w - 1))
        y0, y1 = int(max(np.floor(p[:, 1].min()), 0)), int(min(np.ceil(p[:, 1].max()), h - 1))
        if x1 < x0 or y1 < y0:
            continue
        den = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(den) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        l1 = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / den
        l2 = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / den
        l3 = 1 - l1 - l2
        m = (l1 >= -0.01) & (l2 >= -0.01) & (l3 >= -0.01)
        if not m.any():
            continue
        z = l1 * depth[tri[0]] + l2 * depth[tri[1]] + l3 * depth[tri[2]]
        sub = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        upd = m & (z > sub)
        if not upd.any():
            continue
        if tex is not None and uv is not None:
            u = l1[upd] * uv[tri[0]][0] + l2[upd] * uv[tri[1]][0] + l3[upd] * uv[tri[2]][0]
            v = l1[upd] * uv[tri[0]][1] + l2[upd] * uv[tri[1]][1] + l3[upd] * uv[tri[2]][1]
            s = sample_texture(tex, np.stack([u, v], axis=1))
            keep = s[:, 3] / 255.0 >= cutoff
            rgb = s[:, :3] * (color / 255.0 if tint is None else np.asarray(tint, dtype=np.float64) / 255.0)
            sel = np.zeros_like(upd)
            sel[upd] = keep
            zsub = sub
            zsub[sel] = z[sel]
            if keep.any():
                out = img[y0 : y1 + 1, x0 : x1 + 1]
                out[sel] = (rgb[keep] * shade[t]).clip(0, 255).astype(np.uint8)
        else:
            sub[upd] = z[upd]
            img[y0 : y1 + 1, x0 : x1 + 1][upd] = (color * shade[t]).clip(0, 255).astype(np.uint8)


class Scene:
    """Orthographic camera over a region, with a pre-drawn base (the body) that is reused for every `render`.

    view 'front': camera on +Z looking at -Z, image x = world x; 'side': camera on +X, image x = -world z (face left).
    bbox (min, max): the world region shown (the longest axis of its projection fills `size`).
    """

    def __init__(self, base_layers: list[dict], view: str, bbox: tuple[np.ndarray, np.ndarray], size: int = 800,
                 background=(238, 238, 244)) -> None:
        lo, hi = bbox
        if view == "front":
            self.ax_u, self.sign_u, self.dep = 0, 1.0, 2
        else:
            self.ax_u, self.sign_u, self.dep = 2, -1.0, 0
        u0, u1 = sorted((self.sign_u * lo[self.ax_u], self.sign_u * hi[self.ax_u]))
        v0, v1 = lo[1], hi[1]
        self.u0, self.v1 = u0, v1
        self.scale = size / max(u1 - u0, v1 - v0)
        w, h = int((u1 - u0) * self.scale) + 1, int((v1 - v0) * self.scale) + 1
        self.img = np.empty((h, w, 3), dtype=np.uint8)
        self.img[:] = background
        self.zbuf = np.full((h, w), -1e9)
        self._draw(self.img, self.zbuf, base_layers)

    def _draw(self, img, zbuf, layers) -> None:
        h, w, _ = img.shape
        for layer in layers:
            pos = layer["pos"]
            pts = np.stack([(self.sign_u * pos[:, self.ax_u] - self.u0) * self.scale, (self.v1 - pos[:, 1]) * self.scale], axis=1)
            depth = pos[:, self.dep]
            tris = layer["tris"]
            # drop triangles entirely outside the image before the per-triangle loop
            tp = pts[tris]
            inside = (tp[:, :, 0].max(1) >= 0) & (tp[:, :, 0].min(1) <= w) & (tp[:, :, 1].max(1) >= 0) & (tp[:, :, 1].min(1) <= h)
            tris = tris[inside]
            n = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            facing = n[:, self.dep]
            shade = 0.5 + 0.5 * np.abs(facing)
            if layer.get("cull"):  # single-sided material: skip back faces (the body's socket cavity faces into the head)
                tris, shade = tris[facing > 0], shade[facing > 0]
            _raster_layer(img, zbuf, pts, depth, tris, shade, layer["color"], layer.get("uv"), layer.get("tex"),
                          layer.get("cutoff", 0.5), layer.get("tint"))

    def render(self, layers: list[dict]) -> np.ndarray:
        img, zbuf = self.img.copy(), self.zbuf.copy()
        self._draw(img, zbuf, layers)
        return img
