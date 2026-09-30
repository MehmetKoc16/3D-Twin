#!/usr/bin/env python
"""Tiny CPU flat-shaded mesh previewer (numpy + OpenCV painter's algorithm, orthographic). No GPU, no GL context.

Views use the output convention (+Y up, character faces +Z). yaw 0 = camera in front (+Z), 90 = camera on the
character's left side (+X), 180 = back, 270 = right side. Meshes are expected in metres.

    python preview.py path/to/mesh.glb  out_dir            # front / 3-4 / side / back + head + feet/hands crops
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh

LIGHT = np.array([0.35, 0.55, 0.75])
LIGHT /= np.linalg.norm(LIGHT)
FILL = np.array([-0.6, 0.1, 0.5])
FILL /= np.linalg.norm(FILL)
CLAY = np.array([0.80, 0.76, 0.72])  # RGB


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def render(
    v: np.ndarray,
    f: np.ndarray,
    yaw_deg: float,
    center_y: float,
    view_h: float,
    size: tuple[int, int] = (600, 900),
    center_x: float = 0.0,
    ss: int = 2,
    bg: int = 236,
    smooth: bool = False,
) -> np.ndarray:
    """Render one view. `view_h` = vertical extent (m) mapped to the image height; `center_y` the world height at the
    image centre. Returns an RGB uint8 image of `size` = (width, height)."""
    w, h = size
    a = math.radians(yaw_deg)
    ca, sa = math.cos(a), math.sin(a)
    x = v[:, 0] * ca - v[:, 2] * sa
    z = v[:, 0] * sa + v[:, 2] * ca
    vr = np.stack([x, v[:, 1], z], axis=1)

    tri = vr[f]  # (m,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-12)
    facing = n[:, 2] > 0.0
    order = np.argsort(tri[:, :, 2].mean(axis=1))  # far -> near
    order = order[facing[order]]

    if smooth:
        vn = np.zeros_like(vr)
        np.add.at(vn, f[:, 0], n * ln)
        np.add.at(vn, f[:, 1], n * ln)
        np.add.at(vn, f[:, 2], n * ln)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        nf = vn[f].mean(axis=1)
        nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-12)
    else:
        nf = n
    shade = 0.28 + 0.62 * np.clip(nf @ LIGHT, 0, 1) + 0.18 * np.clip(nf @ FILL, 0, 1)
    shade = np.clip(shade, 0, 1) ** (1 / 1.1)
    cols = (CLAY[None, :] * shade[:, None] * 255.0).astype(np.uint8)  # RGB

    W, H = w * ss, h * ss
    s = H / view_h
    px = (W * 0.5 + (tri[:, :, 0] - center_x) * s)
    py = (H * 0.5 - (tri[:, :, 1] - center_y) * s)
    pts = np.round(np.stack([px, py], axis=2) * 16.0).astype(np.int32)

    img = np.full((H, W, 3), bg, dtype=np.uint8)
    for i in order:
        c = cols[i]
        cv2.fillConvexPoly(img, pts[i], (int(c[0]), int(c[1]), int(c[2])), lineType=cv2.LINE_8, shift=4)
    if ss > 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def make_previews(mesh_path: str | Path, out_dir: str | Path, prefix: str = "") -> list[Path]:
    """Write the standard preview set for a normalised mesh (metres, feet on y = 0). Returns the written files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = load_mesh(mesh_path)
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces)
    height = float(v[:, 1].max() - v[:, 1].min())
    ymin = float(v[:, 1].min())
    files: list[Path] = []

    def save(name: str, img: np.ndarray) -> None:
        p = out / f"{prefix}{name}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        files.append(p)

    full = {}
    for name, yaw in (("front", 0.0), ("threequarter", 35.0), ("side", 90.0), ("back", 180.0)):
        img = render(v, f, yaw, center_y=ymin + height / 2, view_h=height * 1.06, size=(520, 900))
        full[name] = img
        save(name, label(img, name))
    save("contact", np.concatenate([label(full[k], k) for k in ("front", "threequarter", "side", "back")], axis=1))

    # head close-ups (top 16 % of the body)
    head_c = ymin + height * 0.925
    head_h = height * 0.17
    heads = [render(v, f, yaw, head_c, head_h, size=(560, 700), smooth=True) for yaw in (0.0, 35.0, 90.0)]
    save("head", np.concatenate(heads, axis=1))
    # hands hang at roughly 40-50 % of the height, feet are in the bottom 12 %
    hands = render(v, f, 0.0, ymin + height * 0.44, height * 0.30, size=(1000, 500), smooth=True)
    feet = render(v, f, 0.0, ymin + height * 0.07, height * 0.20, size=(1000, 500), smooth=True)
    feet_side = render(v, f, 90.0, ymin + height * 0.07, height * 0.20, size=(500, 500), smooth=True)
    save("hands", hands)
    save("feet", np.concatenate([feet, feet_side], axis=1))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    for p in make_previews(args.mesh, args.out_dir, args.prefix):
        print(p)


if __name__ == "__main__":
    main()
