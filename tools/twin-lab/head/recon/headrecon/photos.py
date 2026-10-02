"""The four head photos: images, person masks (rembg through the shape stage's Python, GrabCut fallback), and the
optional de-glassed versions (``<photos>/clean/<name>.jpg`` + ``<name>_mask.png``)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import LAB_DIR, RECON_DIR, log

VIEWS = ("front", "back", "profile_nose_right", "profile_nose_left")
# Nominal camera azimuth around the head: 0 front, +90 on the character's left (+X), 180 back. A nose pointing to the
# image right means the camera is on the character's right side (-90), a nose pointing left on the left side (+90).
NOMINAL_YAW = {"front": 0.0, "back": 180.0, "profile_nose_right": -90.0, "profile_nose_left": 90.0}


@dataclass
class Photo:
    name: str
    rgb: np.ndarray  # (H, W, 3) uint8, original (geometry)
    tex_rgb: np.ndarray  # same size; de-glassed when available (texture)
    mask: np.ndarray  # (H, W) bool person mask
    glass: np.ndarray  # (H, W) bool pixels that must not feed the texture (glasses), may be all False
    clean: bool = False


def load_rgb(path: Path) -> np.ndarray:
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def grabcut_person(rgb: np.ndarray) -> np.ndarray:
    H, W = rgb.shape[:2]
    s = 512.0 / max(H, W)
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = np.full(small.shape[:2], cv2.GC_PR_BGD, np.uint8)
    h, w = small.shape[:2]
    m[int(h * 0.15):, int(w * 0.15):int(w * 0.85)] = cv2.GC_PR_FGD
    m[: int(h * 0.05)] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(small, m, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    out = np.isin(m, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST) > 0


def segment_person(paths: list[Path], cache: Path) -> dict[str, np.ndarray]:
    """Person masks. rembg u2net_human_seg runs in the shape stage's environment (it has rembg); cached per file."""
    cache.mkdir(parents=True, exist_ok=True)
    need = [p for p in paths if not (cache / f"mask_{p.stem}.png").exists()
            or (cache / f"mask_{p.stem}.png").stat().st_mtime < p.stat().st_mtime]
    if need:
        py = LAB_DIR / "shape" / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if py.exists():
            r = subprocess.run([str(py), str(RECON_DIR / "segment_worker.py"), str(cache), *map(str, need)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                log(f"segmentation worker failed: {r.stderr[-400:]}")
        else:
            log("shape venv not found: GrabCut fallback for the person masks")
    out = {}
    for p in paths:
        mp = cache / f"mask_{p.stem}.png"
        out[p.stem] = (np.array(Image.open(mp)) > 127) if mp.exists() else grabcut_person(load_rgb(p))
    return out


def load_photos(photo_dir: Path, cache: Path, use_clean: bool = True) -> dict[str, Photo]:
    paths = [photo_dir / f"{v}.jpg" for v in VIEWS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing head photos: {missing}")
    masks = segment_person(paths, cache)
    out: dict[str, Photo] = {}
    for v, p in zip(VIEWS, paths, strict=True):
        rgb = load_rgb(p)
        tex, clean, glass = rgb, False, np.zeros(rgb.shape[:2], bool)
        cp = photo_dir / "clean" / f"{v}.jpg"
        if use_clean and cp.exists():
            c = load_rgb(cp)
            if c.shape == rgb.shape:
                tex, clean = c, True
        if use_clean:
            for gm in (photo_dir / "clean" / f"{v}_mask.png", photo_dir / "clean" / f"mask_{v}.png"):
                if gm.exists():
                    g = np.array(Image.open(gm).convert("L")) > 127
                    if g.shape == rgb.shape[:2]:
                        glass = g
        out[v] = Photo(v, rgb, tex, masks[v], glass, clean)
    return out


def head_blob(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the largest mask component (the person)."""
    n, _lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0, 0, mask.shape[1], mask.shape[0]
    best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _a = st[best]
    return int(x), int(y), int(x + w), int(y + h)
