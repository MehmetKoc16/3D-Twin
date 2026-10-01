"""2-D face landmarks with MediaPipe Face Landmarker (Apache-2.0), run locally on the front photo.

The model file is the one the web app already ships (``apps/web/public/models/face_landmarker.task``). Nothing leaves
the machine. A full-body photo is cropped around the head first (the face is only ~10 % of its height, below what the
face detector handles) and the crop is upscaled; the returned landmarks are in pixels of the ORIGINAL image.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import LANDMARKER_TASK, log


@dataclass
class FaceLandmarks:
    xy: np.ndarray  # (478, 2) pixels of the original image (x right, y down)
    z: np.ndarray  # (478,) depth in pixel units of the original image (MediaPipe's relative z; smaller = closer)
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 of the detector input in original pixels
    score: float = 1.0


def _import_mediapipe():
    """mediapipe's vision package imports matplotlib for its drawing helpers only; stub it when it is missing."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the environment
        for name in ("matplotlib", "matplotlib.pyplot"):
            sys.modules[name] = types.ModuleType(name)
    from mediapipe.tasks.python import BaseOptions, vision

    return BaseOptions, vision


def person_head_crop(alpha: np.ndarray | None, shape: tuple[int, int], side_frac: float = 0.20) -> tuple[int, int, int, int]:
    """Square crop around the head of a full-body photo from its person mask (or the top-centre when there is none)."""
    H, W = shape
    if alpha is None or not (alpha > 0.5).any():
        s = int(side_frac * H)
        return max(W // 2 - s // 2, 0), 0, min(W // 2 + s // 2, W), min(s, H)
    mask = alpha > 0.5
    ys = np.flatnonzero(mask.any(axis=1))
    y0, y1 = int(ys[0]), int(ys[-1])
    body_h = y1 - y0 + 1
    top_rows = mask[y0 : y0 + max(int(0.06 * body_h), 3)]
    cols = np.flatnonzero(top_rows.any(axis=0))
    xc = int(cols.mean()) if len(cols) else W // 2
    s = int(side_frac * body_h)
    x0 = int(np.clip(xc - s // 2, 0, max(W - s, 0)))
    y_start = int(np.clip(y0 - 0.03 * s, 0, max(H - s, 0)))
    return x0, y_start, min(x0 + s, W), min(y_start + s, H)


def _run_detector(rgb: np.ndarray, conf: float):
    BaseOptions, vision = _import_mediapipe()
    import mediapipe as mp

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_TASK)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=conf,
        min_face_presence_confidence=conf,
        output_face_blendshapes=False,
    )
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return lm.detect(img)


def detect_landmarks(
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    full_frame: bool = False,
    target: int = 768,
) -> FaceLandmarks | None:
    """Detect the 478 landmarks of the single face in ``rgb`` (uint8 HxWx3). ``full_frame``: the image is already a
    head-and-shoulders portrait (no head crop). Returns None when no face is found."""
    if not LANDMARKER_TASK.exists():
        log(f"landmarker model missing: {LANDMARKER_TASK}")
        return None
    H, W = rgb.shape[:2]
    attempts: list[tuple[tuple[int, int, int, int], float]] = []
    if full_frame:
        attempts.append(((0, 0, W, H), 0.5))
        attempts.append(((0, 0, W, H), 0.2))
    else:
        for frac in (0.20, 0.16, 0.26):
            c = person_head_crop(alpha, (H, W), frac)
            attempts += [(c, 0.5), (c, 0.2)]
    for crop, conf in attempts:
        x0, y0, x1, y1 = crop
        sub = rgb[y0:y1, x0:x1]
        scale = target / max(sub.shape[:2])
        if scale > 1.0 or scale < 0.5:
            sub = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        res = _run_detector(sub, conf)
        if res.face_landmarks:
            lms = res.face_landmarks[0]
            sh, sw = sub.shape[:2]
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            cw, ch = x1 - x0, y1 - y0
            xy = np.stack([x0 + pts[:, 0] * cw, y0 + pts[:, 1] * ch], axis=1)
            z = pts[:, 2] * cw
            return FaceLandmarks(xy, z, crop)
    return None


def draw_landmarks(rgb: np.ndarray, lm: FaceLandmarks, crop: bool = True, scale: int = 3) -> np.ndarray:
    """Debug overlay: landmark dots (green) on the head crop, upscaled."""
    x0, y0, x1, y1 = lm.crop
    sub = rgb[y0:y1, x0:x1] if crop else rgb
    img = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).copy()
    for x, y in lm.xy:
        cv2.circle(img, (int((x - (x0 if crop else 0)) * scale), int((y - (y0 if crop else 0)) * scale)), 2, (0, 255, 0), -1)
    return img


def load_landmarker_path() -> Path:
    return LANDMARKER_TASK
