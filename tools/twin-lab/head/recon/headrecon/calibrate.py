"""Per-photo weak-perspective cameras: coarse search + Powell refinement of the silhouette IoU between the (clipped)
twin head and the person mask of the photo, optionally tightened with face landmarks (front view)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from . import log
from .photos import Photo, head_blob
from .viewcam import ViewCam, iou, sample_surface, silhouette

SHRINK = 0.25


@dataclass
class Calib:
    cam: ViewCam
    iou: float
    cut_row: float  # photo row (full resolution) of the neck cut plane


class HeadFitMesh:
    """Welded, simplified head sub-mesh used for silhouette tests (vertices above ``y_cut``)."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray, y_cut: float, target_faces: int = 0):
        keep = (verts[faces][:, :, 1] > y_cut).all(axis=1)
        f = faces[keep]
        used, inv = np.unique(f, return_inverse=True)
        v, f = verts[used], inv.reshape(-1, 3)
        try:
            import fast_simplification as fs

            if target_faces and len(f) > target_faces:
                v, f = fs.simplify(v.astype(np.float32), f.astype(np.int32), target_count=target_faces)
                v, f = v.astype(np.float64), f.astype(np.int64)
        except ImportError:  # pragma: no cover
            pass
        self.verts, self.faces, self.y_cut = v, f, y_cut
        self.points = sample_surface(v, f)


def _cut_row(cam: ViewCam, y_cut: float, pivot_z: float) -> float:
    return float(cam.project(np.array([[0.0, y_cut, pivot_z]]))[0, 1])


def _loss_terms(cam: ViewCam, hm: HeadFitMesh, photo_mask_s: np.ndarray, pivot_z: float, height: int, width: int):
    """(1 - IoU) between the mesh silhouette and the photo mask, both clipped above the neck cut row."""
    row = _cut_row(cam, hm.y_cut, pivot_z) * SHRINK
    r = int(np.clip(round(row), 4, photo_mask_s.shape[0]))
    sm = silhouette(cam, hm.points, width, height, SHRINK)[:r]
    return 1.0 - iou(sm > 0, photo_mask_s[:r]), row


def calibrate_view(
    photo: Photo, nominal_yaw: float, hm: HeadFitMesh, ymax: float, pivot: np.ndarray, yaw_range: float = 10.0, pitch_range: float = 30.0,
    init: ViewCam | None = None, f_rel: float = 0.71, lm3d: np.ndarray | None = None, lm_px: np.ndarray | None = None,
) -> Calib:
    H, W = photo.mask.shape
    f = f_rel * max(H, W)
    mask_s = cv2.resize(photo.mask.astype(np.uint8), None, fx=SHRINK, fy=SHRINK, interpolation=cv2.INTER_AREA) > 0
    x0, y0, x1, y1 = head_blob(photo.mask)
    top = np.array([[0.0, ymax, pivot[2]]])
    rows = photo.mask[y0 : y0 + max(int(0.12 * (y1 - y0)), 8)]
    cols = np.flatnonzero(rows.any(axis=0))
    xc = float(cols.mean()) if len(cols) else 0.5 * (x0 + x1)
    pz = float(pivot[2])
    base = ViewCam(f=f, cx=W / 2, cy=H / 2, pivot=pivot.copy())

    def anchor(cam: ViewCam) -> ViewCam:
        # translation that puts the mesh top at the blob top and the mesh x-centre at the hair centre
        cam.tx = cam.ty = 0.0
        t = cam.project(top)[0]
        cam.tx, cam.ty = xc - t[0], y0 - t[1]
        return cam

    def prior(p: np.ndarray) -> float:
        return 0.03 * np.log(p[3] / 0.5) ** 2 + 0.0004 * (p[1] / 10.0) ** 2

    def loss(p: np.ndarray) -> float:
        pen = 0.0
        if init is not None:  # re-fits after a deformation stay close to the first calibration (soft bounds)
            ex = np.array([abs(p[0] - init.yaw) / 4.0, abs(p[1] - init.pitch) / 6.0, abs(p[2] - init.roll) / 4.0,
                           abs(np.log(p[3] / init.dist)) / 0.1])
            pen = 0.3 * float(np.sum(np.maximum(ex - 1.0, 0.0) ** 2))
        if p[3] < 0.33 or p[3] > 0.9 or abs(p[0] - nominal_yaw) > yaw_range or abs(p[1]) > pitch_range or abs(p[2]) > 10:
            return 10.0
        cam_p = base.with_params(p)
        val, _ = _loss_terms(cam_p, hm, mask_s, pz, H, W)
        if lm3d is not None and lm_px is not None:
            err = np.linalg.norm(cam_p.project(lm3d)[:, :2] - lm_px, axis=1)
            val += 0.0017 * float(np.sqrt(np.mean(err**2)))  # 30 px rms ~ 0.05 IoU
        return val + prior(p) + pen

    dists = np.geomspace(0.36, 0.8, 11) if init is None else [init.dist]
    pitches = tuple(x for x in (-30, -20, -10, 0, 10, 20, 30) if abs(x) <= pitch_range) if init is None else (init.pitch,)
    yaws = (-0.7 * yaw_range, 0.0, 0.7 * yaw_range) if init is None else (init.yaw - nominal_yaw,)
    cands: list[tuple[float, np.ndarray]] = []
    for yo in yaws:
        for pt in pitches:
            for d in dists:
                cam = anchor(base.with_params(np.array([nominal_yaw + yo, pt, 0.0, d, 0.0, 0.0])))
                cands.append((loss(cam.params()), cam.params()))
    cands.sort(key=lambda t: t[0])
    # silhouettes of a different-shaped head make the landscape multi-modal: refine the best few distinct starts
    starts: list[np.ndarray] = []
    for _l, p in cands:
        if all(abs(p[3] - q[3]) > 0.04 or abs(p[1] - q[1]) > 5 for q in starts):
            starts.append(p)
        if len(starts) == (3 if init is None else 1):
            break
    sc = np.array([4.0, 4.0, 3.0, 0.05, 0.01 * W, 0.01 * H])
    best_x, best_l = starts[0], float("inf")
    for x0_ in starts:
        x = x0_.copy()
        for _round in range(2):
            r = minimize(lambda z, x=x: loss(x + z * sc), np.zeros(6), method="Powell",
                         options={"xtol": 1e-2, "ftol": 1e-4, "maxiter": 40})
            x = x + r.x * sc
        lx = loss(x)
        if lx < best_l:
            best_l, best_x = lx, x
    x = best_x
    cam = base.with_params(x)
    val, row = _loss_terms(cam, hm, mask_s, pz, H, W)
    log(f"{photo.name}: yaw {cam.yaw:.1f} pitch {cam.pitch:.1f} roll {cam.roll:.1f} dist {cam.dist:.2f} m IoU {1 - val:.3f}")
    return Calib(cam, 1.0 - val, row / SHRINK)
