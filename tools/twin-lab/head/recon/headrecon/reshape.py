"""Silhouette-driven reshaping of the twin head.

Each round: for every photo the contour of the (clipped) head silhouette is matched to the contour of the photo mask;
the silhouette vertices get image-plane displacement targets (converted to world space through the camera), the front
photo's face landmarks add targets for the face features, and a smooth 3-D thin-plate field interpolates them over the
whole head. The field fades to zero over the neck. Hair volume, head width, profile and the back of the head follow the
photos; a few rounds of contour matching approximate the visual hull of the four silhouettes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from twinrefine.scan import weld_ids, welded_faces

from . import log
from .calibrate import SHRINK, Calib
from .photos import Photo
from .twinhead import TwinHead
from .viewcam import ViewCam, sample_surface, silhouette

# stable face landmarks (MediaPipe indices): eyes, brows, nose, mouth and the lower jaw line
STABLE_LM = [33, 133, 263, 362, 159, 386, 55, 285, 105, 334, 1, 168, 129, 358, 61, 291, 13, 14, 152, 172, 397, 234, 454]


@dataclass
class ReshapeParams:
    rounds: int = 6
    step: float = 0.85  # fraction of the matched displacement applied per round
    smoothing: float = 2e-3  # thin-plate smoothing (larger = stiffer field)
    max_offset_px: float = 260.0  # contour matches further away than this are ignored (other objects, mismatch)
    max_move_m: float = 0.05  # cap of the per-round displacement of a constraint
    voxel_m: float = 0.008  # constraints are averaged on this grid
    landmark_weight: float = 0.7
    max_landmark_move_m: float = 0.02
    use_landmarks: bool = True


@dataclass
class ReshapeResult:
    verts: np.ndarray  # (n, 3) new positions (index space)
    disp: np.ndarray  # (n, 3)
    gate: np.ndarray  # (n,) neck falloff weight 0..1
    stats: dict = field(default_factory=dict)
    calibs: dict = field(default_factory=dict)
    y_cut: float = 0.0
    lm_pos: np.ndarray | None = None  # final 3-D positions of the tracked landmarks


def _largest(mask: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    return lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


def _contour(mask: np.ndarray, max_row: int) -> np.ndarray:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2))
    c = max(cnts, key=len)[:, 0, :].astype(np.float64)
    return c[c[:, 1] < max_row - 4]


def contour_constraints(
    cam: ViewCam, calib: Calib, photo_mask_s: np.ndarray, Vw: np.ndarray, Nw: np.ndarray, Fw: np.ndarray,
    head_idx: np.ndarray, y_cut: float, pivot_z: float, prm: ReshapeParams, size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Vertex ids (into Vw) and world displacement targets from matching the silhouette contours in one view."""
    H, W = size
    keep = (Vw[Fw][:, :, 1] > y_cut).all(axis=1)
    pts = sample_surface(Vw, Fw[keep])
    r = int(np.clip(round(cam.project(np.array([[0.0, y_cut, pivot_z]]))[0, 1] * SHRINK), 8, photo_mask_s.shape[0]))
    sm = _largest(silhouette(cam, pts, W, H, SHRINK) > 0)
    cm = _contour(sm, r)
    cp = _contour(_largest(photo_mask_s), r)
    if len(cm) < 20 or len(cp) < 20:
        return np.zeros(0, np.int64), np.zeros((0, 3)), float("nan")
    tree = cKDTree(cp)
    dist, j = tree.query(cm)
    off = (cp[j] - cm) / SHRINK  # full-resolution pixels
    ok = dist / SHRINK < prm.max_offset_px
    cm, off = cm[ok], off[ok]
    mean_px = float(np.mean(np.linalg.norm(off, axis=1))) if len(off) else float("nan")
    # silhouette vertices: projected vertices closest to the mesh contour with grazing normals
    cand = head_idx[Vw[head_idx, 1] > y_cut + 0.004]
    P = cam.project(Vw[cand])
    vt = cKDTree(P[:, :2])
    _right, _up, fwd = cam.axes()
    graz = np.abs(Nw[cand] @ fwd)
    dd, nn = vt.query(cm / SHRINK, k=6, distance_upper_bound=4.0)
    sel_v, sel_o = [], []
    for k in range(len(cm)):
        ok_k = np.isfinite(dd[k])
        if not ok_k.any():
            continue
        cand_k = nn[k][ok_k]
        best = cand_k[int(np.argmin(graz[cand_k]))]
        sel_v.append(best)
        sel_o.append(off[k])
    if not sel_v:
        return np.zeros(0, np.int64), np.zeros((0, 3)), mean_px
    sel_v, sel_o = np.array(sel_v), np.array(sel_o)
    disp = cam.plane_offset_to_world(sel_o, P[sel_v, 2])
    return cand[sel_v], disp, mean_px


def landmark_constraints(cam: ViewCam, Vw: np.ndarray, lm_vid: np.ndarray, lm_px: np.ndarray, prm: ReshapeParams):
    P = cam.project(Vw[lm_vid])
    off = lm_px - P[:, :2]
    return lm_vid, cam.plane_offset_to_world(off, P[:, 2]), float(np.mean(np.linalg.norm(off, axis=1)))


def aggregate(Vw: np.ndarray, vid: np.ndarray, disp: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Average displacements per vertex, then per voxel (RBF centres must be distinct)."""
    order = np.argsort(vid)
    vid, disp = vid[order], disp[order]
    u, start = np.unique(vid, return_index=True)
    sums = np.add.reduceat(disp, start, axis=0)
    cnt = np.diff(np.append(start, len(vid)))
    d = sums / cnt[:, None]
    X = Vw[u]
    key = np.floor(X / voxel).astype(np.int64)
    _k, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    n = len(first)
    Xs = np.zeros((n, 3))
    Ds = np.zeros((n, 3))
    c = np.bincount(inv, minlength=n).astype(np.float64)
    for k in range(3):
        Xs[:, k] = np.bincount(inv, weights=X[:, k], minlength=n) / c
        Ds[:, k] = np.bincount(inv, weights=d[:, k], minlength=n) / c
    return Xs, Ds


def reshape_head(
    verts: np.ndarray,
    faces: np.ndarray,
    th: TwinHead,
    photos: dict[str, Photo],
    calibs: dict[str, Calib],
    recalibrate=None,
    front_landmarks: np.ndarray | None = None,
    prm: ReshapeParams | None = None,
) -> ReshapeResult:
    """Deform ``verts`` (index space) so the head silhouettes match the photos. ``recalibrate(verts) -> calibs`` is
    called with (verts, y_cut) after rounds 3 and 5 (the cameras were fitted to the undeformed head)."""
    prm = prm or ReshapeParams()
    inv, first = weld_ids(verts)
    Vw0 = verts[first].copy()
    Vw = Vw0.copy()
    Fw = welded_faces(faces, inv)
    below = 0.14
    head_idx = np.flatnonzero(Vw0[:, 1] > th.chin_y - below)
    gate_w = th.falloff(Vw0[:, 1])
    y_cut = th.chin_y + 0.02

    # landmark vertex ids (front face features), tracked through the deformation
    lm_vid = None
    if prm.use_landmarks and th.lm3d is not None and front_landmarks is not None:
        tree = cKDTree(Vw0[head_idx])
        _d, k = tree.query(th.lm3d)
        lm_vid = head_idx[k][STABLE_LM]
    # zero anchors in the neck band keep the field quiet below the head
    band = np.flatnonzero((Vw0[:, 1] < th.chin_y - 0.11) & (Vw0[:, 1] > th.chin_y - 0.20))
    rng = np.random.default_rng(1)
    anchors = rng.choice(band, size=min(400, len(band)), replace=False) if len(band) else np.zeros(0, np.int64)

    masks_s = {}
    for v, ph in photos.items():
        masks_s[v] = cv2.resize(ph.mask.astype(np.uint8), None, fx=SHRINK, fy=SHRINK, interpolation=cv2.INTER_AREA) > 0
    history = []
    for rnd in range(prm.rounds):
        if recalibrate is not None and rnd in (2, 4):
            tmp = verts.copy()
            tmp += (Vw - Vw0)[inv]
            calibs = recalibrate(tmp, y_cut, None if lm_vid is None else Vw[lm_vid])
        if lm_vid is not None:  # the chin moves with the landmarks: the neck cut plane follows it
            y_cut = float(Vw[lm_vid[STABLE_LM.index(152)], 1]) + 0.02
        Nw = _vertex_normals(Vw, Fw)
        vids, disps, stats = [], [], {}
        for v, ph in photos.items():
            cal = calibs[v]
            vi, dv, m_px = contour_constraints(cal.cam, cal, masks_s[v], Vw, Nw, Fw, head_idx, y_cut, th.pivot[2], prm, ph.mask.shape)
            stats[v] = round(m_px, 1)
            if len(vi):
                vids.append(vi)
                disps.append(dv)
        if lm_vid is not None:
            vi, dv, m_px = landmark_constraints(calibs["front"].cam, Vw, lm_vid, front_landmarks[STABLE_LM], prm)
            stats["landmarks_px"] = round(m_px, 1)
            dv = dv * prm.landmark_weight
            nl = np.linalg.norm(dv, axis=1, keepdims=True)
            vids.append(vi)
            disps.append(dv * np.minimum(1.0, prm.max_landmark_move_m / np.maximum(nl, 1e-9)))
        history.append(stats)
        log(f"reshape round {rnd + 1}/{prm.rounds}: mean contour distance px {stats}")
        vid = np.concatenate(vids)
        disp = np.concatenate(disps)
        n = np.linalg.norm(disp, axis=1, keepdims=True)
        disp = disp * np.minimum(1.0, prm.max_move_m / np.maximum(n, 1e-9))
        X, D = aggregate(Vw, vid, disp * prm.step, prm.voxel_m)
        if len(anchors):
            X = np.concatenate([X, Vw[anchors]])
            D = np.concatenate([D, np.zeros((len(anchors), 3))])
        rbf = RBFInterpolator(X, D, kernel="thin_plate_spline", smoothing=prm.smoothing, degree=1)
        field_ = rbf(Vw[head_idx])
        Vw[head_idx] += field_ * gate_w[head_idx, None]
    disp_w = Vw - Vw0
    new = verts + disp_w[inv]
    return ReshapeResult(new, new - verts, gate_w[inv], {"rounds": history}, calibs, y_cut,
                         None if lm_vid is None else Vw[lm_vid])


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    t = V[F]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    acc = np.zeros_like(V)
    for k in range(3):
        np.add.at(acc, F[:, k], fn)
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
