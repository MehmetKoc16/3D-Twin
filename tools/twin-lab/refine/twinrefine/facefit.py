"""Fit the MakeHuman face to the photo's 2-D landmarks (python port of ``packages/avatar-core/src/face/fitFace.ts``).

Method (same as the TS): every canonical MediaPipe landmark is bound (``face-map.json``) to a point of the MakeHuman
surface (a render-vertex triangle + barycentric weights). For face-modifier values ``theta`` the bound point moves by
the piecewise-linear morph deltas; the frontal (x, y) layout of the bound points is compared with the photo landmarks
after the best 2-D similarity, and ``theta`` is solved by bounded Levenberg-Marquardt with a Tikhonov pull to the
neutral face. Only "stable" landmarks take part (no eye contours, brows, lips except the mouth corners).

Differences to the TS: the target is not a photo in pixels but the same landmarks expressed in the scan's world frame
(front camera + flow of the texture stage), and the body the face is added to is the articulated fit of the scan (the
head is posed like the scan's head). The similarity found here is applied to the face; the remaining per-landmark
residual is closed later by a thin-plate warp (``facedepth``), which only moves points in x / y.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from mh import MHModel  # noqa: E402  (rig stage)
from rigfit import FitResult  # noqa: E402
from scipy.optimize import least_squares

from . import FACE_MAP_JSON, log

# MediaPipe FACEMESH_*_EYEBROW vertex indices (face_mesh_connections.py, Apache-2.0), as in fitFace.ts
MEDIAPIPE_EYEBROWS = (
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
)  # fmt: skip
LIP_CORNERS = (61, 291)
# Face modifiers of the TS fit, plus the head width / height / fat: the body fit sets these from the whole scan (hair
# included) and they differ a lot between runs, while the photo's face landmarks determine them well. They are fitted
# as an additive correction on top of the body fit's value; head depth is not observable from frontal landmarks.
FACE_FIT_MODIFIERS = (
    "chin/chin-width", "chin/chin-height", "nose/nose-scale-horiz", "nose/nose-scale-vert",
    "mouth/mouth-scale-horiz", "eyes/eye-scale", "cheek/cheek-volume",
    "head/head-scale-horiz", "head/head-scale-vert", "head/head-fat",
)  # fmt: skip


@dataclass
class FaceMap:
    tri: np.ndarray  # (468, 3) render-vertex ids
    bary: np.ndarray  # (468, 3)
    oval: list[int]
    regions: dict[str, list[int]]
    triangles: np.ndarray  # (898, 3) canonical mesh triangles (landmark indices)


def load_face_map() -> FaceMap:
    d = json.loads(FACE_MAP_JSON.read_text(encoding="utf-8"))
    lm = d["landmarks"]
    return FaceMap(
        np.array([e["tri"] for e in lm], dtype=np.int64),
        np.array([e["bary"] for e in lm], dtype=np.float64),
        list(d["faceOval"]),
        {k: list(v) for k, v in d["regions"].items()},
        np.array(d["triangles"], dtype=np.int64),
    )


def stable_landmarks(fm: FaceMap) -> np.ndarray:
    drop = set(fm.regions["leftEye"]) | set(fm.regions["rightEye"]) | set(MEDIAPIPE_EYEBROWS)
    drop |= {i for i in fm.regions["lips"] if i not in LIP_CORNERS}
    return np.array([i for i in range(468) if i not in drop], dtype=np.int64)


def similarity_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares (s, R, t) with dst ~ s R src + t (2-D, Umeyama)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, sv, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    S = np.diag([1.0, d])
    R = u @ S @ vt
    s = float(np.trace(np.diag(sv) @ S) / max(a.var(axis=0).sum(), 1e-18))
    t = mu_d - s * R @ mu_s
    return s, R, t


class FaceModel:
    """MakeHuman head posed like the fitted scan, with morphable face modifiers."""

    def __init__(self, model: MHModel, res: FitResult, fm: FaceMap, modifiers: tuple[str, ...] = FACE_FIT_MODIFIERS) -> None:
        self.m = model
        self.res = res
        self.fm = fm
        self.modifiers = [m for m in modifiers if m in model.mod_index]
        self.lo = np.array([model.modifiers[model.mod_index[m]]["min"] for m in self.modifiers])
        self.hi = np.array([model.modifiers[model.mod_index[m]]["max"] for m in self.modifiers])
        # rest-space delta of each face modifier on the render vertices (incr and decr columns)
        self._delta: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        nr = model.nr
        for mid in self.modifiers:
            mod = model.modifiers[model.mod_index[mid]]
            cols = []
            for key in ("incrTarget", "decrTarget"):
                if key in mod:
                    cols.append((model.D[:, model.target_index[mod[key]]].toarray().reshape(-1, 3))[:nr])
                else:
                    cols.append(np.zeros((nr, 3)))
            self._delta[mid] = (cols[0], cols[1])
        from scipy.spatial.transform import Rotation

        self._rots = {b: Rotation.from_rotvec(v).as_matrix() for b, v in res.pose_rotvec.items()}
        self._pose_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # landmark points on the zero-modifier posed body
        self.posed0 = self.posed(np.zeros(len(self.modifiers)))
        self.P0 = self.bound(self.posed0)
        # per-modifier landmark deltas, posed (head joint rotation is tiny; use the blended linear map of the base)
        self._lm_delta = {}
        A, _b = self._blend()
        for mid in self.modifiers:
            inc, dec = self._delta[mid]
            self._lm_delta[mid] = (self.bound_delta(A, inc), self.bound_delta(A, dec))

    def _blend(self):
        res, m = self.res, self.m
        heads = m.rest_heads(res.rest_positions)
        rots, posed_h = m.fk(heads, self._rots, res.root_t)
        A, b = m.blended_affine(m.skin_j, m.skin_w, heads, rots, posed_h)
        return A, b

    def posed(self, theta: np.ndarray) -> np.ndarray:
        """Posed render vertices (nr, 3) of the fitted body plus the face modifiers ``theta``."""
        A, b = self._blend()
        pos = self.res.rest_positions[: self.m.nr].copy()
        for t, mid in zip(theta, self.modifiers, strict=True):
            inc, dec = self._delta[mid]
            pos += t * inc if t >= 0 else -t * dec
        return np.einsum("nij,nj->ni", A, pos) + b

    def bound(self, posed: np.ndarray) -> np.ndarray:
        """The 468 landmark points of a posed body (468, 3)."""
        return np.einsum("lk,lkj->lj", self.fm.bary, posed[self.fm.tri])

    def bound_delta(self, A: np.ndarray, delta: np.ndarray) -> np.ndarray:
        d = np.einsum("nij,nj->ni", A, delta)
        return np.einsum("lk,lkj->lj", self.fm.bary, d[self.fm.tri])

    def landmark_points(self, theta: np.ndarray) -> np.ndarray:
        P = self.P0.copy()
        for t, mid in zip(theta, self.modifiers, strict=True):
            di, dd = self._lm_delta[mid]
            P += t * di if t >= 0 else -t * dd
        return P


@dataclass
class FaceFitResult:
    theta: np.ndarray
    modifiers: list[str]
    s: float
    R: np.ndarray
    t: np.ndarray
    rms_mm: float
    rms_neutral_mm: float
    iterations: int


def fit_face(fmodel: FaceModel, target_xy: np.ndarray, regularization: float = 2.0, max_nfev: int = 60) -> FaceFitResult:
    """Solve the face modifiers so that the stable landmarks match ``target_xy`` (468, 2; metres, world x / y)."""
    idx = stable_landmarks(fmodel.fm)
    T = target_xy[idx]
    sqrt_reg = np.sqrt(regularization * 1e-6)  # TS: mm^2 / unit^2 -> m^2 / unit^2

    def residual(theta: np.ndarray) -> np.ndarray:
        M = fmodel.landmark_points(theta)[idx, :2]
        s, R, t = similarity_2d(M, T)
        e = T - (s * (M @ R.T) + t)
        return np.concatenate([e.ravel(), sqrt_reg * theta])

    x0 = np.zeros(len(fmodel.modifiers))
    r0 = residual(x0)
    res = least_squares(residual, x0, bounds=(fmodel.lo, fmodel.hi), method="trf", x_scale=0.3, max_nfev=max_nfev, diff_step=1e-3)
    theta = res.x
    M = fmodel.landmark_points(theta)[idx, :2]
    s, R, t = similarity_2d(M, T)
    e = T - (s * (M @ R.T) + t)
    rms = float(np.sqrt((e**2).sum(axis=1).mean()) * 1000)
    rms0 = float(np.sqrt((r0[: 2 * len(idx)] ** 2).sum() / len(idx)) * 1000)
    log(f"face fit: stable-landmark RMS {rms0:.2f} -> {rms:.2f} mm; theta " + ", ".join(
        f"{m.split('/')[1]}={v:+.2f}" for m, v in zip(fmodel.modifiers, theta, strict=True)))
    return FaceFitResult(theta, list(fmodel.modifiers), s, R, t, rms, rms0, int(res.nfev))
