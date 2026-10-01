"""Fit our MakeHuman body to the scan (rig stage's fitter, cached) and segment the scan by the fitted body.

The fitted body gives (a) the posed head surface used for the face relief and (b) per-bone skin weights, from which
the arm / torso labels of the scan vertices come (same transfer the rig stage uses, so the labels agree with the
weights the rig stage will compute on the refined mesh).
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass

import numpy as np
import trimesh
from mh import MHModel  # noqa: E402  (rig stage)
from rigfit import FitResult, Fitter, top4, transfer_weights  # noqa: E402

from . import log
from .scan import weld_ids, welded_faces

FINGERS = ("index", "middle", "ring", "pinky", "thumb")


@dataclass
class BodyFit:
    model: MHModel
    res: FitResult
    uverts: np.ndarray  # welded scan vertices the fit / weights refer to
    ufaces: np.ndarray
    inv: np.ndarray  # index-space vertex -> welded id
    weights: np.ndarray  # (nu, nbones) transferred + smoothed skin weights (scan pose)


def make_welded(verts: np.ndarray, faces: np.ndarray):
    inv, first = weld_ids(verts)
    uverts = np.asarray(verts, dtype=np.float64)[first]
    ufaces = welded_faces(faces, inv)
    return inv, uverts, ufaces


def mesh_hash(uverts: np.ndarray) -> str:
    return hashlib.sha1(np.round(np.asarray(uverts, dtype=np.float64) * 1e5).astype(np.int64).tobytes()).hexdigest()


def _load_cache(cache, uverts: np.ndarray) -> FitResult | None:
    """A cached fit is only reused for the very mesh it was made on (sha1 sidecar); a pickle without sidecar (e.g. the
    rig stage's own fit.pkl) must at least agree with the scan's extent."""
    sidecar = str(cache) + ".sha1"
    with open(cache, "rb") as fh:
        res = pickle.load(fh)
    if os.path.exists(sidecar):
        if open(sidecar, encoding="utf-8").read().strip() == mesh_hash(uverts):
            return res
        log(f"body fit: {cache} belongs to another mesh, fitting again")
        return None
    lo, hi = uverts.min(axis=0), uverts.max(axis=0)
    plo, phi = res.posed_vertices.min(axis=0), res.posed_vertices.max(axis=0)
    if np.all(np.abs(lo[[0, 2]] - plo[[0, 2]]) < 0.15) and np.all(np.abs(hi - phi) < 0.15) and abs(lo[1] - plo[1]) < 0.08:
        return res
    log(f"body fit: {cache} does not match the scan's extent, fitting again")
    return None


def fit_body(
    model: MHModel,
    uverts: np.ndarray,
    ufaces: np.ndarray,
    cache: str | os.PathLike | None = None,
    samples: int = 120000,
) -> FitResult:
    """Articulated + parametric ICP of the MakeHuman body to the scan (cached on disk when ``cache`` is given)."""
    if cache and os.path.exists(cache):
        res = _load_cache(cache, uverts)
        if res is not None:
            log(f"body fit: reusing {cache}")
            return res
    wm = trimesh.Trimesh(uverts, ufaces, process=False)
    if wm.volume < 0:
        wm.invert()
    pts, fid = trimesh.sample.sample_surface(wm, samples, seed=3)
    nrm = wm.face_normals[fid]
    log(f"body fit: {len(pts)} surface samples (this takes a few minutes) ...")
    res = Fitter(model, pts, nrm).run()
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(res, fh)
        with open(str(cache) + ".sha1", "w", encoding="utf-8") as fh:
            fh.write(mesh_hash(uverts))
    return res


def skin_weights(model: MHModel, res: FitResult, uverts: np.ndarray, ufaces: np.ndarray, smooth_iters: int = 6):
    wm = trimesh.Trimesh(uverts, ufaces, process=False)
    if wm.volume < 0:
        wm.invert()
    W, _stats = transfer_weights(model, res.posed_vertices, uverts, wm.faces, smooth_iters=smooth_iters)
    return W


def arm_bone_ids(model: MHModel, side: str) -> list[int]:
    """Bones that move with the arm: upper arm, forearm, hand and fingers (the clavicle belongs to the trunk side)."""
    names = [f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}"]
    names += [f"{f}_0{i}_{side}" for f in FINGERS for i in (1, 2, 3)]
    return [model.bone_index[n] for n in names if n in model.bone_index]


def arm_weight(model: MHModel, W: np.ndarray, side: str) -> np.ndarray:
    return W[:, arm_bone_ids(model, side)].sum(axis=1)


def top_weights(W: np.ndarray):
    return top4(W)
