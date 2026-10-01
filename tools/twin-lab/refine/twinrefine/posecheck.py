"""Pose a (welded) mesh with the weights the rig stage would compute and measure how much its edges stretch.

This is the in-process twin of ``rig_scan.py`` (weight transfer -> unpose -> pose JSON -> LBS) used to verify the armpit
separation without a Node / browser round trip; ``rig/check_pose.mjs`` remains the final cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from mh import MHModel, load_pose  # noqa: E402  (rig stage)
from rigfit import FitResult, top4, transfer_weights  # noqa: E402

from . import log  # noqa: F401
from .meshops import edge_table  # noqa: E402


def _rot_dict(res: FitResult) -> dict[str, np.ndarray]:
    from scipy.spatial.transform import Rotation

    return {b: Rotation.from_rotvec(v).as_matrix() for b, v in res.pose_rotvec.items()}


@dataclass
class PosedMesh:
    rest: np.ndarray  # unposed (template A-pose) vertices
    posed: np.ndarray  # vertices in the requested pose
    stretch: np.ndarray  # (E,) posed / rest length of every unique welded edge
    edges: np.ndarray  # (E, 2)
    jn: np.ndarray
    jw: np.ndarray


def pose_mesh(
    model: MHModel, res: FitResult, P: np.ndarray, F: np.ndarray, pose: str = "t-pose", smooth_iters: int = 6
) -> PosedMesh:
    wm = trimesh.Trimesh(P, F, process=False)
    W, _ = transfer_weights(model, res.posed_vertices, P, wm.faces, smooth_iters=smooth_iters)
    jn, jw = top4(W)
    heads = model.rest_heads(res.rest_positions)
    rots, posed_h = model.fk(heads, _rot_dict(res), res.root_t)
    A, b = model.blended_affine(jn.astype(np.int64), jw.astype(np.float64), heads, rots, posed_h)
    rest = np.linalg.solve(A, (P - b)[..., None])[..., 0]
    rots_t, posed_t = model.fk(heads, load_pose(pose))
    posed = model.lbs(rest, jn.astype(np.int64), jw.astype(np.float64), heads, rots_t, posed_t)
    et = edge_table(F)
    a = np.zeros(et.n_edges, dtype=np.int64)
    bb = np.zeros(et.n_edges, dtype=np.int64)
    a[et.edge_of] = et.he_a
    bb[et.edge_of] = et.he_b
    l0 = np.linalg.norm(rest[a] - rest[bb], axis=1)
    l1 = np.linalg.norm(posed[a] - posed[bb], axis=1)
    stretch = l1 / np.maximum(l0, 1e-6)
    return PosedMesh(rest, posed, stretch, np.stack([a, bb], axis=1), jn, jw)


def webbing_stats(pm: PosedMesh, y_range=(1.10, 1.38), min_abs_x: float = 0.10, thresh: float = 2.0) -> dict:
    """Stretch statistics of the edges around the armpits (rest frame of the unposed mesh)."""
    mid = 0.5 * (pm.rest[pm.edges[:, 0]] + pm.rest[pm.edges[:, 1]])
    zone = (mid[:, 1] >= y_range[0]) & (mid[:, 1] <= y_range[1]) & (np.abs(mid[:, 0]) >= min_abs_x)
    s = pm.stretch[zone]
    long_rest = np.linalg.norm(pm.rest[pm.edges[zone, 0]] - pm.rest[pm.edges[zone, 1]], axis=1) > 1e-4
    s = s[long_rest]
    # absolute length of the stretched edges in the posed mesh (a "sheet" is a set of long edges)
    pl = np.linalg.norm(pm.posed[pm.edges[zone, 0]] - pm.posed[pm.edges[zone, 1]], axis=1)[long_rest]
    return {
        "edges": int(len(s)),
        "over2": int((s > thresh).sum()),
        "over3": int((s > 3.0).sum()),
        "p99": float(np.percentile(s, 99)) if len(s) else 0.0,
        "max": float(s.max()) if len(s) else 0.0,
        "posed_edges_over_3cm": int((pl > 0.03).sum()),
        "posed_edges_over_6cm": int((pl > 0.06).sum()),
    }
