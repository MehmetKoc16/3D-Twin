"""Skin weights: MPFB2 `weights.game_engine.json` (MakeHuman vertex space) -> glTF JOINTS_0 / WEIGHTS_0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mh_obj import BODY_VERTEX_COUNT, BaseMesh

MAX_INFLUENCES = 4


def load_weights(path: Path, bones: list[str], mesh: BaseMesh) -> tuple[np.ndarray, np.ndarray]:
    """Return JOINTS_0 (R,4) uint16 and WEIGHTS_0 (R,4) float32 over the render vertices.

    Influences are reduced to the top 4 (ties broken by bone order) and renormalised. Skin joint indices are
    positions in `bones` (the rig.json / glTF skin order).
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)["weights"]
    index = {b: i for i, b in enumerate(bones)}
    dense = np.zeros((BODY_VERTEX_COUNT, len(bones)), dtype=np.float64)
    for bone in sorted(raw):
        if bone not in index:
            raise KeyError(f"weights reference unknown bone {bone}")
        for v, w in raw[bone]:
            if v < BODY_VERTEX_COUNT:
                dense[v, index[bone]] += w
    sums = dense.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError(f"{int((sums <= 0).sum())} body vertices have no skin weights")

    order = np.argsort(-dense, axis=1, kind="stable")[:, :MAX_INFLUENCES]
    top = np.take_along_axis(dense, order, axis=1)
    top = top / top.sum(axis=1, keepdims=True)
    j = order.astype(np.uint16)
    j[top == 0] = 0

    w32 = top.astype(np.float32)
    # Make float32 weights sum to exactly 1 (assign the residual to the dominant influence).
    rest = w32[:, 1:].sum(axis=1, dtype=np.float32)
    w32[:, 0] = np.float32(1.0) - rest

    return j[mesh.render_mh], w32[mesh.render_mh]
