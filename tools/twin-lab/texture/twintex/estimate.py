"""Colour harmonisation between views."""

from __future__ import annotations

import numpy as np

from .bake import BakeConfig, sample_view
from .views import View


def estimate_gains(views: list[View], P: np.ndarray, N: np.ndarray, cfg: BakeConfig, ymin: float, ymax: float,
                   ref: str = "front") -> dict[str, np.ndarray]:
    """Per-view RGB gains that match every view to the reference view on the texels both see reliably."""
    body_h = ymax - ymin
    samples = {v.name: sample_view(v, P, N, cfg, body_h) for v in views}
    ref_name = ref if ref in samples else views[0].name
    gains = {v.name: np.ones(3, np.float32) for v in views}
    q_ref, _, c_ref, _ = samples[ref_name]
    for v in views:
        if v.name == ref_name:
            continue
        q, _, c, _ = samples[v.name]
        both = (q > 0.2) & (q_ref > 0.2) & (c.mean(1) > 0.02) & (c_ref.mean(1) > 0.02)
        if both.sum() < 300:
            continue
        ratio = np.median(c_ref[both] / np.maximum(c[both], 1e-4), axis=0)
        gains[v.name] = np.clip(ratio, 0.7, 1.4).astype(np.float32)
    return gains
