"""Bounded pose-space triangle repulsion for offline garment shell review."""
from __future__ import annotations

import numpy as np

from .body import skin
from .geometry import skin_weights, fit
from .pose_safety import POSES
from .visibility import triangle_pairs


def separation(a, b, margin=.0004):
    """Smallest separating-axis translation of triangle A away from B."""
    ea, eb = np.roll(a, -1, axis=1)-a, np.roll(b, -1, axis=1)-b
    na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
    axes = [na, nb]+[np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
    best = np.full(len(a), np.inf)
    movement = np.zeros((len(a), 3))
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1)
        axis = axis/np.maximum(norm[:, None], 1e-30)
        pa, pb = np.einsum('nij,nj->ni', a, axis), np.einsum('nij,nj->ni', b, axis)
        positive, negative = pb.max(axis=1)-pa.min(axis=1), pa.max(axis=1)-pb.min(axis=1)
        distance = np.minimum(positive, negative)+margin
        good = (norm > 1e-15) & (distance < best) & (distance > 0)
        direction = np.where(positive < negative, 1., -1.)
        movement[good] = axis[good]*(distance*direction)[good, None]
        best[good] = distance[good]
    return movement


def untangle(points, faces, bind, body, iterations=12, max_displacement=.012, clearance=None):
    """Average equal/opposite pair corrections back into rest space; retain best pose total.

    All UV copies move together. This reduces intersections; it cannot establish
    a collision-free cloth solution for arbitrary animation or graded shapes.
    """
    ji, jw = skin_weights(bind, body.joints, body.weights)
    transforms = [body.transforms(body.pose(pose)) for _, pose in POSES]
    linear = [(transform[ji, :3, :3]*jw[:, :, None, None]).sum(axis=1) for transform in transforms]
    inverse = [np.linalg.pinv(matrix) for matrix in linear]
    _, copies = np.unique(np.round(points, 7), axis=0, return_inverse=True)
    original, current, best_points = points.copy(), points.copy(), points.copy()
    best_score, initial, best_counts = np.inf, None, None
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    for iteration in range(iterations+1):
        correction, count = np.zeros_like(points), np.zeros(len(points))
        counts = {}
        for (key, _), transform, inv in zip(POSES, transforms, inverse):
            posed = skin(current, ji, jw, transform)
            pairs = triangle_pairs(posed, faces)
            counts[key] = len(pairs)
            if not len(pairs):
                continue
            movement = separation(posed[faces[pairs[:, 0]]], posed[faces[pairs[:, 1]]])*.55
            for column, sign in ((0, 1), (1, -1)):
                ids = faces[pairs[:, column]]
                delta = np.einsum('nkij,nj->nki', inv[ids], movement*sign)
                np.add.at(correction, ids, delta)
                np.add.at(count, ids, 1)
        score = sum(counts.values())
        if initial is None:
            initial = counts.copy()
        within_limit = np.linalg.norm(current-original, axis=1).max() <= max_displacement+1e-7
        if score < best_score and within_limit:
            best_score, best_points, best_counts = score, current.copy(), counts.copy()
        if iteration == iterations or score == 0:
            break
        delta = correction/np.maximum(count[:, None], 1)
        # Spread a small part of the displacement to neighbouring corners to avoid spikes.
        sums, degree = np.zeros_like(points), np.zeros(len(points))
        for u, v in (edges.T, edges[:, ::-1].T):
            np.add.at(sums, u, delta[v]); np.add.at(degree, u, 1)
        delta = .85*delta+.15*sums/np.maximum(degree[:, None], 1)
        group = np.zeros((copies.max()+1, 3))
        np.add.at(group, copies, delta)
        delta = (group/np.bincount(copies)[:, None])[copies]
        displacement = current+delta-original
        distance = np.linalg.norm(displacement, axis=1)
        displacement *= np.minimum(1, max_displacement/np.maximum(distance, 1e-12))[:, None]
        current = original+displacement
        if clearance is not None:
            current, _ = fit(current, faces, body.positions[:body.render_count], body.faces, clearance)
    return best_points, {"beforePairs": initial, "afterPairs": best_counts,
                         "maxDisplacementMm": float(np.linalg.norm(best_points-original, axis=1).max()*1000),
                         "iterations": iterations, "displacementLimitMm": max_displacement*1000,
                         "clearanceProjectedEachIteration": clearance is not None}
