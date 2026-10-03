"""Numeric checks of the procedural hair: penetration, scalp coverage from the main views, skinning to the head.

Every function returns plain numbers (no images), so the checks can run on private data without displaying it.
"""

from __future__ import annotations

import numpy as np
from twintex.camera import OrthoCamera
from twintex.raster import raster_pairs, zbuffer

from .hairgen import HairField, HeadSurface

HEAD_BONES = ("head", "neck_01")


def penetration_report(
    surface: HeadSurface, positions: np.ndarray, faces: np.ndarray, clearance: float, tolerance: float = 1e-4
) -> dict:
    """Signed distance of the card vertices (and of triangle centroids and edge midpoints) to the head surface.

    ``vertices_below_clearance`` counts vertices closer than ``clearance - tolerance`` (so 0 means every vertex keeps
    the minimum offset); ``samples_*`` repeat the test on points inside the triangles.
    """
    vertex, _, _ = surface.signed_distance(positions, candidates=24)
    tri = positions[faces]
    samples = np.concatenate((tri.mean(1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2))
    inner, _, _ = surface.signed_distance(samples, candidates=24)
    return {
        "vertices": int(len(positions)),
        "clearance_mm": clearance * 1000,
        "vertices_inside_head": int((vertex < 0).sum()),
        "vertices_below_clearance": int((vertex < clearance - tolerance).sum()),
        "min_mm": float(vertex.min() * 1000),
        "median_mm": float(np.median(vertex) * 1000),
        "p05_mm": float(np.percentile(vertex, 5) * 1000),
        "p95_mm": float(np.percentile(vertex, 95) * 1000),
        "deeper_than_4mm": float((vertex < -0.004).mean()),
        "deeper_than_10mm": float((vertex < -0.010).mean()),
        "max_mm": float(vertex.max() * 1000),
        "samples_checked": int(len(samples)),
        "samples_inside_head": int((inner < 0).sum()),
        "samples_below_clearance": int((inner < clearance - tolerance).sum()),
        "samples_min_mm": float(inner.min() * 1000),
    }


def view_cameras(bounds_points: np.ndarray, pixel_m: float) -> dict:
    """Orthographic cameras (front, left, right, back, top) framing ``bounds_points`` at about ``pixel_m`` per pixel."""
    top = OrthoCamera("top", np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    cams = {
        "front": OrthoCamera.azimuth("front", 0.0),
        "left": OrthoCamera.azimuth("left", 90.0),
        "right": OrthoCamera.azimuth("right", 270.0),
        "back": OrthoCamera.azimuth("back", 180.0),
        "top": top,
    }
    size = int(np.ceil(np.ptp(bounds_points, axis=0).max() / pixel_m * 1.06)) + 4
    return {name: (cam.fit_bounds(bounds_points, size, size, margin=0.03), size) for name, cam in cams.items()}


def cutout_depth(
    camera: OrthoCamera,
    size: int,
    positions: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    alpha: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Depth map (smaller = nearer) of the cards with the strip texture's alpha test."""
    height, width = alpha.shape
    p = camera.project(positions)
    zbuf = np.full(size * size, np.inf, np.float32)
    for f, px, py, lam in raster_pairs(p[:, :2], faces, size, size):
        t = np.einsum("kj,kjc->kc", lam, uv[faces[f]])
        ix = np.clip((t[:, 0] * width).astype(int), 0, width - 1)
        iy = np.clip((t[:, 1] * height).astype(int), 0, height - 1)
        seen = alpha[iy, ix] >= cutoff
        f, px, py, lam = f[seen], px[seen], py[seen], lam[seen]
        d = np.einsum("kj,kj->k", lam, p[:, 2][faces[f]]).astype(np.float32)
        pix = py * size + px
        order = np.argsort(-d, kind="stable")
        d, pix = d[order], pix[order]
        keep = d < zbuf[pix]
        zbuf[pix[keep]] = d[keep]
    return zbuf.reshape(size, size)


def scalp_coverage(
    surface: HeadSurface,
    field: HairField,
    positions: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    alpha: np.ndarray,
    *,
    samples: int = 60000,
    pixel_mm: float = 0.5,
    cutoff: float = 0.5,
    seed: int = 1,
    min_density: float = 0.5,
) -> dict:
    """Share of the visible scalp (where the hair density is above ``min_density``) hidden behind the cards, per view.

    A scalp sample is *visible* from a view when it faces the camera and nothing of the head is in front of it; it is
    *hidden* when a card fragment (after the strip texture's alpha test) lies in front of it.
    """
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(surface.faces), size=samples, p=surface.area / surface.area.sum())
    points, normals = surface.sample(face_ids, rng)
    scalp = field.weight(points) >= min_density
    points, normals = points[scalp], normals[scalp]
    everything = np.vstack((surface.vertices, positions))
    result = {"scalp_samples": int(len(points))}
    for name, (camera, size) in view_cameras(everything, pixel_mm * 1e-3).items():
        head_depth, _ = zbuffer(camera.project(surface.vertices), surface.faces, size, size)
        hair_depth = cutout_depth(camera, size, positions, faces, uv, alpha, cutoff)
        q = camera.project(points)
        ix = np.clip(q[:, 0].astype(int), 0, size - 1)
        iy = np.clip(q[:, 1].astype(int), 0, size - 1)
        facing = normals @ camera.to_camera > 0.25
        visible = facing & (head_depth[iy, ix] >= q[:, 2] - 0.0008)
        hidden = visible & (hair_depth[iy, ix] < q[:, 2] - 0.0006)
        result[name] = {
            "visible_samples": int(visible.sum()),
            "hidden_fraction": float(hidden.sum() / max(visible.sum(), 1)),
        }
    result["min_hidden_fraction"] = float(
        min(result[v]["hidden_fraction"] for v in ("front", "left", "right", "back", "top"))
    )
    result["pixel_mm"] = pixel_mm
    return result


def skin_weight_report(model, body_positions: np.ndarray, positions: np.ndarray, faces: np.ndarray) -> dict:
    """Skin weights the rig stage will give the hair (its own closest-point transfer, unsmoothed), summed over head bones.

    The rig stage welds vertices at 10 micrometres and transfers the weights of the nearest template vertices.
    """
    from rigfit import transfer_weights

    key = np.round(positions * 1e5).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    welded = positions[first]
    tri = inverse[faces]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    weights, stats = transfer_weights(model, body_positions, welded, tri, smooth_iters=0)
    head = weights[:, [model.bone_index[b] for b in HEAD_BONES]].sum(1)
    dominant = np.argmax(weights, axis=1)
    names, counts = np.unique(np.array(model.bone_names)[dominant], return_counts=True)
    order = np.argsort(-counts)
    return {
        "vertices": int(len(welded)),
        "head_bones": list(HEAD_BONES),
        "head_chain_weight_min": float(head.min()),
        "head_chain_weight_p01": float(np.percentile(head, 1)),
        "head_chain_weight_mean": float(head.mean()),
        "fraction_head_chain_ge_0_95": float((head >= 0.95).mean()),
        "head_bone_weight_mean": float(weights[:, model.bone_index["head"]].mean()),
        "dominant_bones": {str(names[i]): int(counts[i]) for i in order[:4]},
        "nn_distance_median_cm": float(stats["nn_dist_median_cm"]),
        "nn_distance_p99_cm": float(stats["nn_dist_p99_cm"]),
    }
