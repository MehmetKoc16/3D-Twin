"""Synthetic fixtures (no personal data): the CC0 MakeHuman body, a peanut-shaped 'torso + arm', flat sheets."""

from __future__ import annotations

import numpy as np
import trimesh
from rigfit import FitResult


def neutral_fit(model) -> FitResult:
    """A ``FitResult`` for the neutral MakeHuman body at the template pose (what the fitter would return for a scan
    that is exactly that body), without running the (slow) fitter."""
    macro = {"gender": 0.5, "muscle": 0.5, "weight": 0.5, "height": 0.5}
    pos = model.shape(macro, {}, ground=False)
    nr = model.nr
    return FitResult(
        macro=macro,
        mods={},
        rest_positions=pos,
        pose_rotvec={b: np.zeros(3) for b in ("spine_01", "head", "upperarm_l", "upperarm_r")},
        root_t=np.zeros(3),
        posed_vertices=pos[:nr].copy(),
        stats={},
    )


def peanut(radius_a: float = 0.12, radius_b: float = 0.08, centre_b: float = 0.15, subdivisions: int = 4):
    """Closed star-shaped union of two overlapping spheres (centres on the x axis): a fused 'torso + arm' with a pinch
    (concave crease) ring where the spheres meet. Returns (trimesh, x of the pinch plane)."""
    ico = trimesh.creation.icosphere(subdivisions=subdivisions)
    d = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    c0 = 0.05  # star centre, inside both spheres
    o = np.array([c0, 0.0, 0.0])
    r = np.zeros(len(d))
    for ctr, rad in ((0.0, radius_a), (centre_b, radius_b)):
        # distance along d from o to the far side of the sphere (centre (ctr, 0, 0), radius rad)
        oc = o - np.array([ctr, 0.0, 0.0])
        b = d @ oc
        c = oc @ oc - rad * rad
        t = -b + np.sqrt(np.maximum(b * b - c, 0.0))
        r = np.maximum(r, t)
    verts = o + d * r[:, None]
    # pinch plane: radical plane of the two spheres
    x_pinch = (radius_a**2 - radius_b**2 + centre_b**2) / (2 * centre_b)
    return trimesh.Trimesh(verts, ico.faces.copy(), process=False), x_pinch


def flat_sheet(nx: int = 40, ny: int = 40, size: float = 0.2, y0: float = 1.5, noise: float = 0.0, seed: int = 0):
    """Front-facing triangulated sheet (z = 0.1 - 2 x^2 - 2 (y - cy)^2) in the face's world frame, with UVs in [0, 1]."""
    xs = np.linspace(-size / 2, size / 2, nx)
    ys = np.linspace(y0, y0 + size, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.1 - 2.0 * X**2 - 2.0 * (Y - (y0 + size / 2)) ** 2
    if noise:
        Z = Z + np.random.default_rng(seed).normal(0, noise, Z.shape)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    UV = np.stack([(X.ravel() - xs[0]) / size, 1.0 - (Y.ravel() - ys[0]) / size], axis=1)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return V, F, UV


# ---------------------------------------------------------------------------------------------------------------
from twinrefine import facedepth, meshops  # noqa: E402


def sheet_scan(nx=60):
    V, F, UV = flat_sheet(nx, nx, size=0.14, y0=1.55)
    return meshops.to_corners(V, F, UV.astype(np.float32))


def flat_face_depth(grid, mc, offset=0.005, radius=0.05):
    from twintex.raster import zbuffer

    px, py = grid.to_px(mc.P[:, 0], mc.P[:, 1])
    depth, fid = zbuffer(np.stack([px, py, -mc.P[:, 2]], axis=1), mc.F, grid.W, grid.H)
    zscan = np.where(fid >= 0, -depth, np.nan).astype(np.float32)
    gx, gy = grid.centres()
    r = np.hypot(gx - 0.0, gy - 1.62)
    oval = r < radius
    import cv2

    sd = cv2.distanceTransform(oval.astype(np.uint8), cv2.DIST_L2, 5) * grid.res * 1000.0
    out_d = cv2.distanceTransform((~oval).astype(np.uint8), cv2.DIST_L2, 5) * grid.res * 1000.0
    w = facedepth.smoothstep01(sd / 10.0)
    delta = np.where(np.isfinite(zscan), w * offset, 0.0).astype(np.float32)
    return facedepth.FaceDepth(grid, zscan + offset, zscan + offset, zscan, w.astype(np.float32), oval, delta, {}, (sd - out_d).astype(np.float32))


