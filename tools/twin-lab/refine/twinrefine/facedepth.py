"""Face relief: the fitted MakeHuman face as a depth map in the front camera frame, and the displacement it implies.

Pipeline (all maps live on one ``Grid`` of the front camera's image plane, x / y in world metres):

1. rasterise the front-facing triangles of the fitted MakeHuman head (rays along the camera axis) -> depth ``Zmh``;
   the eyelid openings and the lip slit of the MakeHuman head are holes of the head surface;
2. warp it with a thin-plate spline so that every one of the 468 landmarks lands on the photo's landmark (the face
   features then line up with the photo pixels that the texture stage projects onto the scan);
3. close the holes by harmonic interpolation (the eyes stay closed lids, with a slight bulge for the eyeball) so the
   surface stays closed;
4. low frequencies come from the scan, details from the face: ``Znew = Zmh + c`` with ``c`` a smooth (sigma ~ 3 cm)
   correction fitted on the face oval, so the face joins the head without a step;
5. ``delta = w * (Znew - Zscan)`` with ``w`` a feathered mask of the face oval (landmark polygon): hairline, ears and
   neck keep the scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from twintex.raster import zbuffer  # noqa: E402  (texture stage)

from .facefit import FaceFitResult, FaceMap, FaceModel


@dataclass
class Grid:
    x0: float  # world x of the left edge of pixel column 0
    y0: float  # world y of the top edge of pixel row 0
    res: float  # metres per pixel
    W: int
    H: int

    def to_px(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World -> pixel coordinates (top-left corner origin: pixel centre = index + 0.5)."""
        return (np.asarray(x) - self.x0) / self.res, (self.y0 - np.asarray(y)) / self.res

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + (np.arange(self.W) + 0.5) * self.res
        ys = self.y0 - (np.arange(self.H) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    @classmethod
    def around(cls, xy: np.ndarray, margin: float, res: float) -> Grid:
        lo, hi = xy.min(axis=0) - margin, xy.max(axis=0) + margin
        return cls(float(lo[0]), float(hi[1]), res, int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res)))


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------------ MakeHuman head
def head_front_triangles(model, posed: np.ndarray) -> np.ndarray:
    """Front-facing triangles of the head surface island (the part of the render mesh that holds the nose tip)."""
    from scipy.sparse.csgraph import connected_components

    f = model.faces
    n = model.nr
    g = coo_matrix((np.ones(3 * len(f)), (f.reshape(-1), f[:, [1, 2, 0]].reshape(-1))), shape=(n, n))
    _nc, lab = connected_components(g, directed=False)
    high = np.flatnonzero(posed[:, 1] > posed[:, 1].max() - 0.18)  # the head, not the chest
    nose = high[np.argmax(posed[high, 2])]
    head = lab[nose]
    a, b, c = posed[f[:, 0]], posed[f[:, 1]], posed[f[:, 2]]
    nz = np.cross(b - a, c - a)[:, 2]
    return np.flatnonzero((lab[f[:, 0]] == head) & (nz > 0))


def rasterize_depth(xy: np.ndarray, z: np.ndarray, faces: np.ndarray, grid: Grid) -> np.ndarray:
    """Nearest-surface z of a mesh seen along -z on ``grid`` (NaN where nothing is hit)."""
    px, py = grid.to_px(xy[:, 0], xy[:, 1])
    depth, fid = zbuffer(np.stack([px, py, -z], axis=1), faces, grid.W, grid.H)
    out = np.where(fid >= 0, -depth, np.nan)
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------ fills
def harmonic_fill(z: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill the ``hole`` pixels by solving the Laplace equation with the surrounding valid pixels as boundary values."""
    out = z.copy()
    H, W = z.shape
    idx = -np.ones((H, W), dtype=np.int64)
    ys, xs = np.nonzero(hole)
    if len(ys) == 0:
        return out
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(ys))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nb_hole = np.zeros(len(ys), dtype=bool)
        nb_hole[inside] = hole[ny[inside], nx[inside]]
        nb_val = np.zeros(len(ys))
        ok = inside & ~nb_hole & np.isfinite(z[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)])
        nb_val[ok] = z[ny[ok], nx[ok]]
        rhs += np.where(ok, nb_val, 0.0)
        r = np.flatnonzero(nb_hole)
        rows.append(r)
        cols.append(idx[ny[r], nx[r]])
        vals.append(np.full(len(r), -1.0))
    deg = np.zeros(len(ys))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        valid_nb = inside & (hole[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)] | np.isfinite(z[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)]))
        deg += valid_nb
    deg = np.maximum(deg, 1.0)
    rows.append(np.arange(len(ys)))
    cols.append(np.arange(len(ys)))
    vals.append(deg)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ys), len(ys))).tocsc()
    sol = spsolve(A, rhs)
    out[ys, xs] = sol
    return out


def normalized_blur(values: np.ndarray, weight: np.ndarray, sigma_px: float) -> np.ndarray:
    num = cv2.GaussianBlur((values * weight).astype(np.float32), (0, 0), sigma_px)
    den = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), sigma_px)
    return num / np.maximum(den, 1e-6)


# ------------------------------------------------------------------------------------------------ the depth map
@dataclass
class FaceDepth:
    grid: Grid
    zmh: np.ndarray  # warped MakeHuman face depth (holes filled), NaN outside the head surface
    znew: np.ndarray  # target depth on the oval (after the low-frequency correction)
    zscan: np.ndarray  # scan front surface depth
    weight: np.ndarray  # feathered oval mask in [0, 1]
    oval: np.ndarray  # bool mask of the landmark polygon
    delta: np.ndarray  # weight * (znew - zscan), 0 where the scan has no surface
    info: dict
    sd: np.ndarray | None = None  # signed distance to the oval boundary in mm (positive inside)


def oval_mask(grid: Grid, polygon_xy: np.ndarray) -> np.ndarray:
    px, py = grid.to_px(polygon_xy[:, 0], polygon_xy[:, 1])
    img = np.zeros((grid.H, grid.W), np.uint8)
    cv2.fillPoly(img, [np.stack([px, py], axis=1).round().astype(np.int32)], 1)
    return img.astype(bool)


def build_face_depth(
    fmodel: FaceModel,
    fit: FaceFitResult,
    L_world: np.ndarray,
    zscan: np.ndarray,
    grid: Grid,
    fm: FaceMap,
    feather_mm: float = 14.0,
    rim_mm: float = 6.0,
    rim_sigma_mm: float = 6.0,
    smooth_mm: float = 1.0,
    warp_rms_mm: float = 0.8,
    max_delta_mm: float = 35.0,
    slope_max: float = 4.5,
    eye_bulge_mm: float = 1.2,
    oval_fn=None,
    log=print,
) -> FaceDepth:
    """``L_world``: (468, 2) photo landmarks in world x / y (metres). ``zscan``: (H, W) scan front depth on ``grid``."""
    model = fmodel.m
    posed = fmodel.posed(fit.theta)
    lm3 = fmodel.landmark_points(fit.theta)
    # similarity in the image plane (uniform scale also applies to depth), then the thin-plate residual warp
    def sim(p: np.ndarray) -> np.ndarray:
        q = np.empty_like(p)
        q[:, :2] = fit.s * (p[:, :2] @ fit.R.T) + fit.t
        q[:, 2] = fit.s * p[:, 2]
        return q

    V = sim(posed)
    Q = sim(lm3)
    tri = head_front_triangles(model, posed)
    zmh_raw = rasterize_depth(V[:, :2], V[:, 2], model.faces[tri], grid)

    # thin-plate map: world (photo) coordinates -> MakeHuman-aligned coordinates
    rbf = None
    for lam in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-8):  # the smoothest warp that follows the landmarks to warp_rms_mm
        rbf = RBFInterpolator(L_world, Q[:, :2], kernel="thin_plate_spline", smoothing=lam, degree=1)
        if np.sqrt(((rbf(L_world) - Q[:, :2]) ** 2).sum(axis=1).mean()) * 1000 <= warp_rms_mm:
            break
    gx, gy = grid.centres()
    oval = oval_mask(grid, L_world[fm.oval])
    oval_info: dict = {}
    if oval_fn is not None:  # adjust the landmark oval to the photo (hairline)
        oval, oval_info = oval_fn(grid, oval)
    # surfaces that turn away from the camera (cheek sides, jaw, hairline folds): a few millimetres of x error would be
    # tens of millimetres of depth error there, so the relief region ends where the scan surface gets steeper than
    # ``slope_max`` (an interior steep patch, like the sides of the nose, stays inside)
    scan_ok0 = np.isfinite(zscan)
    zf = np.where(scan_ok0, zscan, np.nanmin(zscan[scan_ok0]) if scan_ok0.any() else 0.0).astype(np.float32)
    zb = cv2.GaussianBlur(zf, (0, 0), 1.0e-3 / grid.res)
    gy_, gx_ = np.gradient(zb, grid.res)
    steep = np.hypot(gx_, gy_) > slope_max
    k5 = max(int(round(3e-3 / grid.res)) | 1, 3)
    cut = oval & ~steep
    cut = cv2.morphologyEx(cut.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k5, k5), np.uint8))
    n_, lab_ = cv2.connectedComponents(cut)
    if n_ > 2:
        cut = (lab_ == (1 + int(np.argmax(np.bincount(lab_.ravel())[1:])))).astype(np.uint8)
    # fill holes (interior steep patches) so only the outer steep rim is removed
    ff = cut.copy()
    cv2.floodFill(ff, np.zeros((grid.H + 2, grid.W + 2), np.uint8), (0, 0), 2)
    oval_slope = (cut > 0) | (ff == 0)
    oval_info["steep_removed_px"] = int(oval.sum() - (oval & oval_slope).sum())
    oval = oval & oval_slope
    near = cv2.dilate(oval.astype(np.uint8), np.ones((int(0.03 / grid.res) | 1,) * 2, np.uint8)).astype(bool)
    ys, xs = np.nonzero(near)
    h = rbf(np.stack([gx[ys, xs], gy[ys, xs]], axis=1))
    hx, hy = grid.to_px(h[:, 0], h[:, 1])
    mapx = np.full((grid.H, grid.W), -1.0, np.float32)
    mapy = np.full((grid.H, grid.W), -1.0, np.float32)
    mapx[ys, xs] = hx - 0.5
    mapy[ys, xs] = hy - 0.5
    zfill = np.where(np.isfinite(zmh_raw), zmh_raw, 0.0).astype(np.float32)
    cov = np.isfinite(zmh_raw).astype(np.float32)
    zw = cv2.remap(zfill, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)
    cw = cv2.remap(cov, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)
    valid = (cw > 0.995) & near
    zmh = np.where(valid, zw / np.maximum(cw, 1e-6), np.nan).astype(np.float32)

    # holes inside the (slightly dilated) oval: eyelid openings, lip slit; fill harmonically, add the eyeball bulge
    inner = oval & ~valid
    hole = cv2.dilate(inner.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & near & ~valid
    zfilled = harmonic_fill(zmh, hole)
    if eye_bulge_mm > 0 and hole.any():
        dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
        # per connected hole: normalise the distance by its own maximum
        n, comp = cv2.connectedComponents(hole.astype(np.uint8))
        bulge = np.zeros_like(dist)
        for c in range(1, n):
            m = comp == c
            mx = dist[m].max()
            if mx > 1.5:
                bulge[m] = (dist[m] / mx) ** 0.8 * eye_bulge_mm * 1e-3
        zfilled = zfilled + bulge
    zmh = np.where(np.isfinite(zfilled), zfilled, zmh)

    # join with the scan (Poisson-style): Znew = Zmh + c, with c the harmonic extension of the residual
    # (Zscan - Zmh) on a band inside the oval boundary, so the face matches the head at the boundary and keeps its own
    # details inside, without a step or a crease
    sd = cv2.distanceTransform(oval.astype(np.uint8), cv2.DIST_L2, 5) * grid.res * 1000.0  # mm inside the oval
    scan_ok = np.isfinite(zscan)
    rim = oval & (sd <= rim_mm) & np.isfinite(zmh) & scan_ok
    interior = oval & (sd > rim_mm) & np.isfinite(zmh)
    r = np.where(rim, zscan - zmh, 0.0)
    if rim.any():
        # robust residual: outliers of the scan (hair locks, dents) must not become bumps of the face
        from scipy import ndimage as ndi

        idx = ndi.distance_transform_edt(~rim, return_distances=False, return_indices=True)
        filled = r[idx[0], idx[1]].astype(np.float32)
        f = 4
        small = cv2.resize(filled, (grid.W // f, grid.H // f), interpolation=cv2.INTER_AREA)
        ksz = max(int(0.02 / (grid.res * f)) | 1, 3)
        med = ndi.median_filter(small, size=ksz, mode="nearest")
        med = cv2.resize(med, (grid.W, grid.H), interpolation=cv2.INTER_LINEAR)
        r = np.where(rim, np.clip(r, med - 0.008, med + 0.008), 0.0)
    rs = normalized_blur(r, rim.astype(np.float32), rim_sigma_mm * 1e-3 / grid.res)
    seed = np.where(rim | (oval & ~interior), rs, np.nan).astype(np.float32)
    seed = np.where(interior, np.nan, seed)
    c = harmonic_fill(np.where(oval, seed, np.nan), interior)
    c = np.where(np.isfinite(c), c, rs)
    znew = zmh + c
    # no detail finer than the mesh can carry: light blur, only inside the oval
    zs_ = cv2.GaussianBlur(np.where(np.isfinite(znew), znew, 0.0).astype(np.float32), (0, 0), smooth_mm * 1e-3 / grid.res)
    ok_ = cv2.GaussianBlur(np.isfinite(znew).astype(np.float32), (0, 0), smooth_mm * 1e-3 / grid.res)
    znew = np.where(np.isfinite(znew), zs_ / np.maximum(ok_, 1e-6), np.nan)
    w = smoothstep01(sd / feather_mm)
    delta = np.where(np.isfinite(znew) & scan_ok, w * np.clip(znew - zscan, -max_delta_mm * 1e-3, max_delta_mm * 1e-3), 0.0).astype(np.float32)
    info = {
        "oval_px": int(oval.sum()),
        **oval_info,
        "hole_px": int(hole.sum()),
        "correction_mm": [float(np.nanmin(c[oval]) * 1000), float(np.nanmax(c[oval]) * 1000)] if oval.any() else None,
        "delta_mm": [float(delta.min() * 1000), float(delta.max() * 1000)],
        "rms_landmark_warp_mm": float(
            np.sqrt(((rbf(L_world) - Q[:, :2]) ** 2).sum(axis=1).mean()) * 1000
        ),
    }
    log(f"face depth: {info}")
    out_d = cv2.distanceTransform((~oval).astype(np.uint8), cv2.DIST_L2, 5) * grid.res * 1000.0
    sd_signed = (sd - out_d).astype(np.float32)
    return FaceDepth(grid, zmh, znew.astype(np.float32), zscan, w.astype(np.float32), oval, delta, info, sd_signed)
