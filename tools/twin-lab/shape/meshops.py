"""Mesh post-processing, coordinate normalisation and silhouette camera fitting for the shape lab.

Everything here is plain numpy / trimesh / scipy / OpenCV (no torch), so it can be unit-tested and reused by the
texture agent. Conventions of the OUTPUT (same as the web app, see docs/ARCHITECTURE.md): metres, right handed,
+Y up, the character faces +Z (so the character's left is +X), feet on y = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

# --------------------------------------------------------------------------------------------------------------------
# cleanup
# --------------------------------------------------------------------------------------------------------------------


def _rebuild(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces), process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def keep_main_components(mesh: trimesh.Trimesh, floater_ratio: float = 0.02) -> tuple[trimesh.Trimesh, dict]:
    """Keep the largest connected component (by surface area) plus any component whose area is at least
    `floater_ratio` of the largest one. Everything else is a floater and is dropped."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, {"components": len(parts), "kept": len(parts), "removedFaces": 0}
    areas = np.array([p.area for p in parts])
    main = float(areas.max())
    keep = [p for p, a in zip(parts, areas, strict=True) if a >= floater_ratio * main]
    removed_faces = int(sum(len(p.faces) for p in parts) - sum(len(p.faces) for p in keep))
    merged = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    return merged, {"components": len(parts), "kept": len(keep), "removedFaces": removed_faces}


def decimate(mesh: trimesh.Trimesh, target_faces: int, agg: float = 6.0) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation to about `target_faces` triangles (fast_simplification)."""
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    import fast_simplification

    pts, tris = fast_simplification.simplify(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int32),
        target_count=int(target_faces),
        agg=agg,
    )
    return _rebuild(pts, tris)


def cleanup(
    mesh: trimesh.Trimesh,
    floater_ratio: float = 0.02,
    smooth_iters: int = 0,
    target_faces: int = 0,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Full cleanup. Returns (final mesh, high-res cleaned mesh, stats).

    order: weld/degenerate removal -> floaters -> hole fill -> (optional Taubin smoothing) -> normals -> decimation ->
    normals again. The high-res mesh is the cleaned mesh BEFORE decimation."""
    stats: dict = {"inputVertices": int(len(mesh.vertices)), "inputFaces": int(len(mesh.faces))}
    mesh = _rebuild(mesh.vertices, mesh.faces)
    mesh, comp = keep_main_components(mesh, floater_ratio)
    stats["components"] = comp
    mesh = _rebuild(mesh.vertices, mesh.faces)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    trimesh.repair.fix_normals(mesh)
    stats["hiresFaces"] = int(len(mesh.faces))
    stats["hiresWatertight"] = bool(mesh.is_watertight)
    hires = mesh.copy()
    if target_faces > 0:
        mesh = decimate(mesh, target_faces)
        trimesh.repair.fix_normals(mesh)
    stats["finalVertices"] = int(len(mesh.vertices))
    stats["finalFaces"] = int(len(mesh.faces))
    stats["finalWatertight"] = bool(mesh.is_watertight)
    return mesh, hires, stats


# --------------------------------------------------------------------------------------------------------------------
# orientation and normalisation
# --------------------------------------------------------------------------------------------------------------------


def _slab_bbox_area(v: np.ndarray, axis: int, lo: float, hi: float) -> float:
    sel = v[(v[:, axis] >= lo) & (v[:, axis] <= hi)]
    if len(sel) < 3:
        return 0.0
    lat = [a for a in range(3) if a != axis]
    ext = sel[:, lat].max(0) - sel[:, lat].min(0)
    return float(ext[0] * ext[1])


def detect_frame(v: np.ndarray, default_up: int = 1, default_forward: int = 2) -> dict:
    """Detect which source axis is 'up' and which is 'forward' for a standing person.

    up axis      : the axis with the largest extent; its sign from the footprint area of the lowest 3 % slab (two
                   shoes, large) versus the highest 3 % slab (top of the head, small).
    forward axis : the horizontal axis with the smaller extent (chest depth < shoulder / arm span); its sign from the
                   fact that the soles' centroid lies in front of the shank centroid (toes point forward).
    Returns the detection with a confidence flag and the fallback used when a cue is ambiguous."""
    ext = v.max(0) - v.min(0)
    up = int(np.argmax(ext))
    lo, height = float(v[:, up].min()), float(ext[up])
    bottom = _slab_bbox_area(v, up, lo, lo + 0.03 * height)
    top = _slab_bbox_area(v, up, lo + 0.97 * height, lo + height)
    ratio = bottom / top if top > 0 else float("inf")
    if ratio > 1.5:
        up_sign, up_conf = +1, True
    elif ratio < 1 / 1.5:
        up_sign, up_conf = -1, True
    else:
        up_sign, up_conf = +1, False  # ambiguous: assume the generator's native +Y up

    lat = [a for a in range(3) if a != up]
    forward = lat[0] if ext[lat[0]] < ext[lat[1]] else lat[1]
    fwd_conf = abs(ext[lat[0]] - ext[lat[1]]) > 0.03 * height
    if not fwd_conf:
        forward = default_forward if default_forward in lat else lat[0]

    t = (v[:, up] - lo) / height
    if up_sign < 0:
        t = 1.0 - t
    foot = v[t < 0.04]
    shank = v[(t > 0.12) & (t < 0.28)]
    diff = float(foot[:, forward].mean() - shank[:, forward].mean()) if len(foot) and len(shank) else 0.0
    fwd_sign = +1 if diff >= 0 else -1
    sign_conf = abs(diff) > 0.01 * height
    if not sign_conf:
        fwd_sign = +1
    return {
        "upAxis": up,
        "upSign": up_sign,
        "forwardAxis": forward,
        "forwardSign": fwd_sign,
        "confident": bool(up_conf and fwd_conf and sign_conf),
        "cues": {
            "footprintBottomOverTop": None if math.isinf(ratio) else round(ratio, 3),
            "extentsSource": [round(float(e), 4) for e in ext],
            "soleMinusShankAlongForward": round(diff, 4),
        },
    }


def rotation_from_detection(det: dict) -> np.ndarray:
    """3x3 rotation R with  v_std = R @ v_src  so that source up -> +Y and source forward -> +Z (proper rotation)."""
    up_vec = np.zeros(3)
    up_vec[det["upAxis"]] = det["upSign"]
    fwd_vec = np.zeros(3)
    fwd_vec[det["forwardAxis"]] = det["forwardSign"]
    x_vec = np.cross(up_vec, fwd_vec)  # y cross z = x in a right handed frame
    # rows of R are the images of the standard axes expressed in the source frame
    return np.stack([x_vec, up_vec, fwd_vec], axis=0)


@dataclass
class Normalization:
    rotation: np.ndarray  # 3x3
    scale: float  # metres per source unit
    translation: np.ndarray  # 3, applied after rotation and scale
    height_m: float
    source_bbox: dict = field(default_factory=dict)
    final_bbox: dict = field(default_factory=dict)

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation * self.scale
        m[:3, 3] = self.translation
        return m


def normalize(v: np.ndarray, det: dict, height_m: float) -> tuple[np.ndarray, Normalization]:
    """Rotate to the standard frame, scale the height to `height_m`, put the feet on y = 0 and centre x / z on the
    contact patch of the feet (mid of the bbox of the lowest 3 % slab)."""
    src_min, src_max = v.min(0), v.max(0)
    r = rotation_from_detection(det)
    vr = v @ r.T
    h_src = float(vr[:, 1].max() - vr[:, 1].min())
    scale = height_m / h_src
    vs = vr * scale
    y_min = float(vs[:, 1].min())
    foot = vs[vs[:, 1] < y_min + 0.03 * height_m]
    cx = 0.5 * float(foot[:, 0].min() + foot[:, 0].max())
    cz = 0.5 * float(foot[:, 2].min() + foot[:, 2].max())
    t = np.array([-cx, -y_min, -cz])
    out = vs + t
    norm = Normalization(
        rotation=r,
        scale=scale,
        translation=t,
        height_m=height_m,
        source_bbox={"min": src_min.tolist(), "max": src_max.tolist()},
        final_bbox={"min": out.min(0).tolist(), "max": out.max(0).tolist()},
    )
    return out, norm


# --------------------------------------------------------------------------------------------------------------------
# silhouette camera fit
# --------------------------------------------------------------------------------------------------------------------

VIEW_YAW_DEG = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}


def project_x(v: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Horizontal image coordinate (metres, +right) of an orthographic camera on the +Z side rotated by `yaw_deg`
    about +Y (0 front, 90 = camera on the character's left side (+X), 180 back, 270 right side)."""
    a = math.radians(yaw_deg)
    return v[:, 0] * math.cos(a) - v[:, 2] * math.sin(a)


def raster_silhouette(
    v: np.ndarray,
    f: np.ndarray,
    yaw_deg: float,
    s: float,
    u0: float,
    v0: float,
    w: int,
    h: int,
    ds: float = 0.5,
    mirror: bool = False,
    sy: float | None = None,
) -> np.ndarray:
    """Binary silhouette (uint8 0/255) of the mesh under  u = u0 + s * x',  v = v0 - s * y  at scale `ds`.
    `mirror` flips x' (mirror about the image column of the world origin), used as a handedness sanity check.
    `sy` optional separate vertical scale (px per metre) for the anisotropic diagnostic fit."""
    xp = project_x(v, yaw_deg)
    if mirror:
        xp = -xp
    pts = np.stack([(u0 + s * xp) * ds, (v0 - (s if sy is None else sy) * v[:, 1]) * ds], axis=1)
    tri = np.round(pts[f] * 16.0).astype(np.int32)
    img = np.zeros((max(1, int(round(h * ds))), max(1, int(round(w * ds)))), dtype=np.uint8)
    # one convex fill per triangle: cv2.fillPoly over many contours uses the even-odd rule, which would punch holes
    # wherever front and back surfaces overlap in the projection.
    for t in tri:
        cv2.fillConvexPoly(img, t, 255, lineType=cv2.LINE_8, shift=4)
    return img


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def alpha_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fit_orthographic_camera(
    v: np.ndarray, f: np.ndarray, mask: np.ndarray, yaw_deg: float, ds: float = 0.5, max_faces: int = 25000
) -> dict:
    """Fit (px per metre, image position of the world origin) so the mesh silhouette matches the person mask.

    Model:  u = u0 + s * x'   (x' = x cos yaw - z sin yaw),   v = v0 - s * y      (v grows downwards).
    Initial guess: s from the mask bbox height vs. the mesh height, u0 from the bbox centre, v0 = bbox bottom (feet).
    Then Nelder-Mead on (s, u0, v0) maximising the silhouette IoU."""
    h_img, w_img = mask.shape[:2]
    m = mask.astype(bool)  # any non-zero pixel is person (works for 0/1 and 0/255 masks)
    x0, y0, x1, y1 = alpha_bbox(m)
    height_m = float(v[:, 1].max() - v[:, 1].min())
    s_init = (y1 - y0 + 1) / height_m
    xp = project_x(v, yaw_deg)
    u_init = 0.5 * (x0 + x1 + 1) - s_init * 0.5 * float(xp.min() + xp.max())
    v_init = float(y1 + 1) + s_init * float(v[:, 1].min())

    # proxy mesh for speed
    if len(f) > max_faces:
        try:
            import fast_simplification

            pv, pf = fast_simplification.simplify(
                np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(f, dtype=np.int32), target_count=max_faces
            )
        except Exception:  # noqa: BLE001 - fall back to the full mesh
            pv, pf = v, f
    else:
        pv, pf = v, f

    target = cv2.resize(m.astype(np.uint8), (max(1, int(round(w_img * ds))), max(1, int(round(h_img * ds)))), interpolation=cv2.INTER_AREA) > 0

    def loss(p: np.ndarray) -> float:
        s, u0, v0 = p
        sil = raster_silhouette(pv, pf, yaw_deg, s, u0, v0, w_img, h_img, ds) > 0
        return 1.0 - _iou(sil, target)

    iou_init = 1.0 - loss(np.array([s_init, u_init, v_init]))
    simplex = np.array(
        [
            [s_init, u_init, v_init],
            [s_init * 1.02, u_init, v_init],
            [s_init, u_init + 8.0, v_init],
            [s_init, u_init, v_init + 8.0],
        ]
    )
    res = minimize(loss, x0=np.array([s_init, u_init, v_init]), method="Nelder-Mead",
                   options={"initial_simplex": simplex, "xatol": 0.05, "fatol": 1e-5, "maxiter": 400})
    s, u0, v0 = (float(x) for x in res.x)
    iou_fit = 1.0 - float(res.fun)
    if iou_fit < iou_init:  # never accept a worse fit
        s, u0, v0, iou_fit = s_init, u_init, v_init, iou_init
    # diagnostic: separate horizontal / vertical scale (how much the generated proportions deviate from the photo)
    def loss_aniso(p: np.ndarray) -> float:
        sx, sy_, u0_, v0_ = p
        sil = raster_silhouette(pv, pf, yaw_deg, sx, u0_, v0_, w_img, h_img, ds, sy=sy_) > 0
        return 1.0 - _iou(sil, target)

    simplex4 = np.array(
        [[s, s, u0, v0], [s * 1.02, s, u0, v0], [s, s * 1.02, u0, v0], [s, s, u0 + 8.0, v0], [s, s, u0, v0 + 8.0]]
    )
    res2 = minimize(loss_aniso, x0=np.array([s, s, u0, v0]), method="Nelder-Mead",
                    options={"initial_simplex": simplex4, "xatol": 0.05, "fatol": 1e-5, "maxiter": 500})
    aniso = {
        "pxPerMeterX": float(res2.x[0]),
        "pxPerMeterY": float(res2.x[1]),
        "originPx": [float(res2.x[2]), float(res2.x[3])],
        "silhouetteIoU": round(1.0 - float(res2.fun), 4),
        "horizontalOverVertical": round(float(res2.x[0] / res2.x[1]), 4),
        "note": "diagnostic only: the generated body is this much narrower (<1) or wider (>1) relative to its height than the photo",
    }
    # mirror check: is a horizontally mirrored silhouette a better match? (mesh handedness sanity check)
    iou_mirror = _iou(raster_silhouette(pv, pf, yaw_deg, s, u0, v0, w_img, h_img, ds, mirror=True) > 0, target)
    return {
        "yawDeg": yaw_deg,
        "imageSize": [int(w_img), int(h_img)],
        "pxPerMeter": s,
        "originPx": [u0, v0],
        "silhouetteIoU": round(iou_fit, 4),
        "silhouetteIoUInitialGuess": round(iou_init, 4),
        "silhouetteIoUMirroredMesh": round(iou_mirror, 4),
        "maskBBoxPx": [x0, y0, x1, y1],
        "initialGuess": {"pxPerMeter": s_init, "originPx": [u_init, v_init]},
        "anisotropicFit": aniso,
    }


def overlay_image(v: np.ndarray, f: np.ndarray, cam: dict, mask: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """BGR debug image: green = photo person mask, red = projected mesh silhouette, yellow = both."""
    w, h = cam["imageSize"]
    sil = raster_silhouette(v, f, cam["yawDeg"], cam["pxPerMeter"], cam["originPx"][0], cam["originPx"][1], w, h, ds=scale) > 0
    m = cv2.resize(mask.astype(np.uint8), (sil.shape[1], sil.shape[0]), interpolation=cv2.INTER_AREA) > 0
    out = np.zeros((*sil.shape, 3), dtype=np.uint8)
    out[..., 1] = m * 200
    out[..., 2] = sil * 200
    return out
