"""CIELAB diagnostics and photo-referenced skin transitions; RGB inputs are sRGB."""

import cv2
import numpy as np
from scipy.spatial import cKDTree
from twintex.raster import rasterize_uv

from .seam import smoothstep
from .texture import sample


def to_lab(rgb):
    a = np.asarray(rgb, np.float32)
    if np.asarray(rgb).dtype == np.uint8:
        a = a / 255
    shape = a.shape
    if a.size == 0:
        return np.empty(shape, np.float32)
    return cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(shape)


def from_lab(lab):
    shape = np.asarray(lab).shape
    if np.asarray(lab).size == 0:
        return np.empty(shape, np.float32)
    return cv2.cvtColor(np.asarray(lab, np.float32).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(shape)


def skin_samples(lab):
    # Midtone, warm skin; exclude beard, glasses, eyes, hair and clipped highlights.
    return (lab[:, 0] > 32) & (lab[:, 0] < 88) & (lab[:, 1] > 2) & (lab[:, 1] < 32) & (lab[:, 2] > 2) & (lab[:, 2] < 40)


def paired_colour(reference, actual, mask):
    if np.sum(mask) < 16:
        raise ValueError("Too few matched skin samples for colour diagnostics")
    a, b = to_lab(reference)[mask], to_lab(actual)[mask]
    photo, baked = a.mean(0, dtype=np.float64), b.mean(0, dtype=np.float64)
    delta = baked - photo
    return {
        "sample_count": int(mask.sum()),
        "photo_mean_lab": photo.tolist(),
        "texture_mean_lab": baked.tolist(),
        "delta_lab": delta.tolist(),
        "chroma_pass": bool(np.all(np.abs(delta[1:]) < 3)),
    }


def raster_surface(mc, size, face_mask=None):
    faces = np.arange(len(mc.F)) if face_mask is None else np.flatnonzero(face_mask)
    # Corners carry separate UVs at seams and at each interpolated cut edge.
    uv = mc.C[faces].reshape(-1, 2)
    fid, bary = rasterize_uv(uv * size, np.arange(len(uv)).reshape(-1, 3), size, size)
    y, x = np.nonzero(fid >= 0)
    tri = mc.F[faces[fid[y, x]]]
    points = np.einsum("ij,ijk->ik", bary[y, x], mc.P[tri])
    return y, x, tri, bary[y, x], points


def recolour_and_crossfade(
    scan_mc, flame_mc, scan_atlas, flame_atlas, scan_ring, flame_ring, target_lab, band=0.022, diagnostics=None
):
    """Photo chroma on scan skin over 75mm; spatially varying texture blend over 22mm."""
    scan_atlas, flame_atlas = scan_atlas.copy(), flame_atlas.copy()
    curve = scan_mc.P[scan_ring]
    tree = cKDTree(curve)
    dist = tree.query(scan_mc.P)[0]
    near_faces = (dist[scan_mc.F] < 0.08).any(1)
    sy, sx, st, sb, sp = raster_surface(scan_mc, scan_atlas.shape[0], near_faces)
    sd = tree.query(sp)[0]
    rgb = scan_atlas[sy, sx]
    lab = to_lab(rgb)
    skin = skin_samples(lab)
    neck_height = np.quantile(curve[:, 1], 0.20) + 0.012
    near_neck = skin & (sd < 0.025) & (sp[:, 1] < neck_height)
    neck_before = lab[near_neck].mean(0, dtype=np.float64) if near_neck.any() else None
    # Preserve hair and beard detail while correcting the skin surrounding it.
    omega = (1 - smoothstep((sd - 0.025) / 0.050)) * skin
    source_mean = (
        lab[skin & (sd < 0.025)].mean(0, dtype=np.float64) if (skin & (sd < 0.025)).any() else np.asarray(target_lab)
    )
    delta_l = np.clip(float(target_lab[0] - source_mean[0]), -12, 12)
    lab[:, 0] += delta_l * omega
    if near_neck.any():
        # Neck illumination differs from the face. Match its low-frequency level
        # separately, retaining pixel contrast and fading continuously up the jaw.
        neck_delta_l = np.clip(float(target_lab[0] - lab[near_neck, 0].mean(dtype=np.float64)), -12, 12)
        neck_weight = 1 - smoothstep((sp[:, 1] - neck_height) / 0.020)
        lab[:, 0] += neck_delta_l * neck_weight * omega
    lab[:, 1:] += (np.asarray(target_lab)[None, 1:] - lab[:, 1:]) * omega[:, None]
    corrected = np.clip(np.rint(from_lab(lab) * 255), 0, 255).astype(np.uint8)
    scan_atlas[sy, sx] = corrected
    # Bake the scan texture onto the inserted band's actual surface by closest points.
    fd = cKDTree(flame_mc.P[flame_ring]).query(flame_mc.P)[0]
    near_flame = (fd[flame_mc.F] < band).any(1)
    fy, fx, ft, fb, fp = raster_surface(flame_mc, flame_atlas.shape[0], near_flame)
    distance = cKDTree(flame_mc.P[flame_ring]).query(fp)[0]
    import trimesh

    from .geometry import closest_surface

    # Use the original scan's retained collar only: no sample from deleted generic face.
    cp, ids = closest_surface(scan_mc.P, scan_mc.F[near_faces], fp)
    selected_corners = scan_mc.C[near_faces][ids]
    bary = trimesh.triangles.points_to_barycentric(scan_mc.P[scan_mc.F[near_faces][ids]], cp)
    uv = np.einsum("ij,ijk->ik", bary, selected_corners)
    scan_rgb = sample(scan_atlas, uv * scan_atlas.shape[0]).astype(np.float32) / 255
    flame_rgb = flame_atlas[fy, fx].astype(np.float32) / 255
    # A 22mm smooth transition retains photographed under-chin beard in the inner band.
    alpha = smoothstep(distance / band)
    colour = to_lab(flame_rgb) * alpha[:, None] + to_lab(scan_rgb) * (1 - alpha[:, None])
    flame_atlas[fy, fx] = np.clip(np.rint(from_lab(colour) * 255), 0, 255).astype(np.uint8)
    neck_after = to_lab(scan_atlas[sy, sx])[near_neck]
    if diagnostics is not None:
        diagnostics.update(neck_uv=np.column_stack((sx[near_neck] + 0.5, sy[near_neck] + 0.5)) / scan_atlas.shape[0])
    report = {
        "scan_colour_falloff_mm": 75.0,
        "texture_band_mm": band * 1000,
        "modified_scan_texels": int((omega > 0).sum()),
        "crossfaded_flame_texels": len(fy),
        "scan_band_before_mean_lab": source_mean.tolist(),
        "photo_face_mean_lab": list(target_lab),
        "neck_sample_count": int(near_neck.sum()),
        "neck_before_mean_lab": neck_before.tolist() if neck_before is not None else None,
    }
    if len(neck_after):
        neck_mean = neck_after.mean(0, dtype=np.float64)
        report["neck_mean_lab"] = neck_mean.tolist()
        report["neck_minus_photo_lab"] = (neck_mean - target_lab).tolist()
        report["neck_chroma_pass"] = bool(np.all(np.abs(neck_mean[1:] - target_lab[1:]) < 3))
    return scan_atlas, flame_atlas, report
