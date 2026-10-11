"""Numeric single-view UV bake consistency; private source images are never exported."""

from __future__ import annotations

import cv2
import numpy as np
from flamehead.colour import skin_samples, to_lab
from flamehead.texture import project_samples, sample, visibility
from scipy import ndimage
from twinrefine.scan import welded_vertex_normals


def ghost_alignment(front, side, overlap, *, radius=8, patch=24):
    """Estimate local UV displacement from high-pass luminance, with masked NCC.

    This is a texture-registration proxy, not a validated perception metric.
    Only fully overlapping, textured patches with NCC >= 0.5 contribute;
    peak-at-search-bound patches are reported rather than called aligned.
    """
    front, side = np.asarray(front, np.float32), np.asarray(side, np.float32)
    a = front - ndimage.gaussian_filter(front, 2)
    b = side - ndimage.gaussian_filter(side, 2)
    valid = ndimage.binary_erosion(overlap, iterations=radius + 3)
    shifts, before, after, correlation = [], [], [], []
    attempted, at_bound = 0, 0
    feature_support = ndimage.binary_erosion(overlap, iterations=3)
    unaligned_rms, unaligned_ncc = None, None
    if feature_support.sum() >= 32:
        av, bv = a[feature_support], b[feature_support]
        unaligned_rms = float(np.sqrt(np.mean((av - bv) ** 2)))
        if av.std() > 1e-5 and bv.std() > 1e-5:
            unaligned_ncc = float(np.corrcoef(av, bv)[0, 1])
    for y in range(radius + 3, a.shape[0] - patch - radius - 3, patch):
        for x in range(radius + 3, a.shape[1] - patch - radius - 3, patch):
            if not valid[y:y + patch, x:x + patch].all():
                continue
            reference = a[y:y + patch, x:x + patch]
            if reference.std() < 0.5:
                continue
            attempted += 1
            search = b[y - radius:y + patch + radius, x - radius:x + patch + radius]
            score = cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (dx, dy) = cv2.minMaxLoc(score)
            if peak < 0.5:
                continue
            dx, dy = dx - radius, dy - radius
            at_bound += int(abs(dx) == radius or abs(dy) == radius)
            shifts.append([dx, dy])
            original = b[y:y + patch, x:x + patch]
            aligned = b[y + dy:y + dy + patch, x + dx:x + dx + patch]
            before.append(float(np.mean((reference - original) ** 2)))
            after.append(float(np.mean((reference - aligned) ** 2)))
            correlation.append(peak)
    report = {"attempted_patches": attempted, "accepted_patches": len(shifts),
              "search_radius_px": radius, "search_bound_patches": at_bound,
              "method": "masked local high-pass L* NCC; displacement of side features relative to front",
              "feature_texels": int(feature_support.sum()),
              "unaligned_highpass_rms_l": unaligned_rms, "unaligned_highpass_ncc": unaligned_ncc}
    if not shifts:
        return {**report, "median_displacement_px": None, "reason": "insufficient textured overlap"}
    shifts = np.asarray(shifts)
    return {**report, "median_displacement_px": float(np.median(np.linalg.norm(shifts, axis=1))),
            "median_shift_xy_px": np.median(shifts, axis=0).tolist(),
            "p95_displacement_px": float(np.percentile(np.linalg.norm(shifts, axis=1), 95)),
            "median_ncc": float(np.median(correlation)),
            "highpass_rms_before": float(np.sqrt(np.mean(before))),
            "highpass_rms_after_local_alignment": float(np.sqrt(np.mean(after)))}


def single_view_samples(posed, faces, tri, bary, cameras, photos, masks, face_masks, gate, unmasked_weights=None):
    """Bake each source independently at identical UV texels; no blending or filling.

    When ``unmasked_weights`` is a dict it receives each view's visibility weight BEFORE the photo face-oval mask, so
    regions outside the oval (ears) can be inspected too.
    """
    member = np.zeros(len(next(iter(posed.values()))), bool)
    member[masks["face"]] = True
    semantic_face = member[tri].all(1)
    for key in ("eye_region", "lips", "left_eyeball", "right_eyeball"):
        member[masks.get(key, [])] = False
    semantic_skin = member[tri].all(1)
    samples, weights = {}, {}
    for name, camera in cameras.items():
        mesh = posed[name]
        points = np.einsum("ij,ijk->ik", bary, mesh[tri])
        normals = welded_vertex_normals(mesh, faces)
        normals = np.einsum("ij,ijk->ik", bary, normals[tri])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        depth, edge, scale = visibility(mesh, faces, camera)
        colour, weight = project_samples(points, normals, photos[name], camera, depth, edge, scale)
        if unmasked_weights is not None:
            unmasked_weights[name] = weight * gate
        if face_masks is not None:
            mask = sample(face_masks[name], camera.project(points)[:, :2]).reshape(-1)
            weight *= mask
        weights[name] = weight * gate
        samples[name] = to_lab(colour.astype(np.float32) / 255)
    return samples, weights, semantic_skin, semantic_face


def compare_single_views(samples, weights, semantic_skin, semantic_face, shape, y, x, *, support=None, skin_support=None):
    """Colour and feature agreement with front on independent source bakes.

    Optionally restrict to the exact same texels visible before AND after a
    camera change; this avoids improvements obtained by shrinking overlap.
    """
    result = {}
    front = samples["front"]
    front_l = np.zeros(shape, np.float32)
    front_l[y, x] = front[:, 0]
    for name in ("left", "right"):
        overlap = (weights["front"] > 0.08) & (weights[name] > 0.08)
        if support is not None:
            overlap &= support[name]
        skin = overlap & semantic_skin & skin_samples(front) & skin_samples(samples[name])
        if skin_support is not None:
            skin &= skin_support[name]
        report = {"overlap_texels": int(overlap.sum()), "skin_texels": int(skin.sum())}
        if skin.sum() >= 32:
            a, b = front[skin], samples[name][skin]
            offset = np.clip(np.median(a - b, axis=0), [-15, -20, -20], [15, 20, 20])
            difference = a - (b + offset)
            report.update(
                median_offset_lab=offset.tolist(),
                after_delta_e76_of_mean=float(np.linalg.norm(difference.mean(0))),
                after_mean_paired_delta_e76=float(np.linalg.norm(difference, axis=1).mean()),
                after_median_paired_delta_e76=float(np.median(np.linalg.norm(difference, axis=1))),
            )
        mask = np.zeros(shape, bool)
        mask[y, x] = overlap & semantic_face
        side_l = np.zeros(shape, np.float32)
        side_l[y, x] = samples[name][:, 0]
        report["ghost_alignment"] = ghost_alignment(front_l, side_l, mask)
        result[name] = report
    return result


def _skin_statistics(lab):
    if len(lab) == 0:
        return {"texels": 0, "skin_fraction": None, "dark_fraction": None}
    return {"texels": int(len(lab)), "skin_fraction": float(skin_samples(lab).mean()),
            "dark_fraction": float((lab[:, 0] < 38).mean())}


def projection_skin_check(samples, weights, unmasked_weights, ear_members, semantic_face):
    """Fraction of each side photo's samples on the ears and the face that look like skin.

    A side camera that is misaligned samples hair or background onto the ear (a 2 cm structure: a few millimetres
    of error already move it onto hair), which shows as dark/non-skin samples. This is independent of the landmarks
    the camera was fitted to, but it is a proxy: hair genuinely covering part of an ear lowers every camera's score,
    so only compare cameras on the same photos.
    """
    result = {}
    for name in ("left", "right"):
        ear = (unmasked_weights[name] > 0.08) & ear_members[name]
        face = (weights[name] > 0.08) & semantic_face
        result[name] = {"ear": _skin_statistics(samples[name][ear]), "face": _skin_statistics(samples[name][face])}
    return result
