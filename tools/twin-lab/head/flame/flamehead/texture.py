"""Perspective-visible photo baking and deterministic atlas fallback."""

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from twinrefine.scan import welded_vertex_normals
from twintex.colorspace import linear_to_srgb, srgb_to_linear
from twintex.raster import rasterize_uv, zbuffer


def uv_atlas(vertices, faces, size, use_xatlas=True):
    if use_xatlas:
        try:
            import xatlas
        except ImportError:
            pass
        else:
            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
            pack = xatlas.PackOptions()
            pack.resolution, pack.padding = size, 6
            atlas.generate(pack_options=pack)
            mapping, indices, uv = atlas[0]
            # xatlas uses a Cartesian UV axis; glTF uses a top-left image origin.
            uv = np.asarray(uv, np.float32)
            uv[:, 1] = 1 - uv[:, 1]
            return mapping.astype(int), indices.astype(int), uv, "xatlas"
    # An independent right-triangle chart per face cannot fold or overlap. Its
    # density is deliberately conservative; this is a fallback, not a facial unwrap.
    side = int(np.ceil(np.sqrt(len(faces))))
    if size / side < 5:
        raise ValueError("Fallback atlas needs at least five pixels per triangle chart")
    cell = 1 / side
    base = np.column_stack((np.arange(len(faces)) % side, np.arange(len(faces)) // side)) * cell
    pad = 1.5 / size
    uv = np.stack((base + pad, base + [cell - pad, pad], base + [pad, cell - pad]), axis=1).reshape(-1, 2)
    return faces.ravel(), np.arange(len(faces) * 3).reshape(-1, 3), uv.astype(np.float32), "triangle-grid"


def sample(image, pixels):
    """Image-edge pixels to OpenCV's integer sample-center convention."""
    pixels = np.asarray(pixels, np.float32) - 0.5
    if len(pixels) == 0:
        return np.empty((0, image.shape[2] if image.ndim == 3 else 1), image.dtype)
    # OpenCV remap uses signed 16-bit row indices internally.
    return np.concatenate(
        [
            cv2.remap(
                image,
                chunk[:, 0].reshape(-1, 1),
                chunk[:, 1].reshape(-1, 1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(len(chunk), -1)
            for chunk in (pixels[start : start + 24000] for start in range(0, len(pixels), 24000))
        ]
    )


def visibility(mesh, faces, camera, resolution=768):
    projection = camera.project(mesh)
    scale = np.array([resolution / camera.size[0], resolution / camera.size[1]])
    xyz = np.column_stack((projection[:, :2] * scale, -1 / np.maximum(projection[:, 2], 1e-9)))
    front = faces[(projection[faces, 2] > 0).all(1)]
    # Inverse depth is linear in screen barycentrics; ordinary depth is not.
    inverse, fid = zbuffer(xyz, front, resolution, resolution)
    depth = np.full_like(inverse, np.inf)
    np.divide(-1, inverse, out=depth, where=np.isfinite(inverse) & (inverse < 0))
    valid = fid >= 0
    filled = np.where(valid, depth, 0)
    discontinuity = np.zeros_like(valid)
    for axis in (0, 1):
        difference = np.abs(np.diff(filled, axis=axis)) > 0.006
        if axis == 0:
            discontinuity[:-1] |= difference
            discontinuity[1:] |= difference
        else:
            discontinuity[:, :-1] |= difference
            discontinuity[:, 1:] |= difference
    edge_distance = ndimage.distance_transform_edt(valid & ~discontinuity).astype(np.float32)
    return depth, edge_distance, scale


def project_samples(points, normals, photo, camera, depth, edge_distance, scale):
    p = camera.project(points)
    xy = p[:, :2] * scale
    x, y = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
    inside = (p[:, 2] > 0) & (x >= 0) & (y >= 0) & (x < depth.shape[1]) & (y < depth.shape[0])
    x, y = np.clip(x, 0, depth.shape[1] - 1), np.clip(y, 0, depth.shape[0] - 1)
    visible = inside & np.isfinite(depth[y, x]) & (np.abs(p[:, 2] - depth[y, x]) < 0.0025)
    direction = camera.center - points
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    angle = np.clip(np.einsum("ij,ij->i", normals, direction), 0, 1) ** 3
    weight = visible * angle * np.clip(edge_distance[y, x] / 12, 0, 1) ** 2
    original = camera.to_original(p[:, :2])
    weight *= (
        (original[:, 0] >= 0.5)
        & (original[:, 1] >= 0.5)
        & (original[:, 0] < photo.shape[1] - 0.5)
        & (original[:, 1] < photo.shape[0] - 0.5)
    )
    return sample(photo, original), weight.astype(np.float32)


def multiband(images, weights, levels=5):
    """Laplacian image pyramids with Gaussian confidence weights."""
    totals = None
    normalizers = None
    for image, weight in zip(images, weights, strict=True):
        gaussian, confidence = [image], [weight]
        for _ in range(levels - 1):
            gaussian.append(cv2.pyrDown(gaussian[-1]))
            confidence.append(cv2.pyrDown(confidence[-1]))
        laplacian = [
            gaussian[i] - cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
            for i in range(levels - 1)
        ] + [gaussian[-1]]
        if totals is None:
            totals = [np.zeros_like(x) for x in laplacian]
            normalizers = [np.zeros_like(x) for x in confidence]
        for i in range(levels):
            totals[i] += laplacian[i] * confidence[i][:, :, None]
            normalizers[i] += confidence[i]
    blended = [x / np.maximum(w[:, :, None], 1e-12) for x, w in zip(totals, normalizers, strict=True)]
    result = blended[-1]
    for i in range(levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended[i].shape[1], blended[i].shape[0])) + blended[i]
    return result


def bake(neutral, faces, mapping, atlas_faces, uv, fitted, cameras, photos, symmetry, masks, size, diagnostics=None):
    fid, bary = rasterize_uv(uv * size, atlas_faces, size, size)
    y, x = np.nonzero(fid >= 0)
    return bake_texels(
        neutral, faces, fid, y, x, faces[fid[y, x]], bary[y, x], fitted, cameras, photos, symmetry, masks, diagnostics
    )


def bake_texels(
    neutral, faces, fid, y, x, tri, lam, fitted, cameras, photos, symmetry, masks, diagnostics=None, gate=None,
    view_names=None, view_preferences=None, harmonize_chroma=False,
    view_face_masks=None,
    reject_forehead_hair=False,
):
    """Bake the photos into the texels ``(y, x)`` of a (rows, cols) canvas whose triangle ids are ``fid``.

    Each texel lies in the triangle ``tri`` (vertex ids into ``neutral`` / ``fitted``) at barycentrics ``lam``. The
    canvas may be any rectangle (the hybrid stage bakes only the head island's UV rectangle), ``gate`` (optional,
    one value per texel) scales every view's confidence, and ``diagnostics`` also receives the total ``confidence``.
    """
    shape = fid.shape
    points = np.einsum("ij,ijk->ik", lam, neutral[tri])
    mask_rejections = {}
    images, weights = [], []
    names = list(view_names) if view_names is not None else ["front", "right", "mirrored_right"]
    face_member = np.zeros(len(neutral), bool)
    face_member[masks.get("photo_face_region", masks.get("face", []))] = True
    if not names or names[0] != "front":
        raise ValueError("Front must be the first colour reference view")
    for name in names:
        print(f"[flame] projecting {name}", flush=True)
        view = "right" if name == "mirrored_right" else name
        mesh, camera = fitted[view], cameras[view]
        depth, edge_distance, scale = visibility(mesh, faces, camera)
        normal = welded_vertex_normals(mesh, faces)
        image = np.zeros((*shape, 3), np.float32)
        weight = np.zeros(shape, np.float32)
        rejected = 0
        for start in range(0, len(y), 24000):
            end = start + 24000
            ti = symmetry[tri[start:end]] if name == "mirrored_right" else tri[start:end]
            q = np.einsum("ij,ijk->ik", lam[start:end], mesh[ti])
            n = np.einsum("ij,ijk->ik", lam[start:end], normal[ti])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            color, w = project_samples(q, n, photos[view], camera, depth, edge_distance, scale)
            if view_face_masks is not None:
                photo_xy = camera.to_original(camera.project(q)[:, :2])
                face_weight = sample(view_face_masks[view], photo_xy).ravel()
                # A photographed ear/hair/background can never be evidence for
                # a face texel, even when the camera's local reprojection agrees.
                is_face = face_member[tri[start:end]].any(1)
                rejected += int((is_face & (w > 1e-5) & (face_weight < 0.5)).sum())
                w *= np.where(is_face, face_weight, 1.0)
            if gate is not None:
                w = w * gate[start:end]
            if view_preferences is not None:
                w *= view_preferences[name][start:end]
            if name == "mirrored_right":
                # Symmetry augments only the side opposite the fitted right camera.
                side = np.sign(camera.center[0]) or 1
                w *= np.clip(-points[start:end, 0] * side / 0.012, 0, 1)
            image[y[start:end], x[start:end]] = srgb_to_linear(color.astype(np.float32) / 255)
            weight[y[start:end], x[start:end]] = w
        images.append(image)
        weights.append(weight)
        if view_face_masks is not None:
            mask_rejections[name] = rejected
    from .colour import from_lab, paired_colour, skin_samples, to_lab

    forehead_rejections = {}
    if reject_forehead_hair and len(masks.get("eye_region", [])):
        # The new photo's hairline need not agree with the existing shell. Keep
        # photo hair off exposed forehead skin; the unchanged shell tint supplies
        # the actual scalp. A height guard preserves brows, eyes and beard.
        eye = neutral[masks["eye_region"]].mean(0)
        forehead = (points[:, 1] > eye[1] + 0.025) & (points[:, 2] > eye[2] - 0.025)
        forehead &= np.abs(points[:, 0] - eye[0]) < 0.065
        front_lab = to_lab(linear_to_srgb(images[0][y, x]))
        reference = skin_samples(front_lab) & (weights[0][y, x] > 0.1) & face_member[tri].all(1)
        if reference.sum() >= 32:
            skin_l = float(np.median(front_lab[reference, 0]))
            ramp = np.clip((points[:, 1] - eye[1] - 0.025) / 0.012, 0, 1)
            ramp = ramp * ramp * (3 - 2 * ramp)
            for name, image, weight in zip(names, images, weights, strict=True):
                lab = to_lab(linear_to_srgb(image[y, x]))
                hair = forehead & (lab[:, 0] < skin_l - 12)
                forehead_rejections[name] = int((hair & (weight[y, x] > 1e-5)).sum())
                weight[y, x] *= 1 - ramp * hair

    # Legacy photos retain luminance-only matching. New camera/lighting sources
    # additionally match chroma on overlapping semantic skin, never RGB gains.
    gains = {}
    semantic_overlap = np.zeros(shape, bool)
    semantic_overlap[y, x] = face_member[tri].all(1)
    source_front = images[0][y, x].copy()
    for i in range(1, len(names)):
        overlap = (weights[0] > 0.08) & (weights[i] > 0.08)
        l0, li = images[0].mean(2), images[i].mean(2)
        overlap &= (l0 > 0.05) & (li > 0.05) & (l0 < 0.75) & (li < 0.75)
        if harmonize_chroma:
            overlap &= semantic_overlap
            oy, ox = np.nonzero(overlap)
            skin = skin_samples(to_lab(linear_to_srgb(images[0][oy, ox])))
            skin &= skin_samples(to_lab(linear_to_srgb(images[i][oy, ox])))
            overlap[oy[~skin], ox[~skin]] = False
        if overlap.sum() >= 32:
            a = to_lab(linear_to_srgb(images[0][overlap]))
            b = to_lab(linear_to_srgb(images[i][overlap]))
            offset_l = float(np.clip(np.median(a[:, 0] - b[:, 0]), -15, 15))
            offset_ab = np.clip(np.median(a[:, 1:] - b[:, 1:], axis=0), -20, 20) if harmonize_chroma else np.zeros(2)
        else:
            offset_l = 0.0
            offset_ab = np.zeros(2)
        lab = to_lab(linear_to_srgb(images[i]))
        lab[:, :, 0] += offset_l * np.clip(lab[:, :, 0] / 30, 0, 1) ** 2
        lab[:, :, 1:] += offset_ab
        images[i] = srgb_to_linear(np.clip(from_lab(lab), 0, 1))
        colour_overlap = None
        if overlap.sum() >= 32:
            matched = to_lab(linear_to_srgb(images[i][overlap]))
            difference = a - matched
            colour_overlap = {"before_mean_delta_e76": float(np.linalg.norm(a.mean(0) - b.mean(0))),
                              "after_mean_delta_e76": float(np.linalg.norm(difference.mean(0))),
                              "after_median_delta_e76": float(np.median(np.linalg.norm(difference, axis=1)))}
        gains[names[i]] = {
            "luminance_offset_lab": offset_l,
            "chroma_modified": harmonize_chroma,
            "chroma_offset_lab": offset_ab.tolist(),
            "overlap_texels": int(overlap.sum()),
            "overlap_colour": colour_overlap,
        }
    # Front is the measured identity/white-balance reference where it is visible.
    if view_preferences is None:
        weights[0] *= 4
    total = sum(weights)
    observed = total[y, x] > 1e-5
    if observed.sum() < 32:
        raise ValueError("Cameras produced too few visible photo texels")
    raw = sum(im * w[:, :, None] for im, w in zip(images, weights, strict=True)) / np.maximum(total[:, :, None], 1e-12)
    # Fill each source in surface space before pyramids; UV chart neighbours are
    # not necessarily anatomical neighbours. Never let black margins enter the blend.
    fallback = raw[y, x].copy()
    observed_ids = np.flatnonzero(observed)
    # Fill uses a bounded surface sample; measured texels retain full photo detail.
    observed_ids = observed_ids[:: max(1, len(observed_ids) // 80000)]
    fallback_ids = cKDTree(points[observed_ids]).query(neutral, eps=0.15)[1]
    vertex_fill = raw[y[observed_ids], x[observed_ids]][fallback_ids]
    fallback[~observed] = np.einsum("ij,ijk->ik", lam[~observed], vertex_fill[tri[~observed]])
    eye = np.zeros(len(neutral), bool)
    eye[np.concatenate((masks["left_eyeball"], masks["right_eyeball"]))] = True
    unobserved_eye = eye[tri].all(1) & ~observed
    fallback[unobserved_eye] = srgb_to_linear(np.array([0.80, 0.77, 0.73], np.float32))
    for i in range(len(names)):
        print(f"[flame] surface-space fill {names[i]}", flush=True)
        usable = weights[i][y, x] > 1e-5
        if usable.sum() >= 32:
            usable_ids = np.flatnonzero(usable)
            usable_ids = usable_ids[:: max(1, len(usable_ids) // 80000)]
            nearest = cKDTree(points[usable_ids]).query(neutral, eps=0.15)[1]
            vertex_fill = images[i][y[usable_ids], x[usable_ids]][nearest]
            fill = fallback.copy()
            fill[~usable] = np.einsum("ij,ijk->ik", lam[~usable], vertex_fill[tri[~usable]])
        else:
            fill = fallback
        images[i][y[~usable], x[~usable]] = fill[~usable]
        # Pad charts before downsampling to prevent background colour bleeding.
        nearest = ndimage.distance_transform_edt(fid < 0, return_distances=False, return_indices=True)
        images[i][fid < 0] = images[i][nearest[0][fid < 0], nearest[1][fid < 0]]
    labs = [to_lab(linear_to_srgb(im)) for im in images]
    # Blend shading at multiple scales, while chroma comes only from actual visible
    # photo samples. Occluded fill cannot dilute the measured a/b channels.
    luminance = multiband([np.repeat(lab[:, :, :1], 3, axis=2) for lab in labs], weights)[:, :, 0]
    chroma = sum(lab[:, :, 1:] * w[:, :, None] for lab, w in zip(labs, weights, strict=True)) / np.maximum(
        total[:, :, None], 1e-12
    )
    result = srgb_to_linear(np.clip(from_lab(np.dstack((luminance, chroma))), 0, 1))
    result[y[~observed], x[~observed]] = fallback[~observed]
    # Small positive floor is only a numerical fallback; dark photographed glasses
    # are otherwise preserved. Padding fills the entire canvas for valid mipmaps.
    result = np.clip(result, 0.001, 1)
    nearest = ndimage.distance_transform_edt(fid < 0, return_distances=False, return_indices=True)
    result[fid < 0] = result[nearest[0][fid < 0], nearest[1][fid < 0]]
    texture = np.clip(np.rint(linear_to_srgb(result) * 255), 1, 255).astype(np.uint8)
    semantic_skin = np.zeros(len(neutral), bool)
    semantic_skin[masks.get("face", [])] = True
    for key in ("eye_region", "lips", "left_eyeball", "right_eyeball"):
        semantic_skin[masks.get(key, [])] = False
    reference = np.clip(linear_to_srgb(source_front), 0, 1)
    skin = semantic_skin[tri].all(1) & (weights[0][y, x] > 0.1) & skin_samples(to_lab(reference))
    colour = paired_colour(reference, texture[y, x], skin) if skin.sum() >= 16 else None
    neck = np.zeros(len(neutral), bool)
    neck[masks.get("neck", [])] = True
    chin = neutral[masks["face"], 1].min() if len(masks.get("face", [])) else -np.inf
    underchin = neck[tri].any(1) & (points[:, 1] < chin) & (points[:, 1] > chin - 0.035) & (weights[0][y, x] > 0.1)
    if diagnostics is not None:
        diagnostics.update(y=y, x=x, points=points, reference=reference, skin=skin, confidence=total)
    return texture, {
        "texture_fill_ratio": 1.0,
        "photo_observed_ratio": float(observed.mean()),
        "atlas_surface_texels": len(y),
        "gains": gains,
        "photo_colour": colour,
        "underchin_front_visible_texels": int(underchin.sum()),
        "view_weight_share": {n: float(w.sum() / max(total.sum(), 1e-9)) for n, w in zip(names, weights, strict=True)},
        "face_mask_rejected_texels": mask_rejections,
        "forehead_photo_hair_rejected_texels": forehead_rejections,
    }


def pack_atlases(scan_atlas, flame_atlas, scan_uv, flame_uv):
    """Cap the square atlas at 4096; keep face density before scan body density."""
    max_side = 4096
    face_side = min(max(flame_atlas.shape[:2]), 3072)
    scan_side = min(max(scan_atlas.shape[:2]), max_side - face_side - 16)
    if max(flame_atlas.shape[:2]) != face_side:
        flame_atlas = cv2.resize(flame_atlas, (face_side, face_side), interpolation=cv2.INTER_AREA)
    if max(scan_atlas.shape[:2]) != scan_side:
        scan_atlas = cv2.resize(scan_atlas, (scan_side, scan_side), interpolation=cv2.INTER_AREA)
    sh, sw = scan_atlas.shape[:2]
    fh, fw = flame_atlas.shape[:2]
    side = max(sw + fw + 16, sh, fh)
    atlas = np.empty((side, side, 3), np.uint8)
    atlas[:] = np.median(flame_atlas.reshape(-1, 3), axis=0).astype(np.uint8)
    atlas[:sh, :sw] = scan_atlas
    atlas[:fh, sw + 16 : sw + 16 + fw] = flame_atlas
    # Replicate border pixels in the empty padding for mipmaps.
    atlas[:sh, sw : sw + 8] = scan_atlas[:, -1:]
    atlas[:fh, sw + 8 : sw + 16] = flame_atlas[:, :1]
    suv = scan_uv * [sw / side, sh / side]
    fuv = flame_uv * [fw / side, fh / side] + [(sw + 16) / side, 0]
    return atlas, suv, fuv
