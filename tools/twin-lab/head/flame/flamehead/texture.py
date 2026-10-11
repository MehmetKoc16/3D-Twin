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


def hair_vertex_regions(neutral, masks):
    """Derive skin reference and beard sub-regions in native FLAME coordinates.

    Official FLAME masks have no cheek/jaw labels. The nose base separates the
    upper cheeks from the jaw/chin/upper-lip allow-mask; eye/lip masks are always
    protected. Derive these before transferring correspondence to another mesh.
    """
    def member(*keys):
        selected = np.zeros(len(neutral), bool)
        for key in keys:
            selected[masks.get(key, [])] = True
        return selected

    face = member("face", "forehead")
    eyes = member("eye_region", "left_eye_region", "right_eye_region", "left_eyeball", "right_eyeball")
    lips, nose, forehead = member("lips"), member("nose"), member("forehead")
    beard = member("jaw", "chin", "upper_lip", "beard_allow")
    reference = member("cheeks", "skin_reference")
    temple = np.zeros(len(neutral), bool)
    if face.any() and eyes.any() and nose.any():
        centre = float(np.median(neutral[face, 0]))
        width = float(np.max(np.abs(neutral[face, 0] - centre)))
        eye_y = float(np.median(neutral[eyes, 1]))
        nose_y = float(np.min(neutral[nose, 1]))
        dx = np.abs(neutral[:, 0] - centre)
        # Include marginal triangles through any-member interpolation below.
        beard |= face & (neutral[:, 1] <= nose_y)
        reference |= face & (dx > width * 0.32) & (dx < width * 0.92)
        reference &= (neutral[:, 1] > nose_y) & (neutral[:, 1] < eye_y)
        reference |= forehead & (dx < width * 0.4)
        temple = face & (dx > width * 0.65) & (neutral[:, 1] > eye_y - width * 0.15)
    beard |= lips
    reference &= ~eyes & ~nose & ~lips & ~beard
    eligible = (face | member("left_ear", "right_ear")) & ~eyes & ~nose & ~lips & ~beard
    return {"skin_reference": np.flatnonzero(reference), "beard_allow": np.flatnonzero(beard),
            "hair_skin_region": np.flatnonzero(eligible), "temple": np.flatnonzero(temple)}


FRINGE_RADIUS = 0.0015  # metres: anti-aliased hair edge to drop around rejected ear/temple samples
EAR_HAIR_SHARE = 0.12  # an ear is repaired only if at least this share of what a view sees of it is hair-like
EAR_MIN_SEEN = 500  # texels a view must see of an ear (confidence > 0.08) before its share counts


def skin_colour_model(lab, reference, weight):
    """Robust per-photo a*/b* model; L* is deliberately not a rejection axis."""
    from .colour import skin_samples

    threshold = 0.08
    selected = reference & skin_samples(lab) & (weight > threshold)
    if selected.sum() < 32:
        return None
    colours = lab[selected]
    centre = np.median(colours, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(colours[:, 1:] - centre[1:]), axis=0), [2.5, 3.0])
    return {"median_lab": centre.tolist(), "mad_ab": spread.tolist(), "texels": int(selected.sum()),
            "confidence_threshold": threshold}


def hair_like_samples(lab, model):
    """Reject chroma outliers, including moderately lit grey/brown hair.

    Shadow with the skin hue is retained: luminance alone cannot imply hair, and
    bright desaturated skin (a specular highlight) is not hair either. A sample
    must be far from the robust model AND lose chroma or skin hue; only samples
    far darker than lit skin are rejected on darkness alone (hue is meaningless
    there, and black hair carries a faint warm cast).
    """
    if model is None:
        return np.zeros(len(lab), bool)
    centre = np.asarray(model["median_lab"])[1:]
    ab = lab[:, 1:]
    chroma = np.linalg.norm(ab, axis=1)
    reference_chroma = np.linalg.norm(centre)
    cosine = (ab @ centre) / np.maximum(chroma * reference_chroma, 1e-9)
    distance = np.linalg.norm((ab - centre) / np.asarray(model["mad_ab"]), axis=1)
    # Skin chroma also falls in shadow. Compare with the expected dimmed chroma,
    # rather than rejecting a warm shadow for being less saturated than a cheek.
    shadow = np.clip(lab[:, 0] / max(model["median_lab"][0], 1), 0.25, 1)
    chroma_outlier = (distance > 2.5) & ((chroma < reference_chroma * 0.5 * shadow) | (cosine < 0.8))
    return chroma_outlier | (lab[:, 0] < 0.3 * model["median_lab"][0])


def temple_dark_samples(lab, model, margin=22.0):
    """Hairline stubble on the exposed temple: far darker than this photo's lit skin, whatever its hue.

    The temple is convex and lit, so (unlike the cheek hollow or the eye socket) skin there is not in deep
    shadow. Only applied to the FLAME temple band; the forehead has the same rule in ``reject_forehead_hair``.
    """
    if model is None:
        return np.zeros(len(lab), bool)
    return lab[:, 0] < model["median_lab"][0] - margin


def mirrored_ear_colour(points, mirrored_points, raw, own, opposite, missing, sigma=0.0012, neighbours=48):
    """Colours for the ``missing`` ear texels from the ``opposite`` ear around their mirror positions.

    A topology mirror is only accurate to a millimetre or two, so the opposite ear's texels are averaged under a
    Gaussian (``sigma`` metres): a mirrored fold line would otherwise show as a doubled dark stroke. The opposite
    photo's white balance differs from this side's, so the median skin tone of this ear's own surviving texels
    (``own``, when there are enough) replaces the opposite ear's median. Returns the missing texel ids, a flag for
    those with a mirror partner within 8 mm, and their linear colours.
    """
    from .colour import from_lab, skin_samples, to_lab

    ids = np.flatnonzero(opposite)
    want = np.flatnonzero(missing)
    distance, nearest = cKDTree(points[ids]).query(
        mirrored_points[want], k=min(neighbours, len(ids)), distance_upper_bound=0.008,
    )
    distance, nearest = distance.reshape(len(want), -1), nearest.reshape(len(want), -1)
    found = np.isfinite(distance)
    source = to_lab(linear_to_srgb(raw[ids]))
    weight = np.exp(-0.5 * (np.where(found, distance, 1.0) / sigma) ** 2) * found
    mixed = (weight[:, :, None] * source[np.minimum(nearest, len(ids) - 1)]).sum(1)
    mixed /= np.maximum(weight.sum(1, keepdims=True), 1e-300)
    kept = to_lab(linear_to_srgb(raw[own]))
    kept, theirs = kept[skin_samples(kept)], source[skin_samples(source)]
    if len(kept) >= 32 and len(theirs) >= 32:
        mixed += np.clip(np.median(kept, axis=0) - np.median(theirs, axis=0), [-6, -8, -8], [6, 8, 8])
    return want, found[:, 0], srgb_to_linear(np.clip(from_lab(mixed), 0, 1))


def ear_fallback(points, mirrored_points, raw, observed, ears, labs, accepted, models, lost=None, rims=None):
    """Ear texels whose photo evidence was all rejected: clean opposite ear, then ear skin tone plus L* detail.

    ``raw`` already combines the other accepted views, so they have priority. ``lost`` marks the texels that were
    photographed but rejected everywhere (default: every unobserved texel); ears never photographed keep the legacy
    surface fill. ``rims`` (side -> flags) are rim triangles that straddle the ear mask: they may be mirrored
    from the opposite ear too, but never take the ear tone. Mirroring uses topology correspondence, never UV
    adjacency or face samples.
    """
    from .colour import from_lab, skin_samples

    fallback = raw.copy()
    mirrored = np.zeros(len(points), bool)
    toned = np.zeros(len(points), bool)
    lost = ~observed if lost is None else lost & ~observed
    for side, ear in ears.items():
        missing = (ear | (rims or {}).get(side, False)) & lost
        opposite = ears["right" if side == "left" else "left"] & observed
        if opposite.any() and missing.any():
            want, matched, colour = mirrored_ear_colour(points, mirrored_points, raw, ear & observed, opposite, missing)
            fallback[want[matched]] = colour[matched]
            mirrored[want[matched]] = True
        remaining = missing & ~mirrored & ear
        if not remaining.any():
            continue
        skin = []
        for lab, weight in zip(labs, accepted, strict=True):
            keep = ear & (weight > 1e-5) & skin_samples(lab)
            skin.extend(lab[keep])
        if skin:
            tone = np.median(np.asarray(skin), axis=0)
        else:
            tones = [m["median_lab"] for m in models.values() if m is not None]
            if not tones:
                continue  # Insufficient evidence: retain the legacy surface fill.
            tone = np.median(tones, axis=0)
        # Keep only fine luminance variation, not the hair's broad dark patch.
        ids = np.flatnonzero(ear)
        source_l = np.mean([lab[ear, 0] for lab in labs], axis=0)
        tree = cKDTree(points[ids])
        _, neighbours = tree.query(points[ids], k=min(16, len(ids)))
        if neighbours.ndim == 1:
            neighbours = neighbours[:, None]
        detail = np.clip(source_l - np.median(source_l[neighbours], axis=1), -3, 3)
        colour = np.tile(tone, (len(ids), 1))
        colour[:, 0] += detail
        fill = srgb_to_linear(from_lab(colour))
        fallback[ids[remaining[ids]]] = fill[remaining[ids]]
        toned[remaining] = True
    return fallback, mirrored, toned


LOCAL_LOW_WEIGHT = 0.12  # every contributing view weight below this: the texel rests on weak photo evidence
LOCAL_CONFIDENT_WEIGHT = 0.15  # total view weight of a texel that may serve as the neighbouring confident skin
LOCAL_DISAGREE_AB = 4.0  # a*b* distance between two views' chroma that counts as a cross-view mismatch
LOCAL_DEVIATION_AB = 3.5  # chroma distance from the confident neighbourhood that makes a weak texel a mismatch
OUTLIER_AB = 7.0  # chroma distance from the local median that makes a texel a colour outlier
OUTLIER_MAX_TEXELS = 800  # a cluster of outliers larger than this is a real feature (mole, brow, lip), not a speck
OUTLIER_L_CLAMP = 5.0  # an outlier's lightness is pulled to within this of its neighbourhood median
LOCAL_RADIUS = 0.008  # metres: confident-skin neighbourhood of a weak texel
OUTLIER_RADIUS = 0.006  # metres: neighbourhood for the outlier median
OUTLIER_LINK = 0.0015  # metres: outlier texels closer than this form one cluster
OUTLIER_MAX_EXTENT = 0.006  # metres: longest box diagonal of a speck (a feature's boundary ring is longer)
NEIGHBOUR_VOXEL = 0.0006  # metres: texels are averaged per voxel so a median covers millimetres, not a texel cluster


class _VoxelTable:
    """Texel colours averaged per ``NEIGHBOUR_VOXEL`` cell: neighbourhood medians then span a physical radius whatever
    the local texel density (a 2 mm speck can cover most of the nearest few hundred texels, but only a few voxels)."""

    def __init__(self, points, values):
        key = np.floor(points / NEIGHBOUR_VOXEL).astype(np.int64)
        _, self.inverse = np.unique(key, axis=0, return_inverse=True)
        self.inverse = self.inverse.ravel()
        count = np.bincount(self.inverse).astype(np.float64)
        self.centroids = np.stack([np.bincount(self.inverse, points[:, c]) for c in range(3)], axis=1) / count[:, None]
        self.means = np.stack([np.bincount(self.inverse, values[:, c]) for c in range(values.shape[1])], axis=1)
        self.means = (self.means / count[:, None]).astype(np.float32)
        self.tree = cKDTree(self.centroids)

    def median(self, query, radius, k=160, minimum=8):
        """Median of the voxel colours within ``radius`` of every query point (NaN where fewer than ``minimum``)."""
        result = np.full((len(query), self.means.shape[1]), np.nan, np.float32)
        for start in range(0, len(query), 4000):
            distance, index = self.tree.query(query[start : start + 4000], k=k, distance_upper_bound=radius)
            found = np.isfinite(distance)
            gathered = self.means[np.minimum(index, len(self.means) - 1)]
            gathered[~found] = np.nan
            enough = found.sum(1) >= minimum
            median = np.full((len(distance), self.means.shape[1]), np.nan, np.float32)
            if enough.any():
                median[enough] = np.nanmedian(gathered[enough], axis=1)
            result[start : start + 4000] = median
        return result


def _small_clusters(points, link, max_texels, max_extent):
    """Mask of the points in connected clusters (closer than ``link``) that are compact specks, and their count.

    The boundary ring of a large feature can also be a small cluster, but it is long and thin: a speck must fit in a
    ``max_extent`` (metres) box as well.
    """
    from scipy.sparse.csgraph import connected_components

    if len(points) == 0:
        return np.zeros(0, bool), 0
    tree = cKDTree(points)
    count, labels = connected_components(tree.sparse_distance_matrix(tree, link, output_type="coo_matrix"), directed=False)
    sizes = np.bincount(labels, minlength=count)
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    np.minimum.at(low, labels, points)
    np.maximum.at(high, labels, points)
    small = (sizes <= max_texels) & (np.linalg.norm(high - low, axis=1) <= max_extent)
    return small[labels], int(small.sum())


def local_chroma_repair(points, lab, view_weights, view_ab, region):
    """Local, conservative repair of two small colour artefacts of the face bake; the global blend is untouched.

    1. Cross-view chroma mismatch on weak evidence. Where EVERY contributing view weight is low and the views
       disagree in chroma, the blended a*/b* is a mix of mutually inconsistent colours (a pale grey smudge). If that
       blend also deviates from the neighbouring confident skin, the chroma comes from the dominant (most confident)
       view when that view agrees with the neighbourhood, otherwise from the neighbouring confident skin. L* keeps
       its detail.
    2. Small chroma outliers (an isolated saturated speck): compact clusters (at most ``OUTLIER_MAX_TEXELS`` texels in
       a ``OUTLIER_MAX_EXTENT`` box) whose a*/b* is far from the local median and that do not sit on the edge of a
       larger area of their own colour are painted with the median chroma of the non-outlier neighbours, and their
       L* is pulled to within ``OUTLIER_L_CLAMP`` of the local median.

    ``lab`` is (n, 3) for the texels, ``view_weights`` (v, n), ``view_ab`` (v, n, 2), ``region`` (n,) bool: texels
    allowed to change. Returns the new Lab and a statistics dict.
    """
    new = lab.copy()
    stats = {"weak_view_mismatch": 0, "outlier_specks": 0, "outlier_texels": 0}
    total = view_weights.sum(0)
    active = view_weights > 1e-5
    top = view_weights.max(0)
    disagree = np.zeros(len(lab), np.float32)
    for i in range(len(view_weights)):
        for j in range(i + 1, len(view_weights)):
            both = active[i] & active[j]
            disagree = np.where(both, np.maximum(disagree, np.linalg.norm(view_ab[i] - view_ab[j], axis=1)), disagree)
    weak = region & (active.sum(0) >= 2) & (top < LOCAL_LOW_WEIGHT) & (disagree > LOCAL_DISAGREE_AB)
    confident = region & (total >= LOCAL_CONFIDENT_WEIGHT)
    if weak.any() and confident.sum() >= 32:
        ids = np.flatnonzero(weak)
        reference = _VoxelTable(points[confident], lab[confident][:, 1:]).median(points[ids], LOCAL_RADIUS, 96)
        deviation = np.linalg.norm(lab[ids, 1:] - reference, axis=1)
        bad = np.isfinite(deviation) & (deviation > LOCAL_DEVIATION_AB)
        ids, reference = ids[bad], reference[bad]
        if len(ids):
            dominant = view_ab[view_weights[:, ids].argmax(0), ids]
            use_dominant = np.linalg.norm(dominant - reference, axis=1) <= LOCAL_DEVIATION_AB
            new[ids, 1:] = np.where(use_dominant[:, None], dominant, reference)
            stats["weak_view_mismatch"] = len(ids)
    if region.sum() >= 32:
        region_ids = np.flatnonzero(region)
        table = _VoxelTable(points[region_ids], new[region_ids])
        median = table.median(table.centroids, OUTLIER_RADIUS)[table.inverse]
        outlier = np.isfinite(median[:, 0]) & (np.linalg.norm(new[region_ids, 1:] - median[:, 1:], axis=1) > OUTLIER_AB)
        ids = region_ids[outlier]
        if len(ids):
            ids = ids[_small_clusters(points[ids], OUTLIER_LINK, OUTLIER_MAX_TEXELS, OUTLIER_MAX_EXTENT)[0]]
        if len(ids):
            # A speck stands alone. A texel whose surroundings (outside the outlier set) share its colour sits on the
            # edge of a larger feature: that is a boundary corner, not a speck.
            flagged = np.zeros(len(lab), bool)
            flagged[ids] = True
            alone = np.ones(len(ids), bool)
            for n, neighbours in enumerate(cKDTree(points[region_ids]).query_ball_point(points[ids], OUTLIER_LINK * 2)):
                others = region_ids[neighbours]
                others = others[~flagged[others]]
                if len(others):
                    similar = np.linalg.norm(new[others, 1:] - new[ids[n], 1:], axis=1) < OUTLIER_AB * 0.6
                    alone[n] = similar.mean() < 0.08
            ids = ids[alone]
        if len(ids):
            flagged = np.zeros(len(lab), bool)
            flagged[ids] = True
            clean = region & ~flagged
            reference = _VoxelTable(points[clean], new[clean]).median(points[ids], OUTLIER_RADIUS, minimum=4)
            ok = np.isfinite(reference[:, 0])
            ids, reference = ids[ok], reference[ok]
            new[ids, 1:] = reference[:, 1:]
            new[ids, 0] = np.clip(new[ids, 0], reference[:, 0] - OUTLIER_L_CLAMP, reference[:, 0] + OUTLIER_L_CLAMP)
            stats["outlier_texels"] = len(ids)
            if len(ids):
                stats["outlier_specks"] = _small_clusters(points[ids], OUTLIER_LINK, len(ids), np.inf)[1]
    return new, stats


def bake(neutral, faces, mapping, atlas_faces, uv, fitted, cameras, photos, symmetry, masks, size, diagnostics=None,
         **options):
    fid, bary = rasterize_uv(uv * size, atlas_faces, size, size)
    y, x = np.nonzero(fid >= 0)
    return bake_texels(
        neutral, faces, fid, y, x, faces[fid[y, x]], bary[y, x], fitted, cameras, photos, symmetry, masks, diagnostics,
        **options,
    )


def bake_texels(
    neutral, faces, fid, y, x, tri, lam, fitted, cameras, photos, symmetry, masks, diagnostics=None, gate=None,
    view_names=None, view_preferences=None, harmonize_chroma=False,
    view_face_masks=None,
    reject_forehead_hair=False,
    reject_photo_hair=False,
    local_chroma_fix=False,
):
    """Bake the photos into the texels ``(y, x)`` of a (rows, cols) canvas whose triangle ids are ``fid``.

    Each texel lies in the triangle ``tri`` (vertex ids into ``neutral`` / ``fitted``) at barycentrics ``lam``. The
    canvas may be any rectangle (the hybrid stage bakes only the head island's UV rectangle), ``gate`` (optional,
    one value per texel) scales every view's confidence, and ``diagnostics`` also receives the total ``confidence``.
    """
    shape = fid.shape
    points = np.einsum("ij,ijk->ik", lam, neutral[tri])
    beard_reference = None
    if reject_photo_hair:
        # Pyramid lighting has a broad footprint: exempting beard weights alone
        # cannot protect stubble from nearby ear/temple edits. Retain the legacy
        # bake on this explicit allow-mask, including its original fine detail.
        regions = hair_vertex_regions(neutral, masks) if "beard_allow" not in masks else masks
        member = np.zeros(len(neutral), bool)
        member[regions["beard_allow"]] = True
        protected = member[tri].any(1)
        if protected.any():
            legacy, _ = bake_texels(
                neutral, faces, fid, y, x, tri, lam, fitted, cameras, photos, symmetry, masks,
                gate=gate, view_names=view_names, view_preferences=view_preferences,
                harmonize_chroma=harmonize_chroma, view_face_masks=view_face_masks,
                reject_forehead_hair=reject_forehead_hair,
            )
            beard_reference = legacy[y[protected], x[protected]].copy()
            del legacy
    mask_rejections = {}
    images, weights, model_weights = [], [], []
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
        model_weight = np.zeros(shape, np.float32) if reject_photo_hair else None
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
            if model_weight is not None:
                # Registration distance and anatomical blending preferences are
                # not photo visibility. A low gate can suppress a valid cheek
                # colour even though it should not contribute texture detail.
                model_weight[y[start:end], x[start:end]] = w
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
        if model_weight is not None:
            model_weights.append(model_weight)
        if view_face_masks is not None:
            mask_rejections[name] = rejected
    from .colour import from_lab, paired_colour, skin_samples, to_lab

    projection_weights = [w[y, x].copy() for w in weights] if reject_photo_hair else None
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
    # Overlap colour matching keeps the pre-photo-hair weights (forehead rejection included, as in the legacy bake).
    colour_weights = [w.copy() for w in weights] if reject_photo_hair else weights

    # Preserve the existing overlap correction and front colour reference. Hair
    # rejection must not change white balance on the protected beard.
    source_labs = [to_lab(linear_to_srgb(im[y, x])) for im in images]
    hair_rejections, skin_models, rejected_samples = {}, {}, {}
    ear_members = {}
    beard_allow = np.zeros(len(y), bool)
    if reject_photo_hair:
        regions = hair_vertex_regions(neutral, masks) if "hair_skin_region" not in masks else masks

        def texel_member(key, any_vertex=False):
            member = np.zeros(len(neutral), bool)
            member[regions.get(key, masks.get(key, []))] = True
            return member[tri].any(1) if any_vertex else member[tri].all(1)

        beard_allow = texel_member("beard_allow", True)
        eligible = texel_member("hair_skin_region", True) & ~beard_allow
        for key in ("eye_region", "left_eye_region", "right_eye_region", "nose", "lips",
                    "left_eyeball", "right_eyeball"):
            eligible &= ~texel_member(key, True)
        reference_skin = texel_member("skin_reference") & ~beard_allow
        temple = texel_member("temple", True)
        ear_members = {side: texel_member(f"{side}_ear") for side in ("left", "right")}
        ear_rims = {side: texel_member(f"{side}_ear", True) & eligible & ~ear_members[side] for side in ear_members}
        models, rejects = {}, {}
        for name, lab, model_weight in zip(names, source_labs, model_weights, strict=True):
            models[name] = skin_colour_model(lab, reference_skin, model_weight[y, x])
            rejects[name] = eligible & (
                hair_like_samples(lab, models[name]) | (temple & temple_dark_samples(lab, models[name]))
            )
        # An ear is repaired only when a view that sees it well finds a substantial share of it hair-like. A clean ear
        # (a few percent) has only fold shadows and warm creases that a cheek-based skin model calls outliers, and
        # repainting those from the other ear would leave visible patches.
        ear_hair_share, dirty = {}, np.zeros(len(y), bool)
        for side in ear_members:
            ear_all = ear_members[side] | ear_rims[side]
            shares = []
            for name, weight in zip(names, weights, strict=True):
                seen = ear_all & (weight[y, x] > 0.08)
                if seen.sum() >= EAR_MIN_SEEN:
                    shares.append(float((rejects[name] & seen).sum() / seen.sum()))
            ear_hair_share[side] = max(shares, default=None)
            if ear_hair_share[side] is not None and ear_hair_share[side] < EAR_HAIR_SHARE:
                for reject in rejects.values():
                    reject &= ~ear_all
            else:
                dirty |= ear_all
        fringe_region = eligible & (temple | dirty)
        for name, weight in zip(names, weights, strict=True):
            skin_models[name], reject = models[name], rejects[name]
            # A hair edge is anti-aliased with skin: grow the rejection a little on the ears and temple, where it
            # would otherwise leave a thin dark fringe along the repaired skin. The rest of the face is unchanged.
            grow = np.flatnonzero(fringe_region & ~reject & (weight[y, x] > 1e-5))
            if reject.any() and len(grow):
                near, _ = cKDTree(points[reject]).query(points[grow], distance_upper_bound=FRINGE_RADIUS)
                reject[grow[np.isfinite(near)]] = True
            rejected_samples[name] = reject
            hair_rejections[name] = int((reject & (weight[y, x] > 1e-5)).sum())
            weight[y, x] *= ~reject

    # Legacy photos retain luminance-only matching. New camera/lighting sources
    # additionally match chroma on overlapping semantic skin, never RGB gains.
    gains = {}
    semantic_overlap = np.zeros(shape, bool)
    semantic_overlap[y, x] = face_member[tri].all(1)
    source_front = images[0][y, x].copy()
    for i in range(1, len(names)):
        overlap = (colour_weights[0] > 0.08) & (colour_weights[i] > 0.08)
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
    mirrored_ear = toned_ear = np.zeros(len(y), bool)
    repaired_ear = np.zeros(len(y), bool)
    if reject_photo_hair:
        photographed = np.any([w[y, x] > 1e-5 for w in colour_weights], axis=0)
        mirrored_points = np.einsum("ij,ijk->ik", lam, neutral[symmetry[tri]])
        matched_labs = [to_lab(linear_to_srgb(im[y, x])) for im in images]
        # Tone evidence is in the same corrected colour space as the baked atlas.
        matched_models = {name: skin_colour_model(lab, reference_skin, model_weights[i][y, x])
                          for i, (name, lab) in enumerate(zip(names, matched_labs, strict=True))}
        ear_fill, mirrored_ear, toned_ear = ear_fallback(
            points, mirrored_points, raw[y, x], observed, ear_members, matched_labs,
            [w[y, x] for w in weights], matched_models, lost=photographed & ~observed, rims=ear_rims,
        )
        for before, after in zip(colour_weights, weights, strict=True):
            repaired_ear |= (before[y, x] - after[y, x]) > 1e-5
        repaired_ear &= ear_members["left"] | ear_members["right"] | ear_rims["left"] | ear_rims["right"]
        repaired_ear |= mirrored_ear | toned_ear
        fallback[repaired_ear] = ear_fill[repaired_ear]
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
        # Rejected ear samples must not leak back through the luminance pyramid.
        missing_ear = repaired_ear & ~usable
        images[i][y[missing_ear], x[missing_ear]] = fallback[missing_ear]
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
    result[y[repaired_ear], x[repaired_ear]] = fallback[repaired_ear]
    local_fix = None
    if reject_photo_hair and local_chroma_fix:
        # Local repair of weak cross-view chroma mismatches and tiny colour specks (never the beard allow-mask, the
        # nose, eyes and lips, and never the blend of confident texels; clusters above OUTLIER_MAX_TEXELS are kept).
        keep = ~beard_allow & ~ear_members["left"] & ~ear_members["right"] & eligible & ~texel_member("nose", True)
        texel_lab = to_lab(linear_to_srgb(result[y, x]))
        view_w = np.stack([w[y, x] for w in weights])
        view_ab = np.stack([lab[y, x, 1:] for lab in labs])
        fixed, local_fix = local_chroma_repair(points, texel_lab, view_w, view_ab, keep)
        changed = np.flatnonzero(np.abs(fixed - texel_lab).max(1) > 1e-4)
        result[y[changed], x[changed]] = srgb_to_linear(np.clip(from_lab(fixed[changed]), 0, 1))
        local_fix["changed_texels"] = len(changed)
        local_fix["changed_centroid"] = points[changed].mean(0).tolist() if len(changed) else None
    # Small positive floor is only a numerical fallback; dark photographed glasses
    # are otherwise preserved. Padding fills the entire canvas for valid mipmaps.
    result = np.clip(result, 0.001, 1)
    nearest = ndimage.distance_transform_edt(fid < 0, return_distances=False, return_indices=True)
    result[fid < 0] = result[nearest[0][fid < 0], nearest[1][fid < 0]]
    texture = np.clip(np.rint(linear_to_srgb(result) * 255), 1, 255).astype(np.uint8)
    if beard_reference is not None:
        texture[y[protected], x[protected]] = beard_reference
        texture[fid < 0] = texture[nearest[0][fid < 0], nearest[1][fid < 0]]
    semantic_skin = np.zeros(len(neutral), bool)
    semantic_skin[masks.get("face", [])] = True
    for key in ("eye_region", "lips", "left_eyeball", "right_eyeball"):
        semantic_skin[masks.get(key, [])] = False
    reference = np.clip(linear_to_srgb(source_front), 0, 1)
    skin = semantic_skin[tri].all(1) & (colour_weights[0][y, x] > 0.1) & skin_samples(to_lab(reference))
    colour = paired_colour(reference, texture[y, x], skin) if skin.sum() >= 16 else None
    neck = np.zeros(len(neutral), bool)
    neck[masks.get("neck", [])] = True
    chin = neutral[masks["face"], 1].min() if len(masks.get("face", [])) else -np.inf
    underchin = neck[tri].any(1) & (points[:, 1] < chin) & (points[:, 1] > chin - 0.035) & (weights[0][y, x] > 0.1)
    if diagnostics is not None:
        confidence = total.copy()
        diagnostics.update(y=y, x=x, points=points, reference=reference, skin=skin, confidence=confidence)
        if reject_photo_hair:
            # Resolved ear evidence should survive hybrid's unobserved-skin fade.
            confidence[y[mirrored_ear | toned_ear], x[mirrored_ear | toned_ear]] = 0.12
            diagnostics.update(hair={"models": skin_models, "rejected_texels": hair_rejections,
                                    "ear_hair_share": ear_hair_share, "local_chroma_fix": local_fix,
                                    "ear_fallback": {"other_views": int((repaired_ear & observed).sum()),
                                                     "mirrored": int(mirrored_ear.sum()),
                                                     "skin_tone_detail": int(toned_ear.sum())}},
                               source_labs=dict(zip(names, source_labs, strict=True)),
                               source_weights=dict(zip(names, projection_weights, strict=True)),
                               accepted_weights={n: w[y, x] for n, w in zip(names, weights, strict=True)},
                               ear_members=ear_members, beard_allow=beard_allow,
                               rejected_samples=rejected_samples,
                               temple=texel_member("temple"))
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
        "photo_hair_rejected_texels": hair_rejections,
        "photo_skin_models": skin_models,
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
