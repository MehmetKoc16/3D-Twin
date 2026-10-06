"""Keep the user's real hair and stubble out of the skin behind the ear and on the neck.

The right profile photo sees the hair behind the ear and the nape hair or stubble, and the baked head texture carries
them into the UV island (the left side gets the same through the mirror fill). Hair belongs to the separate hair shell,
so the photo is only kept for the face (beard, cheeks, jaw, chin, upper lip, brows, eyes), the ears and the front of
the neck under the chin (handled by the neck hand-over). Everything of the head UV island that is

- not in a FLAME face-like region (face, forehead, eye regions, lips, nose) and not in an ear, and
- behind the middle of the ear (the sideburn / temple tint in front of the ear is left exactly as it was)

is replaced by the clean body skin of the same UV texels, with a smooth geodesic blend over ``blend_m`` (13 mm) into
the photographed region, so the jaw-line beard fades out naturally instead of being cut. Ears get a short protective
ramp (``ear_ramp_m``) so their rim keeps the photo.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from flamehead.colour import from_lab, to_lab
from scipy.sparse.csgraph import dijkstra

from .register import smoothstep

# FLAME mask keys whose photo colour is kept (the face with its beard, the brows and eyes, the lips).
FACE_KEYS = ("face", "forehead", "eye_region", "left_eye_region", "right_eye_region", "lips", "nose")
EAR_KEYS = ("left_ear", "right_ear")
EYE_KEYS = ("eye_region", "left_eye_region", "right_eye_region", "left_eyeball", "right_eyeball")
BLEND_M = 0.013
EAR_RAMP_M = 0.004


def flame_vertex_groups(fit, flame, head) -> dict:
    """Per compact head vertex: ``face``, ``ear`` and ``eye`` (eyelid / eyeball regions) membership of the corresponding FLAME vertex."""
    dominant = flame.faces[np.maximum(fit.face_ids, 0)][np.arange(len(fit.face_ids)), fit.face_bary.argmax(1)]
    flame_vertex = dominant[head.welded]
    valid = fit.face_ids[head.welded] >= 0

    def member(keys):
        flag = np.zeros(len(flame.neutral), bool)
        for key in keys:
            if key in flame.masks:
                flag[flame.masks[key]] = True
        return flag[flame_vertex] & valid

    return {"face": member(FACE_KEYS), "ear": member(EAR_KEYS), "eye": member(EYE_KEYS)}


def _welded_distance(positions, faces, welded, sources, limit):
    """Geodesic (edge graph) distance in metres from the compact vertices flagged in ``sources``; ``limit`` caps it."""
    n_w = int(welded.max()) + 1
    wpos = np.zeros((n_w, 3))
    wpos[welded] = positions
    wf = welded[faces]
    edges = np.concatenate((wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]))
    edges = edges[edges[:, 0] != edges[:, 1]]
    length = np.linalg.norm(wpos[edges[:, 0]] - wpos[edges[:, 1]], axis=1)
    graph = sp.coo_matrix((length, (edges[:, 0], edges[:, 1])), shape=(n_w, n_w)).tocsr()
    graph = graph.maximum(graph.T)
    src = np.unique(welded[sources])
    if len(src) == 0:
        return np.full(len(welded), np.inf)
    dist = dijkstra(graph, directed=False, indices=src, min_only=True, limit=limit)
    return dist[welded]


def restriction_weights(
    positions: np.ndarray,
    faces: np.ndarray,
    welded: np.ndarray,
    groups: dict,
    *,
    blend_m: float = BLEND_M,
    ear_ramp_m: float = EAR_RAMP_M,
) -> dict:
    """Per compact head vertex weight in 0..1 of the clean-skin replacement (1 = body skin, 0 = photo)."""
    face, ear = groups["face"], groups["ear"]
    x, z = positions[:, 0], positions[:, 2]
    z_mid = {}
    for side in (1, -1):
        sel = ear & (np.sign(x) == side)
        z_mid[side] = float(np.median(z[sel])) if sel.any() else None
    fallback = next((v for v in z_mid.values() if v is not None), float(z.max()))
    z_ref = np.where(x >= 0, z_mid[1] if z_mid[1] is not None else fallback,
                     z_mid[-1] if z_mid[-1] is not None else fallback)
    core = ~face & ~ear & (z < z_ref)
    # fade out of the core over blend_m on the photo side: the core itself is full strength
    d_core = _welded_distance(positions, faces, welded, core, blend_m * 1.2)
    outside = 1.0 - smoothstep(d_core / blend_m)
    w = np.where(core, 1.0, np.where(ear, 0.0, outside))
    # keep the ear rim: ramp from 0 on the ear to 1 at ear_ramp_m away
    d_ear = _welded_distance(positions, faces, welded, ear, ear_ramp_m * 1.2)
    w = w * np.where(ear, 0.0, smoothstep(d_ear / ear_ramp_m))
    return {"weight": w, "core": core, "z_ear_mid": z_mid}


def texel_weights(vertex_weight: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Interpolate the per-vertex weight to texels (``tri`` compact vertex ids per texel, ``bary`` barycentrics)."""
    return np.einsum("ij,ij->i", bary, vertex_weight[tri])


def restrict_photo(texture: np.ndarray, ty, tx, weight: np.ndarray, skin_rgb: np.ndarray) -> np.ndarray:
    """Mix the photographed texels toward ``skin_rgb`` (the clean body skin of the same texels, uint8) in Lab."""
    out = texture.copy()
    lab = to_lab(texture[ty, tx])
    clean = to_lab(skin_rgb)
    mixed = lab * (1 - weight[:, None]) + clean * weight[:, None]
    out[ty, tx] = np.clip(np.rint(from_lab(mixed) * 255), 1, 255).astype(np.uint8)
    return out


def _local_contrast(texture: np.ndarray, ty, tx, selection: np.ndarray, radius: int = 2) -> np.ndarray:
    """Std of L* in a small window around the selected texels (strand-like structure)."""
    from scipy import ndimage

    lab = to_lab(texture)[..., 0].astype(np.float32)
    mean = ndimage.uniform_filter(lab, 2 * radius + 1)
    sq = ndimage.uniform_filter(lab * lab, 2 * radius + 1)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return std[ty[selection], tx[selection]]


def region_stats(texture: np.ndarray, ty, tx, selection: np.ndarray, tone_lab, *, dark_l: float = 40.0) -> dict:
    """Numbers of a texel set: dark hair-like texels, L* offset against the body skin, local contrast, Lab mean / std."""
    if not selection.any():
        return {"texels": 0}
    lab = to_lab(texture[ty[selection], tx[selection]]).astype(np.float64)
    contrast = _local_contrast(texture, ty, tx, selection)
    tone = np.asarray(tone_lab, np.float64)
    return {
        "texels": int(selection.sum()),
        f"dark_texels_l_lt_{int(dark_l)}": int((lab[:, 0] < dark_l).sum()),
        "strand_texels_dl_lt_minus15": int((lab[:, 0] < tone[0] - 15).sum()),
        "high_contrast_texels_std_gt_6": int((contrast > 6.0).sum()),
        "delta_l_vs_body": float(lab[:, 0].mean() - tone[0]),
        "mean_lab": lab.mean(0).tolist(),
        "std_lab": lab.std(0).tolist(),
        "p05_l": float(np.percentile(lab[:, 0], 5)),
    }
