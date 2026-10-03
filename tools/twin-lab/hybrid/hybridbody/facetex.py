"""Bake the two fitted photos into the template head's fixed UV island with the exact FLAME cameras.

Every template head vertex corresponds to one point on the neutral FLAME surface (closest point after the
registration). Posing that point with each view's fitted FLAME mesh gives the vertex's position in the view's
camera frame, so the template head is "posed" like FLAME in each photo and the proven FLAME baker (z-buffer
visibility, view-angle weights, luminance matching, multiband blending, mirror for the unseen side) runs unchanged
on the template's own triangles and UV texels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from flamehead.camera import load_cameras
from flamehead.colour import from_lab, to_lab
from flamehead.texture import bake_texels
from PIL import Image
from scipy import ndimage
from twintex.raster import rasterize_uv

from .headfit import FlameFit, HeadFit, Template
from .register import smoothstep


@dataclass
class HeadMesh:
    """The render vertices and faces of the head region in compact numbering."""

    ids: np.ndarray  # (n,) render vertex ids
    faces: np.ndarray  # (m, 3) compact
    face_island: np.ndarray  # (m,)
    welded: np.ndarray  # (n,) welded id
    symmetry: np.ndarray  # (n,) compact id of the x-mirrored vertex (itself where there is none)
    compact_of_render: np.ndarray  # (nr,) compact id or -1


def build_head_mesh(template: Template) -> HeadMesh:
    free_render = template.free[template.inverse]
    ids = np.flatnonzero(free_render)
    compact = np.full(len(template.positions), -1, np.int64)
    compact[ids] = np.arange(len(ids))
    keep = free_render[template.faces].all(1)
    faces = compact[template.faces[keep]]
    face_island = template.islands[template.faces[keep][:, 0]]
    welded = template.inverse[ids]
    representative = np.full(len(template.base), -1, np.int64)
    representative[welded[::-1]] = np.arange(len(ids))[::-1]  # first compact vertex of every welded id
    mirror = template.mirror[welded]
    symmetry = np.where(mirror >= 0, representative[np.maximum(mirror, 0)], -1)
    symmetry = np.where(symmetry >= 0, symmetry, np.arange(len(ids)))
    return HeadMesh(ids, faces, face_island, welded, symmetry, compact)


def posed_vertices(
    template: Template,
    fit: HeadFit,
    flame: FlameFit,
    view: np.ndarray,
    head: HeadMesh,
    *,
    valid_mm: float = 8.0,
    smooth_iterations: int = 200,
) -> np.ndarray:
    """Compact head vertices carried into one fitted view (FLAME-native metres, camera-ready)."""
    s, r, t = fit.similarity["scale"], fit.similarity["rotation"], fit.similarity["translation"]
    moved = template.base + fit.displacement
    native = ((moved - t) / s) @ r  # inverse of the FLAME -> template similarity
    view_normals = trimesh.Trimesh(view, flame.faces, process=False).vertex_normals
    free = np.flatnonzero(template.free)
    face, bary = fit.face_ids[free], fit.face_bary[free]
    valid = (fit.snap_distance[free] * 1000 <= valid_mm) & (face >= 0)
    posed = np.zeros((len(template.base), 3))
    delta = np.zeros((len(template.base), 3))
    known = np.zeros(len(template.base), bool)
    tri = flame.faces[face[valid]]
    surface = np.einsum("nk,nkj->nj", bary[valid], view[tri])
    normal = np.einsum("nk,nkj->nj", bary[valid], view_normals[tri])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    ids = free[valid]
    posed[ids] = surface + (fit.signed_offset[ids] / s)[:, None] * normal
    delta[ids] = posed[ids] - native[ids]
    known[ids] = True
    # Vertices without a FLAME counterpart (cavities, upper neck) follow the smooth field of their neighbours.
    if (~known[free]).any():
        edges = template.edges
        graph = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(template.base),) * 2)
        graph = (graph + graph.T).tocsr()
        degree = np.maximum(np.asarray(graph.sum(1)).ravel(), 1)
        walk = sp.diags(1 / degree) @ graph
        for _ in range(smooth_iterations):
            averaged = walk @ delta
            delta = np.where(known[:, None], delta, averaged)
    unknown = free[~known[free]]
    posed[unknown] = native[unknown] + delta[unknown]
    return posed[head.welded]


def flame_masks_on_head(template: Template, fit: HeadFit, flame: FlameFit, head: HeadMesh) -> dict:
    """Semantic masks (compact head vertex ids) transferred from FLAME through the registration correspondence."""
    dominant = flame.faces[np.maximum(fit.face_ids, 0)][np.arange(len(fit.face_ids)), fit.face_bary.argmax(1)]
    result = {}
    welded_flame = dominant[head.welded]
    for key in ("face", "eye_region", "lips", "neck"):
        member = np.zeros(len(flame.neutral), bool)
        member[flame.masks[key]] = True
        result[key] = np.flatnonzero(member[welded_flame] & (fit.face_ids[head.welded] >= 0))
    result["left_eyeball"] = np.zeros(0, np.int64)
    result["right_eyeball"] = np.zeros(0, np.int64)
    return result


def load_photos(photos_dir: Path, cameras: dict) -> dict:
    photos = {}
    for name in ("front", "right"):
        with Image.open(Path(photos_dir) / f"{name}.jpg") as image:
            if image.size != cameras[name].original_size:
                raise ValueError(f"{name} photo dimensions differ from the exported FLAME camera")
            photos[name] = np.array(image.convert("RGB"))
    return photos


@dataclass
class FaceBake:
    texture: np.ndarray  # (h, w, 3) uint8 crop of the atlas
    origin: tuple[int, int]  # (x, y) of the crop inside the atlas
    covered: np.ndarray  # (h, w) bool: texels of the head island
    confidence: np.ndarray  # (h, w) float: total view weight
    report: dict
    texel_y: np.ndarray
    texel_x: np.ndarray
    texel_face: np.ndarray  # head-mesh triangle per texel
    texel_bary: np.ndarray
    texel_points: np.ndarray  # (n, 3) template-frame position of every texel
    skin_mask: np.ndarray  # (h, w) bool: photographed skin samples (no eyes, lips, beard, glints)
    posed: dict  # view name -> compact head vertices carried into that view
    cameras: dict
    photos: dict


def uv_window(uv: np.ndarray, size: int, pad: int = 8, multiple: int = 32):
    low = np.floor(uv.min(0) * size).astype(int) - pad
    high = np.ceil(uv.max(0) * size).astype(int) + pad
    low = np.maximum(low, 0)
    extent = np.minimum(((high - low + multiple - 1) // multiple) * multiple, size - low)
    return int(low[0]), int(low[1]), int(extent[0]), int(extent[1])


def bake_face(
    template: Template,
    fit: HeadFit,
    flame: FlameFit,
    head: HeadMesh,
    uv: np.ndarray,
    photos_dir: Path,
    size: int,
    *,
    log=None,
) -> FaceBake:
    cameras = load_cameras(flame.fit_dir / "cameras.json")
    photos = load_photos(photos_dir, cameras)
    views, posed = {}, {}
    from flamehead.assets import read_mesh

    for name in ("front", "right"):
        views[name], faces = read_mesh(flame.fit_dir / "fitted_views" / f"{name}.ply")
        if views[name].shape != flame.neutral.shape or not np.array_equal(faces, flame.faces):
            raise ValueError(f"{name} fitted mesh topology differs from the neutral FLAME mesh")
        posed[name] = posed_vertices(template, fit, flame, views[name], head)
    moved = (template.base + fit.displacement)[head.welded]
    island = head.face_island == template.head_island
    island_faces = head.faces[island]
    vertex_uv = uv[head.ids]
    x0, y0, w, h = uv_window(vertex_uv[np.unique(island_faces)], size)
    local = vertex_uv * size - np.array([x0, y0])
    fid, bary = rasterize_uv(local, island_faces, w, h)
    y, x = np.nonzero(fid >= 0)
    face = fid[y, x]
    lam = bary[y, x]
    tri = island_faces[face]
    masks = flame_masks_on_head(template, fit, flame, head)
    # Texels whose template surface is far from FLAME (ears' backs, neck) carry no photo evidence.
    snapped = fit.snap_distance[head.welded] * 1000
    texel_gap = np.einsum("ij,ij->i", lam, snapped[tri])
    gate = 1.0 - smoothstep((texel_gap - 3.0) / 4.0)
    diagnostics = {}
    if log:
        log("baking photos into the head UV island")
    texture, report = bake_texels(
        moved,
        island_faces,
        fid,
        y,
        x,
        tri,
        lam,
        posed,
        cameras,
        photos,
        head.symmetry,
        masks,
        diagnostics,
        gate=gate.astype(np.float32),
    )
    report = dict(report)
    report["uv_window"] = {"x": x0, "y": y0, "width": w, "height": h}
    report["island_texels"] = int(len(y))
    report["gated_texels"] = int((gate < 0.5).sum())
    skin_mask = np.zeros((h, w), bool)
    skin_mask[y[diagnostics["skin"]], x[diagnostics["skin"]]] = True
    return FaceBake(
        texture,
        (x0, y0),
        fid >= 0,
        diagnostics["confidence"],
        report,
        y,
        x,
        face,
        lam,
        diagnostics["points"],
        skin_mask,
        posed,
        cameras,
        photos,
    )


def blend_unobserved(
    texture: np.ndarray,
    covered: np.ndarray,
    confidence: np.ndarray,
    skin_lab: np.ndarray,
    *,
    full_at: float = 0.06,
) -> tuple[np.ndarray, dict]:
    """Where the photos say little (grazing, hidden), fade to the flat skin tone; return the new texture."""
    alpha = smoothstep(confidence / full_at)
    # Smooth the transition in texture space so charts never show seams between confident and fallback texels.
    alpha = ndimage.gaussian_filter(alpha.astype(np.float32), 2.0) * covered
    lab = to_lab(texture)
    mixed = lab * alpha[..., None] + np.asarray(skin_lab, np.float32) * (1 - alpha[..., None])
    out = np.clip(np.rint(from_lab(mixed) * 255), 1, 255).astype(np.uint8)
    out[~covered] = texture[~covered]
    return out, {
        "observed_fraction": float((confidence[covered] > full_at).mean()) if covered.any() else 0.0,
        "mean_alpha": float(alpha[covered].mean()) if covered.any() else 0.0,
    }


def vertex_photo_check(
    face: FaceBake,
    head: HeadMesh,
    template: Template,
    vertex_uv: np.ndarray,
    atlas: np.ndarray,
    size: int,
    seed: int = 7,
) -> dict:
    """Independent check of the whole UV chain: photo colour at a vertex's projection vs the atlas at its UV.

    Uses only vertices of the head UV island that the front camera sees head-on. A correct bake gives a small
    Lab distance; the same vertices with shuffled photo colours give the chance level.
    """
    from flamehead.texture import project_samples, sample, visibility
    from twinrefine.scan import welded_vertex_normals

    island = np.unique(head.faces[head.face_island == template.head_island])
    mesh, camera = face.posed["front"], face.cameras["front"]
    depth, edge, scale = visibility(mesh, head.faces, camera)
    normals = welded_vertex_normals(mesh, head.faces)
    photo, weight = project_samples(mesh[island], normals[island], face.photos["front"], camera, depth, edge, scale)
    keep = weight > 0.5
    if keep.sum() < 50:
        return {"vertices": int(keep.sum()), "matched_mean_de": None, "shuffled_mean_de": None}
    ids = island[keep]
    baked = sample(atlas, vertex_uv[ids] * size).astype(np.uint8)
    photo_lab = to_lab(photo[keep].astype(np.float32) / 255)
    baked_lab = to_lab(baked.astype(np.float32) / 255)
    matched = np.linalg.norm(photo_lab - baked_lab, axis=1)
    shuffled = np.linalg.norm(photo_lab[np.random.default_rng(seed).permutation(len(ids))] - baked_lab, axis=1)
    return {
        "vertices": int(len(ids)),
        "matched_mean_de": float(matched.mean()),
        "matched_median_de": float(np.median(matched)),
        "shuffled_mean_de": float(shuffled.mean()),
    }
