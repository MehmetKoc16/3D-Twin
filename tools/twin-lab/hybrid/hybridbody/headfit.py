"""Fit the template head to the FLAME fit: landmark similarity, region weights, registration, correspondences.

Frames: ``template`` is the solved MakeHuman body (metres, +Y up, +Z front, character's left = +X). The FLAME fit lives
in native FLAME metres; one similarity (estimated from the FLAME/MediaPipe landmark embedding and the app's face-map)
carries FLAME into the template frame, and its scale is reported (the template keeps the body's own head size, so
FLAME's absolute scale is only used through the landmark layout).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from flamehead.assets import landmarks as flame_landmarks
from flamehead.assets import load_masks, read_mesh
from flamehead.geometry import similarity, transform
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .register import Landmarks, Params, Surface, register, smoothstep
from .template import triangle_quality, uv_islands, weld, welded_edges

REGION_WEIGHT = {"face": 1.0, "ear": 1.0, "scalp": 1.0, "other": 0.5, "neck": 0.2, "eye": 1.0, "none": 0.0}
FACE_KEYS = ("face", "nose", "lips", "eye_region", "forehead", "left_eye_region", "right_eye_region")


@dataclass
class Template:
    """The solved body in welded form, with its head region."""

    positions: np.ndarray  # (nr, 3) render vertices of the solved body (grounded A-pose)
    faces: np.ndarray  # (F, 3) render faces
    inverse: np.ndarray  # (nr,) welded id of every render vertex
    first: np.ndarray  # (W,) one render vertex per welded id
    base: np.ndarray  # (W, 3)
    faces_w: np.ndarray  # (F, 3) welded faces
    edges: np.ndarray  # (E, 2)
    islands: np.ndarray  # (nr,) UV island of every render vertex
    head_island: int
    in_head: np.ndarray  # (W,) welded vertex belongs to the head UV island
    anchor_y: float  # everything at or below stays exactly in place
    free: np.ndarray  # (W,) bool
    landmark_bary: sp.csr_matrix  # (468, W)
    landmark_index: np.ndarray  # MediaPipe index per row
    mirror: np.ndarray  # (W,) welded id of the x-mirrored vertex (-1 where none)


def build_template(positions: np.ndarray, faces: np.ndarray, face_map: dict, neck_vertices, margin: float = 5e-4):
    pos = np.asarray(positions, np.float64)
    inverse, first = weld(pos)
    base = pos[first]
    islands = uv_islands(faces, len(pos))
    head_island = int(islands[int(np.argmax(pos[:, 1]))])
    in_head = np.zeros(len(base), bool)
    in_head[inverse[islands == head_island]] = True
    anchor_y = float(pos[np.asarray(neck_vertices), 1].max() + margin)
    rows, cols, vals, index = [], [], [], []
    for row, item in enumerate(face_map["landmarks"]):
        index.append(item["index"])
        for tri, weight in zip(item["tri"], item["bary"], strict=True):
            rows.append(row)
            cols.append(inverse[tri])
            vals.append(weight)
    bary = sp.csr_matrix((vals, (rows, cols)), shape=(len(index), len(base)))
    mirrored = base * np.array([-1.0, 1.0, 1.0])
    distance, nearest = cKDTree(base).query(mirrored)
    mirror = np.where(distance < 1e-5, nearest, -1)
    return Template(
        pos,
        np.asarray(faces, np.int64),
        inverse,
        first,
        base,
        inverse[faces],
        welded_edges(faces, inverse),
        islands,
        head_island,
        in_head,
        anchor_y,
        base[:, 1] > anchor_y,
        bary,
        np.asarray(index),
        mirror,
    )


def seam_vertices(template: Template, torso_island: int) -> np.ndarray:
    """Welded vertices shared by the head UV island and the torso island: the UV seam around the neck."""
    in_torso = np.zeros(len(template.base), bool)
    in_torso[template.inverse[template.islands == torso_island]] = True
    return np.flatnonzero(template.in_head & in_torso)


@dataclass
class FlameFit:
    neutral: np.ndarray
    faces: np.ndarray
    masks: dict
    landmark_ids: np.ndarray
    landmark_points: np.ndarray
    fit_dir: Path
    assets_dir: Path


def load_flame_fit(fit_dir: Path, assets_dir: Path) -> FlameFit:
    neutral_path = Path(fit_dir) / "head_neutral.obj"
    if not neutral_path.is_file():
        neutral_path = Path(fit_dir) / "head_neutral.ply"
    neutral, faces = read_mesh(neutral_path)
    masks = load_masks(Path(assets_dir), len(neutral))
    ids, points = flame_landmarks(Path(assets_dir), neutral, faces)
    return FlameFit(neutral, faces, masks, ids, points, Path(fit_dir), Path(assets_dir))


def landmark_similarity(flame: FlameFit, template: Template, positions: np.ndarray, scale_band=(0.85, 1.2)):
    """Trimmed similarity FLAME -> template from the shared MediaPipe landmarks (outliers re-weighted, not removed)."""
    lookup = {int(i): row for row, i in enumerate(template.landmark_index)}
    ids = np.array([int(i) in lookup for i in flame.landmark_ids])
    if ids.sum() < 12:
        raise ValueError("FLAME and the face-map share too few landmarks for a similarity")
    rows = np.array([lookup[int(i)] for i in flame.landmark_ids[ids]])
    # Landmarks are bound to render triangles; evaluate them on the template's welded surface.
    target_all = np.asarray(template.landmark_bary @ template.base)
    target = target_all[rows]
    source = flame.landmark_points[ids]
    keep = np.ones(len(source), bool)
    for _ in range(4):
        scale, rotation, translation, clamped = similarity(source[keep], target[keep], band=scale_band)
        residual = np.linalg.norm(transform(source, scale, rotation, translation) - target, axis=1)
        keep = residual <= max(np.quantile(residual, 0.8), 0.002)
    if clamped:
        raise ValueError(f"FLAME to template scale left the plausible band {scale_band}")
    residual = np.linalg.norm(transform(source, scale, rotation, translation) - target, axis=1)
    return {
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "rows": rows,
        "source_ids": flame.landmark_ids[ids],
        "inliers": keep,
        "initial_rms_mm": float(np.sqrt(np.mean(residual**2)) * 1000),
        "initial_inlier_rms_mm": float(np.sqrt(np.mean(residual[keep] ** 2)) * 1000),
        "rotation_euler_deg": Rotation.from_matrix(rotation).as_euler("xyz", degrees=True).tolist(),
    }


def flame_labels(flame: FlameFit) -> np.ndarray:
    labels = np.full(len(flame.neutral), "other", dtype=object)
    for name, key in (("scalp", ["scalp"]), ("neck", ["neck", "boundary"]), ("face", FACE_KEYS)):
        for k in key:
            labels[flame.masks.get(k, [])] = name
    for k in ("left_ear", "right_ear"):
        labels[flame.masks[k]] = "ear"
    for k in ("left_eyeball", "right_eyeball"):
        labels[flame.masks[k]] = "eye"
    return labels


def region_weights(template: Template, aligned: np.ndarray, labels: np.ndarray, fade: float = 0.012):
    """Per welded vertex: label of the nearest FLAME vertex and the gate of the surface term."""
    distance, nearest = cKDTree(aligned).query(template.base)
    label = labels[nearest].copy()
    label[distance > 0.03] = "none"
    # Skin around the eyes is face; cavities (not in the head UV island) are never matched to the surface.
    label[(label == "eye") & template.in_head] = "face"
    weight = np.array([REGION_WEIGHT[k] for k in label])
    weight[~template.in_head] = 0.0
    weight[label == "eye"] = 0.0
    weight = weight * smoothstep((template.base[:, 1] - template.anchor_y) / fade)
    return label, weight


def repair_foldovers(base, displacement, faces, free, iterations: int = 40):
    """Pull the displacement of flipped triangles toward their mean (sub-millimetre slivers at lip and lid corners).

    Returns the repaired displacement and the number of triangles still flipped.
    """
    d = displacement.copy()
    count = len(base)
    flipped = np.zeros(0, int)
    for _ in range(iterations):
        a, b = (base + d)[faces], base[faces]
        na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
        nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
        flipped = np.flatnonzero(np.einsum("ij,ij->i", na, nb) <= 0)
        if len(flipped) == 0:
            break
        tri_mean = d[faces[flipped]].mean(1)
        total = np.zeros_like(d)
        owners = np.zeros(count)
        for k in range(3):
            np.add.at(total, faces[flipped, k], tri_mean)
            np.add.at(owners, faces[flipped, k], 1.0)
        vertices = np.flatnonzero((owners > 0) & free)
        d[vertices] = 0.5 * d[vertices] + 0.5 * total[vertices] / owners[vertices][:, None]
    return d, int(len(flipped))


@dataclass
class HeadFit:
    displacement: np.ndarray  # (W, 3) welded
    aligned_flame: np.ndarray  # FLAME neutral in the template frame
    similarity: dict
    labels: np.ndarray
    weights: np.ndarray
    face_ids: np.ndarray  # (W,) closest FLAME face per welded vertex (-1 outside the head region)
    face_bary: np.ndarray  # (W, 3)
    signed_offset: np.ndarray  # (W,) metres along the FLAME normal (template frame)
    snap_distance: np.ndarray  # (W,) metres from the registered vertex to the FLAME surface
    surface: Surface
    report: dict = field(default_factory=dict)


def fit_head(
    template: Template,
    flame: FlameFit,
    *,
    params: Params | None = None,
    neck_measure=None,
    log=None,
) -> HeadFit:
    sim = landmark_similarity(flame, template, template.positions)
    s, r, t = sim["scale"], sim["rotation"], sim["translation"]
    aligned = transform(flame.neutral, s, r, t)
    labels = flame_labels(flame)
    eye = np.zeros(len(aligned), bool)
    eye[np.r_[flame.masks["left_eyeball"], flame.masks["right_eyeball"]]] = True
    skin_faces = np.flatnonzero(eye[flame.faces].sum(1) < 2)
    surface = Surface(aligned, flame.faces[skin_faces])
    ear = np.zeros(len(aligned), bool)
    ear[np.r_[flame.masks["left_ear"], flame.masks["right_ear"]]] = True
    ear_faces = skin_faces[ear[flame.faces[skin_faces]].any(1)]
    ear_surface = Surface(aligned, flame.faces[ear_faces]) if len(ear_faces) else None
    label, weight = region_weights(template, aligned, labels)
    ears = (label == "ear") & template.in_head
    e = template.edges
    stiff = np.where(ears[e[:, 0]] & ears[e[:, 1]], 6.0, 1.0)
    source = flame.landmark_points[np.isin(flame.landmark_ids, sim["source_ids"])]
    targets = transform(source, s, r, t)
    bary = template.landmark_bary[sim["rows"]]
    inliers = np.where(sim["inliers"], 1.0, 0.3)
    constraint = Landmarks(bary.tocsr(), targets, inliers)
    displacement, info = register(
        template.base,
        template.faces_w,
        template.edges,
        template.free,
        surface,
        weight,
        landmarks=constraint,
        edge_stiffness=stiff,
        alt_surface=ear_surface,
        alt_mask=ears,
        params=params,
        log=log,
    )
    displacement, remaining = repair_foldovers(template.base, displacement, template.faces_w, template.free)
    moved = template.base + displacement
    ids = np.flatnonzero(template.free)
    point, face, bary_f, normal = surface.closest(moved[ids])
    snap = np.linalg.norm(point - moved[ids], axis=1)
    offset = np.einsum("ij,ij->i", moved[ids] - point, normal)
    face_ids = np.full(len(moved), -1, np.int64)
    face_ids[ids] = skin_faces[face]  # ids of the full FLAME topology (the views share it)
    face_bary = np.zeros((len(moved), 3))
    face_bary[ids] = bary_f
    signed = np.zeros(len(moved))
    signed[ids] = offset
    snapped = np.full(len(moved), np.inf)
    snapped[ids] = snap
    residual_by_region = {}
    for name in ("face", "ear", "scalp", "neck"):
        sel = (label[ids] == name) & template.in_head[ids]
        if sel.any():
            mm = snap[sel] * 1000
            residual_by_region[name] = {
                "vertices": int(sel.sum()),
                "mean_mm": float(mm.mean()),
                "rms_mm": float(np.sqrt(np.mean(mm**2))),
                "p95_mm": float(np.percentile(mm, 95)),
                "max_mm": float(mm.max()),
            }
    after = np.linalg.norm(constraint.bary @ moved - constraint.targets, axis=1) * 1000
    before = np.linalg.norm(constraint.bary @ template.base - constraint.targets, axis=1) * 1000
    norm = np.linalg.norm(displacement, axis=1)
    report = {
        "similarity_scale": s,
        "similarity_rotation_euler_deg": sim["rotation_euler_deg"],
        "landmarks": {
            "count": int(len(targets)),
            "inliers": int(sim["inliers"].sum()),
            "similarity_rms_mm": sim["initial_rms_mm"],
            "similarity_inlier_rms_mm": sim["initial_inlier_rms_mm"],
            "before_mean_mm": float(before.mean()),
            "after_mean_mm": float(after.mean()),
            "after_max_mm": float(after.max()),
        },
        "residual_to_flame_surface": residual_by_region,
        "displacement_mm": {
            "max": float(norm.max() * 1000),
            "mean_free": float(norm[template.free].mean() * 1000),
            "moved_vertices": int((norm > 1e-6).sum()),
            "fixed_below_anchor_max": float(norm[~template.free].max() * 1000) if (~template.free).any() else 0.0,
        },
        "anchor_y_m": template.anchor_y,
        "triangle_quality": triangle_quality(moved, template.faces_w, template.base),
        "foldovers_after_repair": remaining,
        "iterations": info["history"][-1] if info["history"] else {},
        "labels": {
            k: int(((label == k) & template.free).sum()) for k in ("face", "ear", "scalp", "neck", "other", "none")
        },
    }
    if neck_measure is not None:
        report["neck"] = neck_measure(moved)
    return HeadFit(displacement, aligned, sim, label, weight, face_ids, face_bary, signed, snapped, surface, report)


def read_face_map(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))
