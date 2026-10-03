"""Synthetic head, face-map and FLAME-like fit builders (no real person, model file or photo is involved)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from flamehead.pipeline import split_uv
from hybridbody.headfit import FlameFit
from PIL import Image
from scipy.spatial.transform import Rotation

LANDMARK_IDS = np.arange(1, 41) * 7  # arbitrary MediaPipe-like indices, 40 landmarks


def sphere_template(radius=0.1, subdivisions=3, centre_y=1.6):
    """A closed sphere split at its UV seam like the app's render mesh: (positions, faces, uv, neck-loop ids)."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices = np.asarray(mesh.vertices) + [0.0, centre_y, 0.0]
    faces = np.asarray(mesh.faces)
    rel = vertices - [0.0, centre_y, 0.0]
    u = np.arctan2(rel[:, 0], rel[:, 2]) / (2 * np.pi) + 0.5
    v = np.arccos(np.clip(rel[:, 1] / radius, -1, 1)) / np.pi
    corner_u = u[faces].copy()
    wrap = (corner_u.max(1) - corner_u.min(1)) > 0.5
    corner_u[wrap] = np.where(corner_u[wrap] < 0.25, corner_u[wrap] + 1.0, corner_u[wrap])
    corner = np.stack((corner_u * 0.7 + 0.02, v[faces] * 0.9 + 0.04), axis=-1)
    positions, faces_out, uv = split_uv(vertices, faces, corner)
    below = np.flatnonzero(positions[:, 1] < centre_y - 0.075)
    return positions, faces_out, uv, below


def face_map_for(positions, faces, count=40, seed=3):
    """Landmarks bound to front-facing triangles, in the format of apps/web/public/assets/body/face-map.json."""
    centroid = positions[faces].mean(1)
    front = np.flatnonzero((centroid[:, 2] > 0.03) & (centroid[:, 1] > positions[:, 1].mean() - 0.05))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(front, count, replace=False)
    landmarks = []
    for index, tri in zip(LANDMARK_IDS[:count], chosen, strict=True):
        bary = rng.dirichlet([2, 2, 2])
        landmarks.append({"index": int(index), "tri": faces[tri].tolist(), "bary": bary.tolist(), "uv": [0, 0]})
    return {"landmarks": landmarks}


def flame_like(
    positions,
    faces,
    face_map,
    *,
    scale=0.95,
    euler=(6.0, 0.0, 0.0),
    shift=(0.0, -1.5, 0.1),
    stretch=(1.06, 1.03, 1.0),
    folder=None,
):
    """A FLAME-like fit: the template head stretched, expressed in its own frame (inverse similarity).

    Returns (FlameFit, truth); ``truth`` carries the similarity FLAME -> template that ``fit_head`` should recover.
    """
    rotation = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    translation = np.array(shift)
    centre = positions.mean(0)
    target = (positions - centre) * stretch + centre  # what the registration should reach
    keys = np.round(positions * 1e8).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    to_flame = lambda p: ((p - translation) / scale) @ rotation  # noqa: E731
    landmark_points = np.array(
        [np.asarray(item["bary"]) @ to_flame(target)[item["tri"]] for item in face_map["landmarks"]], float
    )
    ids = np.array([item["index"] for item in face_map["landmarks"]])
    welded_target = target[first]
    rel = welded_target - welded_target.mean(0)
    masks = {
        key: np.zeros(0, np.int64)
        for key in (
            "nose",
            "lips",
            "eye_region",
            "forehead",
            "left_eye_region",
            "right_eye_region",
            "left_eyeball",
            "right_eyeball",
            "left_ear",
            "right_ear",
            "boundary",
        )
    }
    masks["face"] = np.flatnonzero(rel[:, 2] > 0.02)
    masks["scalp"] = np.flatnonzero(rel[:, 1] > 0.06)
    masks["neck"] = np.flatnonzero(rel[:, 1] < -0.06)
    folder = Path(folder) if folder else Path(".")
    fit = FlameFit(to_flame(welded_target), inverse[faces], masks, ids, landmark_points, folder, folder)
    return fit, {"scale": scale, "rotation": rotation, "translation": translation, "stretch": stretch}


def orbit_extrinsic(centre, azimuth_deg, distance=0.6):
    """OpenGL world->camera for a camera orbiting ``centre`` (azimuth 0 = +z, negative = the subject's right)."""
    a = np.radians(azimuth_deg)
    direction = np.array([np.sin(a), 0.0, np.cos(a)])
    position = np.asarray(centre) + distance * direction
    forward = -direction
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack((right, up, -forward))
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ position
    return extrinsic


def write_fit_files(folder: Path, flame: FlameFit, *, size=(256, 320)):
    """cameras.json, fitted_views/{front,right}.ply and gradient photos; the right camera orbits to the subject's right."""
    folder = Path(folder)
    (folder / "fitted_views").mkdir(parents=True, exist_ok=True)
    photos = folder / "photos"
    photos.mkdir(exist_ok=True)
    centre = flame.neutral.mean(0)
    cameras = {}
    for name, azimuth in (("front", 0.0), ("right", -70.0)):
        trimesh.Trimesh(flame.neutral, flame.faces, process=False).export(folder / "fitted_views" / f"{name}.ply")
        cameras[name] = {
            "intrinsics": [[420.0, 0, size[0] / 2], [0, 420.0, size[1] / 2], [0, 0, 1]],
            "worldToCamera": orbit_extrinsic(centre, azimuth).tolist(),
            "cropBoundsYminYmaxXminXmax": [0, size[1], 0, size[0]],
            "originalSizeWH": list(size),
            "imageSizeWH": list(size),
        }
        x = np.broadcast_to(np.linspace(20, 235, size[0])[None, :], (size[1], size[0]))
        y = np.broadcast_to(np.linspace(20, 235, size[1])[:, None], (size[1], size[0]))
        image = np.stack((x, y, np.full(x.shape, 120.0)), axis=2)
        Image.fromarray(image.astype(np.uint8)).save(photos / f"{name}.jpg", quality=100, subsampling=0)
    (folder / "cameras.json").write_text(
        json.dumps(
            {
                "schema": "dt-flame-head-cameras/1",
                "projection": "q = worldToCamera @ [x,y,z,1]; depth=-q.z; u=fx*q.x/depth+cx; v=cy-fy*q.y/depth",
                "views": cameras,
            }
        )
    )
    return photos
