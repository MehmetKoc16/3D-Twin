"""Iris and hair colour read from the fitted photos (numbers only: nothing is displayed or stored but two colours)."""

from __future__ import annotations

import numpy as np

from .partstex import DEFAULT_IRIS_SRGB, robust_colour, sample_ring

IRIS_RADIUS_M = 0.0058  # human iris radius, in native FLAME metres


def iris_colour(flame, front_vertices, camera, photo, ring=(0.4, 0.85)) -> dict:
    """Median sRGB of the iris ring around each eyeball's front pole in the front photo."""
    pixels, poles = [], []
    for key in ("left_eyeball", "right_eyeball"):
        vertices = front_vertices[flame.masks[key]]
        centre = vertices.mean(0)
        toward = camera.center - centre
        toward /= np.linalg.norm(toward)
        pole = vertices[np.argmax((vertices - centre) @ toward)]
        projected = camera.project(pole[None])[0]
        original = camera.to_original(projected[None, :2])[0]
        focal = camera.intrinsic[0, 0] * (camera.crop[3] - camera.crop[2]) / camera.size[0]
        radius = focal * IRIS_RADIUS_M / projected[2]
        pixels.append(sample_ring(photo, original, radius, *ring))
        poles.append({"photo_px": original.tolist(), "radius_px": float(radius)})
    pixels = np.concatenate(pixels)
    colour, count = robust_colour(pixels, (8, 78))
    return {
        "srgb": (np.asarray(colour) if colour is not None else DEFAULT_IRIS_SRGB).tolist(),
        "samples": count,
        "from_photo": colour is not None,
        "poles": poles,
    }


def scalp_pixels(flame, mesh, camera, photo, min_facing: float = 0.1) -> np.ndarray:
    """sRGB photo pixels under the front-facing triangles of FLAME's scalp (the hair-covered region)."""
    import cv2

    scalp = np.zeros(len(mesh), bool)
    scalp[np.asarray(flame.masks["scalp"], int)] = True
    faces = flame.faces[scalp[flame.faces].all(1)]
    normals = np.cross(mesh[faces[:, 1]] - mesh[faces[:, 0]], mesh[faces[:, 2]] - mesh[faces[:, 0]])
    centre = mesh[faces].mean(1)
    toward = camera.center - centre
    facing = np.einsum("ij,ij->i", normals, toward) / np.maximum(
        np.linalg.norm(normals, axis=1) * np.linalg.norm(toward, axis=1), 1e-12
    )
    mask = np.zeros(photo.shape[:2], np.uint8)
    for tri in faces[np.abs(facing) > min_facing]:
        projected = camera.project(mesh[tri])
        pixels = camera.to_original(projected[:, :2])
        cv2.fillConvexPoly(mask, np.rint(pixels).astype(np.int32), 1)
    return photo[mask > 0].astype(np.float64)


def hair_colour(flame, views, cameras, photos) -> dict:
    """Median sRGB of the dark pixels covering FLAME's scalp, seen through each fitted view."""
    pool = [scalp_pixels(flame, views[name], cameras[name], photos[name]) for name in ("front", "right")]
    pixels = np.concatenate(pool)
    # Hair, not skin or a bright background: dark pixels only.
    colour, count = robust_colour(pixels, (4, 36))
    if colour is None:
        return {"srgb": None, "samples": count, "from_photo": False}
    return {"srgb": np.asarray(colour).tolist(), "samples": count, "from_photo": True}
