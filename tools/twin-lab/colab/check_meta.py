"""Check a shape-stage GLB and twin-shape/1 metadata without displaying private data.

Requires numpy and trimesh (already used by the local shape stage).
Geometry alone cannot prove which anatomical side is the front; inspect that locally.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import trimesh

VIEW_YAW_DEG = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape and bool(np.isfinite(result).all()), f"Invalid {label}")
    return result


def project_points(points: np.ndarray, camera: dict) -> np.ndarray:
    """Project final-frame metres to ORIGINAL-image pixels, with v downwards."""
    yaw = np.deg2rad(camera["yawDeg"])
    horizontal = points[:, 0] * np.cos(yaw) - points[:, 2] * np.sin(yaw)
    scale = camera["pxPerMeter"]
    origin = camera["originPx"]
    return np.column_stack((origin[0] + scale * horizontal, origin[1] - scale * points[:, 1]))


def _embedded_glb(path: Path) -> None:
    data = path.read_bytes()
    _require(len(data) >= 20, "Truncated GLB")
    magic, version, length = struct.unpack_from("<4sII", data)
    _require(magic == b"glTF" and version == 2 and length == len(data), "Invalid GLB header")
    size, kind = struct.unpack_from("<II", data, 12)
    _require(kind == 0x4E4F534A and 20 + size <= len(data), "Missing GLB JSON chunk")
    document = json.loads(data[20 : 20 + size])
    for item in document.get("buffers", []) + document.get("images", []):
        _require("uri" not in item or item["uri"].startswith("data:"), "External GLB resource")


def validate_pair(mesh_path: Path, meta_path: Path, tolerance_m: float = 2e-5) -> dict:
    """Raise ValueError for an inconsistent bundle; return only a small numeric summary."""
    _require(np.isfinite(tolerance_m) and tolerance_m > 0, "Invalid tolerance")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _require(meta.get("schema") == "twin-shape/1", "Expected twin-shape/1")
    system = meta["coordinateSystem"]
    for key, expected in {"units": "meters", "handedness": "right", "up": "+Y", "characterLeft": "+X"}.items():
        _require(system.get(key) == expected, f"Invalid coordinateSystem.{key}")
    _require(system.get("forward") == "+Z (the character faces +Z)", "Invalid forward convention")
    _require(meta["files"]["glb"] == "mesh.glb", "Expected mesh.glb output")
    _embedded_glb(mesh_path)
    scene = trimesh.load_scene(mesh_path, process=False)
    mesh = scene.to_geometry()
    _require(isinstance(mesh, trimesh.Trimesh), "GLB must contain triangle geometry")
    vertices = np.asarray(mesh.vertices)
    _require(len(vertices) > 0 and len(mesh.faces) > 0 and bool(np.isfinite(vertices).all()), "Empty or nonfinite mesh")
    bounds = np.array([vertices.min(0), vertices.max(0)])
    normalization = meta["normalization"]
    height = float(normalization["heightCm"]) / 100
    _require(np.isfinite(height) and height > 0, "Invalid heightCm")
    _require(abs(bounds[0, 1]) <= tolerance_m, "Feet must be at y=0")
    _require(abs(bounds[1, 1] - bounds[0, 1] - height) <= tolerance_m, "Mesh height disagrees with heightCm")
    feet = vertices[vertices[:, 1] < bounds[0, 1] + 0.03 * height]
    center = (feet.min(0) + feet.max(0)) / 2
    _require(bool(np.abs(center[[0, 2]]).max() <= tolerance_m), "Feet contact patch must be centered in x,z")
    final = normalization["finalBBox"]
    for key, actual in (("min", bounds[0]), ("max", bounds[1]), ("sizeMeters", bounds[1] - bounds[0])):
        claimed = _array(final[key], (3,), f"finalBBox.{key}")
        _require(bool(np.allclose(claimed, actual, atol=tolerance_m, rtol=0)), f"Incorrect finalBBox.{key}")
    rotation = _array(normalization["rotationSourceToStandard"], (3, 3), "rotation")
    _require(bool(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)), "Rotation must be orthonormal")
    _require(abs(np.linalg.det(rotation) - 1) < 1e-6, "Rotation must be proper (no reflection)")
    scale = float(normalization["scaleMetersPerSourceUnit"])
    _require(np.isfinite(scale) and scale > 0, "Scale must be finite and positive")
    translation = _array(normalization["translationMeters"], (3,), "translation")
    transform = _array(normalization["matrix4x4RowMajor"], (4, 4), "matrix")
    expected = np.eye(4)
    expected[:3, :3] = rotation * scale
    expected[:3, 3] = translation
    _require(bool(np.allclose(transform, expected, atol=1e-7, rtol=0)), "Matrix disagrees with R, scale, translation")
    for key, actual in (("vertices", len(vertices)), ("faces", len(mesh.faces))):
        _require(meta["mesh"][key] == actual, f"Incorrect mesh.{key}")
    _require(meta["mesh"]["watertight"] == bool(mesh.is_watertight), "Incorrect mesh.watertight")
    inputs = meta["inputs"]
    cameras = meta["cameras"]["views"]
    _require("front" in inputs and set(inputs) == set(cameras), "One camera is required per input, including front")
    for view, camera in cameras.items():
        _require(view in VIEW_YAW_DEG, "Unknown camera view")
        _require(camera.get("yawDeg") == VIEW_YAW_DEG[view], f"Incorrect {view} yaw")
        image_size = _array(camera["imageSize"], (2,), "imageSize")
        _require(bool((image_size > 0).all() and (image_size == np.floor(image_size)).all()), "Invalid imageSize")
        _array(camera["originPx"], (2,), "originPx")
        px_per_meter = float(camera["pxPerMeter"])
        _require(np.isfinite(px_per_meter) and px_per_meter > 0, "Invalid pxPerMeter")
        iou = float(camera["silhouetteIoU"])
        _require(np.isfinite(iou) and 0 <= iou <= 1, "Invalid silhouetteIoU")
        bbox = _array(camera["maskBBoxPx"], (4,), "maskBBoxPx")
        _require(bool((bbox >= 0).all() and (bbox[:2] <= bbox[2:]).all() and (bbox[2:] < image_size).all()), "Invalid maskBBoxPx")
    model = meta["model"]
    used = model["viewsUsedForShape"]
    _require(bool(used) and set(used).issubset(inputs), "Invalid viewsUsedForShape")
    if model.get("repo") == "microsoft/TRELLIS.2-4B":
        _require(model["multiView"] is False and used == ["front"], "TRELLIS.2 does not support multi-view conditioning")
    return {"valid": True, "heightCm": height * 100, "cameraViews": list(cameras), "anatomicalFrontVerified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("meta", type=Path)
    args = parser.parse_args()
    try:
        result = validate_pair(args.mesh, args.meta)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
