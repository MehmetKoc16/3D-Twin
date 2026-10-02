"""Numeric camera export for Pixel3DMM's OpenGL pinhole model (synthetic-testable)."""

import numpy as np


def camera_matrices(checkpoint: dict) -> tuple:
    size = np.asarray(checkpoint["img_size"], dtype=int).reshape(2)
    if size[0] != size[1]:
        raise ValueError("This adapter requires square cropped images")
    side = int(size[0])
    camera = checkpoint["camera"]
    focal = float(np.asarray(camera["fl"]).reshape(-1)[0]) * side
    pp = np.asarray(camera["pp"]).reshape(-1, 2)[0]
    center = (1 + pp) * (side / 2 + 0.5)
    intrinsics = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]])
    base = np.eye(4)
    base[:3, :3] = np.asarray(camera["R_base_0"]).reshape(-1, 3, 3)[0]
    base[:3, 3] = np.asarray(camera["t_base_0"]).reshape(-1, 3)[0]
    head = np.eye(4)
    head[:3, :3] = np.asarray(checkpoint["flame"]["R_rotation_matrix"]).reshape(-1, 3, 3)[0]
    head[:3, 3] = np.asarray(checkpoint["flame"]["t"]).reshape(-1, 3)[0]
    extrinsics = base @ head
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all() or focal <= 0:
        raise ValueError("Invalid fitted camera")
    return intrinsics, extrinsics, side


def projection_matrix(intrinsics: np.ndarray, side: int) -> np.ndarray:
    """Near/far = 0.1/5 m, matching upstream tracking's rasterizer.

    Pixel-array row 0 corresponds to NDC y=-1: negative fy is deliberate.
    This matrix uses pixel edge coordinates; pixel i's center is i+0.5.
    """
    near, far = 0.1, 5.0
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = 2 * intrinsics[0, 0] / side
    result[1, 1] = -2 * intrinsics[1, 1] / side
    result[0, 2] = 1 - 2 * intrinsics[0, 2] / side
    result[1, 2] = 1 - 2 * intrinsics[1, 2] / side
    result[2, 2] = -(far + near) / (far - near)
    result[2, 3] = -2 * far * near / (far - near)
    result[3, 2] = -1
    return result
