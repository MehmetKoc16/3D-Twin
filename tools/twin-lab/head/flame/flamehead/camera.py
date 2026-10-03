"""The exported Pixel3DMM crop camera, with no estimated photo cameras."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    crop: np.ndarray  # ymin, ymax, xmin, xmax, continuous image-edge coordinates
    original_size: tuple[int, int]
    size: tuple[int, int] = (256, 256)

    def project(self, points: np.ndarray) -> np.ndarray:
        q = points @ self.extrinsic[:3, :3].T + self.extrinsic[:3, 3]
        depth = -q[:, 2]
        safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
        k = self.intrinsic
        return np.column_stack((k[0, 0] * q[:, 0] / safe + k[0, 2], k[1, 2] - k[1, 1] * q[:, 1] / safe, depth))

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        k = self.intrinsic
        q = np.column_stack(
            ((pixels[:, 0] - k[0, 2]) * depth / k[0, 0], (k[1, 2] - pixels[:, 1]) * depth / k[1, 1], -depth)
        )
        return (q - self.extrinsic[:3, 3]) @ np.linalg.inv(self.extrinsic[:3, :3]).T

    def to_original(self, pixels: np.ndarray) -> np.ndarray:
        y0, y1, x0, x1 = self.crop
        return pixels * np.array([(x1 - x0) / self.size[0], (y1 - y0) / self.size[1]]) + [x0, y0]

    def from_original(self, pixels: np.ndarray) -> np.ndarray:
        y0, y1, x0, x1 = self.crop
        return (pixels - [x0, y0]) * np.array([self.size[0] / (x1 - x0), self.size[1] / (y1 - y0)])

    @property
    def center(self) -> np.ndarray:
        return np.linalg.inv(self.extrinsic)[:3, 3]


def load_cameras(path: Path) -> dict[str, Camera]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "dt-flame-head-cameras/1":
        raise ValueError("Expected dt-flame-head-cameras/1")
    if data.get("projection") != "q = worldToCamera @ [x,y,z,1]; depth=-q.z; u=fx*q.x/depth+cx; v=cy-fy*q.y/depth":
        raise ValueError("Unsupported camera projection convention")
    result = {}
    for name in ("front", "right"):
        v = data["views"][name]
        cam = Camera(
            np.asarray(v["intrinsics"], float),
            np.asarray(v["worldToCamera"], float),
            np.asarray(v["cropBoundsYminYmaxXminXmax"], float),
            tuple(v["originalSizeWH"]),
            tuple(v.get("imageSizeWH", [256, 256])),
        )
        if cam.intrinsic.shape != (3, 3) or cam.extrinsic.shape != (4, 4):
            raise ValueError("Invalid camera matrices")
        if not all(np.isfinite(a).all() for a in (cam.intrinsic, cam.extrinsic, cam.crop)):
            raise ValueError("Nonfinite camera")
        if cam.crop[1] <= cam.crop[0] or cam.crop[3] <= cam.crop[2] or min(cam.size) <= 0:
            raise ValueError("Invalid crop dimensions")
        result[name] = cam
    return result
