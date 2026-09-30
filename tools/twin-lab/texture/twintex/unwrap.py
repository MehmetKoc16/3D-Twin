"""UV unwrapping with xatlas, cached on disk."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import xatlas


@dataclass
class Unwrapped:
    vertices: np.ndarray  # (V', 3) float32, split at UV seams
    faces: np.ndarray  # (F, 3) uint32
    uv: np.ndarray  # (V', 2) float32 in [0, 1], v pointing DOWN (glTF / image convention)
    vmapping: np.ndarray  # (V',) index into the source mesh vertices
    size: int
    chart_count: int
    utilization: float


def unwrap(
    mesh: trimesh.Trimesh,
    size: int = 4096,
    padding: int = 6,
    max_cost: float = 3.0,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
) -> Unwrapped:
    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    key = hashlib.sha1(positions.tobytes() + indices.tobytes() + f"{size}|{padding}|{max_cost}".encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"unwrap_{key}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        return Unwrapped(z["vertices"], z["faces"], z["uv"], z["vmapping"], int(z["size"]), int(z["charts"]), float(z["util"]))

    atlas = xatlas.Atlas()
    atlas.add_mesh(positions, indices)
    co = xatlas.ChartOptions()
    co.max_cost = max_cost
    po = xatlas.PackOptions()
    po.resolution = size
    po.padding = padding
    po.bilinear = True
    po.rotate_charts = True
    po.bruteForce = False
    atlas.generate(co, po, verbose=verbose)
    vmapping, new_indices, uvs = atlas[0]
    uv = np.asarray(uvs, dtype=np.float32).copy()
    # xatlas returns uvs in texel units of the atlas when a resolution is given; normalise to [0, 1].
    if uv.max() > 1.0 + 1e-3:
        uv[:, 0] /= atlas.width
        uv[:, 1] /= atlas.height
    out = Unwrapped(
        vertices=positions[vmapping].astype(np.float32),
        faces=np.asarray(new_indices, dtype=np.uint32).reshape(-1, 3),
        uv=uv,
        vmapping=np.asarray(vmapping, dtype=np.uint32),
        size=size,
        chart_count=int(atlas.chart_count),
        utilization=float(atlas.utilization[0]) if np.ndim(atlas.utilization) else float(atlas.utilization),
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, vertices=out.vertices, faces=out.faces, uv=out.uv, vmapping=out.vmapping,
                 size=size, charts=out.chart_count, util=out.utilization)
    return out
