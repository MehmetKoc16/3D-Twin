"""A textured scan in memory: index-space arrays (UV seams duplicated), welded views, GLB IO."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from glbio import GlbScene, Prim, read_glb, write_static_glb  # noqa: E402  (rig stage)
from PIL import Image

from . import log  # noqa: F401  (also puts rig/ and texture/ on sys.path)


def weld_ids(verts: np.ndarray, decimals: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Position-based welding like the rig stage (1e-5 m grid). Returns (inv, first): ``inv[i]`` is the welded id of
    vertex i and ``first[w]`` the index of one representative vertex of welded id w."""
    key = np.round(np.asarray(verts, dtype=np.float64) * 10.0**decimals).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return inv.ravel().astype(np.int64), first.astype(np.int64)


def face_normals_area(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Non-normalised face normals (length = 2 x area)."""
    t = verts[faces]
    return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])


def welded_vertex_normals(verts: np.ndarray, faces: np.ndarray, inv: np.ndarray | None = None) -> np.ndarray:
    """Angle-weighted vertex normals computed on the welded mesh (like ``trimesh``, which the rig stage uses) and
    returned per index-space vertex, so UV-seam duplicates share one normal."""
    if inv is None:
        inv, _ = weld_ids(verts)
    t = verts[faces]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-18)
    nw = int(inv.max()) + 1
    acc = np.zeros((nw, 3))
    for k in range(3):
        e1 = t[:, (k + 1) % 3] - t[:, k]
        e2 = t[:, (k + 2) % 3] - t[:, k]
        cosang = np.einsum("ij,ij->i", e1, e2) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-18)
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        np.add.at(acc, inv[faces[:, k]], fn * ang[:, None])
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    acc = acc / np.maximum(ln, 1e-18)
    return acc[inv]


def welded_faces(faces: np.ndarray, inv: np.ndarray) -> np.ndarray:
    f = inv[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return f[keep]


@dataclass
class Scan:
    verts: np.ndarray  # (n, 3) float64, metres, +Y up, +Z front
    faces: np.ndarray  # (m, 3) int64
    uv: np.ndarray  # (n, 2) float32, glTF convention (v down)
    atlas: np.ndarray | None  # (S, S, 3) uint8 sRGB baseColor
    mime: str = "image/jpeg"
    scene: GlbScene | None = None
    name: str = "twin"
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.verts)

    def copy(self) -> Scan:
        return Scan(
            self.verts.copy(), self.faces.copy(), self.uv.copy(),
            None if self.atlas is None else self.atlas.copy(), self.mime, self.scene, self.name, dict(self.extras),
        )


def load_scan(path: str | Path) -> Scan:
    """Read a (textured) single-primitive GLB written by the texture stage (or the shape stage: no atlas)."""
    scene = read_glb(str(path))
    if len(scene.prims) != 1:
        raise ValueError(f"expected a single-primitive mesh, found {len(scene.prims)}")
    p = scene.prims[0]
    verts = p.positions.astype(np.float64)
    faces = p.indices.astype(np.int64).reshape(-1, 3)
    uv = p.uv.astype(np.float32) if p.uv is not None else np.zeros((len(verts), 2), np.float32)
    atlas, mime = None, "image/jpeg"
    if p.uv is not None and scene.images and "data" in scene.images[0]:
        atlas = np.array(Image.open(io.BytesIO(scene.images[0]["data"])).convert("RGB"))
        mime = scene.images[0].get("mimeType", "image/jpeg")
    return Scan(verts, faces, uv, atlas, mime, scene, p.name)


def encode_atlas(atlas: np.ndarray, mime: str, jpeg_quality: int = 95) -> tuple[bytes, str]:
    im = Image.fromarray(atlas, "RGB")
    buf = io.BytesIO()
    if "jpeg" in mime or "jpg" in mime:
        im.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
        return buf.getvalue(), "image/jpeg"
    im.save(buf, format="PNG", compress_level=6)
    return buf.getvalue(), "image/png"


def save_scan(path: str | Path, scan: Scan, normals: np.ndarray | None = None, jpeg_quality: int = 95) -> None:
    """Write a single-primitive textured GLB (same layout the rig stage reads)."""
    if normals is None:
        normals = welded_vertex_normals(scan.verts, scan.faces)
    scene = scan.scene
    materials = list(scene.materials) if scene and scene.materials else []
    textures = list(scene.textures) if scene and scene.textures else []
    samplers = list(scene.samplers) if scene and scene.samplers else []
    images: list[dict] = []
    if scan.atlas is not None:
        data, mime = encode_atlas(scan.atlas, scan.mime, jpeg_quality)
        images = [{"data": data, "mimeType": mime}]
        if not materials:
            materials = [{"name": "twin_skin_cloth", "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.9}}]
            textures = [{"sampler": 0, "source": 0}]
            samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    prim = Prim(
        scan.verts.astype(np.float32), scan.faces.astype(np.uint32), normals.astype(np.float32),
        scan.uv.astype(np.float32) if scan.atlas is not None else None, 0 if materials else None, scan.name,
    )
    write_static_glb(str(path), GlbScene([prim], materials, textures, samplers, images))
