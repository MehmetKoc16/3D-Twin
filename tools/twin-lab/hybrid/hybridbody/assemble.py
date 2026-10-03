"""Single-primitive twin mesh: the deformed MakeHuman body plus its bound parts, and the GLB writer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from glbio import GlbScene, Prim, _split, read_glb, write_static_glb
from twinrefine.scan import encode_atlas, welded_vertex_normals

from .partstex import Tile
from .template import Part

KIND = {"body": 0, "eyes": 1, "eyebrows": 2, "eyelashes": 3, "hair": 4}
BACK_OFFSET = 1.5e-4  # metres: optional reversed copy of a card (``double_sided``; the material is double sided itself)
ALPHA_CUTOFF = 0.5


@dataclass
class Assembled:
    positions: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    kind: np.ndarray
    source_render: np.ndarray  # body render vertex id per output vertex (-1 for parts)
    parts: dict = field(default_factory=dict)  # name -> {"vertices": (start, stop), "faces": (start, stop)}

    def without(self, *names: str) -> Assembled:
        """A copy with whole parts dropped (previews of the bare body)."""
        drop = np.zeros(len(self.positions), bool)
        for name in names:
            drop |= self.kind == KIND[name]
        keep_face = ~drop[self.faces].any(1)
        remap = np.cumsum(~drop) - 1
        return Assembled(
            self.positions[~drop],
            remap[self.faces[keep_face]],
            self.uv[~drop],
            self.kind[~drop],
            self.source_render[~drop],
            {},
        )


def sphere_fit(points: np.ndarray):
    """Least-squares sphere (centre, radius) of a point set."""
    a = np.c_[2 * points, np.ones(len(points))]
    b = (points**2).sum(1)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    centre = x[:3]
    return centre, float(np.sqrt(max(x[3] + centre @ centre, 0.0)))


def eye_offsets(bound: np.ndarray, flame_eyes: dict, limit: float = 0.004) -> tuple[np.ndarray, dict]:
    """Per-vertex translation putting each bound eyeball where FLAME's eyeball is (clamped, centre in x/y, pole in z)."""
    shift = np.zeros_like(bound)
    report = {}
    for side, key in ((1.0, "left"), (-1.0, "right")):
        sel = np.sign(bound[:, 0]) == side
        centre, _ = sphere_fit(bound[sel])
        target_centre, target_radius = sphere_fit(flame_eyes[key])
        dx = target_centre[:2] - centre[:2]
        dz = flame_eyes[key][:, 2].max() - bound[sel][:, 2].max()
        delta = np.clip(np.array([dx[0], dx[1], dz]), -limit, limit)
        shift[sel] = delta
        report[key] = {
            "translation_mm": (delta * 1000).tolist(),
            "unclamped_mm": (np.array([dx[0], dx[1], dz]) * 1000).tolist(),
            "flame_radius_mm": target_radius * 1000,
        }
    return shift, report


def assemble(
    body_positions: np.ndarray,
    body_faces: np.ndarray,
    body_uv: np.ndarray,
    parts: list[tuple[str, Part, np.ndarray, Tile]],
    atlas_size: tuple[int, int],
    *,
    drop_vertices: np.ndarray | None = None,
    double_sided=(),
) -> Assembled:
    """Concatenate the body (minus ``drop_vertices`` faces) and the bound parts; UVs are atlas coordinates."""
    width, height = atlas_size
    keep_face = np.ones(len(body_faces), bool)
    if drop_vertices is not None and len(drop_vertices):
        dead = np.zeros(len(body_positions), bool)
        dead[drop_vertices] = True
        keep_face = ~dead[body_faces].any(1)
    faces = body_faces[keep_face]
    used = np.zeros(len(body_positions), bool)
    used[faces] = True
    remap = np.cumsum(used) - 1
    positions = [body_positions[used]]
    faces_out = [remap[faces]]
    uv = [body_uv[used]]
    kind = [np.zeros(int(used.sum()), np.int8)]
    source = [np.flatnonzero(used)]
    count = int(used.sum())
    slices = {"body": {"vertices": (0, count), "faces": (0, len(faces))}}
    face_total = len(faces)
    for name, part, bound, tile in parts:
        atlas_uv = tile.remap(part.uv, width, height)
        n = len(bound)
        v_start, f_start = count, face_total
        positions.append(bound)
        faces_out.append(part.faces + count)
        uv.append(atlas_uv)
        kind.append(np.full(n, KIND[part.category], np.int8))
        source.append(np.full(n, -1))
        count += n
        face_total += len(part.faces)
        if part.category in double_sided:
            normals = welded_vertex_normals(bound, part.faces)
            positions.append(bound - normals * BACK_OFFSET)
            faces_out.append(part.faces[:, ::-1] + count)
            uv.append(atlas_uv)
            kind.append(np.full(n, KIND[part.category], np.int8))
            source.append(np.full(n, -1))
            count += n
            face_total += len(part.faces)
        slices[name] = {"vertices": (v_start, count), "faces": (f_start, face_total)}
    return Assembled(
        np.vstack(positions),
        np.vstack(faces_out).astype(np.int64),
        np.vstack(uv),
        np.concatenate(kind),
        np.concatenate(source).astype(np.int64),
        slices,
    )


def mesh_validity(mesh: Assembled, decimals: int = 5) -> dict:
    """Closed-surface statistics of the body triangles (welded by position) and sanity of everything else."""
    from flamehead.geometry import edge_table
    from twinrefine.scan import weld_ids

    body = mesh.faces[: mesh.parts["body"]["faces"][1]] if mesh.parts else mesh.faces[mesh.kind[mesh.faces[:, 0]] == 0]
    inverse, _ = weld_ids(mesh.positions, decimals)
    faces = inverse[body]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    directed, count = edge_table(faces)
    both = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    signs = np.where(both[:, 0] < both[:, 1], 1, -1)
    _, index = np.unique(np.sort(both, axis=1), axis=0, return_inverse=True)
    winding = np.bincount(index.ravel(), weights=signs)
    tri = mesh.positions[mesh.faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    return {
        "body_boundary_edges": int((count == 1).sum()),
        "body_nonmanifold_edges": int((count > 2).sum()),
        "body_inconsistent_winding_edges": int(((count == 2) & (winding != 0)).sum()),
        "degenerate_triangles": int((area < 1e-12).sum()),
        "finite": bool(np.isfinite(mesh.positions).all() and np.isfinite(mesh.uv).all()),
        "uv_in_range": bool(((mesh.uv >= 0) & (mesh.uv <= 1)).all()),
        "min_triangle_area_mm2": float(area.min() * 1e6),
    }


def encode_png(atlas: np.ndarray) -> bytes:
    """Lossless PNG (RGBA when the atlas has an alpha channel)."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(atlas, "RGBA" if atlas.shape[2] == 4 else "RGB").save(
        buffer, format="PNG", compress_level=9, optimize=True
    )
    return buffer.getvalue()


def material_json() -> dict:
    """The single twin material: alpha cut-out cards (hair, lashes) in the base colour texture's alpha channel."""
    return {
        "name": "twin",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0,
            "roughnessFactor": 0.88,
        },
        "alphaMode": "MASK",
        "alphaCutoff": ALPHA_CUTOFF,
        "doubleSided": True,
    }


def write_glb(path, mesh: Assembled, atlas: np.ndarray, extras: dict, *, mime: str | None = None) -> dict:
    """Write one skinless textured primitive (the rig stage skins it) and validate a re-read.

    An RGBA atlas is stored as PNG (the alpha channel drives the cut-outs); an RGB atlas as JPEG unless ``mime`` says
    otherwise.
    """
    normals = welded_vertex_normals(mesh.positions, mesh.faces)
    if atlas.shape[2] == 4:
        data, mime = encode_png(atlas), "image/png"
    else:
        data, mime = encode_atlas(atlas, mime or "image/jpeg")
    prim = Prim(
        mesh.positions.astype(np.float32),
        mesh.faces.astype(np.uint32),
        normals.astype(np.float32),
        mesh.uv.astype(np.float32),
        0,
        "twin",
    )
    scene = GlbScene(
        [prim],
        [material_json()],
        [{"sampler": 0, "source": 0}],
        [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        [{"data": data, "mimeType": mime}],
        extras,
    )
    write_static_glb(str(path), scene)
    checked = read_glb(str(path))
    if len(checked.prims) != 1 or len(checked.prims[0].positions) != len(mesh.positions):
        raise ValueError("Written hybrid GLB failed its re-read")
    if not np.isfinite(checked.prims[0].positions).all():
        raise ValueError("Written hybrid GLB has non-finite positions")
    js, _ = _split(open(path, "rb").read())
    return {
        "bytes": len(open(path, "rb").read()),
        "image_bytes": len(data),
        "image_mime": mime,
        "material": js["materials"][0],
        "extras_keys": sorted(js["asset"]["extras"]),
    }
