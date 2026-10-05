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


def hair_material_json(hair, texture_index: int = 1, normal_index: int | None = None) -> dict:
    """The ``dtHair`` material.

    Strand cards (``rcov-groot-bvar/1``): the strand data atlas as base colour texture, ``MASK`` 0.5 as fallback for plain
    viewers. The atlas is linear data (R coverage, G root to tip, B variation), not a colour image: the hair shader reads
    it with ``NoColorSpace`` and takes the colours from ``extras.dtHair``; ``baseColorFactor`` only tints what a viewer
    without the shader draws (dark brown cut-out cards).

    Solid shell (``shell/1``, contract addendum v1.1): a normal PBR material. The base colour texture is sRGB colour with
    the soft hairline fringe in its alpha channel (``MASK`` 0.5), an optional tangent-space normal texture, double sided.
    """
    from twintex.colorspace import srgb_to_linear

    from .hair import FORMAT_SHELL
    from .hairtex import FORMAT

    if getattr(hair, "format", FORMAT) == FORMAT_SHELL:
        material = {
            "name": hair.node,
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": texture_index},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.62,
            },
            "alphaMode": "MASK",
            "alphaCutoff": ALPHA_CUTOFF,
            "doubleSided": True,
            "extras": {"dtHair": {"format": FORMAT_SHELL, "colorHex": hair.colours["colorHex"], "cardCount": 0}},
        }
        if normal_index is not None:
            material["normalTexture"] = {"index": normal_index, "scale": 1.0}
        return material
    colour = np.array([int(hair.colours["colorHex"][i : i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0
    factor = np.clip(srgb_to_linear(colour) * 2.0, 0.0, 1.0)
    return {
        "name": hair.node,
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": texture_index},
            "baseColorFactor": [float(factor[0]), float(factor[1]), float(factor[2]), 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.55,
        },
        "alphaMode": "MASK",
        "alphaCutoff": ALPHA_CUTOFF,
        "doubleSided": True,
        "extras": {
            "dtHair": {
                "format": FORMAT,
                "colorHex": hair.colours["colorHex"],
                "rootHex": hair.colours["rootHex"],
                "tipHex": hair.colours["tipHex"],
                "cardCount": int(hair.card_count),
            }
        },
    }


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    """JPEG without chroma subsampling (normal maps keep their tilt in the colour channels)."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, "RGB").save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=True)
    return buffer.getvalue()


def write_glb(path, mesh: Assembled, atlas: np.ndarray, extras: dict, *, mime: str | None = None, hair=None) -> dict:
    """Write the skinless textured primitive of the body (the rig stage skins it) and validate a re-read.

    An RGBA atlas is stored as PNG (the alpha channel drives the cut-outs); an RGB atlas as JPEG unless ``mime`` says
    otherwise. ``hair`` (a ``HairMesh``) is written as a second primitive on its own node ``dtHair`` with its own
    material and strand data atlas; ``asset.extras.dtHairNode`` names the node.
    """
    normals = welded_vertex_normals(mesh.positions, mesh.faces)
    if atlas.shape[2] == 4:
        data, mime = encode_png(atlas), "image/png"
    else:
        data, mime = encode_atlas(atlas, mime or "image/jpeg")
    prims = [
        Prim(
            mesh.positions.astype(np.float32),
            mesh.faces.astype(np.uint32),
            normals.astype(np.float32),
            mesh.uv.astype(np.float32),
            0,
            "twin",
        )
    ]
    materials = [material_json()]
    textures = [{"sampler": 0, "source": 0}]
    images = [{"data": data, "mimeType": mime}]
    extras = dict(extras)
    hair_data = b""
    if hair is not None:
        hair_data = encode_png(hair.atlas)
        prims.append(
            Prim(
                hair.positions.astype(np.float32),
                hair.faces.astype(np.uint32),
                hair.normals.astype(np.float32),
                hair.uv.astype(np.float32),
                1,
                hair.node,
            )
        )
        textures.append({"sampler": 0, "source": 1})
        images.append({"data": hair_data, "mimeType": "image/png"})
        normal_index = None
        if getattr(hair, "normal_atlas", None) is not None:
            normal_data = encode_jpeg(hair.normal_atlas)
            textures.append({"sampler": 0, "source": 2})
            images.append({"data": normal_data, "mimeType": "image/jpeg"})
            normal_index = len(textures) - 1
        materials.append(hair_material_json(hair, 1, normal_index))
        extras["dtHairNode"] = hair.node
    scene = GlbScene(
        prims,
        materials,
        textures,
        [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        images,
        extras,
    )
    write_static_glb(str(path), scene)
    checked = read_glb(str(path))
    if len(checked.prims) != len(prims) or len(checked.prims[0].positions) != len(mesh.positions):
        raise ValueError("Written hybrid GLB failed its re-read")
    if not all(np.isfinite(p.positions).all() for p in checked.prims):
        raise ValueError("Written hybrid GLB has non-finite positions")
    js, _ = _split(open(path, "rb").read())
    info = {
        "bytes": len(open(path, "rb").read()),
        "image_bytes": len(data),
        "image_mime": mime,
        "material": js["materials"][0],
        "extras_keys": sorted(js["asset"]["extras"]),
    }
    if hair is not None:
        hair_prim = checked.prims[1]
        if hair_prim.name != hair.node or len(hair_prim.positions) != len(hair.positions) or hair_prim.uv is None:
            raise ValueError("Written hair node failed its re-read")
        info["hair"] = {
            "node": hair.node,
            "vertices": int(len(hair.positions)),
            "triangles": int(len(hair.faces)),
            "atlas_bytes": len(hair_data),
            "atlas_size": [int(hair.atlas.shape[1]), int(hair.atlas.shape[0])],
            "material": js["materials"][1],
        }
        if getattr(hair, "normal_atlas", None) is not None:
            info["hair"]["normal_bytes"] = len(normal_data)
            info["hair"]["normal_size"] = [int(hair.normal_atlas.shape[1]), int(hair.normal_atlas.shape[0])]
    return info
