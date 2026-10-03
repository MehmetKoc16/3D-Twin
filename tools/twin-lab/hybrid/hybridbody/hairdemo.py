"""Procedural hair on the GENERIC MakeHuman head (no person, no photo): previews to judge the hairstyle shape.

The head is the plain MakeHuman body for a chosen gender (default male), its skin a flat tone, the scalp darkened under
the hair exactly like the twin pipeline does, eyes from the CC0 eyes part. The same ``build_procedural_hair`` as in the
twin pipeline grows the hair (a separate mesh with its strand data atlas, as in the twin GLB), so the renders show the
real algorithm and its default style. The renders approximate the hair shader (coverage as soft alpha, root to tip
colour ramp): they are rough, the real look is the shader in the app.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flamehead.colour import to_lab
from mh import MHModel
from PIL import Image
from twinrefine.render import DEFAULT_LIGHTS, RAKING_LIGHTS

from . import BODY_ASSETS, PARTS_ASSETS
from .assemble import assemble
from .body import neck_definition
from .facetex import build_head_mesh
from .hair import DEFAULT_HAIR_HEX, build_procedural_hair
from .hairgen import HairStyle
from .headfit import build_template, read_face_map
from .partstex import eye_tile, make_tile, pack_strip
from .previews import HEAD_VIEWS, _atlas_tables, _label, _pyramid, _render, head_camera, render_all
from .skin import pad_texture, rasterize_bands, tint_scalp
from .template import load_part

SKIN_SRGB = (190, 146, 122)
DEFAULT_HAIR_SRGB = tuple(int(DEFAULT_HAIR_HEX[i : i + 2], 16) for i in (1, 3, 5))
SCALP_TINT = 0.5


def generic_head(gender: float = 1.0):
    """The undeformed MakeHuman body for ``gender`` with its head region: ``(model, combined, template, head)``."""
    model = MHModel()
    macro = {k: v["default"] for k, v in model.macro_vars.items()}
    macro["gender"] = gender
    combined = model.shape(macro, {}, ground=True)
    face_map = read_face_map(BODY_ASSETS / "face-map.json")
    template = build_template(combined[: model.nr], model.faces, face_map, neck_definition()["verts"])
    return model, combined, template, build_head_mesh(template)


def build_generic_twin(
    *,
    gender: float = 1.0,
    colour_srgb=None,
    style: HairStyle | None = None,
    size: int = 2048,
    coverage: bool = True,
    atlas_size: tuple[int, int] = (2048, 1024),
) -> dict:
    """Body and eyes on the generic head as an ``Assembled`` mesh plus its atlas, and the separate procedural hair."""
    model, combined, _, head = generic_head(gender)
    body = combined[: model.nr]
    eyes = load_part(PARTS_ASSETS, "eyes-default")
    eyes_bound = eyes.bind(combined)
    eye_y = float(eyes_bound[:, 1].mean())
    colour = np.asarray(DEFAULT_HAIR_SRGB if colour_srgb is None else colour_srgb, float)
    strip_height = size // 4
    build = build_procedural_hair(
        body[head.ids], head.faces, eye_y, colour, style=style, atlas_size=atlas_size, coverage=coverage
    )
    # skin: flat tone, darkened under the hair like the twin's head texture
    canvas = np.empty((size, size, 3), np.uint8)
    canvas[:] = SKIN_SRGB
    covered = np.zeros((size, size), bool)
    rows, cols, cover = [], [], []
    for r0, y, x, face, bary in rasterize_bands(model.uv, model.faces, size):
        points = np.einsum("ij,ijk->ik", bary, body[model.faces[face]])
        covered[r0 + y, x] = True
        near = points[:, 1] > eye_y - 0.09
        if near.any():
            rows.append(r0 + y[near])
            cols.append(x[near])
            cover.append(build.field.cover(points[near]))
    scalp = to_lab(np.clip(colour * SCALP_TINT, 1, 255).astype(np.float32).reshape(1, 3) / 255.0)[0]
    canvas = tint_scalp(canvas, np.concatenate(rows), np.concatenate(cols), np.concatenate(cover), scalp)
    canvas = pad_texture(canvas, covered)
    eyes_tile = make_tile(
        "eyes", eye_tile(eyes, np.array([76.0, 52.0, 34.0])), eyes.uv, 512 * size // 4096, strip_height
    )
    strip = pack_strip([eyes_tile], size, strip_height, y_offset=size)
    atlas = np.vstack((np.dstack((canvas, np.full((size, size), 255, np.uint8))), strip))
    height = atlas.shape[0]
    cavity = eyes.delete_verts
    mesh = assemble(
        body,
        model.faces,
        model.uv * np.array([1.0, size / height]),
        [("eyes", eyes, eyes_bound, eyes_tile)],
        (size, height),
        drop_vertices=cavity[cavity < model.nr],
    )
    head_y = float(body[head.ids][:, 1].min())
    head_points = np.vstack((mesh.positions, build.mesh.positions))
    head_points = head_points[head_points[:, 1] > head_y]
    return {
        "mesh": mesh,
        "atlas": atlas,
        "build": build,
        "head_points": head_points,
        "model": model,
        "combined": combined,
        "gender": gender,
    }


def write_generic_previews(folder: Path, twin: dict, size: int = 640, variants=(False, True), workers: int = 6) -> dict:
    """Head views (front, side, back, 3/4, 3/4 back, top), textured and clay, plus contact sheets."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mesh, atlas, head = twin["mesh"], twin["atlas"], twin["head_points"]
    _atlas_tables(atlas, 0.5)
    _pyramid(twin["build"].mesh.atlas)
    plan = [(clay, name, angle) for clay in variants for name, angle in HEAD_VIEWS]
    jobs = [
        lambda c=clay, n=name, a=angle: _render(
            mesh,
            atlas,
            head_camera(n, a, head, size, margin=0.06),
            (size, size),
            clay=c,
            lights=RAKING_LIGHTS if c else DEFAULT_LIGHTS,
            hair=twin["build"].mesh,
        )
        for clay, name, angle in plan
    ]
    images = render_all(jobs, workers)
    paths, rows = {}, {}
    for (clay, name, _), image in zip(plan, images, strict=True):
        path = folder / f"generic_head_{name}{'_clay' if clay else ''}.png"
        Image.fromarray(image).save(path)
        paths[path.stem] = str(path)
        rows.setdefault(clay, []).append(_label(image, f"{name}{' clay' if clay else ''}"))
    for clay, tiles in rows.items():
        Image.fromarray(np.concatenate(tiles, axis=1)).save(
            folder / f"generic_head_contact{'_clay' if clay else ''}.png"
        )
    return paths


def summary(twin: dict) -> dict:
    """The numbers of a generic build (what the twin report carries for the hair)."""
    r = twin["build"].report
    keys = ("cards", "triangles", "layers", "penetration", "coverage", "lengths_mm", "hairline", "strip")
    out = {k: r[k] for k in keys if k in r}
    out["body_atlas_size"] = [int(twin["atlas"].shape[1]), int(twin["atlas"].shape[0])]
    out["frame"] = r["frame"]
    return out


def dump_summary(folder: Path, twin: dict) -> Path:
    path = Path(folder) / "generic_hair_report.json"
    path.write_text(json.dumps(summary(twin), indent=2, default=float) + "\n", encoding="utf8")
    return path
