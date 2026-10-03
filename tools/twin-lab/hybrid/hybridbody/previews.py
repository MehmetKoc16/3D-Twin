"""Private renders for the lead (bare and with hair, full body and head close-ups). Never open or display them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from headrecon.previews import _label
from PIL import Image
from twinrefine.render import DEFAULT_LIGHTS, RAKING_LIGHTS, render
from twinrefine.scan import welded_vertex_normals
from twintex.camera import OrthoCamera

from .assemble import Assembled

VIEWS = (("front", 0.0), ("side", 270.0), ("back", 180.0), ("three_quarter", 35.0))


def _render(mesh: Assembled, atlas: np.ndarray, camera, size, *, clay=False, lights=DEFAULT_LIGHTS, ss=1):
    height, width = atlas.shape[:2]
    uv = mesh.uv * np.array([width / height, 1.0])  # the renderer assumes a square atlas
    normals = welded_vertex_normals(mesh.positions, mesh.faces)
    return render(
        mesh.positions,
        mesh.faces,
        normals,
        camera,
        size[0],
        size[1],
        uv,
        atlas,
        lights=lights,
        clay=clay,
        ss=ss,
    )


def write_previews(folder: Path, full: Assembled, bare: Assembled, atlas: np.ndarray, head_y: float) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant, mesh in (("bare", bare), ("hair", full)):
        tiles = []
        for name, angle in VIEWS:
            camera = OrthoCamera.azimuth(name, angle).fit_bounds(mesh.positions, 640, 960, margin=0.05)
            image = _render(mesh, atlas, camera, (640, 960))
            path = folder / f"full_body_{variant}_{name}.png"
            Image.fromarray(image).save(path)
            paths[path.stem] = str(path)
            tiles.append(_label(image, f"{variant} {name}"))
        Image.fromarray(np.concatenate(tiles, axis=1)).save(folder / f"full_body_{variant}_contact.png")
        head = mesh.positions[mesh.positions[:, 1] > head_y]
        rows = []
        for clay in (False, True):
            tiles = []
            for name, angle in VIEWS:
                camera = OrthoCamera.azimuth(name, angle).fit_bounds(head, 640, 640, margin=0.08)
                image = _render(
                    mesh, atlas, camera, (640, 640), clay=clay, lights=RAKING_LIGHTS if clay else DEFAULT_LIGHTS
                )
                path = folder / f"head_{variant}_{name}{'_clay' if clay else ''}.png"
                Image.fromarray(image).save(path)
                paths[path.stem] = str(path)
                tiles.append(_label(image, f"{variant} {name}{' clay' if clay else ''}"))
            rows.append(np.concatenate(tiles, axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(folder / f"head_{variant}_contact.png")
    return paths
