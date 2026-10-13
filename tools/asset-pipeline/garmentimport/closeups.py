"""Close-up review renders (hem, sleeve openings, neck) of an imported top, with open boundary edges overlaid.

Run with the pipeline venv:

    python -m garmentimport.closeups <dir with <id>.glb, index.json> --id mh-jacket-t01 [--out previews/]

Uses the same CPU preview renderer as the visibility review (flat shading, no PBR normal maps), the exported body
delete mask plus the app's rest footprint, and draws every welded boundary edge in red so cracks or ragged openings
are obvious. Debug output only; nothing here is shipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .body import load_body
from .geometry import prism_coverage
from .mesh_io import load_parts
from .visibility import hidden_triangles

BODY_COLOR = [195, 154, 123]


def welded_boundary(positions, faces, digits=6, with_faces=False):
    """Boundary (single-use) edges of the position-welded mesh, as pairs of indices into `positions` (first copy).

    With `with_faces` also returns the index of the one face using each edge."""
    key = np.round(positions * 10**digits).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    welded = inverse.reshape(-1)[faces]
    edges = np.sort(np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]]), axis=1)
    unique, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    single = np.flatnonzero(counts == 1)
    if not with_faces:
        return first[unique[single]]
    face_of_edge = np.full(len(unique), -1)
    face_of_edge[inv.reshape(-1)] = np.tile(np.arange(len(faces)), 3)
    return first[unique[single]], face_of_edge[single]


def _boxes(points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    left = points[points[:, 0].argmax()]
    right = points[points[:, 0].argmin()]
    return {
        "hem": (np.array([-.22, lo[1] - .02, -.3]), np.array([.22, lo[1] + .12, .3])),
        "neck": (np.array([-.12, hi[1] - .1, -.3]), np.array([.12, hi[1] + .02, .3])),
        "sleeve_l": (left - [.1, .1, .3], left + [.1, .1, .3]),
        "sleeve_r": (right - [.1, .1, .3], right + [.1, .1, .3]),
    }


def render(directory: Path, identifier: str, out: Path, size=420):
    from PIL import Image, ImageDraw
    from parts_debug import Scene
    body = load_body(Path(__file__).resolve().parents[3]/"apps/web/public/assets/body")
    entry = next(e for e in json.loads((directory/"index.json").read_text())["garments"] if e["id"] == identifier)
    part = load_parts(directory/entry["mesh"])[0]
    deleted = np.frombuffer((directory/entry["deleteVerts"]).read_bytes(), dtype="<u4") if entry.get("deleteVerts") else []
    bp = body.positions[:body.render_count]
    coverage = prism_coverage(bp, part.positions, part.faces)
    hidden = hidden_triangles(body.faces, body.render_count, np.union1d(deleted, np.flatnonzero(coverage)).astype(int))
    boundary, boundary_faces = welded_boundary(part.positions, part.faces, with_faces=True)
    corners = part.positions[part.faces]
    normals = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    texture = part.material.get("base")
    garment = {"pos": part.positions, "tris": part.faces, "uv": part.uv, "color": [90, 135, 195]}
    if texture:
        garment["tex"] = np.array(texture["image"])
        garment["color"] = [255, 255, 255]
    tiles = []
    for name, bbox in _boxes(part.positions).items():
        for view in ("front", "side"):
            scene = Scene([{"pos": bp, "tris": body.faces[~hidden], "color": BODY_COLOR}], view, bbox, size)
            image = Image.fromarray(scene.render([garment]))
            draw = ImageDraw.Draw(image)
            toward = normals[boundary_faces][:, 2 if view == "front" else 0] > 0.05
            for a, b in boundary[toward]:
                pa, pb = part.positions[a], part.positions[b]
                ends = [((scene.sign_u*p[scene.ax_u]-scene.u0)*scene.scale, (scene.v1-p[1])*scene.scale) for p in (pa, pb)]
                draw.line(ends, fill=(230, 20, 20), width=1)
            tile = Image.new("RGB", (size+4, size+24), "white")
            tile.paste(image, (2, 20))
            ImageDraw.Draw(tile).text((4, 4), f"{identifier} {name} {view}", fill="black")
            tiles.append(tile)
    columns = 4
    rows = (len(tiles)+columns-1)//columns
    width, height = max(t.width for t in tiles), max(t.height for t in tiles)
    canvas = Image.new("RGB", (columns*width, rows*height), "white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % columns)*width, (i//columns)*height))
    out.mkdir(parents=True, exist_ok=True)
    path = out/f"{identifier}-closeups.png"
    canvas.save(path)
    return path


if __name__ == "__main__":
    from garment_import import require_venv
    require_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--size", type=int, default=420)
    args = parser.parse_args()
    print(render(args.directory, args.id, args.out or args.directory, args.size))
