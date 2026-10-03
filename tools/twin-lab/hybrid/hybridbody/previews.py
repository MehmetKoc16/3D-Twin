"""Private renders for the lead (bare and with hair, full body and head close-ups). Never open or display them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from headrecon.previews import _label
from PIL import Image
from twinrefine.render import DEFAULT_LIGHTS, RAKING_LIGHTS
from twinrefine.scan import welded_vertex_normals
from twintex.camera import OrthoCamera

from .assemble import Assembled

VIEWS = (("front", 0.0), ("side", 270.0), ("back", 180.0), ("three_quarter", 35.0))


def render_cutout(verts, faces, normals, cam, width, height, uv, atlas, *, lights, clay=False, ss=2, cutoff=0.5):
    """The twinrefine renderer with the glTF material contract: alpha below the cutoff is discarded, faces are two-sided."""
    import cv2
    from twinrefine.render import BG, shade
    from twintex.bake import remap_points
    from twintex.colorspace import linear_to_srgb, srgb_to_linear
    from twintex.raster import barycentric_at, raster_pairs

    atlas_h, atlas_w = atlas.shape[:2]
    alpha = atlas[..., 3] if atlas.shape[2] == 4 else None
    W, H = width * ss, height * ss
    p = cam.project(verts)
    p[:, 0] *= ss
    p[:, 1] *= ss
    z = p[:, 2]
    zbuf = np.full(W * H, np.inf, np.float32)
    fid = np.full(W * H, -1, np.int32)
    for f, px, py, lam in raster_pairs(p[:, :2], faces, W, H):
        if alpha is not None:
            t = np.einsum("kj,kjc->kc", lam, uv[faces[f]].astype(np.float64))
            ix = np.clip((t[:, 0] * atlas_w).astype(int), 0, atlas_w - 1)
            iy = np.clip((t[:, 1] * atlas_h).astype(int), 0, atlas_h - 1)
            seen = alpha[iy, ix] >= cutoff * 255
            f, px, py, lam = f[seen], px[seen], py[seen], lam[seen]
        d = np.einsum("kj,kj->k", lam, z[faces[f]]).astype(np.float32)
        pix = py * W + px
        order = np.argsort(-d, kind="stable")
        d, pix, f = d[order], pix[order], f[order]
        keep = d < zbuf[pix]
        zbuf[pix[keep]] = d[keep]
        fid[pix[keep]] = f[keep]
    fid = fid.reshape(H, W)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    lam = barycentric_at(p[:, :2], faces, f, xs, ys)
    tri = faces[f]
    n = np.einsum("ij,ijk->ik", lam, normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.where(n @ cam.to_camera < 0, -1.0, 1.0)[:, None]  # two-sided shading
    ncam = np.stack([n @ cam.right, n @ cam.up, n @ cam.to_camera], axis=1)
    if clay:
        albedo = np.full((len(f), 3), srgb_to_linear(np.float32(0.72)), dtype=np.float32)
    else:
        t = np.einsum("ij,ijk->ik", lam, uv[tri].astype(np.float64))
        linear = srgb_to_linear(atlas[..., :3].astype(np.float32) / 255.0)
        albedo = np.clip(
            remap_points(
                linear, (t[:, 0] * atlas_w - 0.5).astype(np.float32), (t[:, 1] * atlas_h - 0.5).astype(np.float32)
            ),
            0,
            None,
        )
    colour = shade(ncam, albedo, lights)
    image = np.tile(srgb_to_linear(BG)[None, None, :], (H, W, 1)).astype(np.float32)
    image[ys, xs] = colour
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) if ss > 1 else image
    return np.clip(np.rint(linear_to_srgb(image) * 255), 0, 255).astype(np.uint8)


def _render(mesh: Assembled, atlas: np.ndarray, camera, size, *, clay=False, lights=DEFAULT_LIGHTS, ss=2):
    normals = welded_vertex_normals(mesh.positions, mesh.faces)
    return render_cutout(
        mesh.positions, mesh.faces, normals, camera, size[0], size[1], mesh.uv, atlas, lights=lights, clay=clay, ss=ss
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
