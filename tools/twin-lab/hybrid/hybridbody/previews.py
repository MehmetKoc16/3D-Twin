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
HEAD_VIEWS = (*VIEWS, ("three_quarter_back", 215.0), ("top", None))


def head_camera(name: str, angle: float | None, points: np.ndarray, size: int, margin: float = 0.08) -> OrthoCamera:
    """Orthographic camera framing ``points``; ``top`` looks straight down with the front of the face up."""
    if name == "top":
        camera = OrthoCamera("top", np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    else:
        camera = OrthoCamera.azimuth(name, angle)
    return camera.fit_bounds(points, size, size, margin=margin)


_TABLES: dict = {}


def _atlas_tables(atlas: np.ndarray, cutoff: float) -> dict:
    """Per-atlas tables shared by all renders of one preview run: integral image of the cut-out texels, linear colours."""
    from twintex.colorspace import srgb_to_linear

    key = (atlas.__array_interface__["data"][0], atlas.shape, cutoff, int(atlas[::97, ::89].sum()))
    if key not in _TABLES:
        _TABLES.clear()
        tables = {"integral": None, "linear": None}
        if atlas.shape[2] == 4:
            holes = (atlas[..., 3] < cutoff * 255).astype(np.int32)
            tables["integral"] = np.pad(holes.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
        _TABLES[key] = tables
    tables = _TABLES[key]
    if tables["linear"] is None:
        tables["linear"] = srgb_to_linear(atlas[..., :3].astype(np.float32) / 255.0)
    return tables


def render_cutout(verts, faces, normals, cam, width, height, uv, atlas, *, lights, clay=False, ss=2, cutoff=0.5):
    """The twinrefine renderer with the glTF material contract: alpha below the cutoff is discarded, faces are two-sided."""
    import cv2
    from twinrefine.render import BG, shade
    from twintex.bake import remap_points
    from twintex.colorspace import linear_to_srgb, srgb_to_linear
    from twintex.raster import barycentric_at, raster_pairs

    atlas_h, atlas_w = atlas.shape[:2]
    alpha = atlas[..., 3] if atlas.shape[2] == 4 else None
    tables = _atlas_tables(atlas, cutoff) if not clay or alpha is not None else None
    needs_test = None
    if alpha is not None:
        # only triangles whose UV box touches a cut-out texel need the per-fragment alpha test (skin never does)
        integral = tables["integral"]
        box = uv[faces] * np.array([atlas_w, atlas_h])
        x0 = np.clip(np.floor(box[..., 0].min(1)).astype(int), 0, atlas_w - 1)
        x1 = np.clip(np.ceil(box[..., 0].max(1)).astype(int) + 1, 1, atlas_w)
        y0 = np.clip(np.floor(box[..., 1].min(1)).astype(int), 0, atlas_h - 1)
        y1 = np.clip(np.ceil(box[..., 1].max(1)).astype(int) + 1, 1, atlas_h)
        needs_test = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) > 0
    W, H = width * ss, height * ss
    p = cam.project(verts)
    p[:, 0] *= ss
    p[:, 1] *= ss
    z = p[:, 2]
    zbuf = np.full(W * H, np.inf, np.float32)
    fid = np.full(W * H, -1, np.int32)
    for f, px, py, lam in raster_pairs(p[:, :2], faces, W, H):
        if alpha is not None:
            seen = np.ones(len(f), bool)
            test = np.flatnonzero(needs_test[f])
            if len(test):
                t = np.einsum("kj,kjc->kc", lam[test], uv[faces[f[test]]].astype(np.float64))
                ix = np.clip((t[:, 0] * atlas_w).astype(int), 0, atlas_w - 1)
                iy = np.clip((t[:, 1] * atlas_h).astype(int), 0, atlas_h - 1)
                seen[test] = alpha[iy, ix] >= cutoff * 255
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
        linear = tables["linear"] if tables is not None else srgb_to_linear(atlas[..., :3].astype(np.float32) / 255.0)
        albedo = np.clip(
            remap_points(
                linear, (t[:, 0] * atlas_w - 0.5).astype(np.float32), (t[:, 1] * atlas_h - 0.5).astype(np.float32)
            ),
            0,
            None,
        )
    colour = shade(ncam, albedo, lights, spec=0.04)  # the twin material is rough (0.88): only a faint highlight
    image = np.tile(srgb_to_linear(BG)[None, None, :], (H, W, 1)).astype(np.float32)
    image[ys, xs] = colour
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) if ss > 1 else image
    return np.clip(np.rint(linear_to_srgb(image) * 255), 0, 255).astype(np.uint8)


def _render(mesh: Assembled, atlas: np.ndarray, camera, size, *, clay=False, lights=DEFAULT_LIGHTS, ss=2):
    normals = welded_vertex_normals(mesh.positions, mesh.faces)
    return render_cutout(
        mesh.positions, mesh.faces, normals, camera, size[0], size[1], mesh.uv, atlas, lights=lights, clay=clay, ss=ss
    )


def render_all(jobs: list, workers: int = 6) -> list:
    """Run render callables in a thread pool (numpy releases the GIL in the heavy parts: about 2.5x faster)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(lambda job: job(), jobs))


def write_previews(folder: Path, full: Assembled, bare: Assembled, atlas: np.ndarray, head_y: float, workers=6) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    _atlas_tables(atlas, 0.5)  # built once before the threads start
    plan = []  # (kind, variant, name, clay, job)
    for variant, mesh in (("bare", bare), ("hair", full)):
        for name, angle in VIEWS:
            camera = OrthoCamera.azimuth(name, angle).fit_bounds(mesh.positions, 640, 960, margin=0.05)
            plan.append(("full_body", variant, name, False, lambda m=mesh, c=camera: _render(m, atlas, c, (640, 960))))
        head = mesh.positions[mesh.positions[:, 1] > head_y]
        for clay in (False, True):
            for name, angle in HEAD_VIEWS if variant == "hair" else VIEWS:  # top / 3-4 back only with hair
                camera = head_camera(name, angle, head, 640)
                lights = RAKING_LIGHTS if clay else DEFAULT_LIGHTS
                plan.append(
                    (
                        "head",
                        variant,
                        name,
                        clay,
                        lambda m=mesh, c=camera, k=clay, li=lights: _render(m, atlas, c, (640, 640), clay=k, lights=li),
                    )
                )
    images = render_all([item[4] for item in plan], workers)
    sheets: dict = {}
    for (kind, variant, name, clay, _), image in zip(plan, images, strict=True):
        suffix = "_clay" if clay else ""
        path = folder / f"{kind}_{variant}_{name}{suffix}.png"
        Image.fromarray(image).save(path)
        paths[path.stem] = str(path)
        sheets.setdefault((kind, variant, clay), []).append(_label(image, f"{variant} {name}{' clay' if clay else ''}"))
    for variant in ("bare", "hair"):
        Image.fromarray(np.concatenate(sheets[("full_body", variant, False)], axis=1)).save(
            folder / f"full_body_{variant}_contact.png"
        )
        rows = [np.concatenate(sheets[("head", variant, clay)], axis=1) for clay in (False, True)]
        Image.fromarray(np.concatenate(rows, axis=0)).save(folder / f"head_{variant}_contact.png")
    return paths
