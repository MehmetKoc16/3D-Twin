"""Private twin renders (bare and with hair, full body and head close-ups), never source photos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from headrecon.previews import _label
from PIL import Image
from twinrefine.render import DEFAULT_LIGHTS, RAKING_LIGHTS, shade
from twinrefine.scan import welded_vertex_normals
from twintex.camera import OrthoCamera
from twintex.colorspace import srgb_to_linear

from .assemble import Assembled

VIEWS = (("front", 0.0), ("side", 270.0), ("back", 180.0), ("three_quarter", 35.0))
FACE_VIEWS = (("front", 0.0), ("three_quarter_left", 35.0), ("three_quarter_right", 325.0),
              ("side_left", 90.0), ("side_right", 270.0))
HEAD_VIEWS = (*VIEWS, *FACE_VIEWS[1:], ("three_quarter_back", 215.0), ("top", None))
# hairline / temple QA close-ups from both sides (hair variant only, 960 px, not part of the contact sheets)
HAIR_QA_VIEWS = (
    ("qa_three_quarter_left", 35.0),
    ("qa_three_quarter_right", 325.0),
    ("qa_side_left", 90.0),
    ("qa_side_right", 270.0),
    ("qa_front", 0.0),
)


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


def render_cutout(
    verts, faces, normals, cam, width, height, uv, atlas, *, lights, clay=False, ss=2, cutoff=0.5, raw=False
):
    """The twinrefine renderer with the glTF material contract: alpha below the cutoff is discarded, faces are two-sided.

    ``raw=True`` returns the supersampled linear image and its depth buffer (for the hair pass) instead of the final image.
    """
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
    if raw:
        return image, zbuf
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) if ss > 1 else image
    return np.clip(np.rint(linear_to_srgb(image) * 255), 0, 255).astype(np.uint8)


BAYER3 = (np.array([[0, 7, 3], [6, 5, 2], [4, 1, 8]]) + 0.5) / 9.0  # 3 x 3 ordered dither: one threshold per sample
_PYRAMIDS: dict = {}


def _pyramid(atlas: np.ndarray, levels: int = 6) -> list:
    """Box-filtered float mip pyramid of a strand data atlas (cached per atlas array)."""
    from .hairtex import mip_levels

    key = (atlas.__array_interface__["data"][0], atlas.shape, int(atlas[::53, ::47].sum()))
    if key not in _PYRAMIDS:
        _PYRAMIDS.clear()
        _PYRAMIDS[key] = mip_levels(atlas, levels)
    return _PYRAMIDS[key]


def _sample_level(pyramid: list, level: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear (R, G, B) of the mip ``level`` chosen per fragment; ``uv`` is (k, 2) in atlas UV (clamped at the border)."""
    from twintex.bake import remap_points

    out = np.zeros((len(uv), 3), np.float32)
    for lv in np.unique(level):
        m = level == lv
        image = pyramid[int(lv)]
        h, w = image.shape[:2]
        x = np.clip(uv[m, 0] * w - 0.5, 0, w - 1).astype(np.float32)
        y = np.clip(uv[m, 1] * h - 0.5, 0, h - 1).astype(np.float32)
        out[m] = remap_points(image, x, y)
    return out


def render_hair_pass(
    image: np.ndarray,
    zbuf: np.ndarray,
    hair,
    cam,
    ss: int,
    *,
    lights,
    clay: bool = False,
    gain: float = 2.5,
    spec: float = 0.05,
    anisotropy: float = 8.0,
) -> np.ndarray:
    """Approximation of the creategamecharacters hair shader in its MSAA mode, on a supersampled image.

    The strand data atlas is mip filtered by the card's texel footprint, ``alpha = min(1, 2.5 R)`` decides per sample
    (ordered dither, like alpha to coverage), covered samples write depth, and their colour is the root to tip ramp
    (root colour at G = 0 to the base colour at G = 1, ``seedVariation`` 0.36 on B, the shader's normal-based self
    occlusion) lit with the smooth card normals. Rough: no anisotropic highlight, no blended fringe.
    """
    from twintex.raster import barycentric_at, raster_pairs

    H, W = image.shape[:2]
    faces, uv, normals = hair.faces, hair.uv, hair.normals
    pyramid = _pyramid(hair.atlas)
    atlas_h, atlas_w = hair.atlas.shape[:2]
    p = cam.project(hair.positions)
    uv_tex = uv * np.array([atlas_w, atlas_h])
    face_level = _mip_level(p[:, :2], uv_tex, faces, len(pyramid), anisotropy)
    p[:, 0] *= ss
    p[:, 1] *= ss
    z = p[:, 2]
    zb = zbuf.copy()
    fid = np.full(W * H, -1, np.int32)
    for f, px, py, lam in raster_pairs(p[:, :2], faces, W, H):
        t = np.einsum("kj,kjc->kc", lam, uv[faces[f]].astype(np.float64))
        cov = _sample_level(pyramid, face_level[f], t)[:, 0]
        seen = np.minimum(gain * cov, 1.0) > BAYER3[py % 3, px % 3]
        f, px, py, lam = f[seen], px[seen], py[seen], lam[seen]
        d = np.einsum("kj,kj->k", lam, z[faces[f]]).astype(np.float32)
        pix = py * W + px
        order = np.argsort(-d, kind="stable")
        d, pix, f = d[order], pix[order], f[order]
        keep = d < zb[pix]
        zb[pix[keep]] = d[keep]
        fid[pix[keep]] = f[keep]
    fid = fid.reshape(H, W)
    ys, xs = np.nonzero(fid >= 0)
    if len(ys) == 0:
        return image
    f = fid[ys, xs]
    lam = barycentric_at(p[:, :2], faces, f, xs, ys)
    tri = faces[f]
    n = np.einsum("ij,ijk->ik", lam, normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    ndv = np.abs(n @ cam.to_camera)  # the shader flips back faces towards the viewer: only |n . v| matters
    n *= np.where(n @ cam.to_camera < 0, -1.0, 1.0)[:, None]
    ncam = np.stack([n @ cam.right, n @ cam.up, n @ cam.to_camera], axis=1)
    t = np.einsum("ij,ijk->ik", lam, uv[tri].astype(np.float64))
    data = _sample_level(pyramid, face_level[f], t)
    colours = {k: srgb_to_linear(np.array(_hex_rgb(v), np.float32) / 255.0) for k, v in hair.colours.items()}
    if clay:
        albedo = np.full((len(f), 3), srgb_to_linear(np.float32(0.72)), np.float32)
    else:
        strand = 1.0 + (data[:, 2:3] - 0.5) * 2.0 * 0.36
        ao = 0.3 + 0.7 * _smoothstep((ndv - 0.05) / 0.5)[:, None]
        ramp = colours["rootHex"][None, :] * (1 - data[:, 1:2]) + colours["colorHex"][None, :] * data[:, 1:2]
        albedo = ramp * strand * ao
    image[ys, xs] = shade(ncam, albedo.astype(np.float32), lights, spec=spec)
    return image


_SHELL_LINEAR: dict = {}


def _shell_linear(atlas: np.ndarray) -> np.ndarray:
    """Linear colours of the shell's colour atlas (cached; a separate cache from the body atlas tables)."""
    key = (atlas.__array_interface__["data"][0], atlas.shape)
    if key not in _SHELL_LINEAR:
        _SHELL_LINEAR.clear()
        _SHELL_LINEAR[key] = srgb_to_linear(atlas[..., :3].astype(np.float32) / 255.0)
    return _SHELL_LINEAR[key]


def render_shell_pass(
    image: np.ndarray,
    zbuf: np.ndarray,
    hair,
    cam,
    ss: int,
    *,
    lights,
    clay: bool = False,
    spec: float = 0.14,
) -> np.ndarray:
    """The solid ``shell/1`` hair of the twin over a supersampled image: the glTF material as the app draws it.

    ``MASK`` 0.5 on the alpha channel of the colour texture, two-sided, depth tested against ``zbuf``, shaded with the
    smooth vertex normals bent by the tangent-space normal map (glTF convention: +Y up the image) when there is one.
    """
    from twintex.bake import remap_points
    from twintex.raster import barycentric_at, raster_pairs

    from .hairshell import orthonormal_frame, tangent_frames

    H, W = image.shape[:2]
    faces, uv, normals = hair.faces, hair.uv, hair.normals
    atlas = hair.atlas
    atlas_h, atlas_w = atlas.shape[:2]
    alpha = atlas[..., 3]
    p = cam.project(hair.positions)
    p[:, 0] *= ss
    p[:, 1] *= ss
    z = p[:, 2]
    zb = zbuf.copy()
    fid = np.full(W * H, -1, np.int32)
    for f, px, py, lam in raster_pairs(p[:, :2], faces, W, H):
        t = np.einsum("kj,kjc->kc", lam, uv[faces[f]].astype(np.float64))
        ix = np.clip((t[:, 0] * atlas_w).astype(int), 0, atlas_w - 1)
        iy = np.clip((t[:, 1] * atlas_h).astype(int), 0, atlas_h - 1)
        seen = alpha[iy, ix] >= 128
        f, px, py, lam = f[seen], px[seen], py[seen], lam[seen]
        d = np.einsum("kj,kj->k", lam, z[faces[f]]).astype(np.float32)
        pix = py * W + px
        order = np.argsort(-d, kind="stable")
        d, pix, f = d[order], pix[order], f[order]
        keep = d < zb[pix]
        zb[pix[keep]] = d[keep]
        fid[pix[keep]] = f[keep]
    fid = fid.reshape(H, W)
    ys, xs = np.nonzero(fid >= 0)
    if len(ys) == 0:
        return image
    f = fid[ys, xs]
    lam = barycentric_at(p[:, :2], faces, f, xs, ys)
    tri = faces[f]
    n = np.einsum("ij,ijk->ik", lam, normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    t = np.einsum("ij,ijk->ik", lam, uv[tri].astype(np.float64))
    x, y = (t[:, 0] * atlas_w - 0.5).astype(np.float32), (t[:, 1] * atlas_h - 0.5).astype(np.float32)
    if hair.normal_atlas is not None:
        t_face, b_face = tangent_frames(hair.positions, faces, uv)
        tangent, bitangent = orthonormal_frame(n, t_face[f], b_face[f])
        mapped = remap_points(hair.normal_atlas, x, y).astype(np.float32) / 255.0 * 2.0 - 1.0
        n = mapped[:, 0:1] * tangent + mapped[:, 1:2] * bitangent + mapped[:, 2:3] * n
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.where(n @ cam.to_camera < 0, -1.0, 1.0)[:, None]  # two-sided shading
    ncam = np.stack([n @ cam.right, n @ cam.up, n @ cam.to_camera], axis=1)
    if clay:
        albedo = np.full((len(f), 3), srgb_to_linear(np.float32(0.72)), np.float32)
    else:
        albedo = np.clip(remap_points(_shell_linear(atlas), x, y), 0, None)
    image[ys, xs] = shade(ncam, albedo.astype(np.float32), lights, spec=spec)
    return image


def _mip_level(screen: np.ndarray, texels: np.ndarray, faces: np.ndarray, levels: int, anisotropy: float) -> np.ndarray:
    """Per-face mip level like a GPU with ``anisotropy``x filtering: the pixel footprint in texture space is an ellipse
    (a, b texels); the level is ``log2(max(a / anisotropy, b))`` from the singular values of the screen/texel Jacobian."""
    e_screen = np.stack((screen[faces[:, 1]] - screen[faces[:, 0]], screen[faces[:, 2]] - screen[faces[:, 0]]), axis=2)
    e_tex = np.stack((texels[faces[:, 1]] - texels[faces[:, 0]], texels[faces[:, 2]] - texels[faces[:, 0]]), axis=2)
    det = e_tex[:, 0, 0] * e_tex[:, 1, 1] - e_tex[:, 0, 1] * e_tex[:, 1, 0]
    inverse = (
        np.stack(
            (
                np.stack((e_tex[:, 1, 1], -e_tex[:, 0, 1]), axis=1),
                np.stack((-e_tex[:, 1, 0], e_tex[:, 0, 0]), axis=1),
            ),
            axis=1,
        )
        / np.where(np.abs(det) < 1e-12, 1e-12, det)[:, None, None]
    )
    jacobian = e_screen @ inverse  # screen pixels per texel
    sigma = np.linalg.svd(jacobian, compute_uv=False)  # (faces, 2), descending
    major = 1.0 / np.maximum(sigma[:, 1], 1e-6)  # texels per pixel along the elongated axis
    minor = 1.0 / np.maximum(sigma[:, 0], 1e-6)
    lod = np.log2(np.maximum(np.maximum(major / anisotropy, minor), 1.0))
    return np.clip(np.rint(lod), 0, levels - 1).astype(int)


def _area2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Areas of 2-D triangles."""
    return np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 2


def _hex_rgb(text: str) -> tuple:
    text = text.lstrip("#")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _render(mesh: Assembled, atlas: np.ndarray, camera, size, *, clay=False, lights=DEFAULT_LIGHTS, ss=2, hair=None):
    """The body (cut-outs, two-sided) and, when ``hair`` (a ``HairMesh``) is given, the soft hair pass on top of it."""
    import cv2
    from twintex.colorspace import linear_to_srgb

    normals = welded_vertex_normals(mesh.positions, mesh.faces)
    if hair is None:
        return render_cutout(
            mesh.positions,
            mesh.faces,
            normals,
            camera,
            size[0],
            size[1],
            mesh.uv,
            atlas,
            lights=lights,
            clay=clay,
            ss=ss,
        )
    hair_ss = max(ss, 3)  # the dither matrix has 3 x 3 samples
    image, zbuf = render_cutout(
        mesh.positions, mesh.faces, normals, camera, size[0], size[1], mesh.uv, atlas,
        lights=lights, clay=clay, ss=hair_ss, raw=True,
    )  # fmt: skip
    if getattr(hair, "format", None) == "shell/1":
        image = render_shell_pass(image, zbuf, hair, camera, hair_ss, lights=lights, clay=clay)
    else:
        image = render_hair_pass(image, zbuf, hair, camera, hair_ss, lights=lights, clay=clay)
    image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(linear_to_srgb(image) * 255), 0, 255).astype(np.uint8)


def render_all(jobs: list, workers: int = 6) -> list:
    """Run render callables in a thread pool (numpy releases the GIL in the heavy parts: about 2.5x faster)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(lambda job: job(), jobs))


def write_previews(
    folder: Path, full: Assembled, bare: Assembled, atlas: np.ndarray, head_y: float, workers=6, hair=None
) -> dict:
    """Full body and head views, bare and with hair. ``hair`` (a ``HairMesh``) is the separate procedural hair of the
    twin: it is drawn with the soft shader approximation over ``full`` (cards of a MakeHuman hair part stay in ``full``)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    _atlas_tables(atlas, 0.5)  # built once before the threads start
    if hair is not None:
        if getattr(hair, "format", None) == "shell/1":
            _shell_linear(hair.atlas)  # the colour table is built once before the threads start
        else:
            _pyramid(hair.atlas)
    plan = []  # (kind, variant, name, clay, job)
    for variant, mesh, extra in (("bare", bare, None), ("hair", full, hair)):
        points = mesh.positions if extra is None else np.vstack((mesh.positions, extra.positions))
        for name, angle in VIEWS:
            camera = OrthoCamera.azimuth(name, angle).fit_bounds(points, 640, 960, margin=0.05)
            plan.append(
                (
                    "full_body",
                    variant,
                    name,
                    False,
                    lambda m=mesh, c=camera, h=extra: _render(m, atlas, c, (640, 960), hair=h),
                )
            )
        head = points[points[:, 1] > head_y]
        for clay in (False, True):
            for name, angle in HEAD_VIEWS if variant == "hair" else (*VIEWS, *FACE_VIEWS[1:]):
                camera = head_camera(name, angle, head, 640)
                lights = RAKING_LIGHTS if clay else DEFAULT_LIGHTS
                plan.append(
                    (
                        "head",
                        variant,
                        name,
                        clay,
                        lambda m=mesh, c=camera, k=clay, li=lights, h=extra: _render(
                            m, atlas, c, (640, 640), clay=k, lights=li, hair=h
                        ),
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
    if hair is not None:
        qa_head = np.vstack((full.positions, hair.positions))
        qa_head = qa_head[qa_head[:, 1] > head_y]
        qa_cameras = [(n, head_camera("qa", a, qa_head, 960, margin=0.04)) for n, a in HAIR_QA_VIEWS]
        qa_images = render_all(
            [lambda c=c: _render(full, atlas, c, (960, 960), hair=hair) for _, c in qa_cameras], workers
        )
        for (name, _), image in zip(qa_cameras, qa_images, strict=True):
            path = folder / f"head_hair_{name}.png"
            Image.fromarray(image).save(path)
            paths[path.stem] = str(path)
    for variant in ("bare", "hair"):
        Image.fromarray(np.concatenate(sheets[("full_body", variant, False)], axis=1)).save(
            folder / f"full_body_{variant}_contact.png"
        )
        rows = [np.concatenate(sheets[("head", variant, clay)], axis=1) for clay in (False, True)]
        Image.fromarray(np.concatenate(rows, axis=0)).save(folder / f"head_{variant}_contact.png")
    # Dedicated texture seam QA; these renders never load original photographs.
    shoulder = full.positions[(full.positions[:, 1] > head_y - .20) & (full.positions[:, 1] < head_y + .06)
                              & (np.abs(full.positions[:, 0]) < .34)]
    for name, angle in (("shoulder_neck_front", 0), ("shoulder_neck_back", 180), ("shoulder_neck_three_quarter", 45)):
        camera = OrthoCamera.azimuth(name, angle).fit_bounds(shoulder, 1000, 650, margin=.06)
        image = _render(full, atlas, camera, (1000, 650), hair=hair)
        path = folder / f"{name}.png"
        Image.fromarray(image).save(path)
        paths[name] = str(path)
    return paths


EAR_NECK_VIEWS = {"left": (("side", 90.0), ("three_quarter_front", 55.0), ("three_quarter_back", 125.0)),
                  "right": (("side", 270.0), ("three_quarter_front", 305.0), ("three_quarter_back", 235.0))}


def write_ear_neck_closeups(folder: Path, full: Assembled, atlas: np.ndarray, ear_points: dict, hair=None, size: int = 900) -> dict:
    """Close-ups of each ear with the neck below and behind it (side, 3/4 front and 3/4 back) for texture QA."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    points = full.positions if hair is None else np.vstack((full.positions, hair.positions))
    paths = {}
    for side, views in EAR_NECK_VIEWS.items():
        ear = ear_points[side]
        if len(ear) == 0:
            continue
        centre = ear.mean(0)
        near = points[
            (np.abs(points[:, 0] - centre[0]) < 0.12) & (points[:, 1] > centre[1] - 0.11) & (points[:, 1] < centre[1] + 0.05)
            & (np.abs(points[:, 2] - centre[2]) < 0.10)
        ]
        for name, angle in views:
            camera = OrthoCamera.azimuth(name, angle).fit_bounds(near, size, size, margin=0.03)
            image = _render(full, atlas, camera, (size, size), hair=hair)
            path = folder / f"closeup_ear_neck_{side}_{name}.png"
            Image.fromarray(image).save(path)
            paths[path.stem] = str(path)
    return paths
