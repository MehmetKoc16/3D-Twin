"""Hunyuan3D bust of the person (``user-data/twin/hy3d/hy3d.glb``): load, find axes and scale, map it onto the twin head.

The bust is a PRIVATE, user-generated asset (head and shoulders with hair, glasses and clothes, one mesh, one PBR
material with 4096 px textures, no skin). It never leaves ``user-data/``; only the hair shell cut out of it
(``hairshell.py``) goes into the twin bundle, which is private as well.

Frames. ``read_glb`` bakes the node transforms: Hunyuan3D's exporter writes a +90 degree rotation about X on the node
(its raw files are Z-up), after which the bust is +Y up and faces +Z, the repo convention. The orientation is verified
by a face detector on renders (``find_orientation`` tries the 24 axis-aligned rotations when the identity fails), the
scale is metric only after the landmark similarity of ``trimmed_similarity`` (the bust is not in metres).

Landmarks. The front render of the bust goes through MediaPipe's face landmarker (the model file the web app ships);
every landmark pixel is lifted to 3-D with the depth buffer of the same render and compared with the same landmark on
the twin head (the app's ``face-map.json`` binding). Eyes, brows and lips are not used (glasses, a smile, closed eyes);
the similarity is trimmed so glasses frames and the smile cannot pull it. A small rigid ICP on the forehead and
temple skin then makes the hairline land where the twin's forehead is.

The mesh is position-welded on load (Hunyuan3D splits vertices at every UV seam, which would cut the surface into
islands for any graph operation); vertex colours are the base colour texture sampled at the vertices.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from glbio import read_glb
from twintex.camera import OrthoCamera
from twintex.raster import barycentric_at, zbuffer

from . import log
from .template import weld


@dataclass
class Bust:
    """The welded bust in its own (non-metric) units; ``+Y`` up and ``+Z`` front after the orientation step."""

    positions: np.ndarray  # (n, 3) float64
    normals: np.ndarray  # (n, 3) unit
    colours: np.ndarray  # (n, 3) float32, sRGB 0..255 (mean of the seam copies)
    lab: np.ndarray  # (n, 3) float32 CIELAB of ``colours``
    faces: np.ndarray  # (m, 3) int64
    face_uv: np.ndarray  # (m, 3, 2) float32: glTF UV of the three corners of every face (v down)
    base_texture: np.ndarray  # (H, W, 3) uint8 sRGB base colour
    normal_texture: np.ndarray | None  # (H, W, 3) uint8 tangent-space normal map (glTF convention) or None
    info: dict = field(default_factory=dict)


def sample_texture(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest texel colours at glTF UVs (u right, v down, image row 0 on top)."""
    h, w = image.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(np.int64), 0, w - 1)
    y = np.clip((uv[:, 1] * h).astype(np.int64), 0, h - 1)
    return image[y, x]


def decode_image(data: bytes) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    import trimesh

    return np.asarray(trimesh.Trimesh(positions, faces, process=False).vertex_normals)


def weld_vertices(
    positions: np.ndarray, normals: np.ndarray, colours: np.ndarray, faces: np.ndarray, decimals: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge duplicate positions (UV seam copies): welded positions, normals, colours, faces without collapsed ones, and
    the index of the original face every kept face came from."""
    inverse, first = weld(positions, decimals)
    count = len(first)
    owners = np.bincount(inverse, minlength=count).astype(np.float64)
    n = np.zeros((count, 3))
    np.add.at(n, inverse, normals)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    c = np.stack([np.bincount(inverse, weights=colours[:, k], minlength=count) / owners for k in range(3)], axis=1)
    f = inverse[faces]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return positions[first], n, c.astype(np.float32), f[ok], np.flatnonzero(ok)


def load_bust(path) -> Bust:
    """Read the GLB: the largest triangle primitive with its base colour texture, node transforms baked, welded."""
    from flamehead.colour import to_lab

    scene = read_glb(str(path))
    if not scene.prims:
        raise ValueError(f"{path}: no triangle mesh")
    prim = max(scene.prims, key=lambda p: len(p.positions))
    if prim.uv is None or prim.material is None:
        raise ValueError(f"{path}: the bust needs UVs and a material with a base colour texture")
    info = scene.materials[prim.material].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if info is None:
        raise ValueError(f"{path}: the bust material has no baseColorTexture")
    image = scene.images[scene.textures[info["index"]]["source"]]
    if "data" not in image:
        raise ValueError(f"{path}: external textures are not supported")
    base = decode_image(image["data"])
    normal_info = scene.materials[prim.material].get("normalTexture")
    normal_image = scene.images[scene.textures[normal_info["index"]]["source"]] if normal_info else None
    normal_texture = decode_image(normal_image["data"]) if normal_image is not None and "data" in normal_image else None
    positions = prim.positions.astype(np.float64)
    faces = prim.indices.astype(np.int64)
    normals = prim.normals.astype(np.float64) if prim.normals is not None else vertex_normals(positions, faces)
    colours = sample_texture(base, prim.uv.astype(np.float64)).astype(np.float32)
    raw_min, raw_max = positions.min(0), positions.max(0)
    p, n, c, f, kept = weld_vertices(positions, normals, colours, faces)
    uv = prim.uv.astype(np.float32)
    return Bust(
        p,
        n,
        c,
        to_lab(c / 255.0).astype(np.float32),
        f,
        uv[faces[kept]],
        base,
        normal_texture,
        info={
            "file": Path(path).name,
            "raw_vertices": int(len(positions)),
            "welded_vertices": int(len(p)),
            "triangles": int(len(f)),
            "texture_size": [int(base.shape[1]), int(base.shape[0])],
            "has_normal_map": normal_texture is not None,
            "bbox_min": raw_min.tolist(),
            "bbox_max": raw_max.tolist(),
        },
    )


def rotated(bust: Bust, rotation: np.ndarray) -> Bust:
    return replace(bust, positions=bust.positions @ rotation.T, normals=bust.normals @ rotation.T)


def axis_rotations() -> list[np.ndarray]:
    """The 24 proper rotation matrices that permute and flip the axes, identity first."""
    mats = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for signs in np.ndindex(2, 2, 2):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs, strict=True)):
                m[row, col] = 1.0 - 2.0 * s
            if np.linalg.det(m) > 0:
                mats.append(m)
    mats.sort(key=lambda m: float(np.abs(m - np.eye(3)).sum()))
    return mats


# ------------------------------------------------------------------------------------------------ rendering
def vertex_colour_render(
    positions: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    colours: np.ndarray,
    camera: OrthoCamera,
    width: int,
    height: int,
    *,
    ss: int = 1,
    lights=None,
    spec: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Lit render with per-vertex sRGB colours (0..255); returns the image (uint8) and the depth buffer (``ss`` x size)."""
    import cv2
    from twinrefine.render import BG, DEFAULT_LIGHTS, shade
    from twintex.colorspace import linear_to_srgb, srgb_to_linear

    lights = lights or DEFAULT_LIGHTS
    p = camera.project(positions)
    p[:, 0] *= ss
    p[:, 1] *= ss
    depth, fid = zbuffer(p, faces, width * ss, height * ss)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    lam = barycentric_at(p[:, :2], faces, f, xs, ys)
    tri = faces[f]
    n = np.einsum("ij,ijk->ik", lam, normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.where(n @ camera.to_camera < 0, -1.0, 1.0)[:, None]
    ncam = np.stack([n @ camera.right, n @ camera.up, n @ camera.to_camera], axis=1)
    linear = srgb_to_linear(np.asarray(colours, np.float32) / 255.0)
    albedo = np.einsum("ij,ijk->ik", lam, linear[tri]).astype(np.float32)
    image = np.tile(srgb_to_linear(BG)[None, None, :], (height * ss, width * ss, 1)).astype(np.float32)
    image[ys, xs] = shade(ncam, albedo, lights, spec=spec)
    if ss > 1:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(linear_to_srgb(image) * 255), 0, 255).astype(np.uint8), depth


# --------------------------------------------------------------------------------------------------- landmarks
def detect_front_landmarks(
    bust: Bust, size: int = 1024, region: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """MediaPipe landmarks of the bust's front render, lifted to 3-D: ``(points (478, 3), valid (478,))`` or None.

    Two passes: the head (the face is small in a bust render), then a render framed on the face found in the first
    pass. ``region`` optionally restricts the render to a vertex mask. Depth comes from the z-buffer of the render.
    """
    from twinrefine.landmarks import detect_landmarks

    faces = bust.faces
    if region is not None:
        faces = faces[region[faces].all(1)]
    base = OrthoCamera.azimuth("front", 0.0)
    camera = base.fit_bounds(bust.positions[np.unique(faces)], size, size, margin=0.04)
    result = None
    for attempt in range(2):
        image, depth = vertex_colour_render(bust.positions, faces, bust.normals, bust.colours, camera, size, size)
        found = detect_landmarks(image, full_frame=True)
        if found is None:
            return result
        xy = found.xy
        xi = np.clip(np.rint(xy[:, 0] - 0.5).astype(int), 0, size - 1)
        yi = np.clip(np.rint(xy[:, 1] - 0.5).astype(int), 0, size - 1)
        d = depth[yi, xi]
        valid = np.isfinite(d)
        u = (xy[:, 0] - camera.tx) / (camera.scale * camera.aspect)
        v = (camera.ty - xy[:, 1]) / camera.scale
        points = u[:, None] * camera.right + v[:, None] * camera.up + np.where(valid, d, 0.0)[:, None] * camera.forward
        result = (points, valid)
        if attempt == 0:  # second pass: a render framed on the face found in the first one
            seen = xy[valid]
            if len(seen) < 100:
                return result
            lo, hi = seen.min(0), seen.max(0)
            half = float(max(hi - lo)) * 0.62
            centre = (lo + hi) / 2
            c_world = (centre[0] - camera.tx) / (camera.scale * camera.aspect) * camera.right + (
                camera.ty - centre[1]
            ) / camera.scale * camera.up
            extent = half / camera.scale
            focus = np.array(
                [c_world - extent * (camera.right + camera.up), c_world + extent * (camera.right + camera.up)]
            )
            camera = base.fit_bounds(focus, size, size, margin=0.0)
    return result


def trimmed_similarity(
    source: np.ndarray, target: np.ndarray, *, keep: float = 0.75, rounds: int = 6, band=(0.2, 3.0)
) -> dict:
    """Umeyama similarity ``target ~ s R source + t`` refit on the best ``keep`` fraction of the points."""
    selected = np.ones(len(source), bool)
    s, r, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(rounds):
        a, b = source[selected], target[selected]
        mu_a, mu_b = a.mean(0), b.mean(0)
        x, y = a - mu_a, b - mu_b
        u, d, vt = np.linalg.svd(y.T @ x / len(x))
        sign = np.ones(3)
        sign[-1] = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag(sign) @ vt
        s = float((d * sign).sum() / np.mean((x * x).sum(1)))
        t = mu_b - s * r @ mu_a
        residual = np.linalg.norm(s * source @ r.T + t - target, axis=1)
        selected = residual <= max(np.quantile(residual, keep), 1e-9)
    if not band[0] <= s <= band[1]:
        raise ValueError(f"The landmark similarity scale {s:.3f} is implausible (expected {band})")
    residual = np.linalg.norm(s * source @ r.T + t - target, axis=1)
    return {"scale": s, "rotation": r, "translation": t, "inliers": selected, "residual": residual}


def apply_similarity(points: np.ndarray, sim: dict) -> np.ndarray:
    return sim["scale"] * points @ sim["rotation"].T + sim["translation"]


# MediaPipe canonical landmarks that stay put under a smile, closed eyes and glasses: the nose bridge and base, the
# forehead and the upper face oval (temples). Nose wings, cheeks, jaw and lips move with the expression or hide behind
# the glasses frame, eyes and brows sit behind the glasses.
STABLE_FRONT = (6, 168, 197, 195, 5, 4, 1, 19, 94, 2)
STABLE_FRONT += (10, 109, 67, 103, 54, 21, 162, 338, 297, 332, 284, 251, 389, 8, 9, 151, 108, 337)


def landmark_correspondences(
    bust_points: np.ndarray, bust_valid: np.ndarray, twin_points: np.ndarray, twin_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matching (bust, twin) landmark points restricted to the stable subset; also the MediaPipe ids used."""
    lookup = {int(i): row for row, i in enumerate(twin_index)}
    ids = [i for i in STABLE_FRONT if i in lookup and bust_valid[i]]
    return (
        np.array([bust_points[i] for i in ids]),
        np.array([twin_points[lookup[i]] for i in ids]),
        np.array(ids),
    )


def find_orientation(bust: Bust, size: int = 640) -> np.ndarray:
    """The rotation (3 x 3, applied as ``points @ R.T``) that makes the bust +Y up and +Z front.

    The identity when the front render shows a face to the landmark detector (the case for Hunyuan3D's glTF, whose node
    rotation ``read_glb`` already baked), otherwise the first of the 24 axis rotations where it does.
    """
    for rotation in axis_rotations():
        found = detect_front_landmarks(rotated(bust, rotation), size=size)
        if found is not None and found[1].sum() > 300:
            points, valid = found
            # the face must look at the camera: the nose tip (landmark 1) is nearer than the face's median depth
            if valid[1] and points[1][2] >= np.median(points[valid][:, 2]):
                log(f"hy3d orientation: {np.rint(rotation).astype(int).tolist()}")
                return rotation
    raise ValueError("No axis-aligned orientation of the bust shows a face to the landmark detector")


# ------------------------------------------------------------------------------------------- ICP refinement
def small_rigid(
    source: np.ndarray, target: np.ndarray, normals: np.ndarray, weights: np.ndarray, rotation_ridge: float = 0.0
):
    """Linearised point-to-plane rigid motion ``(R, t)`` moving ``source`` towards ``target`` along ``normals``.

    ``source`` should be centred on the region of interest. ``rotation_ridge`` (relative to the total weight and the
    point spread) damps the rotation: a forehead patch and some side hair cannot determine the pitch and yaw of a head.
    """
    a = np.concatenate((np.cross(source, normals), normals), axis=1) * np.sqrt(weights)[:, None]
    b = np.einsum("ij,ij->i", target - source, normals) * np.sqrt(weights)
    normal_matrix = a.T @ a
    spread = float(np.mean((source * source).sum(1)))
    normal_matrix[:3, :3] += rotation_ridge * float(weights.sum()) * spread * np.eye(3)
    x = np.linalg.lstsq(normal_matrix + 1e-12 * np.eye(6), a.T @ b, rcond=None)[0]
    wx, wy, wz = x[:3]
    angle = float(np.linalg.norm(x[:3]))
    k = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    if angle < 1e-12:
        return np.eye(3), x[3:]
    r = np.eye(3) + np.sin(angle) / angle * k + (1 - np.cos(angle)) / angle**2 * (k @ k)
    return r, x[3:]


def refine_on_skin(
    points: np.ndarray,
    skin: np.ndarray,
    stubble: np.ndarray,
    surface,
    *,
    stubble_offset: float = 0.003,
    stubble_weight: float = 0.35,
    iterations: int = 12,
    keep: float = 0.85,
    samples: int = 4000,
    rotation_ridge: float = 2.0,
    seed: int = 5,
) -> dict:
    """Rigid ICP of the aligned bust onto the twin head: forehead / temple SKIN (offset 0) and the short side hair
    (``stubble``, offset ``stubble_offset`` for the hair thickness, lower weight). Returns ``{"R", "t", "report"}``
    with ``p' = p @ R.T + t``.

    ``points`` are the vertex positions (already carried by the landmark similarity); ``skin`` / ``stubble`` are index
    arrays. The motion is rigid on purpose, with a damped rotation: the scale comes from the landmark layout (a flat
    forehead patch cannot determine it) and the shell is deformed non-rigidly afterwards.
    """
    rng = np.random.default_rng(seed)

    def pick(ids):
        return ids if len(ids) <= samples else rng.choice(ids, samples, replace=False)

    skin, stubble = pick(skin), pick(stubble)
    centre = points[skin].mean(0)
    rot, shift = np.eye(3), np.zeros(3)  # p' = R (p - centre) + centre + shift
    history = []

    def moved(ids):
        return (points[ids] - centre) @ rot.T + centre + shift

    def stubble_gap():
        signed, *_ = surface.signed_distance(moved(stubble))
        return signed

    gap_before = stubble_gap() if len(stubble) else np.zeros(0)
    for _ in range(iterations):
        moved_skin, moved_hair = moved(skin), moved(stubble)
        cs, _, _, ns = surface.closest(moved_skin)
        source, target, normals = moved_skin, cs, ns
        weights = np.ones(len(skin))
        if len(stubble):
            ch, _, _, nh = surface.closest(moved_hair)
            source = np.vstack((moved_skin, moved_hair))
            target = np.vstack((cs, ch + nh * stubble_offset))
            normals = np.vstack((ns, nh))
            weights = np.concatenate((weights, np.full(len(stubble), stubble_weight)))
        gap = np.abs(np.einsum("ij,ij->i", source - target, normals))
        weights = weights * (gap <= max(np.quantile(gap, keep), 1e-6))
        dr, dt = small_rigid(source - centre, target - centre, normals, weights, rotation_ridge)
        rot, shift = dr @ rot, dr @ shift + dt
        history.append(float(np.average(gap, weights=np.maximum(weights, 1e-12))) * 1000)
    skin_gap, *_ = surface.signed_distance(moved(skin))
    gap_after = stubble_gap() if len(stubble) else np.zeros(0)
    translation = centre - rot @ centre + shift
    return {
        "R": rot,
        "t": translation,
        "report": {
            "iterations": iterations,
            "mean_plane_gap_mm": history,
            "skin_signed_distance_mm": {
                "mean": float(skin_gap.mean() * 1000),
                "std": float(skin_gap.std() * 1000),
                "p95_abs": float(np.percentile(np.abs(skin_gap), 95) * 1000),
            },
            "side_hair_signed_distance_mm": {
                "before_median": float(np.median(gap_before) * 1000) if len(gap_before) else None,
                "after_median": float(np.median(gap_after) * 1000) if len(gap_after) else None,
            },
            "shift_mm": (shift * 1000).tolist(),
            "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))),
            "skin_points": int(len(skin)),
            "stubble_points": int(len(stubble)),
        },
    }


@dataclass
class Alignment:
    """The bust carried into the twin frame (metres): similarity from the landmarks, then a small rigid ICP."""

    positions: np.ndarray
    normals: np.ndarray
    scale: float
    rotation: np.ndarray  # total rotation (bust -> twin)
    translation: np.ndarray  # total translation (applied after the rotation and the scale)
    report: dict


def align_bust(
    bust: Bust, twin_points: np.ndarray, twin_index: np.ndarray, surface, frame, *, size: int = 1024
) -> Alignment:
    """Metric, upright and registered: landmark similarity, then rigid ICP of the forehead skin / side hair on the twin head.

    ``twin_points`` / ``twin_index`` are the face-map landmarks on the deformed twin head (468 points and their MediaPipe
    ids), ``surface`` its ``HeadSurface`` and ``frame`` its ``HeadFrame`` (skull centre, eye line).
    """
    rotation0 = find_orientation(bust)
    oriented = rotated(bust, rotation0)
    head = oriented.positions[:, 1] > oriented.positions[:, 1].max() - 0.55 * np.ptp(oriented.positions[:, 1])
    found = detect_front_landmarks(oriented, size=size, region=head)
    if found is None:
        raise ValueError("MediaPipe found no face on the bust's front render")
    points, valid = found
    source, target, ids = landmark_correspondences(points, valid, twin_points, twin_index)
    if len(ids) < 12:
        raise ValueError(f"Only {len(ids)} usable face landmarks on the bust")
    sim = trimmed_similarity(source, target)
    moved = apply_similarity(oriented.positions, sim)
    normals = oriented.normals @ sim["rotation"].T
    # ICP on forehead / temple skin and the short hair of the sides (lightness: skin >> hair)
    az = np.degrees(np.arctan2(moved[:, 0] - frame.x0, moved[:, 2] - frame.zc))
    dy = moved[:, 1] - frame.eye_y
    lab = bust.lab
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    skin = np.flatnonzero(
        (lab[:, 0] > 58) & (chroma > 8) & (np.abs(az) < 62) & (dy > 0.02) & (dy < 0.075) & (moved[:, 2] > 0.05)
    )
    stubble = np.flatnonzero((lab[:, 0] < 40) & (np.abs(az) > 62) & (np.abs(az) < 118) & (dy > 0.015) & (dy < 0.05))
    if len(skin) >= 60:
        icp = refine_on_skin(moved, skin, stubble, surface)
        r_icp, t_icp = icp["R"], icp["t"]
        icp_report = icp["report"]
    else:  # a fringe over the forehead: the landmark similarity stands
        r_icp, t_icp = np.eye(3), np.zeros(3)
        icp_report = {"skipped": f"only {len(skin)} forehead skin vertices (needs 60): landmark similarity only"}
    moved = moved @ r_icp.T + t_icp
    normals = normals @ r_icp.T
    rotation = r_icp @ sim["rotation"] @ rotation0
    translation = r_icp @ sim["translation"] + t_icp
    residual = sim["residual"] * 1000
    return Alignment(
        moved,
        normals,
        sim["scale"],
        rotation,
        translation,
        {
            "orientation_rotation": np.rint(rotation0).astype(int).tolist(),
            "landmarks_used": int(len(ids)),
            "landmark_inliers": int(sim["inliers"].sum()),
            "landmark_residual_mm": {
                "median": float(np.median(residual)),
                "p90": float(np.percentile(residual, 90)),
                "inlier_mean": float(residual[sim["inliers"]].mean()),
            },
            "scale_to_metres": sim["scale"],
            "similarity_rotation_deg": float(
                np.degrees(np.arccos(np.clip((np.trace(sim["rotation"]) - 1) / 2, -1, 1)))
            ),
            "similarity_translation_m": sim["translation"].tolist(),
            "icp": icp_report,
            "total_rotation": rotation.tolist(),
            "total_translation_m": translation.tolist(),
            "formula": "p_twin = scale * (R_total @ p_bust_world) + t_total  (R_total includes the ICP and orientation)",
        },
    )
