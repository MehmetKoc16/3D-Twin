"""Hide dark shoe-lining triangles that show through on the outside of a fitted shoe GLB.

Evidence (lead, running app, 2026-10-19/20, ``sf-converse-style.glb``): after fitting, part of the dark inner lining
lies a few mm OUTSIDE the red canvas near the heel and faces outward, so it shows as irregular dark patches (a
single-sided material changed nothing). Deleting every dark-texel triangle removed the patches but also the black
sole stripes, so a colour rule alone is wrong. A dark triangle is hidden only when a light, roughly parallel surface
lies directly BEHIND it (towards the shoe centre), i.e. it is a stacked layer on top of the canvas:

1. ``classify_dark``: median texture colour of 4 UV samples (centroid + 3 corners) is dark and unsaturated.
2. Method B ``proximity_covered`` (main pass): 7 samples per dark triangle (corners, edge midpoints, centroid); a
   sample is covered when a light sample within ``radius_mm`` is parallel (``|n_dark . n_light| >= min_parallel``)
   and lies on the inner side (``(q - p) . n_out < -inner_margin``), ``n_out`` = dark normal oriented away from the
   shoe's vertex centroid. Cull when ``>= min_covered`` of 7 samples are covered.
3. Method A ``ray_covered`` (second pass, on dark triangles B did not cull): 4 rays (centroid + centroid/corner
   midpoints) cast inward along ``-n_out``; the nearest light triangle hit within ``max_gap_mm`` must be parallel
   (``>= min_parallel``). Cull when ``>= min_hits`` rays are accepted.
4. ``sole_mask``: triangles whose centroid is in the bottom ``sole_mm`` of their shoe (the rigid sole) are never
   culled, which keeps the black sole stripes.

Chaining B (radius 8 mm) then A (gap 10 mm, 1 hit) with dark luminance 130 and the 35 mm sole guard removed every
patch in A-pose and walk with the sole stripes intact. Each pass alone removed only part of the patches.

Only the index buffer changes: a culled triangle becomes the degenerate ``(a, a, a)``. Vertex count, attributes, JSON
chunk and file length stay identical because the app binds every vertex to the body through a separate ``.bind.bin``.
Assumptions (same as the experiment scripts): one mesh, one triangle primitive, one material whose baseColorTexture
is embedded in the GLB, +Y up, a pair is mirrored at x = 0 (split by triangle-centroid x sign).
"""
from __future__ import annotations

import argparse
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

# Defaults: the parameter set the lead verified in the app on 2026-10-20.
DEFAULT_DARK_LUM = 130.0  # mean RGB below this is "dark"; 90 missed the lighter grey lining texels
DEFAULT_DARK_SAT = 40.0  # max-min channel spread below this; keeps saturated red canvas (lum ~93) light
DEFAULT_RADIUS_MM = 8.0  # method B search radius; 4 mm missed lining lying farther outside the canvas
DEFAULT_MIN_PARALLEL = 0.6  # method B: stacked layers are roughly parallel (|cos| >= 0.6, about 53 degrees)
DEFAULT_INNER_MARGIN_MM = 0.2  # method B: covering light sample must be clearly behind, not coplanar
DEFAULT_MIN_COVERED = 4  # method B: majority of the 7 samples must be covered
DEFAULT_MAX_GAP_MM = 10.0  # method A: farthest backing canvas accepted along the inward ray
DEFAULT_RAY_MIN_PARALLEL = 0.5  # method A: looser parallel test, the ray already proves "directly behind"
DEFAULT_MIN_HITS = 1  # method A: one accepted ray of 4 is enough (2 left patches behind)
DEFAULT_RAY_OFFSET_MM = 0.05  # method A: start rays slightly inside to avoid hitting coplanar neighbours
DEFAULT_SOLE_MM = 35.0  # rigid sole height of the shoe (importer's --rigid-sole-height-mm); never culled

SAMPLES_PER_TRIANGLE = 7  # corners, edge midpoints, centroid (method B)
RAYS_PER_TRIANGLE = 4  # centroid + 3 centroid/corner midpoints (method A)
BARY_EPS = 1e-6  # barycentric tolerance so rays on a shared edge still hit
GAP_EPS_M = 1e-6  # 1 micrometre tolerance on the max-gap test
DET_EPS = 1e-12  # Moeller-Trumbore: rays parallel to the triangle plane never hit
NORMAL_EPS = 1e-12  # zero-area triangles get a zero normal (they never pass a parallel test)
PAIR_CHUNK = 1_000_000  # point/triangle pairs evaluated per batch to bound peak memory

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_INDEX_TYPES = (5121, 5123, 5125)


@dataclass
class GlbMesh:
    """The single textured primitive of a GLB plus the raw bytes needed to rewrite its index buffer in place."""

    data: bytearray
    positions: np.ndarray  # (N, 3) float64, metres
    uvs: np.ndarray  # (N, 2) float64
    triangles: np.ndarray  # (M, 3) int64
    index_offset: int  # byte offset of the first index in ``data``
    index_dtype: np.dtype  # little-endian on-disk index type
    texture: np.ndarray  # (H, W, 3) float64 base colour, 0..255


# --------------------------------------------------------------------------------------------------- GLB reading


def read_glb_mesh(path: str | Path) -> GlbMesh:
    """Parse a GLB with one mesh, one triangle primitive and one material with an embedded baseColorTexture."""
    data = bytearray(Path(path).read_bytes())
    if len(data) < 20 or data[:4] != _GLB_MAGIC:
        raise ValueError(f"{path}: not a binary glTF (GLB) file")
    chunks: list[tuple[int, int, int]] = []
    off = 12
    while off + 8 <= len(data):
        length, ctype = struct.unpack_from("<II", data, off)
        chunks.append((ctype, off + 8, length))
        off += 8 + length
    if len(chunks) < 2 or chunks[0][0] != _CHUNK_JSON or chunks[1][0] != _CHUNK_BIN:
        raise ValueError(f"{path}: GLB must have a JSON chunk followed by a BIN chunk")
    js: dict[str, Any] = json.loads(bytes(data[chunks[0][1] : chunks[0][1] + chunks[0][2]]))
    bin_start, bin_len = chunks[1][1], chunks[1][2]

    meshes = js.get("meshes", [])
    if len(meshes) != 1 or len(meshes[0].get("primitives", [])) != 1:
        raise ValueError(f"{path}: expected exactly one mesh with one primitive")
    prim = meshes[0]["primitives"][0]
    if prim.get("mode", 4) != 4:
        raise ValueError(f"{path}: primitive mode must be TRIANGLES (4)")
    attrs = prim.get("attributes", {})
    if "POSITION" not in attrs or "TEXCOORD_0" not in attrs or "indices" not in prim:
        raise ValueError(f"{path}: primitive needs POSITION, TEXCOORD_0 and indices")

    def view_start(view_index: int) -> tuple[dict[str, Any], int]:
        view = js["bufferViews"][view_index]
        if view.get("buffer", 0) != 0:
            raise ValueError(f"{path}: bufferView {view_index} is not in the GLB BIN chunk")
        return view, bin_start + view.get("byteOffset", 0)

    def accessor(index: int, kind: str, types: tuple[int, ...]) -> tuple[np.ndarray, int, np.dtype]:
        acc = js["accessors"][index]
        if acc.get("type") != kind or acc.get("componentType") not in types:
            raise ValueError(f"{path}: accessor {index} must be {kind} of componentType {types}")
        if "sparse" in acc or "bufferView" not in acc:
            raise ValueError(f"{path}: accessor {index} is sparse or has no bufferView (not supported)")
        n = _COMPONENTS[kind]
        dtype = _DTYPES[acc["componentType"]]
        view, start = view_start(acc["bufferView"])
        stride = view.get("byteStride")
        if stride is not None and stride != n * dtype.itemsize:
            raise ValueError(f"{path}: accessor {index} is interleaved (byteStride {stride}, not supported)")
        start += acc.get("byteOffset", 0)
        count = int(acc["count"])
        if start + count * n * dtype.itemsize > bin_start + bin_len:
            raise ValueError(f"{path}: accessor {index} runs past the BIN chunk")
        arr = np.frombuffer(bytes(data[start : start + count * n * dtype.itemsize]), dtype=dtype)
        return (arr.reshape(-1, n) if n > 1 else arr), start, dtype

    positions, _, _ = accessor(attrs["POSITION"], "VEC3", (5126,))
    uvs, _, _ = accessor(attrs["TEXCOORD_0"], "VEC2", (5126,))
    indices, index_offset, index_dtype = accessor(prim["indices"], "SCALAR", _INDEX_TYPES)
    if indices.size % 3:
        raise ValueError(f"{path}: index count {indices.size} is not a multiple of 3")
    if indices.size and int(indices.max()) >= len(positions):
        raise ValueError(f"{path}: index out of range of {len(positions)} vertices")

    materials = js.get("materials", [])
    if len(materials) != 1 or prim.get("material", 0) != 0:
        raise ValueError(f"{path}: expected exactly one material used by the primitive")
    tex_info = materials[0].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if tex_info is None:
        raise ValueError(f"{path}: material has no baseColorTexture")
    if tex_info.get("texCoord", 0) != 0:
        raise ValueError(f"{path}: baseColorTexture must use TEXCOORD_0")
    image = js["images"][js["textures"][tex_info["index"]]["source"]]
    if "bufferView" not in image:
        raise ValueError(f"{path}: baseColorTexture image is not embedded in the GLB")
    view, img_start = view_start(image["bufferView"])
    img_bytes = bytes(data[img_start : img_start + view["byteLength"]])
    texture = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"), dtype=np.float64)

    return GlbMesh(
        data=data,
        positions=positions.astype(np.float64),
        uvs=uvs.astype(np.float64),
        triangles=indices.astype(np.int64).reshape(-1, 3),
        index_offset=index_offset,
        index_dtype=index_dtype,
        texture=texture,
    )


# --------------------------------------------------------------------------------------------- classification


def _sample_texture(uv: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Nearest-texel lookup with wrapping; u or v of exactly 1.0 maps to the last texel instead of wrapping to 0."""
    h, w, _ = texture.shape
    u, v = uv[..., 0], uv[..., 1]
    u_c = np.where((u > 0.9999) & (u <= 1.0001), 1.0 - 1e-6, u % 1.0)
    v_c = np.where((v > 0.9999) & (v <= 1.0001), 1.0 - 1e-6, v % 1.0)
    x = np.clip(u_c * w, 0, w - 1).astype(np.int64)
    y = np.clip(v_c * h, 0, h - 1).astype(np.int64)
    return texture[y, x]


def classify_dark(
    triangles: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
    dark_lum: float = DEFAULT_DARK_LUM,
    dark_sat: float = DEFAULT_DARK_SAT,
) -> np.ndarray:
    """Dark when the median of the 4 texture samples (UV centroid + 3 corner UVs) is dark and unsaturated."""
    corner_uv = uvs[triangles]  # (M, 3, 2)
    samples_uv = np.concatenate([corner_uv.mean(axis=1, keepdims=True), corner_uv], axis=1)  # (M, 4, 2)
    med = np.median(_sample_texture(samples_uv, texture), axis=1)  # (M, 3)
    lum = med.mean(axis=1)
    sat = med.max(axis=1) - med.min(axis=1)
    return (lum < dark_lum) & (sat < dark_sat)


# ------------------------------------------------------------------------------------------------ geometry helpers


def split_pair(positions: np.ndarray, triangles: np.ndarray) -> list[np.ndarray]:
    """One triangle mask per shoe by centroid x sign (pair mirrored at x = 0); empty sides are omitted."""
    cx = positions[triangles].mean(axis=1)[:, 0]
    return [mask for mask in (cx < 0.0, cx >= 0.0) if mask.any()]


def _face_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    corners = positions[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), NORMAL_EPS)


def _shoe_center(positions: np.ndarray, triangles: np.ndarray, shoe: np.ndarray) -> np.ndarray:
    """Vertex centroid of one shoe (unique vertices of its triangles)."""
    return positions[np.unique(triangles[shoe])].mean(axis=0)


def _orient_out(normals: np.ndarray, from_center: np.ndarray) -> np.ndarray:
    """Flip normals so they point away from the shoe centre."""
    dots = np.sum(normals * from_center, axis=-1, keepdims=True)
    return np.where(dots >= 0.0, normals, -normals)


def _seven_samples(corners: np.ndarray) -> np.ndarray:
    """(K, 3, 3) corners -> (K, 7, 3): p0, p1, p2, m01, m12, m20, centroid."""
    p0, p1, p2 = corners[:, 0], corners[:, 1], corners[:, 2]
    return np.stack([p0, p1, p2, 0.5 * (p0 + p1), 0.5 * (p1 + p2), 0.5 * (p2 + p0), (p0 + p1 + p2) / 3.0], axis=1)


def _longest_edge(corners: np.ndarray) -> float:
    if len(corners) == 0:
        return 0.0
    edges = corners[:, [1, 2, 0]] - corners
    return float(np.linalg.norm(edges, axis=2).max())


def _pairs_within(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """All (i, j) with |a_i - b_j| <= radius, as two int64 arrays (C-level k-d tree join, no Python loop)."""
    if len(a) == 0 or len(b) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    res = cKDTree(a).sparse_distance_matrix(cKDTree(b), radius, output_type="ndarray")
    return res["i"].astype(np.int64), res["j"].astype(np.int64)


# -------------------------------------------------------------------------------------------- method B (proximity)


def _proximity_pass(
    positions: np.ndarray,
    triangles: np.ndarray,
    dark: np.ndarray,
    radius_mm: float,
    min_parallel: float,
    inner_margin_mm: float,
    min_covered: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Method B; returns (covered mask, gap in mm from centroid to nearest covering light sample, NaN if not)."""
    if not 1 <= min_covered <= SAMPLES_PER_TRIANGLE:
        raise ValueError(f"min_covered must be 1..{SAMPLES_PER_TRIANGLE}, got {min_covered}")
    m = len(triangles)
    covered = np.zeros(m, dtype=bool)
    gaps_mm = np.full(m, np.nan)
    normals = _face_normals(positions, triangles)
    radius = radius_mm * 1e-3
    margin = inner_margin_mm * 1e-3
    k7 = SAMPLES_PER_TRIANGLE

    for shoe in split_pair(positions, triangles):
        dark_idx = np.flatnonzero(shoe & dark)
        light_idx = np.flatnonzero(shoe & ~dark)
        if dark_idx.size == 0 or light_idx.size == 0:
            continue
        center = _shoe_center(positions, triangles, shoe)

        light_pts = _seven_samples(positions[triangles[light_idx]]).reshape(-1, 3)
        light_n = np.repeat(normals[light_idx], k7, axis=0)
        dark_samples = _seven_samples(positions[triangles[dark_idx]])  # (K, 7, 3)
        dark_pts = dark_samples.reshape(-1, 3)
        dark_n = np.repeat(normals[dark_idx], k7, axis=0)
        n_out = _orient_out(dark_n, dark_pts - center)

        pi, qi = _pairs_within(dark_pts, light_pts, radius)
        sample_covered = np.zeros(len(dark_pts), dtype=bool)
        k = len(dark_idx)
        nearest = np.full(k, np.inf)
        for s in range(0, pi.size, PAIR_CHUNK):
            p, q = pi[s : s + PAIR_CHUNK], qi[s : s + PAIR_CHUNK]
            parallel = np.abs(np.sum(dark_n[p] * light_n[q], axis=1)) >= min_parallel
            inner = np.sum((light_pts[q] - dark_pts[p]) * n_out[p], axis=1) < -margin
            valid = parallel & inner
            if not valid.any():
                continue
            pv, qv = p[valid], q[valid]
            sample_covered[pv] = True
            tri_local = pv // k7
            dist = np.linalg.norm(light_pts[qv] - dark_samples[tri_local, k7 - 1], axis=1)
            np.minimum.at(nearest, tri_local, dist)

        tri_covered = sample_covered.reshape(k, k7).sum(axis=1) >= min_covered
        covered[dark_idx[tri_covered]] = True
        gaps_mm[dark_idx[tri_covered]] = nearest[tri_covered] * 1000.0
    return covered, gaps_mm


def proximity_covered(
    positions: np.ndarray,
    triangles: np.ndarray,
    dark: np.ndarray,
    *,
    radius_mm: float = DEFAULT_RADIUS_MM,
    min_parallel: float = DEFAULT_MIN_PARALLEL,
    inner_margin_mm: float = DEFAULT_INNER_MARGIN_MM,
    min_covered: int = DEFAULT_MIN_COVERED,
) -> np.ndarray:
    """Method B: dark triangles with a parallel light layer within ``radius_mm`` behind them (per shoe)."""
    return _proximity_pass(positions, triangles, dark, radius_mm, min_parallel, inner_margin_mm, min_covered)[0]


# ------------------------------------------------------------------------------------------------ method A (rays)


def _ray_pass(
    positions: np.ndarray,
    triangles: np.ndarray,
    dark: np.ndarray,
    candidates: np.ndarray,
    max_gap_mm: float,
    min_parallel: float,
    min_hits: int,
    ray_offset_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Method A; returns (covered mask, gap in mm: centroid ray if accepted, else mean of accepted rays)."""
    if not 1 <= min_hits <= RAYS_PER_TRIANGLE:
        raise ValueError(f"min_hits must be 1..{RAYS_PER_TRIANGLE}, got {min_hits}")
    m = len(triangles)
    covered = np.zeros(m, dtype=bool)
    gaps_mm = np.full(m, np.nan)
    normals = _face_normals(positions, triangles)
    centroids = positions[triangles].mean(axis=1)
    offset = ray_offset_mm * 1e-3
    max_gap = max_gap_mm * 1e-3

    for shoe in split_pair(positions, triangles):
        cand_idx = np.flatnonzero(shoe & dark & candidates)
        light_idx = np.flatnonzero(shoe & ~dark)
        if cand_idx.size == 0 or light_idx.size == 0:
            continue
        center = _shoe_center(positions, triangles, shoe)

        n_cand = normals[cand_idx]
        c_cand = centroids[cand_idx]
        n_out = _orient_out(n_cand, c_cand - center)
        cand_corners = positions[triangles[cand_idx]]  # (K, 3, 3)
        ray_pts = np.concatenate([c_cand[:, None], 0.5 * (c_cand[:, None] + cand_corners)], axis=1)  # (K, 4, 3)
        origins = ray_pts - offset * n_out[:, None]
        dirs = -n_out

        light_corners = positions[triangles[light_idx]]  # (L, 3, 3)
        # Any light triangle a ray can hit within max_gap has its centroid inside this radius.
        radius = max_gap + 2.0 * _longest_edge(light_corners) + _longest_edge(cand_corners)
        pair_d, pair_l = _pairs_within(c_cand, centroids[light_idx], radius)

        k = len(cand_idx)
        best_t = np.full((k, RAYS_PER_TRIANGLE), np.inf)
        best_l = np.full((k, RAYS_PER_TRIANGLE), -1, dtype=np.int64)
        for s in range(0, pair_d.size, PAIR_CHUNK):
            d, lt = pair_d[s : s + PAIR_CHUNK], pair_l[s : s + PAIR_CHUNK]
            ray_dir = dirs[d]
            v0 = light_corners[lt, 0]
            e1 = light_corners[lt, 1] - v0
            e2 = light_corners[lt, 2] - v0
            pvec = np.cross(ray_dir, e2)
            det = np.sum(e1 * pvec, axis=1)
            ok = np.abs(det) > DET_EPS
            inv_det = np.zeros_like(det)
            inv_det[ok] = 1.0 / det[ok]
            for r in range(RAYS_PER_TRIANGLE):
                tvec = origins[d, r] - v0
                u = np.sum(tvec * pvec, axis=1) * inv_det
                qvec = np.cross(tvec, e1)
                v = np.sum(ray_dir * qvec, axis=1) * inv_det
                t = np.sum(e2 * qvec, axis=1) * inv_det
                hit = (
                    ok
                    & (u >= -BARY_EPS)
                    & (v >= -BARY_EPS)
                    & (u + v <= 1.0 + BARY_EPS)
                    & (t > 0.0)
                    & (t + offset <= max_gap + GAP_EPS_M)
                )
                if not hit.any():
                    continue
                hd, hl, ht = d[hit], lt[hit], t[hit]
                order = np.lexsort((ht, hd))  # nearest hit first within each ray
                hd, hl, ht = hd[order], hl[order], ht[order]
                first = np.unique(hd, return_index=True)[1]
                bd, bl, bt = hd[first], hl[first], ht[first]
                better = bt < best_t[bd, r]
                best_t[bd[better], r] = bt[better]
                best_l[bd[better], r] = bl[better]

        has_hit = best_l >= 0
        n_hit = normals[light_idx][np.where(has_hit, best_l, 0)]  # (K, 4, 3)
        parallel = np.abs(np.sum(n_cand[:, None] * n_hit, axis=2)) >= min_parallel
        accepted = has_hit & parallel
        n_acc = accepted.sum(axis=1)
        tri_covered = n_acc >= min_hits

        ray_gap = np.where(accepted, best_t + offset, 0.0)
        mean_gap = np.where(n_acc > 0, ray_gap.sum(axis=1) / np.maximum(n_acc, 1), np.nan)
        gap = np.where(accepted[:, 0], ray_gap[:, 0], mean_gap)
        covered[cand_idx[tri_covered]] = True
        gaps_mm[cand_idx[tri_covered]] = gap[tri_covered] * 1000.0
    return covered, gaps_mm


def ray_covered(
    positions: np.ndarray,
    triangles: np.ndarray,
    dark: np.ndarray,
    candidates: np.ndarray,
    *,
    max_gap_mm: float = DEFAULT_MAX_GAP_MM,
    min_parallel: float = DEFAULT_RAY_MIN_PARALLEL,
    min_hits: int = DEFAULT_MIN_HITS,
    ray_offset_mm: float = DEFAULT_RAY_OFFSET_MM,
) -> np.ndarray:
    """Method A: dark ``candidates`` whose inward rays hit a parallel light triangle within ``max_gap_mm``."""
    return _ray_pass(positions, triangles, dark, candidates, max_gap_mm, min_parallel, min_hits, ray_offset_mm)[0]


# ------------------------------------------------------------------------------------------------------ sole guard


def sole_mask(positions: np.ndarray, triangles: np.ndarray, sole_mm: float = DEFAULT_SOLE_MM) -> np.ndarray:
    """Triangles whose centroid lies less than ``sole_mm`` above their shoe's lowest vertex (+Y up)."""
    cy = positions[triangles].mean(axis=1)[:, 1]
    mask = np.zeros(len(triangles), dtype=bool)
    for shoe in split_pair(positions, triangles):
        floor_y = positions[np.unique(triangles[shoe]), 1].min()
        mask |= shoe & (cy < floor_y + sole_mm * 1e-3)
    return mask


# -------------------------------------------------------------------------------------------------------- pipeline


def _gap_stats(gaps_mm: np.ndarray) -> dict[str, float | int]:
    g = gaps_mm[np.isfinite(gaps_mm)]
    if g.size == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "count": int(g.size),
        "min": round(float(g.min()), 4),
        "max": round(float(g.max()), 4),
        "mean": round(float(g.mean()), 4),
        "median": round(float(np.median(g)), 4),
    }


def cull_lining(
    src: str | Path,
    dst: str | Path,
    *,
    dark_lum: float = DEFAULT_DARK_LUM,
    dark_sat: float = DEFAULT_DARK_SAT,
    radius_mm: float = DEFAULT_RADIUS_MM,
    min_parallel: float = DEFAULT_MIN_PARALLEL,
    inner_margin_mm: float = DEFAULT_INNER_MARGIN_MM,
    min_covered: int = DEFAULT_MIN_COVERED,
    max_gap_mm: float = DEFAULT_MAX_GAP_MM,
    ray_min_parallel: float = DEFAULT_RAY_MIN_PARALLEL,
    min_hits: int = DEFAULT_MIN_HITS,
    ray_offset_mm: float = DEFAULT_RAY_OFFSET_MM,
    sole_mm: float = DEFAULT_SOLE_MM,
) -> dict[str, Any]:
    """Run B, then A on what B left, drop sole triangles, write ``dst`` with only the index buffer changed.

    Report counts ``culled_proximity`` / ``culled_ray`` are after the sole guard and sum to ``culled_total``;
    ``protected_sole`` counts triangles either pass would have culled but the sole guard kept.
    """
    mesh = read_glb_mesh(src)
    pos, tri = mesh.positions, mesh.triangles
    dark = classify_dark(tri, mesh.uvs, mesh.texture, dark_lum, dark_sat)
    by_prox, prox_gap = _proximity_pass(pos, tri, dark, radius_mm, min_parallel, inner_margin_mm, min_covered)
    by_ray, ray_gap = _ray_pass(pos, tri, dark, dark & ~by_prox, max_gap_mm, ray_min_parallel, min_hits, ray_offset_mm)
    sole = sole_mask(pos, tri, sole_mm)
    raw = by_prox | by_ray
    culled = raw & ~sole

    new_tri = tri.copy()
    new_tri[culled, 1] = new_tri[culled, 0]
    new_tri[culled, 2] = new_tri[culled, 0]
    out = mesh.data
    encoded = new_tri.astype(mesh.index_dtype).tobytes()
    out[mesh.index_offset : mesh.index_offset + len(encoded)] = encoded
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(bytes(out))

    prox_final = by_prox & culled
    ray_final = by_ray & culled
    return {
        "triangles": int(len(tri)),
        "dark": int(dark.sum()),
        "culled_proximity": int(prox_final.sum()),
        "culled_ray": int(ray_final.sum()),
        "protected_sole": int((raw & sole).sum()),
        "culled_total": int(culled.sum()),
        "gap_mm": {
            "proximity": _gap_stats(prox_gap[prox_final]),
            "ray": _gap_stats(ray_gap[ray_final]),
        },
        "params": {
            "dark_lum": dark_lum,
            "dark_sat": dark_sat,
            "radius_mm": radius_mm,
            "min_parallel": min_parallel,
            "inner_margin_mm": inner_margin_mm,
            "min_covered": min_covered,
            "max_gap_mm": max_gap_mm,
            "ray_min_parallel": ray_min_parallel,
            "min_hits": min_hits,
            "ray_offset_mm": ray_offset_mm,
            "sole_mm": sole_mm,
        },
    }


# ------------------------------------------------------------------------------------------------------------- CLI


def parser() -> argparse.ArgumentParser:
    """Command line parser for lining_cull."""
    p = argparse.ArgumentParser(
        description="Hide dark lining triangles that show through on the outside of a fitted shoe GLB."
    )
    p.add_argument("src", type=Path, help="Source GLB (fitted shoe pair)")
    p.add_argument("dst", type=Path, help="Destination GLB (same bytes except culled triangle indices)")
    p.add_argument("--report", type=Path, default=None, help="Write the JSON report here")
    p.add_argument("--dark-lum", type=float, default=DEFAULT_DARK_LUM, help="dark: mean RGB below this")
    p.add_argument("--dark-sat", type=float, default=DEFAULT_DARK_SAT, help="dark: RGB spread below this")
    p.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM, help="method B search radius")
    p.add_argument("--min-parallel", type=float, default=DEFAULT_MIN_PARALLEL, help="method B |cos| minimum")
    p.add_argument("--inner-margin-mm", type=float, default=DEFAULT_INNER_MARGIN_MM, help="method B inner margin")
    p.add_argument("--min-covered", type=int, default=DEFAULT_MIN_COVERED, help="method B covered samples of 7")
    p.add_argument("--max-gap-mm", type=float, default=DEFAULT_MAX_GAP_MM, help="method A max ray gap")
    p.add_argument("--ray-min-parallel", type=float, default=DEFAULT_RAY_MIN_PARALLEL, help="method A |cos| minimum")
    p.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS, help="method A accepted rays of 4")
    p.add_argument("--ray-offset-mm", type=float, default=DEFAULT_RAY_OFFSET_MM, help="method A ray start offset")
    p.add_argument("--sole-mm", type=float, default=DEFAULT_SOLE_MM, help="never cull the bottom N mm of a shoe")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: python -m garmentimport.lining_cull SRC DST [--report JSON] [params]."""
    args = parser().parse_args(argv)
    report = cull_lining(
        args.src,
        args.dst,
        dark_lum=args.dark_lum,
        dark_sat=args.dark_sat,
        radius_mm=args.radius_mm,
        min_parallel=args.min_parallel,
        inner_margin_mm=args.inner_margin_mm,
        min_covered=args.min_covered,
        max_gap_mm=args.max_gap_mm,
        ray_min_parallel=args.ray_min_parallel,
        min_hits=args.min_hits,
        ray_offset_mm=args.ray_offset_mm,
        sole_mm=args.sole_mm,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"triangles {report['triangles']} dark {report['dark']} culled {report['culled_total']} "
        f"(proximity {report['culled_proximity']}, ray {report['culled_ray']}) "
        f"protected_sole {report['protected_sole']}"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
