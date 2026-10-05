"""The hair shell cut out of the aligned bust: a regular equal-area remesh, its UV atlas and the texture bake.

The shell is the OUTER hair surface of the Hunyuan3D bust (open, a few hundred thousand triangles) turned into a game-ready
mesh. Direct quadric decimation of the noisy bust previously produced folds and non-manifold edges. Preserve the surface
repair by radial resampling first, then apply quadric reduction and xatlas UV packing. The hair cap is star-shaped from the
middle of the head, so it is a height field over the sphere of directions. The directions are mapped to the plane by the
Lambert azimuthal equal-area projection centred on the vertical axis, a square grid in that plane gives a regular,
manifold mesh of uniform quality, and every grid node takes the
outermost surface of the bust along its direction (the silhouette is kept; a forward-leaning brim loses its few millimetre
underside). The hair edge is NOT cut in the geometry: the mesh extends a margin beyond the hairline over the skin and the
alpha channel of the colour texture cuts the shell exactly at the hairline (a noisy ramp of a few millimetres).

Textures are baked from the original high-poly surface and ITS textures (sampled bilinearly through the original UVs, so the
strand detail of the 4096 px textures survives, not just the vertex colours):

* **colour** (sRGB, 2048 px), bled outwards so neither mipmaps nor bilinear taps mix in skin or unrelated colours;
* **alpha**: 255 inside the hair region, a ramp of ``fringe`` millimetres at its edge, 0 beyond (``MASK`` 0.5);
* **normal map** (tangent space, glTF convention: +X right, +Y up the image, +Z out): the original normal map and the
  relief of the high-poly surface, seen from the low-poly one.

Everything works in the aligned frame of the bust (metres, +Y up, +Z front); ``shellfit`` then moves the shell onto the
twin head (the baked textures stay valid: they live in the UV space).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
from scipy.spatial import cKDTree
from twintex.raster import rasterize_uv

MM = 1e-3
INT_PARAMS = ("target_triangles", "texture_size", "padding", "seed")


@dataclass(frozen=True)
class ShellParams:
    """Tunables of the shell (millimetres unless noted)."""

    target_triangles: int = 24000
    texture_size: int = 2048
    padding: int = 6  # texels the charts are bled outwards
    margin: float = 10.0  # the mesh extends this far beyond the hairline (the alpha channel cuts the shell)
    sample_spacing: float = 0.25  # spacing of the dense samples of the high-poly surface that the bake looks up
    fringe: float = 2.5  # width of the alpha ramp at the hairline
    fringe_noise: float = 0.7  # irregularity of the cut-out edge (mm of ramp offset)
    colour_bleed: float = 2.5  # colours closer than this to the hairline are taken from this far inside (no skin tint)
    grain: float = 0.2  # fine speckle (relative lightness) on the replaced colour, like short stubble
    fill_fade: float = (
        0.45  # the replaced colour is this much lighter at the hairline (a fade), fading out 30 mm inside
    )
    normal_strength: float = 1.0  # 0 flat .. 1 full relief of the high-poly surface and its normal map
    fill_rings: int = 5  # graph rings the glitch / flat-fill mask is softened over
    max_jump: float = 30.0  # grid cells whose corners differ more in radius than this are dropped
    seed: int = 11

    @classmethod
    def with_overrides(cls, overrides: dict) -> ShellParams:
        names = {f.name for f in fields(cls)}
        values = {}
        for key, value in overrides.items():
            if key not in names:
                raise ValueError(f"Unknown hair shell parameter {key!r}; known: {', '.join(sorted(names))}")
            number = float(value)
            if not np.isfinite(number) or number < 0:
                raise ValueError(f"Hair shell parameter {key} must be a finite, non-negative number")
            values[key] = int(number) if key in INT_PARAMS else number
        return replace(cls(), **values)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ----------------------------------------------------------------------------------------------- high poly
@dataclass
class HighShell:
    """The high-poly hair surface plus a margin around it (compact vertex numbering)."""

    positions: np.ndarray  # (n, 3) metres, aligned frame
    normals: np.ndarray  # (n, 3)
    colours: np.ndarray  # (n, 3) float32 sRGB 0..255
    faces: np.ndarray  # (m, 3)
    face_uv: np.ndarray  # (m, 3, 2) original glTF UV of the corners
    inside: np.ndarray  # (n,) bool: vertex belongs to the hair region (not just to the margin)
    signed_edge: np.ndarray  # (n,) metres: distance to the hair edge, positive inside, negative in the margin
    source: np.ndarray  # (n,) index into the arrays the shell was cut from
    fill_weight: np.ndarray | None = None  # (n,) 0..1: the texture is the generator's glitch or featureless fill here
    fill_colour: np.ndarray | None = None  # (n, 3) sRGB 0..255: the colour that replaces it (plus a grain)


def edge_vertices(faces: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Vertices of the triangles that straddle the border of ``inside`` (both sides of the edge)."""
    mixed = inside[faces].any(1) & ~inside[faces].all(1)
    return np.unique(faces[mixed])


def extract_shell(
    positions: np.ndarray,
    normals: np.ndarray,
    colours: np.ndarray,
    faces: np.ndarray,
    face_uv: np.ndarray,
    mask: np.ndarray,
    region: np.ndarray,
) -> HighShell:
    """The triangles whose three vertices are in ``region`` (the hair ``mask`` plus a margin), compacted.

    The arrays run over the whole bust. ``signed_edge`` is the distance of each kept vertex to the border of ``mask``.
    """
    keep = region[faces].all(1)
    selected = faces[keep]
    if len(selected) == 0:
        raise ValueError("The hair mask contains no triangle")
    used = np.unique(selected)
    remap = np.full(len(positions), -1, np.int64)
    remap[used] = np.arange(len(used))
    compact = remap[selected]
    inside = mask[used]
    edge = edge_vertices(compact, inside)
    if len(edge):
        distance = cKDTree(positions[used][edge]).query(positions[used], workers=-1)[0]
    else:
        distance = np.full(len(used), 1.0)
    return HighShell(
        positions[used],
        normals[used],
        np.asarray(colours[used], np.float32),
        compact,
        face_uv[keep],
        inside,
        np.where(inside, distance, -distance),
        used,
    )


def stubble_colour(lab: np.ndarray, points: np.ndarray, frame, fallback=(24.0, 0.8, -1.5)) -> np.ndarray:
    """CIELAB of the short side hair: the median of the darker 40 percent of the neutral hair vertices above the ears."""
    az = np.abs(np.degrees(np.arctan2(points[:, 0] - frame.x0, points[:, 2] - frame.zc)))
    dy = points[:, 1] - frame.eye_y
    pick = (
        (az > 60)
        & (az < 125)
        & (dy > 0.005)
        & (dy < 0.06)
        & (lab[:, 0] > 12)
        & (lab[:, 0] < 32)
        & (np.hypot(lab[:, 1], lab[:, 2]) < 6)
    )
    if pick.sum() <= 200:
        return np.asarray(fallback, np.float64)
    dark = pick & (lab[:, 0] <= np.percentile(lab[pick, 0], 40))
    return np.median(lab[dark], axis=0)


def harmonic_fill(
    adjacency, values: np.ndarray, unknown: np.ndarray, usable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The harmonic (graph-Laplacian) extension of ``values`` into the ``unknown`` vertices of the sub-graph ``usable``.

    Known vertices (``usable`` and not ``unknown``) are boundary conditions; the rest of the graph is ignored, so the
    margin of the shell (skin) never leaks in. Returns the filled values and a flag per vertex that was reached from a
    known vertex (an unknown island with no known neighbour keeps its input).
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    out = values.copy()
    reached = np.zeros(len(values), bool)
    ids = np.flatnonzero(usable)
    local = np.full(len(values), -1, np.int64)
    local[ids] = np.arange(len(ids))
    sub = adjacency[ids][:, ids].tocsr()
    free = np.flatnonzero(unknown[ids])
    known = np.flatnonzero(~unknown[ids])
    if len(free) == 0 or len(known) == 0:
        return out, reached
    # islands of unknown vertices that touch no known vertex cannot be solved: drop them
    count, label = connected_components(sub[free][:, free], directed=False)
    touches = np.zeros(count, bool)
    contact = sub[free][:, known]
    touches[label[np.asarray(contact.sum(1)).ravel() > 0]] = True
    solvable = free[touches[label]]
    if len(solvable) == 0:
        return out, reached
    degree = sp.diags(np.asarray(sub.sum(1)).ravel())
    laplacian = (degree - sub).tocsr()
    a = laplacian[solvable][:, solvable].tocsc()
    rest = np.setdiff1d(np.arange(len(ids)), solvable)
    b = -(laplacian[solvable][:, rest] @ values[ids[rest]])
    solution = np.asarray(spsolve(a, b)).reshape(len(solvable), -1)
    out[ids[solvable]] = solution
    reached[ids[solvable]] = True
    return out, reached


def attach_fill(
    high: HighShell, graph, pale: np.ndarray, lab: np.ndarray, short_hair_lab, rings: int, flat_std: float = 1.8
) -> dict:
    """Mark the parts of the texture that are not the person's hair and give them the colours of their surroundings.

    Two kinds: ``pale`` vertices (a light patch the generator painted on the unseen back of the head) and the featureless
    flat grey it fills the rest of that back with (the local lightness hardly varies, chroma near zero). Their colour is
    the harmonic extension of the real hair around them (only the vertices inside the hairline count; where nothing real
    is near, ``short_hair_lab`` takes over). The mask is softened over ``rings`` graph rings so the replacement blends in;
    the bake adds a fine grain to it.
    """
    from flamehead.colour import from_lab

    mean = graph.smooth(lab[:, 0], 3)
    spread = np.sqrt(np.maximum(graph.smooth(lab[:, 0] ** 2, 3) - mean**2, 0.0))
    chroma = np.hypot(graph.smooth(lab[:, 1], 3), graph.smooth(lab[:, 2], 3))
    flat = (spread < flat_std) & (chroma < 4.0) & (mean > 20.0) & (mean < 50.0) & high.inside
    flagged = (flat | pale) & high.inside
    unknown = graph.dilate(flagged, 2) & high.inside
    filled, reached = harmonic_fill(graph.adjacency, np.asarray(lab, np.float64), unknown, high.inside)
    fallback = np.asarray(short_hair_lab, np.float64)
    filled[unknown & ~reached] = fallback
    weight = np.clip(graph.smooth(flagged.astype(np.float64), rings) * 1.8, 0.0, 1.0) if rings else flagged * 1.0
    weight[flagged] = 1.0
    colour = np.clip(from_lab(filled.astype(np.float32)) * 255.0, 0, 255).astype(np.float32)
    high.fill_weight, high.fill_colour = weight, colour
    return {
        "pale_vertices": int(pale.sum()),
        "flat_vertices": int(flat.sum()),
        "filled_share_of_shell": float((weight > 0.5).mean()),
        "unreached_vertices": int((unknown & ~reached).sum()),
        "short_hair_lab": [float(v) for v in short_hair_lab],
    }


def triangle_area(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = positions[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2


def unit_rows(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-18)


def surface_samples(positions: np.ndarray, faces: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Regular barycentric lattice of sample points on every triangle: ``(face ids, barycentrics (k, 3))``.

    A triangle with ``m * m`` sub-triangles contributes their centroids, ``m`` chosen so that one sample covers about
    ``spacing`` squared of surface.
    """
    area = triangle_area(positions, faces)
    m = np.clip(np.rint(np.sqrt(area) / spacing).astype(np.int64), 1, 24)
    face_ids, bary = [], []
    for k in np.unique(m):
        sel = np.flatnonzero(m == k)
        lattice = []
        for i in range(k):
            for j in range(k - i):
                lattice.append(((i + 1 / 3) / k, (j + 1 / 3) / k))  # up-pointing sub-triangles
        for i in range(k - 1):
            for j in range(k - 1 - i):
                lattice.append(((i + 2 / 3) / k, (j + 2 / 3) / k))  # down-pointing ones
        lattice = np.array(lattice)
        b = np.column_stack((1 - lattice.sum(1), lattice))
        face_ids.append(np.repeat(sel, len(b)))
        bary.append(np.tile(b, (len(sel), 1)))
    return np.concatenate(face_ids), np.concatenate(bary)


# ------------------------------------------------------------------------------------------ radial remesh
@dataclass
class Unwrapped:
    positions: np.ndarray  # (v, 3)
    faces: np.ndarray  # (m, 3)
    uv: np.ndarray  # (v, 2) in [0, 1], v down (image row = v * size)
    report: dict


def lambert(directions: np.ndarray) -> np.ndarray:
    """Lambert azimuthal equal-area projection of unit vectors about +Y: ``(x, y)`` in a disc of radius 2."""
    psi = np.arccos(np.clip(directions[:, 1], -1, 1))
    rho = 2 * np.sin(psi / 2)
    alpha = np.arctan2(directions[:, 0], directions[:, 2])
    return np.stack([rho * np.sin(alpha), rho * np.cos(alpha)], axis=1)


def inverse_lambert(xy: np.ndarray) -> np.ndarray:
    rho = np.hypot(xy[:, 0], xy[:, 1])
    psi = 2 * np.arcsin(np.clip(rho / 2, 0, 1))
    alpha = np.arctan2(xy[:, 0], xy[:, 1])
    return np.stack([np.sin(psi) * np.sin(alpha), np.cos(psi), np.sin(psi) * np.cos(alpha)], axis=1)


def grid_mesh(
    sample_xy: np.ndarray, tree: cKDTree, radius: np.ndarray, centre: np.ndarray, step: float, max_jump: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """One square grid of spacing ``step`` over the occupied part of the disc: ``(node xy, node positions, faces)``.

    A node's radius is that of the outer layer of the samples nearest to its direction (inverse distance weights).
    """
    cells = np.unique(np.floor(sample_xy / step).astype(np.int64), axis=0)
    count = len(cells)
    corner = np.concatenate([cells + np.array(o) for o in ((0, 0), (1, 0), (0, 1), (1, 1))])
    nodes, inverse = np.unique(corner, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    # quad corners in cyclic order: (0,0), (1,0), (1,1), (0,1)
    quad = np.stack(
        [inverse[:count], inverse[count : 2 * count], inverse[3 * count :], inverse[2 * count : 3 * count]], axis=1
    )
    xy = nodes * step
    direction = inverse_lambert(xy)
    distance, found = tree.query(direction, k=6, distance_upper_bound=2.5 * step, workers=-1)
    finite = np.isfinite(distance)
    found = np.where(finite, found, 0)
    r = np.where(finite, radius[found], -np.inf)
    top = r.max(1)
    layer = finite & (r >= top[:, None] - 0.003)
    weight = np.where(layer, 1.0 / (np.where(finite, distance, 1.0) + 1e-4), 0.0)
    r_node = (np.where(layer, r, 0.0) * weight).sum(1) / np.maximum(weight.sum(1), 1e-12)
    valid = np.isfinite(top) & (np.hypot(xy[:, 0], xy[:, 1]) < 1.999)
    ok = valid[quad].all(1)
    ok &= (r_node[quad].max(1) - r_node[quad].min(1)) <= max_jump
    quad = quad[ok]
    if len(quad) == 0:
        return None
    positions = centre + r_node[:, None] * direction
    a, b, c, d = quad.T  # split along the shorter 3-D diagonal
    short_ac = np.linalg.norm(positions[a] - positions[c], axis=1) <= np.linalg.norm(
        positions[b] - positions[d], axis=1
    )
    first = np.where(short_ac[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
    second = np.where(short_ac[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
    faces = np.concatenate((first, second))
    used = np.unique(faces)
    remap = np.full(len(positions), -1, np.int64)
    remap[used] = np.arange(len(used))
    return xy[used], positions[used], remap[faces]


def radial_remesh(samples: np.ndarray, centre: np.ndarray, params: ShellParams) -> Unwrapped:
    """The equal-area grid shell of the high-poly surface; ``samples`` are dense surface points (the hair plus margin)."""
    rel = samples - centre
    radius = np.linalg.norm(rel, axis=1)
    direction = rel / radius[:, None]
    tree = cKDTree(direction)
    sample_xy = lambert(direction)
    step = 0.02
    result = None
    for _ in range(8):  # tune the grid step until the triangle count is within 4 percent of the target
        result = grid_mesh(sample_xy, tree, radius, centre, step, params.max_jump * MM)
        if result is None:
            step *= 0.6
            continue
        ratio = len(result[2]) / params.target_triangles
        if abs(ratio - 1) < 0.04:
            break
        step *= float(np.clip(np.sqrt(ratio), 0.6, 1.6))
    if result is None:
        raise ValueError("The radial remesh produced no triangles")
    xy, positions, faces = result
    size, pad = params.texture_size, params.padding
    lo, hi = xy.min(0), xy.max(0)
    span = float((hi - lo).max()) / max(1.0 - 2.0 * pad / size, 0.5)
    origin = (lo + hi) / 2 - span / 2
    uv = np.stack([(xy[:, 0] - origin[0]) / span, (origin[1] + span - xy[:, 1]) / span], axis=1)
    tri = positions[faces]  # outward winding: the radial direction and the face normal agree
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.median(np.einsum("ij,ij->i", normal, tri.mean(1) - centre)) < 0:
        faces = faces[:, ::-1].copy()
    edge = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    length = np.linalg.norm(positions[edge[:, 0]] - positions[edge[:, 1]], axis=1) * 1000
    return Unwrapped(
        positions,
        faces,
        uv,
        {
            "method": "Lambert equal-area grid over the outer hair surface seen from the head centre",
            "grid_step": float(step),
            "vertices": int(len(positions)),
            "triangles": int(len(faces)),
            "edge_mm": {
                "median": float(np.median(length)),
                "p95": float(np.percentile(length, 95)),
                "max": float(length.max()),
            },
            "area_cm2": float(triangle_area(positions, faces).sum() * 1e4),
            "uv_span": span,
            "centre": centre.tolist(),
        },
    )


def surface_error(low_positions: np.ndarray, samples: np.ndarray) -> dict:
    """Distance of the shell vertices to the dense samples of the high-poly surface (a bound of the error), in mm."""
    distance = cKDTree(samples).query(low_positions, workers=-1)[0] * 1000
    return {
        "vertices": int(len(low_positions)),
        "mean_mm": float(distance.mean()),
        "p95_mm": float(np.percentile(distance, 95)),
        "max_mm": float(distance.max()),
    }


def decimate_shell(shell: Unwrapped, target: int) -> Unwrapped:
    """Quadric reduction of the repaired radial surface, retaining a welded mesh for fitting.

    Decimating the original noisy bust can fold triangles and produce hundreds of UV islands.
    The existing radial remesh repairs that surface first, at twice the requested budget.
    UVs here are temporary; ``unwrap_shell`` generates the final atlas after reduction.
    """
    import fast_simplification

    centre = np.asarray(shell.report["centre"])
    attempts = []
    for aggressiveness in (5.0, 4.0, 3.0, 2.0):
        points, faces = fast_simplification.simplify(
            shell.positions, shell.faces, target_count=min(target, len(shell.faces)), agg=aggressiveness
        )
        points, faces = np.asarray(points, np.float64), np.asarray(faces, np.int64)
        tri = points[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(normal, axis=1)
        edges = np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        outward = np.einsum("ij,ij->i", normal, tri.mean(1) - centre)
        unique_faces = np.unique(np.sort(faces, axis=1), axis=0)
        quality = {
            "aggressiveness": aggressiveness,
            "triangles": len(faces),
            "degenerate": int((area <= 1e-14).sum()),
            "nonmanifold": int((counts > 2).sum()),
            "duplicate": len(faces) - len(unique_faces),
            "inward_radial": int((outward <= 0).sum()),
        }
        attempts.append(quality)
        if not any(quality[key] for key in ("degenerate", "nonmanifold", "duplicate", "inward_radial")):
            break
    else:
        raise ValueError(f"Quadric shell reduction failed geometry checks: {attempts}")
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1) * 1000
    return Unwrapped(
        points,
        faces,
        np.zeros((len(points), 2)),
        {
            **shell.report,
            "method": "Lambert surface repair, quadric decimation, xatlas UVs",
            "repair_triangles": len(shell.faces),
            "quadric_attempts": attempts,
            "vertices": len(points),
            "triangles": len(faces),
            "edge_mm": {
                "median": float(np.median(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "max": float(lengths.max()),
            },
            "area_cm2": float(area.sum() * 0.5 * 1e4),
        },
    )


def unwrap_shell(shell: Unwrapped, params: ShellParams) -> tuple[Unwrapped, np.ndarray]:
    """Pack xatlas charts; return render geometry and its mapping to welded fit vertices.

    The mapping keeps UV seam copies coincident during scalp fitting and clearance correction.
    glTF UVs use image rows down, so the xatlas vertical coordinate is flipped.
    """
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(shell.positions.astype(np.float32), shell.faces.astype(np.uint32))
    charts = xatlas.ChartOptions()
    pack = xatlas.PackOptions()
    pack.resolution = params.texture_size
    pack.padding = params.padding
    pack.bilinear = True
    atlas.generate(charts, pack)
    mapping, faces, uv = atlas[0]
    mapping = np.asarray(mapping, np.int64)
    uv = np.asarray(uv, np.float64).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return Unwrapped(
        shell.positions[mapping],
        np.asarray(faces, np.int64),
        uv,
        {
            **shell.report,
            "uv_charts": int(atlas.chart_count),
            "render_vertices": len(mapping),
        },
    ), mapping


# ------------------------------------------------------------------------------------------------- baking
def tangent_frames(positions: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle tangent (towards +u) and bitangent (UP the image, i.e. towards decreasing v), unit vectors.

    glTF normal maps: +X right, +Y up the image, +Z out of the surface; with the image's v axis pointing down the
    bitangent is minus the derivative of the position with respect to v. ``uv`` is (v, 2) per vertex or (m, 3, 2)
    per face corner.
    """
    corner = uv[faces] if uv.ndim == 2 else uv
    p0, p1, p2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det = np.where(np.abs(det) < 1e-18, 1e-18, det)
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
    d_dv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det[:, None]
    return unit_rows(tangent), unit_rows(-d_dv)


def orthonormal_frame(normal: np.ndarray, tangent: np.ndarray, bitangent: np.ndarray):
    """Gram-Schmidt of the tangent and bitangent against the normal, the way the renderer builds its frame."""
    t = unit_rows(tangent - normal * np.einsum("ij,ij->i", tangent, normal)[:, None])
    b = bitangent - normal * np.einsum("ij,ij->i", bitangent, normal)[:, None]
    b = unit_rows(b - t * np.einsum("ij,ij->i", b, t)[:, None])
    return t, b


def texture_lookup(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a uint8 (H, W, 3) image at glTF UVs (v down, texel centres at half-integers): float32 (n, 3)."""
    import cv2

    h, w = image.shape[:2]
    count = len(uv)
    columns = 2048  # cv2.remap needs both map dimensions below 32767
    rows = -(-count // columns)
    x = np.zeros(rows * columns, np.float32)
    y = np.zeros(rows * columns, np.float32)
    x[:count] = uv[:, 0] * w - 0.5
    y[:count] = uv[:, 1] * h - 0.5
    out = cv2.remap(
        image, x.reshape(rows, columns), y.reshape(rows, columns), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return out.reshape(rows * columns, -1)[:count].astype(np.float32)


@dataclass
class Baked:
    colour: np.ndarray  # (S, S, 4) uint8: sRGB colour, alpha = hairline fringe
    normal: np.ndarray  # (S, S, 3) uint8 tangent-space normal map
    covered: np.ndarray  # (S, S) bool: texels inside a triangle
    visible: np.ndarray  # (S, S) bool: covered and alpha >= 0.5
    mean_colour: np.ndarray  # (3,) mean sRGB colour of the visible texels
    report: dict


def texel_noise(size: int, ys: np.ndarray, xs: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Unit-variance random noise on the texel grid, blurred by ``sigma`` texels, read at the texels ``(ys, xs)``.

    Texel space noise has no preferred direction and no regular ripples (plane-wave noise in 3-D showed moire rings).
    """
    from scipy import ndimage

    field = np.random.default_rng(seed).normal(size=(size, size)).astype(np.float32)
    if sigma > 0:
        field = ndimage.gaussian_filter(field, sigma)
        field /= max(float(field.std()), 1e-9)
    return field[ys, xs].astype(np.float64)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pad_chart_colours(image: np.ndarray, covered: np.ndarray, pixels: int) -> np.ndarray:
    """Bleed the covered texels outwards by ``pixels`` (nearest covered texel); far texels get the mean colour."""
    from scipy import ndimage

    out = image.copy()
    distance, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    ring = ~covered & (distance <= pixels)
    out[ring] = image[iy[ring], ix[ring]]
    far = ~covered & ~ring
    if far.any():
        out[far] = np.rint(image[covered].mean(0)).astype(image.dtype)
    return out


def welded_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth, area-weighted vertex normals."""
    from .template import weld

    inverse, first = weld(positions)
    tri = positions[faces]
    face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    accumulated = np.zeros((len(first), 3))
    for k in range(3):
        np.add.at(accumulated, inverse[faces[:, k]], face_normal)
    return unit_rows(accumulated)[inverse]


def bake(
    shell: Unwrapped,
    high: HighShell,
    sample_face: np.ndarray,
    sample_bary: np.ndarray,
    base_texture: np.ndarray,
    normal_texture: np.ndarray | None,
    params: ShellParams,
) -> Baked:
    """Colour, alpha and tangent-space normal textures of the shell from the high-poly surface and its textures."""
    size = params.texture_size
    face_id, bary = rasterize_uv(shell.uv * size, shell.faces, size, size)
    ys, xs = np.nonzero(face_id >= 0)
    face = face_id[ys, xs]
    lam = bary[ys, xs].astype(np.float64)
    tri = shell.faces[face]
    position = np.einsum("ij,ijk->ik", lam, shell.positions[tri])
    n_low = unit_rows(np.einsum("ij,ijk->ik", lam, welded_normals(shell.positions, shell.faces)[tri]))
    t_face, b_face = tangent_frames(shell.positions, shell.faces, shell.uv)
    t_low, b_low = orthonormal_frame(n_low, t_face[face], b_face[face])

    # dense samples of the high-poly surface carry its face and barycentrics: the original textures are looked up there
    sample_tri = high.faces[sample_face]
    sample_point = np.einsum("ij,ijk->ik", sample_bary, high.positions[sample_tri])
    sample_edge = np.einsum("ij,ij->i", sample_bary, high.signed_edge[sample_tri])  # metres, + inside the hair region
    distance, nearest = cKDTree(sample_point).query(position, workers=-1)
    edge_mm = sample_edge[nearest] * 1000
    # texels at and beyond the hairline take the colour and relief of the nearest sample ``colour_bleed`` inside the hair
    # (the texture there is the skin-to-hair fade, which would tint the cut-out edge pink)
    source = nearest.copy()
    deep = np.flatnonzero(sample_edge >= params.colour_bleed * MM)
    shallow = sample_edge[nearest] < params.colour_bleed * MM
    if shallow.any() and len(deep):
        _, k = cKDTree(sample_point[deep]).query(position[shallow], workers=-1)
        source[shallow] = deep[k]
    f, b = sample_face[source], sample_bary[source]
    uv = np.einsum("ij,ijk->ik", b, high.face_uv[f].astype(np.float64))
    colour = texture_lookup(base_texture, uv)
    corner = high.faces[f]
    n_geo = unit_rows(np.einsum("ij,ijk->ik", b, high.normals[corner]))
    grain_texels = 0
    if high.fill_weight is not None:
        fill = np.einsum("ij,ij->i", b, high.fill_weight[corner])
        # the short hair fades lighter towards the hairline (like a real fade), darker deeper inside
        lighter = 1.0 + params.fill_fade * (1.0 - smoothstep(np.maximum(edge_mm, 0.0) / 30.0))
        replacement = np.clip(np.einsum("ij,ijk->ik", b, high.fill_colour[corner]) * lighter[:, None], 0, 255)
        colour = colour * (1 - fill[:, None]) + replacement * fill[:, None]
        if params.grain > 0:
            speckle = texel_noise(size, ys, xs, 0.8, params.seed)
            colour = colour * (1.0 + params.grain * fill[:, None] * np.clip(speckle, -2.0, 2.0)[:, None])
        grain_texels = int((fill > 0.5).sum())
    if normal_texture is not None:
        t_h, b_h = tangent_frames(high.positions, high.faces, high.face_uv.astype(np.float64))
        t_s, b_s = orthonormal_frame(n_geo, t_h[f], b_h[f])
        map_n = texture_lookup(normal_texture, uv) / 255.0 * 2.0 - 1.0
        n_high = unit_rows(map_n[:, 0:1] * t_s + map_n[:, 1:2] * b_s + map_n[:, 2:3] * n_geo)
    else:
        n_high = n_geo
    # the relief: the high-poly normal in the frame of the low-poly surface, flattened by the strength (never below
    # 0.2 out of the surface: the shading frame of the renderer breaks down for grazing normals)
    local = np.stack(
        [
            np.einsum("ij,ij->i", n_high, t_low),
            np.einsum("ij,ij->i", n_high, b_low),
            np.einsum("ij,ij->i", n_high, n_low),
        ],
        axis=1,
    )
    local[:, :2] *= params.normal_strength
    local[:, 2] = np.maximum(local[:, 2], 0.2)
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(local[:, 2], -1, 1)))

    # alpha: 0.5 at the hairline, a noisy ramp of ``fringe`` mm (irregular, like a real hairline)
    jitter = params.fringe_noise * (
        0.7 * texel_noise(size, ys, xs, 6.0, params.seed + 1) + 0.3 * texel_noise(size, ys, xs, 1.5, params.seed + 2)
    )
    alpha = smoothstep((edge_mm + 0.5 * params.fringe + jitter) / max(params.fringe, 1e-3))

    covered = np.zeros((size, size), bool)
    covered[ys, xs] = True
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[ys, xs] = np.clip(np.rint(colour), 0, 255).astype(np.uint8)
    a = np.zeros((size, size, 1), np.uint8)
    a[ys, xs, 0] = np.rint(alpha * 255).astype(np.uint8)
    normal = np.zeros((size, size, 3), np.uint8)
    normal[ys, xs] = np.clip(np.rint((local * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    rgba = np.dstack((pad_chart_colours(rgb, covered, params.padding), pad_chart_colours(a, covered, params.padding)))
    normal = pad_chart_colours(normal, covered, params.padding)
    visible_texel = alpha >= 0.5
    visible = np.zeros((size, size), bool)
    visible[ys, xs] = visible_texel
    mean_colour = colour[visible_texel].mean(0) if visible_texel.any() else colour.mean(0)
    return Baked(
        rgba,
        normal,
        covered,
        visible,
        mean_colour,
        {
            "high_poly_samples": int(len(sample_point)),
            "texels_covered": int(covered.sum()),
            "texel_coverage": float(covered.mean()),
            "texels_visible": int(visible.sum()),
            "distance_to_high_poly_mm": {
                "median": float(np.median(distance) * 1000),
                "p99": float(np.percentile(distance, 99) * 1000),
            },
            "mean_colour_srgb": mean_colour.tolist(),
            "grain_texels": grain_texels,
            "normal_map_used": normal_texture is not None,
            "normal_tilt_deg": {"median": float(np.median(tilt)), "p95": float(np.percentile(tilt, 95))},
        },
    )
