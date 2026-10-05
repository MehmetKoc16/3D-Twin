"""The user's own hair from a Hunyuan3D bust: segment it, cut it out as a textured shell, fit it to the twin head.

``build_hy3d_hair`` ties the steps together (each lives in its own module and is tested on its own):

1. ``hy3d``      load the bust, find axes and a metric scale from face landmarks, register it onto the twin head;
2. ``hairseg``    segment the hair (colour + a hairline floor + the twin's ears), clean it on the mesh graph;
3. ``hairshell``  repair radially, decimate with quadrics, unwrap with xatlas, bake colour, alpha
                  (a noisy ramp at the hairline) and a normal map from the bust and ITS textures;
4. ``shellfit``   warp the shell onto the scalp where the hair is short, keep the volume on top, push it
                  ``clearance`` outside the whole head.

The result is a ``HairMesh`` of format ``shell/1`` (contract addendum v1.1: a normal PBR material with an sRGB colour
texture, a normal map, ``MASK`` 0.5, double sided) and a ``ShellField`` that tells the body texture where the shell covers
the scalp, so the baked head texture is darkened under it with the hair's own colours.

The bust is a private user asset (``user-data/twin/hy3d/hy3d.glb``): the code only ever reads it from there and everything
it writes stays in ``user-data/`` like the rest of the hybrid stage.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import REPO, log
from .hair import FORMAT_SHELL, NODE_NAME, HairMesh
from .haircheck import penetration_report, scalp_coverage
from .hairgen import HairField, HairStyle, HeadSurface, measure_head
from .hairseg import SegmentParams, VertexGraph, azimuth_of, pale_artefacts, segment_hair
from .hairshell import (
    ShellParams,
    attach_fill,
    bake,
    decimate_shell,
    extract_shell,
    grade_colour,
    radial_remesh,
    stubble_colour,
    surface_error,
    surface_samples,
    unwrap_shell,
    welded_normals,
)
from .hy3d import align_bust, load_bust
from .register import Surface
from .shellfit import FitParams, enforce_clearance, warp_to_scalp

DEFAULT_BUST = REPO / "user-data/twin/hy3d/hy3d.glb"
MM = 1e-3
EAR_PATCH = 0.008  # faces within this distance of an ear vertex are not scalp (the warp never pulls hair onto an ear)
HEAD_CROP_X, HEAD_CROP_Z = 0.16, 0.20  # metres around the skull centre kept for the segmentation (aligned frame)
TINT_DARKEN = 0.85  # the scalp under the shell is a little darker than the shell's own colour


@dataclass(frozen=True)
class ShellStyle:
    """Every tunable of the hy3d hair in one place (``--hair-param NAME=VALUE`` addresses them by field name)."""

    segment: SegmentParams = SegmentParams()
    shell: ShellParams = ShellParams()
    fit: FitParams = FitParams()

    @classmethod
    def with_overrides(cls, overrides: dict | None) -> ShellStyle:
        """The defaults with ``overrides`` (name -> number) routed to the group that has a field of that name."""
        groups = {"segment": SegmentParams, "shell": ShellParams, "fit": FitParams}
        names = {name: {f.name for f in fields(kind)} for name, kind in groups.items()}
        routed: dict = {name: {} for name in groups}
        for key, value in (overrides or {}).items():
            owners = [name for name, known in names.items() if key in known]
            if not owners:
                known = sorted(set().union(*names.values()))
                raise ValueError(f"Unknown hy3d hair parameter {key!r}; known: {', '.join(known)}")
            routed[owners[0]][key] = value
        built = {name: kind.with_overrides(routed[name]) for name, kind in groups.items()}
        return cls(**built)

    def to_dict(self) -> dict:
        return {
            "segment": self.segment.to_dict(),
            "shell": self.shell.to_dict(),
            "fit": self.fit.to_dict(),
        }


class ShellField:
    """Where the visible shell covers the scalp, and in which colour (duck-typed like ``HairField`` for the pipeline).

    A head point is covered when the shell is near it along its normal (hair lying on the scalp) or along the ray from the
    skull centre through it (the shell was built radially from there, so this finds the shell standing off the scalp:
    the quiff above the forehead, the volume on top).

    Beyond the visible edge the skin gets a graduated stubble tint (``tint_weight``): full under the shell, fading to the
    bare skin over a few millimetres at the hairline (with a fine dither, so it reads as sparse short hair, never as a
    line) and carrying on further down the temple / sideburn band, where a faded cut shows dark short stubble.
    """

    REACH_NORMAL = (0.002, 0.006, 0.012, 0.02, 0.03, 0.04)
    REACH_RADIAL = (0.004, 0.012, 0.022, 0.032, 0.042, 0.054)
    NEAR = 0.004  # legacy cover: full within this distance of the shell, none beyond twice this
    SMOOTH_NEIGHBOURS = 96  # shell samples (about 12 mm around) the tint colour is averaged over: no Voronoi wedges
    HAIRLINE_FULL = 0.002  # m beyond the visible edge the stubble tint starts to fade (front hairline)
    HAIRLINE_FADE = 0.007  # m: width of that fade
    HAIRLINE_STRENGTH = 0.92  # the tint never quite reaches the full replacement colour outside the edge
    TEMPLE_FULL = 0.009  # sideburn band: full stubble to here, none beyond TEMPLE_FULL + TEMPLE_FADE
    TEMPLE_FADE = 0.006
    DITHER = 0.45  # relative amplitude of the grain in the fade zone

    def __init__(self, head: HeadSurface, points: np.ndarray, colours_srgb: np.ndarray, centre: np.ndarray, frame=None):
        from flamehead.colour import to_lab

        self.head = head
        self.tree = cKDTree(points)
        self.points = points
        self.centre = np.asarray(centre, np.float64)
        self.frame = frame
        lab = to_lab(colours_srgb.astype(np.float32) / 255.0).astype(np.float32)
        lab[:, 0] *= TINT_DARKEN
        self.lab = lab
        k = int(min(self.SMOOTH_NEIGHBOURS, len(points)))
        neighbours = self.tree.query(points, k=k, workers=-1)[1].reshape(len(points), k)
        self.lab_smooth = lab[neighbours].mean(1)

    def normals_at(self, points: np.ndarray) -> np.ndarray:
        return self.head.closest(points)[3]

    def distance(self, points: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
        """Distance (m) from each head point to the nearest visible shell sample, looking along its normal and radially."""
        normals = self.normals_at(points) if normals is None else normals
        radial = points - self.centre
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        nearest = np.full(len(points), np.inf)
        for direction, reach in ((normals, self.REACH_NORMAL), (radial, self.REACH_RADIAL)):
            for t in reach:
                nearest = np.minimum(nearest, self.tree.query(points + direction * t, workers=-1)[0])
        return nearest

    def cover_from(self, distance: np.ndarray) -> np.ndarray:
        from .register import smoothstep

        return 1.0 - smoothstep((distance - self.NEAR) / self.NEAR)

    def cover(self, points: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
        """0..1: how much hair stands above each head point, along its normal or along the radial ray from the centre."""
        return self.cover_from(self.distance(points, normals))

    def weight(self, points: np.ndarray) -> np.ndarray:
        return self.cover(points)

    def temple_band(self, points: np.ndarray) -> np.ndarray:
        """Skin just beside the lower temple edge; never eyebrows, ear surfaces or the cheeks."""
        return self.band_weight(points) > 0.5

    def band_weight(self, points: np.ndarray) -> np.ndarray:
        """0..1 (smooth edges): the temple / sideburn band, away from the ears, eyebrows and cheeks."""
        from .register import smoothstep

        if self.frame is None:
            return np.zeros(len(points))
        az = np.abs(azimuth_of(points, self.frame))
        ear = (cKDTree(self.frame.ear_points).query(points, workers=-1)[0]
               if len(self.frame.ear_points) else np.full(len(points), np.inf))
        eye = self.frame.eye_y
        return (smoothstep((az - 42.0) / 6.0) * (1.0 - smoothstep((az - 112.0) / 6.0))
                * smoothstep((points[:, 1] - (eye - 0.009)) / 0.006)
                * (1.0 - smoothstep((points[:, 1] - (eye + 0.062)) / 0.006))
                * smoothstep((ear - 0.004) / 0.003))

    def tint_cover(self, points: np.ndarray, normals: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Legacy (round 2) extension: stubble up to 9 mm outside the shell, fading out by 15 mm at the temples."""
        from .register import smoothstep

        distance = self.tree.query(points, workers=-1)[0]
        band = self.temple_band(points)
        fringe = (1.0 - smoothstep((distance - 0.009) / 0.006)) * band
        return np.maximum(base, fringe), band & (fringe > 0.85) & (base < 0.7)

    def hairline_term(self, distance: np.ndarray) -> np.ndarray:
        """The graded stubble fade just beyond the visible edge (forehead hairline; no band, no brow / ear guards)."""
        from .register import smoothstep

        beyond = np.maximum(distance - self.HAIRLINE_FULL, 0.0)
        return self.HAIRLINE_STRENGTH * (1.0 - smoothstep(beyond / self.HAIRLINE_FADE))

    def front_profile(self, distance: np.ndarray, legacy: bool = False) -> np.ndarray:
        """Tint weight at the front hairline as a function of the distance to the shell (``legacy``: round 2)."""
        under = self.cover_from(distance)
        return under if legacy else np.maximum(under, self.hairline_term(distance))

    def tint_weight(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """0..1 stubble tint per head point: full under the shell, a few mm of graded fade at the hairline, a longer one
        in the sideburn band. ``distance`` comes from ``distance()``; ``skin.tint_scalp`` dithers the fade with ``DITHER``."""
        from .register import smoothstep

        under = self.cover_from(distance)
        hairline = self.hairline_term(distance)
        eye = self.frame.eye_y if self.frame is not None else -1e9
        # the fade is for the forehead hairline and the temple: nothing over the eyebrows / cheeks / ears
        above_brow = smoothstep((points[:, 1] - (eye + 0.012)) / 0.010) if self.frame is not None else np.ones(len(points))
        if self.frame is not None and len(self.frame.ear_points):
            above_brow = above_brow * smoothstep((cKDTree(self.frame.ear_points).query(points, workers=-1)[0] - 0.004) / 0.004)
        band = self.band_weight(points)
        temple = band * (1.0 - smoothstep((distance - self.TEMPLE_FULL) / self.TEMPLE_FADE))
        return np.clip(np.maximum(np.maximum(under, hairline * above_brow), temple), 0.0, 1.0)

    def tint_lab(self, points: np.ndarray) -> np.ndarray:
        """CIELAB of the shell around each head point (averaged over ~12 mm: the scalp tint continues the hair's colour)."""
        return self.lab_smooth[self.tree.query(points, workers=-1)[1]]

    def tint_lab_nearest(self, points: np.ndarray) -> np.ndarray:
        """Legacy: CIELAB of the single nearest shell sample (streaks the tint in wedges)."""
        return self.lab[self.tree.query(points, workers=-1)[1]]


@dataclass
class ShellBuild:
    """What the pipeline needs of the hy3d hair (mirrors ``hair.HairBuild`` where it matters)."""

    mesh: HairMesh
    field: ShellField
    style: ShellStyle
    report: dict


def visible_samples(positions: np.ndarray, faces: np.ndarray, uv: np.ndarray, colour: np.ndarray, spacing: float):
    """Sample points of the shell where the texture alpha is at least 0.5, with their colours (for the scalp field)."""
    face_ids, bary = surface_samples(positions, faces, spacing)
    tri = faces[face_ids]
    points = np.einsum("ij,ijk->ik", bary, positions[tri])
    t = np.einsum("ij,ijk->ik", bary, uv[tri])
    h, w = colour.shape[:2]
    ix = np.clip((t[:, 0] * w).astype(int), 0, w - 1)
    iy = np.clip((t[:, 1] * h).astype(int), 0, h - 1)
    texel = colour[iy, ix]
    seen = texel[:, 3] >= 128
    return points[seen], texel[seen, :3]


def alpha_samples(positions: np.ndarray, faces: np.ndarray, uv: np.ndarray, colour: np.ndarray, spacing: float):
    """Sample points over the whole shell with the texture alpha (0..1) at each: the density the cut-out leaves."""
    face_ids, bary = surface_samples(positions, faces, spacing)
    tri = faces[face_ids]
    points = np.einsum("ij,ijk->ik", bary, positions[tri])
    t = np.einsum("ij,ijk->ik", bary, uv[tri])
    h, w = colour.shape[:2]
    ix = np.clip((t[:, 0] * w).astype(int), 0, w - 1)
    iy = np.clip((t[:, 1] * h).astype(int), 0, h - 1)
    return points, colour[iy, ix, 3].astype(np.float64) / 255.0


def edge_report(depth: np.ndarray, distance: np.ndarray) -> dict:
    """Distance of the shell's visible edge ring (0 to 1.5 mm inside the hairline) to the head surface, in mm."""
    ring = (depth >= 0.0) & (depth <= 1.5 * MM)
    if not ring.any():
        return {"vertices": 0}
    d = distance[ring] * 1000
    return {
        "vertices": int(ring.sum()),
        "median_mm": float(np.median(d)),
        "p95_mm": float(np.percentile(d, 95)),
        "max_mm": float(d.max()),
        "share_above_4mm": float((d > 4.0).mean()),
    }


def hairline_profile(
    points: np.ndarray, alpha: np.ndarray, frame, eye_y: float, az_max: float = 100.0, min_height: float = 0.02,
    pixel: float = 0.5,
) -> dict:
    """Shape of the hair edge on the forehead and the temple corner, per side, from shell samples with their alpha.

    The shell is unrolled to (arc length, height above the eye line) in ``pixel``-mm cells (azimuth 0..``az_max``, heights
    from ``min_height`` m up) and its mean alpha is the density image D. The hair edge is the D = 0.5 level line, so
    stray dithered hairs do not count. Everything is measured on that line without tracing it:

    * ``min_radius_mm``: the tightest curve radius (99th percentile of the level-line curvature of D smoothed by 1.5 mm);
      an angled step or a notch is a small radius, a natural rounded corner a large one;
    * ``p50_radius_mm``: the median radius (large = mostly straight);
    * ``roughness_mm``: RMS distance of the line from the same line of a 4 mm-smoothed D (small = a clean curve).
    """
    from scipy.ndimage import gaussian_filter

    az = azimuth_of(points, frame)
    radius = float(np.median(np.hypot(points[:, 0] - frame.x0, points[:, 2] - frame.zc)))
    out = {"radius_mm": radius * 1000, "method": f"D=0.5 level line of the unrolled alpha density, {pixel} mm pixels"}
    for name, sign in (("left", 1.0), ("right", -1.0)):
        a = sign * az
        keep = (a >= 0.0) & (a <= az_max) & (points[:, 1] >= eye_y + min_height)
        if keep.sum() < 200:
            out[name] = None
            continue
        u = np.radians(a[keep]) * radius * 1000.0 / pixel
        v = (points[keep, 1] - eye_y - min_height) * 1000.0 / pixel
        shape = (int(v.max()) + 8, int(u.max()) + 8)
        iy, ix = np.clip(v.astype(int), 0, shape[0] - 1), np.clip(u.astype(int), 0, shape[1] - 1)
        total, count = np.zeros(shape), np.zeros(shape)
        np.add.at(total, (iy, ix), alpha[keep])
        np.add.at(count, (iy, ix), 1.0)

        def density(sigma_mm, total=total, count=count):
            sigma = sigma_mm / pixel
            return gaussian_filter(total, sigma) / np.maximum(gaussian_filter(count, sigma), 1e-6)

        fine, wide = density(1.5), density(4.0)
        # rows / columns the samples really cover (the unrolled image has empty corners): near the 0.5 line only
        line = (np.abs(fine - 0.5) < 0.06) & (gaussian_filter(count, 2.0) > 0.3)
        gy, gx = np.gradient(fine)
        gyy, gyx = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        norm2 = gx**2 + gy**2
        kappa = np.abs(gxx * gy**2 - (gxy + gyx) * gx * gy + gyy * gx**2) / np.maximum(norm2, 1e-9) ** 1.5
        grad_wide = np.hypot(*np.gradient(wide))
        offset = np.abs(fine - wide) / np.maximum(grad_wide, 1e-6)
        line &= norm2 > 1e-6
        if line.sum() < 20:
            out[name] = None
            continue
        radii = pixel / np.maximum(kappa[line], 1e-9)
        out[name] = {
            "line_pixels": int(line.sum()),
            "min_radius_mm": float(np.percentile(radii, 1)),
            "p50_radius_mm": float(np.median(radii)),
            "roughness_mm": float(np.sqrt(np.mean(np.minimum(offset[line] * pixel, 6.0) ** 2))),
        }
    return out


def build_hy3d_hair(
    bust_path,
    head_positions: np.ndarray,
    head_faces: np.ndarray,
    eye_y: float,
    landmark_points: np.ndarray,
    landmark_index: np.ndarray,
    *,
    style: ShellStyle | None = None,
    coverage: bool = True,
    coverage_pixel_mm: float = 0.5,
    colour_hex: str = "#2a1e18",
) -> ShellBuild:
    """The user's hair from the bust, fitted to a head surface (metres, +Y up, +Z front, the character's left = +X).

    ``head_positions`` / ``head_faces`` are the render vertices and faces of the (deformed) head region, ``eye_y`` the
    height of the eye centre line, ``landmark_points`` (468, 3) the app's face-map landmarks on that head and
    ``landmark_index`` their MediaPipe ids.
    """
    style = style or ShellStyle()
    params = style.shell
    head = HeadSurface(head_positions, head_faces)
    frame = measure_head(head, eye_y, HairStyle())
    log(f"hy3d: loading {Path(bust_path).name}")
    bust = load_bust(bust_path)
    log(f"hy3d: {bust.info['welded_vertices']} welded vertices, aligning to the twin head")
    alignment = align_bust(bust, landmark_points, landmark_index, head, frame)
    positions, normals = alignment.positions, alignment.normals
    log(
        f"hy3d: scale to metres {alignment.scale:.4f}, landmark residual median "
        f"{alignment.report['landmark_residual_mm']['median']:.1f} mm"
    )

    # ------------------------------------------------------------------------------------------ segmentation
    near_head = (
        (positions[:, 1] > eye_y - 0.09)
        & (np.abs(positions[:, 0] - frame.x0) < HEAD_CROP_X)
        & (np.abs(positions[:, 2] - frame.zc) < HEAD_CROP_Z)
    )
    ids = np.flatnonzero(near_head)
    remap = np.full(len(positions), -1, np.int64)
    remap[ids] = np.arange(len(ids))
    crop_faces = remap[bust.faces[near_head[bust.faces].all(1)]]
    graph = VertexGraph.from_faces(crop_faces, len(ids))
    segmentation = segment_hair(positions[ids], bust.lab[ids], graph, frame, style.segment)
    pale = pale_artefacts(positions[ids], bust.lab[ids], graph, segmentation.mask, frame, style.segment)
    mask = np.zeros(len(positions), bool)
    mask[ids[segmentation.mask]] = True
    flagged = np.zeros(len(positions), bool)
    flagged[ids[pale]] = True
    if mask.sum() < style.segment.min_vertices:
        raise ValueError("The hair segmentation found almost nothing; check the alignment and the bust")
    log(f"hy3d: hair mask {int(mask.sum())} vertices, {int(pale.sum())} pale-patch vertices")

    # ------------------------------------------------------------------------------------------- shell + bake
    distance_to_hair = cKDTree(positions[ids[segmentation.mask]]).query(positions[ids], workers=-1)[0]
    region = np.zeros(len(positions), bool)
    region[ids[distance_to_hair <= params.margin * MM]] = True
    region |= mask
    high = extract_shell(positions, normals, bust.colours, bust.faces, bust.face_uv, mask, region)
    high_graph = VertexGraph.from_faces(high.faces, len(high.positions))
    if params.edge_smooth > 0 and high_graph.adjacency.nnz:
        rows, cols = high_graph.adjacency.nonzero()
        h = float(np.median(np.linalg.norm(high.positions[rows] - high.positions[cols], axis=1)))
        steps = int(np.clip(np.ceil(2.0 * (params.edge_smooth * MM / max(h, 1e-6)) ** 2), 1, 300))
        # a smoothed distance field: its zero line is the hair edge with the teeth / steps below ~edge_smooth removed
        high.signed_edge = high_graph.smooth(high.signed_edge, steps)
    source_lab = bust.lab[high.source]
    short_hair = stubble_colour(source_lab, high.positions, frame)
    fill = attach_fill(high, high_graph, flagged[high.source], source_lab, short_hair, params.fill_rings)
    sample_face, sample_bary = surface_samples(high.positions, high.faces, params.sample_spacing * MM)
    samples = np.einsum("ij,ijk->ik", sample_bary, high.positions[high.faces[sample_face]])
    centre = np.array([frame.x0, eye_y + 0.035, frame.zc])
    repaired = radial_remesh(samples, centre, replace(params, target_triangles=2 * params.target_triangles))
    shell = decimate_shell(repaired, params.target_triangles)
    render_shell, render_to_fit = unwrap_shell(shell, params)
    log(f"hy3d: shell {len(shell.faces)} triangles, {len(shell.positions)} vertices")
    baked = bake(render_shell, high, sample_face, sample_bary, bust.base_texture, bust.normal_texture, params)
    log(
        f"hy3d: baked {params.texture_size} px colour + normal textures ({baked.report['texels_visible']} visible texels)"
    )

    # ------------------------------------------------------------------------------------------------- fit
    _, nearest = cKDTree(high.positions).query(shell.positions, workers=-1)
    depth = high.signed_edge[nearest]
    inside_points = high.positions[high.inside]
    azimuth = azimuth_of(inside_points, frame)
    front = inside_points[(np.abs(azimuth) < 6.0) & (inside_points[:, 1] > eye_y + 0.02)]
    front_hairline_mm = float((front[:, 1].min() - eye_y) * 1000) if len(front) else float(HairStyle().hairline_front)
    field_style = HairStyle.with_overrides({"hairline_front": max(front_hairline_mm, 30.0)})
    short = 1.0 - HairField(frame, field_style).top_weight(shell.positions)
    ear_distance = cKDTree(frame.ear_points).query(head.vertices, workers=-1)[0] if len(frame.ear_points) else None
    scalp_faces = (
        np.flatnonzero((ear_distance[head.faces] > EAR_PATCH).all(1))
        if ear_distance is not None
        else np.arange(len(head.faces))
    )
    scalp = Surface(head.vertices, head.faces)
    moved, warp = warp_to_scalp(shell.positions, shell.faces, depth, short, scalp, scalp_faces, style.fit)
    final, clearance = enforce_clearance(
        moved, shell.faces, head, style.fit.clearance * MM, style.fit.clearance_iterations, tolerance=2e-4
    )
    shell_normals = welded_normals(final, shell.faces)
    log(
        f"hy3d: warp max {warp['displacement_mm']['max']:.1f} mm, min clearance {clearance['min_vertex_mm']:.2f} mm "
        f"({clearance['vertices_inside_head']} vertices inside the head)"
    )

    # ------------------------------------------------------------------------------------------ outputs
    atlas, colour_grade = grade_colour(baked.colour, colour_hex, baked.visible)
    colour_hex = colour_hex.lower()
    mesh = HairMesh(
        final[render_to_fit],
        shell_normals[render_to_fit],
        render_shell.uv,
        render_shell.faces,
        atlas,
        {"colorHex": colour_hex},
        0,
        NODE_NAME,
        FORMAT_SHELL,
        baked.normal,
    )
    sample_points, sample_colours = visible_samples(mesh.positions, mesh.faces, mesh.uv, atlas, 0.0025)
    field = ShellField(head, sample_points, sample_colours, centre, frame)
    signed, _, _ = head.signed_distance(final, candidates=24)
    penetration = penetration_report(head, final, shell.faces, style.fit.clearance * MM, tolerance=2e-4)
    visible_vertex = depth > 0
    visible = {
        "vertices": int(visible_vertex.sum()),
        "min_mm": float(signed[visible_vertex].min() * 1000) if visible_vertex.any() else None,
        "median_mm": float(np.median(signed[visible_vertex]) * 1000) if visible_vertex.any() else None,
    }
    hair_alpha = (atlas[..., 3] >= 128).astype(np.float32)
    coverage_report = (
        scalp_coverage(
            head, field, mesh.positions, mesh.faces, mesh.uv, hair_alpha, pixel_mm=coverage_pixel_mm, min_density=0.9
        )
        if coverage
        else None
    )
    edge_points, edge_alpha = alpha_samples(mesh.positions, mesh.faces, mesh.uv, atlas, 0.0007)
    edge_profile = hairline_profile(edge_points, edge_alpha, frame, eye_y)
    edge_az = azimuth_of(final, frame)
    edge_front = (np.abs(edge_az) < 6.0) & (depth >= -0.5 * MM) & (depth <= 1.5 * MM)
    report = {
        "kind": "hy3d-shell",
        "format": FORMAT_SHELL,
        "source": bust.info,
        "transform": alignment.report,
        "segmentation": {**segmentation.report, "pale_vertices": int(pale.sum())},
        "fill": fill,
        "shell": {
            **render_shell.report,
            "surface_error": surface_error(shell.positions, samples),
            "bake": baked.report,
            "colour_hex": colour_hex,
            "colour_grade": colour_grade,
        },
        "fit": {"params": style.fit.to_dict(), "warp": warp, "clearance": clearance},
        "penetration": penetration,
        "visible_clearance": visible,
        "edge_gap": edge_report(depth, signed),
        "coverage": coverage_report,
        "hairline": {
            "front_mm_above_eye_bust": front_hairline_mm,
            "front_mm_above_eye_shell": float((final[edge_front, 1].min() - eye_y) * 1000)
            if edge_front.any()
            else None,
            "procedural_target_mm": float(HairStyle().hairline_front),
            "profile": edge_profile,
            "alpha_ramp_mm": {
                "fringe": params.fringe,
                "width_10_90": 0.608 * params.fringe,
                "stipple_mm": params.stipple,
            },
        },
        "frame": {"eye_y_m": float(eye_y), "skull_centre_xz_m": [frame.x0, frame.zc], "notes": frame.notes},
        "vertices": int(len(mesh.positions)),
        "triangles": int(len(shell.faces)),
        "colours": {"colorHex": colour_hex},
        "atlas": {
            "format": FORMAT_SHELL,
            "colour_size": [int(atlas.shape[1]), int(atlas.shape[0])],
            "normal_size": [int(baked.normal.shape[1]), int(baked.normal.shape[0])],
            "alpha": "hairline fringe, MASK 0.5",
        },
        "style": style.to_dict(),
    }
    return ShellBuild(mesh, field, style, report)
