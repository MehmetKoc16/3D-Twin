"""Hair segmentation of the aligned bust: colour AND geometry, cleaned on the mesh graph.

Input: the bust in the twin's frame (metres, +Y up, +Z front, the character's left = +X), its CIELAB vertex colours and
the twin head frame (``hairgen.measure_head``: skull centre, eye line, ears, nape crease).

* **Colour** decides wherever hair meets skin that is clearly lighter: the front hairline, the temples and sideburns,
  behind the ears (``|azimuth| <= colour_azimuth``). Hair texels are dark and nearly neutral, the forehead / temple /
  ear / cheek skin is much lighter; the smoothed lightness (a few rings on the mesh graph) is thresholded half way.
* **Geometry** bounds it from below: a floor curve over the azimuth around the skull (generous in front, so that the
  eyebrows, glasses and beard are never hair, the sideburn ending at the top of the ear, the nape a little above the
  crease) and a margin around the twin's ears. At the back the generated texture is a flat grey (the generator never saw
  it): there only the floor decides, colour cannot tell the nape from the neck.
* **Graph clean-up** on the position-welded mesh: the largest connected component, holes filled, thin spurs removed
  (open / close by rings of the graph) and the boundary smoothed by diffusing the indicator and re-thresholding.

Everything returns plain arrays and numbers (no images).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import PchipInterpolator
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MM = 1e-3


@dataclass(frozen=True)
class SegmentParams:
    """Thresholds of the segmentation (millimetres, CIELAB lightness, degrees)."""

    hair_lightness: float = 42.0  # smoothed L* below this is hair (hair ~10..35, forehead skin ~65..78)
    skin_redness: float = (
        4.5  # smoothed a* above this is skin even when dark (ear canal, sideburn shadow): hair ~0..1.5
    )
    smooth_rings: int = 6  # mesh-graph rings the lightness is averaged over before thresholding
    colour_azimuth: float = 125.0  # colour decides for |azimuth| up to here, geometry alone behind it
    ear_margin: float = 5.0  # no hair this close to the twin's ear vertices
    floor_front: float = 48.0  # lowest hair height above the eye line at the front (az 0), colour decides above it
    floor_temple: float = 22.0  # ... at azimuth 50
    nape_above_crease: float = 18.0  # the nape cut of the shell sits this far above the neck crease
    open_rings: int = 3  # removes spurs thinner than about 2 x this many graph rings
    close_rings: int = 3  # closes gaps and notches
    hole_vertices: int = 4000  # holes of at most this many vertices are filled
    smooth_iterations: int = 25  # boundary smoothing: diffusion steps of the indicator
    boundary_smooth_degrees: float = 6.0  # smooth the lower temple/nape outline in angular space
    hairline_smooth: float = 5.0  # mm: Gaussian-equivalent smoothing of the front hairline / temple outline (0 = off)
    hairline_zone: float = 80.0  # |azimuth| up to here the outline is smoothed spatially (rounds the temple corner)
    pale_lightness: float = 36.0  # hair texels lighter than this, away from the edge, are colour artefacts
    pale_edge_mm: float = 8.0  # ... farther from the shell edge than this
    min_vertices: int = 300  # fewer hair vertices than this is a failed segmentation, not a cropped head

    @classmethod
    def with_overrides(cls, overrides: dict) -> SegmentParams:
        names = {f.name for f in fields(cls)}
        ints = {f.name for f in fields(cls) if isinstance(f.default, int) and not isinstance(f.default, bool)}
        values = {}
        for key, value in overrides.items():
            if key not in names:
                raise ValueError(f"Unknown hair segmentation parameter {key!r}; known: {', '.join(sorted(names))}")
            number = float(value)
            if not np.isfinite(number) or number < 0:
                raise ValueError(f"Hair segmentation parameter {key} must be a finite, non-negative number")
            values[key] = int(number) if key in ints else number
        return replace(cls(), **values)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ----------------------------------------------------------------------------------------------- mesh graph
@dataclass
class VertexGraph:
    adjacency: sp.csr_matrix
    walk: sp.csr_matrix  # row-normalised adjacency: averaging over the neighbours

    @classmethod
    def from_faces(cls, faces: np.ndarray, count: int) -> VertexGraph:
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        graph = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
        adjacency = ((graph + graph.T) > 0).astype(np.float64).tocsr()
        degree = np.asarray(adjacency.sum(1)).ravel()
        return cls(adjacency, (sp.diags(1.0 / np.maximum(degree, 1)) @ adjacency).tocsr())

    def dilate(self, mask: np.ndarray, rings: int = 1) -> np.ndarray:
        out = mask.copy()
        for _ in range(rings):
            out = out | (self.adjacency @ out.astype(np.float64) > 0)
        return out

    def erode(self, mask: np.ndarray, rings: int = 1) -> np.ndarray:
        return ~self.dilate(~mask, rings)

    def smooth(self, values: np.ndarray, iterations: int) -> np.ndarray:
        out = np.asarray(values, np.float64).copy()
        for _ in range(iterations):
            out = self.walk @ out
        return out

    def components(self, mask: np.ndarray) -> tuple[np.ndarray, int]:
        """Labels (``-1`` outside ``mask``) of the connected components of the masked vertices and their count."""
        ids = np.flatnonzero(mask)
        labels = np.full(len(mask), -1, np.int64)
        if len(ids) == 0:
            return labels, 0
        count, found = connected_components(self.adjacency[ids][:, ids], directed=False)
        labels[ids] = found
        return labels, int(count)

    def largest(self, mask: np.ndarray) -> np.ndarray:
        labels, count = self.components(mask)
        if count == 0:
            return mask.copy()
        return labels == int(np.argmax(np.bincount(labels[labels >= 0])))

    def fill_holes(self, mask: np.ndarray, max_vertices: int) -> np.ndarray:
        """Fill the components of the complement that have at most ``max_vertices`` vertices and touch nothing outside."""
        labels, count = self.components(~mask)
        if count == 0:
            return mask.copy()
        sizes = np.bincount(labels[labels >= 0], minlength=count)
        # the biggest complement component is the outside
        small = np.flatnonzero(sizes <= max_vertices)
        small = small[small != int(np.argmax(sizes))]
        return mask | np.isin(labels, small)

    def boundary(self, mask: np.ndarray) -> np.ndarray:
        """Masked vertices with at least one unmasked neighbour."""
        return mask & (self.adjacency @ (~mask).astype(np.float64) > 0)


# ---------------------------------------------------------------------------------------------- floor curve
def floor_curve(frame, side: int, params: SegmentParams) -> PchipInterpolator:
    """Lowest height (relative to the eye line, metres) the hair of one side may reach, over the azimuth in degrees."""
    ear = frame.ears[side]
    top = ear.y_top - frame.eye_y
    nape = frame.crease_y - frame.eye_y + params.nape_above_crease * MM
    ef, eb = float(np.clip(ear.a_front, 68.0, 96.0)), float(np.clip(ear.a_back, 90.0, 128.0))
    eb = max(eb, ef + 10.0)
    h0, ht = params.floor_front * MM, params.floor_temple * MM
    azimuth = [0.0, 30.0, 50.0, ef - 20.0, ef - 8.0, ef + 4.0, 0.5 * (ef + eb), eb - 4.0, eb + 8.0, eb + 22.0]
    height = [h0, h0 - 0.006, ht, 0.008, top - 0.004, top - 0.004, top - 0.004, top - 0.004, top - 0.012, nape + 0.030]
    azimuth += [150.0, 180.0]
    height += [nape, nape]
    a, h = np.array(azimuth), np.array(height)
    order = np.argsort(a)
    a, h = a[order], h[order]
    keep = np.concatenate(([True], np.diff(a) > 0.5))
    return PchipInterpolator(a[keep], h[keep])


def azimuth_of(points: np.ndarray, frame) -> np.ndarray:
    """Signed azimuth in degrees around the skull centre: 0 front, +90 the character's left, +-180 the back."""
    return np.degrees(np.arctan2(points[:, 0] - frame.x0, points[:, 2] - frame.zc))


def floor_height(points: np.ndarray, frame, params: SegmentParams) -> np.ndarray:
    a = azimuth_of(points, frame)
    curves = {1: floor_curve(frame, 1, params), -1: floor_curve(frame, -1, params)}
    absolute = np.clip(np.abs(a), 0.0, 180.0)
    return np.where(a >= 0, curves[1](absolute), curves[-1](absolute)) + frame.eye_y


# ----------------------------------------------------------------------------------------- segmentation
@dataclass
class Segmentation:
    mask: np.ndarray  # (n,) bool over the vertices that were passed in
    candidate: np.ndarray  # (n,) bool before the graph clean-up
    report: dict


def segment_hair(
    points: np.ndarray,
    lab: np.ndarray,
    graph: VertexGraph,
    frame,
    params: SegmentParams | None = None,
) -> Segmentation:
    """The hair vertices of the (cropped, welded) bust; ``points`` in the twin frame, ``lab`` CIELAB per vertex."""
    params = params or SegmentParams()
    lightness = graph.smooth(lab[:, 0], params.smooth_rings)
    azimuth = np.abs(azimuth_of(points, frame))
    floor = floor_height(points, frame, params)
    redness = graph.smooth(lab[:, 1], params.smooth_rings)
    colour_ok = ((lightness < params.hair_lightness) & (redness < params.skin_redness)) | (
        azimuth > params.colour_azimuth
    )
    ear_distance = (
        cKDTree(frame.ear_points).query(points, workers=-1)[0] if len(frame.ear_points) else np.full(len(points), 1.0)
    )
    candidate = (points[:, 1] > floor) & colour_ok & (ear_distance > params.ear_margin * MM)
    mask = graph.largest(candidate)
    mask = graph.dilate(graph.erode(mask, params.open_rings), params.open_rings) & mask  # open: thin spurs
    mask = graph.largest(mask)
    mask = graph.erode(graph.dilate(mask, params.close_rings), params.close_rings) | mask  # close: notches
    mask = graph.fill_holes(mask, params.hole_vertices)
    soft = graph.smooth(mask.astype(np.float64), params.smooth_iterations)  # smooth the edge, keep the area
    mask = graph.largest(graph.fill_holes(soft > 0.5, params.hole_vertices))
    mask, hairline_report = smooth_hairline(points, graph, mask, frame, params)
    mask, boundary_report = smooth_lower_boundary(points, graph, mask, candidate, frame, params)
    count, edge = int(mask.sum()), graph.boundary(mask)
    report = {
        "candidate_vertices": int(candidate.sum()),
        "vertices": count,
        "boundary_vertices": int(edge.sum()),
        "hair_lightness": params.hair_lightness,
        "colour_azimuth_deg": params.colour_azimuth,
        "mean_lightness": float(lab[mask, 0].mean()) if count else None,
        "boundary_smoothing": boundary_report,
        "hairline_smoothing": hairline_report,
    }
    return Segmentation(mask, candidate, report)


def smooth_hairline(points, graph, mask, frame, params):
    """Round the front hairline and the temple corner: diffuse the indicator over the mesh graph, re-threshold at 0.5.

    A straight edge does not move; convex corners (the temple corner of the hair region) recede and round off, and the
    vertex-scale staircase of the colour threshold disappears. The reach is ``hairline_smooth`` millimetres (Gaussian
    equivalent) and only the vertices inside ``hairline_zone`` degrees of azimuth may change.
    """
    if params.hairline_smooth <= 0 or graph.adjacency.nnz == 0:
        return mask, {"applied": False}
    rows, cols = graph.adjacency.nonzero()
    h = float(np.median(np.linalg.norm(points[rows] - points[cols], axis=1)))
    iterations = int(np.clip(np.ceil(2.0 * (params.hairline_smooth * MM / max(h, 1e-6)) ** 2), 1, 400))
    soft = graph.smooth(mask.astype(np.float64), iterations)
    zone = np.abs(azimuth_of(points, frame)) < params.hairline_zone
    result = graph.largest(graph.fill_holes(np.where(zone, soft > 0.5, mask), params.hole_vertices))
    return result, {
        "applied": True,
        "sigma_mm": params.hairline_smooth,
        "edge_length_mm": h * 1000,
        "iterations": iterations,
        "changed_vertices": int((result != mask).sum()),
    }


def smooth_lower_boundary(points, graph, mask, candidate, frame, params):
    """Regularize the lower silhouette, preserving the front hairline and disconnected exclusions."""
    from scipy.ndimage import gaussian_filter1d

    from .register import smoothstep

    edge = graph.boundary(mask)
    if edge.sum() < 12 or params.boundary_smooth_degrees == 0:
        return mask, {"applied": False}
    az = azimuth_of(points, frame)
    step = 3.0
    bins = np.arange(-180.0, 180.0, step)
    labels = np.clip(np.floor((az + 180.0) / step).astype(int), 0, len(bins) - 1)
    height = np.full(len(bins), np.nan)
    for i in np.unique(labels[edge]):
        height[i] = np.median(points[edge & (labels == i), 1])
    known = np.isfinite(height)
    centres = bins + step / 2
    raw = np.interp(centres, centres[known], height[known], period=360)
    smooth = gaussian_filter1d(raw, params.boundary_smooth_degrees / step, mode="wrap")
    blend = smoothstep((np.abs(centres) - 30) / 15)
    curve = raw + blend * (smooth - raw)
    floor = np.interp(az, centres, curve, period=360)
    near = (np.abs(az) > 35) & (np.abs(points[:, 1] - floor) < 0.006)
    result = np.where(near, (points[:, 1] >= floor) & (mask | candidate), mask)
    result = graph.largest(result)
    return result, {"applied": True, "sigma_degrees": params.boundary_smooth_degrees,
                    "changed_vertices": int((result != mask).sum()),
                    "profile_second_difference_mm_before": float(np.abs(np.roll(raw, 1)-2*raw+np.roll(raw, -1)).mean()*1000),
                    "profile_second_difference_mm_after": float(np.abs(np.roll(curve, 1)-2*curve+np.roll(curve, -1)).mean()*1000)}


# --------------------------------------------------------------------------------------- colour artefacts
def pale_artefacts(
    points: np.ndarray,
    lab: np.ndarray,
    graph: VertexGraph,
    mask: np.ndarray,
    frame,
    params: SegmentParams | None = None,
) -> np.ndarray:
    """Hair-zone vertices that are far too light for hair and far from the shell edge: the generator's texture glitch
    (a pale patch on the unseen back of the head). Only behind ``colour_azimuth``, where colour could not decide."""
    params = params or SegmentParams()
    lightness = graph.smooth(lab[:, 0], 3)
    edge = graph.boundary(mask)
    far = np.ones(len(points), bool)
    if edge.any():
        far = cKDTree(points[edge]).query(points, workers=-1)[0] > params.pale_edge_mm * MM
    back = np.abs(azimuth_of(points, frame)) > params.colour_azimuth
    pale = mask & back & far & (lightness > params.pale_lightness)
    return graph.dilate(pale, 2) & mask & back
