"""Procedural short-hair cards grown on the deformed MakeHuman head.

The hairstyle is a modern short cut: cropped and tapered sides and back (about 1 to 1.5 cm, fading to a few
millimetres at the hairline, above the ears and at the nape), more volume on top (4 to 6 cm) swept up and back from
the forehead, a visible forehead with the hairline about 6.5 to 7 cm above the eye centre line.

Algorithm (all lengths in the style are millimetres, everything else is metres in the head's frame: +Y up, +Z front,
the character's left is +X):

1. **Head frame** (``measure_head``): skull centre, front/crown references, the ears (found from the normal
   deviation of the head surface), the back-of-neck crease (where the skull starts to bulge over the neck).
2. **Hair field** (``HairField``): a smooth hairline curve in (azimuth, height) around the skull, a taper line that
   separates the long top from the short sides, and from them the hair *density* (0..1, soft hairline), strand
   *length*, *lift* and the combing *flow* direction at any surface point.
3. **Guides**: dense Poisson-disc roots on the scalp, in layers (an undercoat of short cards that hides the scalp, a
   body layer, an outer silhouette layer of the longest cards, fine stubble in the soft hairline band). Every guide
   is grown over the scalp along the flow field; its height above the surface follows an arch-shaped lift profile
   (strands rise at the front, lie flat on the sides, tips settle back towards the flow surface) and is never below
   the clearance (2 mm). Neighbouring guides share smooth direction noise, so they clump instead of bristling.
4. **Cards**: every guide becomes a camera-independent ribbon (a narrow strip with a few segments) whose width lies
   in the surface tangent plane and whose UVs address one slot of the strand data atlas (``hairtex``): U across the
   card, V from the root (v of the slot's top) to the tip. Ribbon normals are blended with the scalp normal (smooth
   shading of the hair volume).
5. **Clearance pass**: any card vertex that is closer than the clearance to the head surface is pushed out along the
   surface normal; the result is verified numerically (``haircheck``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .hairtex import StripLayout
from .register import smoothstep
from .template import weld

MM = 1e-3


# ----------------------------------------------------------------------------------------------------- style
@dataclass(frozen=True)
class HairStyle:
    """Parameters of the hairstyle. Lengths and heights are in millimetres (``hairline_*`` above the eye centre line)."""

    # hairline
    hairline_front: float = 68.0  # height of the front hairline above the eye centre line
    temple_recession: float = 6.0  # extra height of the temple corners (M-shaped hairline)
    sideburn_drop: float = 9.0  # the sideburn ends this far below the top of the ear
    ear_margin: float = 7.0  # hair keeps this far from the ear
    nape_above_crease: float = 20.0  # the nape hairline sits this far above the neck crease
    hairline_fade: float = 11.0  # width of the soft hairline edge (density ramps from 0 to 1 over this)
    hairline_noise: float = 1.2  # irregularity of the hairline
    # taper line between the long top and the short sides (above the eye centre line)
    taper_side: float = 50.0
    taper_back: float = 58.0
    taper_band: float = 16.0
    # strand lengths
    length_front: float = 55.0  # top, near the front hairline
    length_mid: float = 50.0  # top, middle
    length_crown: float = 30.0  # top, at the crown
    length_side: float = 8.0  # sides and back
    length_edge: float = 4.0  # sides and back at the hairline edge
    length_front_edge: float = 25.0  # top at the front hairline
    fade_band: float = 35.0  # distance above the hairline over which the sides grow from edge to full length
    # lift (fraction of the strand length that stands off the scalp at the tip)
    lift_front: float = 0.36
    lift_crown: float = 0.12
    lift_side: float = 0.05
    top_spread: float = 0.18  # sideways spreading of the combed-back top hair
    strand_noise: float = 1.0  # waviness and direction noise of the strands (1 = default texture)
    clump_noise: float = 1.0  # smooth direction noise shared by neighbouring cards (1 = default)
    # geometry
    clearance: float = 2.0  # minimum distance of every card vertex from the head surface
    card_width: float = 1.0  # multiplier of the layers' card widths
    triangle_target: int = 52000  # the spacing of the cards is adapted to land within +-12 percent of this
    seed: int = 7

    @classmethod
    def with_overrides(cls, overrides: dict | None) -> HairStyle:
        """The default style with ``overrides`` (name -> number); unknown names and non-positive values are errors."""
        names = {f.name for f in fields(cls)}
        values = {}
        for key, value in (overrides or {}).items():
            if key not in names:
                raise ValueError(f"Unknown hair parameter {key!r}; known: {', '.join(sorted(names))}")
            number = float(value)
            if not np.isfinite(number) or number < 0:
                raise ValueError(f"Hair parameter {key} must be a finite, non-negative number")
            values[key] = int(number) if key in ("triangle_target", "seed") else number
        return replace(cls(), **values)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# the card layers; counts follow from the spacing (the style's triangle target rescales it)
@dataclass(frozen=True)
class Layer:
    name: str
    spacing: float  # Poisson-disc radius, millimetres
    width: float  # card width, millimetres
    length_scale: float
    lift_scale: float
    height: float  # extra clearance, millimetres (keeps overlapping cards from being coplanar)
    top_power: float  # 0: everywhere, >0: only where the top-hair weight is high (weight ** power)
    edge_power: float = 0.8  # roots thin out towards the hairline: acceptance = density ** edge_power
    band: bool = False  # only the soft hairline band (fine stubble cards)
    min_length: float = 0.004  # metres
    droop: float = 0.30  # fraction of the lifted height the tip gives back (the tip settles towards the flow surface)
    taper: float = 0.12  # narrowing of the ribbon towards the tip: width * (1 - taper * s ** 1.8), leaf-shaped cards


LAYERS = (
    # an undercoat of short, nearly flat cards that hides the scalp, the body of the hair, the outer layer that
    # defines the silhouette (longest, narrowest cards, tips resting near the flow surface), hairline stubble
    Layer("undercoat", 3.3, 6.0, 0.60, 0.30, 0.0, 0.0, 1.2, droop=0.20, taper=0.45),
    Layer("body", 4.6, 5.2, 0.88, 0.70, 0.9, 0.0, 1.8, droop=0.30, taper=0.65),
    Layer("outer", 5.6, 4.6, 1.08, 1.00, 1.8, 0.5, 2.4, droop=0.40, taper=0.85),
    Layer("stubble", 2.4, 2.8, 0.50, 0.08, 0.0, 0.0, 1.0, band=True, min_length=0.0035, droop=0.0, taper=0.6),
)


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


class Noise:
    """Smooth 3-D value noise from a few random plane waves; roughly zero-mean with unit-ish spread."""

    def __init__(self, seed: int, terms: int = 5):
        rng = np.random.default_rng(seed)
        self.directions = unit(rng.normal(size=(terms, 3)))
        self.phase = rng.uniform(0, 2 * np.pi, terms)
        self.terms = terms

    def __call__(self, points: np.ndarray, frequency: float) -> np.ndarray:
        arg = 2 * np.pi * frequency * (np.asarray(points) @ self.directions.T) + self.phase
        return np.sin(arg).sum(1) / np.sqrt(self.terms / 2.0)


# --------------------------------------------------------------------------------------------- head surface
class HeadSurface:
    """The head triangles, welded by position (UV seam copies merged), with closest-point and signed distance."""

    def __init__(self, positions: np.ndarray, faces: np.ndarray):
        positions = np.asarray(positions, np.float64)
        inverse, first = weld(positions)
        vertices = positions[first]
        f = inverse[np.asarray(faces, np.int64)]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        self.vertices, self.faces = vertices, f
        self.normals = trimesh.Trimesh(vertices, f, process=False).vertex_normals.copy()
        self.triangles = vertices[f]
        cross = np.cross(self.triangles[:, 1] - self.triangles[:, 0], self.triangles[:, 2] - self.triangles[:, 0])
        self.area = np.linalg.norm(cross, axis=1) / 2
        self.tree = cKDTree(
            np.concatenate((self.triangles.mean(1), self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]))
        )
        edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
        graph = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
        graph = ((graph + graph.T) > 0).astype(np.float64).tocsr()
        self.graph = graph
        self.walk = sp.diags(1.0 / np.maximum(np.asarray(graph.sum(1)).ravel(), 1)) @ graph

    def closest(self, points: np.ndarray, candidates: int = 12):
        """Closest surface points: ``(point, face, bary, normal)`` with the normal interpolated from the vertices."""
        points = np.asarray(points, np.float64)
        faces = len(self.faces)
        out_p, out_f = [], []
        for start in range(0, len(points), 8192):
            chunk = points[start : start + 8192]
            _, found = self.tree.query(chunk, k=min(candidates, 4 * faces))
            found = np.asarray(found).reshape(len(chunk), -1) % faces
            tri = self.triangles[found].reshape(-1, 3, 3)
            cp = trimesh.triangles.closest_point(tri, np.repeat(chunk, found.shape[1], axis=0))
            cp = cp.reshape(len(chunk), -1, 3)
            best = np.argmin(((cp - chunk[:, None]) ** 2).sum(2), axis=1)
            out_p.append(cp[np.arange(len(chunk)), best])
            out_f.append(found[np.arange(len(chunk)), best])
        point, face = np.concatenate(out_p), np.concatenate(out_f)
        bary = trimesh.triangles.points_to_barycentric(self.triangles[face], point)
        bary = np.clip(bary, 0.0, 1.0)
        bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
        normal = unit(np.einsum("nk,nkj->nj", bary, self.normals[self.faces[face]]))
        return point, face, bary, normal

    def signed_distance(self, points: np.ndarray, candidates: int = 12):
        """``(signed distance, closest point, normal)``; negative means inside (behind the surface)."""
        point, _, _, normal = self.closest(points, candidates)
        delta = np.asarray(points, np.float64) - point
        sign = np.where(np.einsum("ij,ij->i", delta, normal) >= 0, 1.0, -1.0)
        return sign * np.linalg.norm(delta, axis=1), point, normal

    def sample(self, face_ids: np.ndarray, rng: np.random.Generator):
        """Uniform random points on the given faces: ``(points, normals)``."""
        r = rng.random((len(face_ids), 2))
        flip = r.sum(1) > 1
        r[flip] = 1 - r[flip]
        bary = np.column_stack((1 - r.sum(1), r))
        tri = self.triangles[face_ids]
        points = np.einsum("nk,nkj->nj", bary, tri)
        normals = unit(np.einsum("nk,nkj->nj", bary, self.normals[self.faces[face_ids]]))
        return points, normals

    def smoothed(self, iterations: int = 60, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
        """Taubin-smoothed vertices (low-pass shape of the head without ears and other small features)."""
        s = self.vertices.copy()
        for _ in range(iterations):
            s = s + lam * (self.walk @ s - s)
            s = s + mu * (self.walk @ s - s)
        return s


# ---------------------------------------------------------------------------------------------- head frame
@dataclass
class EarBox:
    y_top: float
    y_bottom: float
    a_front: float  # azimuth (degrees from the front, absolute) of the ear's front edge
    a_back: float
    detected: bool = False
    vertices: int = 0


@dataclass
class HeadFrame:
    eye_y: float
    x0: float
    zc: float  # centre of the skull in the horizontal plane
    top_y: float
    bottom_y: float
    front_z: float  # z of the head surface at the front hairline
    crown_z: float
    crease_y: float
    ears: dict  # +1 (the character's left, +X) / -1 -> EarBox
    ear_points: np.ndarray  # vertices of the detected ears (dilated by one ring)
    notes: dict = field(default_factory=dict)


def detect_ears(surface: HeadSurface, eye_y: float, x0: float, zc: float):
    """Ear vertices from the normal deviation against a Taubin-smoothed head (side zone, largest patch per side)."""
    v = surface.vertices
    smooth = surface.smoothed()
    smooth_normals = trimesh.Trimesh(smooth, surface.faces, process=False).vertex_normals
    angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", surface.normals, smooth_normals), -1, 1)))
    zone = (
        (np.abs(v[:, 0] - x0) > 0.045)
        & (v[:, 1] > eye_y - 0.075)
        & (v[:, 1] < eye_y + 0.05)
        & (v[:, 2] > zc - 0.07)
        & (v[:, 2] < zc + 0.045)
    )
    candidate = zone & (angle > 22.0)
    mask = np.zeros(len(v), bool)
    boxes = {}
    for side in (1, -1):
        selected = candidate & (np.sign(v[:, 0] - x0) == side)
        if selected.sum() < 12:
            continue
        sub = surface.graph[selected][:, selected]
        count, label = connected_components(sub, directed=False)
        ids = np.flatnonzero(selected)
        best = np.argmax(np.bincount(label))
        patch = ids[label == best]
        if len(patch) < 25:
            continue
        # one ring of dilation: the ear's rim
        ring = np.zeros(len(v), bool)
        ring[patch] = True
        ring = (surface.graph @ ring.astype(float) > 0) | ring
        ring &= (np.sign(v[:, 0] - x0) == side) & (np.abs(v[:, 0] - x0) > 0.04)
        mask |= ring
        pts = v[ring]
        az = np.degrees(np.arctan2(np.abs(pts[:, 0] - x0), pts[:, 2] - zc))
        boxes[side] = EarBox(
            float(np.percentile(pts[:, 1], 99)),
            float(np.percentile(pts[:, 1], 1)),
            float(np.percentile(az, 2)),
            float(np.percentile(az, 98)),
            True,
            int(ring.sum()),
        )
    return mask, boxes


def find_crease(surface: HeadSurface, x0: float, zc: float, eye_y: float) -> float:
    """Height where the back of the skull starts to bulge over the neck (the nape crease)."""
    v = surface.vertices
    sel = (np.abs(v[:, 0] - x0) < 0.02) & (v[:, 2] < zc)
    low = float(v[:, 1].min())
    fallback = eye_y - 0.065
    if sel.sum() < 20:
        return fallback
    edges = np.arange(low, eye_y + 0.012, 0.005)
    ys, zs = v[sel, 1], v[sel, 2]
    centres, depth = [], []
    for a in edges[:-1]:
        inside = (ys >= a) & (ys < a + 0.005)
        if inside.any():
            centres.append(a + 0.0025)
            depth.append(zs[inside].min())
    if len(centres) < 6:
        return fallback
    centres, depth = np.array(centres), np.array(depth)
    neck = centres < low + 0.045
    if neck.sum() < 3:
        return fallback
    reference = np.median(depth[neck])
    beyond = np.flatnonzero((centres > low + 0.025) & (depth < reference - 0.006))
    if len(beyond) == 0:
        return fallback
    k = int(beyond[0])
    crease = float(centres[k])
    if k > 0 and depth[k - 1] > depth[k]:  # interpolate where the bulge crosses 6 mm
        t = (reference - 0.006 - depth[k - 1]) / (depth[k] - depth[k - 1])
        crease = float(centres[k - 1] + np.clip(t, 0, 1) * (centres[k] - centres[k - 1]))
    return float(np.clip(crease, eye_y - 0.085, eye_y - 0.045))


def measure_head(surface: HeadSurface, eye_y: float, style: HairStyle, ears: np.ndarray | None = None) -> HeadFrame:
    v = surface.vertices
    band = (v[:, 1] > eye_y - 0.01) & (v[:, 1] < eye_y + 0.05)
    x0 = float((v[band, 0].min() + v[band, 0].max()) / 2) if band.any() else float(v[:, 0].mean())
    cross_band = (v[:, 1] > eye_y + 0.03) & (v[:, 1] < eye_y + 0.05)
    cross = v[cross_band] if cross_band.any() else v
    zc = float((cross[:, 2].min() + cross[:, 2].max()) / 2)
    hairline_y = eye_y + style.hairline_front * MM
    near = (np.abs(v[:, 0] - x0) < 0.015) & (np.abs(v[:, 1] - hairline_y) < 0.008) & (v[:, 2] > zc)
    front_z = float(np.median(v[near, 2])) if near.any() else float(cross[:, 2].max())
    notes = {}
    if ears is None:
        mask, boxes = detect_ears(surface, eye_y, x0, zc)
    else:
        mask = np.asarray(ears, bool)
        boxes = {}
        for side in (1, -1):
            pts = v[mask & (np.sign(v[:, 0] - x0) == side)]
            if len(pts) >= 25:
                az = np.degrees(np.arctan2(np.abs(pts[:, 0] - x0), pts[:, 2] - zc))
                boxes[side] = EarBox(
                    float(np.percentile(pts[:, 1], 99)),
                    float(np.percentile(pts[:, 1], 1)),
                    float(np.percentile(az, 2)),
                    float(np.percentile(az, 98)),
                    True,
                    len(pts),
                )
    for side in (1, -1):
        if side not in boxes:
            boxes[side] = EarBox(eye_y + 0.008, eye_y - 0.047, 80.0, 102.0, False, 0)
            notes[f"ear_{'left' if side > 0 else 'right'}"] = "not detected, default ear box"
    return HeadFrame(
        eye_y=float(eye_y),
        x0=x0,
        zc=zc,
        top_y=float(v[:, 1].max()),
        bottom_y=float(v[:, 1].min()),
        front_z=front_z,
        crown_z=zc - 0.02,
        crease_y=find_crease(surface, x0, zc, eye_y),
        ears=boxes,
        ear_points=v[mask],
        notes=notes,
    )


# -------------------------------------------------------------------------------------------------- field
def poisson_select(points: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Maximal Poisson-disc subset (random priorities, parallel rounds): indices, pairwise distance >= radius."""
    n = len(points)
    if n == 0:
        return np.zeros(0, np.int64)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    priority = rng.permutation(n)
    alive = np.ones(n, bool)
    chosen = []
    while alive.any():
        keep = alive[pairs[:, 0]] & alive[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
        pe = pairs[keep]
        beaten = np.zeros(n, bool)
        if len(pe):
            lower = priority[pe[:, 0]] < priority[pe[:, 1]]
            beaten[pe[lower, 0]] = True
            beaten[pe[~lower, 1]] = True
        winners = alive & ~beaten
        chosen.append(np.flatnonzero(winners))
        dead = winners.copy()
        if len(pe):
            dead[pe[winners[pe[:, 0]], 1]] = True
            dead[pe[winners[pe[:, 1]], 0]] = True
        alive &= ~dead
    return np.concatenate(chosen)


class HairField:
    """Hair density, length, lift and flow at points of the head surface (analytic, shared by hair and scalp tint)."""

    def __init__(self, frame: HeadFrame, style: HairStyle):
        self.frame, self.style = frame, style
        self.noise = Noise(style.seed + 101)
        self._ear_tree = cKDTree(frame.ear_points) if len(frame.ear_points) else None
        self.hairline = {side: self._hairline_curve(side) for side in (1, -1)}
        self.taper = {side: self._taper_curve(side) for side in (1, -1)}

    # ---- curves
    def _hairline_curve(self, side: int) -> PchipInterpolator:
        f, s = self.frame, self.style
        ear = f.ears[side]
        h0, rec = s.hairline_front * MM, s.temple_recession * MM
        top = ear.y_top - f.eye_y
        bottom = ear.y_bottom - f.eye_y
        mid = 0.5 * (top + bottom)
        nape = f.crease_y - f.eye_y + s.nape_above_crease * MM
        sideburn = top - s.sideburn_drop * MM
        ef = float(np.clip(ear.a_front, 68.0, 96.0))
        eb = float(np.clip(ear.a_back, ef + 12.0, 128.0))
        a = [0.0, 22.0, 38.0, ef - 22.0, ef - 14.0, ef - 8.0, ef + 1.0, 0.5 * (ef + eb), eb - 1.0, eb + 7.0, eb + 17.0]
        a += [eb + 33.0, 150.0, 180.0]
        h = [h0, h0 + 0.001, h0 + rec, top + 0.045, top + 0.014, sideburn, top + 0.006, top + 0.006, top + 0.006]
        h += [mid + 0.004, max(bottom + 0.008, nape - 0.004), nape + 0.012, nape + 0.006, nape - 0.004]
        a, h = np.array(a), np.array(h)
        order = np.argsort(a)
        a, h = a[order], h[order]
        keep = np.concatenate(([True], np.diff(a) > 0.5))
        return PchipInterpolator(a[keep], h[keep])

    def _taper_curve(self, side: int) -> PchipInterpolator:
        s = self.style
        h0, rec = s.hairline_front * MM, s.temple_recession * MM
        hair = self.hairline[side]
        ts, tb = s.taper_side * MM, s.taper_back * MM
        a = np.array([0.0, 38.0, 56.0, 80.0, 120.0, 160.0, 180.0])
        t = np.array(
            [
                h0 - 0.004,
                h0 + rec - 0.004,
                max(ts + 0.006, float(hair(56.0)) + 0.010),
                ts,
                ts + 0.004,
                tb,
                tb,
            ]
        )
        return PchipInterpolator(a, t)

    # ---- coordinates
    def azimuth(self, p: np.ndarray) -> np.ndarray:
        """Signed azimuth in degrees around the skull centre: 0 front, +90 the character's left, +-180 the back."""
        return np.degrees(np.arctan2(p[:, 0] - self.frame.x0, p[:, 2] - self.frame.zc))

    def _by_side(self, curves: dict, p: np.ndarray) -> np.ndarray:
        a = self.azimuth(p)
        side = a >= 0
        absolute = np.abs(a)
        return np.where(side, curves[1](absolute), curves[-1](absolute)) + self.frame.eye_y

    def hairline_y(self, p: np.ndarray) -> np.ndarray:
        return self._by_side(self.hairline, p)

    def taper_y(self, p: np.ndarray) -> np.ndarray:
        return self._by_side(self.taper, p)

    def ear_distance(self, p: np.ndarray) -> np.ndarray:
        if self._ear_tree is None:
            return np.full(len(p), 1.0)
        return self._ear_tree.query(p, workers=-1)[0]

    # ---- fields
    def height_above_hairline(self, p: np.ndarray) -> np.ndarray:
        wobble = self.style.hairline_noise * MM * self.noise(p, 70.0)
        return p[:, 1] + wobble - self.hairline_y(p)

    def weight(self, p: np.ndarray) -> np.ndarray:
        """Hair density 0..1: 0.5 on the hairline, 0 below and around the ears, 1 inside."""
        fade = max(self.style.hairline_fade * MM, 1e-4)
        inside = smoothstep(self.height_above_hairline(p) / fade + 0.5)
        ear = smoothstep((self.ear_distance(p) - self.style.ear_margin * MM) / 0.004)
        return inside * ear

    def cover(self, p: np.ndarray) -> np.ndarray:
        """Darkening of the skin texture: full under the hair, a faint stubble shadow just outside the hairline."""
        wide = smoothstep(self.height_above_hairline(p) / 0.014 + 0.9)
        ear = smoothstep((self.ear_distance(p) - 0.004) / 0.004)
        return np.maximum(self.weight(p), 0.32 * wide * ear)

    def top_weight(self, p: np.ndarray) -> np.ndarray:
        band = max(self.style.taper_band * MM, 1e-4)
        return smoothstep((p[:, 1] - self.taper_y(p)) / band + 0.5)

    def ap_position(self, p: np.ndarray) -> np.ndarray:
        """0 at the crown, 1 at the front hairline along the head."""
        f = self.frame
        return np.clip((p[:, 2] - f.crown_z) / max(f.front_z - f.crown_z, 1e-3), 0.0, 1.0)

    def lengths(self, p: np.ndarray) -> np.ndarray:
        s = self.style
        above = np.maximum(self.height_above_hairline(p) + 0.5 * s.hairline_fade * MM, 0.0)
        tau = self.top_weight(p)
        u = self.ap_position(p)
        rear = smoothstep(u / 0.65)
        front = smoothstep((u - 0.65) / 0.35)
        top = np.where(
            u < 0.65,
            s.length_crown + (s.length_mid - s.length_crown) * rear,
            s.length_mid + (s.length_front - s.length_mid) * front,
        )
        top = np.minimum(top, s.length_front_edge + 1.2 * above / MM)
        near_ear = 0.55 + 0.45 * smoothstep(self.ear_distance(p) / 0.025)
        side = s.length_edge + (s.length_side - s.length_edge) * smoothstep(above / max(s.fade_band * MM, 1e-4))
        side = side * near_ear
        return (tau * top + (1 - tau) * side) * MM

    def lifts(self, p: np.ndarray) -> np.ndarray:
        s = self.style
        tau = self.top_weight(p)
        u = self.ap_position(p)
        top = s.lift_crown + (s.lift_front - s.lift_crown) * smoothstep(u)
        return tau * top + (1 - tau) * s.lift_side

    def flow(self, p: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Unit combing direction in the surface tangent plane: top swept back, sides and back combed down."""
        a = np.abs(self.azimuth(p))
        sgn = np.sign(p[:, 0] - self.frame.x0)
        tau = self.top_weight(p)
        spread = self.style.top_spread * np.clip(np.abs(p[:, 0] - self.frame.x0) / 0.06, 0, 1) * sgn
        top = np.column_stack((spread, np.zeros(len(p)), -np.ones(len(p))))
        back = smoothstep((a - 95.0) / 40.0)
        side = np.column_stack((-sgn * 0.18 * back, np.full(len(p), -0.88), -0.40 * (1 - back)))
        direction = tau[:, None] * top + (1 - tau[:, None]) * side
        tangent = direction - normals * np.einsum("ij,ij->i", direction, normals)[:, None]
        length = np.linalg.norm(tangent, axis=1)
        fallback = unit(np.array([0.0, -1.0, 0.0]) - normals * normals[:, 1:2])
        return np.where((length < 1e-3)[:, None], fallback, tangent / np.maximum(length, 1e-12)[:, None])


# -------------------------------------------------------------------------------------------------- cards
def sample_roots(surface: HeadSurface, field_: HairField, layer: Layer, scale: float, rng: np.random.Generator):
    """Poisson-disc roots of one layer: candidates on the scalp triangles, thinned by the density, then spaced."""
    centroid = surface.triangles.mean(1)
    face_w = field_.weight(centroid)
    corner_w = np.max([field_.weight(surface.triangles[:, k]) for k in range(3)], axis=0)
    faces = np.flatnonzero((face_w > 0.01) | (corner_w > 0.05))
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    radius = layer.spacing * MM * scale
    area = float(surface.area[faces].sum())
    count = int(np.ceil(4.0 * area / (radius * radius)))
    pick = rng.choice(faces, size=count, p=surface.area[faces] / surface.area[faces].sum())
    points, normals = surface.sample(pick, rng)
    w = field_.weight(points)
    if layer.band:
        accept = rng.random(count) < np.clip(0.6 * (1.0 - w), 0.0, 1.0) * smoothstep(w / 0.25)
    else:
        accept = rng.random(count) < w**layer.edge_power
    if layer.top_power > 0:
        accept &= rng.random(count) < field_.top_weight(points) ** layer.top_power
    points, normals = points[accept], normals[accept]
    keep = poisson_select(points, radius, rng)
    return points[keep], normals[keep]


def segments_for(length: np.ndarray) -> np.ndarray:
    return np.clip(np.ceil(length / 0.0105).astype(int), 1, 6)


def lift_profile(u: np.ndarray, droop: float = 0.28) -> np.ndarray:
    """Height fraction at relative arclength ``u``: rises at the root, flattens, then the tip settles back by ``droop``."""
    return (1 - (1 - u) ** 1.6) * (1 - droop * smoothstep((u - 0.55) / 0.45))


def lift_gate(normal: np.ndarray) -> np.ndarray:
    """0.25..1: hair stands off surfaces that face up or forward, and lies closer on the lateral flanks of the vault
    (a lift along a sideways normal would push tips out of the silhouette as spikes)."""
    facing = normal[:, 1] + 0.8 * np.maximum(normal[:, 2], 0.0)
    return 0.25 + 0.75 * smoothstep((facing - 0.25) / 0.55)


def rotate_about(v: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * c + np.cross(axis, v) * s + axis * np.einsum("ij,ij->i", axis, v)[:, None] * (1 - c)


def grow(
    surface: HeadSurface,
    field_: HairField,
    roots: np.ndarray,
    root_normals: np.ndarray,
    length: np.ndarray,
    lift: np.ndarray,
    clear0: np.ndarray,
    segments: int,
    rng: np.random.Generator,
    droop: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Strand polylines ``(n, K + 1, 3)`` and the surface normals under every point, grown along the flow field."""
    n = len(roots)
    noise = field_.noise
    style = field_.style
    top = field_.top_weight(roots)
    tex = style.strand_noise * (0.4 + 0.6 * top)  # sides and back lie neater than the top
    clump = style.clump_noise
    # smooth fields shared by neighbouring roots (clumps, parting) dominate; the per-card part is small, so tips do not
    # fan out into spikes
    bias = clump * (
        np.radians(6.5) * noise(roots, 46.0) + np.radians(3.5) * noise(roots + 0.37, 26.0)
    ) + tex * rng.normal(0.0, np.radians(1.4), n)
    wave = tex * rng.uniform(np.radians(1.0), np.radians(4.5), n)
    wave_phase = rng.uniform(0, 2 * np.pi, n)
    up = np.array([0.0, 1.0, 0.0])
    pos = np.zeros((n, segments + 1, 3))
    nrm = np.zeros((n, segments + 1, 3))
    foot, normal = roots.copy(), root_normals.copy()
    pos[:, 0] = foot + normal * clear0[:, None]
    nrm[:, 0] = normal
    previous = clear0.copy()
    step = length / segments
    height_total = lift * length
    for k in range(1, segments + 1):
        u = np.full(n, k / segments)
        height = clear0 + height_total * lift_profile(u, droop) * lift_gate(normal)
        rise = height - previous
        travel = np.sqrt(np.maximum(step**2 - rise**2, (0.3 * step) ** 2))
        direction = field_.flow(foot, normal)
        angle = bias + wave * np.sin(2 * np.pi * (u * 1.3) + wave_phase)
        direction = rotate_about(direction, normal, angle)
        moved = foot + direction * travel[:, None]
        foot, _, _, normal = surface.closest(moved, candidates=10)
        # lean the lift slightly upwards (a quiff stands up instead of leaning over the forehead)
        lean = unit(normal + 0.15 * up * (field_.top_weight(foot)[:, None]))
        factor = np.maximum(np.einsum("ij,ij->i", lean, normal), 0.6)
        pos[:, k] = foot + lean * (height / factor)[:, None]
        nrm[:, k] = normal
        previous = height
    return pos, nrm


def build_ribbons(
    pos: np.ndarray,
    nrm: np.ndarray,
    width: np.ndarray,
    roll: np.ndarray,
    taper: float = 0.12,
    blend: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Ribbon vertices and normals ``(n, 2 * (K + 1), 3)`` (left, right per ring) with parallel-transported frames.

    The normal of a vertex is the card's own normal (facing away from the scalp) mixed with the surface normal under
    it (``blend`` is the scalp share), so overlapping cards shade like one smooth volume instead of flat facets.
    """
    n, rings, _ = pos.shape
    k = rings - 1
    segment = unit(pos[:, 1:] - pos[:, :-1])
    tangent = np.zeros_like(pos)
    tangent[:, 0] = segment[:, 0]
    tangent[:, k] = segment[:, -1]
    if k > 1:
        tangent[:, 1:k] = unit(segment[:, :-1] + segment[:, 1:])
    side = np.zeros_like(pos)
    first = np.cross(nrm[:, 0], tangent[:, 0])
    weak = np.linalg.norm(first, axis=1) < 1e-3
    if weak.any():
        first[weak] = np.cross(np.array([0.0, 1.0, 0.0]), tangent[weak, 0])
    side[:, 0] = unit(first)
    for i in range(1, rings):
        previous = side[:, i - 1]
        side[:, i] = unit(previous - tangent[:, i] * np.einsum("ij,ij->i", previous, tangent[:, i])[:, None])
    # a constant roll about the strand varies the cards' tilt (overlapping cards never coincide)
    c, s = np.cos(roll)[:, None, None], np.sin(roll)[:, None, None]
    side = side * c + np.cross(tangent, side) * s
    along = np.linspace(0.0, 1.0, rings)[None, :]
    half = 0.5 * width[:, None] * (1.0 - taper * along**1.8)
    left = pos - side * half[..., None]
    right = pos + side * half[..., None]
    own = np.cross(side, tangent)  # the card's own normal; the scalp normal decides which of the two is "outwards"
    own = own * np.where(np.einsum("nkj,nkj->nk", own, nrm) < 0, -1.0, 1.0)[..., None]
    smooth = unit(blend * nrm + (1.0 - blend) * own)
    vertices = np.stack((left, right), axis=2).reshape(n, 2 * rings, 3)
    normals = np.stack((smooth, smooth), axis=2).reshape(n, 2 * rings, 3)
    return vertices, normals


def ribbon_faces(rings: int) -> np.ndarray:
    """Triangles of one ribbon with the vertex order (left, right) per ring; the normal faces away from the scalp."""
    k = np.arange(rings - 1)
    i0, i1, i2, i3 = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
    return np.stack((np.stack((i0, i3, i1), 1), np.stack((i0, i2, i3), 1)), axis=1).reshape(-1, 3)


def push_out(
    surface: HeadSurface, positions: np.ndarray, faces: np.ndarray, clearance: float, passes: int = 3
) -> np.ndarray:
    """Keep every card vertex (and the interior of every card triangle) at least ``clearance`` from the head.

    A vertex that is closer (or inside) moves along the line from its closest surface point (the surface normal
    when it is inside). Triangle centroids and edge midpoints that dip below the clearance lift their vertices.
    """
    out = positions.copy()
    eps = 2e-5
    for _ in range(passes):
        distance, closest, normal = surface.signed_distance(out, candidates=24)
        bad = distance < clearance - 1e-6
        away = out - closest
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        direction = np.where((distance >= 0)[:, None] & (norm > 1e-9), away / np.maximum(norm, 1e-12), normal)
        out[bad] = closest[bad] + direction[bad] * (clearance + eps)
        near = np.arange(len(faces))
        tri = out[faces[near]]
        samples = np.concatenate(
            (tri.mean(1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 2] + tri[:, 0]) / 2)
        )
        inner, _, _ = surface.signed_distance(samples, candidates=24)
        deficit = np.maximum(clearance + eps - inner, 0.0).reshape(4, -1).max(0)
        lift = np.zeros(len(out))
        for k in range(3):
            np.maximum.at(lift, faces[near, k], deficit)
        move = lift > 1e-7
        if not bad.any() and not move.any():
            break
        out[move] += normal[move] * lift[move, None]
    return out


def card_slots(
    layout: StripLayout,
    lengths: np.ndarray,
    rng: np.random.Generator,
    long_from: float = 0.032,
    mid_from: float = 0.014,
) -> np.ndarray:
    """``(cards, 4)``: ``u_left, u_right, v_root, v_tip`` of a random slot of the matching class (u mirrored half of the time).

    The slot's root row is on top (``v_root < v_tip``): the atlas G channel is 0 there and 1 at the tip row.
    """
    out = np.zeros((len(lengths), 4))
    kinds = np.where(lengths >= long_from, 2, np.where(lengths >= mid_from, 1, 0))
    for code, kind in ((2, "long"), (1, "mid"), (0, "short")):
        mask = kinds == code
        slots = layout.slots(kind)
        count = int(mask.sum())
        if count == 0:
            continue
        pick = rng.integers(0, len(slots), count)
        boxes = np.array([slots[i].uv_box(layout.width, layout.height) for i in pick])  # (u0, v_root, u1, v_tip)
        mirror = rng.random(count) < 0.5
        left = np.where(mirror, boxes[:, 2], boxes[:, 0])
        right = np.where(mirror, boxes[:, 0], boxes[:, 2])
        out[mask] = np.column_stack((left, right, boxes[:, 1], boxes[:, 3]))
    return out


@dataclass
class LayerPlan:
    """Everything decided before growing: roots, lengths, lifts, widths and strip slots of one layer."""

    index: int
    roots: np.ndarray
    normals: np.ndarray
    length: np.ndarray
    lift: np.ndarray
    clearance: np.ndarray
    width: np.ndarray
    roll: np.ndarray
    segments: np.ndarray
    slots: np.ndarray
    droop: float = 0.3
    taper: float = 0.12


@dataclass
class Plan:
    layers: list
    scale: float

    @property
    def triangles(self) -> int:
        return int(sum(2 * p.segments.sum() for p in self.layers))

    @property
    def cards(self) -> int:
        return int(sum(len(p.roots) for p in self.layers))


def plan_cards(
    surface: HeadSurface, field_: HairField, style: HairStyle, layout: StripLayout, scale: float = 1.0
) -> Plan:
    """Roots and per-card parameters of all layers for the spacing ``scale`` (1.0 = the layers' nominal spacing)."""
    rng = np.random.default_rng(style.seed)
    plans = []
    for index, layer in enumerate(LAYERS):
        roots, normals = sample_roots(surface, field_, layer, scale, rng)
        if len(roots) == 0:
            continue
        n = len(roots)
        jitter = np.clip(1.0 + 0.12 * field_.noise(roots, 48.0) + rng.normal(0.0, 0.07, n), 0.75, 1.3)
        density = field_.weight(roots)
        length = np.maximum(field_.lengths(roots) * layer.length_scale * jitter, layer.min_length)
        plans.append(
            LayerPlan(
                index,
                roots,
                normals,
                length,
                field_.lifts(roots) * layer.lift_scale * rng.uniform(0.8, 1.2, n),
                (style.clearance + layer.height + rng.uniform(0.0, 0.8, n)) * MM,
                layer.width
                * MM
                * style.card_width
                * rng.uniform(0.85, 1.2, n)
                * (0.5 + 0.5 * smoothstep(density / 0.9)),
                rng.normal(0.0, np.radians(8.0), n),
                segments_for(length),
                card_slots(layout, length, rng),
                layer.droop,
                layer.taper,
            )
        )
    return Plan(plans, scale)


@dataclass
class Cards:
    positions: np.ndarray
    faces: np.ndarray
    uv: np.ndarray  # atlas UV (0..1), root on the slot's top row
    normals: np.ndarray  # smooth vertex normals (card normal blended with the scalp normal)
    layer_of_face: np.ndarray
    card_of_face: np.ndarray
    roots: np.ndarray  # (cards, 3)
    root_normals: np.ndarray
    lengths: np.ndarray  # (cards,)
    layer_of_card: np.ndarray
    segments: np.ndarray  # (cards,)


def grow_cards(surface: HeadSurface, field_: HairField, style: HairStyle, plan: Plan) -> Cards:
    """Grow every planned guide and turn it into a ribbon; vertices below the clearance are pushed out afterwards."""
    rng = np.random.default_rng(style.seed + 1)
    positions, normals_out, faces, uvs, layer_of_face, card_of_face = [], [], [], [], [], []
    roots, root_normals, lengths, layer_of_card, segment_count = [], [], [], [], []
    vertex_total = card_total = 0
    for p in plan.layers:
        for k in np.unique(p.segments):
            g = np.flatnonzero(p.segments == k)
            k = int(k)
            pos, nrm = grow(
                surface, field_, p.roots[g], p.normals[g], p.length[g], p.lift[g], p.clearance[g], k, rng, p.droop
            )
            vertices, vertex_normals = build_ribbons(pos, nrm, p.width[g], p.roll[g], p.taper)
            vertices = vertices.reshape(-1, 3)
            rings, n = k + 1, len(g)
            tri = np.tile(ribbon_faces(rings), (n, 1)) + np.repeat(np.arange(n) * 2 * rings, 2 * k)[:, None]
            slot = p.slots[g]
            v_coord = slot[:, 2:3] + (slot[:, 3:4] - slot[:, 2:3]) * np.linspace(0.0, 1.0, rings)[None, :]
            uv = np.empty((n, rings, 2, 2))
            uv[:, :, 0, 0] = slot[:, 0:1]
            uv[:, :, 1, 0] = slot[:, 1:2]
            uv[:, :, :, 1] = v_coord[:, :, None]
            positions.append(vertices)
            normals_out.append(vertex_normals.reshape(-1, 3))
            faces.append(tri + vertex_total)
            uvs.append(uv.reshape(-1, 2))
            layer_of_face.append(np.full(len(tri), p.index))
            card_of_face.append(np.repeat(card_total + np.arange(n), 2 * k))
            roots.append(p.roots[g])
            root_normals.append(p.normals[g])
            lengths.append(p.length[g])
            layer_of_card.append(np.full(n, p.index))
            segment_count.append(np.full(n, k))
            vertex_total += len(vertices)
            card_total += n
    all_faces = np.vstack(faces).astype(np.int64)
    all_positions = push_out(surface, np.vstack(positions), all_faces, style.clearance * MM)
    return Cards(
        all_positions,
        all_faces,
        np.vstack(uvs),
        unit(np.vstack(normals_out)),
        np.concatenate(layer_of_face),
        np.concatenate(card_of_face),
        np.vstack(roots),
        np.vstack(root_normals),
        np.concatenate(lengths),
        np.concatenate(layer_of_card),
        np.concatenate(segment_count),
    )
