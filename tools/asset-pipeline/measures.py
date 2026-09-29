"""measures.json generation: measurement loops and landmarks derived from geometry (plane slices) on the neutral body.

Nothing is copied from MakeHuman's measurement plugin: landmark planes come from joint points, skin-weight body
regions and mesh topology; MakeHuman's own vertex lists were only used (read-only) to sanity-check the results.

Measure semantics (must match the runtime, see docs/ARCHITECTURE.md):
  circumference   convex hull of the loop vertices projected onto the plane of the ordered loop (Newell's method),
                  perimeter of the hull (tape-measure semantics); loops are stored in angular order
  distance        Euclidean distance of 2 vertices, or |difference| along `axis` if given
  polyline        sum of consecutive vertex distances (shoulder: acromion -> across the upper back over C7 -> acromion)
  height          bbox Y extent of the render vertices
  vertexHeight    y of one vertex minus the lowest render vertex (floor)
All vertex indices live in the combined index space (render vertices, then joint points).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mh_obj import BaseMesh

SIDE_L = 1.0  # MakeHuman: character's left = +X

# Which modifiers drive which measure. Drivers of one measure are meant to move together (same slider value).
DRIVERS: dict[str, list[str]] = {
    "height": ["armslegs/upperlegs-height", "armslegs/lowerlegs-height"],
    "neck": ["measure/measure-neck-circ"],
    "shoulder": ["measure/measure-shoulder-dist"],
    "chest": ["measure/measure-bust-circ"],
    "waist": ["measure/measure-waist-circ"],
    "hip": ["measure/measure-hips-circ"],
    "thigh": ["measure/measure-thigh-circ"],
    "upperArm": ["measure/measure-upperarm-circ"],
    "armLength": ["measure/measure-upperarm-length", "measure/measure-lowerarm-length"],
    "inseam": ["measure/measure-upperleg-height", "measure/measure-lowerleg-height"],
    "footLength": ["armslegs/foot-scale-depth"],
}

MEASURE_ORDER = ["height", "neck", "shoulder", "chest", "waist", "hip", "thigh", "upperArm", "armLength",
                 "inseam", "footLength"]


# ---------------------------------------------------------------------------------------------------------------
# Evaluation (numpy reference of the runtime semantics; used for derivation and tests)
# ---------------------------------------------------------------------------------------------------------------


def _hull_perimeter(pts: np.ndarray) -> float:
    p = sorted(set(map(tuple, np.round(pts, 9).tolist())))
    if len(p) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    h = np.array(lower[:-1] + upper[:-1])
    return float(np.linalg.norm(h - np.roll(h, -1, axis=0), axis=1).sum())


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u)], axis=1)  # (3, 2)


def newell_normal(points: np.ndarray) -> np.ndarray:
    """Plane normal (area vector) of the closed polygon through the ordered points (Newell's method)."""
    return np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)


def circumference(points: np.ndarray) -> float:
    """Tape-measure length: perimeter of the convex hull of the ordered loop projected on its Newell plane."""
    n = newell_normal(points)
    if np.linalg.norm(n) < 1e-12:  # degenerate ordering -> fall back to the least-squares plane
        c = points - points.mean(axis=0)
        n = np.linalg.eigh(c.T @ c)[1][:, 0]
    return _hull_perimeter((points - points.mean(axis=0)) @ _plane_basis(n))


def order_loop(points: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Order loop vertex ids (ascending, closed loop, first vertex not repeated) by angle around the loop centroid in
    the least-squares plane, as needed by Newell's method. `points` = positions of `ids` (same order)."""
    c = points - points.mean(axis=0)
    normal = np.linalg.eigh(c.T @ c)[1][:, 0]
    uv = c @ _plane_basis(normal)
    angle = np.arctan2(uv[:, 1], uv[:, 0])
    ordered = ids[np.lexsort((np.hypot(uv[:, 0], uv[:, 1]), angle))]
    # one consistent direction for every loop: the Newell normal points up (+Y), i.e. counter-clockwise seen from above
    if newell_normal(points[np.searchsorted(ids, ordered)])[1] < 0:
        ordered = ordered[::-1]
    return ordered


def evaluate(defn: dict, P: np.ndarray, render_count: int) -> float:
    """Value in meters of one measure definition on combined positions P (V, 3)."""
    t = defn["type"]
    if t == "circumference":
        return circumference(P[defn["verts"]])
    if t == "distance":
        a, b = P[defn["verts"][0]], P[defn["verts"][1]]
        if "axis" in defn:
            k = "xyz".index(defn["axis"])
            return float(abs(a[k] - b[k]))
        return float(np.linalg.norm(a - b))
    if t == "polyline":
        pts = P[defn["verts"]]
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if t == "height":
        y = P[:render_count, 1]
        return float(y.max() - y.min())
    if t == "vertexHeight":
        return float(P[defn["vert"], 1] - P[:render_count, 1].min())
    raise ValueError(t)


# ---------------------------------------------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Context:
    mesh: BaseMesh
    P: np.ndarray  # (V, 3) neutral body, combined space, meters (feet ~ y 0)
    joint_names: list[str]
    bone_names: list[str]
    joints: np.ndarray  # (R, 4) skin joint ids
    weights: np.ndarray  # (R, 4)
    nipple_y: float | None = None  # bust level in meters (same frame as P); default 0.727 * height

    def __post_init__(self) -> None:
        R = self.mesh.render_count
        self.R = R
        dom = np.array(self.bone_names)[self.joints[np.arange(R), self.weights.argmax(axis=1)]]
        self.dom = dom
        q = self.mesh.quads
        e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
        self.edges = np.unique(np.sort(e, axis=1), axis=0)
        # first render copy of every MakeHuman vertex (loops use one copy per position)
        self.first_copy = self.mesh.mh_to_render_ids[self.mesh.mh_to_render_start[:13380]]

    def bones_mask(self, *keys: str, exact: bool = False) -> np.ndarray:
        if exact:
            return np.isin(self.dom, keys)
        return np.array([any(k in d for k in keys) for d in self.dom])

    def joint(self, name: str) -> np.ndarray:
        return self.P[self.R + self.joint_names.index(name)]

    def slice_ring(self, normal: np.ndarray, origin: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Render ids of the vertices nearest to the plane on each crossed quad edge (masked), one per position."""
        n = normal / np.linalg.norm(normal)
        s = (self.P[: self.R] - origin) @ n
        a, b = self.edges[:, 0], self.edges[:, 1]
        crossed = (s[a] * s[b] < 0) | ((s[a] == 0) & (s[b] != 0))
        ea, eb = a[crossed], b[crossed]
        pick = np.where(np.abs(s[ea]) <= np.abs(s[eb]), ea, eb)
        ids = np.unique(pick)
        ids = ids[mask[ids]]
        mh = np.unique(self.mesh.render_mh[ids])
        ids = np.sort(self.first_copy[mh])
        return order_loop(self.P[ids], ids) if len(ids) >= 3 else ids

    def mirror(self, render_id: int) -> int:
        """Render id of the mirrored (x -> -x) vertex on the raw base mesh (exact match required)."""
        v = self.mesh.verts
        mh = self.mesh.render_mh[render_id]
        target = v[mh] * np.array([-1.0, 1.0, 1.0])
        d = np.linalg.norm(v[:13380] - target, axis=1)
        j = int(d.argmin())
        if d[j] > 1e-3:
            raise ValueError("no exact mirror vertex")
        return int(self.first_copy[j])


def _horizontal_search(ctx: Context, mask: np.ndarray, ys: np.ndarray, pick: str) -> tuple[float, np.ndarray]:
    best = None
    up = np.array([0.0, 1.0, 0.0])
    for y in ys:
        ids = ctx.slice_ring(up, np.array([0.0, y, 0.0]), mask)
        if len(ids) < 8:
            continue
        val = circumference(ctx.P[ids])
        key = val if pick == "min" else -val
        if best is None or key < best[0] - 1e-12:
            best = (key, y, ids)
    if best is None:
        raise ValueError("no valid slice")
    return best[1], best[2]


def _c7_vertex(ctx: Context) -> int:
    """Midline back vertex (x = 0, behind the body axis) closest in height to the neck joint (base of the neck)."""
    ren = ctx.P[: ctx.R]
    y = ctx.joint("joint-neck")[1]
    mid = np.where((np.abs(ren[:, 0]) < 1e-4) & (ren[:, 2] < 0.0))[0]
    return int(mid[np.abs(ren[mid, 1] - y).argmin()])


def _back_path(ctx: Context, acro_l: int, acro_r: int, c7: int) -> list[int]:
    """Ordered render vertices (acro_l first, acro_r last) of the back-surface slice through acro_l, c7, acro_r.

    One vertex (nearest to the plane) per crossed quad edge, restricted to the shoulder girdle / upper back bones,
    to |x| <= the acromion x and to the back side of the chord acromion -> C7 -> acromion; ordered by decreasing x.
    """
    ren = ctx.P[: ctx.R]
    pa, pb, pc = ren[acro_l], ren[acro_r], ren[c7]
    n = np.cross(pb - pa, pc - pa)
    n /= np.linalg.norm(n)
    mask = ctx.bones_mask(
        "spine_02", "spine_03", "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r", "neck_01", exact=True
    )
    s = (ren - pa) @ n
    a, b = ctx.edges[:, 0], ctx.edges[:, 1]
    crossed = (s[a] * s[b] < 0) | ((s[a] == 0) & (s[b] != 0))
    ea, eb = a[crossed], b[crossed]
    ids = np.unique(np.where(np.abs(s[ea]) <= np.abs(s[eb]), ea, eb))
    ids = ids[mask[ids]]
    half = abs(pa[0])

    def chord_z(x: np.ndarray) -> np.ndarray:
        t = np.clip(np.abs(x) / half, 0.0, 1.0)
        return pc[2] * (1.0 - t) + pa[2] * t

    ids = ids[(np.abs(ren[ids, 0]) <= half + 1e-6) & (ren[ids, 2] <= chord_z(ren[ids, 0]) + 0.004)]
    mh = np.unique(ctx.mesh.render_mh[ids])
    ids = np.sort(ctx.first_copy[mh])
    ids = ids[(ids != acro_l) & (ids != acro_r)]
    ids = ids[np.argsort(-ren[ids, 0], kind="stable")]
    return [acro_l, *[int(i) for i in ids], acro_r]


def derive(ctx: Context) -> tuple[list[dict], dict]:
    P, R = ctx.P, ctx.R
    ren = P[:R]
    floor = ren[:, 1].min()
    H = ren[:, 1].max() - floor
    left = ren[:, 0] > 1e-4
    ys = lambda lo, hi: np.arange(lo, hi, 0.0025)  # noqa: E731
    lm: dict = {}

    # --- landmarks ------------------------------------------------------------------------------------------
    # crotch: lowest vertex on the mid-sagittal plane (legs are separate below it, so nothing lower is on x = 0)
    mid = np.where(np.abs(ren[:, 0]) < 1e-4)[0]
    crotch = int(mid[ren[mid, 1].argmin()])
    crotch_y = ren[crotch, 1]
    lm["crotch"] = crotch

    trunk = ctx.bones_mask("pelvis", "spine_01", "spine_02", "spine_03", exact=True)
    legs = ctx.bones_mask("thigh", "calf", "foot", "ball")
    head_neck = ctx.bones_mask("neck_01", "head", exact=True)

    out: dict[str, dict] = {}

    # height: bbox of the render vertices
    out["height"] = {"id": "height", "type": "height"}

    # neck: narrowest ring between the neck base and the point where the jaw starts to intrude
    jn = ctx.joint("joint-neck")[1]
    y_neck, ids = _horizontal_search(
        ctx, (trunk | head_neck) & (np.abs(ren[:, 0]) < 0.12), ys(jn - 0.01, jn + 0.045), "min"
    )
    out["neck"] = {"id": "neck", "type": "circumference", "verts": ids.tolist()}

    # shoulder: acromion = highest shoulder-cap vertex in a narrow window just lateral/posterior to the shoulder joint
    sj = ctx.joint("joint-l-shoulder")
    win = (
        (ren[:, 0] > sj[0]) & (ren[:, 0] < sj[0] + 0.02) & (ren[:, 1] > sj[1]) & (ren[:, 2] <= sj[2] - 0.02)
        & ctx.bones_mask("upperarm_l", "clavicle_l", "spine_03", exact=True)
    )
    cand = np.where(win)[0]
    acro_l = int(cand[ren[cand, 1].argmax()])
    acro_r = ctx.mirror(acro_l)
    lm["acromion_l"], lm["acromion_r"] = acro_l, acro_r
    # shoulder width as garment size charts define it: measured ACROSS THE BACK SURFACE from one acromion over the
    # upper back at C7 to the other acromion (tape-measure path, longer than the straight biacromial distance).
    # C7 = midline back vertex at the height of the neck joint. The path is the slice of the back surface by the plane
    # through both acromia and C7, listed from the subject's left acromion (+X) to the right one (-X).
    c7 = _c7_vertex(ctx)
    lm["c7"] = c7
    path = _back_path(ctx, acro_l, acro_r, c7)
    out["shoulder"] = {"id": "shoulder", "type": "polyline", "verts": path}

    # chest: bust level = height of the nipples (located by the caller from the nipple-point target); if absent it
    # falls back to 0.727 * height, which is where the nipples sit on the neutral body
    y_bust = ctx.nipple_y if ctx.nipple_y is not None else floor + 0.727 * H
    ids = ctx.slice_ring(np.array([0.0, 1.0, 0.0]), np.array([0.0, y_bust, 0.0]), trunk)
    out["chest"] = {"id": "chest", "type": "circumference", "verts": ids.tolist()}
    lm["bust_y"] = y_bust

    # waist: narrowest torso between hip and bust level
    y_waist, ids = _horizontal_search(ctx, trunk, ys(floor + 0.58 * H, floor + 0.70 * H), "min")
    out["waist"] = {"id": "waist", "type": "circumference", "verts": ids.tolist()}
    lm["waist_y"] = y_waist

    # hip: widest trunk + upper-thigh ring above the crotch
    y_hip, ids = _horizontal_search(ctx, trunk | legs, ys(crotch_y + 0.02, crotch_y + 0.14), "max")
    out["hip"] = {"id": "hip", "type": "circumference", "verts": ids.tolist()}
    lm["hip_y"] = y_hip

    # thigh: left leg ring just below the crotch
    ids = ctx.slice_ring(np.array([0.0, 1.0, 0.0]), np.array([0.0, crotch_y - 0.03, 0.0]), legs & left)
    out["thigh"] = {"id": "thigh", "type": "circumference", "verts": ids.tolist()}

    # upper arm: left arm ring at mid upper arm, perpendicular to the shoulder->elbow axis
    elbow, wrist = ctx.joint("joint-l-elbow"), ctx.joint("joint-l-hand")
    axis = elbow - sj
    ids = ctx.slice_ring(axis, (sj + elbow) / 2, ctx.bones_mask("upperarm_l", exact=True) & left)
    out["upperArm"] = {"id": "upperArm", "type": "circumference", "verts": ids.tolist()}

    # arm length: acromion -> elbow (olecranon side) -> wrist (ulnar side)
    arm = ctx.bones_mask("upperarm_l", "lowerarm_l", "hand_l", exact=True) & left

    def outer_vertex(joint: np.ndarray, axis_v: np.ndarray, direction: np.ndarray) -> int:
        ring = ctx.slice_ring(axis_v, joint, arm)
        rel = ren[ring] - joint
        return int(ring[(rel @ (direction / np.linalg.norm(direction))).argmax()])

    elbow_v = outer_vertex(elbow, axis, np.array([0.5, 0.0, -1.0]))
    wrist_v = outer_vertex(wrist, wrist - elbow, np.array([1.0, 0.0, 0.0]))
    out["armLength"] = {"id": "armLength", "type": "polyline", "verts": [acro_l, elbow_v, wrist_v]}
    lm["elbow_v"], lm["wrist_v"] = elbow_v, wrist_v

    # inseam: crotch height above the floor
    out["inseam"] = {"id": "inseam", "type": "vertexHeight", "vert": crotch}

    # foot length: heel (min z) to toe (max z) of the left foot
    foot = np.where(ctx.bones_mask("foot_l", "ball_l", exact=True) & left)[0]
    heel = int(foot[ren[foot, 2].argmin()])
    toe = int(foot[ren[foot, 2].argmax()])
    out["footLength"] = {"id": "footLength", "type": "distance", "verts": [heel, toe], "axis": "z"}
    lm["heel"], lm["toe"] = heel, toe

    measures = []
    for mid_ in MEASURE_ORDER:
        d = dict(out[mid_])
        d["drivers"] = list(DRIVERS[mid_])
        measures.append(d)
    return measures, lm


def nipple_height(ctx_mesh: BaseMesh, P: np.ndarray, idx: np.ndarray, delta: np.ndarray) -> float:
    """Height of the nipples: the body vertex moved most by the `nipple-point` target (and its x-mirror)."""
    body = idx < 13380
    i, d = idx[body], delta[body]
    tip = int(i[np.linalg.norm(d, axis=1).argmax()])
    first = ctx_mesh.mh_to_render_ids[ctx_mesh.mh_to_render_start[:13380]]
    return float(P[first[tip], 1])
