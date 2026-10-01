"""Separate the arms from the torso where a scan has them fused (armpit webbing).

Why. A scan whose arms hang against the torso is one continuous surface across the contact strip (sleeve / upper arm
against the chest wall): a surface that pinches between the arm and the torso along two crease lines (front and
back) and has no triangles inside the strip itself. Skinned with a T-pose, the faces next to the crease lines are
stretched into a sheet between the arm and the torso. The rig stage cannot fix that (its bridge cutter only handles
parts that are far apart in the skeleton), so the refine stage changes the mesh:

1. Label every scan vertex arm / not-arm with the fitted MakeHuman body (the skin weights the rig stage will compute:
   upper arm + forearm + hand + fingers), clean the labels (largest connected arm patch, no islands).
2. Cut the surface along the arm / torso interface inside the contact zone (below the armpit apex taken from the
   fitted body, above the free part of the arm). The cut is an open arc: the surface stays attached above it (the
   shoulder keeps deforming like a shoulder) and two lips open below it.
3. Close both lips with a small cap each (the arm's slot, the torso's window), so both sides stay closed surfaces. Cap
   triangles use a solid colour patch in unused atlas space (the lips' mean colour).
4. Nudge the arm side outward a few millimetres (tapering to zero at the apex and below the contact) so the two lips
   are no longer the same points: the rig stage welds vertices by position, which would stitch them together again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from twintex.colorspace import linear_to_u8, u8_to_linear  # noqa: E402  (texture stage)

from . import log
from .atlaspatch import PatchAllocator
from .meshops import (
    Corners,
    EdgeTable,
    chain_halfedges,
    cut_along_edges,
    edge_endpoints,
    edge_table,
    face_components,
    refine_marked_edges,
    triangulate_loop_3d,
    zipper_triangulate,
)

LEFT, RIGHT = 1, 2


@dataclass
class ArmpitParams:
    y_top: float | None = None  # cut zone top (m); default: lowest point of the fitted body's arm interface + apex_margin
    apex_margin: float = 0.03
    y_low: float = 1.02  # no cut below this height (hands touching thighs are not handled here)
    gap_mm: float = 4.0  # outward nudge of the arm side
    taper_top: float = 0.05  # m below y_top over which the nudge fades in
    taper_bottom: float = 0.08  # m over which the nudge fades out below the lowest cut edge
    min_cut_edges: int = 6
    smooth_iters: int = 6  # Laplacian smoothing of the arm / not-arm field before cutting along its zero level
    chain_smooth: int = 4  # Laplacian iterations along the cut line (straightens the lips)
    t_clip: float = 0.12  # keep the inserted cut vertices at least this fraction away from the old vertices


@dataclass
class ArmpitReport:
    sides: dict = field(default_factory=dict)
    y_top: float = 0.0


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def vertex_labels(aw_l: np.ndarray, aw_r: np.ndarray) -> np.ndarray:
    lab = np.zeros(len(aw_l), dtype=np.int8)
    lab[aw_l > 0.5] = LEFT
    lab[aw_r > 0.5] = RIGHT
    return lab


def face_labels(F: np.ndarray, vlab: np.ndarray) -> np.ndarray:
    lv = vlab[F]
    out = np.zeros(len(F), dtype=np.int8)
    for L in (LEFT, RIGHT):
        out[(lv == L).sum(axis=1) >= 2] = L
    return out


def clean_face_labels(F: np.ndarray, flab: np.ndarray, et: EdgeTable) -> np.ndarray:
    """Keep the largest connected patch of every arm label, absorb isolated not-arm islands into the arm."""
    flab = flab.copy()
    for L in (LEFT, RIGHT):
        n, comp = face_components(F, flab == L, et)
        if n > 1:
            sizes = np.bincount(comp[comp >= 0])
            keep = int(np.argmax(sizes))
            flab[(flab == L) & (comp != keep)] = 0
    # not-arm islands: components of label 0 other than the largest take the majority label of their neighbours
    n, comp = face_components(F, flab == 0, et)
    if n > 1:
        sizes = np.bincount(comp[comp >= 0])
        big = int(np.argmax(sizes))
        h = np.flatnonzero(et.he_pair >= 0)
        f1, f2 = h // 3, et.he_pair[h] // 3
        for c in range(n):
            if c == big:
                continue
            inside = comp == c
            nb = flab[f2[inside[f1] & (flab[f2] != 0)]]
            if len(nb):
                flab[inside] = np.bincount(nb, minlength=3).argmax()
    return flab


def apex_height(model, res) -> float:
    """Lowest point of the arm / trunk interface of the fitted (posed) MakeHuman body: the armpit apex."""
    from twintex import bake  # noqa: F401  (keeps the import graph explicit)

    nr = model.nr
    Wt = np.zeros((nr, len(model.bones)))
    for c in range(4):
        np.add.at(Wt, (np.arange(nr), model.skin_j[:, c]), model.skin_w[:, c])
    pv = res.posed_vertices
    f = model.faces
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    ys = []
    from .bodyfit import arm_bone_ids

    for side in ("l", "r"):
        lab = Wt[:, arm_bone_ids(model, side)].sum(axis=1) > 0.5
        m = lab[e[:, 0]] != lab[e[:, 1]]
        ev = np.unique(e[m])
        ev = ev[pv[ev, 1] > 1.0]  # ignore the hands
        if len(ev):
            ys.append(float(pv[ev, 1].min()))
    return float(np.mean(ys)) if ys else 1.28


def is_new_cut_vertex(is_new: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[: len(is_new)] = is_new
    return out


def smooth_field(F: np.ndarray, phi: np.ndarray, iters: int, alpha: float = 0.6) -> np.ndarray:
    """Laplacian smoothing of a vertex field on the welded mesh (uniform weights)."""
    from scipy.sparse import coo_matrix

    et = edge_table(F)
    lo = np.zeros(et.n_edges, dtype=np.int64)
    hi = np.zeros(et.n_edges, dtype=np.int64)
    lo[et.edge_of] = np.minimum(et.he_a, et.he_b)
    hi[et.edge_of] = np.maximum(et.he_a, et.he_b)
    n = len(phi)
    A = coo_matrix((np.ones(2 * len(lo)), (np.r_[lo, hi], np.r_[hi, lo])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
    out = phi.astype(np.float64).copy()
    for _ in range(iters):
        out = (1 - alpha) * out + alpha * (A @ out) / deg
    return out


def separate_arms(
    mc: Corners,
    atlas: np.ndarray | None,
    aw_l: np.ndarray,
    aw_r: np.ndarray,
    y_apex: float,
    params: ArmpitParams | None = None,
) -> tuple[Corners, np.ndarray | None, ArmpitReport]:
    """Cut + cap + nudge. ``aw_l`` / ``aw_r``: arm skin weight of every welded vertex (row = ``mc.P`` row)."""
    prm = params or ArmpitParams()
    rep = ArmpitReport()
    y_top = prm.y_top if prm.y_top is not None else y_apex + prm.apex_margin
    rep.y_top = float(y_top)
    et = edge_table(mc.F)
    vlab = vertex_labels(aw_l, aw_r)
    flab = clean_face_labels(mc.F, face_labels(mc.F, vlab), et)

    # ---- smooth arm / not-arm field from the cleaned labels; its zero level is the cut line
    phi0 = np.full(len(mc.P), -1.0)
    for L in (LEFT, RIGHT):
        phi0[mc.F[flab == L].reshape(-1)] = 1.0
    # vertices touched by both kinds of faces sit on the interface
    touch_arm = np.zeros(len(mc.P), dtype=bool)
    touch_non = np.zeros(len(mc.P), dtype=bool)
    touch_arm[mc.F[flab > 0].reshape(-1)] = True
    touch_non[mc.F[flab == 0].reshape(-1)] = True
    phi0[touch_arm & touch_non] = 0.0
    phi = smooth_field(mc.F, phi0, prm.smooth_iters)
    phi[np.abs(phi) < 1e-6] = 1e-6

    # ---- insert vertices on the zero level inside the zone
    lo, hi = edge_endpoints(et)
    zlo, zhi = phi[lo], phi[hi]
    ymid = 0.5 * (mc.P[lo, 1] + mc.P[hi, 1])
    zone_e = (ymid >= prm.y_low - 0.02) & (ymid <= y_top + 0.015)
    crossing = (zlo * zhi < 0) & zone_e
    t = np.clip(zlo / np.where(crossing, zlo - zhi, 1.0), prm.t_clip, 1.0 - prm.t_clip)
    mc2, is_new = refine_marked_edges(mc, crossing, et, t)
    phi2 = np.concatenate([phi, np.zeros(int(is_new.sum()))])
    orig = ~is_new
    fs = np.where(orig[mc2.F], phi2[mc2.F], 0.0).sum(axis=1)
    arm_face = fs > 0

    # ---- cut edges: interior edges between arm faces and not-arm faces inside the zone
    et2 = edge_table(mc2.F)
    h = np.flatnonzero(et2.he_pair >= 0)
    f1, f2 = h // 3, et2.he_pair[h] // 3
    differ = arm_face[f1] != arm_face[f2]
    ymid2 = 0.5 * (mc2.P[et2.he_a[h], 1] + mc2.P[et2.he_b[h], 1])
    zone = (ymid2 <= y_top) & (ymid2 >= prm.y_low)
    sel = h[differ & zone]
    cut_edge = np.zeros(et2.n_edges, dtype=bool)
    cut_edge[et2.edge_of[sel]] = True
    n_cut = int(cut_edge.sum())
    if prm.chain_smooth > 0 and n_cut:
        # straighten the cut line (the scan is noisy at the crease): Laplacian smoothing along the chain of cut edges
        a_e = np.zeros(et2.n_edges, dtype=np.int64)
        b_e = np.zeros(et2.n_edges, dtype=np.int64)
        a_e[et2.edge_of] = et2.he_a
        b_e[et2.edge_of] = et2.he_b
        ce = np.flatnonzero(cut_edge)
        nbr_sum = np.zeros_like(mc2.P)
        nbr_cnt = np.zeros(len(mc2.P))
        for u, v in ((a_e[ce], b_e[ce]), (b_e[ce], a_e[ce])):
            np.add.at(nbr_sum, u, mc2.P[v])
            np.add.at(nbr_cnt, u, 1.0)
        on_chain = (nbr_cnt == 2) & is_new_cut_vertex(is_new, len(mc2.P))
        P_s = mc2.P.copy()
        for _ in range(prm.chain_smooth):
            nbr_sum = np.zeros_like(P_s)
            for u, v in ((a_e[ce], b_e[ce]), (b_e[ce], a_e[ce])):
                np.add.at(nbr_sum, u, P_s[v])
            avg = nbr_sum / np.maximum(nbr_cnt, 1.0)[:, None]
            P_s = np.where(on_chain[:, None], 0.5 * P_s + 0.5 * avg, P_s)
        mc2 = Corners(P_s, mc2.F, mc2.C)
    rep.sides["cut_edges"] = n_cut
    if n_cut < prm.min_cut_edges:
        log(f"armpit: only {n_cut} interface edges in the zone, nothing to separate")
        return mc, atlas, rep
    log(f"armpit: zone y <= {y_top:.3f} m, {n_cut} interface edges cut")
    cmc, _src = cut_along_edges(mc2, cut_edge, et2)

    # ---- caps
    F2 = cmc.F
    P2 = cmc.P
    hes = np.flatnonzero(cut_edge[et2.edge_of])  # half-edges of cut edges (faces keep their index): both sides
    alloc = PatchAllocator(atlas, cmc.C) if atlas is not None else None
    cap_faces: list[np.ndarray] = []
    cap_uv: list[np.ndarray] = []
    cap_is_arm: list[bool] = []
    extra_pts: list[np.ndarray] = []
    nP = len(P2)
    stats = {"fan_fallback": 0, "ear_clipped": 0, "zippered": 0}

    def cap_batch(path: list[int], closed: bool, rgb: tuple[int, int, int] | None, is_arm: bool) -> None:
        nonlocal nP
        tris_ids = None
        if not closed:
            tris_ids = zipper_triangulate(path, P2)
            if tris_ids is not None:
                stats["zippered"] += 1
        if tris_ids is None:
            # cycle reversed against the lip's half-edge direction (the open ends are joined by a chord)
            cyc = np.array(list(reversed(path)), dtype=np.int64)
            tris, extra = triangulate_loop_3d(P2[cyc])
            if extra is not None:
                cyc = np.concatenate([cyc, [nP]])
                extra_pts.append(extra)
                nP += 1
                stats["fan_fallback"] += 1
            else:
                stats["ear_clipped"] += 1
            tris_ids = cyc[tris]
        cap_faces.append(tris_ids)
        uvc = alloc.paint(rgb) if (alloc is not None and rgb is not None) else None
        cap_uv.append(np.tile(uvc if uvc is not None else np.zeros(2, np.float32), (len(tris_ids), 3, 1)))
        cap_is_arm.append(is_arm)

    def lip_colour(hs: np.ndarray) -> tuple[int, int, int] | None:
        if atlas is None or len(hs) == 0:
            return None
        fa, ka = hs // 3, hs % 3
        uv = mc2.C[fa, ka]
        S = atlas.shape[0]
        px = np.clip((uv[:, 0] * S).astype(int), 0, S - 1)
        py = np.clip((uv[:, 1] * S).astype(int), 0, S - 1)
        col = u8_to_linear(atlas[py, px]).mean(axis=0)
        return tuple(int(x) for x in linear_to_u8(col[None, :])[0])

    arm_hs = hes[arm_face[hes // 3]]
    torso_hs = hes[~arm_face[hes // 3]]
    for hs, is_arm in ((arm_hs, True), (torso_hs, False)):
        rgb = lip_colour(hs)
        for path, closed in chain_halfedges(F2, hs):
            if len(path) >= 3:
                cap_batch(path, closed, rgb, is_arm)
    if extra_pts:
        P2 = np.vstack([P2, *extra_pts])
    n_main = len(F2)
    F3 = np.vstack([F2, *cap_faces]) if cap_faces else F2
    C3 = np.vstack([cmc.C, *cap_uv]) if cap_faces else cmc.C
    rep.sides["cap_faces"] = int(len(F3) - n_main)
    rep.sides.update(stats)

    # ---- nudge the arm side outward
    nV = len(P2)
    arm_flag = np.concatenate([arm_face, np.concatenate([np.full(len(cf), a) for cf, a in zip(cap_faces, cap_is_arm, strict=True)]) if cap_faces else np.zeros(0, bool)])
    arm_v = np.zeros(nV, dtype=bool)
    non_arm_v = np.zeros(nV, dtype=bool)
    arm_v[F3[arm_flag].reshape(-1)] = True
    non_arm_v[F3[~arm_flag].reshape(-1)] = True
    ys = P2[:, 1]
    cut_ys = 0.5 * (mc2.P[et2.he_a[sel], 1] + mc2.P[et2.he_b[sel], 1])
    y_lo_cut = float(cut_ys.min())
    omega = smoothstep(y_top - ys, 0.0, prm.taper_top) * smoothstep(ys - (y_lo_cut - prm.taper_bottom), 0.0, prm.taper_bottom)
    P3 = P2.copy()
    gap = prm.gap_mm * 1e-3
    mv = arm_v & ~non_arm_v
    sgn = np.where(P3[:, 0] >= 0, 1.0, -1.0)
    P3[mv, 0] += sgn[mv] * gap * omega[mv]
    rep.sides["nudged"] = int(mv.sum())
    return Corners(P3, F3, C3), atlas, rep
