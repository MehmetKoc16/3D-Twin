"""Fit our MakeHuman body (shape + pose) to a scan, then transfer skin weights and unpose the scan.

Pipeline (see README.md):
  1. articulated + parametric ICP: alternating pose LM (per-bone rotations, LBS) and shape solve
     (macro variables + bounded modifier least squares) against the scan point cloud;
  2. skin weights: closest-point transfer from the posed template (normal gated, densified),
     Laplacian smoothing on the scan mesh, top-4;
  3. unpose the scan with the estimated pose so the output rests in the exact MakeHuman A-pose frame the
     pose library was authored for; joints come from the fitted body's joint points.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.optimize import least_squares, lsq_linear, minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from mh import MHModel

POSE_BONES = [
    "spine_01", "spine_02", "spine_03", "neck_01", "head",
    "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
    "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r",
]  # fmt: skip
POSE_SIGMA = {  # prior std-dev (rad) of each bone's local rotation
    "spine_01": 0.08, "spine_02": 0.08, "spine_03": 0.08, "neck_01": 0.15, "head": 0.2,
    "clavicle_l": 0.12, "clavicle_r": 0.12, "upperarm_l": 0.6, "upperarm_r": 0.6,
    "lowerarm_l": 0.5, "lowerarm_r": 0.5, "hand_l": 0.3, "hand_r": 0.3,
    "thigh_l": 0.3, "thigh_r": 0.3, "calf_l": 0.25, "calf_r": 0.25, "foot_l": 0.2, "foot_r": 0.2,
}  # fmt: skip
MACRO_FIT = ["gender", "muscle", "weight", "height"]
# The point-to-nearest-point data term cannot see height (surfaces slide tangentially), so without a constraint the
# macro/modifier solve drifts to a body several cm shorter than the scan. The fitted body's head-top to sole height is
# therefore tied to the scan's height minus what hair adds above the skull; the scan's lowest point (sole of the shoe)
# is the body's sole, so only the hair is subtracted here. twin_export.CLOTHING_ALLOWANCE_CM["height"] covers hair + sole.
HAIR_M = 0.015
HEIGHT_WEIGHT = 5.0  # weight of the height row, as a fraction of the summed correspondence weights


@dataclass
class FitResult:
    macro: dict[str, float]
    mods: dict[str, float]
    rest_positions: np.ndarray  # (N,3) combined space, template rest pose, ungrounded
    pose_rotvec: dict[str, np.ndarray]
    root_t: np.ndarray
    posed_vertices: np.ndarray  # (nr,3) template posed to the scan
    stats: dict = field(default_factory=dict)


def log(msg: str) -> None:
    print(f"[rigfit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rots_from_rotvec(rv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: R.from_rotvec(v).as_matrix() for k, v in rv.items()}


class Fitter:
    def __init__(self, model: MHModel, scan_pts: np.ndarray, scan_nrm: np.ndarray | None,
                 *, fixed_shape: tuple[dict[str, float], dict[str, float]] | None = None) -> None:
        self.m = model
        self.pts = scan_pts
        self.nrm = scan_nrm
        self.tree = cKDTree(scan_pts)
        self.macro = {k: model.macro_vars[k]["default"] for k in MACRO_FIT}
        self.z = np.zeros(0)
        self.cols = model.fit_modifier_columns()
        cols_idx = [c[2] for c in self.cols]
        # dense rest-space columns for the render vertices (+ joint rows kept separately)
        Dcols = model.D[:, cols_idx].toarray()
        n = model.base.shape[0]
        self.A_all = Dcols.reshape(n, 3, len(cols_idx))  # (N,3,C)
        self.z = np.zeros(len(cols_idx))
        self.zmax = np.array(
            [model.modifiers[model.mod_index[c[0]]]["max" if c[1] == "incr" else "min"] for c in self.cols], dtype=float
        )
        self.zmax = np.abs(self.zmax)
        # macro-only target columns
        w0 = model.macro_target_weights({})
        self.macro_targets = np.nonzero(
            np.array([bool(t.get("macroConditions")) for t in model.manifest["targets"]])
        )[0]
        self.Dm = model.D[:, self.macro_targets]
        self.rv = {b: np.zeros(3) for b in POSE_BONES}
        self.root_t = np.zeros(3)
        self.stats: dict = {}
        self.fixed_shape = fixed_shape
        if fixed_shape is not None:
            self.macro = dict(fixed_shape[0])
        _ = w0

    # ---------------------------------------------------------------- shape helpers
    def macro_positions(self, macro: dict[str, float]) -> np.ndarray:
        wfull = self.m.macro_target_weights(macro)
        return self.m.base + (self.Dm @ wfull[self.macro_targets]).reshape(-1, 3)

    def shape_positions(self) -> np.ndarray:
        if self.fixed_shape is not None:
            return self.m.shape(self.macro, self.fixed_shape[1], ground=False)
        return self.macro_positions(self.macro) + np.einsum("nkc,c->nk", self.A_all, self.z)

    # ---------------------------------------------------------------- pose helpers
    def pose_matrices(self, rv: dict[str, np.ndarray], heads: np.ndarray):
        return self.m.fk(heads, _rots_from_rotvec(rv), self.root_t)

    def posed_vertices(self, pos: np.ndarray, rv=None) -> np.ndarray:
        rv = self.rv if rv is None else rv
        heads = self.m.rest_heads(pos)
        rots, posed = self.pose_matrices(rv, heads)
        return self.m.lbs(pos[: self.m.nr], self.m.skin_j, self.m.skin_w, heads, rots, posed)

    # ---------------------------------------------------------------- init
    def init_alignment(self) -> None:
        m = self.m
        top = self.pts[:, 1].max()
        bot = self.pts[:, 1].min()
        # height macro by bisection so the template height matches the scan (ignores hair and pose)
        target_h = top - bot
        self.target_body_h = target_h - HAIR_M

        def height(hv: float) -> float:
            mc = dict(self.macro)
            mc["height"] = hv
            p = self.macro_positions(mc)[: m.nr]
            return p[:, 1].max() - p[:, 1].min()

        lo, hi = 0.0, 1.0
        if self.fixed_shape is None:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if height(mid) < self.target_body_h:
                    lo = mid
                else:
                    hi = mid
            self.macro["height"] = 0.5 * (lo + hi)
        p = self.shape_positions()[: m.nr]
        # translate: feet to scan bottom, xz centroid to scan centroid (torso band)
        self.root_t = np.array([0.0, bot - p[:, 1].min(), 0.0])
        band = (self.pts[:, 1] > 0.3 * target_h) & (self.pts[:, 1] < 0.6 * target_h)
        sc = self.pts[band][:, [0, 2]].mean(axis=0)
        fc = p[m.faces].mean(axis=1)  # area-weighted centroid of the same height band (vertex density is non-uniform)
        area = 0.5 * np.linalg.norm(np.cross(p[m.faces[:, 1]] - p[m.faces[:, 0]], p[m.faces[:, 2]] - p[m.faces[:, 0]]), axis=1)
        tb = (fc[:, 1] > p[:, 1].min() + 0.3 * target_h) & (fc[:, 1] < p[:, 1].min() + 0.6 * target_h)
        tc = (fc[tb][:, [0, 2]] * area[tb, None]).sum(axis=0) / area[tb].sum()
        self.root_t[[0, 2]] = sc - tc
        log(f"init: scan height {target_h:.3f} m, height macro {self.macro['height']:.3f}, root_t {self.root_t.round(3)}")

    def _group_weight(self, bone_names: list[str]) -> np.ndarray:
        m = self.m
        ids = [m.bone_index[n] for n in bone_names]
        w = np.zeros(m.nr)
        for k in range(4):
            w += np.where(np.isin(m.skin_j[:, k], ids), m.skin_w[:, k], 0.0)
        return w

    def limb_search(self) -> None:
        """Coarse 2-DOF grid search of each limb segment (swing about z = abduction, about x = forward/back) by
        symmetric chamfer distance: template limb -> scan + scan points the rest of the template does not explain
        -> template limb, so a limb hidden inside the torso is not rewarded.

        Chains: upperarm (scored on upperarm+lowerarm+hand) then lowerarm (lowerarm+hand); thigh then calf.
        """
        m = self.m
        pos = self.shape_positions()
        fingers = ("index", "middle", "ring", "pinky", "thumb")
        sub = self.pts[:: max(1, len(self.pts) // 30000)]
        for side, sx in (("l", 1.0), ("r", -1.0)):
            hand = [f"hand_{side}"] + [f"{f}_0{i}_{side}" for f in fingers for i in (1, 2, 3)]
            chains = [
                (f"upperarm_{side}", [f"upperarm_{side}", f"lowerarm_{side}", *hand], range(-70, 26, 5), range(-50, 21, 10)),
                (f"lowerarm_{side}", [f"lowerarm_{side}", *hand], range(-30, 31, 10), range(-100, 21, 10)),
                (f"thigh_{side}", [f"thigh_{side}", f"calf_{side}", f"foot_{side}", f"ball_{side}"], range(-25, 26, 5), range(-30, 31, 10)),
            ]
            for bone, group, zr, xr in chains:
                w = self._group_weight(group)
                limb = np.nonzero(w > 0.9)[0][::2]
                rest = np.nonzero(w < 0.1)[0][::2]
                vp0 = self.posed_vertices(pos)
                dU, _ = cKDTree(vp0[rest]).query(sub)
                U = sub[(dU > 0.04) & (sub[:, 0] * sx > 0.03)]
                best = (1e9, 0, 0)
                for dz in zr:
                    for dx in xr:
                        rv = {b: v.copy() for b, v in self.rv.items()}
                        rv[bone] = R.from_euler("xyz", [dx, 0, dz], degrees=True).as_rotvec()
                        vl = self.posed_vertices(pos, rv)[limb]
                        d1, _ = self.tree.query(vl)
                        score = float(np.mean(np.minimum(d1, 0.15) ** 2))
                        if len(U):
                            d2, _ = cKDTree(vl).query(U)
                            score += float(np.mean(np.minimum(d2, 0.15) ** 2))
                        if score < best[0]:
                            best = (score, dx, dz)
                self.rv[bone] = R.from_euler("xyz", [best[1], 0, best[2]], degrees=True).as_rotvec()
                log(f"limb search {bone}: x {best[1]} z {best[2]} deg (score {np.sqrt(best[0]) * 100:.1f} cm, {len(U)} unexplained pts)")

    # ---------------------------------------------------------------- correspondences
    def correspondences(self, vp: np.ndarray, normals: np.ndarray | None, gate: float):
        d, j = self.tree.query(vp)
        w = 1.0 / (1.0 + (d / (0.5 * gate)) ** 2)
        w[d > gate] = 0.0
        if normals is not None and self.nrm is not None:
            dots = np.einsum("ij,ij->i", normals, self.nrm[j])
            w = w * np.clip((dots + 0.2) / 0.6, 0.0, 1.0)
        return self.pts[j], w, d

    # ---------------------------------------------------------------- pose LM
    def update_pose(self, pos: np.ndarray, gate: float, iters: int = 8, sub: int = 2500) -> None:
        m = self.m
        rng = np.random.default_rng(1)
        sel = rng.choice(m.nr, size=min(sub, m.nr), replace=False)
        heads = m.rest_heads(pos)
        base = pos[: m.nr][sel]
        sj, sw = m.skin_j[sel], m.skin_w[sel]
        # vertex normals for gating
        tm = trimesh.Trimesh(pos[: m.nr], m.faces, process=False)
        nrm_rest = tm.vertex_normals[sel]
        sigma = np.array([POSE_SIGMA[b] for b in POSE_BONES])
        kappa = 0.0035

        def unpack(x):
            rv = {b: x[3 * i : 3 * i + 3] for i, b in enumerate(POSE_BONES)}
            return rv, x[3 * len(POSE_BONES) :]

        x0 = np.concatenate([np.concatenate([self.rv[b] for b in POSE_BONES]), self.root_t])
        state = {"c": None, "w": None}

        def residual(x):
            rv, t = unpack(x)
            old = self.root_t
            self.root_t = t
            rots, posed = m.fk(heads, _rots_from_rotvec(rv), t)
            self.root_t = old
            vp = m.lbs(base, sj, sw, heads, rots, posed)
            c, w = state["c"], state["w"]
            r = ((vp - c) * np.sqrt(w / len(sel))[:, None]).ravel()
            prior = (x[: 3 * len(POSE_BONES)].reshape(-1, 3) / sigma[:, None]).ravel() * kappa
            return np.concatenate([r, prior])

        x = x0
        for it in range(iters):
            rv, t = unpack(x)
            rots, posed = m.fk(heads, _rots_from_rotvec(rv), t)
            vp = m.lbs(base, sj, sw, heads, rots, posed)
            # posed normals ~ rotate rest normals by blended rotation
            A, _b = m.blended_affine(sj, sw, heads, rots, posed)
            nn = np.einsum("nij,nj->ni", A, nrm_rest)
            nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
            c, w, _d = self.correspondences(vp, nn, gate)
            state["c"], state["w"] = c, w
            res = least_squares(residual, x, method="trf", loss="soft_l1", f_scale=0.02, max_nfev=4, x_scale=0.1)
            dx = np.abs(res.x - x).max()
            x = res.x
            if dx < 2e-4:
                break
        rv, t = unpack(x)
        self.rv = {b: v.copy() for b, v in rv.items()}
        self.root_t = t.copy()

    # ---------------------------------------------------------------- shape solve
    def update_shape(self, gate: float, ridge: float, macro_iters: int = 1) -> None:
        m = self.m
        pos = self.shape_positions()
        heads = m.rest_heads(pos)
        rots, posed = self.pose_matrices(self.rv, heads)
        vp = m.lbs(pos[: m.nr], m.skin_j, m.skin_w, heads, rots, posed)
        tm = trimesh.Trimesh(pos[: m.nr], m.faces, process=False)
        A, b = m.blended_affine(m.skin_j, m.skin_w, heads, rots, posed)
        nn = np.einsum("nij,nj->ni", A, tm.vertex_normals)
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        c, w, d = self.correspondences(vp, nn, gate)
        # targets back in the rest frame
        tgt = np.linalg.solve(A, (c - b)[..., None])[..., 0]
        sw_ = np.sqrt(w)[:, None]
        Ar = self.A_all[: m.nr]  # (nr,3,C)
        ncols = Ar.shape[2]
        h = 0.1
        for _ in range(macro_iters):
            mp = self.macro_positions(self.macro)[: m.nr]
            # macro variables enter as extra columns: one-sided steps up and down (the tent weights are piecewise linear)
            mcols, mvars = [], []
            for k in MACRO_FIT:
                for sgn in (1.0, -1.0):
                    v2 = float(np.clip(self.macro[k] + sgn * h, 0.0, 1.0))
                    step = abs(v2 - self.macro[k])
                    if step < 1e-6:
                        mcols.append(np.zeros((m.nr, 3)))
                    else:
                        mcols.append((self.macro_positions({**self.macro, k: v2})[: m.nr] - mp) * (h / step))
                    mvars.append((k, sgn))
            Mc = np.stack(mcols, axis=2)  # (nr,3,8)
            Aall = np.concatenate([Ar, Mc], axis=2)
            ntot = Aall.shape[2]
            r = (tgt - mp) * sw_
            Aw = (Aall * sw_[:, :, None]).reshape(-1, ntot)
            # height row: (top vertex y - sole vertex y) of the rest body equals the target body height
            it, ib = int(np.argmax(mp[:, 1])), int(np.argmin(mp[:, 1]))
            sh = np.sqrt(HEIGHT_WEIGHT * float(np.sum(w)))
            Ah = (Aall[it, 1, :] - Aall[ib, 1, :]) * sh
            rh = (self.target_body_h - (mp[it, 1] - mp[ib, 1])) * sh
            Aw = np.vstack([Aw, Ah[None, :]])
            AtA = Aw.T @ Aw
            diag = np.concatenate([np.full(ncols, ridge), np.full(len(mvars), ridge * 0.02)]) * np.trace(AtA[:ncols, :ncols]) / ncols
            H = AtA + np.diag(diag)
            g = Aw.T @ np.concatenate([r.reshape(-1), [rh]])
            L = np.linalg.cholesky(H)
            rhs = np.linalg.solve(L, g)
            ub = np.concatenate([self.zmax, np.full(len(mvars), 3.0)])
            res = lsq_linear(L.T, rhs, bounds=(np.zeros(ntot), ub), method="bvls")
            self.z = res.x[:ncols]
            for (k, sgn), a in zip(mvars, res.x[ncols:]):
                self.macro[k] = float(np.clip(self.macro[k] + sgn * h * a, 0.0, 1.0))
        bp = self.shape_positions()[: m.nr]
        log(f"  body height {bp[:, 1].max() - bp[:, 1].min():.3f} m (target {self.target_body_h:.3f})")
        mask = w > 0
        self.stats["shape_rms_cm"] = float(np.sqrt(np.sum(w * d**2) / max(np.sum(w), 1e-9)) * 100)
        self.stats["shape_coverage"] = float(mask.mean())

    # ---------------------------------------------------------------- driver
    def run(self) -> FitResult:
        m = self.m
        self.init_alignment()
        self.limb_search()
        schedule = [(0.20, 0.3), (0.14, 0.1), (0.10, 0.05), (0.07, 0.02), (0.05, 0.01), (0.04, 0.005), (0.035, 0.003)]
        for gi, (gate, ridge) in enumerate(schedule):
            pos = self.shape_positions()
            self.update_pose(pos, gate)
            if self.fixed_shape is None:
                self.update_shape(gate, ridge, macro_iters=2)
            pos = self.shape_positions()
            vp = self.posed_vertices(pos)
            d, _ = self.tree.query(vp)
            log(
                f"iter {gi} gate {gate:.2f} ridge {ridge:g}: template->scan median {np.median(d) * 100:.2f} cm, "
                f"p90 {np.percentile(d, 90) * 100:.2f} cm; macro { {k: round(v, 2) for k, v in self.macro.items()} }"
            )
        # final pose polish with tight gate
        pos = self.shape_positions()
        self.update_pose(pos, 0.04, iters=10)
        vp = self.posed_vertices(pos)
        mods: dict[str, float] = {}
        for (mid, kind, _t), zv in zip(self.cols, self.z):
            mods[mid] = mods.get(mid, 0.0) + (zv if kind == "incr" else -zv)
        if self.fixed_shape is not None:
            mods = dict(self.fixed_shape[1])
            self.stats["shape_source"] = "bodyfix"
        d, _ = self.tree.query(vp)
        d2, _ = cKDTree(vp).query(self.pts[:: max(1, len(self.pts) // 20000)])
        self.stats.update(
            {
                "template_to_scan_cm": {"median": float(np.median(d) * 100), "p90": float(np.percentile(d, 90) * 100)},
                "scan_to_template_cm": {"median": float(np.median(d2) * 100), "p90": float(np.percentile(d2, 90) * 100)},
                "pose_deg": {b: np.degrees(v).round(1).tolist() for b, v in self.rv.items() if np.linalg.norm(v) > 0.01},
            }
        )
        return FitResult(dict(self.macro), mods, pos, {b: v.copy() for b, v in self.rv.items()}, self.root_t.copy(), vp, self.stats)


# --------------------------------------------------------------------------------------- skinning transfer
def _densify(verts: np.ndarray, faces: np.ndarray, weights: np.ndarray):
    """One midpoint subdivision of the template with interpolated per-bone weights."""
    tm = trimesh.Trimesh(verts, faces, process=False)
    nv, nf = trimesh.remesh.subdivide(tm.vertices, tm.faces)
    # subdivide appends midpoints in unique-edge order; recover the weights by averaging the endpoint weights
    n0 = len(verts)
    edges = tm.edges_unique  # trimesh.remesh.subdivide appends one midpoint per unique edge, same ordering
    ok = len(nv) == n0 + len(edges) and np.allclose(nv[n0:], 0.5 * (verts[edges[:, 0]] + verts[edges[:, 1]]), atol=1e-9)
    if ok:
        w_new = np.vstack([weights, 0.5 * (weights[edges[:, 0]] + weights[edges[:, 1]])])
    else:  # ordering assumption failed: nearest-vertex weights
        _, nn = cKDTree(verts).query(nv)
        w_new = weights[nn]
    return nv, nf, w_new


def transfer_weights(
    model: MHModel,
    posed_template: np.ndarray,
    scan_verts: np.ndarray,
    scan_faces: np.ndarray,
    smooth_iters: int = 6,
    smooth_alpha: float = 0.5,
    k: int = 24,
):
    nb = len(model.bones)
    Wt = np.zeros((model.nr, nb))
    for c in range(4):
        np.add.at(Wt, (np.arange(model.nr), model.skin_j[:, c]), model.skin_w[:, c])
    tv, tf, tw = _densify(posed_template, model.faces, Wt)
    ttm = trimesh.Trimesh(tv, tf, process=False)
    tnrm = ttm.vertex_normals
    stm = trimesh.Trimesh(scan_verts, scan_faces, process=False)
    snrm = stm.vertex_normals
    tree = cKDTree(tv)
    dist, idx = tree.query(scan_verts, k=k)
    dots = np.einsum("nkj,nj->nk", tnrm[idx], snrm)
    score = dist + 0.03 * np.clip(0.6 - dots, 0, None) / 0.6 * 4  # penalise opposing normals (about 12 cm at worst)
    pick = np.argmin(score, axis=1)
    sel = idx[np.arange(len(idx)), pick]
    W = tw[sel].copy()
    # Laplacian smoothing over the scan mesh
    edges = stm.edges_unique
    n = len(scan_verts)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    P = sp.diags(1.0 / deg) @ adj
    for _ in range(smooth_iters):
        W = (1 - smooth_alpha) * W + smooth_alpha * (P @ W)
    return W, dict(nn_dist_median_cm=float(np.median(dist[:, 0]) * 100), nn_dist_p99_cm=float(np.percentile(dist[:, 0], 99) * 100))


def top4(W: np.ndarray):
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, axis=1)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return idx.astype(np.uint16), (w / s).astype(np.float32)
