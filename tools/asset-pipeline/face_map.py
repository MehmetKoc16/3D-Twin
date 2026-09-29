"""face-map.json: binding of the 468 MediaPipe canonical face landmarks to the MakeHuman head surface.

Data sources (both Apache-2.0, pinned commit, see config.MEDIAPIPE):
  canonical_face_model.obj   468 vertices (landmark index == vertex index), UVs, 898 triangles
  face_mesh_connections.py   connection index lists (face oval, eyes, eyebrows, lips)

Only the index lists and the triangle topology of the canonical model are copied into the output; its vertex
positions and UVs are used during the build as the alignment target and are not redistributed.

Frames.  MakeHuman body (meters, +Y up, +Z front, +X = the subject's LEFT).  Canonical model (centimeters, +Y up,
+Z out of the face towards the camera, +X = the subject's LEFT: landmark 263 = MediaPipe "left eye" has x > 0, and a
photo shows the subject's left on the image right, i.e. image +x as well).  So both frames share their axes and the
alignment is a similarity that keeps mirror symmetry: uniform scale s, pitch (rotation about X), translation (0, ty, tz).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np

from config import CANONICAL_OBJ, FACE_CONNECTIONS_PY, MEDIAPIPE
from mh_obj import BaseMesh

# Weight of the depth (z) distance when a canonical point is bound to the head surface. The photo is warped and the
# face is fitted in the frontal projection, so the bound point should be the surface point that is closest in (x, y);
# a small depth weight keeps the binding well defined at grazing angles (cheeks) where a pure z-ray would slide.
# 1.0 would be plain closest-point.
Z_WEIGHT = 0.2


# ---------------------------------------------------------------------------------------------------------------
# Canonical model + MediaPipe connection lists
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Canonical:
    verts: np.ndarray  # (468, 3) cm
    faces: np.ndarray  # (898, 3) int


def load_canonical() -> Canonical:
    v: list[list[float]] = []
    f: list[list[int]] = []
    with open(CANONICAL_OBJ, "r", encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                v.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                if len(p) != 4:
                    raise ValueError("canonical face is not a triangle")
                f.append([int(t.split("/")[0]) - 1 for t in p[1:4]])
    verts, faces = np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)
    if verts.shape != (468, 3) or faces.shape != (898, 3):
        raise ValueError(f"unexpected canonical face model: {verts.shape}, {faces.shape}")
    return Canonical(verts, faces)


def load_connections() -> dict[str, list[tuple[int, int]]]:
    """Parse `NAME = frozenset([(a, b), ...])` assignments with ast (no code is executed); order is preserved."""
    tree = ast.parse(FACE_CONNECTIONS_PY.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[int, int]]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if getattr(call.func, "id", "") != "frozenset" or len(call.args) != 1 or not isinstance(call.args[0], ast.List):
            continue
        out[node.targets[0].id] = [tuple(int(x) for x in e) for e in ast.literal_eval(call.args[0])]
    return out


def chain_loop(pairs: list[tuple[int, int]], start: int) -> list[int]:
    """Order an undirected closed loop of edges into a vertex walk starting at `start` (first vertex not repeated)."""
    nxt: dict[int, list[int]] = {}
    for a, b in pairs:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    loop = [start]
    prev, cur = -1, start
    while True:
        cands = [n for n in nxt[cur] if n != prev] or nxt[cur]
        # the list order gives the intended direction: take the first neighbour that is not where we came from
        step = cands[0]
        if step == start:
            break
        loop.append(step)
        prev, cur = cur, step
        if len(loop) > len(pairs):
            raise ValueError("connection list is not a single loop")
    return loop


# ---------------------------------------------------------------------------------------------------------------
# Head surface
# ---------------------------------------------------------------------------------------------------------------


def uv_islands(tris: np.ndarray, count: int) -> np.ndarray:
    """Island label per render vertex: connected components of the render-triangle graph (render vertices are
    (vertex, uv) pairs, so triangles sharing a render vertex share its UV, i.e. they are on the same island)."""
    parent = np.arange(count)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in tris:
        r0 = find(int(t[0]))
        for k in (1, 2):
            r = find(int(t[k]))
            if r != r0:
                parent[r] = r0
    return np.array([find(i) for i in range(count)], dtype=np.int64)


@dataclass
class Head:
    island: np.ndarray  # (R,) island label per render vertex
    head_label: int
    eye_labels: dict[str, int]  # 'left' / 'right' (subject's own side) eye-socket islands
    mouth_label: int
    tri_ids: np.ndarray  # triangles of the head island facing the camera (candidates for binding)


def analyse_head(mesh: BaseMesh, P: np.ndarray, neck_y: float) -> Head:
    """Find the head UV island, the eye-socket and mouth-cavity islands that hang on it, and the frontal triangles.

    The eyeballs are separate helper geometry, so eyelid openings and the lip slit are real holes of the head island
    (bounded by the socket / mouth-cavity islands): the head island is the only surface a photo can be painted on.
    """
    R = mesh.render_count
    ren = P[:R]
    island = uv_islands(mesh.tris, R)
    high = np.where(ren[:, 1] > neck_y + 0.02)[0]
    head_label = int(island[high[ren[high, 2].argmax()]])  # the island that holds the nose tip
    head_mh = set(mesh.render_mh[island == head_label].tolist())
    eye: dict[str, int] = {}
    mouth = -1
    for lab in np.unique(island):
        if lab == head_label:
            continue
        m = island == lab
        shared = [i for i in np.where(m)[0] if int(mesh.render_mh[i]) in head_mh]
        if not shared:
            continue
        c = ren[m].mean(axis=0)
        if ren[m, 1].mean() < neck_y + 0.02 or m.sum() > 1000:
            continue  # the neck / body island
        if abs(c[0]) > 0.02:
            eye["left" if c[0] > 0 else "right"] = int(lab)
        else:
            mouth = int(lab)
    if set(eye) != {"left", "right"} or mouth < 0:
        raise RuntimeError(f"could not identify eye/mouth islands: {eye}, {mouth}")
    tri_lab = island[mesh.tris[:, 0]]
    a, b, c_ = (ren[mesh.tris[:, i]] for i in range(3))
    nz = np.cross(b - a, c_ - a)[:, 2]
    front = (tri_lab == head_label) & (nz > 0)
    return Head(island, head_label, eye, mouth, np.where(front)[0])


def _closest_dense(pts, A, B, C):
    """Exact closest point of each query on the triangles A, B, C, all given per query: arrays are (n, k, 3) and pts
    is (n, 3). Interior projection, else the closest of the three edges. Returns (index into k, bary, sq dist)."""
    p = pts[:, None, :]
    e0, e1 = B - A, C - A
    nrm = np.cross(e0, e1)
    nn = np.maximum((nrm * nrm).sum(-1), 1e-30)
    d00, d01, d11 = (e0 * e0).sum(-1), (e0 * e1).sum(-1), (e1 * e1).sum(-1)
    det = np.where(d00 * d11 - d01 * d01 == 0, 1e-30, d00 * d11 - d01 * d01)
    ap = p - A
    d = (ap * nrm).sum(-1)
    rel = ap - (d / nn)[..., None] * nrm
    d20, d21 = (rel * e0).sum(-1), (rel * e1).sum(-1)
    v = (d11 * d20 - d01 * d21) / det
    w = (d00 * d21 - d01 * d20) / det
    u = 1.0 - v - w
    inside = (u >= 0) & (v >= 0) & (w >= 0)
    dist = np.where(inside, d * d / nn, np.inf)
    bary = np.stack([u, v, w], -1)
    for org, ev, i0, i1 in ((A, B - A, 0, 1), (B, C - B, 1, 2), (C, A - C, 2, 0)):
        t = np.clip(((p - org) * ev).sum(-1) / np.maximum((ev * ev).sum(-1), 1e-30), 0.0, 1.0)
        dd = ((p - (org + t[..., None] * ev)) ** 2).sum(-1)
        better = dd < dist
        eb = np.zeros_like(bary)
        eb[..., i0] = 1.0 - t
        eb[..., i1] = t
        dist = np.where(better, dd, dist)
        bary = np.where(better[..., None], eb, bary)
    k = dist.argmin(axis=1)
    rows = np.arange(len(k))
    return k, bary[rows, k], dist[rows, k]


def closest_on_triangles(
    pts: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, candidates: int = 160
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point of each query on a triangle set: returns (triangle index, barycentric (N, 3), squared distance).
    Exact per candidate; the candidates of a query are its `candidates` triangles with the nearest centroids (the
    head triangles are small and fairly uniform, so the true closest triangle is always among them; asserted by the
    unit test against the brute-force result)."""
    k = min(candidates, len(A))
    cen = (A + B + C) / 3.0
    out_t = np.zeros(len(pts), dtype=np.int64)
    out_b = np.zeros((len(pts), 3))
    out_d = np.zeros(len(pts))
    for s in range(0, len(pts), 64):
        q = pts[s : s + 64]
        d2 = ((q[:, None, :] - cen[None]) ** 2).sum(-1)
        cand = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < len(A) else np.tile(np.arange(len(A)), (len(q), 1))
        cand.sort(axis=1)  # deterministic tie handling
        kk, bary, dist = _closest_dense(q, A[cand], B[cand], C[cand])
        out_t[s : s + 64] = cand[np.arange(len(q)), kk]
        out_b[s : s + 64] = bary
        out_d[s : s + 64] = dist
    return out_t, out_b, out_d


# ---------------------------------------------------------------------------------------------------------------
# Alignment: canonical (cm) -> MakeHuman head (m). Symmetric similarity T(q) = s * Rx(theta) q + (0, ty, tz)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Align:
    s: float  # meters per canonical cm
    theta: float  # pitch in radians (rotation about +X)
    ty: float
    tz: float

    def apply(self, q: np.ndarray) -> np.ndarray:
        c, sn = np.cos(self.theta), np.sin(self.theta)
        out = np.empty_like(q, dtype=np.float64)
        out[..., 0] = self.s * q[..., 0]
        out[..., 1] = self.s * (c * q[..., 1] - sn * q[..., 2]) + self.ty
        out[..., 2] = self.s * (sn * q[..., 1] + c * q[..., 2]) + self.tz
        return out


def _fit_given_theta(q: np.ndarray, c: np.ndarray, w: np.ndarray, m: np.ndarray, theta: float):
    """Weighted LS for (s, ty, tz) at a fixed pitch, in the metric diag(m). Returns (params, cost)."""
    co, si = np.cos(theta), np.sin(theta)
    rq = np.stack([q[:, 0], co * q[:, 1] - si * q[:, 2], si * q[:, 1] + co * q[:, 2]], axis=1)  # (n, 3)
    n = len(q)
    A = np.zeros((3 * n, 3))
    b = np.zeros(3 * n)
    for k in range(3):
        sw = np.sqrt(w) * m[k]
        A[k::3, 0] = rq[:, k] * sw
        b[k::3] = c[:, k] * sw
        if k == 1:
            A[k::3, 1] = sw
        if k == 2:
            A[k::3, 2] = sw
    # x has no translation (symmetry); its row only constrains s
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cost = float(((A @ sol - b) ** 2).sum())
    return sol, cost


def fit_alignment(q: np.ndarray, c: np.ndarray, w: np.ndarray | None = None, m=(1.0, 1.0, 1.0)) -> Align:
    """Least-squares (s, theta, ty, tz) mapping canonical points q (cm) onto targets c (m)."""
    w = np.ones(len(q)) if w is None else w
    mm = np.asarray(m, dtype=np.float64)
    thetas = np.deg2rad(np.arange(-30.0, 30.0001, 0.5))
    costs = [_fit_given_theta(q, c, w, mm, t)[1] for t in thetas]
    t0 = thetas[int(np.argmin(costs))]
    fine = t0 + np.deg2rad(np.arange(-0.5, 0.5001, 0.02))
    costs = [_fit_given_theta(q, c, w, mm, t)[1] for t in fine]
    t = float(fine[int(np.argmin(costs))])
    (s, ty, tz), _ = _fit_given_theta(q, c, w, mm, t)
    return Align(float(s), t, float(ty), float(tz))


def anchors_mh(mesh: BaseMesh, P: np.ndarray, head: Head) -> dict[str, np.ndarray]:
    """Anchor points of the MakeHuman head from its own geometry: centres of the eyelid openings (boundary of the
    eye-socket islands), lip slit (boundary of the mouth-cavity island) with its corners, and the nose tip."""
    R = mesh.render_count
    ren = P[:R]
    head_mh = set(mesh.render_mh[head.island == head.head_label].tolist())

    def rim(label: int) -> np.ndarray:
        ids = [i for i in np.where(head.island == label)[0] if int(mesh.render_mh[i]) in head_mh]
        return ren[ids]

    left, right, mouth = rim(head.eye_labels["left"]), rim(head.eye_labels["right"]), rim(head.mouth_label)
    hi = np.where(head.island == head.head_label)[0]
    return {
        "eye_l": left.mean(axis=0),
        "eye_r": right.mean(axis=0),
        "mouth": mouth.mean(axis=0),
        "mouth_l": mouth[mouth[:, 0].argmax()],  # subject's left = +X
        "mouth_r": mouth[mouth[:, 0].argmin()],
        "nose": ren[hi[ren[hi, 2].argmax()]],
    }


def anchors_canonical(can: Canonical, conn: dict) -> dict[str, np.ndarray]:
    def idx(name: str) -> list[int]:
        return sorted({i for e in conn[name] for i in e})

    v = can.verts
    inner_lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    return {
        "eye_l": v[idx("FACEMESH_LEFT_EYE")].mean(axis=0),  # MediaPipe "left" = subject's left = +X (asserted below)
        "eye_r": v[idx("FACEMESH_RIGHT_EYE")].mean(axis=0),
        "mouth": v[inner_lips].mean(axis=0),
        "mouth_l": v[308],
        "mouth_r": v[78],
        "nose": v[int(v[:, 2].argmax())],
    }


ANCHOR_ORDER = ("eye_l", "eye_r", "mouth", "mouth_l", "mouth_r", "nose")
ANCHOR_WEIGHT = 12.0  # each anchor counts like this many surface points in the refinement
TRIM = 0.85  # fraction of the surface points (smallest distances) used by each ICP step
ICP_ITERATIONS = 20


def align_canonical(
    can: Canonical, conn: dict, mesh: BaseMesh, P: np.ndarray, head: Head
) -> tuple[Align, dict]:
    """Anchor fit, then trimmed ICP (anchors stay in the objective so the scale cannot drift): returns the alignment
    and a small report dict."""
    ren = P[: mesh.render_count]
    am, ac = anchors_mh(mesh, P, head), anchors_canonical(can, conn)
    if not (ac["eye_l"][0] > 0 > ac["eye_r"][0] and am["eye_l"][0] > 0 > am["eye_r"][0]):
        raise RuntimeError("left/right convention mismatch between canonical model and MakeHuman")
    q_a = np.stack([ac[k] for k in ANCHOR_ORDER])
    c_a = np.stack([am[k] for k in ANCHOR_ORDER])
    al = fit_alignment(q_a, c_a)
    tid = head.tri_ids
    A, B, C = (ren[mesh.tris[tid, i]] for i in range(3))
    # surface points used for the refinement: everything except the eye / lip contours (the MakeHuman lids are
    # holes of the head island, so contour points would be pulled to the hole rims)
    sel = np.ones(468, dtype=bool)
    for name in ("FACEMESH_LIPS", "FACEMESH_LEFT_EYE", "FACEMESH_RIGHT_EYE"):
        sel[sorted({i for e in conn[name] for i in e})] = False
    q_s = can.verts[sel]
    for _ in range(ICP_ITERATIONS):
        ti, bb, d2 = closest_on_triangles(al.apply(q_s), A, B, C)
        keep = np.sqrt(d2) <= np.quantile(np.sqrt(d2), TRIM)
        tri_pts = np.stack([A[ti], B[ti], C[ti]], axis=1)
        cp = (bb[:, :, None] * tri_pts).sum(axis=1)
        q = np.concatenate([q_s[keep], q_a])
        c = np.concatenate([cp[keep], c_a])
        w = np.concatenate([np.ones(int(keep.sum())), np.full(len(q_a), ANCHOR_WEIGHT)])
        al = fit_alignment(q, c, w)
    return al, {"anchors_mh": am, "anchors_canonical": ac}


# ---------------------------------------------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Binding:
    tri: np.ndarray  # (468, 3) render-vertex ids
    bary: np.ndarray  # (468, 3)
    uv: np.ndarray  # (468, 2) glTF convention (u, 1 - v_obj)
    points: np.ndarray  # (468, 3) bound positions on the neutral body, meters
    aligned: np.ndarray  # (468, 3) canonical vertices after alignment, meters


def gltf_uv(mesh: BaseMesh) -> np.ndarray:
    """Render-vertex UVs in the base.glb TEXCOORD_0 convention (v flipped)."""
    return np.stack([mesh.render_uv[:, 0], 1.0 - mesh.render_uv[:, 1]], axis=1)


def bind_landmarks(mesh: BaseMesh, P: np.ndarray, head: Head, can: Canonical, al: Align) -> Binding:
    """Bind every canonical vertex to the closest point (depth-weighted metric, see Z_WEIGHT) on the frontal
    triangles of the head UV island. The result is normalised: barycentrics are clipped at 0 and sum to exactly 1
    after rounding to 6 decimals."""
    ren = P[: mesh.render_count]
    aligned = al.apply(can.verts)
    tid = head.tri_ids
    A, B, C = (ren[mesh.tris[tid, i]] for i in range(3))
    m = np.array([1.0, 1.0, Z_WEIGHT])
    ti, bary, _ = closest_on_triangles(aligned * m, A * m, B * m, C * m)
    bary = np.clip(bary, 0.0, None)
    bary /= bary.sum(axis=1, keepdims=True)
    bary = np.round(bary, 6)
    bary[:, 2] = np.round(1.0 - bary[:, 0] - bary[:, 1], 6)
    tri = mesh.tris[tid[ti]]
    uv = (bary[:, :, None] * gltf_uv(mesh)[tri]).sum(axis=1)
    points = (bary[:, :, None] * ren[tri]).sum(axis=1)
    return Binding(tri, bary, np.round(uv, 6), points, aligned)


def uv_triangle_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Signed area (twice) of each canonical face in UV space (glTF convention, v down)."""
    t = uv[faces]
    return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])


def seam_report(mesh: BaseMesh, head: Head, b: Binding, faces: np.ndarray, contour_ids: set[int]) -> dict:
    """UV island / seam checks of the face region.

    1. every bound triangle has its 3 render vertices on the head island (so the UV interpolation is continuous);
    2. no canonical face straddles a UV cut: the centroid and edge midpoints of every canonical face, taken in UV
       space, must fall inside some head-island triangle. Samples outside are counted; they may only belong to faces
       that touch an eye or lip contour (the eyelid openings and the lip slit are holes of the head island).
    """
    islands = np.unique(head.island[b.tri])
    uvg = gltf_uv(mesh)
    hid = np.where(head.island[mesh.tris[:, 0]] == head.head_label)[0]
    T = uvg[mesh.tris[hid]]  # (n, 3, 2)
    f = b.uv[faces]  # (F, 3, 2)
    samples = np.concatenate(
        [f.mean(axis=1, keepdims=True), (f[:, [0, 1, 2]] + f[:, [1, 2, 0]]) / 2.0], axis=1
    )  # (F, 4, 2): centroid + 3 edge midpoints
    flat = samples.reshape(-1, 2)
    inside = np.zeros(len(flat), dtype=bool)
    a, bb, c = T[:, 0], T[:, 1], T[:, 2]
    d = (bb[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - bb[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(d) > 1e-15
    a, bb, c, d = a[ok], bb[ok], c[ok], d[ok]
    for s in range(0, len(flat), 256):
        p = flat[s : s + 256][:, None, :]
        w0 = ((bb[:, 1] - c[:, 1]) * (p[..., 0] - c[:, 0]) + (c[:, 0] - bb[:, 0]) * (p[..., 1] - c[:, 1])) / d
        w1 = ((c[:, 1] - a[:, 1]) * (p[..., 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[..., 1] - c[:, 1])) / d
        w2 = 1.0 - w0 - w1
        inside[s : s + 256] = ((w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)).any(axis=1)
    bad_face = ~inside.reshape(-1, 4).all(axis=1)
    on_contour = np.array([any(int(i) in contour_ids for i in fc) for fc in faces])
    return {
        "islands_of_bound_triangles": [int(x) for x in islands],
        "single_island": bool(len(islands) == 1 and islands[0] == head.head_label),
        "faces_with_samples_outside_island": int(bad_face.sum()),
        "faces_outside_not_on_contour": int((bad_face & ~on_contour).sum()),
    }


# ---------------------------------------------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------------------------------------------


def _members(conn: dict, name: str) -> list[int]:
    return sorted({i for e in conn[name] for i in e})


def build_regions(can: Canonical, conn: dict) -> dict[str, list[int]]:
    """Landmark groups; left/right are the SUBJECT's own sides (+X in both frames).

    eyes / lips come straight from MediaPipe's connection lists (Apache-2.0). MediaPipe has no cheek or forehead
    list, so those two are defined here from the canonical geometry: forehead = above the highest eyebrow landmark;
    cheeks = the skin between the lower lid line and the mouth corners, outside the nose wings and inside the face
    oval, split by the sign of x (canonical +X = subject's left, verified through the eye lists).
    """
    v = can.verts
    left_eye, right_eye = _members(conn, "FACEMESH_LEFT_EYE"), _members(conn, "FACEMESH_RIGHT_EYE")
    if not (v[left_eye, 0].mean() > 0 > v[right_eye, 0].mean()):
        raise RuntimeError("MediaPipe 'left eye' is not on canonical +X")
    lips = _members(conn, "FACEMESH_LIPS")
    brows = _members(conn, "FACEMESH_LEFT_EYEBROW") + _members(conn, "FACEMESH_RIGHT_EYEBROW")
    eye_all = set(left_eye) | set(right_eye)
    nose = _members(conn, "FACEMESH_NOSE")
    oval = _members(conn, "FACEMESH_FACE_OVAL")
    taken = eye_all | set(lips) | set(brows)

    brow_top = v[brows, 1].max()
    forehead = [i for i in range(468) if v[i, 1] > brow_top + 0.2 and i not in taken]

    y_top = v[list(eye_all), 1].min() - 0.5  # below the lower lid line
    y_bot = min(v[61, 1], v[291, 1])  # level of the mouth corners
    nose_wing = max(np.abs(v[nose, 0]).max(), abs(v[61, 0]), abs(v[291, 0]))  # also clear of the mouth corners
    oval_v = v[oval]

    def oval_half_width(y: float) -> float:
        near = np.abs(oval_v[:, 1] - y) < 0.7
        return float(np.abs(oval_v[near, 0]).max()) if near.any() else 0.0

    left_cheek, right_cheek = [], []
    for i in range(468):
        x, y = v[i, 0], v[i, 1]
        if i in taken or not (y_bot <= y <= y_top):
            continue
        if not (abs(x) > nose_wing + 0.3 and abs(x) < oval_half_width(y) - 0.9):
            continue
        (left_cheek if x > 0 else right_cheek).append(i)
    return {
        "leftEye": left_eye,
        "rightEye": right_eye,
        "lips": lips,
        "leftCheek": left_cheek,
        "rightCheek": right_cheek,
        "forehead": forehead,
    }


# ---------------------------------------------------------------------------------------------------------------
# Fit modifiers: which face modifiers visibly change the frontal 2-D landmark layout
# ---------------------------------------------------------------------------------------------------------------

# Candidates (all exist in the manifest). Selection by finite differences, see select_fit_modifiers.
FIT_CANDIDATES = (
    "head/head-scale-horiz",
    "head/head-scale-vert",
    "head/head-fat",
    "chin/chin-width",
    "chin/chin-height",
    "nose/nose-scale-horiz",
    "nose/nose-scale-vert",
    "mouth/mouth-scale-horiz",
    "mouth/mouth-scale-vert",
    "eyes/eye-scale",
    "forehead/forehead-scale-vert",
    "cheek/cheek-volume",
    "chin/chin-jaw-drop",
    "nose/nose-nostrils-width",
)
# A modifier qualifies when, on every defined side (-1 and/or +1), the 2-D layout of the stable landmarks changes by
# at least this much (RMS after removing translation, rotation and uniform scale), in millimeters. The head is
# about 150 mm wide; typical MediaPipe landmark noise on a selfie is well below 1 mm-equivalent.
MIN_EFFECT_MM = 0.6


def stable_landmarks(can: Canonical, conn: dict) -> list[int]:
    """Landmarks used for shape fitting: everything except the parts that move with expression or blinking, i.e.
    the eye contours (blink / squint), the eyebrows (raise / frown) and the lips (open / smile / pucker), keeping only
    the two outer lip corners (61, 291) which carry the mouth width. The jaw stays in (chin height matters) and a
    closed mouth is assumed."""
    drop = set(_members(conn, "FACEMESH_LEFT_EYE")) | set(_members(conn, "FACEMESH_RIGHT_EYE"))
    drop |= set(_members(conn, "FACEMESH_LEFT_EYEBROW")) | set(_members(conn, "FACEMESH_RIGHT_EYEBROW"))
    drop |= set(_members(conn, "FACEMESH_LIPS")) - {61, 291}
    return [i for i in range(468) if i not in drop]


def similarity_residual_rms(a: np.ndarray, b: np.ndarray) -> float:
    """RMS of the 2-D points b after the best similarity (translation, rotation, uniform scale) of a onto b."""
    za = (a[:, 0] - a[:, 0].mean()) + 1j * (a[:, 1] - a[:, 1].mean())
    zb = (b[:, 0] - b[:, 0].mean()) + 1j * (b[:, 1] - b[:, 1].mean())
    k = np.vdot(za, zb) / np.vdot(za, za)
    return float(np.sqrt((np.abs(zb - k * za) ** 2).mean()))


def frontal_layout(P: np.ndarray, b: Binding) -> np.ndarray:
    """(468, 2) frontal (x, y) of the bound surface points on the body positions P."""
    return (b.bary[:, :, None] * P[b.tri]).sum(axis=1)[:, :2]


def modifier_effects(morphset, manifest: dict, b: Binding, stable: list[int], ids) -> dict[str, list[float | None]]:
    """[effect at -1, effect at +1] in mm for each modifier id (None where the side does not exist)."""
    defs = {m["id"]: m for m in manifest["modifiers"]}
    base = frontal_layout(morphset.positions(), b)[stable]
    out: dict[str, list[float | None]] = {}
    for mid in ids:
        d = defs[mid]
        row: list[float | None] = []
        for val in (-1.0, 1.0):
            if val < d["min"] or val > d["max"]:
                row.append(None)
                continue
            lay = frontal_layout(morphset.positions(mods={mid: val}), b)[stable]
            row.append(round(similarity_residual_rms(base, lay) * 1000.0, 3))
        out[mid] = row
    return out


def select_fit_modifiers(effects: dict[str, list[float | None]]) -> list[str]:
    return [
        m
        for m, row in effects.items()
        if any(e is not None for e in row) and all(e >= MIN_EFFECT_MM for e in row if e is not None)
    ]


# A fit modifier must also be identifiable: the part of its frontal effect that the other fit modifiers cannot
# reproduce (after the similarity is removed) must be at least this fraction of its total effect. Example: with a
# uniform scale removed, head-scale-horiz and head-scale-vert only differ by their ratio (about 5 % unique effect
# each), so the weaker one is dropped.
MIN_UNIQUE_FRACTION = 0.2


def _effect_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    za = (a[:, 0] - a[:, 0].mean()) + 1j * (a[:, 1] - a[:, 1].mean())
    zb = (b[:, 0] - b[:, 0].mean()) + 1j * (b[:, 1] - b[:, 1].mean())
    r = zb - (np.vdot(za, zb) / np.vdot(za, za)) * za
    return np.concatenate([r.real, r.imag]) * 1000.0  # mm


def prune_unidentifiable(morphset, manifest: dict, b: Binding, stable: list[int], ids: list[str]) -> list[str]:
    """Backward elimination on the frontal Jacobian (similarity removed): drop the modifier with the smallest unique
    effect while it is below MIN_UNIQUE_FRACTION of its own norm."""
    defs = {m["id"]: m for m in manifest["modifiers"]}
    base = frontal_layout(morphset.positions(), b)[stable]
    cols: dict[str, np.ndarray] = {}
    for mid in ids:
        val = 1.0 if defs[mid]["max"] >= 1.0 else -1.0
        cols[mid] = _effect_vector(base, frontal_layout(morphset.positions(mods={mid: val}), b)[stable])
    keep = list(ids)
    while len(keep) > 1:
        J = np.stack([cols[m] for m in keep], axis=1)
        frac = []
        for i in range(len(keep)):
            others = np.delete(J, i, axis=1)
            coef, *_ = np.linalg.lstsq(others, J[:, i], rcond=None)
            frac.append(np.linalg.norm(J[:, i] - others @ coef) / max(np.linalg.norm(J[:, i]), 1e-12))
        worst = int(np.argmin(frac))
        if frac[worst] >= MIN_UNIQUE_FRACTION:
            break
        keep.pop(worst)
    return keep


# ---------------------------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------------------------

# Faces whose UV area is below this (UV units squared, doubled area) are collapsed slivers at the eye / lip holes.
SLIVER_AREA = 1e-6


@dataclass
class FaceMapResult:
    data: dict  # contents of face-map.json
    report: dict  # statistics for logs / tests / the ADR (not written to the output)
    binding: Binding
    can: Canonical
    head: Head


def build_face_map(mesh: BaseMesh, P: np.ndarray, morphset, manifest: dict) -> FaceMapResult:
    """P: neutral body positions in the combined index space (meters). morphset: macro.MorphSet of the same assets."""
    names = [j["name"] for j in manifest["jointPoints"]]
    R = mesh.render_count
    neck_y = float(P[R + names.index("joint-neck"), 1])
    head = analyse_head(mesh, P, neck_y)
    can, conn = load_canonical(), load_connections()
    al, arep = align_canonical(can, conn, mesh, P, head)
    b = bind_landmarks(mesh, P, head, can, al)

    regions = build_regions(can, conn)
    oval = chain_loop(conn["FACEMESH_FACE_OVAL"], 10)
    contour_ids = set(regions["leftEye"]) | set(regions["rightEye"]) | set(regions["lips"])
    seams = seam_report(mesh, head, b, can.faces, contour_ids)

    areas = uv_triangle_areas(b.uv, can.faces)
    sign = 1.0 if (areas > 0).sum() > (areas < 0).sum() else -1.0
    flipped = (areas * sign < 0) & (np.abs(areas) >= 0.0)
    sliver = np.abs(areas) < SLIVER_AREA
    on_contour = np.array([any(int(i) in contour_ids for i in f) for f in can.faces])
    d3 = np.linalg.norm(b.points - b.aligned, axis=1)
    dxy = np.linalg.norm((b.points - b.aligned)[:, :2], axis=1)

    stable = stable_landmarks(can, conn)
    effects = modifier_effects(morphset, manifest, b, stable, FIT_CANDIDATES)
    qualified = select_fit_modifiers(effects)
    fit_mods = prune_unidentifiable(morphset, manifest, b, stable, qualified)
    ids = {m["id"] for m in manifest["modifiers"]}
    missing = [m for m in fit_mods if m not in ids]
    if missing:
        raise RuntimeError(f"fitModifiers not in the manifest: {missing}")

    data = {
        "version": 1,
        "source": {"mediapipeCommit": MEDIAPIPE["sha"], "license": "Apache-2.0"},
        "triangles": can.faces.tolist(),
        "landmarks": [
            {
                "index": i,
                "tri": [int(x) for x in b.tri[i]],
                "bary": [float(x) for x in b.bary[i]],
                "uv": [float(x) for x in b.uv[i]],
            }
            for i in range(468)
        ],
        "faceOval": oval,
        "regions": regions,
        "uvBounds": {
            "min": [round(float(x), 6) for x in b.uv.min(axis=0)],
            "max": [round(float(x), 6) for x in b.uv.max(axis=0)],
        },
        "fitModifiers": fit_mods,
    }
    report = {
        "alignment": {"scaleMPerCm": al.s, "pitchDeg": float(np.rad2deg(al.theta)), "ty": al.ty, "tz": al.tz},
        "anchors_mh": arep["anchors_mh"],
        "bound": 468,
        "max_distance_mm": float(d3.max() * 1000),
        "mean_distance_mm": float(d3.mean() * 1000),
        "max_xy_distance_mm": float(dxy.max() * 1000),
        "mean_xy_distance_mm": float(dxy.mean() * 1000),
        "seams": seams,
        "uv_orientation_sign": sign,
        "flipped_faces": [int(i) for i in np.where(flipped)[0]],
        "flipped_off_contour": int((flipped & ~on_contour).sum()),
        "sliver_faces": int(sliver.sum()),
        "sliver_off_contour": int((sliver & ~on_contour).sum()),
        "stable_landmarks": stable,
        "effects_mm": effects,
        "qualified_modifiers": qualified,
        "fit_modifiers": fit_mods,
        "regions": {k: len(v) for k, v in regions.items()},
        "uv_bounds": data["uvBounds"],
        "areas": areas,
    }
    return FaceMapResult(data, report, b, can, head)


# ---------------------------------------------------------------------------------------------------------------
# Debug image
# ---------------------------------------------------------------------------------------------------------------


def write_debug_images(path, mesh: BaseMesh, res: FaceMapResult) -> None:
    """Body UV layout (grey; head island lighter) with the mapped canonical triangles in orange; eye contours cyan,
    lips magenta, face oval yellow, flipped / sliver triangles red. A zoomed crop of the face is written next to it."""
    from pathlib import Path

    from debug_png import draw_dot, draw_line, write_png

    path = Path(path)
    uvg = gltf_uv(mesh)
    b, can, head = res.binding, res.can, res.head
    regions = res.data["regions"]
    oval = res.data["faceOval"]
    bad = np.where(np.abs(res.report["areas"]) < SLIVER_AREA)[0].tolist() + res.report["flipped_faces"]
    bad_set = set(bad)

    def render(size: int, lo: np.ndarray, hi: np.ndarray, layout: bool) -> np.ndarray:
        w = int(size)
        h = int(round(size * (hi[1] - lo[1]) / (hi[0] - lo[0])))
        img = np.full((h, w, 3), 18, dtype=np.uint8)

        def px(p):
            return ((p[0] - lo[0]) / (hi[0] - lo[0]) * (w - 1), (p[1] - lo[1]) / (hi[1] - lo[1]) * (h - 1))

        if layout:
            for t in mesh.tris:
                col = (110, 110, 110) if head.island[t[0]] == head.head_label else (55, 55, 55)
                for k in range(3):
                    draw_line(img, px(uvg[t[k]]), px(uvg[t[(k + 1) % 3]]), col)
        for fi, f in enumerate(can.faces):
            col = (235, 40, 40) if fi in bad_set else (255, 140, 0)
            for k in range(3):
                draw_line(img, px(b.uv[f[k]]), px(b.uv[f[(k + 1) % 3]]), col)
        for i in oval:
            draw_dot(img, px(b.uv[i]), (255, 255, 0), 1)
        for i in regions["leftEye"] + regions["rightEye"]:
            draw_dot(img, px(b.uv[i]), (0, 255, 255), 1)
        for i in regions["lips"]:
            draw_dot(img, px(b.uv[i]), (255, 0, 255), 1)
        return img

    write_png(path, render(1024, np.array([0.0, 0.0]), np.array([1.0, 1.0]), True))
    pad = 0.02
    lo, hi = b.uv.min(axis=0) - pad, b.uv.max(axis=0) + pad
    write_png(path.with_name(path.stem + "_zoom.png"), render(1400, lo, hi, True))
