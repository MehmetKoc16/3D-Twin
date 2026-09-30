"""Body-part proxies (eyes, eyebrows, eyelashes, hair): MakeHuman MHCLO -> `parts/<id>.glb`, `.bind.bin`, `.delete.bin`.

Same conventions as the garment templates (ADR 0007): the glb POSITION is the neutral-body rest frame in meters (feet
on y = 0), the binding is 36 bytes per glb vertex in the combined index space (ADR 0005), scale refs are render ids.
ADR 0008 documents what is different for body parts:

* Eyes and eyelashes (and some hair styles) are authored against HELPER geometry of `base.obj` (eyeballs, lash cards,
  a hair cap) that is not part of the runtime index space. They are re-bound to the body: every vertex (or, for the
  rigid eyeballs, every vertex of one eye) gets a triangle of the body mesh, chosen by how well it predicts the helper
  geometry over 48 sample bodies (macro extremes, face modifiers, random mixes; `mh_morph.MhMorpher`), see `fit_*`.
* The eye proxy keeps only the opaque inner shell (the outer shell is a fully transparent "cornea" in MakeHuman) and
  both eyes share one texture island (`irisUv` is then one circle).
* Hair / eyebrow textures are neutral luminance + alpha maps (`parts_tex`), the colour is the material colour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fetch
import garments
import gltf_writer
import mh_obj
import mhclo as mhclo_mod
import parts_tex
import writers
from config import DM_TO_M, PART_ASSETS, PART_PACK_URL, PART_TEXTURE_MAX
from mh_morph import MhMorpher

BODY_VERTS = mh_obj.BODY_VERTEX_COUNT

# Catalogue (order = index.json order). `title` is used for the attribution line.
PARTS: dict[str, dict] = {
    "eyes-default": {"category": "eyes", "title": "High-poly eyes (grey iris)",
                     "label": {"tr": "Gözler", "en": "Eyes"}},
    "eyebrows-default": {"category": "eyebrows", "title": "Eyebrow 001",
                         "label": {"tr": "Kaş (doğal)", "en": "Eyebrows (natural)"}},
    "eyebrows-thick": {"category": "eyebrows", "title": "Eyebrow 009",
                       "label": {"tr": "Kaş (gür)", "en": "Eyebrows (thick)"}},
    "eyebrows-thin": {"category": "eyebrows", "title": "Eyebrow 006",
                      "label": {"tr": "Kaş (ince)", "en": "Eyebrows (thin)"}},
    "eyelashes-default": {"category": "eyelashes", "title": "Eyelashes 01",
                          "label": {"tr": "Kirpikler", "en": "Eyelashes"}},
    "hair-short": {"category": "hair", "title": "Short hair 02 (male)",
                   "label": {"tr": "Kısa saç", "en": "Short hair"}},
    "hair-tousled": {"category": "hair", "title": "Hair 05",
                     "label": {"tr": "Dağınık saç", "en": "Tousled hair"}},
    "hair-bob": {"category": "hair", "title": "Bob 02 (female)",
                 "label": {"tr": "Kakülli küt saç", "en": "Bob with fringe"}},
    "hair-medium": {"category": "hair", "title": "Inverted bob",
                    "label": {"tr": "Orta boy saç", "en": "Medium hair"}},
    "hair-long": {"category": "hair", "title": "Long hair 01 (female)",
                  "label": {"tr": "Uzun saç", "en": "Long hair"}},
    "hair-ponytail": {"category": "hair", "title": "Ponytail 01 (female)",
                      "label": {"tr": "At kuyruğu", "en": "Ponytail"}},
}
DEFAULTS = {"eyes": "eyes-default", "eyebrows": "eyebrows-default", "eyelashes": "eyelashes-default"}

# Materials (glTF): doubleSided / tintable / roughness per category. Hair, brows and lashes are alpha-MASKed cards (opaque
# pass, depth-written, no sorting) unless their texture is fully opaque; the cutoff is derived from each texture
# (`parts_tex.coverage_cutoff`). The eyes are opaque, single-sided and only their iris is tintable.
CATEGORY_MATERIAL = {
    "eyes": {"double_sided": False, "tintable": True, "roughness": 0.25},
    "eyebrows": {"double_sided": True, "tintable": True, "roughness": 0.9},
    "eyelashes": {"double_sided": True, "tintable": False, "roughness": 0.9},
    "hair": {"double_sided": True, "tintable": True, "roughness": 0.6},
}

SAMPLE_MODIFIERS = (
    "head/head-scale-horiz", "head/head-scale-vert", "head/head-scale-depth", "head/head-fat", "forehead/forehead-scale-vert",
    "chin/chin-width", "chin/chin-height", "nose/nose-scale-horiz", "nose/nose-scale-vert", "mouth/mouth-scale-horiz",
    "eyes/eye-scale", "cheek/cheek-volume",
)
BODY_SAMPLE_MODIFIERS = ("neck/neck-scale-horiz", "neck/neck-scale-vert", "torso/torso-scale-horiz", "torso/torso-scale-vert",
                         "measure/measure-neck-circ", "measure/measure-shoulder-dist")
MACRO_IDS = ("gender", "height", "weight", "muscle")

# ---------------------------------------------------------------------------------------------------------------
# context: sample bodies in the full MakeHuman vertex space
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class PartsContext:
    mesh: mh_obj.BaseMesh
    morph: MhMorpher
    offset_y: float
    first: np.ndarray  # (13380,) canonical render id per MakeHuman body vertex
    tri_mh: np.ndarray  # (T, 3) MakeHuman ids of the body triangles
    sample_specs: list[tuple[dict, dict]]
    Hs: np.ndarray  # (1 + B, 19158, 3) meters, ground-offset; index 0 = neutral body

    def body(self, macros: dict | None = None, mods: dict | None = None) -> np.ndarray:
        """(19158, 3) meters, ground-offset positions of any body."""
        p = self.morph.positions(macros, mods)
        p[:, 1] += self.offset_y
        return p


def sample_bodies() -> list[tuple[dict, dict]]:
    """Deterministic set of (macros, modifiers): macro corners, every face modifier at +-1, random mixes."""
    out: list[tuple[dict, dict]] = []
    for g in (0.0, 1.0):
        for h in (0.0, 1.0):
            for w in (0.0, 1.0):
                for m in (0.0, 1.0):
                    out.append(({"gender": g, "height": h, "weight": w, "muscle": m}, {}))
    for mid in SAMPLE_MODIFIERS:
        for v in (-1.0, 1.0):
            out.append(({}, {mid: v}))
    rng = np.random.default_rng(20260929)
    pool = SAMPLE_MODIFIERS + BODY_SAMPLE_MODIFIERS
    for _ in range(8):
        macros = {k: float(round(rng.uniform(0.0, 1.0), 4)) for k in MACRO_IDS}
        picks = rng.choice(len(pool), size=4, replace=False)
        mods = {pool[int(i)]: float(round(rng.uniform(-1.0, 1.0), 4)) for i in sorted(picks)}
        out.append((macros, mods))
    return out


def make_context(mesh: mh_obj.BaseMesh, morph: MhMorpher, offset_y: float) -> PartsContext:
    first = mesh.mh_to_render_ids[mesh.mh_to_render_start[:BODY_VERTS]]
    specs = sample_bodies()
    bodies = [morph.positions()] + [morph.positions(m, d) for m, d in specs]
    Hs = np.stack(bodies)
    Hs[:, :, 1] += offset_y
    return PartsContext(mesh, morph, offset_y, first, mesh.render_mh[mesh.tris], specs, Hs)


# ---------------------------------------------------------------------------------------------------------------
# asset loading
# ---------------------------------------------------------------------------------------------------------------


def load_asset(pid: str, directory: Path | None = None) -> dict:
    """Read + validate one cached upstream asset. Raises on an unacceptable or inconsistent licence."""
    a = PART_ASSETS[pid]
    d = directory or fetch.part_dir(pid)
    names = {Path(m).name: m for m in a["files"]}
    clo = next(n for n in names if n.endswith(".mhclo"))
    obj_name = next(n for n in names if n.endswith(".obj"))
    mat_name = next(n for n in names if n.endswith(".mhmat"))
    mat_text = mhclo_mod.read_text(d / mat_name)
    cl = mhclo_mod.parse_mhclo(mhclo_mod.read_text(d / clo))
    lic_text = mhclo_mod.license_text(cl, mat_text)
    lic = mhclo_mod.classify_license(lic_text)
    pack_json = json.loads((d / "pack.json").read_text(encoding="utf-8"))
    entry = pack_json[clo[: -len(".mhclo")]]
    pack_lic = mhclo_mod.classify_license(entry.get("license", ""))
    if lic is None or lic != pack_lic:
        raise RuntimeError(f"{pid}: licence not acceptable or inconsistent (mhclo {lic_text!r}, pack {entry.get('license')!r})")
    if lic == "CC0-1.0" and not a["pack"].endswith(("cc0", "ccby")):
        raise RuntimeError(f"{pid}: unexpected pack {a['pack']}")
    mat = mhclo_mod.parse_mhmat(mat_text)
    diffuse = mat.get("diffuseTexture")
    tex_path = d / diffuse.split("/")[-1] if diffuse else None
    return {
        "gid": pid, "pid": pid, "asset": a, "dir": d, "mhclo": cl, "obj": mhclo_mod.parse_garment_obj(mhclo_mod.read_text(d / obj_name)),
        "mhmat": mat, "texture": tex_path, "license": lic, "license_text": lic_text, "pack_entry": entry, "clo_name": clo,
    }


# ---------------------------------------------------------------------------------------------------------------
# helper-bound parts: truth on any body, re-binding to body triangles
# ---------------------------------------------------------------------------------------------------------------


def truth_positions(cl: mhclo_mod.Mhclo, P: np.ndarray) -> np.ndarray:
    """MHCLO reconstruction on a full-mesh body P (19158, 3) meters: (n_obj_vertices, 3). Uses helper vertices too."""
    s = mhclo_mod.axis_scales({k: (a, b, d * DM_TO_M) for k, (a, b, d) in cl.scale.items()}, P)
    return mhclo_mod.reconstruct(cl.indices, cl.weights, cl.offsets * DM_TO_M, s, P)


def scales_on(cl: mhclo_mod.Mhclo, Hs: np.ndarray) -> np.ndarray:
    """(B1, 3) per-axis scale factors of the MHCLO scale refs on every sample body."""
    return np.stack([mhclo_mod.axis_scales({k: (a, b, d * DM_TO_M) for k, (a, b, d) in cl.scale.items()}, P) for P in Hs])


def _bary(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric weights of p in triangle (a, b, c), clamped into the triangle and renormalised. (..., 3)."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-18, 1e-18, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bar = np.stack([1.0 - v - w, v, w], axis=-1)
    bar = np.clip(bar, 0.0, None)
    tot = bar.sum(-1, keepdims=True)
    return np.where(tot > 0, bar / np.where(tot > 0, tot, 1.0), 1.0 / 3.0)


def fit_per_vertex(ctx: PartsContext, T: np.ndarray, s: np.ndarray, k: int = 32, chunk: int = 96):
    """Bind every vertex to the body triangle that best predicts it over all sample bodies.

    T (B1, V, 3): truth positions on every sample (0 = neutral); s (B1, 3): scale factors. The candidates are the k
    triangles closest to the vertex on the neutral body; weights are the (clamped) barycentric coordinates of the vertex
    on the triangle; the offset is exact on the neutral body (`o = (T0 - sum w P0) / s0`), and the candidate with the
    smallest squared error on the other bodies wins. Returns (tri_mh (V, 3), w (V, 3), off (V, 3) meters).
    """
    Hs, tri_mh = ctx.Hs, ctx.tri_mh
    cent = Hs[0][tri_mh].mean(axis=1)
    V = T.shape[1]
    out_tri = np.zeros((V, 3), dtype=np.int64)
    out_w = np.zeros((V, 3))
    out_off = np.zeros((V, 3))
    for lo in range(0, V, chunk):
        hi = min(V, lo + chunk)
        p = T[0, lo:hi]
        d2 = ((p[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        cand = np.argpartition(d2, k - 1, axis=1)[:, :k]
        cd2 = np.take_along_axis(d2, cand, axis=1)
        tri = tri_mh[cand]  # (c, k, 3)
        P0 = Hs[0][tri]  # (c, k, 3, 3)
        w = _bary(p[:, None, :], P0[..., 0, :], P0[..., 1, :], P0[..., 2, :])  # (c, k, 3)
        PB = Hs[:, tri]  # (B1, c, k, 3, 3)
        body = (w[None, :, :, :, None] * PB).sum(axis=3)  # (B1, c, k, 3)
        D = T[:, lo:hi, None, :] - body
        off = D[0] / s[0]
        res = D[1:] - off[None] * s[1:, None, None, :]
        r2 = (res ** 2).sum(axis=(0, 3)) + 1e-8 * np.sqrt(cd2)
        j = np.argmin(r2, axis=1)
        rows = np.arange(hi - lo)
        out_tri[lo:hi] = tri[rows, j]
        out_w[lo:hi] = w[rows, j]
        out_off[lo:hi] = off[rows, j]
    return out_tri, out_w, out_off


def fit_group(ctx: PartsContext, T: np.ndarray, s: np.ndarray, groups: list[np.ndarray], k: int = 96):
    """Rigid variant: all vertices of a group share ONE triangle and weights (the eyeballs stay spheres).

    Weights are the barycentric coordinates of the group centroid; per-vertex offsets are exact on the neutral body; the
    triangle with the smallest squared prediction error over the sample bodies (summed over the group) wins.
    """
    Hs, tri_mh = ctx.Hs, ctx.tri_mh
    cent = Hs[0][tri_mh].mean(axis=1)
    V = T.shape[1]
    out_tri = np.zeros((V, 3), dtype=np.int64)
    out_w = np.zeros((V, 3))
    out_off = np.zeros((V, 3))
    for idx in groups:
        pc = T[0, idx].mean(axis=0)
        d2 = ((cent - pc) ** 2).sum(-1)
        cand = np.argpartition(d2, k - 1)[:k]
        tri = tri_mh[cand]
        P0 = Hs[0][tri]
        w = _bary(pc, P0[:, 0], P0[:, 1], P0[:, 2])  # (k, 3)
        PB = Hs[:, tri]  # (B1, k, 3, 3)
        body = (w[None, :, :, None] * PB).sum(axis=2)  # (B1, k, 3)
        D = T[:, idx, None, :] - body[:, None, :, :]  # (B1, G, k, 3)
        off = D[0] / s[0]
        res = D[1:] - off[None] * s[1:, None, None, :]
        r2 = (res ** 2).sum(axis=(0, 1, 3)) + 1e-8 * np.sqrt(d2[cand])
        j = int(np.argmin(r2))
        out_tri[idx] = tri[j]
        out_w[idx] = w[j]
        out_off[idx] = off[:, j]
    return out_tri, out_w, out_off


def binding_error(ctx: PartsContext, tri: np.ndarray, w: np.ndarray, off: np.ndarray, T: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(B1, V) distance (meters) between the binding prediction and the truth on every sample body."""
    pred = np.stack([(w[:, :, None] * ctx.Hs[b][tri]).sum(axis=1) + off * s[b] for b in range(ctx.Hs.shape[0])])
    return np.linalg.norm(pred - T, axis=2)


# ---------------------------------------------------------------------------------------------------------------
# eyes: inner shell only, one shared texture island, cavity of the body
# ---------------------------------------------------------------------------------------------------------------


def prepare_eyes(asset: dict, tex_alpha: np.ndarray) -> dict:
    """Drop the fully transparent outer shell (faces whose UV lands on alpha < 0.5), map the second eye onto the texture
    island of the first (they are mirrored copies), and return the reduced OBJ + the MHCLO row ids kept."""
    obj: mhclo_mod.GarmentObj = asset["obj"]
    h, w = tex_alpha.shape
    keep_faces = []
    for f in obj.faces:
        uv = obj.uvs[[c[1] for c in f]].mean(axis=0)
        a = tex_alpha[min(h - 1, int((1.0 - uv[1]) * h)), min(w - 1, int(uv[0] * w))]
        if a >= 128:
            keep_faces.append(f)
    used = sorted({c[0] for f in keep_faces for c in f})
    remap = -np.ones(len(obj.verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    vx = obj.verts[used]
    left = vx[:, 0] > 0  # +x = the character's left
    # mirror correspondence (exact mirror copies): position key of the right-eye vertices -> left-eye vertex
    lkey = {tuple(np.round(p, 4)): j for j, p in enumerate(vx[left])}
    left_ids = np.nonzero(left)[0]
    right_ids = np.nonzero(~left)[0]
    mate = {}
    for i in right_ids:
        j = lkey.get(tuple(np.round(vx[i] * [-1, 1, 1], 4)))
        if j is None:
            raise RuntimeError("eye meshes are not mirror copies")
        mate[int(i)] = int(left_ids[j])
    face_key_left: dict[tuple, dict[int, int]] = {}
    for f in keep_faces:
        vs = [int(remap[c[0]]) for c in f]
        if left[vs[0]]:
            face_key_left[tuple(sorted(vs))] = {int(remap[c[0]]): c[1] for c in f}
    new_faces = []
    for f in keep_faces:
        vs = [int(remap[c[0]]) for c in f]
        if left[vs[0]]:
            new_faces.append([(int(remap[c[0]]), c[1]) for c in f])
        else:
            key = tuple(sorted(mate[v] for v in vs))
            vt_of = face_key_left[key]
            new_faces.append([(v, vt_of[mate[v]]) for v in vs])
    new_obj = mhclo_mod.GarmentObj(vx, obj.uvs, new_faces)
    return {"obj": new_obj, "rows": np.array(used, dtype=np.int64), "left": left}


def _sphere_fit(P: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.c_[2.0 * P, np.ones(len(P))]
    x = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    c = x[:3]
    return c, float(np.sqrt(max(x[3] + c @ c, 0.0)))


def eye_cavity(ctx: PartsContext, centre: np.ndarray, body: np.ndarray | None = None) -> dict:
    """The eye socket of the body mesh around one eyeball centre (meters, ground-offset frame).

    In MakeHuman the socket is a closed sphere-like cavity whose faces point INTO the head; the eyeball proxy sits inside.
    Returns quads (Q, 4) of the cavity, `rim` (MakeHuman ids where the eyelid skin meets the cavity, the eyelid
    opening) and `interior` (cavity vertices that touch no skin quad; the ones an eye proxy may hide).
    """
    mesh = ctx.mesh
    P = ctx.Hs[0] if body is None else body
    B = P[:BODY_VERTS]
    q = mesh.render_mh[mesh.quads]
    qc = B[q].mean(axis=1)
    p4 = B[q]
    n = np.cross(p4[:, 2] - p4[:, 0], p4[:, 3] - p4[:, 1])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rad = np.linalg.norm(qc - centre, axis=1)
    near = np.nonzero((rad > 0.010) & (rad < 0.021))[0]
    inward = ((qc[near] - centre) * n[near]).sum(axis=1) < 0
    cav = near[inward]
    # keep the connected component that contains the front pole quad
    verts_of = {int(i): set(map(int, q[i])) for i in cav}
    vert_to_q: dict[int, list[int]] = {}
    for i, vs in verts_of.items():
        for v in vs:
            vert_to_q.setdefault(v, []).append(i)
    seed = int(cav[np.argmax(qc[cav][:, 2])])
    comp, stack = {seed}, [seed]
    while stack:
        cur = stack.pop()
        for v in verts_of[cur]:
            for nb in vert_to_q[v]:
                if nb not in comp:
                    comp.add(nb)
                    stack.append(nb)
    comp_q = np.array(sorted(comp), dtype=np.int64)
    cav_verts = set(int(v) for i in comp_q for v in q[i])
    in_cav = np.zeros(len(q), dtype=bool)
    in_cav[comp_q] = True
    incident_skin = set()
    for qi in np.nonzero(~in_cav)[0]:
        for v in q[qi]:
            if int(v) in cav_verts:
                incident_skin.add(int(v))
    rim = np.array(sorted(incident_skin), dtype=np.int64)
    interior = np.array(sorted(cav_verts - incident_skin), dtype=np.int64)
    return {"quads": comp_q, "rim": rim, "interior": interior, "vertices": np.array(sorted(cav_verts), dtype=np.int64)}


def eye_socket_metrics(E: np.ndarray, rim_pts: np.ndarray) -> dict:
    """Eyeball vs eyelid opening on one body. E (n, 3): eyeball vertices; rim_pts (m, 3): eyelid opening vertices.

    `gap` per rim vertex = |rim - c| - r(direction), r = radius of the nearest eyeball vertex about the sphere fit
    centre c: > 0 the lid margin floats above the eyeball surface, < 0 the eyeball surface pokes through the margin.
    `protrusion` = distance of the eyeball's front pole in front of the rim's mean plane (along the rim normal).
    """
    c, r = _sphere_fit(E)
    d = np.linalg.norm(rim_pts[:, None, :] - E[None, :, :], axis=2)
    nn = E[np.argmin(d, axis=1)]
    gap = np.linalg.norm(rim_pts - c, axis=1) - np.linalg.norm(nn - c, axis=1)
    cen = rim_pts.mean(axis=0)
    u, s_, vt = np.linalg.svd(rim_pts - cen)
    nrm = vt[-1]
    if nrm[2] < 0:
        nrm = -nrm
    pole = E[np.argmax(E @ nrm)]
    return {"centre": c, "radius": r, "gap": gap, "gap_min": float(gap.min()), "gap_max": float(gap.max()),
            "protrusion": float((pole - cen) @ nrm)}


def find_iris(rgb: np.ndarray) -> tuple[tuple[float, float], float]:
    """(centre (u, v), radius) of the iris in an eye texture crop (glTF UV, v down, image units), from the texture.

    The pupil is the dark blob (luminance < 30) of the single-eye crop; the iris outer edge (limbus) is the radius of the
    strongest brightening of the mean luminance from the iris ring to the sclera."""
    h, w, _ = rgb.shape
    lum = rgb.astype(np.float64) @ parts_tex.LUMA
    ys, xs = np.nonzero(lum < 30.0)
    if len(ys) < 20:
        raise RuntimeError("no pupil found in the eye texture")
    cy, cx = float(ys.mean()), float(xs.mean())
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - cy, xx - cx)
    rmax = int(0.32 * w)
    prof = np.array([lum[(rr >= r) & (rr < r + 1)].mean() for r in range(rmax)])
    smooth = np.convolve(prof, np.ones(5) / 5.0, mode="same")
    grad = np.gradient(smooth)
    lo = int(0.10 * w)  # beyond the pupil and the iris ring, before the sclera plateau
    r_iris = lo + int(np.argmax(grad[lo:rmax]))
    return ((cx + 0.5) / w, (cy + 0.5) / h), (r_iris + 0.5) / w


# ---------------------------------------------------------------------------------------------------------------
# geometry of one part
# ---------------------------------------------------------------------------------------------------------------


def _rows(cl: mhclo_mod.Mhclo, rows: np.ndarray) -> mhclo_mod.Mhclo:
    return mhclo_mod.Mhclo(cl.header, cl.comments, cl.indices[rows], cl.weights[rows], cl.offsets[rows], cl.scale,
                           cl.delete_verts, cl.tags)


def part_mesh(pid: str, asset: dict, tex_alpha: np.ndarray | None) -> tuple[mhclo_mod.Mhclo, mhclo_mod.GarmentObj, dict]:
    """(MHCLO rows, OBJ, extra) of the mesh that is shipped: the eyes are reduced to the opaque inner shells."""
    if PARTS[pid]["category"] == "eyes":
        red = prepare_eyes(asset, tex_alpha)
        return _rows(asset["mhclo"], red["rows"]), red["obj"], {"left": red["left"]}
    return asset["mhclo"], asset["obj"], {}


def render_arrays(obj: mhclo_mod.GarmentObj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(render_v (obj vertex of each glb vertex), render_uv (OBJ convention), tris): UV-seam split, no orphan vertices."""
    render_v, render_uv, tris = mhclo_mod.render_split(obj)
    keep = np.unique(tris)
    if len(keep) != len(render_v):
        remap = -np.ones(len(render_v), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        render_v, render_uv, tris = render_v[keep], render_uv[keep], remap[tris]
    return render_v, render_uv, tris


def part_geometry(pid: str, asset: dict, ctx: PartsContext, neutral: np.ndarray, tex_alpha: np.ndarray | None) -> dict:
    """Everything derived from the MHCLO on the neutral body: render vertices, positions, binding, deletes, stats."""
    cl, obj, extra = part_mesh(pid, asset, tex_alpha)
    cat = PARTS[pid]["category"]
    first = ctx.first
    if cl.count != len(obj.verts):
        raise RuntimeError(f"{pid}: MHCLO has {cl.count} vertices, OBJ {len(obj.verts)}")
    if not (cl.indices >= BODY_VERTS).any():  # body-bound: exactly the garment path (ADR 0007)
        body = garments.BodyRef(neutral, ctx.mesh.render_count, first, {}, {})
        geo = garments.garment_geometry({"gid": pid, "mhclo": cl, "obj": obj}, body, ctx.mesh)
        geo["mode"] = "direct"
        geo["neutral_error_mm"] = float(np.abs(geo["obj_pos"] - truth_positions(cl, ctx.Hs[0])).max() * 1000.0)
        # the binding after the negative-weight clamp vs the MHCLO itself on every sample body
        T = np.stack([truth_positions(cl, P) for P in ctx.Hs])
        s = scales_on(cl, ctx.Hs)
        geo["error_mm"] = render_binding_error(ctx, geo, T, s) * 1000.0
        geo.update(extra)
        return geo
    # helper-bound: truth on the sample bodies, then re-bind to body triangles
    for k in "xyz":
        if max(cl.scale[k][:2]) >= BODY_VERTS:
            raise RuntimeError(f"{pid}: scale reference outside the body")
    T = np.stack([truth_positions(cl, P) for P in ctx.Hs])  # (B1, V, 3)
    s = scales_on(cl, ctx.Hs)
    if cat == "eyes":
        tri, w, off = fit_group(ctx, T, s, [np.nonzero(extra["left"])[0], np.nonzero(~extra["left"])[0]])
    else:
        tri, w, off = fit_per_vertex(ctx, T, s)
    err = binding_error(ctx, tri, w, off, T, s)
    render_v, render_uv, tris = render_arrays(obj)
    bind_idx = first[tri][render_v]  # canonical render copy of each MakeHuman body vertex
    body_mh = neutral[first]
    refs = {k: [int(first[cl.scale[k][0]]), int(first[cl.scale[k][1]]), round(cl.scale[k][2] * DM_TO_M, 6)] for k in "xyz"}
    s_neutral = np.array([abs(body_mh[cl.scale[k][0], i] - body_mh[cl.scale[k][1], i]) / (cl.scale[k][2] * DM_TO_M)
                          for i, k in enumerate("xyz")])
    obj_pos = (w[:, :, None] * body_mh[tri]).sum(axis=1) + off * s_neutral
    pos = obj_pos[render_v]
    normals = mh_obj.vertex_normals(pos, tris, render_v)
    return {
        "mode": "rebound", "obj_pos": obj_pos, "render_v": render_v, "render_uv": render_uv, "tris": tris, "pos": pos,
        "normals": normals, "bind_idx": bind_idx, "bind_w": w[render_v], "bind_off": off[render_v],
        "delete": np.zeros(0, dtype=np.int64), "scale_refs": refs, "scale": s_neutral, "negative_weight": 0.0,
        "error_mm": err * 1000.0, "neutral_error_mm": float(err[0].max() * 1000.0), "obj_tri": tri, **extra,
    }


def render_binding_error(ctx: PartsContext, geo: dict, T: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(B1, Rv) distance (meters) between a per-render-vertex binding (garment path) and the MHCLO truth T (B1, V, 3)."""
    idx_mh = ctx.mesh.render_mh[geo["bind_idx"]]
    w, off = geo["bind_w"], geo["bind_off"]
    pred = np.stack([(w[:, :, None] * ctx.Hs[b][idx_mh]).sum(axis=1) + off * s[b] for b in range(ctx.Hs.shape[0])])
    return np.linalg.norm(pred - T[:, geo["render_v"]], axis=2)


# ---------------------------------------------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------------------------------------------


def prepare_part_texture(pid: str, asset: dict, geo: dict) -> dict:
    """{'bytes','mime','factor','alpha','cutoff','size','iris'?, 'render_uv'?}: the embedded texture of one part."""
    cat = PARTS[pid]["category"]
    path: Path = asset["texture"]
    rgba = parts_tex.load_rgba(path)
    if cat == "eyes":
        return _eye_texture(rgba, geo)
    rgba = parts_tex.resize_rgba(rgba, PART_TEXTURE_MAX[cat])
    has_alpha = int(rgba[..., 3].min()) < 250  # a fully opaque texture (flat-colour hair) is OPAQUE, not MASK
    rgba = parts_tex.bleed_colors(rgba)
    cutoff = parts_tex.coverage_cutoff(rgba[..., 3]) if has_alpha else None
    if CATEGORY_MATERIAL[cat]["tintable"]:
        la, tint_lin = parts_tex.neutralise(rgba)
        factor = [round(float(x), 6) for x in tint_lin] + [1.0]
    else:
        luma = np.rint(rgba[..., :3].astype(np.float64) @ parts_tex.LUMA)
        la = np.stack([luma, rgba[..., 3]], axis=-1).clip(0, 255).astype(np.uint8)
        factor = [1.0, 1.0, 1.0, 1.0]
    arr = la if has_alpha else la[..., 0]  # LA or L
    return {"bytes": parts_tex.encode_png(arr), "mime": "image/png", "factor": factor, "alpha": has_alpha, "cutoff": cutoff,
            "size": rgba.shape[1::-1]}


def _eye_texture(rgba: np.ndarray, geo: dict) -> dict:
    """Crop the texture island of the first eye (square, with margin), 512 px, JPEG; remap the render UVs accordingly."""
    uv = geo["render_uv"]  # OBJ convention (v up), glb flips v
    h, w, _ = rgba.shape
    px = uv[:, 0] * w
    py = (1.0 - uv[:, 1]) * h
    margin = 0.02 * w
    x0, x1 = px.min() - margin, px.max() + margin
    y0, y1 = py.min() - margin, py.max() + margin
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = int(round(side))
    x0, y0 = max(0, min(x0, w - side)), max(0, min(y0, h - side))
    crop = rgba[y0: y0 + side, x0: x0 + side, :3]
    from PIL import Image

    size = 512
    small = np.asarray(Image.fromarray(crop).resize((size, size), Image.LANCZOS))
    new_uv = np.stack([(px - x0) / side, (py - y0) / side], axis=1)  # glTF convention already (v down)
    centre, radius = find_iris(small)
    return {"bytes": parts_tex.encode_jpeg(small, 92), "mime": "image/jpeg", "factor": [1.0, 1.0, 1.0, 1.0], "alpha": False,
            "cutoff": None, "size": (size, size), "uv_gltf": new_uv, "iris": {"center": centre, "radius": radius}}


# ---------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------


def pack_binding(idx: np.ndarray, w: np.ndarray, off: np.ndarray) -> bytes:
    return garments.pack_binding(idx, w, off)


def _attribution(pid: str, lic: str, upstream_lic: str, author: str, original: str, source_url: str) -> str:
    who = "the MakeHuman project (makehuman_system)" if author == "makehuman_system" else author
    who += f", original by {original}" if original else ""
    if lic == "CC-BY-4.0":
        ver = "4.0" if "4" in upstream_lic else "(version not stated upstream, 4.0 assumed)"
        licence = f"CC BY {ver} (https://creativecommons.org/licenses/by/4.0/)"
    else:
        licence = "CC0 1.0 (public domain dedication)"
    return (f'"{PARTS[pid]["title"]}" by {who}, MakeHuman community assets ({source_url}), {licence}. '
            "Changes: bound to a parametric body through the MHCLO binding, texture downscaled/converted, converted to glTF.")


def build_parts(mesh: mh_obj.BaseMesh, neutral: np.ndarray, morph: MhMorpher, offset_y: float, out_dir: Path,
                verbose: bool = True) -> dict:
    """Write all body parts + index.json into out_dir. Returns per-part details (for logging/tests)."""
    log = print if verbose else (lambda *a, **k: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = make_context(mesh, morph, offset_y)
    if np.abs(neutral[ctx.first] - ctx.Hs[0][:BODY_VERTS]).max() > 5e-5:
        raise RuntimeError("full-mesh neutral body does not match the shipped morph model")
    entries, details = [], {}
    for pid, spec in PARTS.items():
        cat = spec["category"]
        asset = load_asset(pid)
        tex_alpha = parts_tex.load_rgba(asset["texture"])[..., 3] if cat == "eyes" else None
        geo = part_geometry(pid, asset, ctx, neutral, tex_alpha)
        tex = prepare_part_texture(pid, asset, geo)
        uv = geo["render_uv"]
        if cat == "eyes":
            uv = np.stack([tex["uv_gltf"][:, 0], 1.0 - tex["uv_gltf"][:, 1]], axis=1)  # write_static_glb flips v again
        mat = CATEGORY_MATERIAL[cat]
        gltf_writer.write_static_glb(
            out_dir / f"{pid}.glb", pid, geo["pos"], geo["normals"], uv, geo["tris"], tex["bytes"], tex["mime"], tex["factor"],
            tex["alpha"], alpha_cutoff=tex["cutoff"], double_sided=mat["double_sided"], roughness=mat["roughness"],
        )
        (out_dir / f"{pid}.bind.bin").write_bytes(pack_binding(geo["bind_idx"], geo["bind_w"], geo["bind_off"]))
        delete = geo["delete"]
        if cat == "eyes":
            delete = _eyes_delete(ctx, geo)
            geo["delete"] = delete
        delete_name = None
        if len(delete):
            delete_name = f"{pid}.delete.bin"
            (out_dir / delete_name).write_bytes(np.ascontiguousarray(delete, dtype="<u4").tobytes())
        elif (out_dir / f"{pid}.delete.bin").exists():
            (out_dir / f"{pid}.delete.bin").unlink()
        a = asset["asset"]
        author = mhclo_mod.author_of(asset["mhclo"]) or asset["pack_entry"].get("author", "")
        pack_author = asset["pack_entry"].get("author", "")
        if pack_author and pack_author.lower() != author.lower():
            author = f"{author} ({pack_author})" if author else pack_author
        orig = asset["pack_entry"].get("original_author", "")
        src_url = asset["pack_entry"].get("source", "").replace("http://www.makehumancommunity.org", "https://www.makehumancommunity.org")
        pack_dir = a["pack"].rsplit("_", 1)[0]
        member = next(m for m in a["files"] if m.endswith(asset["clo_name"]))
        source = (f"MakeHuman community assets pack {a['pack']} ({PART_PACK_URL.format(name=pack_dir, pack=a['pack'])}) "
                  f"{member} sha256:{a['files'][member]}")
        d: dict = {"id": pid, "category": cat, "label": spec["label"], "license": asset["license"]}
        d["attribution"] = _attribution(pid, asset["license"], asset["license_text"], author, orig, src_url)
        d.update({"source": source, "mesh": f"{pid}.glb", "binding": f"{pid}.bind.bin", "scaleRefs": geo["scale_refs"]})
        if delete_name:
            d["deleteVerts"] = delete_name
        m: dict = {"alphaMode": "MASK" if tex["alpha"] else "OPAQUE"}
        if tex["alpha"]:
            m["alphaCutoff"] = tex["cutoff"]
        m.update({"doubleSided": mat["double_sided"], "tintable": mat["tintable"]})
        d["material"] = m
        if cat == "eyes":
            d["irisUv"] = {"center": [round(tex["iris"]["center"][0], 4), round(tex["iris"]["center"][1], 4)],
                           "radius": round(tex["iris"]["radius"], 4)}
        entries.append(d)
        details[pid] = {"geo": geo, "tex": tex, "asset": asset, "def": d}
        size = sum((out_dir / n).stat().st_size for n in (d["mesh"], d["binding"], *([delete_name] if delete_name else [])))
        err = f"  err mean {geo['error_mm'][1:].mean():.2f} max {geo['error_mm'][1:].max():.2f} mm" if "error_mm" in geo else ""
        log(f"[parts] {pid:18s} {len(geo['pos']):6d} verts {len(geo['tris']):6d} tris {size / 1024:8.1f} KiB  {geo['mode']}{err}")
    writers.write_json(out_dir / "index.json", {"version": 1, "parts": entries, "defaults": dict(DEFAULTS)})
    details["_ctx"] = ctx
    return details


def _eyes_delete(ctx: PartsContext, geo: dict) -> np.ndarray:
    """Render ids of the socket-cavity vertices of both eyes that touch no skin (safe to hide while eyes are shown)."""
    out = []
    pos = geo["obj_pos"]
    left = geo["left"]
    for sel in (left, ~left):
        c, _ = _sphere_fit(pos[sel])
        cav = eye_cavity(ctx, c)
        for v in cav["interior"]:
            out.append(ctx.mesh.render_ids_of(int(v)))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def part_files(out_dir: Path) -> list[str]:
    """Relative names of every file build_parts writes (for hashing / --check)."""
    idx = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    names = ["index.json"]
    for p in idx["parts"]:
        names += [p["mesh"], p["binding"], *([p["deleteVerts"]] if p.get("deleteVerts") else [])]
    return names


# ---------------------------------------------------------------------------------------------------------------
# runtime formula + debug renders (`.cache/debug/parts_*.png`, not shipped)
# ---------------------------------------------------------------------------------------------------------------


def load_binding(path: Path):
    raw = path.read_bytes()
    rec = np.frombuffer(raw, dtype=[("i", "<u4", 3), ("w", "<f4", 3), ("o", "<f4", 3)])
    return rec["i"].astype(np.int64), rec["w"].astype(np.float64), rec["o"].astype(np.float64)


def apply_binding(idx: np.ndarray, w: np.ndarray, off: np.ndarray, refs: dict, body: np.ndarray) -> np.ndarray:
    """The runtime formula (`bindGarment`): sum w.v[i] + offset * per-axis |v[a] - v[b]| / refM."""
    s = np.array([abs(body[refs[k][0], i] - body[refs[k][1], i]) / refs[k][2] for i, k in enumerate("xyz")])
    return (w[:, :, None] * body[idx]).sum(axis=1) + off * s


def read_part_glb(path: Path) -> dict:
    import io

    import pygltflib
    from PIL import Image

    gl = pygltflib.GLTF2().load_binary(str(path))
    blob = gl.binary_blob()
    prim = gl.meshes[0].primitives[0]

    def acc(i, dt, n):
        a = gl.accessors[i]
        v = gl.bufferViews[a.bufferView]
        return np.frombuffer(blob, dtype=dt, count=a.count * n, offset=v.byteOffset)

    ia = gl.accessors[prim.indices]
    mat = gl.materials[0]
    view = gl.bufferViews[gl.images[0].bufferView]
    tex = np.asarray(Image.open(io.BytesIO(blob[view.byteOffset: view.byteOffset + view.byteLength])).convert("RGBA"))
    return {
        "pos": acc(prim.attributes.POSITION, "<f4", 3).reshape(-1, 3).astype(np.float64),
        "uv": acc(prim.attributes.TEXCOORD_0, "<f4", 2).reshape(-1, 2).astype(np.float64),
        "tris": acc(prim.indices, "<u2" if ia.componentType == 5123 else "<u4", 1).reshape(-1, 3).astype(np.int64),
        "tex": tex, "factor": mat.pbrMetallicRoughness.baseColorFactor, "mask": mat.alphaMode == "MASK",
        "cutoff": mat.alphaCutoff if mat.alphaMode == "MASK" else 0.0, "double_sided": bool(mat.doubleSided),
    }


def _debug_layer(g: dict, pos: np.ndarray | None = None) -> dict:
    tint = parts_tex.linear_to_srgb(np.array(g["factor"][:3])) * 255.0
    return {"pos": g["pos"] if pos is None else pos, "tris": g["tris"], "uv": g["uv"], "tex": g["tex"], "cutoff": g["cutoff"],
            "color": (255, 255, 255), "tint": tint, "cull": not g["double_sided"]}


def write_debug_images(ctx: PartsContext, out_dir: Path, debug_dir: Path, hair_ids: list[str] | None = None) -> None:
    """Front / side head renders with the default eyes, brows, lashes and hair, one render per hair style, and the eyes on
    extreme bodies. The body is drawn flat (skin colour), the parts textured with their alpha cutoff."""
    import debug_png
    from parts_debug import Scene

    idx = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    defs = {p["id"]: p for p in idx["parts"]}
    mesh = ctx.mesh
    glbs = {pid: read_part_glb(out_dir / d["mesh"]) for pid, d in defs.items()}
    bind = {pid: load_binding(out_dir / d["binding"]) for pid, d in defs.items()}
    skin = (214, 165, 140)

    def body_layer(P_mh, y_min=1.20):
        Pr = P_mh[mesh.render_mh]
        tris = mesh.tris[(Pr[mesh.tris][:, :, 1] > y_min).all(axis=1)]
        return Pr, {"pos": Pr, "tris": tris, "color": skin, "cull": True}

    def part_layers(Pr, ids):
        out = []
        for pid in ids:
            i, w, o = bind[pid]
            out.append(_debug_layer(glbs[pid], apply_binding(i, w, o, defs[pid]["scaleRefs"], Pr)))
        return out

    eyes, brow, lash = DEFAULTS["eyes"], DEFAULTS["eyebrows"], DEFAULTS["eyelashes"]
    hairs = hair_ids or [p for p in defs if defs[p]["category"] == "hair"]
    face = [eyes, brow, lash]
    Pr, base = body_layer(ctx.Hs[0])
    head_bb = (np.array([-0.13, 1.40, -0.2]), np.array([0.13, 1.76, 0.2]))
    side_bb = (np.array([-1, 1.40, -0.16]), np.array([1, 1.76, 0.16]))
    first_hair = hairs[0] if hairs else None
    dressed = face + ([first_hair] if first_hair else [])
    debug_png.write_png(debug_dir / "parts_head_front.png", Scene([base], "front", head_bb, 700).render(part_layers(Pr, dressed)))
    debug_png.write_png(debug_dir / "parts_head_side.png", Scene([base], "side", side_bb, 700).render(part_layers(Pr, dressed)))
    eye_bb = (np.array([-0.075, 1.505, -1]), np.array([0.075, 1.585, 1]))
    debug_png.write_png(debug_dir / "parts_face_closeup.png", Scene([base], "front", eye_bb, 900).render(part_layers(Pr, face)))
    # every hair style, front | side, on one shared framing (long hair reaches the shoulder blades)
    bb_f = (np.array([-0.17, 1.05, -1]), np.array([0.17, 1.78, 1]))
    bb_s = (np.array([-1, 1.05, -0.22]), np.array([1, 1.78, 0.22]))
    sc_f, sc_s = Scene([base], "front", bb_f, 520), Scene([base], "side", bb_s, 520)
    for pid in hairs:
        f = sc_f.render(part_layers(Pr, face + [pid]))
        s = sc_s.render(part_layers(Pr, face + [pid]))
        h = min(f.shape[0], s.shape[0])
        debug_png.write_png(debug_dir / f"parts_{pid}.png", np.concatenate([f[:h], s[:h]], axis=1))
    # eyes + lashes + brows on extreme bodies (macro corners), close-up around each body's own eyes
    panels = []
    for macros in ({"gender": 0.0}, {"gender": 1.0}, {"height": 0.0, "weight": 1.0}, {"height": 1.0, "gender": 1.0, "muscle": 1.0}):
        P = ctx.body(macros)
        Pr_e = P[mesh.render_mh]
        eyes_pos = apply_binding(*bind[eyes], defs[eyes]["scaleRefs"], Pr_e)
        c = eyes_pos.mean(axis=0)
        Pr_e, base_e = body_layer(P, c[1] - 0.17)
        bb = (np.array([c[0] - 0.075, c[1] - 0.04, -1]), np.array([c[0] + 0.075, c[1] + 0.04, 1]))
        panels.append(Scene([base_e], "front", bb, 520).render(part_layers(Pr_e, face)))
    hh = min(p.shape[0] for p in panels)
    ww = min(p.shape[1] for p in panels)
    grid = np.concatenate([np.concatenate([panels[0][:hh, :ww], panels[1][:hh, :ww]], axis=1),
                           np.concatenate([panels[2][:hh, :ww], panels[3][:hh, :ww]], axis=1)], axis=0)
    debug_png.write_png(debug_dir / "parts_eyes_extremes.png", grid)
