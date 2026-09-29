"""Garment templates: MHCLO proxies -> `<id>.glb`, `<id>.bind.bin`, `<id>.delete.bin`, `index.json`.

Frame: every position is in the body runtime rest frame of `base.glb` / `manifest.json`: meters, +Y up, +Z front, feet
of the NEUTRAL body (macro defaults) on y = 0 (the same `groundOffsetY` as the body). The garment glb POSITION is the
MHCLO reconstruction on that neutral body; at runtime `bindGarment` recomputes it from the solved body.

Index space: binding indices address the body's render vertices (ADR 0005). Every MakeHuman vertex maps to its first
(canonical) render copy (`first_copy`); all copies share one position, so any copy would do. Garment vertices are the
garment OBJ's own (v, vt) render vertices (UV-seam split like the body), one 36-byte record each.

Scale refs: MHCLO `x_scale a b d` -> `scaleRefs.x = [render(a), render(b), d * 0.1]` (decimeters -> meters); the
runtime scale is |body[a].x - body[b].x| / refM, applied to the (meter) offsets.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fetch
import gltf_writer
import measures as measures_mod
import mh_obj
import mhclo as mhclo_mod
import writers
from config import DM_TO_M, GARMENT_ASSETS, GARMENT_TEXTURE_MAX

BODY_VERTS = mh_obj.BODY_VERTEX_COUNT
PACK_URL = "https://files2.makehumancommunity.org/asset_packs/{name}/{pack}.zip"

# Template catalogue (order = index.json order). Measures decide which nativeMeasures are computed.
TEMPLATES: dict[str, dict] = {
    "tshirt": {"kind": "tshirt", "category": "top", "layer": 1, "label": {"tr": "Tişört", "en": "T-shirt"}},
    "sweatshirt": {"kind": "sweatshirt", "category": "top", "layer": 2, "label": {"tr": "Kazak", "en": "Sweater"}},
    "pants": {"kind": "pants", "category": "bottom", "layer": 1, "label": {"tr": "Kumaş pantolon", "en": "Trousers"}},
    "jeans": {"kind": "jeans", "category": "bottom", "layer": 1, "label": {"tr": "Kot pantolon", "en": "Jeans"}},
    "sneakers": {"kind": "sneakers", "category": "shoes", "layer": 1, "label": {"tr": "Spor ayakkabı", "en": "Sneakers"}},
    "shoes": {"kind": "shoes", "category": "shoes", "layer": 1, "label": {"tr": "Kumaş ayakkabı", "en": "Cloth shoes"}},
    "boots": {"kind": "boots", "category": "shoes", "layer": 1, "label": {"tr": "Bot", "en": "Boots"}},
}

SHOE_INNER_MARGIN_M = 0.015  # inner length = outer length - 1.5 cm (sole/toe-cap/heel-counter thickness, see ADR 0007)
EASE_STEP_CM = 0.5


@dataclass
class BodyRef:
    """Neutral body data the garment measurements are taken against (combined index space, meters)."""

    P: np.ndarray
    R: int
    first: np.ndarray  # (13380,) canonical render id per MakeHuman body vertex
    landmarks: dict
    defs: dict[str, dict]  # measure id -> definition (measures.json entries)

    def body_cm(self, mid: str) -> float:
        return measures_mod.evaluate(self.defs[mid], self.P, self.R) * 100.0

    def loop_halfwidth(self, mid: str) -> float:
        return float(np.abs(self.P[self.defs[mid]["verts"], 0]).max())


# ---------------------------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------------------------


def slice_points(pos: np.ndarray, tris: np.ndarray, y: float) -> np.ndarray:
    """Intersection points (n, 3) of the horizontal plane y with every triangle edge that crosses it."""
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p, q = pos[tris[:, a]], pos[tris[:, b]]
        m = (p[:, 1] - y) * (q[:, 1] - y) < 0
        p, q = p[m], q[m]
        t = ((y - p[:, 1]) / (q[:, 1] - p[:, 1]))[:, None]
        pts.append(p + t * (q - p))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def section_circumference(pos, tris, y, x_lo=-np.inf, x_hi=np.inf) -> float | None:
    """Tape-measure length (convex hull perimeter, meters) of the garment cross-section at height y, restricted to
    x in [x_lo, x_hi] (to keep sleeves/legs out of a torso section). None if the garment does not cover y."""
    pts = slice_points(pos, tris, y)
    pts = pts[(pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)]
    if len(pts) < 3:
        return None
    return measures_mod._hull_perimeter(pts[:, [0, 2]])


def _round_ease(v: float) -> float:
    return round(round(v / EASE_STEP_CM) * EASE_STEP_CM, 1)


def sleeve_length(pos: np.ndarray, body: BodyRef) -> float | None:
    """Arc length (meters) from the left acromion along the body arm path (acromion -> elbow -> wrist) to the farthest
    garment vertex on the left arm; the cuff. The path is the body's own `armLength` polyline, so it is comparable."""
    lm, P = body.landmarks, body.P
    a, e, w = P[lm["acromion_l"]], P[lm["elbow_v"]], P[lm["wrist_v"]]
    d1, d2 = e - a, w - e
    l1, l2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
    v = pos[pos[:, 0] > 0]
    best_d = np.full(len(v), np.inf)
    best_s = np.zeros(len(v))
    for start, d, ln, base, t_max in ((a, d1, l1, 0.0, l1), (e, d2, l2, l1, l2 * 1.4)):
        u = d / ln
        t = np.clip((v - start) @ u, 0.0, t_max)  # nearest point on the segment; the last one extends past the wrist
        dist = np.linalg.norm(v - (start + t[:, None] * u), axis=1)
        closer = dist < best_d
        best_d[closer], best_s[closer] = dist[closer], base + t[closer]
    ok = best_d < 0.12
    return float(best_s[ok].max()) if ok.any() else None


def measure_template(gid: str, cat: str, pos: np.ndarray, tris: np.ndarray, body: BodyRef) -> tuple[dict, dict, dict]:
    """(nativeMeasures cm, defaultEase cm, details) of one garment, all on the neutral body."""
    lm, P = body.landmarks, body.P
    native: dict[str, float] = {}
    ease: dict[str, float] = {}
    notes: dict[str, str] = {}
    y_top, y_bot = float(pos[:, 1].max()), float(pos[:, 1].min())

    def add_girth(mid: str, plane_y: float, x_lo=-np.inf, x_hi=np.inf, ref: str | None = None) -> None:
        val = section_circumference(pos, tris, plane_y, x_lo, x_hi)
        if val is None:
            return
        cm = round(val * 100.0, 1)
        native[mid] = cm
        ease[mid] = _round_ease(cm - body.body_cm(ref or mid))

    if cat == "top":
        hw = body.loop_halfwidth("chest") + 0.035
        add_girth("chest", lm["bust_y"], -hw, hw)
        for mid, key in (("waist", "waist_y"), ("hip", "hip_y")):
            add_girth(mid, lm[key], -hw, hw)
        torso = pos[np.abs(pos[:, 0]) <= hw]
        native["length"] = round((y_top - float(torso[:, 1].min())) * 100.0, 1)
        s = sleeve_length(pos, body)
        if s is not None:
            native["sleeve"] = round(float(s) * 100.0, 1)
            ease["sleeve"] = _round_ease(native["sleeve"] - body.body_cm("armLength"))
    elif cat == "bottom":
        hw = body.loop_halfwidth("hip") + 0.035
        waist_y = lm["waist_y"]
        if waist_y > y_top - 0.005:  # low-rise: measure the waistband (top edge) instead of the anatomical waist
            waist_y = y_top - 0.01
            notes["waist"] = "waistband (1 cm below the top edge); the garment does not reach the body waist plane"
        add_girth("waist", waist_y, -hw, hw, ref="waist")
        add_girth("hip", lm["hip_y"], -hw, hw)
        crotch_y = float(P[lm["crotch"], 1])
        add_girth("thigh", crotch_y - 0.03, 0.0, hw)
        native["length"] = round((y_top - y_bot) * 100.0, 1)
        native["inseam"] = round((crotch_y - y_bot) * 100.0, 1)
        ease["inseam"] = _round_ease(native["inseam"] - body.body_cm("inseam"))
    else:  # shoes: left shoe only
        left = pos[pos[:, 0] > 0]
        outer = float(left[:, 2].max() - left[:, 2].min())
        native["footLength"] = round((outer - SHOE_INNER_MARGIN_M) * 100.0, 1)
        ease["footLength"] = _round_ease(native["footLength"] - body.body_cm("footLength"))
        notes["footLength"] = f"outer length {outer * 100:.1f} cm minus {SHOE_INNER_MARGIN_M * 100:.1f} cm"
    return native, ease, notes


# ---------------------------------------------------------------------------------------------------------------
# texture
# ---------------------------------------------------------------------------------------------------------------


def prepare_texture(path: Path | None, diffuse: list[float]) -> tuple[bytes | None, str, str, bool, list[float]]:
    """(bytes, mime, baseColor hex, alpha_mask, glTF baseColorFactor). Downscaled to <= GARMENT_TEXTURE_MAX px."""
    if path is None or not path.exists():
        col = [min(max(c, 0.0), 1.0) for c in diffuse]
        return None, "", _hex(col), False, [*_srgb_to_linear(col), 1.0]
    from PIL import Image

    img = Image.open(path)
    img.load()
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    img = img.convert("RGBA" if has_alpha else "RGB")
    if max(img.size) > GARMENT_TEXTURE_MAX:
        k = GARMENT_TEXTURE_MAX / max(img.size)
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
    arr = np.asarray(img)
    alpha_mask = bool(has_alpha and arr[..., 3].min() < 250)
    rgb = arr[..., :3].reshape(-1, 3).astype(np.float64) / 255.0
    if alpha_mask:
        w = arr[..., 3].reshape(-1).astype(np.float64) / 255.0
        mean = (rgb * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9)
    else:
        mean = rgb.mean(axis=0)
    buf = io.BytesIO()
    if alpha_mask:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=88, optimize=False, progressive=False)
        mime = "image/jpeg"
    return buf.getvalue(), mime, _hex(mean.tolist()), alpha_mask, [1.0, 1.0, 1.0, 1.0]


def _srgb_to_linear(c: list[float]) -> list[float]:
    return [round(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4, 6) for x in c]


def _hex(c: list[float]) -> str:
    return "#" + "".join(f"{int(round(min(max(v, 0.0), 1.0) * 255)):02x}" for v in c[:3])


# ---------------------------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------------------------


TITLES = {
    "tshirt": "T-shirt (basic, tucked)", "sweatshirt": "Fisherman sweater", "pants": "Wool pants",
    "jeans": "Male classic jeans", "sneakers": "Brown sneakers", "shoes": "MJ cloth shoes", "boots": "Hero boots 2",
}


def _attribution(gid: str, lic: str, upstream_lic: str, author: str, original: str, source_url: str) -> str:
    """TASL-style credit line: title, author (+ original author), source, licence, changes."""
    who = f"{author}" + (f", original by {original}" if original else "")
    if lic == "CC-BY-4.0":
        ver = "4.0" if "4" in upstream_lic else "(version not stated upstream, 4.0 assumed)"
        licence = f"CC BY {ver} (https://creativecommons.org/licenses/by/4.0/)"
    else:
        licence = "CC0 1.0 (public domain dedication)"
    return (f'"{TITLES[gid]}" by {who}, MakeHuman community assets ({source_url}), {licence}. '
            "Changes: re-fitted to a parametric body through the MHCLO binding, texture downscaled, converted to glTF.")


def load_asset(gid: str) -> dict:
    """Read + validate one cached upstream asset. Raises on an unacceptable or inconsistent licence."""
    a = GARMENT_ASSETS[gid]
    d = fetch.garment_dir(gid)
    clo = next(n for n in a["files"] if n.endswith(".mhclo"))
    obj_name = next(n for n in a["files"] if n.endswith(".obj"))
    mat_name = next(n for n in a["files"] if n.endswith(".mhmat"))
    mat_text = mhclo_mod.read_text(d / mat_name)
    cl = mhclo_mod.parse_mhclo(mhclo_mod.read_text(d / clo))
    lic_text = mhclo_mod.license_text(cl, mat_text)
    lic = mhclo_mod.classify_license(lic_text)
    pack_json = json.loads((d / "pack.json").read_text(encoding="utf-8"))
    entry = pack_json[clo[: -len(".mhclo")]]
    pack_lic = mhclo_mod.classify_license(entry.get("license", ""))
    if lic is None or lic != pack_lic:
        raise RuntimeError(f"{gid}: licence not acceptable or inconsistent (mhclo {lic_text!r}, pack {entry.get('license')!r})")
    if lic == "CC0-1.0" and not a["pack"].endswith(("cc0", "ccby")):
        raise RuntimeError(f"{gid}: unexpected pack {a['pack']}")
    diffuse = mhclo_mod.parse_mhmat(mat_text).get("diffuseTexture")
    tex_path = d / diffuse.split("/")[-1] if diffuse else None
    return {
        "gid": gid, "asset": a, "dir": d, "mhclo": cl, "obj": mhclo_mod.parse_garment_obj(mhclo_mod.read_text(d / obj_name)),
        "mhmat": mhclo_mod.parse_mhmat(mat_text), "texture": tex_path, "license": lic, "license_text": lic_text,
        "pack_entry": entry, "clo_name": clo,
    }


def _neg_weight_fix(w: np.ndarray, idx: np.ndarray, off_m: np.ndarray, s: np.ndarray, body_mh: np.ndarray):
    """MHCLO weights may extrapolate slightly outside the triangle (negative weights); the runtime rejects them.
    Clamp to >= 0, renormalise, and fold the position difference on the neutral body into the offset (exact there)."""
    w2 = np.clip(w, 0.0, None)
    w2 = w2 / w2.sum(axis=1, keepdims=True)
    delta = ((w - w2)[:, :, None] * body_mh[idx]).sum(axis=1)  # meters
    return w2, off_m + delta / s, float(-w.min()) if w.min() < 0 else 0.0


def garment_geometry(asset: dict, body: BodyRef, mesh: mh_obj.BaseMesh) -> dict:
    """Everything derived from the MHCLO on the neutral body (float64): render vertices, positions, binding."""
    cl: mhclo_mod.Mhclo = asset["mhclo"]
    obj: mhclo_mod.GarmentObj = asset["obj"]
    if cl.count != len(obj.verts):
        raise RuntimeError(f"{asset['gid']}: MHCLO has {cl.count} vertices, OBJ {len(obj.verts)}")
    if cl.indices.max() >= BODY_VERTS or cl.indices.min() < 0:
        raise RuntimeError(f"{asset['gid']}: MHCLO references non-body vertices (max {cl.indices.max()})")
    for k in "xyz":
        if max(cl.scale[k][:2]) >= BODY_VERTS:
            raise RuntimeError(f"{asset['gid']}: scale reference outside the body")
    body_mh = body.P[body.first]  # (13380, 3) meters
    s = mhclo_mod.axis_scales({k: (a, b, d * DM_TO_M) for k, (a, b, d) in cl.scale.items()}, body_mh)
    obj_pos = mhclo_mod.reconstruct(cl.indices, cl.weights, cl.offsets * DM_TO_M, s, body_mh)  # (V, 3) meters
    weights, off_m, neg = _neg_weight_fix(cl.weights, cl.indices, cl.offsets * DM_TO_M, s, body_mh)
    render_v, render_uv, tris = mhclo_mod.render_split(obj)
    keep = np.unique(tris)
    if len(keep) != len(render_v):  # orphan render vertices (referenced by no face): drop and remap
        remap = -np.ones(len(render_v), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        render_v, render_uv, tris = render_v[keep], render_uv[keep], remap[tris]
    pos = obj_pos[render_v]
    normals = mh_obj.vertex_normals(pos, tris, render_v)
    bind_idx = body.first[cl.indices][render_v]  # (Rg, 3) render ids of the body
    delete_mh = cl.delete_verts[cl.delete_verts < BODY_VERTS]
    delete = np.sort(np.concatenate([mesh.render_ids_of(int(v)) for v in delete_mh])) if len(delete_mh) else np.zeros(0, np.int64)
    scale_refs = {k: [int(body.first[cl.scale[k][0]]), int(body.first[cl.scale[k][1]]), round(cl.scale[k][2] * DM_TO_M, 6)] for k in "xyz"}
    return {
        "obj_pos": obj_pos, "render_v": render_v, "render_uv": render_uv, "tris": tris, "pos": pos, "normals": normals,
        "bind_idx": bind_idx, "bind_w": weights[render_v], "bind_off": off_m[render_v], "delete": delete,
        "scale_refs": scale_refs, "scale": s, "negative_weight": neg,
    }


def pack_binding(idx: np.ndarray, w: np.ndarray, off: np.ndarray) -> bytes:
    rec = np.zeros(len(idx), dtype=[("i", "<u4", 3), ("w", "<f4", 3), ("o", "<f4", 3)])
    rec["i"], rec["w"], rec["o"] = idx, w, off
    assert rec.dtype.itemsize == 36
    return rec.tobytes()


def build_garments(mesh: mh_obj.BaseMesh, neutral: np.ndarray, landmarks: dict, measure_defs: list[dict],
                   out_dir: Path, verbose: bool = True) -> dict:
    """Write all garment templates + index.json into out_dir. Returns per-template details (for logging/tests)."""
    log = print if verbose else (lambda *a, **k: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = mesh.mh_to_render_ids[mesh.mh_to_render_start[:BODY_VERTS]]
    body = BodyRef(neutral, mesh.render_count, first, landmarks, {d["id"]: d for d in measure_defs})
    entries, details = [], {}
    for gid, tpl in TEMPLATES.items():
        asset = load_asset(gid)
        geo = garment_geometry(asset, body, mesh)
        a = asset["asset"]
        tex_bytes, mime, base_hex, alpha_mask, factor = prepare_texture(asset["texture"], _diffuse(asset["mhmat"]))
        gltf_writer.write_static_glb(
            out_dir / f"{gid}.glb", gid, geo["pos"], geo["normals"], geo["render_uv"], geo["tris"],
            tex_bytes, mime, factor, alpha_mask,
        )
        (out_dir / f"{gid}.bind.bin").write_bytes(pack_binding(geo["bind_idx"], geo["bind_w"], geo["bind_off"]))
        delete_name = None
        if len(geo["delete"]):
            delete_name = f"{gid}.delete.bin"
            (out_dir / delete_name).write_bytes(np.ascontiguousarray(geo["delete"], dtype="<u4").tobytes())
        elif (out_dir / f"{gid}.delete.bin").exists():
            (out_dir / f"{gid}.delete.bin").unlink()
        native, ease, notes = measure_template(gid, tpl["category"], geo["pos"], geo["tris"], body)
        pack_name = a["pack"].split("_")[0]
        author = mhclo_mod.author_of(asset["mhclo"]) or asset["pack_entry"].get("author", "")
        pack_author = asset["pack_entry"].get("author", "")
        if pack_author and pack_author.lower() != author.lower():
            author = f"{author} ({pack_author})" if author else pack_author
        orig = asset["pack_entry"].get("original_author", "")
        src_url = asset["pack_entry"].get("source", "").replace("http://www.makehumancommunity.org", "https://www.makehumancommunity.org")
        source = (
            f"MakeHuman community assets pack {a['pack']} ({PACK_URL.format(name=pack_name, pack=a['pack'])}) "
            f"clothes/{a['dir']}/{asset['clo_name']} sha256:{a['files'][asset['clo_name']]}"
        )
        d = {
            "id": gid, "kind": tpl["kind"], "category": tpl["category"], "label": tpl["label"],
            "license": asset["license"],
        }
        d["attribution"] = _attribution(gid, asset["license"], asset["license_text"], author, orig, src_url)
        d.update({
            "source": source, "mesh": f"{gid}.glb", "binding": f"{gid}.bind.bin", "scaleRefs": geo["scale_refs"],
        })
        if delete_name:
            d["deleteVerts"] = delete_name
        d.update({"nativeMeasures": native, "defaultEase": ease, "layer": tpl["layer"], "baseColor": base_hex})
        entries.append(d)
        details[gid] = {"geo": geo, "native": native, "ease": ease, "notes": notes, "asset": asset, "def": d}
        size = sum((out_dir / n).stat().st_size for n in (d["mesh"], d["binding"], *([delete_name] if delete_name else [])))
        log(f"[garments] {gid:10s} {len(geo['pos']):6d} verts {len(geo['tris']):6d} tris  {size / 1024:8.1f} KiB  "
            f"native {native}  ease {ease}")
    writers.write_json(out_dir / "index.json", {"version": 1, "garments": entries})
    return details


def _diffuse(mat: dict[str, str]) -> list[float]:
    try:
        return [float(x) for x in mat.get("diffuseColor", "1 1 1").split()[:3]]
    except ValueError:
        return [1.0, 1.0, 1.0]


def garment_files(out_dir: Path) -> list[str]:
    """Relative names of every file build_garments writes (for hashing / --check)."""
    idx = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    names = ["index.json"]
    for g in idx["garments"]:
        names += [g["mesh"], g["binding"], *([g["deleteVerts"]] if g.get("deleteVerts") else [])]
    return names


# ---------------------------------------------------------------------------------------------------------------
# debug renders (front + side orthographic, z-buffered; not shipped)
# ---------------------------------------------------------------------------------------------------------------


def _raster(img: np.ndarray, zbuf: np.ndarray, pts: np.ndarray, depth: np.ndarray, tris: np.ndarray,
            shade: np.ndarray, color: np.ndarray) -> None:
    """Z-buffered flat triangle fill. pts (n, 2) pixel coords, depth (n,) larger = nearer, shade (T,) 0..1."""
    h, w, _ = img.shape
    for t, tri in enumerate(tris):
        p = pts[tri]
        x0, x1 = int(max(np.floor(p[:, 0].min()), 0)), int(min(np.ceil(p[:, 0].max()), w - 1))
        y0, y1 = int(max(np.floor(p[:, 1].min()), 0)), int(min(np.ceil(p[:, 1].max()), h - 1))
        if x1 < x0 or y1 < y0:
            continue
        den = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(den) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        l1 = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / den
        l2 = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / den
        l3 = 1 - l1 - l2
        m = (l1 >= -0.01) & (l2 >= -0.01) & (l3 >= -0.01)
        if not m.any():
            continue
        z = l1 * depth[tri[0]] + l2 * depth[tri[1]] + l3 * depth[tri[2]]
        sub = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        upd = m & (z > sub)
        sub[upd] = z[upd]
        img[y0 : y1 + 1, x0 : x1 + 1][upd] = (color * shade[t]).clip(0, 255).astype(np.uint8)


def render_view(body_pos, body_tris, layers, axis: str, size: int = 900) -> np.ndarray:
    """axis 'front' (look along -Z, x right) or 'side' (look from +X, z to the left). layers: [(pos, tris, rgb)]."""
    allp = np.concatenate([body_pos] + [p for p, _, _ in layers])
    ymin, ymax = float(body_pos[:, 1].min()), float(body_pos[:, 1].max())
    scale = (size - 40) / (ymax - ymin)
    if axis == "front":
        u, dep, nrm_dep = allp[:, 0], 2, 2
    else:
        u, dep, nrm_dep = allp[:, 2], 0, 0
    umin, umax = float(u.min()), float(u.max())
    w = int((umax - umin) * scale) + 40
    img = np.full((size, w, 3), 245, dtype=np.uint8)
    zbuf = np.full((size, w), -1e9)

    def proj(pos):
        uu = pos[:, 0] if axis == "front" else -pos[:, 2]
        x = (uu - (umin if axis == "front" else -umax)) * scale + 20
        y = (ymax - pos[:, 1]) * scale + 20
        return np.stack([x, y], axis=1), pos[:, dep]

    def shade(pos, tris):
        n = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return 0.55 + 0.45 * np.abs(n[:, nrm_dep])

    for pos, tris, col in [(body_pos, body_tris, (205, 160, 135))] + layers:
        pts, depth = proj(pos)
        _raster(img, zbuf, pts, depth, tris, shade(pos, tris), np.array(col, dtype=np.float64))
    return img


def write_debug_images(mesh: mh_obj.BaseMesh, neutral: np.ndarray, garments_dir: Path, debug_dir: Path) -> None:
    """`.cache/debug/garment_<id>.png`: front | side silhouettes of the neutral body with the garment (blue). Where the
    body colour (skin) shows through a garment area, the body pokes through it."""
    import pygltflib

    import debug_png

    R = mesh.render_count
    body_pos, body_tris = neutral[:R], mesh.tris
    for g in json.loads((garments_dir / "index.json").read_text(encoding="utf-8"))["garments"]:
        gl = pygltflib.GLTF2().load_binary(str(garments_dir / g["mesh"]))
        blob = gl.binary_blob()
        prim = gl.meshes[0].primitives[0]

        def acc(i, dt, n):
            a = gl.accessors[i]
            v = gl.bufferViews[a.bufferView]
            return np.frombuffer(blob, dtype=dt, count=a.count * n, offset=v.byteOffset)

        pos = acc(prim.attributes.POSITION, "<f4", 3).reshape(-1, 3).astype(np.float64)
        ia = gl.accessors[prim.indices]
        tris = acc(prim.indices, "<u2" if ia.componentType == 5123 else "<u4", 1).reshape(-1, 3).astype(np.int64)
        col = (70, 110, 200) if g["category"] != "shoes" else (60, 60, 60)
        btris = body_tris
        if g.get("deleteVerts"):  # hide the body triangles under the garment, as the runtime does
            gone = np.zeros(R, dtype=bool)
            gone[np.frombuffer((garments_dir / g["deleteVerts"]).read_bytes(), dtype="<u4")] = True
            btris = body_tris[~gone[body_tris].any(axis=1)]
        front = render_view(body_pos, btris, [(pos, tris, col)], "front")
        side = render_view(body_pos, btris, [(pos, tris, col)], "side")
        h = min(front.shape[0], side.shape[0])
        debug_png.write_png(debug_dir / f"garment_{g['id']}.png", np.concatenate([front[:h], side[:h]], axis=1))
