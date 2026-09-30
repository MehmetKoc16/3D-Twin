"""Body parts (eyes, eyebrows, eyelashes, hair): licences, MHCLO parse, binding files, glb validity, textures, iris,
eyes inside the sockets on extreme bodies, hair clearance to the scalp, determinism, live smoke check."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pygltflib
import pytest
from PIL import Image

import fetch
import mh_obj
import mhclo
import parts
import parts_tex
import targets
from config import DEFAULT_OUT_DIR, DEFAULT_PARTS_DIR, DM_TO_M, PART_ASSETS
from mh_morph import MhMorpher
from test_garments import apply_binding, glb_arrays, load_binding
from test_gltf import HERE, _validator_prefix

BODY_VERTS = 13380
PIDS = list(parts.PARTS)
HELPER_PIDS = ["eyes-default", "eyelashes-default", "hair-long", "hair-ponytail"]
HAIR = [p for p in PIDS if parts.PARTS[p]["category"] == "hair"]


@pytest.fixture(scope="session")
def pdir(out_a) -> Path:
    return out_a / "parts"


@pytest.fixture(scope="session")
def pindex(pdir) -> dict:
    return json.loads((pdir / "index.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pdefs(pindex) -> dict[str, dict]:
    return {p["id"]: p for p in pindex["parts"]}


@pytest.fixture(scope="session")
def first_copy(mesh):
    return mesh.mh_to_render_ids[mesh.mh_to_render_start[:BODY_VERTS]]


@pytest.fixture(scope="session")
def ctx(mesh, manifest):
    """Sample bodies in the full MakeHuman vertex space (helper geometry included), from the raw targets."""
    catalog = targets.build_catalog()
    packer = targets.TargetPacker(mesh, np.zeros((0, 8), dtype=np.int64))
    morph = MhMorpher(mesh, packer, catalog)
    return parts.make_context(mesh, morph, manifest["provenance"]["groundOffsetY"])


@pytest.fixture(scope="session")
def assets():
    return {pid: parts.load_asset(pid) for pid in PIDS}


def read_texture(pdir: Path, pdef: dict) -> Image.Image:
    g = pygltflib.GLTF2().load_binary(str(pdir / pdef["mesh"]))
    view = g.bufferViews[g.images[0].bufferView]
    return Image.open(io.BytesIO(g.binary_blob()[view.byteOffset: view.byteOffset + view.byteLength]))


# ---------------------------------------------------------------------------------------------------------------
# catalogue, licences, upstream files
# ---------------------------------------------------------------------------------------------------------------


def test_index_structure(pindex, pdefs):
    assert pindex["version"] == 1 and set(pdefs) == set(PIDS)
    cats = {c: [p for p in pdefs.values() if p["category"] == c] for c in ("eyes", "eyebrows", "eyelashes", "hair")}
    assert len(cats["eyes"]) >= 1 and len(cats["eyebrows"]) >= 3 and len(cats["eyelashes"]) >= 1 and 4 <= len(cats["hair"]) <= 6
    assert set(pindex["defaults"]) == {"eyes", "eyebrows", "eyelashes"}, "no default hair"
    for cat, pid in pindex["defaults"].items():
        assert pdefs[pid]["category"] == cat
    for d in pdefs.values():
        assert d["label"]["tr"] and d["label"]["en"]
        assert d["license"] in ("CC0-1.0", "CC-BY-4.0") and d["attribution"].strip()
        assert d["mesh"] == f"{d['id']}.glb" and d["binding"] == f"{d['id']}.bind.bin"
        assert set(d["scaleRefs"]) == {"x", "y", "z"} and all(len(v) == 3 and v[2] > 0 for v in d["scaleRefs"].values())
        m = d["material"]
        assert m["alphaMode"] in ("OPAQUE", "MASK") and isinstance(m["doubleSided"], bool) and isinstance(m["tintable"], bool)
        if m["alphaMode"] == "MASK":
            assert 0.2 <= m["alphaCutoff"] <= 0.7
        else:
            assert "alphaCutoff" not in m
        assert ("irisUv" in d) == (d["category"] == "eyes")
    for d in cats["hair"]:
        assert d["material"]["tintable"] and d["material"]["doubleSided"]
    assert sum(d["material"]["alphaMode"] == "MASK" for d in cats["hair"]) >= len(cats["hair"]) - 1, "alpha-masked hair cards"
    for d in cats["eyebrows"] + cats["eyelashes"]:
        assert d["material"]["alphaMode"] == "MASK" and d["material"]["doubleSided"]
    assert all(d["material"]["tintable"] for d in cats["eyebrows"]) and not cats["eyelashes"][0]["material"]["tintable"]
    assert cats["eyes"][0]["material"]["alphaMode"] == "OPAQUE" and not cats["eyes"][0]["material"]["doubleSided"]


def test_mhclo_parser_header_key_inside_verts_block():
    """Some MakeHuman files put `material x` between `verts 0` and the data rows (bob01, long01, ...)."""
    text = ("# license CC0\nname t\nx_scale 1 2 1.5\ny_scale 1 2 1.5\nz_scale 1 2 1.5\nverts 0\nmaterial t.mhmat\n"
            " 1 2 3 0.2 0.3 0.5 0.1 0.2 0.3\n 7\ndelete_verts\n0 - 3\n")
    cl = mhclo.parse_mhclo(text)
    assert cl.count == 2 and cl.header["material"] == "t.mhmat" and cl.delete_verts.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("pid", PIDS)
def test_mhclo_parse(pid, assets, mesh):
    a = assets[pid]
    cl, obj = a["mhclo"], a["obj"]
    assert cl.count == len(obj.verts) > 0
    assert cl.header["basemesh"] == "hm08"
    assert cl.indices.min() >= 0 and cl.indices.max() < mesh.verts.shape[0]
    assert np.abs(cl.weights.sum(axis=1) - 1.0).max() < 2e-5
    assert cl.weights.min() > -3.5  # MHCLO weights may extrapolate; the pipeline clamps them (offset absorbs it)
    assert set(cl.scale) == {"x", "y", "z"}
    for a_, b_, d in cl.scale.values():
        assert 0 <= a_ < BODY_VERTS and 0 <= b_ < BODY_VERTS and d > 0
    helper = bool((cl.indices >= BODY_VERTS).any())
    assert helper == (pid in HELPER_PIDS)


@pytest.mark.parametrize("pid", PIDS)
def test_licence_from_asset_metadata(pid, assets, pdefs):
    a = assets[pid]  # load_asset already cross-checks the .mhclo licence line against the pack json
    d = pdefs[pid]
    assert d["license"] in ("CC0-1.0", "CC-BY-4.0") and d["license"] == a["license"]
    assert a["asset"]["pack"].endswith("cc0" if d["license"] == "CC0-1.0" else "ccby")
    assert a["pack_entry"]["license"].upper().replace("-", "").startswith("CC0" if d["license"] == "CC0-1.0" else "CCBY")
    if d["license"] == "CC-BY-4.0":
        assert a["pack_entry"]["author"] in d["attribution"]
    assert "sha256:" + a["asset"]["files"][next(m for m in a["asset"]["files"] if m.endswith(a["clo_name"]))] in d["source"]
    for n, h in fetch._part_names(pid).items():
        assert hashlib.sha256((fetch.part_dir(pid) / n).read_bytes()).hexdigest() == h


def test_credits_cover_every_part(pdefs, assets):
    credits = (Path(__file__).resolve().parents[3] / "CREDITS.md").read_text(encoding="utf-8")
    assert "Body parts" in credits
    for pid, d in pdefs.items():
        a = assets[pid]
        author = a["pack_entry"]["author"]
        assert author in credits or author == "makehuman_system" and "MakeHuman system assets" in credits, pid
        stem = a["clo_name"][: -len(".mhclo")]
        assert stem in credits, f"{pid}: {stem} not credited"
        if d["license"] == "CC-BY-4.0":
            assert "CC BY" in credits


# ---------------------------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", PIDS)
def test_binding_file(pid, pdir, pdefs, mesh, first_copy, manifest, assets):
    idx, w, off = load_binding(pdir / pdefs[pid]["binding"])
    g, arr = glb_arrays(pdir / pdefs[pid]["mesh"])
    R = manifest["renderVertexCount"]
    assert len(idx) == len(arr["pos"]), "one 36-byte record per glb vertex"
    assert idx.max() < R and w.min() >= 0 and np.abs(w.sum(axis=1) - 1).max() < 1e-6, "runtime rejects negative weights"
    mh = mesh.render_mh[idx]
    assert mh.max() < BODY_VERTS, "bound to body vertices only (helper geometry is not in the runtime index space)"
    assert np.array_equal(idx, first_copy[mh]), "canonical (first) render copy of each MakeHuman vertex"
    assert np.isfinite(off).all() and np.abs(off).max() < 0.45
    cl = assets[pid]["mhclo"]
    for k in "xyz":
        a_, b_, d = pdefs[pid]["scaleRefs"][k]
        assert (a_, b_) == tuple(int(first_copy[v]) for v in cl.scale[k][:2])
        assert abs(d - cl.scale[k][2] * DM_TO_M) < 1e-6


@pytest.mark.parametrize("pid", PIDS)
def test_delete_verts(pid, pdir, pdefs, mesh, assets):
    d = pdefs[pid]
    up = assets[pid]["mhclo"].delete_verts
    if "deleteVerts" not in d:
        assert pid != "eyes-default" and not (up[up < BODY_VERTS]).size
        return
    got = np.frombuffer((pdir / d["deleteVerts"]).read_bytes(), dtype="<u4").astype(np.int64)
    assert (np.diff(got) > 0).all() and got.max() < mesh.render_count
    body = np.unique(mesh.render_mh[got])
    if pid == "eyes-default":  # socket cavity interior of both eyes: every render copy, nothing else, near the eyes
        assert 500 < len(body) < 900 and len(got) >= len(body)
        pts = mesh.verts[body] * DM_TO_M
        assert 0.040 < np.abs(pts[:, 0]).max() < 0.050 and (np.abs(pts[:, 0]) > 0.010).all()
        assert np.array_equal(got, np.sort(np.concatenate([mesh.render_ids_of(int(v)) for v in body])))
    else:
        assert np.array_equal(body, up[up < BODY_VERTS])


@pytest.mark.parametrize("pid", PIDS)
def test_glb_positions_are_runtime_formula_on_neutral(pid, pdir, pdefs, morphset):
    idx, w, off = load_binding(pdir / pdefs[pid]["binding"])
    _, arr = glb_arrays(pdir / pdefs[pid]["mesh"])
    rec = apply_binding(idx, w, off, pdefs[pid]["scaleRefs"], morphset.positions())
    assert np.abs(rec - arr["pos"]).max() < 2e-6


@pytest.mark.parametrize("pid", PIDS)
def test_glb_structure(pid, pdir, pdefs):
    d = pdefs[pid]
    g, arr = glb_arrays(pdir / d["mesh"])
    assert len(g.meshes) == 1 and len(g.meshes[0].primitives) == 1 and len(g.materials) == 1 and not g.skins
    at = g.meshes[0].primitives[0].attributes
    assert at.JOINTS_0 is None and at.WEIGHTS_0 is None and at.NORMAL is not None and at.TEXCOORD_0 is not None
    n = len(arr["pos"])
    assert arr["idx"].max() == n - 1 and len(set(arr["idx"].tolist())) == n, "no orphan vertices"
    assert len(arr["idx"]) % 3 == 0 and np.abs(np.linalg.norm(arr["nor"], axis=1) - 1).max() < 1e-5
    assert arr["uv"].min() >= -1e-6 and arr["uv"].max() <= 1.0 + 1e-6
    mat = g.materials[0]
    m = d["material"]
    assert mat.alphaMode == m["alphaMode"] and bool(mat.doubleSided) == m["doubleSided"]
    if m["alphaMode"] == "MASK":
        assert abs(mat.alphaCutoff - m["alphaCutoff"]) < 1e-6
    assert g.images and g.images[0].bufferView is not None
    img = read_texture(pdir, d)
    limit = {"eyes": 512, "eyebrows": 512, "eyelashes": 512, "hair": 1024}[d["category"]]
    assert max(img.size) <= limit
    # meters, neutral body frame (feet on y = 0, head ~1.5-1.75 m)
    y = arr["pos"][:, 1]
    if d["category"] in ("eyes", "eyebrows", "eyelashes"):
        assert 1.5 < y.min() and y.max() < 1.66
    else:
        assert 1.45 < y.max() < 1.85 and y.min() > 0.9


@pytest.mark.parametrize("pid", PIDS)
def test_texture_and_alpha_strategy(pid, pdir, pdefs):
    d = pdefs[pid]
    img = read_texture(pdir, d)
    m = d["material"]
    if d["category"] == "eyes":
        assert img.format == "JPEG" and img.mode == "RGB" and img.size == (512, 512) and m["alphaMode"] == "OPAQUE"
        return
    assert img.format == "PNG"
    if m["alphaMode"] == "OPAQUE":
        assert img.mode == "L", "a fully opaque texture is a grey PNG without alpha"
        return
    assert img.mode == "LA", "grey + alpha PNG"
    la = np.asarray(img)
    alpha = la[..., 1].astype(np.float64) / 255.0
    assert alpha.min() < 0.05 and alpha.max() > 0.6  # (a hair-thin brow texture peaks at ~0.7)
    # the cutoff keeps the covered area equal to the soft coverage (parts_tex.coverage_cutoff), within the clamp
    covered = float((alpha >= m["alphaCutoff"]).mean())
    soft = float(alpha.mean())
    assert abs(covered - soft) <= 0.35 * soft + 0.02, (covered, soft)
    g = pygltflib.GLTF2().load_binary(str(pdir / d["mesh"]))
    factor = g.materials[0].pbrMetallicRoughness.baseColorFactor
    assert all(0.0 <= c <= 1.0 for c in factor) and factor[3] == 1.0
    if m["tintable"]:
        # neutral map + default tint: the strand highlights reach ~0.9 (or a flat 0.9), the tint (linear factor) is the colour
        solid = alpha > 0.5
        assert np.quantile(la[..., 0][solid], 0.95) >= 0.85 * 255
    else:
        assert factor == [1.0, 1.0, 1.0, 1.0]


def test_texture_prep_helpers():
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[16:48, 16:48] = (200, 100, 50, 255)
    rgba[:16] = (255, 0, 255, 0)  # transparent texels with a junk colour
    bled = parts_tex.bleed_colors(rgba)
    assert (bled[..., 3] == rgba[..., 3]).all() and tuple(bled[0, 32, :3]) == (200, 100, 50), "colour bled from the opaque part"
    small = parts_tex.resize_rgba(np.repeat(np.repeat(rgba, 4, 0), 4, 1), 64)
    assert small.shape == (64, 64, 4) and small[..., 3].max() == 255
    la, tint = parts_tex.neutralise(bled)
    assert la.shape == (64, 64, 2) and tint.max() <= 1.0
    lin = parts_tex.srgb_to_linear(np.array([200, 100, 50]) / 255.0)
    rec = tint * parts_tex.srgb_to_linear(la[32, 32, 0] / 255.0)
    assert np.abs(rec - lin).max() < 0.02, "tint x neutral map reproduces the source colour"
    assert 0.3 <= parts_tex.coverage_cutoff(rgba[..., 3]) <= 0.6


# ---------------------------------------------------------------------------------------------------------------
# eyes: iris, sockets, tracking of the MakeHuman helper geometry
# ---------------------------------------------------------------------------------------------------------------


def test_iris_uv(pdir, pdefs):
    d = pdefs["eyes-default"]
    (cu, cv), r = d["irisUv"]["center"], d["irisUv"]["radius"]
    assert 0.0 < cu - r and cu + r < 1.0 and 0.0 < cv - r and cv + r < 1.0, "the iris circle lies inside the texture"
    assert 0.05 < r < 0.3
    lum = np.asarray(read_texture(pdir, d).convert("L")).astype(np.float64)
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx + 0.5) / w - cu, (yy + 0.5) / h - cv)
    pupil, iris, sclera = lum[rr < 0.3 * r].mean(), lum[(rr > 0.6 * r) & (rr < 0.9 * r)].mean(), lum[(rr > 1.3 * r) & (rr < 1.6 * r)].mean()
    assert pupil < 40 < iris < sclera, "dark pupil, mid iris ring, bright sclera"
    # the mesh covers the iris: many eye vertices have their texture coordinate inside the iris circle
    g, arr = glb_arrays(pdir / d["mesh"])
    inside = np.hypot(arr["uv"][:, 0] - cu, arr["uv"][:, 1] - cv) < r
    assert inside.sum() > 60
    assert (np.hypot(arr["uv"][:, 0] - cu, arr["uv"][:, 1] - cv) < 0.02).any()


def test_eye_mesh_is_opaque_inner_shell_of_both_eyes(pdir, pdefs, assets):
    _, arr = glb_arrays(pdir / pdefs["eyes-default"]["mesh"])
    pos = arr["pos"]
    left, right = pos[pos[:, 0] > 0], pos[pos[:, 0] < 0]
    assert len(left) == len(right) > 200
    assert np.allclose(np.sort(left[:, 1]), np.sort(right[:, 1]), atol=1e-6), "mirrored eyes"
    c, r = parts._sphere_fit(left)
    assert 0.013 < r < 0.017 and 0.025 < c[0] < 0.033 and 1.5 < c[1] < 1.57  # ~15 mm radius, ~29 mm off the mid-plane
    up = assets["eyes-default"]["obj"]
    assert len(up.verts) == 1064, "the upstream mesh has a transparent outer shell that is dropped"


BODIES = {
    "neutral": ({}, {}),
    "female": ({"gender": 0.0}, {}),
    "male": ({"gender": 1.0}, {}),
    "short": ({"height": 0.0}, {}),
    "tall": ({"height": 1.0}, {}),
    "heavy-muscle": ({"gender": 1.0, "weight": 1.0, "muscle": 1.0}, {}),
    "light-short": ({"gender": 0.0, "height": 0.0, "weight": 0.0}, {}),
    "head-wide": ({}, {"head/head-scale-horiz": 1.0}),
    "head-narrow": ({}, {"head/head-scale-horiz": -1.0}),
    "eye-big": ({}, {"eyes/eye-scale": 1.0}),
    "eye-small": ({}, {"eyes/eye-scale": -1.0}),
    "face-mix": ({"gender": 0.3}, {"head/head-fat": 0.8, "cheek/cheek-volume": -0.7, "chin/chin-width": 0.6}),
}


@pytest.fixture(scope="session")
def eye_setup(pdir, pdefs, mesh, first_copy, ctx, assets):
    """Shipped eye binding + the eyelid opening (rim of the socket cavity) found on the neutral body."""
    d = pdefs["eyes-default"]
    idx, w, off = load_binding(pdir / d["binding"])
    _, arr = glb_arrays(pdir / d["mesh"])
    left = arr["pos"][:, 0] > 0
    centres = [parts._sphere_fit(arr["pos"][sel])[0] for sel in (left, ~left)]
    cav = [parts.eye_cavity(ctx, c) for c in centres]
    return {"idx": idx, "w": w, "off": off, "refs": d["scaleRefs"], "left": left, "rim": [c["rim"] for c in cav], "cav": cav}


def eye_metrics(eye_setup, P_combined, first_copy):
    """{eye: socket metrics} of the shipped binding on a solved body (combined index space)."""
    E = apply_binding(eye_setup["idx"], eye_setup["w"], eye_setup["off"], eye_setup["refs"], P_combined)
    out = []
    for sel, rim in ((eye_setup["left"], eye_setup["rim"][0]), (~eye_setup["left"], eye_setup["rim"][1])):
        out.append(parts.eye_socket_metrics(E[sel], P_combined[first_copy[rim]]))
    return out


def test_eye_socket_topology(eye_setup, ctx):
    for rim, cav in zip(eye_setup["rim"], eye_setup["cav"]):
        assert 25 <= len(rim) <= 60, "an eyelid opening ring of a few dozen vertices"
        assert 250 <= len(cav["quads"]) <= 450 and len(cav["interior"]) > 250
        assert set(rim.tolist()).isdisjoint(cav["interior"].tolist())
    # the cavity of a MakeHuman body faces INTO the head (single-sided skin culls it), the eyeball proxy sits inside
    c = parts._sphere_fit(ctx.Hs[0][eye_setup["cav"][0]["vertices"]])[0]
    assert 0.025 < c[0] < 0.033


@pytest.mark.parametrize("name", list(BODIES))
def test_eyes_sit_in_the_sockets_on_extreme_bodies(name, eye_setup, morphset, first_copy):
    """The eyeball surface vs the eyelid opening: no poke-through beyond the lid thickness, no gap, on every body."""
    macros, mods = BODIES[name]
    P = morphset.positions(macros, mods)
    for m in eye_metrics(eye_setup, P, first_copy):
        assert m["gap_min"] > -0.0020, f"{name}: eyeball surface pokes {-m['gap_min'] * 1000:.2f} mm through the lid margin"
        assert m["gap_max"] < 0.0050, f"{name}: gap of {m['gap_max'] * 1000:.2f} mm between the eyeball and the lid margin"
        assert 0.0 < m["protrusion"] < 0.0085, f"{name}: front pole {m['protrusion'] * 1000:.2f} mm from the lid plane"
        assert 0.011 < m["radius"] < 0.022


def test_eyes_socket_metrics_match_the_makehuman_helper_eyes(eye_setup, ctx, first_copy, assets, morphset):
    """The re-bound eyeball has the same lid-margin gap as the helper eyeball MakeHuman itself ships (within 1 mm)."""
    a = assets["eyes-default"]
    cl, _, extra = parts.part_mesh("eyes-default", a, parts_tex.load_rgba(a["texture"])[..., 3])
    for name in ("female", "male", "short", "tall", "head-wide", "eye-big", "face-mix"):
        macros, mods = BODIES[name]
        truth = parts.truth_positions(cl, ctx.body(macros, mods))
        P = morphset.positions(macros, mods)
        ours = eye_metrics(eye_setup, P, first_copy)[0]
        ref = parts.eye_socket_metrics(truth[extra["left"]], P[first_copy[eye_setup["rim"][0]]])
        assert abs(ours["gap_min"] - ref["gap_min"]) < 0.0010 and abs(ours["gap_max"] - ref["gap_max"]) < 0.0010, name


# ---------------------------------------------------------------------------------------------------------------
# tracking of the helper geometry / clearance
# ---------------------------------------------------------------------------------------------------------------

TRACK_MM = {  # (mean, max) allowed distance to the MakeHuman helper truth on the checked bodies, mm
    "eyes-default": (2.5, 6.5),
    "eyelashes-default": (0.6, 3.0),
    "hair-long": (12.0, 75.0),
    "hair-ponytail": (6.0, 45.0),
}


@pytest.mark.parametrize("pid", HELPER_PIDS)
def test_rebound_parts_track_the_helper_geometry(pid, pdir, pdefs, ctx, assets, morphset, first_copy):
    idx, w, off = load_binding(pdir / pdefs[pid]["binding"])
    a = assets[pid]
    cl, obj, extra = parts.part_mesh(pid, a, parts_tex.load_rgba(a["texture"])[..., 3] if pid == "eyes-default" else None)
    render_v, _, _ = parts.render_arrays(obj)
    mean_mm, max_mm = TRACK_MM[pid]
    means, maxes = [], []
    for name in ("neutral", "female", "male", "short", "tall", "head-wide", "eye-big", "face-mix"):
        macros, mods = BODIES[name]
        truth = parts.truth_positions(cl, ctx.body(macros, mods))[render_v]
        pred = apply_binding(idx, w, off, pdefs[pid]["scaleRefs"], morphset.positions(macros, mods))
        err = np.linalg.norm(pred - truth, axis=1) * 1000.0
        means.append(err.mean())
        maxes.append(err.max())
    assert max(means) < mean_mm and max(maxes) < max_mm, (pid, np.round(means, 2), np.round(maxes, 1))
    assert means[0] < 0.01, "exact on the neutral body"


def body_clearance(points: np.ndarray, body_pos: np.ndarray, mesh, near: float = 0.03) -> np.ndarray:
    """Signed distance (meters, + = outside the skin) of points to the nearest body vertex plane, for points within `near`
    of the body; NaN elsewhere. Normals: area weighted per MakeHuman vertex."""
    normals = mh_obj.vertex_normals(body_pos, mesh.tris, mesh.render_mh)
    sel = np.nonzero((body_pos[:, 1] > points[:, 1].min() - 0.05) & (body_pos[:, 1] < points[:, 1].max() + 0.05))[0]
    out = np.full(len(points), np.nan)
    for lo in range(0, len(points), 512):
        p = points[lo: lo + 512]
        d2 = ((p[:, None, :] - body_pos[sel][None]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        v = sel[j]
        dist = np.sqrt(d2[np.arange(len(p)), j])
        signed = ((p - body_pos[v]) * normals[v]).sum(axis=1)
        out[lo: lo + 512] = np.where(dist < near, signed, np.nan)
    return out


@pytest.mark.parametrize("pid", HAIR)
@pytest.mark.parametrize("name", ["neutral", "female", "heavy-muscle", "tall"])
def test_hair_does_not_sink_into_the_scalp(pid, name, pdir, pdefs, mesh, morphset):
    idx, w, off = load_binding(pdir / pdefs[pid]["binding"])
    macros, mods = BODIES[name]
    P = morphset.positions(macros, mods)
    hair = apply_binding(idx, w, off, pdefs[pid]["scaleRefs"], P)
    clr = body_clearance(hair, P[: mesh.render_count], mesh)
    near = clr[~np.isnan(clr)]
    assert len(near) > 300, "the hair sits on the body (scalp, neck)"
    # a few hair roots are tucked into the skull on purpose (invisible behind the skin); the bulk sits outside the skin
    deep4, deep10 = float((near < -0.004).mean()), float((near < -0.010).mean())
    assert deep4 < 0.08 and deep10 < 0.03 and np.percentile(near, 1) > -0.025, (pid, name, round(deep4, 3), round(deep10, 3))
    assert np.percentile(near, 25) > 0.001 and np.percentile(near, 50) > 0.003, "most hair vertices are outside the skin"


@pytest.mark.parametrize("pid", ["eyebrows-default", "eyebrows-thick", "eyebrows-thin"])
@pytest.mark.parametrize("name", ["neutral", "female", "heavy-muscle", "eye-big"])
def test_eyebrows_sit_on_the_skin(pid, name, pdir, pdefs, mesh, morphset):
    idx, w, off = load_binding(pdir / pdefs[pid]["binding"])
    macros, mods = BODIES[name]
    P = morphset.positions(macros, mods)
    brow = apply_binding(idx, w, off, pdefs[pid]["scaleRefs"], P)
    clr = body_clearance(brow, P[: mesh.render_count], mesh, near=0.02)
    assert np.isfinite(clr).all() and np.percentile(clr, 5) > -0.0035 and np.percentile(clr, 95) < 0.015, (pid, name)


# ---------------------------------------------------------------------------------------------------------------
# validation, determinism, live smoke check
# ---------------------------------------------------------------------------------------------------------------


def test_khronos_gltf_validator_parts(pdir, pdefs):
    node = shutil.which("node")
    prefix = _validator_prefix() if node else None
    if not node or prefix is None:
        pytest.skip("node/npm or gltf-validator not available; structural checks still ran")
    for d in pdefs.values():
        res = subprocess.run(
            [node, str(HERE / "validate_glb.cjs"), str(pdir / d["mesh"]), str(prefix)],
            capture_output=True, text=True, timeout=120,
        )
        issues = json.loads(res.stdout.strip().splitlines()[-1])
        assert issues["numErrors"] == 0 and issues["numWarnings"] == 0, (d["id"], issues["messages"])


def test_parts_deterministic(out_a, out_b):
    names = parts.part_files(out_a / "parts")
    assert names == parts.part_files(out_b / "parts")
    for n in names:
        assert (out_a / "parts" / n).read_bytes() == (out_b / "parts" / n).read_bytes(), n


def test_shipped_assets_smoke_live(morphset, first_copy):
    """Real-data smoke check on the committed assets (skipped when they are not built): recompute the eyes from the
    shipped binding + the shipped body morph model with the runtime formula and compare to the shipped glb."""
    idx_file = DEFAULT_PARTS_DIR / "index.json"
    if not idx_file.exists() or not (DEFAULT_OUT_DIR / "base.glb").exists() or not (DEFAULT_PARTS_DIR / "eyes-default.bind.bin").exists():
        pytest.skip("shipped part assets not built (run `npm run assets:build`)")
    if b"git-lfs" in (DEFAULT_PARTS_DIR / "eyes-default.bind.bin").read_bytes()[:64]:
        pytest.skip("shipped part assets are Git LFS pointers (run `git lfs pull`)")
    index = json.loads(idx_file.read_text(encoding="utf-8"))
    d = next(p for p in index["parts"] if p["id"] == "eyes-default")
    idx, w, off = load_binding(DEFAULT_PARTS_DIR / d["binding"])
    _, arr = glb_arrays(DEFAULT_PARTS_DIR / d["mesh"])
    rec = apply_binding(idx, w, off, d["scaleRefs"], morphset.positions())
    assert np.abs(rec - arr["pos"]).max() < 1e-5, "shipped eye glb == runtime formula on the neutral body"
    rec2 = apply_binding(idx, w, off, d["scaleRefs"], morphset.positions({"gender": 1.0, "height": 1.0}))
    assert np.isfinite(rec2).all() and np.abs(rec2 - rec).max() > 0.005
