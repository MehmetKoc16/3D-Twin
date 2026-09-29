"""Garment templates: MHCLO semantics, binding/delete files, glb validity, licences, determinism, live smoke check."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pygltflib
import pytest

import fetch
import garments
import mhclo
from config import DEFAULT_GARMENTS_DIR, DEFAULT_OUT_DIR, DM_TO_M, GARMENT_ASSETS
from test_gltf import HERE, _validator_prefix
from test_rig_skin import read_accessor

BODY_VERTS = 13380
GIDS = list(garments.TEMPLATES)


@pytest.fixture(scope="session")
def gdir(out_a) -> Path:
    return out_a / "garments"


@pytest.fixture(scope="session")
def index(gdir) -> dict:
    return json.loads((gdir / "index.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def defs(index) -> dict[str, dict]:
    return {g["id"]: g for g in index["garments"]}


@pytest.fixture(scope="session")
def first_copy(mesh):
    return mesh.mh_to_render_ids[mesh.mh_to_render_start[:BODY_VERTS]]


def upstream(gid: str) -> dict:
    return garments.load_asset(gid)


def load_binding(path: Path):
    raw = path.read_bytes()
    assert len(raw) % 36 == 0
    rec = np.frombuffer(raw, dtype=[("i", "<u4", 3), ("w", "<f4", 3), ("o", "<f4", 3)])
    return rec["i"].astype(np.int64), rec["w"].astype(np.float64), rec["o"].astype(np.float64)


def glb_arrays(path: Path):
    g = pygltflib.GLTF2().load_binary(str(path))
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    at = prim.attributes
    return g, {
        "pos": read_accessor(g, blob, at.POSITION).astype(np.float64),
        "nor": read_accessor(g, blob, at.NORMAL).astype(np.float64),
        "uv": read_accessor(g, blob, at.TEXCOORD_0),
        "idx": read_accessor(g, blob, prim.indices).reshape(-1),
    }


def apply_binding(idx, w, off, refs, body: np.ndarray) -> np.ndarray:
    """The runtime formula (`bindGarment`): w.v[i] summed + offset * per-axis |v[a] - v[b]| / refM."""
    s = np.array([abs(body[refs[k][0], i] - body[refs[k][1], i]) / refs[k][2] for i, k in enumerate("xyz")])
    return (w[:, :, None] * body[idx]).sum(axis=1) + off * s


# ---------------------------------------------------------------------------------------------------------------


def test_index_structure(index, defs):
    assert index["version"] == 1 and set(defs) == set(GIDS)
    kinds = {d["kind"] for d in defs.values()}
    assert {"tshirt", "sweatshirt", "pants", "jeans", "sneakers", "shoes"} <= kinds
    # at least one short-sleeve top, one long-sleeve top, one pair of long pants, one pair of shoes
    assert defs["tshirt"]["nativeMeasures"]["sleeve"] < 30 < defs["sweatshirt"]["nativeMeasures"]["sleeve"]
    assert defs["pants"]["nativeMeasures"]["inseam"] > 60
    for d in defs.values():
        assert d["category"] in ("top", "bottom", "shoes") and d["layer"] >= 1
        assert d["label"]["tr"] and d["label"]["en"] and d["baseColor"].startswith("#") and len(d["baseColor"]) == 7
        assert d["mesh"] == f"{d['id']}.glb" and d["binding"] == f"{d['id']}.bind.bin"
        assert set(d["scaleRefs"]) == {"x", "y", "z"}
        assert all(len(v) == 3 and v[2] > 0 for v in d["scaleRefs"].values())
        assert "attach" not in d, "MakeHuman shoes are MHCLO proxies: they use `binding`"
        for m, v in {**d["nativeMeasures"], **d["defaultEase"]}.items():
            assert np.isfinite(v)
        for v in d["defaultEase"].values():
            assert abs(v * 2 - round(v * 2)) < 1e-9, "defaultEase is rounded to 0.5 cm"
        assert set(d["defaultEase"]) <= set(d["nativeMeasures"])
    for gid in ("tshirt", "sweatshirt"):
        assert {"chest", "length", "sleeve"} <= set(defs[gid]["nativeMeasures"])
    for gid in ("pants", "jeans"):
        assert {"waist", "hip", "length", "inseam"} <= set(defs[gid]["nativeMeasures"])
    for gid in ("sneakers", "shoes", "boots"):
        assert 20 < defs[gid]["nativeMeasures"]["footLength"] < 30


@pytest.mark.parametrize("gid", GIDS)
def test_mhclo_parse(gid, mesh):
    a = upstream(gid)
    cl, obj = a["mhclo"], a["obj"]
    assert cl.count == len(obj.verts) > 0
    assert cl.header["basemesh"] == "hm08"
    assert cl.indices.min() >= 0 and cl.indices.max() < BODY_VERTS, "garment binds only to body vertices"
    assert np.abs(cl.weights.sum(axis=1) - 1.0).max() < 2e-5
    assert cl.weights.min() > -2.0  # MHCLO weights may extrapolate; the pipeline clamps them (offset absorbs it)
    assert set(cl.scale) == {"x", "y", "z"}
    for a_, b_, d in cl.scale.values():
        assert 0 <= a_ < BODY_VERTS and 0 <= b_ < BODY_VERTS and d > 0
    assert cl.delete_verts.size == 0 or (cl.delete_verts.min() >= 0 and cl.delete_verts.max() < mesh.verts.shape[0])


def test_mhclo_parser_edge_cases():
    text = "# author x\n# license CC0\nname t\nx_scale 1 2 1.5\ny_scale 1 2 1.5\nz_scale 1 2 1.5\nverts 0\n 1 2 3 0.2 0.3 0.5 0.1 0.2 0.3\n 7\ndelete_verts\n0 - 3 9 11 - 12\n"
    cl = mhclo.parse_mhclo(text)
    assert cl.count == 2 and cl.indices[1].tolist() == [7, 7, 7] and cl.weights[1].tolist() == [1.0, 0.0, 0.0]
    assert cl.delete_verts.tolist() == [0, 1, 2, 3, 9, 11, 12]
    assert mhclo.classify_license("CC-BY 4.0") == "CC-BY-4.0" and mhclo.classify_license("CC-0") == "CC0-1.0"
    assert mhclo.classify_license("AGPL3 (see also http://www.makehuman.org/doc/node/external_tools_license.html)") is None
    assert mhclo.classify_license("CC-BY-NC") is None and mhclo.classify_license("CC-BY-SA 3.0") is None


@pytest.mark.parametrize("gid", GIDS)
def test_licence_from_asset_metadata(gid, defs):
    a = upstream(gid)  # load_asset already cross-checks the .mhclo licence line against the pack json
    d = defs[gid]
    assert d["license"] in ("CC0-1.0", "CC-BY-4.0") and d["license"] == a["license"]
    assert a["asset"]["pack"].endswith("cc0" if d["license"] == "CC0-1.0" else "ccby")
    if d["license"] == "CC-BY-4.0":
        assert d["attribution"].strip() and a["pack_entry"]["author"] in d["attribution"]
        if a["pack_entry"].get("original_author"):
            assert a["pack_entry"]["original_author"] in d["attribution"]
    assert "sha256:" + a["asset"]["files"][a["clo_name"]] in d["source"]
    for n, h in fetch._garment_names(gid).items():
        assert hashlib.sha256((fetch.garment_dir(gid) / n).read_bytes()).hexdigest() == h


def test_credits_list_every_ccby_garment(defs):
    credits = (Path(__file__).resolve().parents[3] / "CREDITS.md").read_text(encoding="utf-8")
    for d in defs.values():
        a = upstream(d["id"])
        assert a["pack_entry"]["author"] in credits
        if d["license"] == "CC-BY-4.0":
            assert a["clo_name"][: -len(".mhclo")] in credits


@pytest.mark.parametrize("gid", GIDS)
def test_binding_file(gid, gdir, defs, mesh, first_copy, manifest):
    idx, w, off = load_binding(gdir / defs[gid]["binding"])
    g, arr = glb_arrays(gdir / defs[gid]["mesh"])
    R = manifest["renderVertexCount"]
    assert len(idx) == len(arr["pos"]), "one 36-byte record per garment glb vertex"
    assert idx.max() < R and w.min() >= 0 and np.abs(w.sum(axis=1) - 1).max() < 1e-6
    mh = mesh.render_mh[idx]
    assert mh.max() < BODY_VERTS
    assert np.array_equal(idx, first_copy[mh]), "canonical (first) render copy of each MakeHuman vertex"
    assert np.abs(off).max() < 0.2 and np.isfinite(off).all()  # offsets in meters (< 20 cm)
    # scale refs are render ids of MakeHuman vertices, refM in meters (decimeters * 0.1)
    up = upstream(gid)["mhclo"]
    for k in "xyz":
        a_, b_, d = defs[gid]["scaleRefs"][k]
        assert (a_, b_) == tuple(int(first_copy[v]) for v in up.scale[k][:2])
        assert abs(d - up.scale[k][2] * DM_TO_M) < 1e-6


@pytest.mark.parametrize("gid", GIDS)
def test_delete_verts_cover_all_render_copies(gid, gdir, defs, mesh):
    up = upstream(gid)["mhclo"]
    body = up.delete_verts[up.delete_verts < BODY_VERTS]
    if body.size == 0:
        assert "deleteVerts" not in defs[gid]
        return
    got = np.frombuffer((gdir / defs[gid]["deleteVerts"]).read_bytes(), dtype="<u4").astype(np.int64)
    expected = np.sort(np.concatenate([mesh.render_ids_of(int(v)) for v in body]))
    assert np.array_equal(got, expected) and len(got) >= len(body)
    assert np.array_equal(np.unique(mesh.render_mh[got]), body)
    assert (np.diff(got) > 0).all()


# Mean error (mm) allowed between the reconstruction on the best-fitting authoring body and the upstream OBJ.
RECON_MEAN_MM = {"tshirt": 3.0, "sweatshirt": 4.0, "pants": 7.0, "jeans": 7.0, "sneakers": 25.0, "shoes": 2.0, "boots": 4.0}


@pytest.fixture(scope="session")
def gender_bodies(morphset, first_copy):
    return {g: morphset.positions({"gender": g})[first_copy] / DM_TO_M for g in np.round(np.arange(0, 1.01, 0.1), 1)}


@pytest.mark.parametrize("gid", GIDS)
def test_reconstruction_matches_upstream_obj(gid, gender_bodies):
    """Validates the MHCLO semantics against the upstream garment OBJ (decimeters, MakeHuman axes).

    MHCLO offsets are relative to the body the asset was *authored* on, which is not our neutral body: the male-authored
    items (t-shirt, sweater, pants, jeans, boots) match at gender 1.0, the female-authored cloth shoes at gender ~0.1,
    to 1-6 mm mean error (the residual is the authoring body's race / version). The sneakers' OBJ was evidently edited
    after export (no macro state reproduces it better than ~2 cm). A pure translation (ground offset) is removed.
    On the neutral body (gender 0.5) the same garments are ~15 mm off, which is why positions are recomputed per body."""
    a = upstream(gid)
    cl, obj = a["mhclo"], a["obj"]
    results = []
    for g, body in gender_bodies.items():
        s = mhclo.axis_scales(cl.scale, body)
        rec = mhclo.reconstruct(cl.indices, cl.weights, cl.offsets, s, body)
        err = rec - obj.verts
        err -= np.median(err, axis=0)
        results.append((float(np.linalg.norm(err, axis=1).mean() * 100.0), float(np.abs(s - 1).max()), g, body, s))
    best = min(results)
    assert best[0] < RECON_MEAN_MM[gid], (gid, [(round(r[0], 1), r[2]) for r in sorted(results)[:3]])
    if gid != "sneakers":
        assert best[1] < 0.03, "authoring reference distances agree with the best-fit body"
        # the offset term matters: dropping it (or using the wrong body) is clearly worse
        no_off = mhclo.reconstruct(cl.indices, cl.weights, cl.offsets * 0, best[4], best[3]) - obj.verts
        no_off -= np.median(no_off, axis=0)
        assert np.linalg.norm(no_off, axis=1).mean() * 100.0 > 1.5 * best[0]
        worst = max(results)
        assert worst[0] > 1.5 * best[0]


@pytest.mark.parametrize("gid", GIDS)
def test_glb_positions_are_runtime_formula_on_neutral(gid, gdir, defs, morphset):
    """Same formula as `bindGarment` on the neutral body reproduces the glb POSITION (float32 rounding only)."""
    idx, w, off = load_binding(gdir / defs[gid]["binding"])
    _, arr = glb_arrays(gdir / defs[gid]["mesh"])
    rec = apply_binding(idx, w, off, defs[gid]["scaleRefs"], morphset.positions())
    assert np.abs(rec - arr["pos"]).max() < 2e-6


@pytest.mark.parametrize("gid", GIDS)
def test_glb_structure(gid, gdir, defs):
    g, arr = glb_arrays(gdir / defs[gid]["mesh"])
    assert len(g.meshes) == 1 and len(g.meshes[0].primitives) == 1 and len(g.materials) == 1 and not g.skins
    at = g.meshes[0].primitives[0].attributes
    assert at.JOINTS_0 is None and at.WEIGHTS_0 is None and at.NORMAL is not None and at.TEXCOORD_0 is not None
    n = len(arr["pos"])
    assert arr["idx"].max() == n - 1 and len(set(arr["idx"].tolist())) == n, "no orphan vertices"
    assert len(arr["idx"]) % 3 == 0
    assert np.abs(np.linalg.norm(arr["nor"], axis=1) - 1).max() < 1e-5
    assert arr["uv"].min() > -1.0 and arr["uv"].max() < 2.0
    assert g.images and g.images[0].bufferView is not None and g.images[0].mimeType in ("image/jpeg", "image/png")
    from PIL import Image
    import io

    view = g.bufferViews[g.images[0].bufferView]
    img = Image.open(io.BytesIO(g.binary_blob()[view.byteOffset : view.byteOffset + view.byteLength]))
    assert max(img.size) <= 1024
    # meters, neutral body frame (feet on y = 0, body ~1.67 m tall)
    y = arr["pos"][:, 1]
    assert -0.02 < y.min() < 1.6 and 0.0 < y.max() < 1.75
    if defs[gid]["category"] == "shoes":
        assert -0.02 < y.min() < 0.03 and y.max() < 0.8
    if gid == "pants" or gid == "jeans":
        assert y.min() < 0.1 and 0.95 < y.max() < 1.15
    if gid in ("tshirt", "sweatshirt"):
        assert 1.25 < y.max() < 1.5


def test_khronos_gltf_validator_garments(gdir, defs):
    import shutil as sh

    node = sh.which("node")
    prefix = _validator_prefix() if node else None
    if not node or prefix is None:
        pytest.skip("node/npm or gltf-validator not available; structural checks still ran")
    for d in defs.values():
        res = subprocess.run(
            [node, str(HERE / "validate_glb.cjs"), str(gdir / d["mesh"]), str(prefix)],
            capture_output=True, text=True, timeout=120,
        )
        issues = json.loads(res.stdout.strip().splitlines()[-1])
        assert issues["numErrors"] == 0 and issues["numWarnings"] == 0, (d["id"], issues["messages"])


def test_native_measures_plausible(defs):
    """Garments sit on the neutral body with sane ease (sizes are tape lengths of the garment on that body)."""
    for gid in ("tshirt", "sweatshirt"):
        assert -2 <= defs[gid]["defaultEase"]["chest"] <= 20
    for gid in ("pants", "jeans"):
        assert -5 <= defs[gid]["defaultEase"]["hip"] <= 20
        assert -12 <= defs[gid]["defaultEase"]["inseam"] <= 3
    for gid in ("sneakers", "shoes", "boots"):
        assert -1 <= defs[gid]["defaultEase"]["footLength"] <= 4


def test_garments_deterministic(out_a, out_b):
    names = garments.garment_files(out_a / "garments")
    assert names == garments.garment_files(out_b / "garments")
    for n in names:
        assert (out_a / "garments" / n).read_bytes() == (out_b / "garments" / n).read_bytes(), n


def test_shipped_assets_smoke_live(morphset):
    """Real-data smoke check on the committed assets (skipped when they are not built): recompute one garment from the
    shipped binding + the shipped body morph model with the runtime formula, and compare to the shipped glb."""
    idx_file = DEFAULT_GARMENTS_DIR / "index.json"
    body = DEFAULT_OUT_DIR / "base.glb"
    if not idx_file.exists() or not body.exists() or not (DEFAULT_GARMENTS_DIR / "jeans.bind.bin").exists():
        pytest.skip("shipped garment assets not built (run `npm run assets:build`)")
    if b"git-lfs" in (DEFAULT_GARMENTS_DIR / "jeans.bind.bin").read_bytes()[:64]:
        pytest.skip("shipped garment assets are Git LFS pointers (run `git lfs pull`)")
    index = json.loads(idx_file.read_text(encoding="utf-8"))
    d = next(g for g in index["garments"] if g["id"] == "jeans")
    idx, w, off = load_binding(DEFAULT_GARMENTS_DIR / d["binding"])
    _, arr = glb_arrays(DEFAULT_GARMENTS_DIR / d["mesh"])
    # solved body: neutral macros, same frame as the shipped base.glb / manifest
    P = morphset.positions()
    rec = apply_binding(idx, w, off, d["scaleRefs"], P)
    assert len(rec) == len(arr["pos"])
    assert np.abs(rec - arr["pos"]).max() < 1e-5, "shipped garment glb == runtime formula on the neutral body"
    # a different body (tall + heavy) yields different, finite positions of the same topology
    P2 = morphset.positions({"gender": 1.0, "weight": 1.0, "height": 1.0})
    rec2 = apply_binding(idx, w, off, d["scaleRefs"], P2)
    assert np.isfinite(rec2).all() and np.abs(rec2 - rec).max() > 0.01
