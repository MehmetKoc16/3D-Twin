"""Determinism, counts, index bounds and size budget."""

from __future__ import annotations

import hashlib

import numpy as np

from config import OUTPUT_FILES
from targets import ENTRY_DTYPE, EXTENDED_RANGE

MAX_TOTAL_BYTES = 20 * 1024 * 1024


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_all_outputs_exist(out_a):
    for f in OUTPUT_FILES:
        assert (out_a / f).is_file(), f


def test_determinism_two_builds_identical(out_a, out_b):
    for f in OUTPUT_FILES:
        assert _sha(out_a / f) == _sha(out_b / f), f"{f} differs between two builds"


def test_size_budget(out_a):
    total = sum((out_a / f).stat().st_size for f in OUTPUT_FILES)
    assert total <= MAX_TOTAL_BYTES, f"{total} bytes"


def test_vertex_counts(manifest, mesh):
    assert manifest["renderVertexCount"] == mesh.render_count == 14517
    assert len(manifest["jointPoints"]) == 69
    assert manifest["vertexCount"] == manifest["renderVertexCount"] + len(manifest["jointPoints"])
    names = [j["name"] for j in manifest["jointPoints"]]
    assert names == sorted(names) and len(set(names)) == 69


def test_morphs_bin_layout(manifest, morph_bytes):
    assert len(morph_bytes) % 16 == 0
    entries = np.frombuffer(morph_bytes, dtype=ENTRY_DTYPE)
    expected_offset = 0
    for t in manifest["targets"]:
        assert t["byteOffset"] == expected_offset, t["id"]
        assert t["count"] > 0, t["id"]
        expected_offset += t["count"] * 16
    assert expected_offset == len(morph_bytes)
    assert len({t["id"] for t in manifest["targets"]}) == len(manifest["targets"])
    assert {t["group"] for t in manifest["targets"]} <= {"macro", "measure", "body", "face", "foot"}
    # indices in bounds; each run strictly ascending (unique); finite; meters-scale; tiny deltas dropped
    assert entries["i"].max() < manifest["vertexCount"]
    d = entries["d"]
    assert np.isfinite(d).all()
    assert np.abs(d).max() < 1.0  # meters: no unit mix-up
    assert (np.abs(d).max(axis=1) >= 1e-5 - 1e-12).all(), "deltas below 1e-5 m must be dropped"
    for t in manifest["targets"]:
        s = t["byteOffset"] // 16
        idx = entries["i"][s : s + t["count"]].astype(np.int64)
        assert (np.diff(idx) > 0).all(), t["id"]


def test_target_and_modifier_references(manifest):
    target_ids = {t["id"] for t in manifest["targets"]}
    variables = {v["id"]: {b["name"] for b in v["buckets"]} for v in manifest["macroVariables"]}
    for t in manifest["targets"]:
        for c in t.get("macroConditions", []):
            assert c["bucket"] in variables[c["variable"]], (t["id"], c)
        assert (t["group"] == "macro") == ("macroConditions" in t), t["id"]
    ids = [m["id"] for m in manifest["modifiers"]]
    assert len(ids) == len(set(ids))
    used = set()
    for m in manifest["modifiers"]:
        assert m["min"] <= m["default"] <= m["max"]
        assert "decrTarget" in m or "incrTarget" in m, m["id"]
        # default range +-1; the measure drivers listed in targets.EXTENDED_RANGE reach 1.5 where geometry allows
        lo, hi = EXTENDED_RANGE.get(m["id"], (-1.0, 1.0))
        assert m["max"] == hi
        if "decrTarget" in m:
            assert m["min"] == lo
        else:
            assert m["min"] == 0.0
        for k in ("decrTarget", "incrTarget"):
            if k in m:
                assert m[k] in target_ids, (m["id"], m[k])
                used.add(m[k])
    non_macro = {t["id"] for t in manifest["targets"] if t["group"] != "macro"}
    assert non_macro == used, "every non-macro target must belong to a modifier"


def test_macro_variable_defs(manifest):
    v = {m["id"]: m for m in manifest["macroVariables"]}
    assert set(v) == {"gender", "age", "muscle", "weight", "height", "cupsize", "firmness"}
    assert [b["name"] for b in v["age"]["buckets"]] == ["young"]  # age pinned to 25 y
    for k in ("gender", "muscle", "weight", "height"):
        assert v[k]["default"] == 0.5
    assert [(b["name"], b["at"]) for b in v["weight"]["buckets"]] == [
        ("minweight", 0.0),
        ("averageweight", 0.5),
        ("maxweight", 1.0),
    ]
    # race is pre-merged: no race variable, no per-race targets
    assert "race" not in v
    assert not any("african" in t["id"] or "asian" in t["id"] for t in manifest["targets"])


def test_no_asymmetric_side_targets(manifest):
    # l/r targets must be pre-merged
    for t in manifest["targets"]:
        leaf = t["id"].split("/")[-1]
        assert not leaf.startswith(("l-", "r-")), t["id"]


def test_macro_bucket_positions_unique(manifest):
    for v in manifest["macroVariables"]:
        ats = [b["at"] for b in v["buckets"]]
        assert len(ats) == len(set(ats)) and ats == sorted(ats), v["id"]
        assert all(v["min"] <= a <= v["max"] or v["min"] == v["max"] for a in ats)
    assert {"gender", "height", "weight"} <= {v["id"] for v in manifest["macroVariables"]}
    assert all(t["byteOffset"] % 4 == 0 for t in manifest["targets"])


def test_render_mesh_is_closed(mesh):
    """Every edge is shared by exactly two triangles once UV-seam copies are welded by MakeHuman vertex id."""
    t = mesh.tris
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    _, counts = np.unique(np.sort(mesh.render_mh[e], axis=1), axis=0, return_counts=True)
    assert (counts == 2).all()
