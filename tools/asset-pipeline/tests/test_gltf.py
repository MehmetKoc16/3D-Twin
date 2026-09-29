"""base.glb validity: structural checks (pygltflib) and the Khronos glTF-Validator when npm is available."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pygltflib
import pytest

from config import CACHE_DIR
from test_rig_skin import read_accessor

HERE = Path(__file__).resolve().parent


def glb_json(path):
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    length = int.from_bytes(data[12:16], "little")
    assert data[16:20] == b"JSON"
    return json.loads(data[20 : 20 + length])


def load(out_a):
    return pygltflib.GLTF2().load_binary(str(out_a / "base.glb"))


def test_glb_structure(out_a, manifest):
    g = load(out_a)
    assert g.asset.version == "2.0"
    assert len(g.meshes) == 1 and len(g.meshes[0].primitives) == 1 and len(g.materials) == 1
    prim = g.meshes[0].primitives[0]
    assert prim.targets in (None, []), "morphs live in morphs.bin, not in glTF morph targets"
    at = prim.attributes
    for name in ("POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"):
        assert getattr(at, name) is not None, name
        assert g.accessors[getattr(at, name)].count == manifest["renderVertexCount"]
    assert len(g.skins) == 1 and g.nodes[g.skins[0].skeleton].name == "Root"
    assert g.scenes[0].nodes and g.scene == 0
    raw = glb_json(out_a / "base.glb")
    assert all("children" not in n or len(n["children"]) > 0 for n in raw["nodes"])
    assert "targets" not in raw["meshes"][0]["primitives"][0]


def test_glb_geometry(out_a, manifest, mesh):
    g = load(out_a)
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]
    pos = read_accessor(g, blob, prim.attributes.POSITION).astype(np.float64)
    nor = read_accessor(g, blob, prim.attributes.NORMAL).astype(np.float64)
    uv = read_accessor(g, blob, prim.attributes.TEXCOORD_0)
    idx = read_accessor(g, blob, prim.indices).reshape(-1)
    assert idx.max() == manifest["renderVertexCount"] - 1 and len(idx) == 3 * 26756
    assert len(set(idx.tolist())) == manifest["renderVertexCount"], "no orphan vertices"
    assert np.abs(np.linalg.norm(nor, axis=1) - 1).max() < 1e-5
    assert 0.0 <= uv.min() and uv.max() <= 1.0 + 1e-6
    # raw base mesh (not a neutral body) is ~1.666 m tall in the glb
    assert 1.65 < pos[:, 1].max() - pos[:, 1].min() < 1.68
    # outward-facing winding/normals: on the torso, normals point away from the body axis
    torso = (pos[:, 1] > 0.95) & (pos[:, 1] < 1.30) & (np.abs(pos[:, 0]) < 0.12)
    radial = pos[torso].copy()
    radial[:, 1] = 0.0  # horizontal vector from the body axis (x = z = 0) to the vertex
    assert ((nor[torso] * radial).sum(axis=1) > 0).mean() > 0.95
    acc = g.accessors[prim.attributes.POSITION]
    assert np.allclose(acc.min, pos.min(axis=0), atol=1e-6) and np.allclose(acc.max, pos.max(axis=0), atol=1e-6)


def _validator_prefix() -> Path | None:
    prefix = CACHE_DIR / "node"
    if (prefix / "node_modules" / "gltf-validator").exists():
        return prefix
    npm = shutil.which("npm")
    if not npm:
        return None
    prefix.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [npm, "install", "--prefix", str(prefix), "--no-audit", "--no-fund", "gltf-validator"],
        capture_output=True, text=True, timeout=240,
    )
    return prefix if res.returncode == 0 and (prefix / "node_modules" / "gltf-validator").exists() else None


def test_khronos_gltf_validator(out_a):
    node = shutil.which("node")
    prefix = _validator_prefix() if node else None
    if not node or prefix is None:
        pytest.skip("node/npm or gltf-validator not available; structural checks above still ran")
    res = subprocess.run(
        [node, str(HERE / "validate_glb.cjs"), str(out_a / "base.glb"), str(prefix)],
        capture_output=True, text=True, timeout=120,
    )
    issues = json.loads(res.stdout.strip().splitlines()[-1])
    assert issues["numErrors"] == 0 and issues["numWarnings"] == 0, issues["messages"]
