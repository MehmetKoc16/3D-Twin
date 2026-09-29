"""Shared fixtures: build the assets once (and a second time for the determinism test) into temp dirs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))

import build  # noqa: E402
import fetch  # noqa: E402
import macro  # noqa: E402
from config import DM_TO_M, MH_DATA, OUTPUT_FILES  # noqa: E402


@pytest.fixture(scope="session")
def raw_data():
    try:
        fetch.ensure_present()
    except RuntimeError:
        fetch.fetch_all()


@pytest.fixture(scope="session")
def out_a(tmp_path_factory, raw_data) -> Path:
    d = tmp_path_factory.mktemp("assets_a")
    build.build_all(d, verbose=False)
    return d


@pytest.fixture(scope="session")
def out_b(tmp_path_factory, raw_data) -> Path:
    d = tmp_path_factory.mktemp("assets_b")
    build.build_all(d, verbose=False)
    return d


@pytest.fixture(scope="session")
def manifest(out_a) -> dict:
    return json.loads((out_a / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def rig(out_a) -> dict:
    return json.loads((out_a / "rig.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def measures(out_a) -> dict:
    return json.loads((out_a / "measures.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def morph_bytes(out_a) -> bytes:
    return (out_a / "morphs.bin").read_bytes()


@pytest.fixture(scope="session")
def base_positions(out_a, manifest) -> np.ndarray:
    """Combined-space base positions (render vertices from base.glb, then joint points), meters."""
    import pygltflib

    g = pygltflib.GLTF2().load_binary(str(out_a / "base.glb"))
    acc = g.accessors[g.meshes[0].primitives[0].attributes.POSITION]
    view = g.bufferViews[acc.bufferView]
    blob = g.binary_blob()
    pos = np.frombuffer(blob, dtype="<f4", count=acc.count * 3, offset=view.byteOffset).reshape(-1, 3)
    joints = np.array([j["position"] for j in manifest["jointPoints"]], dtype=np.float64)
    return np.concatenate([pos.astype(np.float64), joints])


@pytest.fixture(scope="session")
def morphset(manifest, morph_bytes, base_positions) -> macro.MorphSet:
    return macro.MorphSet(manifest, morph_bytes, base_positions)


@pytest.fixture(scope="session")
def output_names():
    return OUTPUT_FILES


@pytest.fixture(scope="session")
def mesh(raw_data):
    import mh_obj

    return mh_obj.load_base_mesh(MH_DATA / "3dobjs" / "base.obj")
