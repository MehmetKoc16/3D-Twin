"""Synthetic geometry only: contract rejection, camera semantics and cleanup."""

from __future__ import annotations

import json
import os
import shutil
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import trimesh

from check_meta import project_points, validate_pair

HERE = Path(__file__).resolve().parent


@pytest.fixture
def synthetic_dir():
    # Default mkdir permissions work in restricted Windows workspaces.
    root = HERE / ".cache"
    root.mkdir(exist_ok=True)
    path = root / ("synthetic-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def synthetic_bundle(directory: Path, transformed_node: bool = False) -> tuple[Path, Path, dict]:
    height = 1.78
    mesh = trimesh.creation.box(extents=[0.6, height, 0.3])
    transform = np.eye(4)
    transform[1, 3] = height / 2
    if transformed_node:
        scene = trimesh.Scene()
        scene.add_geometry(mesh, transform=transform)
        scene.export(directory / "mesh.glb")
    else:
        mesh.apply_transform(transform)
        mesh.export(directory / "mesh.glb")
    metadata = {
        "schema": "twin-shape/1",
        "coordinateSystem": {"units": "meters", "handedness": "right", "up": "+Y", "forward": "+Z (the character faces +Z)", "characterLeft": "+X"},
        "normalization": {
            "heightCm": 178, "rotationSourceToStandard": np.eye(3).tolist(), "scaleMetersPerSourceUnit": 1,
            "translationMeters": [0, height / 2, 0], "matrix4x4RowMajor": transform.tolist(),
            "finalBBox": {"min": [-0.3, 0, -0.15], "max": [0.3, height, 0.15], "sizeMeters": [0.6, height, 0.3]},
        },
        "inputs": {view: {"file": f"{view}.png"} for view in ("front", "left", "back", "right")},
        "cameras": {"views": {view: {
            "yawDeg": yaw, "imageSize": [640, 1000], "originPx": [320, 990], "pxPerMeter": 500,
            "silhouetteIoU": 0.9, "maskBBoxPx": [170, 100, 470, 990],
        } for view, yaw in {"front": 0, "left": 90, "back": 180, "right": 270}.items()}},
        "model": {"repo": "microsoft/TRELLIS.2-4B", "multiView": False, "viewsUsedForShape": ["front"]},
        "mesh": {"vertices": 8, "faces": 12, "watertight": True}, "files": {"glb": "mesh.glb"},
    }
    path = directory / "meta.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return directory / "mesh.glb", path, metadata


@pytest.mark.parametrize("transformed_node", [False, True])
def test_valid_glb_including_node_transforms(synthetic_dir, transformed_node):
    mesh, meta, _ = synthetic_bundle(synthetic_dir, transformed_node)
    result = validate_pair(mesh, meta)
    assert result["valid"] is True
    assert result["anatomicalFrontVerified"] is False


@pytest.mark.parametrize("change,message", [
    (lambda m: m["coordinateSystem"].update(units="centimeters"), "units"),
    (lambda m: m["normalization"].update(heightCm=180), "height"),
    (lambda m: m["normalization"].update(rotationSourceToStandard=np.diag([-1, 1, 1]).tolist()), "reflection"),
    (lambda m: m["normalization"]["matrix4x4RowMajor"][0].__setitem__(3, 0.2), "Matrix"),
    (lambda m: m["normalization"]["finalBBox"]["max"].__setitem__(2, 0.4), "finalBBox"),
    (lambda m: m["cameras"]["views"]["left"].update(yawDeg=270), "yaw"),
    (lambda m: m["cameras"]["views"]["front"].update(pxPerMeter=float("nan")), "pxPerMeter"),
    (lambda m: m["cameras"]["views"]["front"].update(originPx=[0, float("inf")]), "originPx"),
    (lambda m: m["cameras"]["views"].pop("back"), "camera"),
    (lambda m: m["model"].update(multiView=True, viewsUsedForShape=["front", "back"]), "multi-view"),
])
def test_reject_inconsistent_metadata(synthetic_dir, change, message):
    mesh, path, meta = synthetic_bundle(synthetic_dir)
    change(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_pair(mesh, path)


@pytest.mark.parametrize("translation,message", [([0, 0.1, 0], "Feet"), ([0.1, 0, 0], "contact patch")])
def test_reject_wrong_ground_or_origin(synthetic_dir, translation, message):
    path, meta, _ = synthetic_bundle(synthetic_dir)
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.apply_translation(translation)
    mesh.export(path)
    with pytest.raises(ValueError, match=message):
        validate_pair(path, meta)


def test_all_camera_yaws_and_pixel_direction():
    points = np.array([[1, 2, 3], [0, 0, 0]])
    expected_horizontal = {0: 1, 90: -3, 180: -1, 270: 3}
    for yaw, horizontal in expected_horizontal.items():
        camera = {"yawDeg": yaw, "originPx": [320, 990], "pxPerMeter": 100}
        pixels = project_points(points, camera)
        np.testing.assert_allclose(pixels, [[320 + 100 * horizontal, 790], [320, 990]])


@pytest.mark.parametrize("failure", [None, "worker", "download", "upload", "interrupt"])
def test_notebook_session_deletes_synthetic_files(synthetic_dir, monkeypatch, failure):
    notebook = json.loads((HERE / "trellis2_shape.ipynb").read_text(encoding="utf-8"))
    source = next(c["source"] for c in notebook["cells"] if c["id"] == "run-with-cleanup")
    session = synthetic_dir / "synthetic-session"
    session.mkdir()
    order = []

    def upload(target_dir):
        name = "unexpected.png" if failure == "upload" else "front.png"
        path = Path(target_dir) / name
        path.write_bytes(b"synthetic data only")
        return {str(path): b"synthetic data only"}

    def command(_args):
        order.append("worker")
        if failure == "interrupt":
            raise KeyboardInterrupt("synthetic cancellation")
        if failure == "worker":
            raise RuntimeError("synthetic worker failure")
        (session / "out/mesh.glb").write_bytes(b"synthetic placeholder")

    def download(path):
        assert path.exists() and (session / "uploads/front.png").exists()
        order.append("download")
        if failure == "download":
            raise RuntimeError("synthetic transfer failure")

    def cleanup(path):
        assert path == session
        order.append("cleanup")
        shutil.rmtree(path)

    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=upload)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    from types import SimpleNamespace

    namespace = dict(
        tempfile=SimpleNamespace(mkdtemp=lambda **_: str(session)), Path=Path, ENV_PYTHON="synthetic-python",
        HEIGHT_CM=178, SEED=42, PIPELINE_TYPE="512", STEPS=12, TARGET_FACES=80000,
        MAX_NUM_TOKENS=49152, SOURCE_UP="auto", SOURCE_FORWARD="auto", FLIP_FORWARD=False,
        TRELLIS_SOURCE="synthetic", TRELLIS_REVISION="synthetic", SPARSE_DECODER_REVISION="synthetic",
        DINO_REVISION="synthetic", EMBEDDED_FILES={}, run_command=command,
        download_and_wait=download, cleanup_session=cleanup, os=os,
    )
    if failure:
        with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt)):
            exec(compile(source, "run-with-cleanup", "exec"), namespace)
    else:
        exec(compile(source, "run-with-cleanup", "exec"), namespace)
    assert not session.exists()
    assert order[-1] == "cleanup"


def test_notebook_matches_its_sources():
    from build_notebook import build

    assert json.loads((HERE / "trellis2_shape.ipynb").read_text(encoding="utf-8")) == build()
