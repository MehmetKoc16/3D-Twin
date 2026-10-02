"""Synthetic tests only: placeholder model bytes and numeric cameras, no user-data."""

from pathlib import Path
import shutil
import stat
import uuid
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from camera import camera_matrices, projection_matrix
from io_utils import cleanup_session, safe_extract, stage_flame, upload_views
from setup_runtime import replace_exact
import setup_runtime


@pytest.fixture
def tmp_path():
    # Windows sandbox ACLs reject pytest's default mode-0700 temp directories.
    parent = Path(__file__).resolve().parent / ".cache"
    parent.mkdir(exist_ok=True)
    directory = parent / ("synthetic-" + uuid.uuid4().hex)
    directory.mkdir(mode=0o777)
    try:
        yield directory
    finally:
        if directory.resolve().parent != parent.resolve():
            raise ValueError("Unsafe synthetic test cleanup")
        shutil.rmtree(directory)


@pytest.mark.parametrize("member", ["../outside", "/absolute", "nested/../../outside", "..\\outside"])
def test_zip_rejects_traversal_before_writing(tmp_path, member):
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("valid.txt", "placeholder")
        bundle.writestr(member, "placeholder")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked/valid.txt").exists()


def test_zip_rejects_symlink(tmp_path):
    archive = tmp_path / "synthetic.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "../outside")
    with pytest.raises(ValueError, match="symlinks"):
        safe_extract(archive, tmp_path / "unpacked")


@pytest.mark.parametrize("version,filename", [("2020", "generic_model.pkl"), ("2023", "flame2023_no_jaw.pkl")])
def test_nested_model_zip_and_existing_auxiliary_assets(tmp_path, version, filename):
    assets = tmp_path / "assets"
    if version == "2020":
        model = assets / "FLAME2020"
        (model / "FLAME_masks").mkdir(parents=True)
        (model / "landmark_embedding.npy").write_bytes(b"synthetic-not-a-model")
        (model / "FLAME_masks/FLAME_masks.pkl").write_bytes(b"synthetic-not-a-model")
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("download/FLAME" + version + "/" + filename, b"synthetic-not-a-model")
    stage_flame(archive, assets, version)
    assert (assets / ("FLAME" + version) / filename).read_bytes() == b"synthetic-not-a-model"
    assert not (assets / ("unpacked-" + version)).exists()


def test_wrong_flame_version_is_not_silently_substituted(tmp_path):
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("generic_model.pkl", b"synthetic")
    with pytest.raises(ValueError, match="flame2023_no_jaw"):
        stage_flame(archive, tmp_path / "assets", "2023")


def test_cleanup_on_error_removes_every_session_artifact(tmp_path):
    session = tmp_path / "dt-pixel3dmm-session-synthetic"
    session.mkdir()
    for name in ("uploads", "export", "pixel3dmm", "cache"):
        (session / name).mkdir()
        (session / name / "placeholder").write_bytes(b"synthetic")
    try:
        raise RuntimeError("synthetic inference failure")
    except RuntimeError:
        pass
    finally:
        cleanup_session(session, parent=tmp_path)
    assert not session.exists()


def test_cleanup_refuses_unowned_directory(tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ValueError, match="Refusing"):
        cleanup_session(unrelated, parent=tmp_path)
    assert unrelated.exists()


def test_view_names_require_both_profiles_and_reject_duplicates():
    assert upload_views(["front.png", "left.jpg", "right.jpeg", "back.png"])["left"] == "left.jpg"
    with pytest.raises(ValueError, match="required"):
        upload_views(["front.png", "back.png"])
    with pytest.raises(ValueError, match="one image"):
        upload_views(["front.png", "front.jpg", "left.png", "right.png"])


def test_camera_projection_and_noncommuting_transform_order():
    # Rotating the global translation in the base camera exposes reversed order.
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    checkpoint = {
        "img_size": [256, 256],
        "camera": {"fl": [[2]], "pp": [[0.1, -0.1]], "R_base_0": rotation[None], "t_base_0": [[0, 0, -2]]},
        "flame": {"R_rotation_matrix": np.eye(3)[None], "t": [[0.25, 0, 0]]},
    }
    intrinsics, extrinsics, side = camera_matrices(checkpoint)
    assert intrinsics[0, 0] == 512
    np.testing.assert_allclose(extrinsics[:3, 3], [0, 0.25, -2])
    point = np.array([0.1, 0.2, 0, 1])
    q = extrinsics @ point
    expected = [512 * q[0] / -q[2] + intrinsics[0, 2], intrinsics[1, 2] - 512 * q[1] / -q[2]]
    clip = projection_matrix(intrinsics, side) @ q
    pixels = (clip[:2] / clip[3] + 1) * side / 2
    np.testing.assert_allclose(pixels, expected, atol=1e-5)


def test_camera_rejects_nonfinite_fit():
    checkpoint = {"img_size": [256, 256], "camera": {"fl": [[float("nan")]], "pp": [[0, 0]],
                  "R_base_0": np.eye(3)[None], "t_base_0": [[0, 0, -1]]},
                  "flame": {"R_rotation_matrix": np.eye(3)[None], "t": [[0, 0, 0]]}}
    with pytest.raises(ValueError, match="Invalid"):
        camera_matrices(checkpoint)


def test_upstream_patch_requires_exact_match(tmp_path):
    path = tmp_path / "synthetic.sh"
    path.write_text("old\nold\n")
    with pytest.raises(RuntimeError, match="adapter mismatch"):
        replace_exact(path, "old", "new")
    path.write_text("old\n")
    replace_exact(path, "old", "new")
    assert path.read_text() == "new\n"


def test_interruption_stops_process_group_before_returning(tmp_path, monkeypatch):
    events = []

    class Process:
        pid = 123

        def wait(self):
            events.append("wait")
            if len(events) == 1:
                raise KeyboardInterrupt
            return -9

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"]
        return Process()

    monkeypatch.setattr(setup_runtime, "os", SimpleNamespace(
        name="posix", environ={}, killpg=lambda pid, signal: events.append("kill_group")))
    monkeypatch.setattr(setup_runtime, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(setup_runtime.subprocess, "Popen", popen)
    with pytest.raises(KeyboardInterrupt):
        setup_runtime.run(["synthetic-worker"], tmp_path)
    assert events == ["wait", "kill_group", "wait"]
