"""Synthetic tests only: placeholder model bytes and numeric cameras, no user-data."""

from pathlib import Path
import shutil
import stat
import uuid
import zipfile
from types import SimpleNamespace
from types import ModuleType
import json
import sys
import subprocess
import notebook_session

import numpy as np
import pytest

from camera import camera_matrices, projection_matrix
from io_utils import cleanup_session, safe_extract, stage_flame, upload_views
from setup_runtime import replace_exact
import setup_runtime
from diagnostics import traceback_tail, print_diagnostics, step_context
from notebook_session import cleanup_everything, private_config
from worker import DetectionError, is_detection_failure, skip_optional
from build_notebook import build


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


def test_view_names_require_front_and_reject_duplicates():
    assert upload_views(["front.png", "left.jpg", "right.jpeg", "back.png"])["left"] == "left.jpg"
    with pytest.raises(ValueError, match="required"):
        upload_views(["back.png", "left.png"])
    assert upload_views(["front.png"]) == {"front": "front.png"}
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


def test_tracebacks_hide_image_bytes_arrays_secrets_and_debug(tmp_path):
    log = tmp_path / "synthetic.log"
    log.write_text("detections [[1,2,3,4]]\nb'IMAGE_BYTES_SENTINEL'\n"
                   "Traceback (most recent call last):\n"
                   '  File "/synthetic/module.py", line 10, in main\n'
                   "    print(image)\nRuntimeError: CUDA out of memory\n"
                   "ValueError: tensor([1,2,3,4]) IMAGE_BYTES_SENTINEL\n"
                   "ValueError: data:image/png;base64,IMAGE_BYTES_SENTINEL\n"
                   "RuntimeError: password=SECRET_SENTINEL\n"
                   "KeyError: 'IMAGE_BYTES_SENTINEL'\n", encoding="utf-8")
    result = "\n".join(traceback_tail(log))
    assert "CUDA out of memory" in result
    assert "module.py" in result
    for hidden in ("IMAGE_BYTES_SENTINEL", "SECRET_SENTINEL", "detections", "print(image)"):
        assert hidden not in result


def test_diagnostics_show_step_and_view_and_bound_tail(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/cropping_landmarks-left.log").write_text("RuntimeError: synthetic failure\n" * 150)
    step_context(tmp_path, "cropping_landmarks", "left")
    print_diagnostics(tmp_path, limit=120)
    output = capsys.readouterr().out
    assert "Failed step: cropping_landmarks; view: left" in output
    assert output.count("RuntimeError:") == 120


def test_optional_profile_is_skipped_but_front_is_required(tmp_path):
    step_context(tmp_path, "cropping_landmarks", "left")
    skipped = skip_optional(tmp_path, "left", "no_usable_crop_or_landmarks")
    assert skipped["step"] == "cropping_landmarks"
    assert json.loads((tmp_path / "warnings.json").read_text())[0]["view"] == "left"
    step_context(tmp_path, "face_detection", "front")
    with pytest.raises(DetectionError, match="Front face required"):
        skip_optional(tmp_path, "front", "no_face_detected")


def test_crop_detection_failure_classification_excludes_install_and_cuda_errors(tmp_path):
    log = tmp_path / "synthetic.log"
    log.write_text("IndexError: list index out of range\n")
    assert is_detection_failure(log)
    log.write_text("RuntimeError: CUDA out of memory\n")
    assert not is_detection_failure(log)
    log.write_text("ImportError: failed to import insightface\n")
    assert not is_detection_failure(log)
    log.write_text("IndexError: list index out of range\nRuntimeError: CUDA out of memory\n")
    assert not is_detection_failure(log)


def populated_cache(tmp_path):
    cache = tmp_path / "dt-pixel3dmm-cache"
    for filename in ("env/bin/python", "FLAME2020.zip",
                     "pixel3dmm/src/pixel3dmm/preprocessing/MICA/data/FLAME2020/generic_model.pkl",
                     "pixel3dmm/src/pixel3dmm/preprocessing/MICA/data/pretrained/mica.tar",
                     "pixel3dmm/pretrained_weights/uv.ckpt", "pixel3dmm/pretrained_weights/normals.ckpt"):
        path = cache / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-not-a-model")
    (cache / "install_complete.json").write_text(json.dumps(setup_runtime.cache_signature("7.5")))
    return cache


def test_ready_cache_reused_without_installer_or_subprocess(tmp_path, monkeypatch):
    cache = populated_cache(tmp_path)
    monkeypatch.setattr(setup_runtime, "apply_runtime_patches", lambda root: None)
    monkeypatch.setattr(setup_runtime, "run", lambda *args, **kwargs: pytest.fail("Should not reinstall"))
    python, env = setup_runtime.install(cache, "7.5", tmp_path / "session")
    assert python == cache / "env/bin/python"
    assert env["TORCH_CUDA_ARCH_LIST"] == "7.5"
    assert env["DT_LOG_ROOT"] == str(tmp_path / "session")
    assert "extensions-sm75" in env["TORCH_EXTENSIONS_DIR"]
    assert env["XFORMERS_DISABLED"] == "1"


def test_cache_rejects_different_gpu_architecture(tmp_path):
    cache = populated_cache(tmp_path)
    with pytest.raises(RuntimeError, match="DELETE_VM_CACHE=True"):
        setup_runtime.require_cache(cache, "8.9")


def test_t4_policy_uses_float32_and_math_attention(monkeypatch):
    from runtime_compat import configure

    calls = {}
    cuda = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))
    for backend in ("flash", "mem_efficient", "math", "cudnn"):
        setattr(cuda, "enable_" + backend + "_sdp",
                lambda enabled, name=backend: calls.update({name: enabled}))
    fake_torch = SimpleNamespace(
        float32="float32", float16="float16",
        set_default_dtype=lambda value: calls.update(dtype=value),
        set_autocast_dtype=lambda device, value: calls.update(autocast=(device, value)),
        backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(allow_tf32=True)))
    fake_timm = ModuleType("timm")
    fake_layers = ModuleType("timm.layers")
    fake_layers.set_fused_attn = lambda enabled: calls.update(fused=enabled)
    fake_timm.layers = fake_layers
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setitem(sys.modules, "timm.layers", fake_layers)
    monkeypatch.setenv("XFORMERS_DISABLED", "0")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    configure()
    assert calls == {"dtype": "float32", "autocast": ("cuda", "float16"),
                     "flash": False, "mem_efficient": False, "math": True,
                     "cudnn": False, "fused": False}
    assert not cuda.matmul.allow_tf32 and not fake_torch.backends.cudnn.allow_tf32


def test_runtime_patches_are_idempotent_and_fail_closed(tmp_path):
    tracker = tmp_path / "pixel3dmm/src/pixel3dmm/tracking/tracker.py"
    network = tmp_path / "pixel3dmm/scripts/network_inference.py"
    tracker.parent.mkdir(parents=True)
    network.parent.mkdir(parents=True)
    tracker.write_text("COMPILE = True\n")
    network.write_text("model = model.cuda()\n")
    setup_runtime.apply_runtime_patches(tmp_path)
    setup_runtime.apply_runtime_patches(tmp_path)
    assert tracker.read_text() == "COMPILE = False\n"
    assert network.read_text() == "model = model.eval().float().cuda()\n"
    network.write_text("synthetic_unknown_upstream\n")
    with pytest.raises(RuntimeError, match="adapter mismatch"):
        setup_runtime.apply_runtime_patches(tmp_path)


def test_private_session_cleanup_keeps_cache_and_final_cleanup_deletes_it(tmp_path):
    cache = populated_cache(tmp_path)
    session = tmp_path / "dt-pixel3dmm-session-synthetic"
    session.mkdir()
    (session / "upload.placeholder").write_bytes(b"synthetic")
    config = tmp_path / "synthetic.env"
    config.write_text("".join(key + "=" + json.dumps(value) + "\n" for key, value in {
        "PIXEL3DMM_CODE_BASE": str(cache / "pixel3dmm"),
        "PIXEL3DMM_PREPROCESSED_DATA": str(session / "preprocessed"),
        "PIXEL3DMM_TRACKING_OUTPUT": str(session / "tracking"),
    }.items()))
    assert private_config(config, cache, tmp_path)
    cleanup_session(session, tmp_path)
    assert cache.exists()
    cleanup_everything(cache, tmp_path, config)
    assert not cache.exists() and not config.exists()


def test_final_cleanup_preserves_unrelated_config_and_refuses_unowned_cache(tmp_path):
    cache = populated_cache(tmp_path)
    config = tmp_path / "unrelated.env"
    config.write_text("UNRELATED=true\n")
    cleanup_everything(cache, tmp_path, config)
    assert config.read_text() == "UNRELATED=true\n"
    with pytest.raises(ValueError, match="Refusing unsafe cache"):
        cleanup_everything(tmp_path / "unrelated", tmp_path, config)


def test_notebook_preserves_lead_parameters_and_has_fit_only_cell():
    cells = {cell["id"]: cell["source"] for cell in build()["cells"]}
    assert "/content/drive/MyDrive/flame/FLAME2020.zip" in cells["parameters"]
    assert '"T4": "7.5"' in cells["gpu-check"]
    assert "run_session(prepare=False)" in cells["fit-only-retry"]
    assert "DELETE_VM_CACHE = False" in cells["privacy-cleanup"]


def test_failed_fit_prints_diagnostics_before_private_cleanup_and_keeps_models(tmp_path, monkeypatch, capsys):
    cache = populated_cache(tmp_path)
    session = tmp_path / "dt-pixel3dmm-session-synthetic"
    session.mkdir()
    fake_google = ModuleType("google")
    fake_colab = ModuleType("google.colab")

    def upload(target_dir):
        (Path(target_dir) / "front.png").write_bytes(b"synthetic-not-an-image")
        return {"front.png": b"synthetic-not-an-image"}

    fake_colab.files = SimpleNamespace(upload=upload)
    fake_google.colab = fake_colab
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(notebook_session.tempfile, "mkdtemp", lambda **kwargs: str(session))
    monkeypatch.setattr(notebook_session, "apply_runtime_patches", lambda root: None)

    def fail(command, root, **kwargs):
        step_context(root, "cropping_landmarks", "front")
        (root / "logs").mkdir()
        (root / "logs/cropping_landmarks-front.log").write_text(
            'Traceback (most recent call last):\n  File "/synthetic.py", line 1, in main\n'
            "IndexError: list index out of range\nIMAGE_BYTES_SENTINEL\n")
        raise subprocess.CalledProcessError(1, ["synthetic-worker"])

    monkeypatch.setattr(notebook_session, "run", fail)
    events = []
    original_print = notebook_session.print_diagnostics

    def print_before_cleanup(root):
        assert root.exists() and (root / "logs").exists()
        events.append("diagnostics")
        original_print(root)

    def cleanup_after_diagnostics(root):
        assert events == ["diagnostics"]
        events.append("cleanup")
        cleanup_session(root, tmp_path)

    monkeypatch.setattr(notebook_session, "print_diagnostics", print_before_cleanup)
    monkeypatch.setattr(notebook_session, "cleanup_session", cleanup_after_diagnostics)
    with pytest.raises(subprocess.CalledProcessError):
        notebook_session.run_session(cache, {}, "2020", "7.5", 1, 1, 1)
    output = capsys.readouterr().out
    assert "Failed step: cropping_landmarks; view: front" in output
    assert "IndexError: list index out of range" in output
    assert "IMAGE_BYTES_SENTINEL" not in output
    assert events == ["diagnostics", "cleanup"]
    assert not session.exists() and (cache / "FLAME2020.zip").exists()
    assert not (tmp_path / ".config/pixel3dmm/.env").exists()
