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
import importlib.util
import worker
import notebook_session
import downloads

import numpy as np
import pytest

from camera import camera_matrices, projection_matrix
from io_utils import cleanup_session, safe_extract, stage_flame, upload_views, VIEW_ORDER
from fit_policy import fit_budget
from export_fit import camera_document, camera_view, expression_indicators
from setup_runtime import replace_exact
import setup_runtime
from diagnostics import traceback_tail, print_diagnostics, step_context, install_log_tail
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
    pipnet = tmp_path / "pixel3dmm/src/pixel3dmm/preprocessing/pipnet_utils.py"
    tracker.parent.mkdir(parents=True)
    network.parent.mkdir(parents=True)
    pipnet.parent.mkdir(parents=True)
    tracker.write_text("COMPILE = True\n")
    network.write_text("model = model.cuda()\n")
    pipnet.write_text("if detections[i][1] < 0.99:\n    pass\n")
    setup_runtime.apply_runtime_patches(tmp_path)
    setup_runtime.apply_runtime_patches(tmp_path)
    assert tracker.read_text() == "COMPILE = False\n"
    assert network.read_text() == "model = model.eval().float().cuda()\n"
    assert "DT_PROFILE_CROP_RETRY" in pipnet.read_text()
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
    kept = tmp_path / "dt-pixel3dmm-result"
    kept.mkdir()
    (kept / "head_fit.zip").write_bytes(b"synthetic")
    cleanup_session(session, tmp_path)
    assert cache.exists() and kept.exists()  # kept result survives per-fit cleanup
    cleanup_everything(cache, tmp_path, config)
    assert not cache.exists() and not config.exists() and not kept.exists()


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

    def print_before_cleanup(root, **kwargs):
        assert root.exists() and (root / "logs").exists()
        assert kwargs == {"allow_install_logs": False}
        events.append("diagnostics")
        original_print(root, **kwargs)

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


def test_install_checkpoints_resume_failed_step_in_order(tmp_path):
    events = []
    steps = setup_runtime.InstallSteps(tmp_path, "7.5", tmp_path, {})

    def run_attempt(fail):
        for identifier in ("env", "torch", "requirements", "pytorch3d", "nvdiffrast", "pixel-editable", "facer", "mica", "pipnet"):
            def operation(name=identifier):
                events.append(name)
                if name == fail:
                    raise RuntimeError("synthetic build failure")
            steps.perform(identifier, identifier, operation)

    with pytest.raises(RuntimeError, match="synthetic build"):
        run_attempt("facer")
    assert events == ["env", "torch", "requirements", "pytorch3d", "nvdiffrast", "pixel-editable", "facer"]
    assert not (tmp_path / "done/facer.json").exists()
    events.clear()
    run_attempt(None)
    assert events == ["facer", "mica", "pipnet"]
    assert json.loads((tmp_path / "done/pipnet.json").read_text())["signature"] == setup_runtime.cache_signature("7.5")
    assert json.loads((tmp_path / "current_step.json").read_text())["step"] == "install: pipnet"


def test_stale_corrupt_and_missing_artifact_markers_are_ignored(tmp_path):
    events = []
    artifact = tmp_path / "artifact"

    def operation():
        events.append("build")
        artifact.write_text("synthetic")

    steps = setup_runtime.InstallSteps(tmp_path, "7.5", tmp_path, {})
    steps.perform("torch", "torch", operation, [artifact])
    steps.perform("torch", "torch", operation, [artifact])
    assert events == ["build"]
    setup_runtime.InstallSteps(tmp_path, "8.9", tmp_path, {}).perform("torch", "torch", operation, [artifact])
    assert events == ["build", "build"]
    (tmp_path / "done/torch.json").write_text("corrupt")
    steps.perform("torch", "torch", operation, [artifact])
    artifact.unlink()
    steps.perform("torch", "torch", operation, [artifact])
    assert len(events) == 4


def test_preprocessing_substep_order_and_individual_weight_steps(tmp_path):
    events = []
    steps = SimpleNamespace(root=tmp_path,
        perform=lambda identifier, label, operation, required=(): events.append(identifier),
        weight=lambda name, destination: events.append("weight:" + name))
    setup_runtime.preprocessing_install(steps)
    assert events == ["verify-upstream", "facer-source", "facer", "mica", "weight:mica.tar",
                      "weight:antelopev2.zip", "extract-antelopev2", "weight:buffalo_l.zip", "extract-buffalo_l",
                      "pipnet-source", "pipnet", "weight:epoch59.pth", "weight:uv.ckpt", "weight:normals.ckpt"]


def test_pipnet_build_uses_env_python_and_checks_artifact(tmp_path):
    directory = tmp_path / "pixel3dmm/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/utils"
    (directory / "nms").mkdir(parents=True)
    (directory / "make.sh").write_text("python3 build.py build_ext --inplace")
    (directory / "build.py").write_text("nms/cpu_nms.pyx")
    calls = []

    def command(args, cwd):
        calls.append((args, cwd))
        (directory / "nms/cpu_nms.synthetic.so").write_bytes(b"synthetic")

    def perform(identifier, label, operation, required=()):
        if identifier == "pipnet":
            operation()

    steps = SimpleNamespace(root=tmp_path, perform=perform, command=command, weight=lambda *args: None)
    setup_runtime.preprocessing_install(steps)
    assert calls == [([tmp_path / "env/bin/python", "build.py", "build_ext", "--inplace"], directory)]


@pytest.mark.parametrize("name", ["uv.ckpt", "normals.ckpt", "antelopev2.zip", "buffalo_l.zip", "mica.tar", "epoch59.pth"])
def test_weight_source_order_retries_and_atomic_download(tmp_path, monkeypatch, name):
    spec = {**downloads.WEIGHTS[name], "minimum": 1}
    monkeypatch.setitem(downloads.WEIGHTS, name, spec)
    calls, delays = [], []

    def fetch(url, temporary):
        calls.append(url)
        if url == spec["sources"][-1][1] and calls.count(url) == 3:
            temporary.write_bytes(b"synthetic-valid-weight")
        else:
            raise subprocess.CalledProcessError(1, ["synthetic-download"])

    destination = tmp_path / "weights" / name
    downloads.download_weight(name, destination, tmp_path, {"https": fetch, "gdown": fetch},
                              lambda message: None, sleep=delays.append)
    assert calls == [url for transport, url in spec["sources"] for _ in range(3)]
    assert delays == [1, 2] * len(spec["sources"])
    assert destination.read_bytes() == b"synthetic-valid-weight"
    assert not destination.with_name(name + ".partial").exists()
    downloads.download_weight(name, destination, tmp_path,
                              {"https": lambda *args: pytest.fail("cached"), "gdown": lambda *args: pytest.fail("cached")},
                              lambda message: None)


def test_download_failure_names_file_source_and_drive_destination(tmp_path, monkeypatch):
    monkeypatch.setitem(downloads.WEIGHTS, "uv.ckpt", {**downloads.WEIGHTS["uv.ckpt"], "minimum": 1})
    logs = []

    def fail(url, temporary):
        temporary.write_bytes(b"<html>synthetic error page</html>")

    with pytest.raises(RuntimeError) as error:
        downloads.download_weight("uv.ckpt", tmp_path / "uv.ckpt", tmp_path,
                                  {"https": fail, "gdown": fail}, logs.append, sleep=lambda delay: None)
    assert "Missing weight: uv.ckpt" in str(error.value)
    assert downloads.HF_BASE + "uv.ckpt" in str(error.value)
    assert "MyDrive/flame/weights/uv.ckpt" in str(error.value)
    assert not (tmp_path / "uv.ckpt").exists()
    assert not (tmp_path / "uv.ckpt.partial").exists()


def test_drive_weight_folder_is_preferred_without_network(tmp_path, monkeypatch):
    folder = tmp_path / "synthetic-Drive/MyDrive/flame/weights"
    folder.mkdir(parents=True)
    (folder / "mica.tar").write_bytes(b"synthetic-model-placeholder")
    (folder / "unrelated.txt").write_text("synthetic ignored file")
    monkeypatch.setitem(downloads.WEIGHTS, "mica.tar", {**downloads.WEIGHTS["mica.tar"], "minimum": 1})
    cache = tmp_path / "cache"
    assert downloads.copy_drive_weights(cache, folder) == ["mica.tar"]
    destination = cache / "models/mica.tar"
    downloads.download_weight("mica.tar", destination, cache,
                              {"gdown": lambda *args: pytest.fail("Drive copy must win")}, lambda message: None)
    assert destination.read_bytes() == b"synthetic-model-placeholder"
    assert not (cache / "drive_weights/unrelated.txt").exists()


def test_unfinished_install_checks_drive_weights_in_same_mount(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "FLAME2020.zip").write_bytes(b"synthetic")
    selected = {"2020": "/content/drive/MyDrive/flame/FLAME2020.zip"}
    (cache / "selected_archives.json").write_text(json.dumps(selected))
    events = []
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda path: events.append("mount"),
                                  flush_and_unmount=lambda: events.append("unmount"))
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(notebook_session, "copy_drive_weights", lambda root: events.append("copy_weights"))
    notebook_session.copy_flame_archives(cache, selected)
    assert events == ["mount", "copy_weights", "unmount"]
    events.clear()
    (cache / "install_complete.json").write_text("synthetic-ready")
    notebook_session.copy_flame_archives(cache, selected)
    assert events == []


def test_install_tail_collapses_progress_redacts_credentials_and_bounds_lines(tmp_path):
    log = tmp_path / "install.log"
    log.write_bytes(("old\n" * 70 + "\x1b[31m10%\r50%\r100%\x1b[0m\r\n"
                     "token=SECRET_ONE\n--password SECRET_TWO\ncredential: SECRET_THREE\n"
                     "Authorization: Bearer SECRET_FOUR\nhttps://user:SECRET_FIVE@example.com/a\n"
                     + "x" * 450 + "\n").encode())
    lines = install_log_tail(log)
    assert len(lines) == 60 and max(map(len, lines)) == 300
    assert "100%" in lines and "10%" not in lines and "50%" not in lines
    assert "\x1b" not in "\n".join(lines) and "SECRET_" not in "\n".join(lines)


@pytest.mark.parametrize("step", ["photo_upload", "cropping_landmarks", "tracking", "export", "archive_download"])
def test_post_upload_steps_never_print_raw_logs_even_if_flag_incorrect(tmp_path, capsys, step):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/install-synthetic-all.log").write_text("PRIVATE_PAYLOAD_SENTINEL\nRuntimeError: synthetic failure\n")
    step_context(tmp_path, step, log_name="install-synthetic-all.log")
    print_diagnostics(tmp_path, allow_install_logs=True)
    output = capsys.readouterr().out
    assert "PRIVATE_PAYLOAD_SENTINEL" not in output
    assert "RuntimeError: synthetic failure" in output
    assert "Last 60 installation" not in output


def test_install_log_requires_explicit_preupload_flag(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/install-facer-all.log").write_text("Resolver failed SYNTHETIC_INSTALL_LINE\n")
    step_context(tmp_path, "install: facer editable install", log_name="install-facer-all.log")
    print_diagnostics(tmp_path)
    assert "SYNTHETIC_INSTALL_LINE" not in capsys.readouterr().out
    print_diagnostics(tmp_path, allow_install_logs=True)
    assert "Resolver failed SYNTHETIC_INSTALL_LINE" in capsys.readouterr().out


def test_install_failure_prints_raw_tail_before_cleanup_without_upload(tmp_path, monkeypatch, capsys):
    session = tmp_path / "dt-pixel3dmm-session-synthetic"
    session.mkdir()
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(upload=lambda *args, **kwargs: pytest.fail("Install failed before upload"))
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(notebook_session.tempfile, "mkdtemp", lambda **kwargs: str(session))
    monkeypatch.setattr(notebook_session, "copy_flame_archives", lambda *args: None)

    def install(cache, architecture, root):
        (root / "logs").mkdir()
        (root / "logs/install-pipnet-all.log").write_text("Compiler failed SYNTHETIC_BUILD_LINE\n")
        step_context(root, "install: PIPNet nms build", log_name="install-pipnet-all.log")
        raise subprocess.CalledProcessError(1, ["synthetic-build"])

    monkeypatch.setattr(notebook_session, "install", install)
    monkeypatch.setattr(notebook_session, "cleanup_session", lambda root: cleanup_session(root, tmp_path))
    with pytest.raises(subprocess.CalledProcessError):
        notebook_session.run_session(tmp_path / "cache", {}, "2020", "7.5", 1, 1, 1, prepare=True)
    output = capsys.readouterr().out
    assert "Failed step: install: PIPNet nms build" in output
    assert "Compiler failed SYNTHETIC_BUILD_LINE" in output
    assert not session.exists()


def test_all_twelve_view_names_are_accepted_in_canonical_order():
    names = [view + ".jpg" for view in reversed(VIEW_ORDER)]
    assert list(upload_views(names)) == list(VIEW_ORDER)
    assert len(upload_views(names)) == 12
    assert list(upload_views(["extra_8.jpeg", "front.png", "extra_2.JPG"])) == ["front", "extra_2", "extra_8"]
    with pytest.raises(ValueError, match="one image"):
        upload_views(["front.png", "extra_1.jpg", "extra_1.png"])
    with pytest.raises(ValueError, match="required"):
        upload_views(["extra_1.png"])


@pytest.mark.parametrize("name", ["extra_0.png", "extra_9.png", "extra_01.png", "extra_1_more.png",
                                  "other.png", "extra_1.gif", "../front.jpg", "folder/front.png", "folder\\front.png"])
def test_extra_names_and_path_components_fail_closed(name):
    with pytest.raises(ValueError, match="Use front"):
        upload_views(["front.png", name])


def test_joint_budget_keeps_two_view_baseline_and_scales_coverage():
    two = fit_budget(2, 1500, 1500, 1)
    nine = fit_budget(9, 1500, 1500, 1)
    twelve = fit_budget(12, 1500, 1500, 1)
    assert [two["jointIters"], nine["jointIters"], twelve["jointIters"]] == [1500, 6750, 9000]
    assert all(value["expectedJointUpdatesPerView"] == 750 for value in (two, nine, twelve))
    assert nine["sampleWorkUnits"] / two["sampleWorkUnits"] == 4.5
    assert twelve["sampleWorkUnits"] / two["sampleWorkUnits"] == 6
    assert fit_budget(12, 1500, 1500, 2)["jointIters"] == 4500
    assert fit_budget(1, 1500, 1500, 1)["expectedJointUpdatesPerView"] == 0
    with pytest.raises(ValueError, match="1 through 12"):
        fit_budget(13, 1500, 1500, 1)


def synthetic_crop(folder):
    (folder / "cropped").mkdir(parents=True, exist_ok=True)
    (folder / "PIPnet_landmarks").mkdir(exist_ok=True)
    (folder / "cropped/00000.jpg").write_bytes(b"synthetic-crop-placeholder")
    np.save(folder / "PIPnet_landmarks/00000.npy", np.full((98, 2), 0.5))
    np.save(folder / "crop_ymin_ymax_xmin_xmax.npy", np.array([0, 30, 5, 35]))


def test_crop_retry_removes_partial_artifacts_and_preserves_coordinates(tmp_path, monkeypatch):
    folder = tmp_path / "preprocessed/left"
    calls = []

    def crop(root, script, args, step, view, cwd):
        step_context(root, step, view)
        calls.append(step)
        if len(calls) == 1:
            synthetic_crop(folder)
            np.save(folder / "PIPnet_landmarks/00000.npy", np.zeros((98, 2)))
        else:
            assert not folder.exists()
            synthetic_crop(folder)

    monkeypatch.setattr(worker, "run_step", crop)
    method = worker.crop_view(tmp_path, tmp_path / "source", tmp_path / "inputs/left", "left")
    assert calls == ["cropping_landmarks", "cropping_landmarks_retry"]
    assert method == "faceboxes_landmark_0.75_retry"
    np.testing.assert_array_equal(np.load(folder / "crop_ymin_ymax_xmin_xmax.npy"), [0, 30, 5, 35])
    assert json.loads((tmp_path / "warnings.json").read_text())[0]["action"] == "used_relaxed_crop"


def test_crop_retry_never_catches_cuda_or_install_errors(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["synthetic-CUDA-error"])

    monkeypatch.setattr(worker, "run_step", fail)
    with pytest.raises(subprocess.CalledProcessError):
        worker.crop_view(tmp_path, tmp_path / "source", tmp_path / "inputs/left", "left")


def test_relaxed_threshold_is_scoped_to_retry_subprocess(tmp_path, monkeypatch):
    flags = []
    monkeypatch.setattr(worker, "run", lambda command, root, **kwargs: flags.append(kwargs["env"]["DT_PROFILE_CROP_RETRY"]))
    worker.run_step(tmp_path, tmp_path / "run_cropping.py", [], "cropping_landmarks", "left")
    worker.run_step(tmp_path, tmp_path / "run_cropping.py", [], "cropping_landmarks_retry", "left")
    worker.run_step(tmp_path, tmp_path / "network.py", [], "normals_prediction")
    assert flags == ["0", "1", "0"]


def test_preprocess_orders_and_compacts_views_after_optional_skips(tmp_path, monkeypatch, capsys):
    from PIL import Image

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    # Uniform synthetic images; no people, model weights or real photographs.
    names = [("extra_8", 6), ("left", 3), ("extra_2", 5), ("back", 4), ("right", 2), ("front", 1)]
    for view, color in names:
        Image.new("RGB", (40, 30), (color, color, color)).save(uploads / (view + ".png"))
    insightface = ModuleType("insightface")
    app = ModuleType("insightface.app")
    app.FaceAnalysis = lambda **kwargs: SimpleNamespace(
        prepare=lambda **kwargs: None, get=lambda image: [] if image[0, 0, 0] == 4 else ["synthetic-face"])
    insightface.app = app
    monkeypatch.setitem(sys.modules, "insightface", insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", app)
    monkeypatch.setenv("DT_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DT_INSIGHTFACE_ROOT", str(tmp_path / "detector-placeholder"))

    def stage(root, script, args, step, view="all", cwd=None):
        step_context(root, step, view)
        folder = root / "preprocessed" / view
        if step.startswith("cropping_landmarks"):
            if view in {"left", "extra_8"}:
                raise DetectionError("no_usable_crop_or_landmarks")
            synthetic_crop(folder)
        elif step == "mica":
            (folder / "mica/00000").mkdir(parents=True)
            np.save(folder / "mica/00000/identity.npy", np.zeros(300))
        elif step == "segmentation":
            (folder / "seg_og").mkdir()
            (folder / "seg_og/00000.png").write_bytes(b"synthetic-seg-placeholder")
        else:
            prediction = "normals" if step == "normals_prediction" else "uv_map"
            destination = root / "preprocessed/head/p3dmm" / prediction
            destination.mkdir(parents=True)
            for index in range(3):
                (destination / f"{index:05d}.png").write_bytes(b"synthetic-prediction-placeholder")

    monkeypatch.setattr(worker, "run_step", stage)
    included = worker.preprocess(tmp_path)
    assert [item["view"] for item in included] == ["front", "right", "extra_2"]
    assert [item["frame"] for item in included] == [0, 1, 2]
    assert all(item["originalSizeWH"] == [40, 30] for item in included)
    assert included[-1]["cropBoundsYminYmaxXminXmax"] == [0, 30, 5, 35]
    skipped = json.loads((tmp_path / "view_map.json").read_text())["skipped"]
    assert next(item for item in skipped if item["view"] == "back")["reason"] == "no_face_detected"
    assert next(item for item in skipped if item["view"] == "extra_8")["reason"] == "no_usable_crop_or_landmarks"
    from diagnostics import print_warnings
    print_warnings(tmp_path)
    output = capsys.readouterr().out
    assert "skipped extra_8" in output and "skipped left" in output


def synthetic_checkpoint(expression):
    return {"img_size": [256, 256],
            "camera": {"fl": [[2]], "pp": [[0, 0]], "R_base_0": np.eye(3)[None], "t_base_0": [[0, 0, -2]]},
            "flame": {"R_rotation_matrix": np.eye(3)[None], "t": [[0, 0, 0]], "exp": expression},
            "joint_transforms": np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()}


def test_expression_proxy_prefers_neutral_and_rejects_bad_coefficients():
    checkpoints = [synthetic_checkpoint(np.zeros((1, 100))),
                   synthetic_checkpoint(np.ones((1, 100))),
                   synthetic_checkpoint(-np.ones((1, 100)))]
    values = expression_indicators(checkpoints)
    assert [item["neutralityRank"] for item in values] == [1, 2, 2]
    assert values[0]["mouthTexturePreference"] > values[1]["mouthTexturePreference"]
    assert values[1]["magnitudeL2"] == 10 and values[1]["magnitudeRMS"] == 1
    assert values[1]["smileProxyOnly"] and values[1] == values[2]
    with pytest.raises(ValueError, match="Invalid fitted expression"):
        expression_indicators([synthetic_checkpoint(np.full(100, float("nan")))])


def test_additive_camera_export_loads_with_current_front_right_consumer(tmp_path):
    source = Path(__file__).resolve().parents[2] / "head/flame/flamehead/camera.py"
    spec = importlib.util.spec_from_file_location("synthetic_flame_consumer_camera", source)
    consumer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = consumer
    try:
        spec.loader.exec_module(consumer)
        document = camera_document([{"view": "left", "reason": "no_face_detected"}])
        checkpoint = synthetic_checkpoint(np.zeros((1, 100)))
        expression = expression_indicators([checkpoint])[0]
        for index, name in enumerate(("front", "right", "extra_1", "extra_8")):
            item = {"view": name, "frame": index, "originalSizeWH": [640, 480],
                    "cropBoundsYminYmaxXminXmax": [20, 420, 100, 500], "cropMethod": "faceboxes_landmark_0.99"}
            document["views"][name] = camera_view(item, checkpoint, expression)
        path = tmp_path / "synthetic-cameras.json"
        path.write_text(json.dumps(document))
        cameras = consumer.load_cameras(path)
        assert set(cameras) == {"front", "right"}
        np.testing.assert_allclose(cameras["front"].to_original([[128, 128]]), [[300, 220]])
        assert document["schema"] == "dt-flame-head-cameras/1"
        extra = document["views"]["extra_8"]
        assert extra["fittedMesh"] == "fitted_views/extra_8.ply" and extra["overlay"] == "overlays/extra_8.png"
        assert extra["originalSizeWH"] == [640, 480] and extra["cropBoundsYminYmaxXminXmax"] == [20, 420, 100, 500]
        np.testing.assert_allclose(extra["worldToCamera"], document["views"]["front"]["worldToCamera"])
    finally:
        sys.modules.pop(spec.name, None)
