"""Colab-only installation adapter. Downloads licensed sources/assets into the VM.

No upstream code, model, credentials or personal data belongs in this repository.
Sources checked 2026-10-02; full CUDA/Colab execution remains untested.
"""

from pathlib import Path
import json
import hashlib
import os
import shutil
import signal
import subprocess
import tarfile
import traceback
import urllib.request

from io_utils import safe_extract, stage_flame
from diagnostics import step_context
from downloads import HF_REVISION, WEIGHTS, download_weight, fetch_https, plausible

PIXEL_REVISION = "fcd1fa973c7715b02a8948dfc679dff53cf85924"
MICA_REVISION = "af22e7a5810d474bc28a1433db533723d6bd2b07"
FACER_REVISION = "ddd35c76ff840174b8a5403ad1c1255e37b8782b"
PIPNET_REVISION = "b9eab58816437403a34aa5bc3adeafe5081fd36b"
PYTORCH3D_REVISION = "75ebeeaea0908c5527e7b1e305fbc7681382db47"  # stable
NVDIFFRAST_REVISION = "729261dc64c4241ea36efda84fbf532cc8b425b8"  # v0.3.3


def run(command, root: Path, cwd=None, env=None, step: str = "runtime", view: str = "all") -> None:
    """Keep upstream numeric/debug output in private VM logs, never notebook outputs."""
    effective_env = os.environ if env is None else env
    log_root = Path(effective_env.get("DT_LOG_ROOT", str(root)))
    directory = log_root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / (step + "-" + view + ".log")
    with log.open("a", encoding="utf-8") as stream:
        # Keep the worker and nested stages in one group so interruption kills
        # subprocesses before the notebook deletes their private directories.
        new_group = os.name == "posix" and effective_env.get("DT_WORKER_ACTIVE") != "1"
        process = subprocess.Popen([str(arg) for arg in command], cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=new_group)
        try:
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, [str(arg) for arg in command])
        except BaseException:
            try:
                if new_group:
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()
            raise


def replace_exact(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"Upstream adapter mismatch in {path.name}; refusing silent fallback")
    path.write_text(source.replace(old, new), encoding="utf-8")


def replace_once_or_done(path: Path, old: str, new: str) -> None:
    if new in path.read_text(encoding="utf-8"):
        return
    replace_exact(path, old, new)


def apply_runtime_patches(root: Path) -> None:
    source = root / "pixel3dmm"
    replace_once_or_done(source / "src/pixel3dmm/tracking/tracker.py", "COMPILE = True", "COMPILE = False")
    replace_once_or_done(source / "scripts/network_inference.py", "model = model.cuda()", "model = model.eval().float().cuda()")


def cache_signature(architecture: str) -> dict:
    return {"version": 2, "architecture": architecture, "pixel3dmm": PIXEL_REVISION,
            "MICA": MICA_REVISION, "facer": FACER_REVISION, "PIPNet": PIPNET_REVISION,
            "pytorch3d": PYTORCH3D_REVISION, "nvdiffrast": NVDIFFRAST_REVISION,
            "torch": "2.7.1", "cuda": "11.8", "python": "3.9", "hf_weights": HF_REVISION}


def require_cache(root: Path, architecture: str, version: str = "2020") -> None:
    marker = root / "install_complete.json"
    if not marker.is_file():
        raise RuntimeError("Compatible installed cache missing; run the install-and-fit cell first")
    if json.loads(marker.read_text()) != cache_signature(architecture):
        raise RuntimeError("Installed cache is incompatible; run final privacy cleanup with DELETE_VM_CACHE=True, then install-and-fit")
    source = root / "pixel3dmm"
    assets = source / "src/pixel3dmm/preprocessing/MICA/data"
    required = [root / "env/bin/python", root / "FLAME2020.zip", assets / "FLAME2020/generic_model.pkl",
                source / "pretrained_weights/uv.ckpt", source / "pretrained_weights/normals.ckpt",
                assets / "pretrained/mica.tar"]
    if version == "2023":
        required.extend([root / "FLAME2023.zip", assets / "FLAME2023/flame2023_no_jaw.pkl"])
    if not all(path.is_file() for path in required):
        raise RuntimeError("Cache assets missing; run the install-and-fit cell to prepare them")


def runtime_environment(root: Path, architecture: str, log_root: Path) -> dict:
    prefix = root / "env"
    env = os.environ.copy()
    env.update({
        "PATH": str(prefix / "bin") + os.pathsep + env["PATH"],
        "CONDA_PREFIX": str(prefix), "CUDA_HOME": str(prefix),
        "LD_LIBRARY_PATH": str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        "CC": str(prefix / "bin/x86_64-conda-linux-gnu-gcc"),
        "CXX": str(prefix / "bin/x86_64-conda-linux-gnu-g++"),
        "CUDAHOSTCXX": str(prefix / "bin/x86_64-conda-linux-gnu-g++"),
        "TORCH_CUDA_ARCH_LIST": architecture, "MAX_JOBS": "2",
        "TORCH_HOME": str(root / "cache/torch"), "HF_HOME": str(root / "cache/hf"),
        "XDG_CACHE_HOME": str(root / "cache"), "MPLCONFIGDIR": str(root / "cache/matplotlib"),
        "TORCH_EXTENSIONS_DIR": str(root / ("cache/extensions-sm" + architecture.replace(".", ""))),
        "PIP_CACHE_DIR": str(root / "cache/pip"), "DT_INSIGHTFACE_ROOT": str(root / "insightface"),
        "WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "TORCHDYNAMO_DISABLE": "1", "XFORMERS_DISABLED": "1",
        "MPLBACKEND": "Agg", "DT_PYTHON": str(prefix / "bin/python"), "DT_SESSION": str(root),
        "DT_CACHE_ROOT": str(root), "DT_LOG_ROOT": str(log_root), "PYTHONUNBUFFERED": "1",
    })
    constraints = root / "constraints.txt"
    if constraints.is_file():
        env["PIP_CONSTRAINT"] = str(constraints)
    return env


class InstallSteps:
    """One versioned checkpoint and log per costly operation; no photo inputs."""

    def __init__(self, root: Path, architecture: str, log_root: Path, env: dict):
        self.root, self.log_root, self.env = root, log_root, env
        self.signature = cache_signature(architecture)
        self.current = "install-initialization"

    def log(self, message: str) -> None:
        directory = self.log_root / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (self.current + "-all.log")).open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    def command(self, command, cwd=None) -> None:
        run(command, self.root, cwd=cwd, env=self.env, step=self.current)

    def perform(self, identifier: str, label: str, operation, required=()) -> None:
        marker = self.root / "done" / (identifier + ".json")
        expected = {"signature": self.signature, "step": identifier}
        try:
            completed = json.loads(marker.read_text(encoding="utf-8")) == expected
        except (OSError, ValueError):
            completed = False
        if completed and all(path.exists() for path in required):
            print("Reusing install step: " + label)
            return
        self.current = "install-" + identifier
        step_context(self.log_root, "install: " + label, log_name=self.current + "-all.log")
        print("Installing step: " + label)
        try:
            operation()
            if not all(path.exists() for path in required):
                raise RuntimeError("Install step output missing: " + label)
        except BaseException:
            directory = self.log_root / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / (self.current + "-all.log")).open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            raise
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(".json.partial")
        temporary.write_text(json.dumps(expected), encoding="utf-8")
        temporary.replace(marker)

    def clone(self, repo: str, revision: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not (destination / ".git").is_dir():
            self.command(["git", "clone", "https://github.com/" + repo + ".git", destination])
        # Discard our tracked patches on a failed/stale step before reapplying.
        self.command(["git", "-C", destination, "restore", "--worktree", "."])
        self.command(["git", "-C", destination, "checkout", "--detach", revision])

    def weight(self, name: str, destination: Path) -> None:
        def fetch_gdown(url, temporary):
            self.command([self.root / "env/bin/python", "-m", "gdown", "--no-cookies", "--fuzzy", url, "-O", temporary])

        def operation():
            download_weight(name, destination, self.root,
                            {"https": fetch_https, "gdown": fetch_gdown}, self.log)

        # Invalid/truncated payloads invalidate even an otherwise matching marker.
        marker = self.root / "done" / ("download-" + name.replace(".", "-") + ".json")
        if not plausible(destination, WEIGHTS[name]["minimum"]):
            marker.unlink(missing_ok=True)
        self.perform("download-" + name.replace(".", "-"), "download " + name, operation, [destination])


REPLACEMENT_HASHES = {
    "farl.py": "fc5e202baf47347f22b60c010f59632fa27d7ce186393aa98d58748a44c43695",
    "facer_transform.py": "34950df2a248132ea3d4e8cf63d497f45076930877c9a53a005c5976078c2bed",
    "mica_demo.py": "06c81c28623afcce95677da6088879bb2e2d40494d086ff7147173f98c015340",
    "mica.py": "5b1c13e5b3cfb9f6bc89bda81c0743a5548d261347eab0001d192d548695993e",
}


def verify_upstream(source: Path) -> None:
    """Check pinned adapter inputs without executing either upstream shell script."""
    script = (source / "install_preprocessing_pipeline.sh").read_text(encoding="utf-8")
    assumptions = ["FacePerceiver/facer.git", "Zielon/MICA.git", "jhb86253817/PIPNet.git",
                   "facer/face_parsing/farl.py", "facer/transform.py", "micalib/models/mica.py",
                   "FaceBoxesV2/utils", "pip_32_16_60_r18_l2_l1_10_1_nb10"]
    assumptions.extend(url.split("/d/")[1].split("/")[0]
                       for name in ("uv.ckpt", "normals.ckpt", "epoch59.pth")
                       for transport, url in WEIGHTS[name]["sources"] if transport == "gdown")
    if any(value not in script for value in assumptions):
        raise RuntimeError("Pinned preprocessing installer layout/weight IDs differ from adapter assumptions")
    replacement = source / "src/pixel3dmm/preprocessing/replacement_code"
    for name, expected in REPLACEMENT_HASHES.items():
        path = replacement / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Pinned replacement_code mismatch: " + name)


def copy_replacements(replacement: Path, destination: Path, pairs) -> None:
    for name, relative in pairs:
        target = destination / relative
        if not target.is_file():
            raise RuntimeError("Pinned upstream destination missing: " + relative)
        shutil.copyfile(replacement / name, target)


def extract_insightface(archive: Path) -> None:
    target = archive.with_suffix("")
    safe_extract(archive, target)
    nested = target / target.name
    if nested.is_dir():
        for child in nested.iterdir():
            shutil.move(str(child), target / child.name)
        nested.rmdir()
    if not any(target.glob("*.onnx")):
        raise RuntimeError("Insightface archive contains no ONNX models: " + archive.name)


def patch_mica_detector(mica: Path) -> None:
    detector = mica / "utils/landmark_detector.py"
    replace_once_or_done(detector, "FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])",
                         "FaceAnalysis(name='antelopev2', root=os.environ['DT_INSIGHTFACE_ROOT'], providers=['CPUExecutionProvider'])")
    text = detector.read_text(encoding="utf-8")
    if not text.startswith("import os\n"):
        detector.write_text("import os\n" + text, encoding="utf-8")


def preprocessing_install(steps: InstallSteps) -> None:
    """Same order/effects as upstream, using only env Python and checked steps."""
    source = steps.root / "pixel3dmm"
    base = source / "src/pixel3dmm/preprocessing"
    replacement = base / "replacement_code"
    python = steps.root / "env/bin/python"
    steps.perform("verify-upstream", "verify pinned upstream adapter", lambda: verify_upstream(source))

    facer = base / "facer"
    steps.perform("facer-source", "facer clone and replacements", lambda: (
        steps.clone("FacePerceiver/facer", FACER_REVISION, facer),
        copy_replacements(replacement, facer, [("farl.py", "facer/face_parsing/farl.py"),
                                              ("facer_transform.py", "facer/transform.py")])), [facer / "setup.py"])
    steps.perform("facer", "facer editable install",
                  lambda: steps.command([python, "-m", "pip", "install", "-e", facer]))

    mica = base / "MICA"
    steps.perform("mica", "MICA clone and replacements", lambda: (
        steps.clone("Zielon/MICA", MICA_REVISION, mica),
        copy_replacements(replacement, mica, [("mica_demo.py", "demo.py"),
                                             ("mica.py", "micalib/models/mica.py")]),
        patch_mica_detector(mica)), [mica / "demo.py", mica / "utils/landmark_detector.py"])
    # In-process asset orchestration; no install.sh and no FLAME credentials.
    steps.weight("mica.tar", mica / "data/pretrained/mica.tar")
    for name in ("antelopev2", "buffalo_l"):
        archive = steps.root / "insightface/models" / (name + ".zip")
        steps.weight(archive.name, archive)
        steps.perform("extract-" + name, "extract " + archive.name,
                      lambda path=archive: extract_insightface(path), [archive.with_suffix("")])

    pipnet = base / "PIPNet"
    steps.perform("pipnet-source", "PIPNet clone", lambda: steps.clone("jhb86253817/PIPNet", PIPNET_REVISION, pipnet),
                  [pipnet / "FaceBoxesV2/utils/build.py"])

    def build_nms():
        directory = pipnet / "FaceBoxesV2/utils"
        make = (directory / "make.sh").read_text(encoding="utf-8")
        build = (directory / "build.py").read_text(encoding="utf-8")
        if "python3 build.py build_ext --inplace" not in make or "nms/cpu_nms.pyx" not in build:
            raise RuntimeError("Pinned PIPNet nms build assumptions differ")
        steps.command([python, "build.py", "build_ext", "--inplace"], cwd=directory)
        if not any((directory / "nms").glob("cpu_nms*.so")):
            raise RuntimeError("PIPNet nms extension was not produced")

    steps.perform("pipnet", "PIPNet nms build", build_nms)
    steps.weight("epoch59.pth", pipnet / "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth")
    for name in ("uv.ckpt", "normals.ckpt"):
        steps.weight(name, source / "pretrained_weights" / name)


def install(root: Path, architecture: str, log_root: Path) -> tuple:
    """Checkpoints survive failed installs; no credentials or photo data enter logs."""
    env = runtime_environment(root, architecture, log_root)
    if (root / "install_complete.json").is_file():
        step_context(log_root, "install: cache validation", log_name="install-cache-validation-all.log")
        require_cache(root, architecture)
        assets = root / "pixel3dmm/src/pixel3dmm/preprocessing/MICA/data"
        if (root / "FLAME2023.zip").is_file() and not (assets / "FLAME2023/flame2023_no_jaw.pkl").is_file():
            stage_flame(root / "FLAME2023.zip", assets, "2023")
        apply_runtime_patches(root)
        print("Reusing installed VM cache; no environment install or weight downloads.")
        return root / "env/bin/python", env
    print("Installing isolated Python 3.9 / CUDA 11.8; completed sub-steps are reused.")
    steps = InstallSteps(root, architecture, log_root, env)
    prefix = root / "env"
    python = prefix / "bin/python"

    def create_env():
        archive = root / "micromamba.tar.bz2"
        urllib.request.urlretrieve("https://micro.mamba.pm/api/micromamba/linux-64/2.0.5", archive)
        with tarfile.open(archive) as bundle:
            member = bundle.getmember("bin/micromamba")
            if not member.isfile():
                raise RuntimeError("Unexpected micromamba package")
            with bundle.extractfile(member) as binary, (root / "micromamba").open("wb") as output:
                shutil.copyfileobj(binary, output)
        mamba = root / "micromamba"
        mamba.chmod(0o755)
        packages = ["cuda-nvcc", "cuda-cccl", "cuda-cudart", "cuda-cudart-dev", "libcusparse", "libcusparse-dev",
                    "libcublas", "libcublas-dev", "libcurand", "libcurand-dev", "libcusolver", "libcusolver-dev"]
        operation = "install" if python.is_file() else "create"
        steps.command([mamba, operation, "-y", "--no-rc", "-r", root / "mamba", "-p", prefix,
                       "-c", "conda-forge", "-c", "nvidia/label/cuda-11.8.0", "python=3.9", "pip",
                       "gcc_linux-64=11", "gxx_linux-64=11",
                       *["nvidia/label/cuda-11.8.0::" + package for package in packages]])

    steps.perform("env", "environment create", create_env, [python])

    def apt():
        steps.command(["apt-get", "update", "-qq"])
        steps.command(["apt-get", "install", "-y", "git", "build-essential", "ffmpeg", "libgl1",
                       "libegl1-mesa-dev", "libgles2-mesa-dev"])

    steps.perform("apt", "system packages", apt)

    def pip(*args):
        steps.command([python, "-m", "pip", "install", *args])

    steps.perform("bootstrap", "pip and numerical build tools", lambda: pip(
        "pip==25.1.1", "setuptools==68.2.2", "wheel", "numpy==1.23.5", "scipy==1.11.4", "Cython<3"))
    steps.perform("torch", "torch cu118", lambda: pip(
        "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu118"))
    source = root / "pixel3dmm"
    steps.perform("pixel-source", "Pixel3DMM clone", lambda: steps.clone("SimonGiebenhain/pixel3dmm", PIXEL_REVISION, source),
                  [source / "requirements.txt"])
    constraints = root / "constraints.txt"
    constraints.write_text(
        "numpy==1.23.5\nscipy==1.11.4\nscikit-image==0.22.0\nopencv-python==4.10.0.84\n"
        "matplotlib==3.7.5\npytorch-lightning==2.5.2\ntorchmetrics==1.7.2\n"
        "librosa==0.10.2.post1\nnumba==0.60.0\ntimm==1.0.15\n"
        "torch==2.7.1\ntorchvision==0.22.1\n", encoding="utf-8")
    env["PIP_CONSTRAINT"] = str(constraints)
    requirements = root / "requirements.txt"
    requirements.write_text((source / "requirements.txt").read_text().replace("numpy==1.23", "numpy==1.23.5"))

    def install_requirements():
        pip("chumpy==0.70", "--no-build-isolation")
        pip("-r", str(requirements), "gdown==5.2.0", "trimesh>=4.6,<5", "onnxruntime==1.19.2", "ninja", "iopath")

    steps.perform("requirements", "requirements", install_requirements)
    for identifier, repo, revision in [("pytorch3d", "facebookresearch/pytorch3d", PYTORCH3D_REVISION),
                                       ("nvdiffrast", "NVlabs/nvdiffrast", NVDIFFRAST_REVISION)]:
        steps.perform(identifier, identifier + " build", lambda repo=repo, revision=revision:
                      pip("git+https://github.com/" + repo + ".git@" + revision, "--no-build-isolation"))
    steps.perform("pixel-editable", "Pixel3DMM editable install", lambda: pip("-e", str(source)))
    preprocessing_install(steps)
    assets = source / "src/pixel3dmm/preprocessing/MICA/data"

    def stage_models():
        stage_flame(root / "FLAME2020.zip", assets, "2020")
        if (root / "FLAME2023.zip").is_file():
            stage_flame(root / "FLAME2023.zip", assets, "2023")

    # stage_model() also restages a changed Drive archive before install().
    steps.perform("flame", "stage local FLAME", stage_models, [assets / "FLAME2020/generic_model.pkl"])
    step_context(log_root, "install: runtime patches", log_name="install-runtime-patches-all.log")
    steps.current = "install-runtime-patches"
    apply_runtime_patches(root)

    def verify():
        steps.command([python, "-m", "pip", "check"])
        steps.command([python, "-c", "import torch, pytorch3d, nvdiffrast.torch, facer, insightface; assert torch.cuda.is_available()"])

    steps.perform("verify", "dependency and CUDA checks", verify)
    completion = root / "install_complete.json"
    temporary = completion.with_suffix(".json.partial")
    temporary.write_text(json.dumps(cache_signature(architecture)), encoding="utf-8")
    temporary.replace(completion)
    return python, env
