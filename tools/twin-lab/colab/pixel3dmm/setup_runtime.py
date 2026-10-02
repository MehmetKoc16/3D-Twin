"""Colab-only installation adapter. Downloads licensed sources/assets into the VM.

No upstream code, model, credentials or personal data belongs in this repository.
Sources checked 2026-10-02; full CUDA/Colab execution remains untested.
"""

from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import urllib.request

from io_utils import safe_extract, stage_flame

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
    return {"version": 1, "architecture": architecture, "pixel3dmm": PIXEL_REVISION,
            "MICA": MICA_REVISION, "facer": FACER_REVISION, "PIPNet": PIPNET_REVISION,
            "pytorch3d": PYTORCH3D_REVISION, "nvdiffrast": NVDIFFRAST_REVISION,
            "torch": "2.7.1", "cuda": "11.8", "python": "3.9"}


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


def mica_assets(root: Path) -> None:
    """Called instead of upstream's credential-based MICA/FLAME install script."""
    os.environ["DT_WORKER_ACTIVE"] = "1"
    source = root / "pixel3dmm"
    mica = source / "src/pixel3dmm/preprocessing/MICA"
    models = root / "insightface/models"
    destinations = [
        ("1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt", mica / "data/pretrained/mica.tar"),
        ("16PWKI_RjjbE4_kqpElG-YFqe8FpXjads", models / "antelopev2.zip"),
        ("1navJMy0DTr1_DHjLWu1i48owCPvXWfYc", models / "buffalo_l.zip"),
    ]
    for identifier, path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "gdown", identifier, "-O", path], root)
        if path.suffix == ".zip":
            target = path.with_suffix("")
            safe_extract(path, target)
            # Accommodate either flat or named-folder archives expected by insightface.
            nested = target / target.name
            if nested.is_dir():
                for child in nested.iterdir():
                    shutil.move(str(child), target / child.name)
                nested.rmdir()
    detector = mica / "utils/landmark_detector.py"
    replace_once_or_done(detector, "FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])",
                  "FaceAnalysis(name='antelopev2', root=os.environ['DT_INSIGHTFACE_ROOT'], providers=['CPUExecutionProvider'])")
    text = detector.read_text(encoding="utf-8")
    if not text.startswith("import os\n"):
        detector.write_text("import os\n" + text, encoding="utf-8")


def install(root: Path, architecture: str, log_root: Path) -> tuple:
    """Python 3.9 + CUDA 11.8 isolated from the Colab kernel; no restart needed."""
    env = runtime_environment(root, architecture, log_root)
    if (root / "install_complete.json").is_file():
        require_cache(root, architecture)
        assets = root / "pixel3dmm/src/pixel3dmm/preprocessing/MICA/data"
        if (root / "FLAME2023.zip").is_file() and not (assets / "FLAME2023/flame2023_no_jaw.pkl").is_file():
            stage_flame(root / "FLAME2023.zip", assets, "2023")
        apply_runtime_patches(root)
        print("Reusing installed VM cache; no environment install or weight downloads.")
        return root / "env/bin/python", env
    print("Installing isolated Python 3.9 / CUDA 11.8; native builds can take tens of minutes.")
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
    prefix = root / "env"
    cuda_packages = ["cuda-nvcc", "cuda-cccl", "cuda-cudart", "cuda-cudart-dev",
                     "libcusparse", "libcusparse-dev", "libcublas", "libcublas-dev",
                     "libcurand", "libcurand-dev", "libcusolver", "libcusolver-dev"]
    operation = "install" if (prefix / "bin/python").is_file() else "create"
    run([mamba, operation, "-y", "--no-rc", "-r", root / "mamba", "-p", prefix,
         "-c", "conda-forge", "-c", "nvidia/label/cuda-11.8.0",
         "python=3.9", "pip", "gcc_linux-64=11", "gxx_linux-64=11",
         *["nvidia/label/cuda-11.8.0::" + package for package in cuda_packages]], root, env=env)
    python = prefix / "bin/python"

    def pip(*args):
        run([python, "-m", "pip", "install", *args], root, env=env)

    run(["apt-get", "update", "-qq"], root, env=env)
    run(["apt-get", "install", "-y", "git", "build-essential", "ffmpeg", "libgl1",
         "libegl1-mesa-dev", "libgles2-mesa-dev"], root, env=env)
    pip("pip==25.1.1", "setuptools==68.2.2", "wheel", "numpy==1.23.5", "scipy==1.11.4", "Cython<3")
    pip("torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
        "--index-url", "https://download.pytorch.org/whl/cu118")
    source = root / "pixel3dmm"
    if not (source / ".git").is_dir():
        run(["git", "clone", "https://github.com/SimonGiebenhain/pixel3dmm.git", source], root, env=env)
    else:
        run(["git", "-C", source, "restore", "--worktree", "."], root, env=env)
    run(["git", "-C", source, "checkout", "--detach", PIXEL_REVISION], root, env=env)
    constraints = root / "constraints.txt"
    constraints.write_text(
        "numpy==1.23.5\nscipy==1.11.4\nscikit-image==0.22.0\nopencv-python==4.10.0.84\n"
        "matplotlib==3.7.5\npytorch-lightning==2.5.2\ntorchmetrics==1.7.2\n"
        "librosa==0.10.2.post1\nnumba==0.60.0\ntimm==1.0.15\n"
        "torch==2.7.1\ntorchvision==0.22.1\n", encoding="utf-8")
    env["PIP_CONSTRAINT"] = str(constraints)
    requirements = root / "requirements.txt"
    requirements.write_text((source / "requirements.txt").read_text().replace("numpy==1.23", "numpy==1.23.5"))
    pip("chumpy==0.70", "--no-build-isolation")
    pip("-r", str(requirements), "gdown==5.2.0", "trimesh>=4.6,<5", "onnxruntime==1.19.2", "ninja", "iopath")
    for repo, revision in [("facebookresearch/pytorch3d", PYTORCH3D_REVISION), ("NVlabs/nvdiffrast", NVDIFFRAST_REVISION)]:
        pip("git+https://github.com/" + repo + ".git@" + revision, "--no-build-isolation")
    pip("-e", str(source))

    # Execute upstream's installer after narrowly adapting its VM copy. Do not
    # execute the credential prompts or its FLAME downloads at any point.
    script = source / "install_preprocessing_pipeline.sh"
    for old, new in [
        ("git clone git@github.com:FacePerceiver/facer.git", "if [ ! -d facer/.git ]; then git clone https://github.com/FacePerceiver/facer.git; fi\ngit -C facer checkout --detach " + FACER_REVISION),
        ("git clone git@github.com:Zielon/MICA.git", "if [ ! -d MICA/.git ]; then git clone https://github.com/Zielon/MICA.git; fi\ngit -C MICA checkout --detach " + MICA_REVISION),
        ("git clone https://github.com/jhb86253817/PIPNet.git", "if [ ! -d PIPNet/.git ]; then git clone https://github.com/jhb86253817/PIPNet.git; fi\ngit -C PIPNet checkout --detach " + PIPNET_REVISION),
    ]:
        replace_exact(script, old, new)
    script.write_text("#!/bin/bash\nset -euo pipefail\n" + script.read_text().replace("gdown --id ", "gdown ").replace("mkdir ", "mkdir -p "))
    replacement = source / "src/pixel3dmm/preprocessing/replacement_code/install_mica_download_flame.sh"
    replacement.write_text('#!/bin/bash\nset -euo pipefail\nexec "$DT_PYTHON" "$DT_SESSION/helpers/setup_runtime.py" mica-assets "$DT_SESSION"\n')
    run(["bash", script], root, cwd=source, env=env)
    assets = source / "src/pixel3dmm/preprocessing/MICA/data"
    stage_flame(root / "FLAME2020.zip", assets, "2020")
    if (root / "FLAME2023.zip").exists():
        stage_flame(root / "FLAME2023.zip", assets, "2023")
    apply_runtime_patches(root)
    run([python, "-m", "pip", "check"], root, env=env)
    run([python, "-c", "import torch, pytorch3d, nvdiffrast.torch, facer, insightface; assert torch.cuda.is_available()"], root, env=env)
    (root / "install_complete.json").write_text(json.dumps(cache_signature(architecture)), encoding="utf-8")
    return python, env


if __name__ == "__main__":
    if sys.argv[1] != "mica-assets":
        raise ValueError("Unknown installation operation")
    mica_assets(Path(sys.argv[2]))
