"""Colab setup and privacy helpers, embedded in the portable notebook."""

import gc
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TRELLIS_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CUMESH_REVISION = "12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM_REVISION = "6dd94a859c26ee8246888502eada3dd8ad85532e"
SPARSE_DECODER_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
BUILD_ROOT = Path("/content/dt-trellis2-software")
ENV_PYTHON = BUILD_ROOT / "venv/bin/python"
TRELLIS_SOURCE = BUILD_ROOT / "TRELLIS.2"


def run_command(arguments, **kwargs):
    # No shell interpolation, and every compiler / installer failure stops the cell.
    return subprocess.run([str(item) for item in arguments], check=True, **kwargs)


def pip_install(*arguments):
    run_command([ENV_PYTHON, "-m", "pip", "install", *arguments])


def clone_pinned(url, revision, destination):
    if not destination.exists():
        run_command(["git", "clone", "--filter=blob:none", url, destination])
    run_command(["git", "-C", destination, "checkout", "--detach", revision])
    run_command(["git", "-C", destination, "submodule", "update", "--init", "--recursive"])


def install_software():
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("This pinned stack targets Python 3.10-3.12. Select a Colab runtime with Python 3.12.")
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
    ).strip()
    name, memory_mib = memory.splitlines()[0].rsplit(",", 1)
    print(f"GPU: {name.strip()}, {int(memory_mib)} MiB")
    if int(memory_mib) < 22000:
        raise RuntimeError("TRELLIS.2 requires a 24 GB class GPU. T4/16 GB is unsupported; select L4 or A100, then Run all again.")
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    toolkit = Path("/usr/local/cuda-12.4")
    if not (toolkit / "bin/nvcc").exists():
        import urllib.request

        # Google Colab uses Ubuntu; install the exact upstream-recommended toolkit.
        release = Path("/etc/os-release").read_text()
        distro = "ubuntu2404" if 'VERSION_ID="24.04"' in release else "ubuntu2204"
        keyring = BUILD_ROOT / "cuda-keyring.deb"
        urllib.request.urlretrieve(
            f"https://developer.download.nvidia.com/compute/cuda/repos/{distro}/x86_64/cuda-keyring_1.1-1_all.deb",
            keyring,
        )
        run_command(["dpkg", "-i", keyring])
        run_command(["apt-get", "update", "-qq"])
        run_command(["apt-get", "install", "-y", "cuda-toolkit-12-4", "build-essential", "git", "libjpeg-dev"])
    os.environ.update({
        "CUDA_HOME": str(toolkit), "PATH": f"{toolkit}/bin:" + os.environ["PATH"],
        "MAX_JOBS": "2", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "ATTN_BACKEND": "flash_attn", "SPARSE_ATTN_BACKEND": "flash_attn",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDA_ARCH_LIST": "8.9" if "L4" in name else "8.0",
    })
    if not ENV_PYTHON.exists():
        run_command([sys.executable, "-m", "venv", BUILD_ROOT / "venv"])
    # Isolated subprocess environment avoids replacing Colab kernel packages or restarting it.
    pip_install("--upgrade", "pip", "setuptools", "wheel")
    pip_install("torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124")
    pip_install(
        "numpy<3", "scipy", "pillow", "opencv-python-headless", "ninja", "packaging", "psutil",
        "trimesh>=4.6,<5", "transformers==4.57.3", "easydict", "tqdm", "einops", "safetensors",
        "huggingface_hub<1", "kornia", "timm", "networkx", "fast-simplification", "zstandard", "rembg[cpu]",
    )
    clone_pinned("https://github.com/microsoft/TRELLIS.2.git", TRELLIS_REVISION, TRELLIS_SOURCE)
    clone_pinned("https://github.com/JeffreyXiang/CuMesh.git", CUMESH_REVISION, BUILD_ROOT / "CuMesh")
    clone_pinned("https://github.com/JeffreyXiang/FlexGEMM.git", FLEXGEMM_REVISION, BUILD_ROOT / "FlexGEMM")
    # O-Voxel eagerly imports postprocess, which imports the noncommercial nvdiffrast.
    # Change only its package entry point in this VM; geometry conversion is unaffected.
    (TRELLIS_SOURCE / "o-voxel/o_voxel/__init__.py").write_text(
        '"""Shape-only entry point modified by Dijital Ikiz Colab notebook."""\nfrom . import convert\n', encoding="utf-8"
    )
    pip_install("flash-attn==2.7.3", "--no-build-isolation")
    for source in (BUILD_ROOT / "CuMesh", BUILD_ROOT / "FlexGEMM", TRELLIS_SOURCE / "o-voxel"):
        pip_install(str(source), "--no-build-isolation")
    run_command([ENV_PYTHON, "-c", "import torch, flash_attn, cumesh, flex_gemm; assert torch.cuda.is_available()"])


def download_and_wait(path):
    """Await the files.download browser transfer before deleting its source file.

    files.download itself returns before transfer. Capture its JavaScript promise;
    fail explicitly if Colab changes that API implementation.
    """
    import IPython.display
    from google.colab import files, output
    from unittest.mock import patch

    original_display = IPython.display.display
    instrumented = False

    def display_with_completion(*objects, **kwargs):
        nonlocal instrumented
        changed = []
        for item in objects:
            if isinstance(item, IPython.display.Javascript) and item.data.strip().startswith("download("):
                item = IPython.display.Javascript("window.dtShapeDownload = " + item.data)
                instrumented = True
            changed.append(item)
        return original_display(*changed, **kwargs)

    with patch.object(IPython.display, "display", display_with_completion):
        files.download(str(path))
    if not instrumented:
        raise RuntimeError("Colab download implementation changed; automatic transfer confirmation needs updating.")
    output.eval_js("window.dtShapeDownload.then(() => true)", timeout_sec=600)


def cleanup_session(path):
    if path is None:
        return
    resolved = Path(path).resolve()
    if resolved.parent != Path("/content") or not resolved.name.startswith("dt-shape-session-"):
        raise ValueError("Refusing cleanup outside the notebook's private session directory")
    if resolved.exists():
        shutil.rmtree(resolved)
    gc.collect()
    print("Privacy cleanup complete: uploaded images, cutouts, meshes, metadata and archive removed from the VM.")
