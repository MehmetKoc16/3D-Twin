"""Rebuild the self-contained head notebook from original helper code only."""

from pathlib import Path
from textwrap import dedent
import json

HERE = Path(__file__).resolve().parent


def cell(kind: str, source: str, identifier: str) -> dict:
    result = {"cell_type": kind, "id": identifier, "metadata": {}, "source": dedent(source).strip() + "\n"}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build() -> dict:
    embedded = {name: (HERE / name).read_text(encoding="utf-8") for name in
                ("io_utils.py", "camera.py", "setup_runtime.py", "worker.py", "export_fit.py",
                 "diagnostics.py", "runtime_compat.py", "notebook_session.py")}
    cells = [cell("markdown", """
        # Multi-image FLAME head fit with Pixel3DMM

        Personal non-commercial research workflow, owner decision 2026-10-02.
        Read [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm),
        [MICA](https://github.com/Zielon/MICA) and your [FLAME](https://flame.is.tue.mpg.de/)
        licence before using their assets. Pixel3DMM is CC BY-NC 4.0; MICA,
        insightface models and NVIDIA's renderer have separate restricted terms.
        No models or upstream source are embedded here or redistributed by this notebook.

        Register at FLAME, download **FLAME2020.zip** yourself and put it in your
        Google Drive. Choose **Runtime > Change runtime type > GPU** (T4 supported; L4/A100 have more headroom).
        Edit the parameter cell, then **Runtime > Run all**.
        A separate Python 3.9 / CUDA 11.8 conda environment avoids restarting Colab.
        Installation/native compilation and multi-GB weight downloads can take tens of minutes.
        **The revised fitting/retry path has not been tested on Colab.**

        Drive is mounted with read-only **usage**, not a read-only permission grant:
        Colab's Drive mount can write. This notebook only reads the selected FLAME
        archive(s), copies them into the VM and immediately unmounts Drive.
        Never place/upload your photos in Drive for this workflow. When prompted,
        select de-glassed `front.png`, `left.png`, `right.png`, `back.png` together
        (.jpg/.jpeg also accepted). Left/right mean the subject's left/right profile.
        Back is optional and omitted from fitting if no face is detected.
        Front must preprocess successfully; undetectable profiles/back are skipped with warnings.

        Photos run on Google's VM only, through user-operated uploads. No remote
        inference service or logging service receives them. No previews are displayed.
        Neutral head OBJ/PLY, JSON/NPZ parameters, cameras, fitted-view meshes and PNG
        overlays download as **head_fit.zip**. Keep it locally in
        **user-data/twin/head/flame/** (gitignored); never ship it with the web app.
        This is a head intermediate, not the rigged `twin.glb` bundle.

        Installation, upload, fitting and download share a `try/finally` cleanup.
        Software, weights and FLAME stay in /content/dt-pixel3dmm-cache for retries.
        The final privacy cell can explicitly delete that cache once you finish.
        Save the browser download, then **Runtime > Disconnect and delete runtime**.
        VM deletion is not secure erasure or a promise about Google's retention.
        Clear notebook outputs before sharing. On hard kernel/VM termination,
        cleanup cannot run; delete the runtime yourself.
    """, "instructions"), cell("code", """
        # Parameters: edit before Runtime > Run all.
        FLAME_ZIP_DRIVE_PATH = "/content/drive/MyDrive/flame/FLAME2020.zip"
        FLAME_VERSION = "2020"  # "2020" (default) or "2023" (no-jaw variant)
        # For 2023, point FLAME_ZIP_DRIVE_PATH to FLAME2023.zip AND provide 2020:
        # preprocessing/MICA and tracking landmark/mask assets still require 2020.
        FLAME2020_ZIP_DRIVE_PATH = "/content/drive/MyDrive/flame/FLAME2020.zip"
        ITERS = 1500
        GLOBAL_ITERS = 1500
        MAX_FIT_BATCH_SIZE = 1  # Small joint batches reduce peak memory on T4.
    """, "parameters"), cell("markdown", """
        ## GPU check

        Fails before Drive access or image uploads unless a T4/L4/A100/H100 GPU is attached.
        These GPU choices are notebook policy; no VRAM/performance guarantee is made.
        FLAME 2023 fitting uses `use_flame2023=True` and `ignore_mica=True`.
        The MICA stage still runs with its own FLAME 2020 model.
    """, "gpu-notes"), cell("code", """
        import subprocess

        if FLAME_VERSION not in {"2020", "2023"}:
            raise ValueError("FLAME_VERSION must be 2020 or 2023")
        if any(not isinstance(value, int) or value < 1 for value in (ITERS, GLOBAL_ITERS, MAX_FIT_BATCH_SIZE)):
            raise ValueError("Iteration counts must be positive integers")
        try:
            gpu_summary = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
            ).splitlines()[0]
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as error:
            raise RuntimeError("Select a T4, L4, A100 or H100 Colab GPU runtime") from error
        gpu_name, gpu_memory = gpu_summary.rsplit(",", 1)
        print(f"GPU: {gpu_name.strip()}, {int(gpu_memory)} MiB")
        # Compile native extensions for the attached GPU, including T4 sm_75.
        GPU_ARCHES = {"T4": "7.5", "L4": "8.9", "A100": "8.0", "H100": "9.0"}
        GPU_ARCH = next((arch for name, arch in GPU_ARCHES.items() if name in gpu_name), None)
        if GPU_ARCH is None or int(gpu_memory) < 15000:
            raise RuntimeError("A T4, L4, A100 or H100 GPU runtime is required")
        if GPU_ARCH == "7.5":
            print("T4 detected: eager float32 execution, no bf16; use MAX_FIT_BATCH_SIZE=1 for memory headroom.")
    """, "gpu-check"), cell("code", "EMBEDDED_FILES = " + repr(embedded), "embedded-original-helpers"), cell("markdown", """
        ## Copy FLAME, install, upload, fit and download

        Drive authorization is only needed to read the model zip(s). No FLAME
        username/password is requested. The credential sections of upstream's
        preprocessing installer are replaced in the VM with model-weight downloads.
        Worker stdout/stderr and each preprocessing step go to private session logs.
        Failures print the failed step/view and the last 120 filtered traceback lines
        BEFORE cleanup. Numeric arrays, image bytes and arbitrary debug output are excluded.
        Clear saved outputs before sharing. The installed cache survives success/failure;
        only photos, derived outputs and logs are automatically removed.

        Each view is cropped separately before the joint fit. `is_discontinuous=True`
        disables temporal smoothness; `global_camera=False` fits intrinsics per photo.
        Overlays are at **256x256 on each face crop**, not original-photo coordinates.
        Final shared shape coefficients generate the neutral mesh anew after joint
        fitting: expression/eyelids zero, jaw/neck/eyes/global rotation identity,
        global translation zero. Do not use upstream's earlier canonical mesh.

        Enable browser downloads. The cell waits for `files.download()` to transfer
        the archive into browser memory before deleting the VM source. It cannot
        verify that you saved the browser download to disk.
    """, "session-notes"), cell("code", """
        from pathlib import Path
        import importlib
        import sys

        CACHE_ROOT = Path("/content/dt-pixel3dmm-cache")
        if CACHE_ROOT.is_symlink() or CACHE_ROOT.resolve().parent != Path("/content"):
            raise ValueError("Refusing unsafe VM cache path")
        (CACHE_ROOT / "helpers").mkdir(parents=True, exist_ok=True)
        for name, source in EMBEDDED_FILES.items():
            (CACHE_ROOT / "helpers" / name).write_text(source, encoding="utf-8")
            sys.modules.pop(Path(name).stem, None)
        helper_path = str(CACHE_ROOT / "helpers")
        if helper_path not in sys.path:
            sys.path.insert(0, helper_path)
        importlib.invalidate_caches()
        from notebook_session import run_session as run_head_session, cleanup_everything

        def run_session(prepare=False):
            selected = {FLAME_VERSION: FLAME_ZIP_DRIVE_PATH}
            if FLAME_VERSION == "2023":
                selected["2020"] = FLAME2020_ZIP_DRIVE_PATH
            run_head_session(CACHE_ROOT, selected, FLAME_VERSION, GPU_ARCH,
                             ITERS, GLOBAL_ITERS, MAX_FIT_BATCH_SIZE, prepare=prepare)
    """, "session-functions"), cell("code", """
        # Install once (or reuse a ready cache), then ask for photos and fit.
        _SKIP_AUTOMATIC_RETRY = False
        run_session(prepare=True)
        _SKIP_AUTOMATIC_RETRY = True
    """, "install-and-fit"), cell("markdown", """
        ## Fit only: retry without installation or Drive access

        After a failed fit, run the next cell to upload the photos again and retry
        with the same installed environment, weights and FLAME files. You may edit
        iteration/batch parameters first. A missing cache requires install-and-fit;
        an incompatible cache requires final privacy cleanup before reinstalling.
        During a successful Run all, this cell skips a
        duplicate upload; run it again manually for another fit. Even a front-only
        fit is accepted when optional profiles cannot be detected; metadata and
        warnings record every omission.
    """, "fit-only-notes"), cell("code", """
        if globals().pop("_SKIP_AUTOMATIC_RETRY", False):
            print("Automatic duplicate fit skipped. Run this cell again for a fit-only retry.")
        else:
            run_session(prepare=False)
    """, "fit-only-retry"), cell("markdown", """
        ## Final privacy cleanup

        Every fit already deletes uploads, outputs, logs and its archive. Keep the
        model cache while retrying. **When finished, set DELETE_VM_CACHE=True below
        and run this cell** to delete everything, including installed software,
        copied/extracted FLAME and downloaded weights. Default False lets Run all
        retain the cache for retries. Then disconnect and delete the runtime.
        No Drive files are deleted. If an earlier cell stops, run this cell manually
        with True when you are done. Clear saved diagnostic outputs before sharing.
    """, "privacy-notes"), cell("code", """
        DELETE_VM_CACHE = False  # Set True and run this cell once finished retrying.
        if DELETE_VM_CACHE:
            cleanup_everything(CACHE_ROOT)
        else:
            print("VM model cache retained for fit-only retries. Set DELETE_VM_CACHE=True here when finished.")
    """, "privacy-cleanup")]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "pixel3dmm_head.ipynb", "provenance": [], "private_outputs": True},
        "accelerator": "GPU",
    }, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    target = HERE.parent / "pixel3dmm_head.ipynb"
    target.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Rebuilt pixel3dmm_head.ipynb from original helper sources; no user data read.")
