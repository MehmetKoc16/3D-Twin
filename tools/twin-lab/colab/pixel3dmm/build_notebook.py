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
                ("io_utils.py", "camera.py", "setup_runtime.py", "worker.py", "export_fit.py")}
    cells = [cell("markdown", """
        # Multi-image FLAME head fit with Pixel3DMM

        Personal non-commercial research workflow, owner decision 2026-10-02.
        Read [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm),
        [MICA](https://github.com/Zielon/MICA) and your [FLAME](https://flame.is.tue.mpg.de/)
        licence before using their assets. Pixel3DMM is CC BY-NC 4.0; MICA,
        insightface models and NVIDIA's renderer have separate restricted terms.
        No models or upstream source are embedded here or redistributed by this notebook.

        Register at FLAME, download **FLAME2020.zip** yourself and put it in your
        Google Drive. Choose **Runtime > Change runtime type > GPU** (T4 works; L4/A100 are faster).
        Edit the parameter cell, then **Runtime > Run all**.
        A separate Python 3.9 / CUDA 11.8 conda environment avoids restarting Colab.
        Installation/native compilation and multi-GB weight downloads can take tens of minutes.
        **Colab execution has not been tested.**

        Drive is mounted with read-only **usage**, not a read-only permission grant:
        Colab's Drive mount can write. This notebook only reads the selected FLAME
        archive(s), copies them into the VM and immediately unmounts Drive.
        Never place/upload your photos in Drive for this workflow. When prompted,
        select de-glassed `front.png`, `left.png`, `right.png`, `back.png` together
        (.jpg/.jpeg also accepted). Left/right mean the subject's left/right profile.
        Back is optional and omitted from fitting if no face is detected.
        Front and both profiles must preprocess successfully.

        Photos run on Google's VM only, through user-operated uploads. No remote
        inference service or logging service receives them. No previews are displayed.
        Neutral head OBJ/PLY, JSON/NPZ parameters, cameras, fitted-view meshes and PNG
        overlays download as **head_fit.zip**. Keep it locally in
        **user-data/twin/head/flame/** (gitignored); never ship it with the web app.
        This is a head intermediate, not the rigged `twin.glb` bundle.

        Installation, upload, fitting and download share a `try/finally` cleanup.
        The final privacy cell also checks that no session directory remains.
        Save the browser download, then **Runtime > Disconnect and delete runtime**.
        VM deletion is not secure erasure or a promise about Google's retention.
        Clear notebook outputs before sharing. On hard kernel/VM termination,
        cleanup cannot run; delete the runtime yourself.
    """, "instructions"), cell("code", """
        # Parameters: edit before Runtime > Run all.
        FLAME_ZIP_DRIVE_PATH = "/content/drive/MyDrive/FLAME/FLAME2020.zip"
        FLAME_VERSION = "2020"  # "2020" (default) or "2023" (no-jaw variant)
        # For 2023, point FLAME_ZIP_DRIVE_PATH to FLAME2023.zip AND provide 2020:
        # preprocessing/MICA and tracking landmark/mask assets still require 2020.
        FLAME2020_ZIP_DRIVE_PATH = "/content/drive/MyDrive/FLAME/FLAME2020.zip"
        ITERS = 1500
        GLOBAL_ITERS = 1500
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
        if any(not isinstance(value, int) or value < 1 for value in (ITERS, GLOBAL_ITERS)):
            raise ValueError("Iteration counts must be positive integers")
        try:
            gpu_summary = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
            ).splitlines()[0]
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as error:
            raise RuntimeError("Select an L4 or A100 Colab GPU runtime") from error
        gpu_name, gpu_memory = gpu_summary.rsplit(",", 1)
        print(f"GPU: {gpu_name.strip()}, {int(gpu_memory)} MiB")
        # T4 (16 GB, sm_75) is the only GPU on Google AI Pro; L4/A100/H100 need AI Ultra.
        GPU_ARCHES = {"T4": "7.5", "L4": "8.9", "A100": "8.0", "H100": "9.0"}
        GPU_ARCH = next((arch for name, arch in GPU_ARCHES.items() if name in gpu_name), None)
        if GPU_ARCH is None or int(gpu_memory) < 15000:
            raise RuntimeError("A T4, L4, A100 or H100 GPU runtime is required")
        if GPU_ARCH == "7.5":
            print("T4 detected: fitting is slower; if it runs out of memory, lower ITERS/GLOBAL_ITERS.")
    """, "gpu-check"), cell("code", "EMBEDDED_FILES = " + repr(embedded), "embedded-original-helpers"), cell("markdown", """
        ## Copy FLAME, install, upload, fit and download

        Drive authorization is only needed to read the model zip(s). No FLAME
        username/password is requested. The credential sections of upstream's
        preprocessing installer are replaced in the VM with model-weight downloads.
        All subprocess diagnostics stay in the session's `runtime.log`, removed
        by cleanup; if installation fails, the cell reports the failing command
        without displaying upstream numeric/photo debug output.

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
        import gc
        import importlib
        import json
        import shutil
        import sys
        import tempfile
        from google.colab import drive, files

        def run_session():
            session = Path(tempfile.mkdtemp(prefix="dt-pixel3dmm-session-", dir="/content"))
            helpers = session / "helpers"
            helpers.mkdir()
            uploaded = {}
            mounted = False
            config_created = False
            env_file = Path.home() / ".config/pixel3dmm/.env"
            try:
                for name, source in EMBEDDED_FILES.items():
                    (helpers / name).write_text(source, encoding="utf-8")
                sys.path.insert(0, str(helpers))
                importlib.invalidate_caches()
                from io_utils import upload_views, download_and_wait
                from setup_runtime import install, run
                selected = {FLAME_VERSION: FLAME_ZIP_DRIVE_PATH}
                if FLAME_VERSION == "2023":
                    selected["2020"] = FLAME2020_ZIP_DRIVE_PATH
                drive.mount("/content/drive")
                mounted = True
                for version, filename in selected.items():
                    model_zip = Path(filename).resolve()
                    if not model_zip.is_relative_to(Path("/content/drive").resolve()) or model_zip.suffix.lower() != ".zip":
                        raise ValueError("Select a FLAME zip inside /content/drive")
                    if not model_zip.is_file():
                        raise FileNotFoundError("FLAME zip not found; check the parameter cell")
                    # Read-only Drive usage: this is the ONLY Drive file operation.
                    shutil.copyfile(model_zip, session / ("FLAME" + version + ".zip"))
                drive.flush_and_unmount()
                mounted = False
                if env_file.exists():
                    raise RuntimeError("Existing Pixel3DMM config found. Use a fresh Colab runtime")
                env_file.parent.mkdir(parents=True, exist_ok=True)
                paths = {
                    "PIXEL3DMM_CODE_BASE": str(session / "pixel3dmm"),
                    "PIXEL3DMM_PREPROCESSED_DATA": str(session / "preprocessed"),
                    "PIXEL3DMM_TRACKING_OUTPUT": str(session / "tracking"),
                }
                env_file.write_text("".join(key + "=" + json.dumps(value) + "\\n" for key, value in paths.items()), encoding="utf-8")
                config_created = True
                python, worker_env = install(session, GPU_ARCH)
                worker_env.update(paths)
                (session / "config.json").write_text(json.dumps({
                    "flame_version": FLAME_VERSION, "iters": ITERS, "global_iters": GLOBAL_ITERS,
                }), encoding="utf-8")
                uploads = session / "uploads"
                uploads.mkdir()
                print("Upload de-glassed front, left, right and back head photos together.")
                uploaded = files.upload(target_dir=str(uploads))
                upload_views(uploaded.keys())
                uploaded.clear()
                print("Preprocessing each view, then fitting shared identity with independent cameras.")
                run([python, helpers / "worker.py", session], session, cwd=session / "pixel3dmm", env=worker_env)
                download_and_wait(session / "head_fit.zip")
                print("head_fit.zip transferred to your browser. Save it to user-data/twin/head/flame/.")
            finally:
                uploaded.clear()
                # Nested finally ensures private files are removed even if Drive unmount fails.
                try:
                    if mounted:
                        drive.flush_and_unmount()
                finally:
                    try:
                        if config_created:
                            env_file.unlink(missing_ok=True)
                    finally:
                        if session.resolve().parent != Path("/content") or not session.name.startswith("dt-pixel3dmm-session-"):
                            raise ValueError("Refusing unsafe session cleanup")
                        shutil.rmtree(session)
                        if str(helpers) in sys.path:
                            sys.path.remove(str(helpers))
                        for name in EMBEDDED_FILES:
                            sys.modules.pop(Path(name).stem, None)
                        gc.collect()
                        print("Privacy cleanup: VM uploads, FLAME, weights, outputs, logs and archive removed.")

        run_session()
    """, "run-with-cleanup"), cell("markdown", """
        ## Privacy cleanup (last cell)

        The session `finally` already deletes all VM photos, derived files, model
        archives/extracted FLAME and weights on success or ordinary errors.
        This final cell checks for leftovers, including after a previous interrupted
        run. If Run all stopped on an error, run this cell manually.
        No Drive file is deleted. Always disconnect and delete the runtime after saving.
    """, "privacy-notes"), cell("code", """
        from pathlib import Path
        import gc
        import shutil

        for leftover in Path("/content").glob("dt-pixel3dmm-session-*"):
            if leftover.resolve().parent != Path("/content") or leftover.is_symlink():
                raise ValueError("Refusing unsafe cleanup path")
            shutil.rmtree(leftover)
        # Remove only this notebook's session-bound configuration, never an unrelated config.
        config_file = Path.home() / ".config/pixel3dmm/.env"
        if config_file.exists() and '"/content/dt-pixel3dmm-session-' in config_file.read_text():
            config_file.unlink()
        gc.collect()
        print("No Pixel3DMM session files remain. Save your download, then disconnect and delete runtime.")
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
