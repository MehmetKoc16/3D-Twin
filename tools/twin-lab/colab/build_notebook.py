"""Rebuild the self-contained notebook using only public code, never user-data.

Run this after changing the worker, setup helpers, validator or shared meshops.
The notebook embeds the existing shape/meshops.py to preserve camera fitting.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent


def cell(kind: str, source: str, identifier: str) -> dict:
    result = {"cell_type": kind, "id": identifier, "metadata": {}, "source": dedent(source).strip() + "\n"}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build() -> dict:
    embedded = {
        "shape_worker.py": (HERE / "shape_worker.py").read_text(encoding="utf-8"),
        "check_meta.py": (HERE / "check_meta.py").read_text(encoding="utf-8"),
        "meshops.py": (HERE.parent / "shape/meshops.py").read_text(encoding="utf-8"),
    }
    cells = [cell("markdown", """
        # TRELLIS.2 shape stage on Colab

        **Licence finding:** Microsoft's code and 4B weights are MIT, but the trained
        image conditioner requires **Meta DINOv3's custom licence and gated access**.
        This is not a permissive-only pipeline. Read the
        [DINOv3 licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
        and request access to [the encoder](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
        before using this notebook. BRIA RMBG-2.0 and NVIDIA's noncommercial renderers
        are excluded. Background removal uses rembg / Apache-2.0 U²-Net instead.

        On **Colab Pro**, use **Runtime → Change runtime type → Python 3 → GPU → A100**
        (recommended, 40 GB class) or **L4 (24 GB)**. Availability and compute units vary;
        Pro does not guarantee either GPU. Upstream requires at least 24 GB and tests
        A100/H100; L4 is an unverified minimum-memory target. Default: 512 and CPU offload.
        For A100 you may select `1024_cascade`. **T4 (16 GB) is unsupported**: switch
        to L4/A100 or wait for availability. No speculative T4 workaround is offered.

        Edit the parameter cell, then **Runtime → Run all**. Installation can take
        tens of minutes and downloads several GB of weights. A separate Python
        environment avoids kernel restarts. Enter a read-only Hugging Face token
        when asked; it must belong to an account approved for the encoder.
        Upload `front.png` and optionally `back.png`, `left.png`, `right.png` together.
        Full-body, standing, consistent A-pose views are best. Only **front** conditions
        the shape: upstream has no supported multi-image API. Other views get cutouts
        and orthographic camera fits for the subsequent texture stage.

        Output: `trellis2_shape.zip` containing `mesh.glb`, `meta.json`, cutouts and
        masks. Save/unzip it into your gitignored `user-data/twin/out/shape/` locally.
        Geometry is untextured/unrigged, in metres, +Y up, facing +Z, feet at y=0,
        with total mesh height (including soles/hair) equal to `HEIGHT_CM`.
        This is a shape intermediate; the rig stage must create the final `twin.glb`
        and its `asset.extras.dtTwin` contract. Inspect orientation locally and rerun
        with source-axis overrides if the automatic anatomical heuristic is wrong.

        **Privacy:** images are processed on Google's VM for this session only.
        This is the user-operated Colab exception to twin-lab's local-only workflow.
        No Drive is mounted and no image/mesh previews are displayed or embedded in
        notebook outputs. Downloads stay on your device. Session files are deleted
        in `finally` after the archive reaches browser memory, including on ordinary
        errors/cancellation. This removes VM files; it is not secure erasure or a
        guarantee about Google's retention. If the kernel/VM crashes or is forcibly
        stopped, use **Runtime → Disconnect and delete runtime**. Do that after every
        session, and clear notebook outputs before sharing it.
    """, "instructions"), cell("code", """
        # Parameters: edit these before Runtime > Run all.
        HEIGHT_CM = 178.0
        SEED = 42
        PIPELINE_TYPE = "512"  # "512" for L4; "1024_cascade" optionally on A100
        STEPS = 12
        TARGET_FACES = 80000
        MAX_NUM_TOKENS = 49152
        SOURCE_UP = "auto"  # "auto" or one of "+X", "-X", "+Y", "-Y", "+Z", "-Z"
        SOURCE_FORWARD = "auto"  # Override BOTH axes if detection is wrong.
        FLIP_FORWARD = False  # Rotate 180 degrees about final +Y when front is reversed.
        # Strict permissive-only use cannot run this model's trained image encoder.
        # Set True only after independently reviewing/accepting Meta's DINOv3 terms.
        DINO_LICENSE_ACKNOWLEDGED = False
    """, "parameters"), cell("markdown", """
        ## Install and check access before uploading images

        The licence prerequisite deliberately fails before installation or upload
        until `DINO_LICENSE_ACKNOWLEDGED` is set. There is no automatic acceptance of
        third-party terms. The worker uses the pinned shape sampling/decoding substeps,
        rather than `pipeline.run`, which also loads BRIA and generates PBR attributes.
        O-Voxel's package entry point is restricted to geometry conversion in this VM
        to avoid an eager renderer import. GPU installation/inference remain untested
        by the notebook authors; compiler failures stop immediately.
    """, "setup-notes"), cell("code", (HERE / "notebook_setup.py").read_text(encoding="utf-8"), "setup-helpers"), cell("code", """
        import math
        from getpass import getpass

        if not DINO_LICENSE_ACKNOWLEDGED:
            raise RuntimeError("Read the DINOv3 licence/access instructions above. This pipeline is not permissive-only.")
        if not math.isfinite(HEIGHT_CM) or HEIGHT_CM <= 0:
            raise ValueError("HEIGHT_CM must be positive and finite.")
        if PIPELINE_TYPE not in ("512", "1024_cascade"):
            raise ValueError("Use 512 or 1024_cascade.")
        if not isinstance(SEED, int) or not isinstance(STEPS, int) or STEPS < 1:
            raise ValueError("Use integer SEED and positive integer STEPS.")
        if not isinstance(TARGET_FACES, int) or TARGET_FACES < 0 or not isinstance(MAX_NUM_TOKENS, int) or MAX_NUM_TOKENS < 1:
            raise ValueError("Invalid face/token limits.")
        allowed_axes = {"auto", "+X", "-X", "+Y", "-Y", "+Z", "-Z"}
        if SOURCE_UP not in allowed_axes or SOURCE_FORWARD not in allowed_axes:
            raise ValueError("Invalid source-axis override.")
        install_software()
        os.environ["HF_HOME"] = str(BUILD_ROOT / "hf-cache")
        os.environ["U2NET_HOME"] = str(BUILD_ROOT / "rembg-cache")
        if not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = getpass("Hugging Face read token (approved DINOv3 account): ")
        # HEAD checks gated weights without uploading any images or logging the token.
        run_command([ENV_PYTHON, "-c",
            "from huggingface_hub import get_hf_file_metadata, hf_hub_url; "
            "get_hf_file_metadata(hf_hub_url('facebook/dinov3-vitl16-pretrain-lvd1689m', 'model.safetensors', revision='" + DINO_REVISION + "')); "
            "print('Encoder access verified.')"])
    """, "install-and-access"), cell("markdown", """
        ## Upload, reconstruct, download, delete

        Select all views in the file picker. Optional views are used for camera fits
        only, never silently treated as multi-view shape conditioning. The original
        pixel grid is preserved in cutouts/masks and camera `imageSize`, `originPx`,
        `pxPerMeter`; yaw front/left/back/right is 0/90/180/270 degrees.
        Enable browser downloads. The cell waits for the download transfer to browser
        memory before removing its VM source; it cannot verify that you saved it to disk.
    """, "run-notes"), cell("code", "EMBEDDED_FILES = " + repr(embedded), "embedded-public-code"), cell("code", """
        import json
        import zipfile
        from google.colab import files

        def run_session():
            session_dir = Path(tempfile.mkdtemp(prefix="dt-shape-session-", dir="/content"))
            uploaded = {}
            try:
                uploads_dir = session_dir / "uploads"
                uploads_dir.mkdir()
                (session_dir / "out").mkdir()
                config = {
                    "height_cm": HEIGHT_CM, "seed": SEED, "pipeline_type": PIPELINE_TYPE,
                    "steps": STEPS, "target_faces": TARGET_FACES, "max_num_tokens": MAX_NUM_TOKENS,
                    "source_up": SOURCE_UP, "source_forward": SOURCE_FORWARD, "flip_forward": FLIP_FORWARD,
                    "trellis_source": str(TRELLIS_SOURCE), "trellis_revision": TRELLIS_REVISION,
                    "sparse_decoder_revision": SPARSE_DECODER_REVISION, "dino_revision": DINO_REVISION,
                }
                (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
                for name, source in EMBEDDED_FILES.items():
                    (session_dir / name).write_text(source, encoding="utf-8")
                print("Upload front.png and optional back.png / left.png / right.png together.")
                # target_dir confines files.upload's on-disk copies, even on rejection.
                uploaded = files.upload(target_dir=str(uploads_dir))
                names = {Path(name).name for name in uploaded}
                if "front.png" not in names or not names.issubset({"front.png", "left.png", "back.png", "right.png"}):
                    raise ValueError("Upload exactly front.png plus optional left.png/back.png/right.png.")
                uploaded.clear()  # No photo bytes kept in notebook globals.
                run_command([ENV_PYTHON, session_dir / "shape_worker.py", session_dir])
                archive = session_dir / "trellis2_shape.zip"
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for path in sorted((session_dir / "out").iterdir()):
                        bundle.write(path, path.name)
                download_and_wait(archive)
                print("Archive transferred to the browser. Save it locally before closing this tab.")
            except BaseException:
                print("Session stopped; private VM files will be removed. Re-run this cell to retry.")
                raise
            finally:
                uploaded.clear()
                cleanup_session(session_dir)
                os.environ.pop("HF_TOKEN", None)

        run_session()
    """, "run-with-cleanup"), cell("markdown", """
        ## Privacy cell (runs automatically at the end)

        The processing cell already removes uploads and every derived output in a
        `finally` block. This final cell verifies that no session directory remains
        and drops the session token. Public software/model caches are not personal
        data and remain until you delete the runtime. Disconnect and delete this
        runtime after use. Never commit the downloaded archive or its contents.
    """, "privacy-notes"), cell("code", """
        remaining_sessions = list(Path("/content").glob("dt-shape-session-*"))
        if remaining_sessions:
            raise RuntimeError("A prior interrupted session remains. Disconnect and delete this runtime.")
        os.environ.pop("HF_TOKEN", None)
        gc.collect()
        print("No private session files remain. Now disconnect and delete this runtime.")
    """, "privacy-cleanup")]
    return {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"name": "trellis2_shape.ipynb", "provenance": [], "private_outputs": True},
            "accelerator": "GPU",
        },
    }


if __name__ == "__main__":
    (HERE / "trellis2_shape.ipynb").write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
