# Resumable Colab installation

The notebook embeds these original helpers. No third-party source or model is
committed. Run `build_notebook.py` after changing helpers. The Python adapter
reproduces the [pinned upstream preprocessing recipe](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/install_preprocessing_pipeline.sh)
without executing that script or MICA's credential-based installer. SHA-256
checks verify the four pinned replacement files before copying them.

The order is environment creation, apt packages, numerical/build tools, torch,
Pixel3DMM clone, requirements, PyTorch3D, nvdiffrast, Pixel3DMM editable install,
adapter verification, facer clone/replacements/editable install, MICA
clone/replacements/CPU detector patch, MICA and insightface weights/extraction,
PIPNet clone/nms build/checkpoint, UV and normals weights, local FLAME staging,
runtime patches, dependency/CUDA checks. All pip/build commands use the isolated
environment's Python; pip keeps the constraints file.

Each expensive operation writes `cache/done/<step>.json` only after success.
Markers include adapter version, GPU architecture, software/source revisions and
the HuggingFace weight revision. Failed or stale markers are ignored. Each
weight has its own marker; incomplete downloads use `.partial` files and atomic
rename. `install_complete.json` is written only after all steps/checks pass.

After an installation failure, re-run **install-and-fit**, not **Fit only**.
Matching completed steps are skipped; the failed step runs again. The installed
environment/sources/weights survive private session cleanup. An old partial
cache without these markers needs one preparatory pass to create them. A complete
incompatible cache requires the notebook's final cache cleanup before rebuilding.
Fit only continues to require a complete compatible installation.

## Downloads and manual fallback

Each source gets three attempts with backoff. Copied Drive files are preferred;
already completed VM files are reused. Use only these official sources and review
their model licences for personal non-commercial use:

| Filename | First official source | Network fallback |
| --- | --- | --- |
| `uv.ckpt` | [Author's pinned HuggingFace file](https://huggingface.co/SimonGiebenhain/Pixel3dmm/resolve/98608f33a2d8af6eed387ac354db6af2030a8f44/uv.ckpt) | [Upstream Drive file](https://drive.google.com/file/d/1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY/view) |
| `normals.ckpt` | [Author's pinned HuggingFace file](https://huggingface.co/SimonGiebenhain/Pixel3dmm/resolve/98608f33a2d8af6eed387ac354db6af2030a8f44/normals.ckpt) | [Upstream Drive file](https://drive.google.com/file/d/1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD/view) |
| `antelopev2.zip` | [Official insightface v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip) | [Upstream Drive file](https://drive.google.com/file/d/16PWKI_RjjbE4_kqpElG-YFqe8FpXjads/view) |
| `buffalo_l.zip` | [Official insightface v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) | [Upstream Drive file](https://drive.google.com/file/d/1navJMy0DTr1_DHjLWu1i48owCPvXWfYc/view) |
| `mica.tar` | [Upstream MICA Drive file](https://drive.google.com/file/d/1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt/view) | User copy |
| `epoch59.pth` | [Upstream PIPNet Drive file](https://drive.google.com/file/d/1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C/view) | User copy |

If Colab downloads are rate-limited, download the reported missing file yourself
and put it in **`MyDrive/flame/weights/<filename>`**. Re-run install-and-fit:
while reading FLAME during the same Drive mount, the notebook copies any of these
six filenames into `cache/drive_weights`, then unmounts Drive. An unfinished
install mounts again to discover newly supplied files even when FLAME was already
copied. The notebook never writes to Drive. No tokens are requested. UV/normals
must exceed 100 MB; other weights must exceed 1 MB; HTML error pages are rejected.

## Diagnostics and privacy boundary

The context names the failing sub-step and its log. Before photo upload only,
failures print a filtered traceback plus the last 60 lines of that log. ANSI
escapes are stripped, carriage-return progress updates collapse to their final
state, credential-related lines are redacted and lines are capped at 300 characters.
The session sets `installation_finished=True` before photo upload. Every later
failure, including uploads/downloads, uses strict traceback-only filtering.
The diagnostic function also independently requires an `install:` step prefix.
Logs are printed before deletion; photos/outputs/logs remain session-bound.
The final privacy cell with `DELETE_VM_CACHE=True` removes the complete VM cache.

## Verification status (2026-10-02)

The pinned recipe/PIPNet build command and replacement-file hashes were checked.
The notebook was regenerated; standard-library JSON parsing and `py_compile` of
all seven code cells and helpers passed using `colab/.venv`.
Standard-library smoke checks also passed for checkpoint reuse/invalidation,
retry/source ordering, atomic download, log filtering and the privacy boundary.
`pytest` and `nbformat` are missing from that venv, so helper tests and
`validate_notebook.py` could not run. No alternate Python was used. Tests were
added for ordering/retries/markers, download fallbacks, Drive preference and the
diagnostic privacy boundary. The lead should run them in the prepared environment.
Actual Colab installation, model downloads, CUDA builds and fitting remain untested.
Repository typecheck passed. Standard lint and npm tests were attempted but hit
Windows `EPERM` errors scanning `.pytest_cache` and spawning test workers.
