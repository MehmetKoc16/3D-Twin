# Pixel3DMM multi-image FLAME head fit on Colab

Research checked **2026-10-02**. This user-operated notebook is the requested
Colab exception to twin-lab's local-only privacy rule. **Colab execution is untested.**
No personal photos, model files, weights or derived meshes were accessed during
development. Only original notebook/helper code is added to the repository.

## User steps

1. Register at [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/), review/accept
   the licence applicable to your use, and download **FLAME 2020** yourself.
   The archive must contain `generic_model.pkl`. Do not download texture-space
   archives instead. The notebook never asks for your FLAME login/password.
2. Upload **FLAME2020.zip** to your Google Drive, e.g. `MyDrive/FLAME2020.zip`.
   Only model archives belong in Drive for this workflow; photos stay on your device.
3. Open/upload [pixel3dmm_head.ipynb](pixel3dmm_head.ipynb) in Google Colab.
4. Select **Runtime > Change runtime type > Python 3 > GPU > L4 or A100**.
   T4 produces a warning and stops before Drive access/photos. GPU availability
   varies; L4/A100 selection is notebook policy, not an upstream memory guarantee.
5. Edit the top parameter cell: set `FLAME_ZIP_DRIVE_PATH` to the exact Drive path
   and leave `FLAME_VERSION="2020"` for the recommended first run. Default fitting
   uses 1500 iterations per view and 1500 joint iterations.
6. Choose **Runtime > Run all**. Authorize Drive when prompted. The notebook copies
   only the selected FLAME zip(s) into its private VM directory and unmounts Drive.
   Installation and native compilation can take tens of minutes and substantial
   RAM/disk; all processing happens in an isolated environment.
7. At `files.upload()`, select all four de-glassed head photos together, named
   `front.png`, `left.png`, `right.png`, `back.png` (JPG/JPEG also accepted).
   Left/right refer to the subject's profiles. Use the same person, neutral face,
   closed mouth, consistent lighting, no glasses and clear face/ears. Back may be
   omitted or skipped if no face is detected. Front and both profiles are required;
   if a profile cannot be detected, retry with a less extreme angle.
8. Save **head_fit.zip** when `files.download()` starts. Put it under
   **`user-data/twin/head/flame/`**, which is gitignored, and extract it there.
   Inspect the overlays in your own local viewer before using the fit downstream.
9. Run the last privacy cell manually if an earlier cell stopped, then select
   **Runtime > Disconnect and delete runtime**. Clear outputs before sharing
   the notebook. To retry after a failed run, use a fresh runtime and Run all.

Drive mounting grants write capability; Colab does not provide a read-only flag
here. **Read-only usage** means this notebook performs only explicit model-zip
reads/copies, then unmounts. It never writes photos, outputs or fitted parameters
to Drive. No other inference/API service is used. Upstream diagnostics are kept
in a VM log rather than displayed in notebook outputs, and are deleted with the
session. Do not share notebook outputs containing your file names.

## Installation and research findings

The [Pixel3DMM README](https://github.com/SimonGiebenhain/pixel3dmm#1-installation)
specifies conda Python 3.9, CUDA 11.8, PyTorch, native PyTorch3D/nvdiffrast builds,
editable package installation and `install_preprocessing_pipeline.sh`.
Its [environment file](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/environment.yml)
uses the torch 2.7 family. Plain pip on a recent Colab kernel is unsuitable for
its legacy NumPy/chumpy stack. This notebook bootstraps
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
**2.0.5** inside the session, creates **Python 3.9 + CUDA 11.8 + GCC 11**, and runs
everything through that environment's interpreter. It avoids condacolab's kernel
restart, keeping Run all continuous. Torch **2.7.1/cu118**, torchvision **0.22.1**,
NumPy **1.23.5** and selected compatibility constraints adapt the upstream recipe.
This is not a fully locked or GPU-tested environment. Nothing is installed in the
project's npm/Python environments by running the notebook.

The installer runs a narrowly adapted VM copy of
[install_preprocessing_pipeline.sh](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/install_preprocessing_pipeline.sh):
HTTPS clones replace SSH, clones check out pinned revisions, shell failures stop,
and the MICA installer replacement performs only weight downloads. Its FLAME
credential/download sections are never executed. Your copied zip supplies FLAME;
MICA's repository supplies auxiliary landmarks, masks and template assets in the
VM. Paths are written to **`~/.config/pixel3dmm/.env`** as upstream documents:
`PIXEL3DMM_CODE_BASE`, `PIXEL3DMM_PREPROCESSED_DATA`, `PIXEL3DMM_TRACKING_OUTPUT`.
Existing configuration is refused; the notebook removes its own config at cleanup.

The [MICA README](https://github.com/Zielon/MICA#pre-trained-models) requires its
pretrained weights plus insightface `antelopev2`/`buffalo_l`. The notebook fetches
them from the identifiers in the
[upstream MICA install replacement](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/preprocessing/replacement_code/install_mica_download_flame.sh).
It keeps insightface models in a session-specific directory and uses CPU ONNX
detection to avoid a second CUDA runtime dependency; MICA itself runs on GPU.
The pipeline installer fetches PIPNet's WFLW checkpoint and Pixel3DMM's UV/normal
checkpoints. Facer fetches its detector/FaRL parsing assets when initialized;
PyTorch/torch-hub also fetch required pretrained encoders/backbones automatically.
All caches stay in the session directory. Weight URLs can expire or hit quotas;
there is no redistribution mirror or credential fallback.

| Source | Pinned revision |
| --- | --- |
| Pixel3DMM | `fcd1fa973c7715b02a8948dfc679dff53cf85924` |
| MICA | `af22e7a5810d474bc28a1433db533723d6bd2b07` |
| facer | `ddd35c76ff840174b8a5403ad1c1255e37b8782b` |
| PIPNet | `b9eab58816437403a34aa5bc3adeafe5081fd36b` |
| PyTorch3D stable | `75ebeeaea0908c5527e7b1e305fbc7681382db47` |
| nvdiffrast v0.3.3 | `729261dc64c4241ea36efda84fbf532cc8b425b8` |

Public checkpoint downloads have no published digest lock in this adaptation.
Legacy upstream checkpoints require trusted pickle loading; the worker environment
enables PyTorch's legacy loader only for these upstream assets. Never substitute
an untrusted model archive/checkpoint.

### FLAME 2023 option

Set `FLAME_VERSION="2023"`, point `FLAME_ZIP_DRIVE_PATH` to your **FLAME2023.zip**,
and also set `FLAME2020_ZIP_DRIVE_PATH` to your **FLAME2020.zip**. Both are manually
downloaded by you after registration and copied read-only from Drive.
The 2023 archive must contain **`flame2023_no_jaw.pkl`**; no automatic substitution
with an explicit-jaw model is made. The
[upstream variant instructions](https://github.com/SimonGiebenhain/pixel3dmm#downloading-flame)
recommend `use_flame2023=True` with `ignore_mica=True` because MICA was trained on
2020. The checked source still loads 2020 landmarks/masks and runs MICA during
preprocessing, so **2023 needs both archives**. MICA results are ignored by the
2023 fit. Zero expression neutralizes jaw-related expression modes in the no-jaw
variant; the jaw rotation parameter has no active articulation there.

## Multi-image fitting and outputs

The [multi-image interface](https://github.com/SimonGiebenhain/pixel3dmm#242-multi-image-inference)
uses `is_discontinuous=True`. This adapter independently crops/preprocesses each
photo using the upstream cropping, MICA and facer scripts, then concatenates valid
results into a contiguous sequence. It checks every landmark/segmentation and
UV/normal prediction because upstream wrappers sometimes catch/ignore errors.
The worker uses upstream `Tracker` with the same YAML/CLI configuration contract;
identity is shared and expression/pose are per view. `global_camera=False` also
fits focal/principal point per photo. Batch size is the accepted photo count.
No temporal smoothness is applied. Neck fitting is disabled because this region
is often underconstrained. Skipped back views have reasons in the metadata.

`head_fit.zip` contains only the export directory; no FLAME files, weights,
repositories, original photo files or pickle checkpoints are bundled.
Overlays do contain the photographed likeness and remain private.

| File | Format and meaning |
| --- | --- |
| `head_neutral.obj`, `head_neutral.ply` | Untextured native FLAME triangle mesh, metres, no normalization/calibration. Regenerated from the final joint-fit identity, with expression/eyelids zero, all joint/global rotations identity and translation zero. |
| `parameters.json` | `dt-flame-head-parameters/1`: neutral coefficients and each view's fitted shape, expression (`exp`), global rotation/translation (`R`, `t`), jaw, neck, eyes, eyelids and global rotation matrix. |
| `parameters.npz` | Numeric arrays; keys `neutral_<field>`, `<view>_<field>`, `<view>_joint_transforms`, `<view>_intrinsics`, `<view>_world_to_camera`. Load with `numpy.load(..., allow_pickle=False)`. |
| `cameras.json` | `dt-flame-head-cameras/1`: raw method parameters, intrinsics, row-major 4x4 OpenGL world-to-camera transforms, head-centric transforms, projection matrices, crop bounds, original image sizes and skipped-view reasons. |
| `fitted_views/<view>.ply` | Final fitted per-view local mesh, including that view's expression/jaw/neck/eyes; global head transform is in its exported camera. |
| `overlays/<view>.png` | Blue shaded mesh blended at 50% over that view's 256x256 face crop; only accepted views have overlays. Generated with nvdiffrast's CUDA rasterizer. |
| `provenance.json` | Timestamp, source revisions, model variant, full fitting settings, view usage, licence caveat and limitations. |

Rotation vectors use **6D first-two-rows** encoding, not axis-angle: identity is
`[1,0,0,0,1,0]`; eyes concatenate two such rotations. Expression has 100 values
and shared identity has 300. Neutral parameters describe the neutral mesh;
fitted view parameters describe the per-view geometry. The early upstream
`canonical.ply` predates joint fitting and is deliberately not exported.

### Camera convention

Camera reconstruction follows the
[upstream camera example](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/scripts/viz_head_centric_cameras.py)
and [tracking rasterizer](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/tracking/nvdiffrast_util.py).
`fl` is normalized focal length; `pp` is normalized principal-point displacement.
For crop side `S=256`: `fx=fy=fl*S`, `cx,cy=(1+pp)*(S/2+0.5)`.
For a fitted-view local vertex, use column vectors:

```text
worldToCamera = baseCamera @ globalHeadTransform
q = worldToCamera @ [x,y,z,1]
depth = -q.z                    # OpenGL looks along -Z, +Y points up
u = fx*q.x/depth + cx
v = cy - fy*q.y/depth
```

These are pixel-edge coordinates: pixel array center `(column,row)` lies at
`(column+0.5,row+0.5)`. The exported projection matrix uses near/far 0.1/5 m and
negative fy, matching upstream's raster-array orientation; do not flip the PNG.
All matrices are row-major in JSON/NPZ, applied to column vectors. Native raw
base camera/`fl`/`pp` fields are preserved to avoid losing the method's semantics.

The head-centric transform additionally composes FLAME joint 1's transform,
following upstream's visualization. It applies only after undoing that transform
on the matching fitted mesh. **It does not compensate nonrigid expression/jaw
changes in `head_neutral`**. Use fitted-view meshes to check exact alignment.
Cameras target face crops, not original photos: original sizes and exclusive slice
bounds `[ymin,ymax,xmin,xmax]` are provided for downstream crop mapping. EXIF
orientation is applied before cropping. Camera/overlay alignment remains GPU-untested.

FLAME does not reconstruct hair; unseen back-of-head anatomy follows its prior.
Scale is inherited from the model, not measured from these photographs. This
archive is an optional head intermediate, not the single-file `twin.glb` contract.
Subsequent integration must create the rig, embedded texture, `asset.extras.dtTwin`,
full twin metadata, forearm skin tone and uint32 LE mh2twin buffer view separately.

## Licensing and privacy cleanup

This is personal non-commercial research under the owner's 2026-10-02 decision,
not a permissive-only asset pipeline. Review the separate
[Pixel3DMM CC BY-NC 4.0](https://github.com/SimonGiebenhain/pixel3dmm#license),
[MICA licence](https://github.com/Zielon/MICA/blob/master/LICENSE),
[FLAME terms](https://flame.is.tue.mpg.de/),
[insightface model terms](https://github.com/deepinsight/insightface/tree/master/python-package#license)
and [nvdiffrast terms](https://github.com/NVlabs/nvdiffrast/blob/v0.3.3/LICENSE.txt).
The repository contains only our orchestration/helpers. Third-party code/models
are fetched into the session VM, never committed or redistributed by the notebook.
Personal outputs stay under gitignored `user-data/`; they are not cleared for
shipping with the app. Other preprocessing dependencies retain their own terms.

Installation, photo uploads, preprocessing, fitting, export and archive transfer
share `try/finally`. It clears uploaded bytes, stops the subprocess group on
ordinary errors/interruption, and removes the entire private
`/content/dt-pixel3dmm-session-*` directory, including FLAME zips/extracted models,
repositories, caches/weights, originals, derived images/meshes, logs and ZIP.
The final cell repeats bounded cleanup and removes only session-owned `.env`.
It never deletes anything from Drive. Browser download completion is awaited using
the same wrapper convention as the TRELLIS notebook. Hard termination can bypass
Python cleanup; disconnect/delete the runtime. Deletion is not secure erasure or
a statement about Google's retention. Your downloaded copy remains your responsibility.

## Developer validation

Helpers under `pixel3dmm/` rebuild/validate the standalone notebook; no repository
dependency manifests are changed. Use an existing validation environment with
`nbformat`, `numpy`, `pytest`:

```powershell
python tools/twin-lab/colab/pixel3dmm/build_notebook.py
python tools/twin-lab/colab/pixel3dmm/validate_notebook.py
python -m pytest tools/twin-lab/colab/pixel3dmm/test_helpers.py -q -p no:cacheprovider
npm run typecheck
npm run lint
npm run test
```

`nbformat` validates notebook JSON; each extracted code cell and every helper
passes `py_compile`. Synthetic tests cover zip traversal/symlinks, model variant
selection, preserving auxiliary files, bounded cleanup on error, upload naming,
camera projection/composition and fail-closed source patches. No tests contain
photos, real model data or personal meshes.

Local checks (2026-10-02): **nbformat passed; all 5 code cells and helpers passed
py_compile; 15 synthetic pytest cases passed; repository typecheck passed.**
Standard `npm run lint` encountered an inaccessible existing `.pytest_cache`;
direct ESLint with `--ignore-pattern '**/.pytest_cache/**'` passed.
Standard `npm run test` hit Windows `spawn EPERM` / native Vite dependency errors;
direct Vitest thread-pool runs passed **290 web + 111 avatar-core tests**.
The web run used `--experimental-strip-types` and `--configLoader=native`.

**Untested:** Colab environment solve, CUDA/native builds, checkpoint URLs/quotas,
FLAME 2020/2023 loading, real detections/segmentation/fitting, likeness, alignment,
GPU memory/performance and browser Drive/upload/download interactions. Local syntax
and synthetic checks cannot prove operational Colab execution; smoke-test there
before treating this as a working stage.
