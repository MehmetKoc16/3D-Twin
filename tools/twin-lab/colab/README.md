# TRELLIS.2 shape on Colab

Research checked **2026-10-01** against Microsoft's source and model card. This
folder is a user-operated Colab alternative to the local Hunyuan shape stage.
No user data was read, displayed or uploaded while building/testing it.

## Licence finding: MIT model, but not a permissive-only pipeline

| Component | Findings and primary sources | Notebook choice |
| --- | --- | --- |
| TRELLIS.2 code | [MIT licence](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/LICENSE) | Pinned Microsoft code; no Tencent components |
| `microsoft/TRELLIS.2-4B` weights | [Model card](https://huggingface.co/microsoft/TRELLIS.2-4B), [upstream licence statement](https://github.com/microsoft/TRELLIS.2#%EF%B8%8F-license): MIT | Pinned weights; no Hunyuan territorial restrictions in their MIT grant |
| Required image encoder | The [pipeline configuration](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/af44b45f2e35a493886929c6d786e563ec68364d/pipeline.json) selects [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). It is gated and uses the [custom DINOv3 licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), with a worldwide grant plus use/redistribution conditions. It is **not MIT/Apache/BSD**. | Retained because these checkpoints were conditioned on its features; the notebook requires explicit acknowledgement and approved HF access. No verified permissive drop-in replacement was found; swapping DINOv2 is not a validated inference path. |
| Default background removal | The same config selects [BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), whose model card restricts the publicly supplied weights to noncommercial use. | Never instantiated or downloaded; replaced by [rembg (MIT)](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt) / `u2net_human_seg` from [U²-Net (Apache-2.0)](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE), on CPU |
| Upstream PBR exporter/renderers | [O-Voxel postprocessing](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/o-voxel/o_voxel/postprocess.py) imports nvdiffrast. The [v0.4.0 NVIDIA licence](https://github.com/NVlabs/nvdiffrast/blob/v0.4.0/LICENSE.txt), used by upstream setup, limits use to noncommercial research/evaluation (§3.3). Upstream also documents separate nvdiffrec terms. | Neither NVIDIA renderer installed or imported; geometry-only decoding and trimesh GLB export |
| Native geometry dependencies | [CuMesh MIT](https://github.com/JeffreyXiang/CuMesh/blob/main/LICENSE), [FlexGEMM MIT](https://github.com/JeffreyXiang/FlexGEMM/blob/main/LICENSE); O-Voxel is part of the Microsoft source tree | Pinned builds; O-Voxel's package entry point is changed in the VM to import `convert` only, avoiding its eager PBR exporter import |

**Consequently, “licence-clean” is an objective, not a blanket clearance claim.**
TRELLIS.2 removes the stated Tencent restriction, but its required encoder prevents
certifying the complete workflow against this repo's permissive-only asset policy.
`DINO_LICENSE_ACKNOWLEDGED = False` is the default. Strict permissive-only users
must leave it false; running requires reviewing/accepting Meta's terms separately.
No encoder weights, third-party assets or project dependencies are added to this
repo. The metadata records the remaining licence caveat, rather than claiming the
entire result/provenance is MIT. Full output rights also depend on the input images.

## Runtime and install findings

- [Microsoft prerequisites/install](https://github.com/microsoft/TRELLIS.2#%EF%B8%8F-installation)
  specify Linux, **at least 24 GB NVIDIA VRAM**, tested A100/H100, and recommend CUDA
  **12.4**. [Upstream setup](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/setup.sh)
  uses torch **2.6.0** / torchvision **0.21.0**, flash-attn **2.7.3**, and native builds.
- Colab Pro: **Runtime → Change runtime type → Python 3 → GPU → A100** is recommended
  for headroom (typically 40 GB); **L4, 24 GB**, is the minimum-memory target with
  `PIPELINE_TYPE="512"` and CPU offload. **A100 is not strictly required by the memory
  minimum**, but L4 is not an upstream-tested GPU and has not been tested here.
  `1024_cascade` is an optional A100 choice, not a memory/performance guarantee.
- **T4, 16 GB**: fails before uploads; switch to L4/A100, retry when available, or use
  another >=24 GB GPU environment. Reducing resolution/offloading does not make T4
  officially supported; we do not offer an unverified T4 recipe.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html) explains that GPU
  availability/limits fluctuate even on paid plans. Pro does not guarantee A100/L4.
- The notebook installs an isolated venv in `/content/dt-trellis2-software`, targeting
  Colab Python 3.10–3.12. It installs the 12.4 toolkit if absent, pins torch/attention,
  transformers **4.57.3**, source revisions and model revisions, then builds CuMesh,
  FlexGEMM and O-Voxel. Remaining CPU utility package versions are constrained or
  current, not a complete lockfile. Failed subprocesses stop immediately; no kernel
  restart or Drive mounting is required. Native builds/weights can take tens of
  minutes and substantial host RAM/disk; CPU offload also needs host RAM.

Pinned revisions are recorded in `notebook_setup.py` and `shape_worker.py`:

| Source | Revision |
| --- | --- |
| TRELLIS.2 | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| TRELLIS.2-4B | `af44b45f2e35a493886929c6d786e563ec68364d` |
| Original TRELLIS sparse decoder | `25e0d31ffbebe4b5a97464dd851910efc3002d96` |
| DINOv3 encoder | `ea8dc2863c51be0a264bab82070e3e8836b02d51` |
| CuMesh | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| FlexGEMM | `6dd94a859c26ee8246888502eada3dd8ad85532e` |

## Inference and output findings

The [public image-to-3D API](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/pipelines/trellis2_image_to_3d.py)
is `Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")`,
`.cuda()`, `.run(PIL_image, seed=..., pipeline_type=...)[0]`. Available pipeline
types are `512`, `1024`, `1024_cascade`, `1536_cascade`; `low_vram=True` moves
individual stages between CPU/GPU. **`run` accepts one image.** A list argument to
the internal feature extractor is batching, not a documented multi-view generation
interface. There is no supported `run_multi_image` in the checked source.

Upstream `run` returns a `MeshWithVoxel` with vertices/faces and PBR voxel attributes;
its [example](https://github.com/microsoft/TRELLIS.2/blob/main/example.py) uses
`o_voxel.postprocess.to_glb` for a textured/PBR GLB (opacity inactive by default).
Our shape worker instead constructs the pipeline with only shape models and the
required encoder, then calls `preprocess_image`, `get_cond`,
`sample_sparse_structure`, `sample_shape_slat` (or its cascade), and
`decode_shape_slat`. It skips texture sampling/decoding, BRIA construction and PBR
export, producing an **untextured, unrigged triangle GLB**, matching the local stage.
This lower-level shape-only adaptation is based on source inspection and remains
GPU-untested. No claim is made that it reconstructs accurate biometric anatomy.

## Using the notebook

1. Open/upload **`trellis2_shape.ipynb`** in Colab and select the runtime above.
2. Review the licence finding, request DINOv3 access on HF and obtain a read-only
   token. Edit the first parameter cell: height, seed, pipeline type, steps, face
   budget and optional axis overrides. Acknowledge DINOv3 terms only if appropriate.
3. Choose **Runtime → Run all**. Enter the token at the masked prompt. Model access
   and GPU checks run before any image upload. Select `front.png` and optional
   back/left/right views together when `files.upload()` opens.
4. Save **`trellis2_shape.zip`** locally and unzip to
   `user-data/twin/out/shape/` (gitignored). It includes `mesh.glb`, `meta.json`,
   `cutout_<view>.png`, `mask_<view>.png`. Original photos are not in the archive.
   Only front generates geometry; optional views supply camera fits/cutouts.
5. Check the numeric contract locally, then inspect orientation in your own viewer.
   Automatic up/front detection uses standing-body/footprint heuristics, not semantic
   recognition. Ambiguous or inaccurate geometry can be reversed; explicitly set
   `SOURCE_UP`, `SOURCE_FORWARD`, or `FLIP_FORWARD` and rerun if needed.
6. **Runtime → Disconnect and delete runtime** after saving the download.

```powershell
# Existing shape environment already has numpy + trimesh; use a recent trimesh (>=4.6).
tools\twin-lab\shape\.venv\Scripts\python.exe tools\twin-lab\colab\check_meta.py `
  user-data\twin\out\shape\mesh.glb user-data\twin\out\shape\meta.json
```

The output convention is metres, right handed, +Y up, character facing +Z, character
left +X; soles at y=0; x/z origin at the midpoint of the bottom 3% contact-patch bbox.
Total height including hair/soles equals `HEIGHT_CM`. `normalization` records the
proper rotation, positive scale, translation and row-major transform matrix.

`meta.json` uses **`twin-shape/1`** and the **same camera fitting code** as
`shape/generate.py`: `shape/meshops.py` is embedded when building the notebook.
Each uploaded view gets `yawDeg`, `imageSize`, `pxPerMeter`, `originPx`, silhouette
IoU, bbox/initial-guess/anisotropic diagnostics, all on the **original pixel grid**:

```text
x' = x*cos(yaw) - z*sin(yaw)
u  = originPx[0] + pxPerMeter*x'
v  = originPx[1] - pxPerMeter*y
yaw: front 0, left 90 (camera at +X), back 180, right 270
```

Orthographic fits are estimates for perspective photos. The validator checks GLB
world-space geometry including node transforms, ground/contact origin, height/bbox,
rotation/matrix consistency, camera fields/yaws, mesh counts and honest view usage.
It cannot prove anatomical front, image alignment/IoU correctness without the masks,
likeness, watertight reconstruction quality or legal clearance. It rejects external
GLB resources. It does not display images/mesh contents.

This is **not `twin.glb`**. The subsequent texture and rig stages must produce the
skinned/embedded-texture twin, `asset.extras.dtTwin`, full `twin.json`, skin-tone
sample, uint32 LE mh2twin buffer view and provenance required by the bundle contract.

## Privacy and validation

Images are processed on Google's VM **for this session only**. This user-operated
workflow is the explicit Colab exception to the parent README's local-only rules.
The notebook uses `files.upload(target_dir=...)` to confine every uploaded copy to
a fresh `/content/dt-shape-session-*` directory; no Drive is mounted, no remote
inference service receives the images and no previews are displayed. The archive
transfer uses `files.download()`; a completion wrapper waits for its JavaScript
promise, because [Colab's helper](https://github.com/googlecolab/colabtools/blob/main/google/colab/files.py)
otherwise returns before the browser has read the file. This implementation detail
may need updating if Colab changes its helper.

The session's `finally` removes original uploads, cutouts/masks, mesh/meta and ZIP,
including ordinary errors, interrupted inference and failed downloads. A final
privacy cell verifies absence of session directories and drops the token. On hard
kernel/VM termination, `finally` cannot run: **disconnect and delete the runtime**.
Deletion is not secure erasure or a promise about Google's retention. Clear saved
notebook outputs before sharing. Public model caches are left until runtime deletion.

Tests use only synthetic boxes and placeholder bytes. Developer checks:

```powershell
python tools/twin-lab/colab/build_notebook.py
python tools/twin-lab/colab/validate_notebook.py   # requires nbformat in the validation environment
python -m pytest tools/twin-lab/colab/test_check_meta.py -q -p no:cacheprovider
```

`build_notebook.py` embeds only the worker, validator and existing public meshops
code. The notebook has no executions or saved outputs. `validate_notebook.py`
validates with **nbformat**, extracts every code cell and calls **py_compile**;
this is syntax validation, not inference or style lint. Synthetic tests exercise
invalid metadata, transformed GLB nodes, projection signs and cleanup after errors.

Local verification (2026-10-01): nbformat passed, all **6** extracted code cells
passed `python -m py_compile`, and **21** synthetic pytest cases passed. Repo
typecheck and lint passed. The standard `npm run test` encountered Windows
`spawn EPERM` / Vite native-module loading errors; direct Vitest runs with the
thread pool passed **111 avatar-core + 269 web tests**. The web run additionally
used Node's `--experimental-strip-types` and Vitest `--configLoader=native`.

**Untested here:** Colab toolkit/native compilation, gated model loading, L4/A100
VRAM peaks/performance, real background removal/reconstruction, anatomical orientation,
multi-photo camera fit quality and browser upload/download completion. The notebook
must be smoke-tested by the user in Colab before treating it as an operational stage.
