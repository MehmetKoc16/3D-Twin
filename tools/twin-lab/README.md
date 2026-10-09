# twin-lab - offline digital-twin tools

This lab prepares private textured, rigged bundles for the web app's twin mode. `--body scan` retains a generated scan;
`--body hybrid` builds a MakeHuman template character with a fitted head and fixed UVs. The Python tools are separate
from `apps/web` and `packages/avatar-core`. Local shape inference needs CUDA; the remaining geometry/texture stages
use their documented environments. Colab notebooks are separate, user-operated alternatives for shape and head fitting.

## Stages and environments

Paths below are relative to `tools/twin-lab/`. Windows interpreters are `<venv>/Scripts/python.exe`; Unix uses
`<venv>/bin/python`. Output names are relative to `--out-dir` unless otherwise stated. The table includes standalone
helpers as well as launcher stages; it is not an execution order.

| Folder                                  | Purpose                                                                           | Venv / runtime                                                                     | Inputs                                                                                                                             | Outputs                                                                                                                                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `shape/`                                | Hunyuan3D shape generation, cleanup, normalisation and camera fits                | `shape/.venv` (CUDA)                                                               | `front.png`, optional back/left/right views, ignored weights                                                                       | `shape/mesh.glb`, OBJ, `mesh_hires.glb`, `raw.glb`, `meta.json`, cutouts/masks, previews                                                                                                   |
| [texture/](texture/README.md)           | UV unwrap and photo projection                                                    | `texture/.venv`                                                                    | shape mesh/meta/cutouts and full-body views                                                                                        | `texture/textured.glb`, `baseColor.png`, `report.json`, cache and previews                                                                                                                 |
| [refine/](refine/README.md)             | Landmark-driven face relief/rebake and armpit separation                          | `refine/.venv`                                                                     | textured GLB, front view (optional higher-resolution face view), app assets/landmarker                                             | `refine/refined.glb`, `refine_report.json`, fit cache and previews                                                                                                                         |
| [head/flame/](head/flame/README.md)     | Replace scan facial surface/ears with an exported FLAME fit                       | `refine/.venv`                                                                     | current scan, neutral fit/cameras/posed views, front/right photos, FLAME masks/embedding                                           | `head/head.glb`, `flame_head_report.json`, previews                                                                                                                                        |
| `head/recon/`                           | Experimental four-photo head deformation and retexture (`--head recon`)           | `refine/.venv`; optional masks via `shape/.venv`                                   | current scan; `front.jpg`, `back.jpg`, `profile_nose_right.jpg`, `profile_nose_left.jpg`; optional cleaned photos/masks            | `head/head.glb`, `head_report.json`, previews                                                                                                                                              |
| [head/glasses/](head/glasses/README.md) | Standalone parametric head-local glasses generator                                | `bundle/.venv` (or `rig/.venv` with bundle dependencies)                           | twin GLB, optional `glasses.json` and refine landmarks                                                                             | requested `glasses.glb`, embedded later by bundle; not the automatic launcher's `glasses` implementation                                                                                   |
| [head/deglass/](head/deglass/README.md) | Standalone offline photo frame masking/inpainting and dimensional estimates       | `head/deglass/.venv`                                                               | head photos, local MediaPipe + Big-LaMa ONNX models                                                                                | requested clean JPEGs, frame masks, `report.json`, debug images; `measure.py` writes `glasses.json`                                                                                        |
| [bodyfix/](bodyfix/README.md)           | Correct scan to measurement targets and remove fused scan hands                   | `rig/.venv`                                                                        | latest scan, optional `measurements.json`, app body assets                                                                         | `bodyfix/bodyfixed.glb` with `dtBodyfix`/hand metadata, `bodyfix_report.json`; absent/empty measurements give a byte-exact no-op                                                           |
| [hybrid/](hybrid/README.md)             | Native MakeHuman template body/head registration, skin/face atlas, parts and hair | `refine/.venv`                                                                     | bodyfix solution, measurements, FLAME fit/masks/embedding, selected head photos, app body/parts; optional bust and cached CC0 skin | `hybrid/hybrid.glb`, `hybrid_report.json`, `face_asset/`, previews; automatic post-rig glasses also writes `hybrid/glasses.glb`, `glasses_rigged.glb`, report and cleaned texture/previews |
| [rig/](rig/README.md)                   | Fit/unpose scan or verify native hybrid; transfer 53-bone weights                 | `rig/.venv`                                                                        | current GLB, app body assets, optional embedded solution/hair                                                                      | `rig/{rigged.glb,twin.json,mh2twin.bin}`, reports/cache/previews (hybrid: `hybrid/rig/`)                                                                                                   |
| `bundle/`                               | Validate and embed definition, mapping, texture and optional accessory            | `bundle/.venv`, fallback `rig/.venv`                                               | rigged GLB, `twin.json`, `mh2twin.bin`, optional glasses GLB                                                                       | `twin.glb` (hybrid: `hybrid/twin.glb`, override `--bundle-out`)                                                                                                                            |
| [colab/](colab/README.md)               | User-operated TRELLIS.2 shape and Pixel3DMM FLAME-fit notebooks                   | TRELLIS isolated VM venv; Pixel3DMM micromamba Python 3.9 env; no local stage venv | explicitly supplied views/model archives, VM-downloaded weights                                                                    | `trellis2_shape.zip` (shape contract) or `head_fit.zip` (neutral head, parameters, cameras, posed views, provenance); extract only under `user-data/`                                      |

`run_all.py` orders stages as **shape -> texture -> refine -> head -> bodyfix -> hybrid -> rig -> glasses -> bundle**.
Refine/bodyfix are included when their scripts exist. Scan omits hybrid and automatic glasses; hybrid requires bodyfix
and FLAME. The launcher constructs this graph before filtering the inclusive `--from-stage` / `--to-stage` range.

- `--body scan` is the default. Rig receives `--fingers merge --cut-bridges`; output is `<out-dir>/twin.glb`.
- `--body hybrid` retains the MakeHuman body topology and its native hands/feet. Earlier scans provide a bodyfix shape
  prior; hybrid re-solves the native body without scan clothing allowance, deforms its own head to FLAME and bakes into
  fixed UVs. Rig receives `--fingers keep --smooth 0`; output is `<out-dir>/hybrid/twin.glb`.
- `--head auto` (default) selects FLAME when `<input-dir>/head/flame/fit/head_neutral.obj` exists, otherwise none.
  Explicit `flame`, `recon`, `none` are supported. Hybrid requires `flame` or a successful auto selection.
  `--with-head` is a deprecated recon alias. FLAME precedes bodyfix even for a full hybrid run.
- Hybrid's automatic `glasses` stage runs after rig when the default `<input-dir>/hy3d/hy3d.glb` exists or
  `--glasses-hy3d` is supplied, unless `--no-glasses`. It invokes `hybrid/hybridbody/glasses_hy3d.py` in the refine venv,
  not `head/glasses/make_glasses.py`. Standalone photo deglass is not a launcher stage; hybrid has its own atlas cleanup.

## Privacy and licensing boundaries

[AGENTS.md](../../AGENTS.md) is authoritative. Personal inputs and **all** derived outputs, reports, previews and
screenshots remain under gitignored `user-data/`; never copy them into docs, app assets, packages or fixtures.
Use only explicitly authorised input files. The web app's face processing stays in the browser; local lab stages run
on the local machine and do not upload inputs. Optional Colab notebooks require a separate user-operated session upload
and the documented cleanup; they are not called by the launcher or app. Downloaded source/model caches and stage venvs
stay ignored (`**/.cache/`, `**/weights/`, `**/outputs/`, `**/.venv/`).

Only CC0, CC-BY, MIT, Apache-2.0 or BSD assets may be committed; CC-BY assets require [credits](../../CREDITS.md).
Research/non-commercial models may be used locally or in Colab under their own terms, with ignored downloads and private
outputs. FLAME/Pixel3DMM-derived hybrids are not permissive web assets despite their CC0 template. Bust-derived hair or
accessories retain their source restrictions as well. Bundle provenance is a short source label, not licence clearance.

## Common runs (repository root, PowerShell)

The launcher uses only Python's standard library. Set up stage dependencies first, using each stage's README.
Bundle may share the rig environment:

```powershell
tools/twin-lab/rig/.venv/Scripts/python.exe -m pip install -r tools/twin-lab/bundle/requirements.txt
```

Alternatively create `bundle/.venv` and install those requirements there. Its standalone tests also need
`pytest==9.1.1` (already in the rig environment).

Prepare a FLAME fit before running hybrid: follow [Pixel3DMM setup](colab/pixel3dmm_README.md), extract the exported fit
into `user-data/twin/head/flame/fit/`, keep original matching `front.jpg`/`right.jpg` in `head/colab_upload/`, and obtain
FLAME masks/MediaPipe embedding into `user-data/flame/`. A full run also needs `front.png`, optional back/left/right PNGs
and `measurements.json` under the input directory. This command does not create the Pixel3DMM fit itself.

```powershell
# Full scan path; --head auto uses a prepared fit when present.
python tools/twin-lab/run_all.py --body scan --input-dir user-data/twin --out-dir user-data/twin/out --height-cm 178
# Full hybrid path, including upstream shape/texture/head/bodyfix.
python tools/twin-lab/run_all.py --body hybrid --head flame --input-dir user-data/twin --out-dir user-data/twin/out --height-cm 178
# Resume at hybrid: bodyfix and FLAME inputs must already exist.
python tools/twin-lab/run_all.py --body hybrid --head flame --from-stage hybrid --to-stage bundle --force
# Re-package existing hybrid rig (and the automatic accessory if configured).
python tools/twin-lab/run_all.py --body hybrid --head flame --from-stage bundle --force
# Print the selected commands without running stages or requiring venvs.
python tools/twin-lab/run_all.py --body hybrid --head flame --from-stage hybrid --dry-run
```

`--force` rebuilds the selected range. Without it, all required outputs must be newer than inputs and local stage code;
`<out-dir>/logs/<stage>.state.json` tracks the command and input list. Existing outputs without state use timestamp
checks (shape also verifies height/views from `meta.json`). Upstream rebuilds invalidate downstream selected stages;
failed stages invalidate state and stop. Logs are overwritten in `<out-dir>/logs/<stage>.log`. Earlier inputs must exist
when resuming; `--dry-run` prints commands without creating files or checking interpreters.

### Photo sets, hair and cleanup flags

`--photos-set`, `--hair` and `--keep-neck-hair` are **hybrid.py flags**, not `run_all.py` flags. The launcher uses their
defaults. `--photos-set auto` requires the complete `nog_front.jpeg`, `nog_left.jpeg`, `nog_right.jpeg` set when its
front file exists in `head/` (or the supplied photo directory), otherwise uses `colab_upload/{front,right}.jpg` and
exported cameras. `noglasses` requires the three-view set; `glasses` chooses the legacy two-view bake. The glasses-free
path runs local `hybridbody/photofit.py` on the existing FLAME identity, fitting robust perspective cameras using
MediaPipe + FLAME's landmark embedding. It disables deglass automatically, uses both observed sides without mirrored
fill and writes private numeric fit diagnostics; it does not refit FLAME identity/expression.

Hair defaults to `hy3d` when the default private bust exists, otherwise `procedural`. `hy3d` writes a separate `dtHair`
shell (`shell/1`); procedural writes separate cards/data atlas (`rcov-groot-bvar/1`); MakeHuman part IDs such as
`hair-short` use the legacy merged body atlas. `--keep-neck-hair` is a QA override of the default neck/behind-ear texture
cleanup. `--no-deglass` is accepted by **both** CLIs; `--no-glasses` belongs to the launcher and also disables deglass.

```powershell
# Explicit automatic photo selection and procedural hair; default neck-hair cleanup stays enabled.
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/hybrid/hybrid.py --bodyfix user-data/twin/out/bodyfix --measurements user-data/twin/measurements.json --fit user-data/twin/head/flame/fit --photos user-data/twin/head/colab_upload --flame-assets user-data/flame --out user-data/twin/out/hybrid/hybrid.glb --photos-set auto --hair procedural
# QA variant: require glasses-free views, shell hair, retain neck hair and skip frame cleanup.
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/hybrid/hybrid.py --bodyfix user-data/twin/out/bodyfix --measurements user-data/twin/measurements.json --fit user-data/twin/head/flame/fit --photos user-data/twin/head/colab_upload --flame-assets user-data/flame --out user-data/twin/out/hybrid/hybrid.glb --photos-set noglasses --hair hy3d --keep-neck-hair --no-deglass
# After either direct hybrid command, resume at RIG so the launcher does not rebuild hybrid with its defaults.
python tools/twin-lab/run_all.py --body hybrid --head flame --from-stage rig --to-stage bundle --force
# Disable the accessory and atlas frame cleanup (hair selection is unchanged).
python tools/twin-lab/run_all.py --body hybrid --head flame --from-stage hybrid --no-glasses --force
# Preserve photographed frames while still adding the accessory when a bust is available.
python tools/twin-lab/run_all.py --body hybrid --head flame --from-stage hybrid --no-deglass --force
```

For legacy cameras use the first direct command with `--photos-set glasses`; select a shipped MakeHuman part with
`--hair hair-short` instead. See [hybrid options and formats](hybrid/README.md) for tuning. The launcher's hybrid cache
currently inventories legacy front/right JPEGs, not the `nog_*.jpeg` inputs: after replacing those photos use
`--from-stage hybrid --force`. The bust is tracked when present; custom direct-CLI settings must be retained by resuming
at rig. With `--no-glasses`, launcher also passes `--no-deglass`; without a bust it does the same and omits glasses.

### Bundle contract and direct packaging

```powershell
tools/twin-lab/rig/.venv/Scripts/python.exe tools/twin-lab/bundle/write_twin_glb.py --rigged user-data/twin/out/rig/rigged.glb --twin user-data/twin/out/rig/twin.json --mh2twin user-data/twin/out/rig/mh2twin.bin --out user-data/twin/out/twin.glb --shape Hunyuan3D-2 --license "Tencent Hunyuan 3D 2.0 Community License"
```

For hybrid direct packaging, use `hybrid/rig/` inputs and `hybrid/twin.glb` output, and label the actual sources
(including any restricted bust source). `--glasses <glasses.glb>` embeds a head accessory. Optional `--texture <image>`
overrides only forearm skin-colour sampling, not the embedded baseColor texture. Without it the bundled albedo is
sampled at lowerarm-weighted vertices, excluding transparent texels, clipped luminance and tails; no samples is an error.
Direct provenance defaults to `unspecified`; launcher supplies scan/hybrid source labels.

`asset.extras.dtTwin` version 1 embeds full `twin.json`, `skinToneHex`, provenance and
`mh2twin {bufferView, count, componentType: "uint32"}`. The mapping is aligned little-endian uint32 in the BIN chunk,
exactly `count * 4` bytes; buffer-view offsets are relative to BIN, not the whole GLB. Mapping/definition describe the
body primitive only. One body primitive is required; separate named hair is supported and shares the rig skin.
Optional `dtTwin.accessories` embeds a complete head-local glasses GLB. Mesh/material/image/normal-map metadata and
extras (`dtBodyfix`, `dtHybrid`, `dtFlameHead`, hand flags, `dtHairNode`) pass through; the writer validates a reread
before atomically replacing output. See [the architecture contract](../../docs/ARCHITECTURE.md#twin-bundle-contract)
for app validation, both hair formats and hand behaviour.

### App and wrist QA

Install npm dependencies and run `npm run dev` in another terminal. The browser tools use installed Chrome/Playwright;
`app_qa.mjs` accepts `--url=...`, while viewer-performance reads `VIEWER_URL` (default `http://localhost:5173`).

```powershell
# Browser poses, camera presets and wardrobe; personal screenshots stay under user-data.
node tools/twin-lab/rig/app_qa.mjs user-data/twin/out/hybrid/rig user-data/twin/out/app --tag=hybrid
# Optional legacy picker path (rigged.glb + twin.json + mapping).
node tools/twin-lab/rig/app_qa.mjs user-data/twin/out/hybrid/rig user-data/twin/out/app --tag=legacy --legacy
# Headless skinning/cross-section QA using the same pose JSONs; optional private hand crops.
New-Item -ItemType Directory -Force user-data/twin/out/wrist | Out-Null
node tools/twin-lab/rig/wrist_qa.mjs user-data/twin/out/hybrid/rig/rigged.glb --json=user-data/twin/out/wrist/metrics.json --png=user-data/twin/out/wrist/png
# Permissive mannequin comparison, also safe for contributor QA.
node tools/twin-lab/rig/wrist_qa.mjs apps/web/public/assets/body/base.glb --json=user-data/twin/out/wrist/mannequin.json
# Synthetic standard/twin orbit benchmark: run from apps/web because fixture paths are relative.
Push-Location apps/web
node scripts/viewer-performance.mjs test-results/viewer-performance
Pop-Location
```

`app_qa.mjs` prefers `<twinDir>/../twin.glb`, then `<twinDir>/twin.glb`; `--legacy` forces sidecars and mapping remains
optional. `--no-wardrobe` skips clothing checks. Wrist QA reports palm direction, swing/twist and posed/rest forearm/wrist
cross-section area for all six poses by default (`--poses=a-pose,relaxed,t-pose,walk,hands-on-hips,side`). It does not need
a dev server. Generated pose JSONs use the corrected palm convention and shared forearm roll from fa2e7a6; see
[pose library](../../apps/web/src/features/poses/README.md).

Viewer-performance always uses the CC0 synthetic twin fixture. It records renderer/drawing buffer, mean/p95 frame
intervals and FPS after orbit warmup, for standard/twin in High/Performance, plus screenshots, console errors and full
HTTP failure URLs. Timing is requestAnimationFrame/vsync limited, not a GPU timer or a hardware performance guarantee.
Never use personal outputs as test fixtures or publish QA results containing personal data.

## Head reconstruction alternatives

FLAME scan insertion details and input schemas are in [head/flame](head/flame/README.md); the Colab fit instructions
are [here](colab/pixel3dmm_README.md), with [multi-photo naming/settings](colab/pixel3dmm/MULTI_PHOTO.md).
The recon alternative uses four head photos and refine's venv:

```powershell
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/head/recon/head.py --in user-data/twin/out/refine/refined.glb --photos user-data/twin/head --out user-data/twin/out/head/head.glb
```

`--use-clean` reads `clean/<name>.jpg` and frame masks. Recon fits perspective cameras/silhouettes and deforms/retextures
the head, fading into the neck; it is experimental and selected explicitly. Standalone [deglass](head/deglass/README.md)
cleans photos, while [parametric glasses](head/glasses/README.md) builds a procedural accessory. Neither is implicitly
run by `--head recon`.

## Shape stage (`shape/`)

Pipeline: background removal (rembg `u2net_human_seg`, Apache-2.0) -> **Hunyuan3D-2mini** (0.6 B, fits 6 GB) shape
diffusion + FlashVDM VAE decode -> mesh cleanup (weld, floaters, holes, normals, quadric decimation) -> orient/scale to the
app convention -> silhouette camera fit -> `mesh.glb`, `mesh.obj`, `meta.json`, previews.
The texture/paint model of Hunyuan3D-2 is **not** used (too big for 6 GB; texture is the `texture/` stage).

### Setup (Windows 11, Python 3.12, NVIDIA GPU with >= 6 GB)

```powershell
cd tools\twin-lab\shape
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 .cache\Hunyuan3D-2      # pinned at f8db630 when this lab was set up
.\.venv\Scripts\python download_weights.py mini mini-turbo mv-turbo                          # weights land in shape\weights\
```

`generate.py` imports `hy3dgen.shapegen` straight from `.cache/Hunyuan3D-2` and stubs the `pymeshlab` import (GPL, only used
by hy3dgen's own post-processors, which this tool replaces with trimesh + fast-simplification). The rembg model
(`u2net_human_seg`, 176 MB) is downloaded on first use into `shape/weights/rembg/`.

### Run

```powershell
# one view (defaults: mini-turbo, octree 380, 5 steps, 80k faces, 178 cm)
.\.venv\Scripts\python generate.py --front ..\..\..\user-data\twin\front.png --out ..\..\..\user-data\twin\out\shape
# quality variant (50 diffusion steps + CFG), smaller grid
.\.venv\Scripts\python generate.py --front ..\..\..\user-data\twin\front.png --variant mini --octree-resolution 256 --name mini_o256
# multi view (needs weights `mv-turbo`): the shape uses all views; every view also gets its own camera fit
.\.venv\Scripts\python generate.py --front f.png --left l.png --back b.png --variant mv-turbo
# re-run only cleanup / normalisation / camera fit on an earlier raw mesh (no GPU)
.\.venv\Scripts\python generate.py --front f.png --from-raw ..\..\..\user-data\twin\out\shape\raw.glb --out ...
# standalone previews of any normalised mesh
.\.venv\Scripts\python preview.py mesh.glb previews_dir
```

Important flags: `--variant {auto,mini,mini-turbo,mv,mv-turbo}`, `--octree-resolution`, `--steps`, `--seed`,
`--num-chunks` (VAE decode chunk, lower = less VRAM), `--offload lite` (DINOv2 conditioner + VAE stay on the CPU between calls),
`--height-cm` (default 178), `--target-faces` (default 80000, 0 = no decimation), `--floater-ratio`, `--smooth-iters`,
`--rembg-model`. On CUDA OOM the tool retries with half the chunk size, then with the next lower octree resolution
(disable with `--no-fallback`). Before touching the GPU it waits (polls every 30 s, up to 10 min) while other processes use
more than 1 GB of VRAM (`--force-gpu` skips that).

### Outputs (`<out>/`)

| File                                   | Content                                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `mesh.glb`, `mesh.obj`                 | cleaned, decimated, normalised mesh (vertex normals, no UVs, no texture), app convention below |
| `mesh_hires.glb`                       | same, before decimation (typically ~300k faces)                                                |
| `raw.glb`                              | untouched generator output in the generator's own frame/scale (input of `--from-raw`)          |
| `cutout_<view>.png`, `mask_<view>.png` | RGBA cut-out fed to the model, and the 0/255 person mask (same pixel grid as the input photo)  |
| `camera_<view>_overlay.png`            | half-size debug image: green = photo mask, red = projected mesh, yellow = both                 |
| `meta.json`                            | everything below                                                                               |
| `previews/*.png`                       | flat-shaded front / 3-4 / side / back, head, hands, feet, `contact.png`                        |

**Coordinate convention of the output** (identical to the web app, see `docs/ARCHITECTURE.md`): metres, right handed, +Y up,
the character faces +Z (the character's left = +X), `y = 0` at the lowest vertex (soles). Horizontal origin `x = z = 0`
is the middle of the bounding box of the lowest 3 % of the mesh (contact patch of the feet). The mesh height (soles to the
highest hair point) equals `--height-cm` (default 178; note that soles and hair are included, so the value is a little more
than the barefoot body height).

**How the generator's frame is mapped** (`meta.json > normalization`): the raw output is oriented by an automatic detector
(largest bbox axis = up, sign from "feet footprint >> head-top footprint"; forward = the smaller horizontal axis, sign from
"soles' centroid lies in front of the shank centroid"), then
`p_final = (R @ p_source) * scale + translation`, with `R` a proper rotation (so triangle winding is unchanged),
`scale` = metres per generator unit and `translation` putting the feet on `y = 0`. `matrix4x4RowMajor` is that transform
as one 4x4. For Hunyuan3D-2 the source frame is already Y-up / +Z-forward (`R = I`), so the detector normally confirms it.

**Camera block** (`meta.json > cameras.views.<view>`): one entry per input view. Model = **orthographic** camera; for a world
point `(x, y, z)` in the final frame (metres):

```text
x'  = x * cos(yaw) - z * sin(yaw)          yaw: front 0, left 90 (camera on the character's left side, +X), back 180, right 270
u   = originPx[0] + pxPerMeter * x'        u, v = pixel coordinates in the ORIGINAL input image (v grows downwards)
v   = originPx[1] - pxPerMeter * y
```

`pxPerMeter` / `originPx` were fitted by maximising the silhouette IoU between the projected mesh and the cleaned person mask
(`silhouetteIoU`; `initialGuess` is the plain bbox alignment, `anisotropicFit` a diagnostic fit with separate horizontal and
vertical scale). A real photo is a perspective view and the generator does not reproduce the photo's proportions exactly, so
the fit is a starting point: expect a few percent (IoU ~0.85-0.9 for a single view) of local misalignment at the silhouette,
and refine per region (or warp the photo onto the mesh silhouette) before baking colours. The model itself only sees a
512 px square crop of the person's bounding box (`inputs.<view>.modelInput`), never the original framing.

### Licence: Tencent Hunyuan 3D 2.0 Community License (read before using outputs beyond this lab)

Hunyuan3D-2 / 2mini / 2mv code and weights: `https://github.com/Tencent-Hunyuan/Hunyuan3D-2` (licence file in the clone at
`shape/.cache/Hunyuan3D-2/LICENSE`, "TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT", release date 2025-01-21).

- **Territory**: worldwide **excluding the European Union, the United Kingdom and South Korea**. The licence "does not apply"
  there. Türkiye is inside the Territory (it is not an EU member), so running the model in Türkiye is licensed.
- Section 5(c) also forbids using, displaying or distributing the **Output** outside the Territory, and the Acceptable Use Policy
  forbids any use outside the Territory. Consequence: the meshes produced with this model must not be shipped to, or shown to,
  users in the EU / UK / South Korea. A public web app that serves them worldwide would break that. Keep the Hunyuan shape as
  a **lab intermediate** (e.g. as a reference for fitting the MakeHuman body, or for a Türkiye-only build) unless legal review
  clears it, or replace it by a different shape source for a global release.
- Other terms: more than 1 M monthly active users needs a separate licence from Tencent; the outputs (or the model) must not be used
  to improve/train any other AI model (5(b)); public content generated with it must be conspicuously labelled as machine
  generated (AUP 12); do not impersonate people without consent (AUP 13); when
  distributing the model or products using it, ship the licence text and the "Powered by Tencent Hunyuan" notice / disclose the
  actual provider and no Tencent affiliation (section 3). Tencent claims no rights in the Outputs (6(d)). Governing law: Hong Kong SAR.
- This lab does not redistribute weights or code: `weights/` and `.cache/` are git-ignored.
- Other components: rembg (MIT) with the `u2net_human_seg` model (Apache-2.0), trimesh (MIT), fast-simplification (MIT), PyTorch (BSD),
  transformers / diffusers / accelerate (Apache-2.0), OpenCV (Apache-2.0), numpy / scipy (BSD).
  `pymeshlab` (GPL-3) is intentionally not installed.

## Shape on Colab

The user-operated [TRELLIS.2 notebook](colab/trellis2_shape.ipynb) targets L4 (24 GB) or preferably A100; T4 is unsupported.
TRELLIS.2 code/4B weights are MIT, but its required DINOv3 encoder has separate custom terms. The workflow is therefore
not permissive-only; it requires a licence acknowledgement, replaces BRIA's remover with rembg/U2Net and avoids the
NVIDIA PBR exporter. See [setup, licences and limitations](colab/README.md).

Only front conditions the shape; optional views provide camera fits/cutouts. Extract `trellis2_shape.zip` under
`user-data/twin/out/shape/`, validate using `colab/check_meta.py`, then resume the local launcher at texture.
The launcher's scan bundle provenance currently remains Hunyuan-labelled even for an imported TRELLIS shape; use the
direct bundler with the actual source/licence label rather than treating that default as accurate provenance.
The notebook involves an explicit session upload; follow its cleanup and disconnect/delete the runtime afterwards.

## Current limits

Scan clothing and fused geometry can limit garment replacement; shape inference estimates unseen depth. Hybrids avoid
scan body topology but inherit FLAME fitting/camera limits, photo lighting and rigid hair geometry. Generated normal
detail is not a measured skin surface. Inspect private outputs locally; synthetic checks establish format/math behaviour.
