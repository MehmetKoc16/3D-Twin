# twin-lab - photorealistic digital-twin spike lab

Goal: turn AI-generated, consistent views of one person (made from that person's own photos) into a **textured, riggable
3D mesh** that looks like them. This is a spike lab, not shipping code: it is deliberately separate from `apps/web`
and `packages/avatar-core`, runs only on the developer machine and needs a CUDA GPU.

| Folder     | Purpose                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------ |
| `shape/`   | image(s) -> watertight 3D shape (Hunyuan3D-2 shape model), cleanup, normalisation, cameras |
| `texture/` | project the photos onto the mesh (own README)                                              |
| `rig/`     | auto-rigging (own README)                                                                  |

Local-only artefacts (venvs, cloned third-party repos, model weights, run outputs) are git-ignored:
`tools/twin-lab/**/.venv/`, `**/.cache/`, `**/weights/`, `**/outputs/`.

## Privacy rules (strict)

1. Personal images live only in `user-data/twin/` (git-ignored). Never copy them, or anything derived from them
   (cut-outs, previews, meshes with a likeness), into `docs/`, `apps/`, `packages/`, or any tracked path.
2. `user-data/twin/refs/solo.json` lists which reference photos show the user alone. Only those files (plus
   `user-data/twin/front.png` and later `back.png` / `left.png` / `right.png`) may be opened, processed or cropped. The other
   photos contain other people and must never be touched.
3. **Nothing is uploaded, ever**: no Hugging Face Spaces, no web APIs, no Colab. Everything runs locally. Downloading open
   model _weights_ is fine (that is a download, not an upload).
4. Outputs go to `user-data/twin/out/<stage>/` (shape: `user-data/twin/out/shape/`). `shape/generate.py` refuses to write
   user outputs anywhere in the repo outside `user-data/` or a git-ignored `outputs/` folder.
5. No face processing outside the local machine (same rule as the web app: `AGENTS.md`, "Privacy rule").

## Usage: one-command pipeline and single-file bundle

Run from the repository root (the launcher itself needs only the Python standard library):

```powershell
python tools/twin-lab/run_all.py --input-dir user-data/twin --out-dir user-data/twin/out --height-cm 178
python tools/twin-lab/run_all.py --dry-run
python tools/twin-lab/run_all.py --from-stage texture --to-stage bundle --force
# Package existing rig outputs without rerunning reconstruction:
python tools/twin-lab/run_all.py --from-stage bundle
```

The order is **shape -> texture -> refine (when installed) -> head (when integrated) -> bodyfix (when installed) -> rig -> bundle**. Each stage uses
its own `<stage>/.venv/Scripts/python.exe` on Windows or `<stage>/.venv/bin/python` on Unix.
Set up shape, texture and rig using their instructions above / their READMEs. Bundle uses
`bundle/.venv` when present, otherwise `rig/.venv`. Bodyfix always reuses `rig/.venv`.
Install the bundle's pinned dependencies once:

```powershell
tools/twin-lab/rig/.venv/Scripts/python.exe -m pip install -r tools/twin-lab/bundle/requirements.txt
```

Alternatively create `tools/twin-lab/bundle/.venv` and install the same requirements there.
For tests in that environment, also install `pytest==9.1.1`. The rig environment
already includes pytest; when using it for bundle dependencies, run:

```powershell
tools/twin-lab/rig/.venv/Scripts/python.exe -m pytest tools/twin-lab/bundle/tests -p no:cacheprovider
```

`front.png` is required. Existing `back.png`, `left.png`, and `right.png` beside it are
passed automatically to shape (`--variant auto` selects multi-view) and texture. Outputs
are placed under `<out-dir>/{shape,texture,refine,head,bodyfix,rig}/`, followed by `<out-dir>/twin.glb`.
Refine is optional: when `refine/refine.py` exists, its expected CLI is
`refine/.venv/Scripts/python.exe refine/refine.py --in <textured.glb> --out <refined.glb>`.
It must preserve embedded texture/UVs and write the specified GLB. When absent, rig
receives the last available intermediate. See the refine stage's own README when installed.

Bodyfix reads `<input-dir>/measurements.json` (any subset of tape measurements),
writes `bodyfix/bodyfixed.glb` and `bodyfix/bodyfix_report.json`, and passes the
corrected mesh to rig. Without the measurements file it logs a no-op and copies
the input byte for byte. It runs after refine today; the separately added head
launcher block must precede bodyfix, so head's output flows into it.
See [bodyfix usage, measurement definitions and clothing allowances](bodyfix/README.md).

```powershell
tools/twin-lab/rig/.venv/Scripts/python.exe tools/twin-lab/run_all.py --from-stage bodyfix --to-stage rig --dry-run
tools/twin-lab/rig/.venv/Scripts/python.exe tools/twin-lab/bodyfix/bodyfix.py --in user-data/twin/out/refine/refined.glb --measurements user-data/twin/measurements.json --out user-data/twin/out/bodyfix/bodyfixed.glb
```

`--from-stage` / `--to-stage` select an inclusive range from `shape`, `texture`, `refine`, `head` (when integrated), `bodyfix`,
`rig`, `bundle`; earlier-stage inputs must already exist. `--dry-run` prints commands
without running stages, creating files, or requiring their environments. `--force`
rebuilds the selected stages. Otherwise a stage is skipped when all required outputs
are newer than all inputs and local stage code; `<out-dir>/logs/<stage>.state.json`
also detects changed commands/settings (including height and available views) after
the first managed run. Pre-existing outputs without a state file use timestamp checks;
shape also checks the height and view set recorded in its `meta.json`.
An upstream rebuild causes downstream selected stages to rerun. Failed stages invalidate
their state and stop the pipeline. Per-stage logs are overwritten at
`<out-dir>/logs/<stage>.log`; the terminal shows stage labels and elapsed times.
Personal inputs and all their outputs must remain under `user-data/`.

The bundler can also run directly:

```powershell
tools/twin-lab/rig/.venv/Scripts/python.exe tools/twin-lab/bundle/write_twin_glb.py --rigged user-data/twin/out/rig/rigged.glb --twin user-data/twin/out/rig/twin.json --mh2twin user-data/twin/out/rig/mh2twin.bin --out user-data/twin/out/twin.glb --shape Hunyuan3D-2 --license "Tencent Hunyuan 3D 2.0 Community License"
```

Optional `--texture <image>` overrides only skin-colour sampling; the GLB's embedded
baseColor texture remains unchanged. Without it, sampling uses that embedded texture.
The colour is the per-channel median of distinct texels at vertices whose
`lowerarm_l` or `lowerarm_r` weight exceeds 0.5, with glTF's top-left UV origin and
sampler wrapping. Transparent texels, luminance <=20 / >=235, and the remaining
5th/95th percentile luminance tails are excluded. No usable samples is an error.
Direct invocation defaults provenance shape/license to `unspecified`; set them for
your actual source. The full pipeline records Hunyuan provenance and its community
license; bundling does not change that license's local-use restrictions described below.

The single-file contract is `asset.extras.dtTwin` with `version: 1`, the full input
`twin` object, `skinToneHex`, `provenance {shape, license, createdAt}`, and
`mh2twin {bufferView, count, componentType: "uint32"}`. The mapping buffer view is
four-byte aligned inside the GLB BIN chunk, contains exactly `count * 4` bytes of
little-endian uint32 data, and has no accessor or GPU target. The web loader must
read it using that buffer view's byte offset **relative to the BIN chunk**, not the
whole file, and use the embedded twin object instead of following its legacy
`glb` / `mapping.file` names. Mesh, skeleton, texture, unknown extensions and other
extras are preserved. A single skinned primitive matching `rig.json` is required.
The writer re-reads and validates the completed bundle before atomically replacing
the destination. Its CLI reports only file size and validation status.

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
  generated (AUP 12); do not impersonate people without consent (AUP 13) - the twin is of the user themself; when
  distributing the model or products using it, ship the licence text and the "Powered by Tencent Hunyuan" notice / disclose the
  actual provider and no Tencent affiliation (section 3). Tencent claims no rights in the Outputs (6(d)). Governing law: Hong Kong SAR.
- This lab does not redistribute weights or code: `weights/` and `.cache/` are git-ignored.
- Other components: rembg (MIT) with the `u2net_human_seg` model (Apache-2.0), trimesh (MIT), fast-simplification (MIT), PyTorch (BSD),
  transformers / diffusers / accelerate (Apache-2.0), OpenCV (Apache-2.0), numpy / scipy (BSD).
  `pymeshlab` (GPL-3) is intentionally not installed.

### Measured on the dev machine (RTX 4050 Laptop, 6 GB, Windows 11, single front view `front.png`)

"Device peak" is the whole GPU memory in use during generation (other processes + CUDA context ~1 GB baseline included); "IoU" is
the silhouette IoU of the fitted front camera (higher = the mesh matches the photo's proportions better). Model load is
~30 s (disk bound, weights are ~4 GB) and is not included in "generation".

| Variant / settings                    | Steps | Octree | Decoder  | Offload | Generation | Device peak | IoU   |
| ------------------------------------- | ----- | ------ | -------- | ------- | ---------- | ----------- | ----- |
| mini-turbo                            | 5     | 380    | FlashVDM | none    | 8 s        | 6140 (full) | 0.859 |
| mini-turbo                            | 5     | 256    | FlashVDM | none    | 9 s        | 5998        | 0.860 |
| mini                                  | 50    | 256    | standard | none    | 41 s       | 5684        | 0.909 |
| mini                                  | 50    | 380    | standard | none    | 94 s       | 6140 (full) | 0.910 |
| mini                                  | 50    | 380    | standard | lite    | 100 s      | 4510        | 0.910 |
| mini                                  | 50    | 380    | FlashVDM | lite    | 15 s       | 4510        | 0.909 |
| **mini (default)**                    | 50    | 512    | FlashVDM | lite    | **16 s**   | **4694**    | 0.909 |
| mini                                  | 50    | 640    | FlashVDM | lite    | 20 s       | 6140 (full) | 0.909 |
| mv-turbo (smoke test, fake back view) | 5     | 380    | FlashVDM | lite    | 10 s       | 5640        | -     |

Take-aways: the 50-step `mini` DiT gives much better proportions than the 5-step turbo DiT (turbo is ~6 % too wide relative to its
height, IoU 0.86 vs 0.91); the FlashVDM VAE decoder makes the 50-step model as fast as the turbo one with the same quality, and
the standard hierarchical volume decoder alone took 80 s of the 94 s; `--offload lite` costs ~6 % time and saves ~1.5 GB (device peak 6.1 ->
4.5 GB); octree 512 adds no visible detail over 380 (face detail is limited by the model, not the grid) and 640 hits the 6 GB
ceiling. Other seeds (7, 99) give the same body with slightly different hair / face relief (IoU 0.907 / 0.908). Taubin
smoothing (`--smooth-iters 8`) makes no visible difference. Default end-to-end run: ~60 s including model load, cut-out, cleanup,
camera fit and previews. The default output is watertight, one component, 80 000 faces (40 002 vertices), 1.78 m, 97 L.

### Visual assessment (untextured mesh vs the input photo)

- **Good**: overall silhouette (IoU 0.91), height/leg/arm proportions, shoulder width, T-shirt sleeves and jeans folds, sneakers,
  ears, the quiff-like hair mass, the back of the body. Orientation and scale need no correction (Y up, +Z front).
- **Weak**: the _clothes are baked into the geometry_ (T-shirt, jeans, shoes), so the mesh is a clothed body, not a nude base
  body; the wardrobe / try-on feature cannot dress it as is. The **face is generic**: soft, eyes are shallow slits, no lips,
  no likeness in the relief (likeness has to come from the texture). The hair is flatter and lower than in the photo (the
  photo's top hair reaches ~2.5 % of the height higher). **Depth is guessed** from one view: belly and buttocks look heavier than
  the photo suggests (97 L clothed is probably too much for a slim 178 cm person). **Hands** are mittens / fists without separate fingers
  (thumb only), **feet** are chunky sneakers whose jeans hem merges into the shoe, no laces. Small artefacts: marching-cubes
  ripple on flat cloth at the 256 grid (gone at 380+), irregular triangle layout after decimation.

### What multi-view (left / back / right) would improve

The `mv` models take front + left + back (+ right) and fix exactly what a single view cannot see: real body depth (belly,
chest, buttocks), the back of the head and hair, the side profile of face / nose, and arm / leg thickness. The cameras
of every view are fitted separately, so the texture stage gets one camera per photo. Views must be consistent (same
person, same pose, same A-pose, same framing height); pass them as `--front/--left/--back/--right`. `mv-turbo` fits in 6 GB
(device peak ~5.6 GB at octree 380 with `--offload lite`; the tool falls back to a lower resolution on OOM).

### Next steps / ideas

- Nude or minimal-clothing views would give a body that can be dressed; alternatively treat this mesh only as a proportion
  reference and fit the MakeHuman body to it (measure the mesh: circumferences, lengths), keeping MakeHuman for wardrobe/rig.
- Head: replace / refine the face with a dedicated head reconstruction (from the face-front photos in `solo.json`) and blend it
  into the neck; Hunyuan geometry stays a low-frequency proxy.
- Fix the fit mismatch at the silhouette by warping the photo onto the mesh silhouette (2D flow) before texture baking.
- UVs are not created here (texture stage: xatlas or similar).

## Shape on Colab (licence-clean)

The user-operated [TRELLIS.2 Colab notebook](colab/trellis2_shape.ipynb) targets Colab Pro
**L4 (24 GB)** or preferably **A100**, with a top parameter cell and **Runtime → Run all**.
T4 (16 GB) is unsupported. Microsoft's code/4B weights are MIT; research found that its
required DINOv3 encoder has separate custom terms, so the complete pipeline cannot be
called permissive-only. The notebook requires acknowledging that caveat, replaces BRIA's
noncommercial background remover and avoids NVIDIA's noncommercial PBR exporter.
See [setup, licences and limitations](colab/README.md) before running.

Only front conditions the shape; optional back/left/right views get camera fits. The
download contains `mesh.glb`, `meta.json` and cutouts/masks in our existing convention.
Save them under `user-data/twin/out/shape/` and use [check_meta.py](colab/check_meta.py)
to validate them. This requested Colab workflow is an explicit exception to the local-only
privacy rules above: you upload images yourself to Google's session VM; no Drive is mounted,
and automatic cleanup removes uploaded images and derived outputs after download transfer.
Disconnect and delete the runtime afterwards. Colab inference has not been tested here.
