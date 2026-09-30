# Asset pipeline

Converts raw MakeHuman + MPFB2 **CC0 data** (no code from either project) into the web assets in
`apps/web/public/assets/body/`: `base.glb`, `morphs.bin`, `manifest.json`, `rig.json`, `measures.json`, `face-map.json`.
Deterministic: two builds produce byte-identical files. No Blender needed.

## Run (Windows 11, Python 3.12)

From the repository root, once:

```powershell
py -3.12 -m venv tools/asset-pipeline/.venv
tools\asset-pipeline\.venv\Scripts\python.exe -m pip install -r tools/asset-pipeline/requirements.txt
```

Then `npm run assets:build` (= `python tools/asset-pipeline/build.py`). If the `python` on `PATH` lacks the packages,
`build.py` re-runs itself with `tools/asset-pipeline/.venv`; with no venv it prints the two setup commands above.

```text
python tools/asset-pipeline/build.py [--out DIR] [--no-fetch] [--check]
  --out DIR     output directory (default apps/web/public/assets/body)
  --no-fetch    do not touch the network, use tools/asset-pipeline/.cache as is
  --check       build into a temp dir and fail (exit 1) if the files in --out differ (CI / pre-commit)
```

The first run fetches the pinned upstream commits (shallow, blobless, sparse: about 180 MB into `.cache/`, gitignored).
`fetch.py` is idempotent: it skips a source that is already at the pinned SHA. Set `DT_ASSET_CACHE` to use another
cache directory.

Tests (about 25 s, builds the assets twice in temp dirs): `tools\asset-pipeline\.venv\Scripts\python.exe -m pytest`
from `tools/asset-pipeline`. The glTF test uses the Khronos validator when `npm` is available (installs
`gltf-validator` into `.cache/node`), and skips only that test otherwise.

## Modules

| file             | role                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| `build.py`       | CLI orchestrator, environment detection                                                                |
| `config.py`      | pinned SHAs, paths, unit constants                                                                     |
| `fetch.py`       | sparse + shallow clone of the pinned sources                                                           |
| `mh_obj.py`      | `base.obj` parser, body extraction, UV-seam split (render vertices), triangulation, normals            |
| `targets.py`     | export catalogue (targets, modifiers), `.target` parser, packing into the combined index space         |
| `macro.py`       | macro variables (tent buckets) and the numpy reference evaluator `MorphSet`                            |
| `rig.py`         | `game_engine` rig -> joint points, `rig.json` bones                                                    |
| `weights.py`     | skin weights -> top-4 `JOINTS_0` / `WEIGHTS_0`                                                         |
| `measures.py`    | measure loops derived from geometry, numpy reference of the measure semantics, driver table            |
| `face_map.py`    | `face-map.json`: MediaPipe canonical landmarks -> head surface (alignment, binding, regions, fit set)  |
| `debug_png.py`   | numpy + zlib PNG helpers for the debug images (`.cache/debug/face_map_uv*.png`)                        |
| `gltf_writer.py` | `base.glb` via pygltflib: skinned mesh, skeleton nodes, inverse bind matrices, one material, no morphs |
| `writers.py`     | JSON writers (stable key order)                                                                        |
| `mhclo.py`       | MHCLO / garment OBJ / MHMAT parsers, licence classification, reconstruction reference                  |
| `garments.py`    | garment templates -> `apps/web/public/assets/garments/` (glb, bind.bin, delete.bin, index.json), ADR 0007 |
| `mh_morph.py`    | float64 morph model on ALL 19158 MakeHuman vertices (helper geometry included), for re-binding + tests |
| `parts.py`       | body parts (eyes, eyebrows, eyelashes, hair) -> `apps/web/public/assets/parts/`, ADR 0008              |
| `parts_tex.py`   | part textures: premultiplied downscale, colour bleed, neutral grey + alpha map, coverage alpha cutoff  |
| `parts_debug.py` | textured z-buffered debug renders (`.cache/debug/parts_*.png`)                                         |

## Outputs

Contract v1.1, described in `docs/ARCHITECTURE.md` (Asset data formats) and `docs/adr/0005-...`. Summary of the
current build:

| file            | size (bytes) | content                                                                    |
| --------------- | ------------ | -------------------------------------------------------------------------- |
| `base.glb`      | 983,164      | 14517 vertices, 26756 triangles, 53 bones, 4 influences max, raw base mesh |
| `morphs.bin`    | 15,789,632   | 311 targets, 986,852 entries of 16 bytes                                   |
| `manifest.json` | 136,204      | 69 joint points, 7 macro variables, 311 targets, 96 modifiers              |
| `rig.json`      | 12,645       | 53 bones, `VERTEX` joint refs into the joint points                        |
| `measures.json` | 3,360        | 11 measures with drivers (`shoulder` is a back-surface polyline)           |
| `face-map.json` | 86,359       | 468 landmark bindings, 898 triangles, regions, uvBounds, 9 fitModifiers    |

`face-map.json` is described in `docs/adr/0006-face-map.md` (contract `FaceMapDef`). The build also writes the debug
images `.cache/debug/face_map_uv.png` (whole body UV layout) and `face_map_uv_zoom.png` (face crop) with the mapped
canonical triangles (orange), eye contours (cyan), lips (magenta), oval (yellow) and collapsed / flipped faces (red).

Index space: `[14517 render vertices] ++ [69 joint points]` = 14586. Meters, +Y up, +Z front, feet on y = 0 for the
neutral body (`groundOffsetY` = 0.817763 m). Rest pose = MakeHuman A-pose, world-aligned bones.

## Licensing and sources

Only **data** is used, all CC0-1.0 (MakeHuman `LICENSE.md` section C, MPFB2 `LICENSE.md` section C). MakeHuman's
application code (AGPL-3.0) and MPFB2's code (GPL-3.0) are neither copied nor imported; the file formats were
re-implemented from the data itself. The generated assets are therefore CC0 as well.

| source                                          | pinned commit                              | used for                                   |
| ----------------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| https://github.com/makehumancommunity/makehuman | `a8bc2d54ff0ac92e78ff71431b1023eda42bf482` | `base.obj`, `.target` files                |
| https://github.com/makehumancommunity/mpfb2     | `3edf9df0551765be43563d047888cf7877eb89b4` | `game_engine` rig JSON + skin weights JSON |
| https://github.com/google-ai-edge/mediapipe     | `9519bb59bf55fc6a79ed5b9f283d72e6cdfb6678` | canonical face model + connection lists    |

The two MediaPipe files (Apache-2.0) are plain downloads verified by sha256 (`config.MEDIAPIPE`), not a git checkout.

Measurement loops are generated from geometry (plane slices between joint points and skin-weight body regions), not
taken from MakeHuman's measurement plugin. MakeHuman's own vertex lists were used read-only as a plausibility
cross-check (the acromion vertex found for the shoulder is the same one MakeHuman uses).

## Design notes and limits

- Age is fixed at 25 years (only `young` targets), race is pre-merged (1/3 each), proportions are not shipped.
- The neutral body is "macro defaults"; `base.glb` is the raw base mesh, so runtime code must always apply the
  morph model (`v = base + sum w_i * delta_i`) before showing the body.
- Measure modifiers: `EXTENDED_RANGE` in `targets.py` lets the solver drivers go beyond +-1 (up to 1.5, linear
  extrapolation) where the mesh stays sane; the manifest carries the per-modifier `min` / `max`.
- `shoulder` is the tape path across the upper back (acromion, C7, acromion), see ADR 0006.
- Circumference loops in `measures.json` are ordered (counter-clockwise from above, first vertex not repeated)
  because the runtime derives the loop plane with Newell's method.
- The render mesh is closed (watertight once UV-seam copies are welded), so volume/mass estimation works on it.
- The bust/chest level is the height of the nipples (found through MakeHuman's `nipple-point` target), the waist is
  the narrowest torso ring between 58% and 70% of body height, the hip is the widest trunk + upper-thigh ring above
  the crotch, the crotch is the lowest vertex on the mid-sagittal plane.
- Size headroom: `morphs.bin` is 15.8 MB of the 20 MB budget. Per-target int16 quantisation (scale in the manifest)
  would halve it if needed.

## Garments (Wave 4a)

`build.py` also writes `apps/web/public/assets/garments/` (`index.json` = `{ version: 1, garments: GarmentTemplateDef[] }`,
per template `<id>.glb`, `<id>.bind.bin`, optional `<id>.delete.bin`) and debug renders `.cache/debug/garment_<id>.png`
(front | side, z-buffered, body triangles under `delete_verts` hidden). Sources are single files of the MakeHuman
community asset packs (`config.GARMENT_ASSETS`, sha256-pinned, fetched by HTTP range into `.cache/garments/`,
about 16 MB). See `docs/adr/0007-garment-templates.md`. Needs Pillow (`requirements.txt`).

## Body parts (Wave 3d)

`build.py` also writes `apps/web/public/assets/parts/` (`index.json` = `{ version: 1, parts: BodyPartDef[], defaults:
{ eyes, eyebrows, eyelashes } }`, per part `<id>.glb`, `<id>.bind.bin`, optional `<id>.delete.bin`) and the debug renders
`.cache/debug/parts_head_front.png`, `parts_head_side.png`, `parts_face_closeup.png`, `parts_eyes_extremes.png` and
`parts_<hair id>.png` (front | side). Sources: single files of the MakeHuman system asset pack and `hair01_cc0`
(`config.PART_ASSETS`, sha256-pinned, range-fetched into `.cache/parts/`, about 25 MB), all CC0. Eyes, eyelashes and the
long / ponytail hair are authored against helper vertices that are not in the runtime index space; they are re-bound to
body triangles over 48 sample bodies. See `docs/adr/0008-body-parts.md`.
