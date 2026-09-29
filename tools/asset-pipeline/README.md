# Asset pipeline

Converts raw MakeHuman + MPFB2 **CC0 data** (no code from either project) into the web assets in
`apps/web/public/assets/body/`: `base.glb`, `morphs.bin`, `manifest.json`, `rig.json`, `measures.json`.
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

Tests (about 5 s, builds the assets twice in temp dirs): `tools\asset-pipeline\.venv\Scripts\python.exe -m pytest`
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
| `gltf_writer.py` | `base.glb` via pygltflib: skinned mesh, skeleton nodes, inverse bind matrices, one material, no morphs |
| `writers.py`     | JSON writers (stable key order)                                                                        |

## Outputs

Contract v1.1, described in `docs/ARCHITECTURE.md` (Asset data formats) and `docs/adr/0005-...`. Summary of the
current build:

| file            | size (bytes) | content                                                                    |
| --------------- | ------------ | -------------------------------------------------------------------------- |
| `base.glb`      | 983,164      | 14517 vertices, 26756 triangles, 53 bones, 4 influences max, raw base mesh |
| `morphs.bin`    | 15,789,632   | 311 targets, 986,852 entries of 16 bytes                                   |
| `manifest.json` | 136,204      | 69 joint points, 7 macro variables, 311 targets, 96 modifiers              |
| `rig.json`      | 12,645       | 53 bones, `VERTEX` joint refs into the joint points                        |
| `measures.json` | 3,290        | 11 measures with drivers                                                   |

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

Measurement loops are generated from geometry (plane slices between joint points and skin-weight body regions), not
taken from MakeHuman's measurement plugin. MakeHuman's own vertex lists were used read-only as a plausibility
cross-check (the acromion vertex found for the shoulder is the same one MakeHuman uses).

## Design notes and limits

- Age is fixed at 25 years (only `young` targets), race is pre-merged (1/3 each), proportions are not shipped.
- The neutral body is "macro defaults"; `base.glb` is the raw base mesh, so runtime code must always apply the
  morph model (`v = base + sum w_i * delta_i`) before showing the body.
- Circumference loops in `measures.json` are ordered (counter-clockwise from above, first vertex not repeated)
  because the runtime derives the loop plane with Newell's method.
- The render mesh is closed (watertight once UV-seam copies are welded), so volume/mass estimation works on it.
- The bust/chest level is the height of the nipples (found through MakeHuman's `nipple-point` target), the waist is
  the narrowest torso ring between 58% and 70% of body height, the hip is the widest trunk + upper-thigh ring above
  the crotch, the crotch is the lowest vertex on the mid-sagittal plane.
- Size headroom: `morphs.bin` is 15.8 MB of the 20 MB budget. Per-target int16 quantisation (scale in the manifest)
  would halve it if needed.
