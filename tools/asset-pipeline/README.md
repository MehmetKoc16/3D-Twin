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

## Garment import (offline review candidates)

`garment_import.py` converts GLB/glTF, OBJ or authored MHCLO garments into the **existing** wardrobe contract. FBX must first be
converted to GLB in the authoring tool. It writes to `.cache/garment-import-out/` by default and refuses outputs
inside `apps/` or `packages/`. It does not add anything to the shipped wardrobe or update credits automatically.
Run it only with the asset-pipeline venv. Install its optional geometry packages into that same venv:

```powershell
tools\asset-pipeline\.venv\Scripts\python.exe -m pip install -r tools/asset-pipeline/garmentimport/requirements.txt

tools\asset-pipeline\.venv\Scripts\python.exe tools/asset-pipeline/garment_import.py path/to/garment.glb `
  --id candidate-tee --category top --kind tshirt --license CC0-1.0 `
  --attribution 'Title by Author, source URL, CC0-1.0' --source-url 'https://source.example/asset' `
  --pose a --source-body-height 1.75 --validate
```

Licence and attribution are required. `--license` takes the app's `GarmentLicense` catalogue values: **CC0-1.0,
CC-BY-3.0 or CC-BY-4.0** (the app's runtime-only `user` value is not an import licence). Other permissive licences and
other CC-BY versions need an app contract change instead of being relabelled, and a source that only says "CC BY"
without a version must not be assumed to be 4.0. CC-BY attribution must include the title, author, source and exact
licence. The lead must review the source declarations, textures and
derivatives, add any required `CREDITS.md` entry and decide what ships. No licence is inferred from a download folder
or mesh appearance. The importer never reads or changes `garment_sources.json` / `GARMENT_SOURCES.md`.

Processing and outputs:

- Load all garment scene parts and bake node transforms. Skinned GLB sources are evaluated in their default node
  pose (animations are not selected); `--skin-pose bind` instead uses the mesh's authored bind pose with node transforms.
  Inspect the evaluated mesh: a T-pose bind mesh may have a relaxed default skeleton and must not be treated as T-pose
  after evaluation. The Quaternius hoodie review uses `--skin-pose bind --pose t`. Apply source morph targets and expand compressed/sparse accessors or material
  extensions in the converter first. Isolate full-character GLBs with repeated `--node` and `--exclude-material`
  arguments (for example `--node Casual_Body --exclude-material Skin`); otherwise only the garment should be present.
  OBJ MTL materials are loaded normally; same-stem MHMAT companions supply MakeHuman diffuse/normal maps. A legacy
  specular map is reported for PBR conversion instead of being mislabeled as metallic/roughness.
- A `.mhclo` source, `--mhclo companion.mhclo`, or an OBJ with a matching `obj_file` companion uses authored bindings
  directly. The tool reconstructs the neutral garment from hm08 bindings, not the OBJ's authoring body. Body indices
  and scale references survive conversion; negative weights are clamped and their positional difference is folded
  into offsets. Clearance fitting adjusts offsets while retaining these indices; the subsequent top-region stability
  pass may replace selected armpit/side-seam rows and reports that count separately. Helpers are reconstructed with
  `mh_morph.MhMorpher` and the pinned CC0 raw targets, then rebound to render-body triangles because the app has no
  helper index space. Reports count both paths; helper deformation is not preserved under grading. Missing raw data,
  other basemeshes and legacy shear bindings fail explicitly. Transforms/reposing are incompatible with this path.
  `--ignore-mhclo` explicitly selects generic refitting instead. Native authored delete masks are converted to all UV
  copies of each body vertex; helper deletes are reported and omitted.
- Default `--shell outer` removes close opposing inner walls using inward rays, opposite face normals and distance
  from the body (8 mm pairing limit). Open fabric without a paired wall survives. Inspect the resulting source shell,
  especially at folds/linings; inconsistent winding or thicker walls need authoring cleanup. `--shell keep` is for a
  reviewed single shell. This pass is batched to bound ray-query memory.
- Drop orphan, degenerate and duplicate triangles; merge identical position/UV pairs while retaining UV seams.
  OBJ polygons are triangulated by the loader; export complex concave polygons as triangles in the authoring tool.
  Above `--max-tris` (default 30000), quadric decimation runs per UV-connected island, followed by barycentric UV
  transfer within each island. Very fragmented meshes can require a higher budget or source simplification.
  Optional `--subdivide 1/2/3` adds linear midpoint samples before clearance fitting, with exact UV interpolation and
  no exterior smoothing. It fails if subdivision would exceed the triangle budget. The coarse Quaternius hoodie uses
  two levels (20,128 triangles) to prevent large facets crossing the torso despite exterior corner vertices.
- Merge parts into one primitive/material, as required by `templateLoader.ts`. Bake base colour, metallic/roughness,
  normal, occlusion and emissive maps plus scalar factors into padded PBR atlases, each at most `--texture-max`
  (default/max 2048). Alternate texture UV sets and texture transforms must be baked in the converter. Atlasing and
  decimation are lossy, so inspect detailed patterns, seam continuity and normal maps before shipping.
- Infer likely units from category dimensions; scale by body height when `--source-body-height` is supplied. This
  argument is the **full authoring body height in source coordinate units** (e.g. `175` for a cm-space mesh,
  `1.75` for meters). A cropped garment cannot uniquely determine body height. Pose detection compares nearest-body
  distances in A/T; bottoms and sleeveless tops often need `--pose a/t`. Landmarks and trimmed translation ICP refine
  alignment; ICP does not deform or shrink exterior fabric. Nearest-body triangle skin influences proxy-repose T
  garments to the MakeHuman A rest pose by inverting the blended affine skin transform.
- Fit inward/near-surface vertices to category clearance (top/bottom 3 mm, shoes 4 mm; `--clearance-mm` accepts 2..4).
  Outer layers add `(layer - 1) * --layer-gap-mm` (default 3 mm) to the baseline; the report records the total.
  Smooth the displacement field, weld displacement across UV copies, then re-project for clearance. Exterior shape
  stays fixed except where clearance is needed. Opposing surfaces may require a bounded outward escape, reported
  separately. An unresolved inside vertex or unmet minimum clearance fails the import.
  Tops use compact smoothstep fields around the shoulder underside and elbow: up to 4 mm and 3 mm extra
  clearance respectively, blended into the category/layer baseline. These are local minimum clearances, not
  additional offsets on already distant fabric. The armpit field fades out above the shoulder underside, preserving
  collar folds and shoulder caps. Armpit and side-seam bindings select nearby real body triangles
  using neighbourhood skin-weight agreement and minimum triangle area over the four validation poses. Rest-space
  differences are folded into the existing world-axis offset record; no new runtime binding format is required.
  A bounded triangle-separation pass evaluates all four poses, averages equal/opposite triangle corrections back
  into rest space, and welds displacement across UV copies. It runs up to 12 iterations with a 12 mm displacement
  cap, then restores regional body clearance. The result is retained only if no pose's intersection-pair count
  increases relative to the input of this pass. Remaining intersections and pre-existing overlapping source folds
  stay in the report; this is a local repair, not a cloth simulation.
- Write `<id>.glb`, `<id>.bind.bin`, optional `<id>.delete.bin`, `<id>.report.json` and an `index.json` entry.
  Binding is **36 bytes per garment render vertex**, little-endian: three uint32 body render indices, three float32
  nonnegative barycentrics, three float32 offsets in meters. Offsets are **world-axis vectors**, matching MHCLO;
  the runtime does not rotate a scalar normal offset. `scaleRefs` are measured on the neutral body. The serialized
  binding/GLB round-trip must be below 1 mm even without `--validate`.
- `--mirror-pair` mirrors an aligned single left shoe across X=0. Shoes also write `<id>.rigid.json` in the app's
  **rigid-sole format v1** (`packages/avatar-core/src/garment/rigid.ts`, `parseGarmentRigidAttachments`), with exactly
  the keys `version: 1`, `soleHeightMm` and `attachments[{bone, vertices, localPositionsM}]`. Sole/heel vertices below
  `--rigid-sole-height-mm` (default 35 mm above each shoe's lowest point) go to `foot_l`/`ball_l`/`foot_r`/`ball_r`,
  split at the ball joint. `vertices` are GLB render-vertex indices (UV copies included, each in a single attachment)
  and `localPositionsM` is the exported float32 neutral position minus the neutral bone head, in metres. The app
  rebuilds the sole as `head + local * scale`, fades it into the upper over a 10 mm geodesic band and keeps the MHCLO
  binding elsewhere. `--validate` also runs the exported sidecar through avatar-core's own parser/preparation and fails
  if the app would reject it or cannot rebuild the exported sole at the neutral scale (`rigidRuntime` in the report).
  Review notes (sole boundary, rigid segments) live in the report's `rigidParts.warnings`, not in the sidecar. Rigid
  override pose metrics (`rigidOverridePoses`) skin each sole segment 100% to its bone and omit the 10 mm blend band.
- Upstream MHCLO `delete_verts` masks are authored lists. Imported masks use the app's normal-prism coverage rule
  instead: body vertices 2..20 mm behind the garment, interior triangle projections, a 6 mm band retained near
  openings, identical decisions for every UV copy. Body triangles are hidden only when all three corners are hidden
  by the app. `--no-delete-mask` disables this optional file. Inspect necklines, cuffs, hems and transparent fabric.
  Texture alpha is not used to infer holes; use `--no-delete-mask` for lace/cutout garments until reviewed.
  Authored lists now seed a union of A/T/relaxed/walk coverage with a 35 mm pose margin. Intersecting body triangles
  contribute all corners within 65 mm of the rest garment. Rest opening bands protect the coverage expansion;
  hand/finger, neck and head influences and the neck-height guard always override deletion, including UV copies.
  The report records coverage per pose, protected vertices and the final exported mask.

Alignment overrides live in `--transform sidecar.json`, or automatically in `<source stem>.import.json`.
World convention is meters, +Y up, +Z front, character left +X. A row-major matrix completely overrides automatic
scaling/placement/ICP; `pose` still controls proxy reposing. For an already aligned exported app garment:

```json
{
  "pose": "a",
  "matrix": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

Other optional keys: `sourceBodyHeightM` (same source-unit meaning as the CLI argument), positive multiplier `scale`,
`rotationDeg: [x,y,z]` (Euler XYZ), `translationM: [x,y,z]` (applied after scaling), `icp: false`, and `landmarks`.
Landmarks map names to **original source-coordinate** points; at least three non-collinear points are needed:
`shoulder_l`, `shoulder_r`, `hip_l`, `hip_r`, `ankle_l`, `ankle_r`. Landmark similarity fitting supersedes the initial
transform. Supply the pose explicitly when landmarks describe a T-pose source. Unknown keys and invalid transforms fail.

Validation (`--validate`; also implied by `--compare-original path/to/tshirt.bind.bin`) requires Node and the repo's
installed npm dependencies. It calls **actual avatar-core** via a read-only Jiti bridge: body solver, binary binding
parser, `bindGarment`, `gradeGarment`, `analyzeFit`, `garmentClearance` and `clearanceToColor`. There is no exported
small/medium/large preset catalogue; the three documented `realData.test.ts` profiles are used:
small 150 cm/45 kg (neutral), medium 165 cm/58 kg (female), large 180 cm/80 kg (male).

The report includes neutral and all three graded bodies in A, T, relaxed and walk, with the percentage of garment
vertices inside the complete posed body, measured by forward/backward ray parity on the welded mesh (near +Z direction
with a slight deterministic tilt to avoid axis-aligned seams). It also records texture sizes, round-trip error,
runtime binding agreement, finite output/heatmap values, fit verdicts, extents and edge stretch. The smoke acceptance
gate is finite geometry, extent below 5 m, 99th-percentile edge stretch below 10x and max edge growth below 100 mm.
Local stretch above 10x produces a warning. These are **numerical stability checks**, not garment-quality approval.
Penetration is diagnostic and may remain nonzero under motion; self-intersecting posed bodies also limit ray parity.
Triangle-centroid penetration is reported separately because exterior vertices can still span a triangle through the
body. Both are samples, not exhaustive triangle intersection tests; the clearance pass constrains vertices only.
The pass is before body hiding or app collision correction and omits app length grading/layering. Persistent arm/torso
collisions would need pose-aware collision handling or cloth deformation in the app; importing alone cannot guarantee
collision-free animation with the current MHCLO/linear-skinning contract. The raw core fit report also compares a short
sleeve with full arm length; the app's kind-specific report UI already treats that as informational.

Each pose also has a `visibility` diagnostic. It applies the exported mask **plus the app's fitted-rest
`coveredBodyVertices` footprint**, retaining a body triangle unless all three corners are hidden. Visible
poke-through counts non-hidden body triangles intersecting the garment, supplemented by exterior body centroids
within 40 mm of an interior garment projection next to a contained cloth vertex/centroid. This local-sample guard
avoids counting nearby bare skin beyond openings. Percentages use **all body triangles** as denominator.
Garment self-intersection uses an R-tree broad phase and triangle separating-axis tests, including coplanar overlap
and contacts within 0.1 micrometers; triangles sharing a welded vertex are excluded as normal adjacency. Its
percentage uses **all garment triangles**, so it is not comparable to vertex containment percentages. Reports
include triangle IDs and rest-anatomy region counts. Self intersections can include original overlapping folds.
`insideDeletedRegionVertices` counts contained cloth vertices whose nearest body triangle is deleted. This is a
local hidden-region diagnostic, not a proof that the vertex has no other contact with visible skin. Posed-body
self-intersection still limits parity. These checks omit app length grading, layer stacking and camera occlusion.

Reproduce masked front/side previews for every pose and a before/after visibility review in the pipeline venv:

```powershell
$env:PYTHONPATH = 'tools/asset-pipeline'
tools\asset-pipeline\.venv\Scripts\python.exe -m garmentimport.review --phase before
# Repeat this import for each reviewed candidate's saved command.json.
$importArgs = Get-Content tools/asset-pipeline/.cache/garment-import-out/mh-jacket-t01/command.json -Raw | ConvertFrom-Json
& tools\asset-pipeline\.venv\Scripts\python.exe tools/asset-pipeline/garment_import.py @importArgs
tools\asset-pipeline\.venv\Scripts\python.exe -m garmentimport.review --phase after
```

The review defaults to jacket-t01, polo, trousers 01/02, sneaker, the existing tshirt and the denim jacket. A candidate's
`before-assets/` snapshot, when present, supplies the baseline instead of its current export. Results are
`before/after-visibility.json`, per-candidate `before/after-preview.png`, and the combined review previews.

`--compare-original` maps cleaned vertices to the original binding's reconstructed vertices, then reports matching
triangle records and base/deformed reconstruction differences. Nearest-triangle bindings need not equal an authored
MHCLO triangle record to reconstruct the same garment. Cleanup/decimation correspondence and clearance fitting are
reported separately. Failed export/validation leaves the old accepted candidate intact and diagnostics in a `.pending`
folder. Accepted files are published into the review folder, with the catalogue written last; concurrent writes to the
same output catalogue are not supported.

Importer tests (synthetic geometry plus optional shipped CC0 MakeHuman integration; no restricted/personal assets):

```powershell
tools\asset-pipeline\.venv\Scripts\python.exe -m pytest tools/asset-pipeline/tests/test_garment_import.py tools/asset-pipeline/tests/test_garment_visibility.py
```

For the synthetic-only import/visibility suite, add
`-k "not real and not synthetic_torso and not align_source and not rigid_shoe" -p no:cacheprovider`.
The excluded tests use the existing permissive body/wardrobe integration fixtures. The visibility suite itself uses
only synthetic meshes, rigs and poses, including collapsing-joint anchors, welded UV copies and protected-hand contacts.

The geometry implementation uses [Trimesh proximity](https://trimesh.org/trimesh.proximity.html) and
[fast-simplification](https://github.com/pyvista/fast-simplification); their pinned versions are in
`garmentimport/requirements.txt`. No MakeHuman application code is imported.

Native example (write each candidate into its own review directory):

```powershell
tools\asset-pipeline\.venv\Scripts\python.exe tools/asset-pipeline/garment_import.py `
  tools/asset-pipeline/.cache/garment-sources/mh-jacket-t01/Jacket_T01.mhclo `
  --id mh-jacket-t01 --category top --kind tshirt --license CC0-1.0 `
  --attribution 'Jacket_T01 by Tobinsky, MakeHuman community, CC0-1.0' `
  --out tools/asset-pipeline/.cache/garment-import-out/mh-jacket-t01 --validate
```

The 2026-10-12 review runs, commands, alignment overrides and per-pose/per-shape reports are in
`.cache/garment-import-out/<source-id>/`; `review-summary.md` in the parent directory summarizes the results.
These are offline candidates. Hoodie hood/flap weights, shoe attachments, texture detail and posed collisions still
require manual review. The MakeHuman Denim Jacket (CC-BY-3.0, punkduck) imports with `--license CC-BY-3.0 --kind
longsleeve --layer 2`; the exact licence version is kept in its catalogue entry and attribution.

## Body parts (Wave 3d)

`build.py` also writes `apps/web/public/assets/parts/` (`index.json` = `{ version: 1, parts: BodyPartDef[], defaults:
{ eyes, eyebrows, eyelashes } }`, per part `<id>.glb`, `<id>.bind.bin`, optional `<id>.delete.bin`) and the debug renders
`.cache/debug/parts_head_front.png`, `parts_head_side.png`, `parts_face_closeup.png`, `parts_eyes_extremes.png` and
`parts_<hair id>.png` (front | side). Sources: single files of the MakeHuman system asset pack and `hair01_cc0`
(`config.PART_ASSETS`, sha256-pinned, range-fetched into `.cache/parts/`, about 25 MB), all CC0. Eyes, eyelashes and the
long / ponytail hair are authored against helper vertices that are not in the runtime index space; they are re-bound to
body triangles over 48 sample bodies. See `docs/adr/0008-body-parts.md`.
