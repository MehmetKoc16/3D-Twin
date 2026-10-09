# Architecture

## Overview

```
 tools/asset-pipeline (Python)
   raw MakeHuman + MPFB2 CC0 data (.cache, gitignored)
        |  python tools/asset-pipeline/build.py
        v
 apps/web/public/assets/body/   (Git LFS)
   base.glb  morphs.bin  manifest.json  rig.json  measures.json
        |  fetch
        v
 packages/avatar-core (pure TS)  <--- runs in a Web Worker via comlink
   macroWeights -> solveBody -> applyMorphs -> computeJoints
        |  Float32Array positions + joints
        v
 apps/web  R3F <Avatar/>  (three.js BufferGeometry update, skeleton rebuild)
        ^
        |  BodyParams from body-panel (zustand store)
```

## Module boundaries

| Module                  | Responsibility                                          | May import          |
| ----------------------- | ------------------------------------------------------- | ------------------- |
| `tools/asset-pipeline`  | Raw data -> web formats; deterministic                  | Python only         |
| `packages/avatar-core`  | Pure math/logic, no React/DOM/three                     | nothing external    |
| `apps/web/src/workers`  | comlink wrapper running avatar-core off the main thread | avatar-core         |
| `apps/web/src/features` | UI + R3F; one folder per feature                        | avatar-core, shared |
| `apps/web/src/store`    | zustand slices                                          | avatar-core types   |

## Conventions

Meters, right-handed, +Y up, character faces +Z, feet on y=0. Rest pose = MakeHuman rest pose; "T-pose" is a
`PoseDef` preset. Quaternions are `[x, y, z, w]`. UI strings only through i18n keys (tr + en).

## Data contracts

All types live in `packages/avatar-core/src/contracts.ts` (source of truth). The files below are produced by
`tools/asset-pipeline` (`npm run assets:build`, see `tools/asset-pipeline/README.md`); this section describes contract
**v1.1** (the JSON `version` field stays `1`).

- `BodyManifest` (`manifest.json`): `vertexCount`, `renderVertexCount`, `jointPoints`, file names, `macroVariables`
  (with tent-interpolation `buckets`), `targets` (offset/count into `morphs.bin`, optional `macroConditions`),
  `modifiers` (slider value -> decr/incr target), license info. The pipeline also writes an informational
  `provenance` object (pinned source SHAs, `groundOffsetY`, rest pose note) that the runtime may ignore.
- Morph model: `v = base + sum_i w_i * delta_i`. Macro targets get weight = product of tent weights of their
  `macroConditions`. A modifier with value < 0 uses `|value| * decrTarget`, > 0 uses `value * incrTarget`.
- `RigDef` (`rig.json`): bones with `head`/`tail` `JointRef` (`VERTEX` of a joint point, `MEAN` of vertices, or `FIXED`
  position) so joints are recomputed from the morphed mesh.
- `MeasuresDef` (`measures.json`): per `MeasureId` a `circumference` / `distance` / `polyline` / `height` /
  `vertexHeight` definition (vertex indices) and the `drivers` modifiers that control it.
- `PoseDef` (`poses/*.json`): bone-local quaternions relative to rest pose, labels tr/en.
- `BodyParams`: user input (gender 0..1, cm values, shoe size). `GarmentDef` describes the shipped wardrobe templates (binding, grading, ease and size data).

### Asset data formats (v1.1)

**Vertex index space.** Every index in `morphs.bin`, `rig.json` and `measures.json` addresses one combined space:
`[render vertices of base.glb] ++ [jointPoints]`, so `vertexCount = renderVertexCount + jointPoints.length`
(currently 14517 + 69 = 14586). Render vertices are the glTF vertices of `base.glb` after the UV-seam split (body only,
helper geometry removed), ordered by (MakeHuman vertex, UV). A MakeHuman vertex delta is duplicated onto all of its
split copies, so seams never open. A joint point is the centroid of one MakeHuman joint helper cube (8 vertices);
its delta is the mean of the 8 cube-vertex deltas (exact, because morphs are linear). Joint points are not part of
the glTF mesh. See `docs/adr/0005-vertex-index-space-and-joint-points.md`.

**Units and ground.** Meters, +Y up, +Z front, character's left = +X. MakeHuman data is in decimeters (x0.1). All
positions are translated by `provenance.groundOffsetY` (0.817763 m) so that the neutral body (macro defaults:
gender 0.5, age 25, muscle/weight/height 0.5, race mix 1/3) has its lowest render vertex at y = 0. `base.glb` holds
the raw base mesh (not the neutral body): the raw mesh is 1.666 m tall, the neutral body 1.659 m (female 1.59 m, male
1.73 m at the other defaults).

**Macro variables.** `gender` (female@0, male@1), `muscle` / `weight` / `height` (min@0, average@0.5, max@1),
`cupsize` and `firmness` (min@0, average@0.5, max@1), all default 0.5. `age` is pinned to 25 years for the MVP:
`min = max = default = 0.5` with the single bucket `young`; baby/child/old targets are not shipped. Race is
pre-merged into a neutral 1/3 mix per gender (no `race` variable). Tent rule: a value between two buckets splits its
weight linearly; outside the range the nearest bucket gets 1; a single bucket always gets 1. This is exactly
MakeHuman's `min = max(0, 1-2t)`, `max = max(0, 2t-1)`, `average = 1-min-max`. Proportions are not shipped.
Targets that are empty upstream (average/average) are omitted.

**Modifiers.** 96 modifiers, ids follow MakeHuman folders (`measure/measure-waist-circ`, `torso/torso-scale-vert`,
`armslegs/upperarm-fat`, `head/head-scale-horiz`, ...). Symmetric left/right targets are pre-merged into one
target (`armslegs/foot-scale-depth-incr` etc.). Two-sided modifiers have `min = -1`, `max = 1` (the measure modifiers may span up to +-1.5, see ADR 0006), one-sided ones
(`head/head-oval` ...) have `min = 0` and only `incrTarget`. All defaults are 0.

**Rig and rest pose.** MPFB2 `game_engine` preset (53 bones: `Root`, `pelvis`, `spine_01..03`, `neck_01`, `head`,
`clavicle_*`, `upperarm_*`, `lowerarm_*`, `hand_*`, 15 finger bones per hand, `thigh_*`, `calf_*`, `foot_*`,
`ball_*`), listed parents-first in `rig.json` and in the glTF skin. Rest pose = MakeHuman relaxed **A-pose** (upper
arm ~50 deg below horizontal, forearm ~41 deg); "T-pose" is a `PoseDef` preset. Bones are world-aligned: identity
rest rotation, `roll: 0`, glTF nodes carry translation only and the inverse bind matrices are pure translations of
`-head`. Joints are `{strategy:'VERTEX', vert: renderVertexCount + j}` for every head and tail (no `MEAN`/`FIXED` is needed;
MPFB2's tiny `Root` tail offset is replaced by the pelvis joint so that the `Root` bone spans ground to pelvis and
follows the leg-height morphs, which move the ground joint). Skin weights are the preset's weights reduced to the top 4
influences per vertex and renormalised (`JOINTS_0` uint16, `WEIGHTS_0` float32).

**Measures.** Semantics (numpy reference: `tools/asset-pipeline/measures.py`):

| type            | definition                                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `circumference` | perimeter of the 2D convex hull of the loop vertices projected on the Newell plane of the ordered loop     |
| `distance`      | Euclidean distance of 2 vertices, or `abs` difference along `axis` (`footLength`: z)                       |
| `polyline`      | sum of consecutive vertex distances (`armLength`: acromion, elbow, wrist; `shoulder`: back surface via C7) |
| `height`        | bbox Y extent over render vertices only                                                                    |
| `vertexHeight`  | `y(vert)` minus the lowest render vertex (`inseam`: crotch)                                                |

Circumference loops are ordered: consecutive vertices walk once around the ring, counter-clockwise seen from above (Newell
normal points +Y), the first vertex is not repeated at the end. They are derived from geometry on the
neutral body (plane slices between joint points and skin-weight body regions), never copied from MakeHuman's
measurement plugin. `drivers` are modifier ids that are meant to move together with one shared slider value
(`armLength`: upper + lower arm length; `inseam` / `height`: upper + lower leg height). The macro `height`
variable is not a modifier and is not listed in `drivers`.

**Render mesh.** 26756 triangles over the 14517 render vertices (quads split along the shorter diagonal, CCW winding,
outward normals). The mesh is closed: every edge is shared by two triangles once UV-seam copies are welded by their
MakeHuman vertex (no eye-socket or mouth holes; hands and feet are closed too), so signed-volume mass estimation works
directly on the render vertices. The raw base mesh encloses 54.9 L. There is one material and no texture; `TEXCOORD_0`
holds the MakeHuman UV layout for later skin textures.

### morphs.bin binary format

Concatenated sparse target entries, little-endian, 16 bytes each:

| bytes | type    | meaning                                  |
| ----- | ------- | ---------------------------------------- |
| 0-3   | uint32  | vertex index (combined space, see above) |
| 4-7   | float32 | dx (meters)                              |
| 8-11  | float32 | dy (meters)                              |
| 12-15 | float32 | dz (meters)                              |

A `TargetDef` addresses its run via `byteOffset` (bytes) and `count` (entries). Entries inside a run are sorted by
strictly ascending vertex index, deltas with all components below 1e-5 m are dropped. Current payload: 311 targets,
986,852 entries, 15.8 MB raw (`base.glb` 0.98 MB, total 16.9 MB). If the size budget tightens, int16 quantisation per
target (scale in the manifest) would halve `morphs.bin` (not implemented).

## avatar-core public API

Exported from `packages/avatar-core/src/index.ts` (pure TypeScript; types live in `contracts.ts` and `types.ts`).

```ts
// morph model
validateManifest(m: BodyManifest): void;
buildBasePositions(gltfPositions: Float32Array, manifest: BodyManifest): Float32Array; // render ++ joint points
parseMorphs(buffer: ArrayBufferLike, manifest: BodyManifest): MorphSet;
getMorphSet(morphs: ArrayBufferLike | MorphSet, manifest: BodyManifest): MorphSet;    // cached
macroWeights(manifest: BodyManifest, vars: Partial<Record<MacroVar, number>>): Map<string, number>;
modifierWeights / mergeWeights / combineWeights;                                      // modifier values -> target weights
applyMorphs(base: Float32Array, morphs: ArrayBufferLike | MorphSet, manifest: BodyManifest,
            weights: ReadonlyMap<string, number>, out: Float32Array): void;
// measures and mass
measure(def: MeasureDef, positions: ArrayLike<number>, renderVertexCount?: number): number; // meters
estimateMassKg(positions, indices, densityKgPerL?, vertexLimit?): number;
meshVolumeM3(positions, indices, vertexLimit?): number;
// solver
interface SolverData { manifest: BodyManifest; base: Float32Array; morphs: ArrayBufferLike | MorphSet;
                       measures: MeasuresDef; indices?: ArrayLike<number> }
solveBody(data: SolverData, params: BodyParams, opts?: SolveOptions): SolveResult;
createBodySolver(data: SolverData): BodySolver;   // prepared + cached; solver.solve(params, opts) is cheap to repeat
bmiToWeightValue(bmi: number, knots?): number;
interface SolveResult { weights; modifierValues; macroVars; positions /* base + morphs, vertexCount * 3 */;
                        achievedCm; residualsCm; unreachable: MeasureId[]; estimatedMassKg; iterations }
// skeleton
computeJoints(rig: RigDef, positions: Float32Array): Map<string, { head: Vec3; tail: Vec3 }>;
validateRig(rig: RigDef, vertexCount?: number): void;  boneOrder(rig: RigDef): string[];
// shoes (approximate table, linear interpolation)
footLengthCmFromShoe(shoe: { system: ShoeSystem; size: number }): number;
shoeFromFootLengthCm(cm: number, system: ShoeSystem): number;
convertShoeSize(shoe: { system: ShoeSystem; size: number }, to: ShoeSystem): number;
roundShoeSize(size: number): number;                    // nearest half size
```

`solveBody`: (1) height by a 1-D root find on the macro `height` variable, (2) weight: macro `weight` seeded from BMI and
refined so the mesh mass matches `weightKg`, (3) provided local measures by bounded Levenberg-Marquardt on their driver
modifiers, nested inside a height root find so the total height stays exact although the leg-height modifiers move it.
Targets that cannot be met (modifier range exhausted, e.g. a waist far from what the weight implies) are reported in
`unreachable` with the closest achievable value in `achievedCm`; they are never thrown. Mass = mesh volume x 1.01 kg/L;
measurements take priority over weight on conflict. A solve takes about 5-15 ms on the real body.

## Runtime flow

```text
main thread                                   avatar worker (workers/avatar.worker.ts, comlink)
-----------                                   --------------------------------------------------
Avatar mounts
  GLTFLoader: base.glb (skinned mesh, raw)
  weld map (UV-seam duplicates) built once
  client.init(raw positions, indices) ------> fetch manifest / measures / rig / morphs.bin (progress -> overlay)
                                               createBodySolver({ manifest, base, morphs, measures, indices })
bodyStore.params changes (slider drag)
  client.solve(params)  [latest wins] ------> solver.solve(params)         (~10 ms)
                                               re-ground: subtract min y over render vertices
                                               computeJoints(rig) -> 53 x (head, tail), grounded
  <------ transferable Float32Arrays: positions, joints; achievedCm, residualsCm, unreachable, mass
applySolveResult:
  write positions into the BufferGeometry, welded normals, bounds
  rebuild rest skeleton: save bone quaternions -> identity -> local = head - parentHead (root = head)
    -> skeleton.calculateInverses() + bindMatrix -> restore quaternions
  avatarRuntimeStore.setSkeleton / bumpRestVersion();  solveStore <- achieved values (body panel)
PoseDriver (features/poses) re-applies the pose on restVersion; focus targets read the posed bones.
```

### Face pipeline (selfie -> texture + head shape)

```text
main thread (all in the browser, nothing is uploaded)               avatar worker
-----------------------------------------------------               -------------
FacePanel: file / webcam frame (raw, un-mirrored; only the preview is mirrored)
faceStore.loadPhoto: decode -> MediaPipe FaceLandmarker (478 landmarks, lazy wasm) -> quality hints
                     photo + landmarks persisted in IndexedDB
faceStore.bake: face-map.json (binding of the 468 canonical landmarks to body UVs, ADR 0006)
  warp: per canonical triangle affine (photo px -> body UV), offscreen WebGL, 2048^2
        (slivers with UV area < 1e-6 or minority orientation are skipped: eye / lip holes)
  mask: feathered face oval, forehead fade along the (rotated) up axis, eyes + mouth slit cut out
  delight + border colour match  ->  overlayCanvas (straight alpha), skinToneHex (cheek median)
  revision++
Avatar (subscribes to faceStore + appearanceStore)
  SkinMap: 2048^2 canvas = skin tone (photo tone or preset) + overlay -> CanvasTexture
           (flipY false, sRGB) as material.map; mannequin mode: no map
  overlay set / removed -> client.setFace(landmarks, w, h) / clearFace() ---> FaceShapeState.set / clear
                           then re-solve                                      solve: fitFaceModifiers on the solved
  <---------------------- faceFit { modifierValues, rmsResidual } ------------ weights (once per photo), then
  faceStore.fit -> "face shape fitted" status in FacePanel                     mergeWeights(body, modifierWeights(face))
                                                                               is applied in EVERY later solve
```

The face-shape fit is done once per photo against the body solved at that moment; the fitted values then stay
constant while the body sliders move. `faceStore.ensureBaked()` runs at startup and restores a stored selfie.

Requests are coalesced on the main thread (`LatestWinsRunner`): at most one solve is in flight and one waits, so
dragging a slider never queues work and the UI thread only does the geometry upload (a few ms). The leg-height morphs push
the feet below y = 0, so grounding is always applied after solving; joints are shifted by the same offset.

### Wardrobe (try-on, Wave 4c)

```text
public/assets/garments/index.json  (GarmentTemplateDef[])  + <id>.glb  <id>.bind.bin  <id>.delete.bin
        |  wardrobeStore.loadCatalog (catalogue)      IndexedDB `wardrobe:item:<id>`, `wardrobe:worn` (StoreItemDef, worn map)
        v
WardrobePanel: catalogue (licence badge, attribution) -> StoreItemForm (chartModel: sizes x measures, flat x2,
               EU shoe sizes -> inner length, colour from a local photo by k-means) -> saved items -> Wear
        |  wardrobeStore.worn (one item per category: top / bottom / shoes)
        v
WardrobeRig (created by Avatar; Avatar forwards every solve after the body geometry + skeleton were updated)
  per worn item: GarmentInstance = SkinnedMesh on the avatar's Skeleton
    once per template: glb index + UV, parseGarmentBinding, garmentSkinWeights from the body's skin attributes
    after EVERY solve (solved, grounded render positions):
      bindGarment -> buildGradeRings (chart[size] - (body + template defaultEase) at the chest / waist / hip / thigh
      planes taken from measures.json loops) -> gradeGarment (leg-aware below the crotch) -> pushOutside (layering)
      -> welded normals -> [heatmap: garmentClearance -> clearanceToColor vertex colours]
    fit order: shoes, bottom, top - but a TUCKED top is fitted before the bottom (see "Tucked and untucked tops")
  body: delete lists + body vertices under each garment's footprint are hidden (index rebuilt when the worn set changes)
  shoes: the avatar (scene) is lifted so the lowest sole vertex stands at y = 0
Fit report (React, no worker): analyzeItemFit = analyzeFit on solveStore.achievedCm; chips per region, overall,
recommended size; recomputed whenever the body or the selected size changes.
```

Details that are not obvious:

- Grading target. The bound garment already follows the body with the template's own ease, so the ring delta is
  `chart - (body + defaultEase)`, not `chart - nativeMeasures` (identical only on the neutral body).
- Leg-aware grade / clearance. avatar-core pools all vertices of a 1 cm height bin, which puts the centroid between
  the legs; below the crotch the pass is run once per leg on a body copy that keeps only that leg in the bins.
- Poke-through. MakeHuman delete lists leave islands (navel); body vertices that project into a garment triangle
  (within 2 cm, prism test) are hidden too, a triangle is dropped only if all three vertices are hidden, so no hole
  opens at a hem. Layering (`garmentCollide.ts`): `pushOutside` moves the vertices of an outer garment out of the inner
  one (up to 5 passes, because a push along one triangle's normal can end inside a neighbouring triangle of a curved
  surface); `pushInside` is its mirror. The inner garment's triangles under the outer one's footprint are hidden.
  Trousers are pushed out of shoes.
- Tucked and untucked tops. `garmentStyle.ts` keeps a per-template style (the catalogue does not carry it):
  `tshirt` (`toigo_basic_tucked_t-shirt`, authored with its hem at the waist, inside the trousers) is `tucked`, every
  other top is `untucked`. Untucked top (sweater): unchanged rule - a top of a higher `layer` than the worn bottom is
  the outer layer (its vertices are pushed out of the trousers, the trousers' triangles under it are hidden), its hem
  is length-graded from the chart. Tucked top worn WITH a bottom: the bottom's waistband is the outer layer, whatever
  the layer numbers say. The top is fitted first, its chart-length pass is skipped (the chart's garment length
  describes a free-hanging hem, the tucked template ends at the waist; sleeve passes still run), then the bottom is
  fitted and pushed out of the top, then the top's hem is tucked (`pushInside`) into the bottom and the top triangles
  behind the bottom are hidden. A tucked top worn ALONE behaves like any top: the hem is length-graded from the chart.
- Length grading (`gradeGarmentLength`). The hem shift is a linear ramp from the anchor plane to the hem plane, so the
  ramp must be long: a top is anchored at the CHEST plane (waist only if no chest plane exists), not at the waist.
  The first version anchored at the waist, 3-8 cm above the hem, and squeezed a 15 cm change into that span: rows near
  the hem were stretched ~5x, giving a ragged, jagged hem (max edge stretch is now ~1.8x, tested). The hem band itself
  moves rigidly (everything below the 10th-percentile height gets the full shift), so its shape is preserved.
- Colour. Templates carry a neutral grey texture whose mean is `baseColor`; the material colour is
  `item / baseColor` per linear channel, so the default shows the texture unchanged and a picked colour keeps the
  fabric detail.
- Fit verdict wording. Girths use tight / snug / comfortable / loose / oversized; sleeve, length and inseam use
  short ... very long. Sleeve and inseam are compared with the template's intended length on the body (body +
  `defaultEase`), and the sleeve of a short-sleeve template is shown as information only.
- Privacy. Store URLs are stored as text and never fetched; a product photo is decoded on a canvas for its colour only.

### Appearance / body parts (Wave 3e)

```text
public/assets/parts/index.json (BodyPartDef[], defaults)  + <id>.glb  <id>.bind.bin  [<id>.delete.bin]     ADR 0008
        |  loadPartsIndex (validatePartsIndex)             IndexedDB `appearance:*` (hair, brows, eye colour, skin)
        v
PartsRig (created by Avatar after the WardrobeRig, disposed before it; Avatar forwards every solve)
  per mounted part: PartInstance = SkinnedMesh on the avatar's Skeleton (same binding format as garments)
    once per part: glb index + UV, parseGarmentBinding, garmentSkinWeights from the body's skin attributes
    after EVERY solve (also the ones that carry the fitted face-shape modifiers): bindGarment(+computeScale from the
      scale refs) on the solved, grounded render positions -> welded normals -> upload
  defaults: eyes, eyebrows-default, eyelashes-default; hair only when chosen (bald = no mesh)
  materials: hair / brows / lashes are MASK -> opaque pass, alphaToCoverage (Canvas has MSAA), no sorting;
    eyes OPAQUE + single-sided; brows / lashes get a small polygon offset over the skin
  body triangles: eye-socket cavity (+ tousled-hair scalp) delete lists are hidden through a `setIndex` hook on the
    body geometry: whatever index the wardrobe assigns is filtered by the parts' mask, so both compose
Appearance store: hairId / hairColor / eyebrowId / eyebrowColor (null = follow hair) / eyeColor
```

- Tint. Hair and brow textures are neutral grey + alpha; the material colour is `picked / meanLinear(texture)` (mean of
  the covered texels, computed once per part), so a picked colour reads true on average whatever the texture's grey level.
- Iris. `irisRecolor.ts` copies the eye texture to a canvas once and rewrites the pixels inside `irisUv` as
  `colour * luminance / referenceLuminance` (reference = mean of the iris ring, so the average is the chosen colour), a
  smooth fade at the rim (0.97 - 1.12 x radius); the grey iris keeps its fibre detail, the pupil stays dark and the
  sclera is untouched. `CanvasTexture`, `flipY = false`, sRGB.
- From the photo. `features/face/photoColors.ts` reads the selfie (up to 2048 px, in the browser) at the iris landmarks
  468 - 477 of both eyes: median colour of the iris ring without pupil, upper-lid shadow and the darkest / brightest
  15 % luminance tails; the brow colour is the darkest 15 % of small windows around the brow contour landmarks. Offered
  as "eye colour from photo" and "hair and brow colour from photo" buttons; nothing is applied automatically.
- Persistence. `appearance:<field>` keys in IndexedDB (try / catch); values the user changed before the stored ones
  arrived win.

### Realistic twin (scan and template-character paths)

The standard model is the parametric MakeHuman mannequin. The Model switch selects `standard` or `twin` in
`twinStore.ts`; the twin panel accepts a self-contained `twin.glb` or the legacy `rigged.glb` + `twin.json` + optional
`mh2twin.bin`. Both reconstruction paths use the same app loader, poses, cameras and wardrobe.

#### Offline pipeline

`tools/twin-lab/run_all.py` is a standard-library launcher. Its selectable stage order is:

```text
shape -> texture -> refine -> head -> bodyfix -> hybrid -> rig -> glasses -> bundle
```

- `--body scan` (default): retains the generated mesh and projected texture. Refine repairs face relief and separates
  fused armpits; an optional head stage precedes bodyfix; bodyfix corrects the scan against supplied measurements.
  Rig fits/unposes it to the 53-bone MakeHuman A-pose, transfers weights and writes `rig/{rigged.glb,twin.json,mh2twin.bin}`.
  The launcher uses `--fingers merge --cut-bridges`; hybrid and automatic glasses are omitted. Bundle is `out/twin.glb`.
- `--body hybrid`: builds a template character from the MakeHuman body, its fixed topology/UVs and native hands/feet.
  Earlier scan stages supply bodyfix's shape prior. Hybrid re-solves the body against measurement targets without scan
  clothing allowances and deforms the template's own head toward the neutral FLAME fit, anchoring the neck loop.
  It bakes the face into the fixed head UVs, adds CC0 eyes/lashes, a CC0 body skin with generated normal detail and painted
  underwear, and hair. The body mesh contains no transplanted scan head geometry. Rig verifies the native A-pose against
  `dtHybrid` and uses `--fingers keep --smooth 0`. Outputs are `out/hybrid/hybrid.glb`, `out/hybrid/rig/` and
  `out/hybrid/twin.glb`; the portable `face_asset/` export is **not yet consumed by the standard-model app**.
- Refine and bodyfix are included when their scripts exist. Head is selected by `--head {auto,flame,recon,none}`:
  `auto` selects FLAME only when `<input-dir>/head/flame/fit/head_neutral.obj` exists, otherwise skips head.
  `--with-head` is a deprecated alias for recon. Hybrid requires FLAME and bodyfix; `auto` without that fit fails.
- In hybrid mode, an existing `<input-dir>/hy3d/hy3d.glb` (or `--glasses-hy3d <bust>`) enables the post-rig `glasses`
  stage unless `--no-glasses` is set. It fits the accessory, cleans the face atlas and supplies `glasses_rigged.glb`
  plus `glasses.glb` to bundle. `--no-deglass` preserves photographed frames while still permitting the accessory;
  `--no-glasses` also disables hybrid deglass. With no bust, the launcher passes `--no-deglass` and omits this stage.

Each stage runs in its own venv, except head/hybrid/automatic glasses share `refine/.venv`, bodyfix shares `rig/.venv`,
and bundle falls back to `rig/.venv`. Inclusive `--from-stage` / `--to-stage` ranges require earlier inputs to exist.
Timestamp and command/input state checks skip fresh stages; `--force` rebuilds the selected range and propagates
rebuilds downstream. Exact commands and stage inputs/outputs are in [twin-lab's README](../tools/twin-lab/README.md).

#### FLAME fit and photo cameras

The user-operated [Pixel3DMM Colab notebook](../tools/twin-lab/colab/pixel3dmm_README.md) fits one shared identity with
per-view expression, pose and cameras. It accepts up to twelve views: front/left/right/back and `extra_1` through
`extra_8`. Its private export supplies the neutral head, `parameters.json`/NPZ, `cameras.json`
(`dt-flame-head-cameras/1`), `fitted_views/*.ply` and provenance. Extract the fit under
`user-data/twin/head/flame/fit/`; masks and the MediaPipe landmark embedding are obtained separately from FLAME and
kept under `user-data/flame/`. The local stages evaluate exported surfaces rather than loading FLAME model weights.

For scans, `head/flame/flame_head.py` replaces the facial surface and ears, smooths/stitches the transition into the
retained scan and writes `head/head.glb` with `dtFlameHead` version 2. `--head recon` instead selects the experimental
four-photo silhouette/landmark deformation in `head/recon/head.py`.

For hybrids, the existing neutral FLAME identity is registered onto the template head. `hybrid.py --photos-set auto`
selects the complete glasses-free `nog_front.jpeg`, `nog_left.jpeg`, `nog_right.jpeg` set when the front file exists
in the head directory; otherwise it uses the legacy `colab_upload/{front,right}.jpg` with exported cameras and mirrored
side fill. `--photos-set noglasses` requires all three files; `glasses` explicitly retains the legacy path.
`hybridbody/photofit.py` fits new perspective cameras locally on the existing neutral identity using MediaPipe and
FLAME's 105-landmark embedding, robust PnP and focal/pose refinement. It honours EXIF orientation and records camera
fits in private reports. The three-view bake uses independently observed sides, visibility masks and overlap colour
matching, and automatically skips deglass. Camera reprojection residuals are not independent 3D shape measurements.
Photo-set, hair and neck-hair selection flags currently belong to `hybrid.py`, not the launcher.

#### Twin bundle contract

The runtime contract is in `twinBundle.ts`, `twinDef.ts`, `twinHair.ts` and `twinAccessories.ts`. Pipeline metadata is
preserved alongside it; metadata presence alone does not change app behaviour.

| Location                           | Meaning and consumer                                                                                                                                                                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `asset.extras.dtTwin`              | Version 1: full `twin.json` as `twin`, `skinToneHex`, `provenance {shape, license, createdAt}`, and `mh2twin {bufferView, count, componentType: "uint32"}`; optional `accessories`. Parsed by the app.                       |
| `asset.extras.dtHasMakeHumanHands` | Optional boolean, absent means false. True keeps the twin's native MakeHuman hands and suppresses replacement/cutting in the app. Hybrid writes true.                                                                        |
| `asset.extras.dtScanHandsRemoved`  | Optional boolean recording upstream scan-hand removal. Bodyfix sets it; rig uses it to repair wrist/hand influences. Hybrid writes false. App validates its type but does not use it to select the hand path.                |
| `asset.extras.dtBodyfix`           | Version 1 full-precision solved macros/net modifiers, targets, achieved/raw measurements, residuals, allowances and measurement basis. Rig reuses the solution; its export retains bodyfix data in `twin.json.bodyfix`.      |
| `asset.extras.dtHybrid`            | Version 1, `frame: "MakeHuman-grounded-A-pose"`, `bodyManifestSha256`, `cutHeightM`, plus method/parts/provenance diagnostics. Rig validates the manifest and anchored body positions, then reuses the native identity pose. |
| `asset.extras.dtFlameHead`         | Version 2 scan-head provenance, input hash, alignment, smooth stitch and ears option. Passed through bodyfix/rig/bundle; not an app head-fitting command.                                                                    |
| `asset.extras.dtHairNode`          | Nonempty name of a separate skinned hair node. Its material must contain `extras.dtHair` and its own base-colour texture. Parsed by the app; mapping describes only the body.                                                |

The mapping is an aligned BIN buffer view, exactly `count * 4` bytes of little-endian uint32 data, with no GPU target
or accessor. Each entry maps a twin **body** vertex to a MakeHuman render vertex. The loader uses
`gltf.parser.getDependency('bufferView', index)`; offsets are relative to the BIN chunk. It validates version, definition,
skin tone, provenance, mapping bounds and length, hand flags, hair and nested accessories before accepting a pack.
Bundles embed all buffers/images and are rejected if they contain resource URIs. IndexedDB stores the original GLB
and reparses embedded data on hydration; legacy sidecars remain supported.

Hair uses one separate indexed skinned mesh with `POSITION`, `TEXCOORD_0`, `JOINTS_0`, `WEIGHTS_0`, normally `NORMAL`, and
one material; the app computes normals if missing. Hair and body share a rest pose. The app accepts
subset/reordered hair joints only when names and inverse bind matrices match the body. `extras.dtHair` requires
`format` and sRGB `colorHex`; optional `rootHex`/`tipHex` are hex colours and `cardCount` is a non-negative integer.

| `dtHair.format`     | Texture and render path                                                                                                                                                                                                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rcov-groot-bvar/1` | Linear strand data: R coverage, G root-to-tip, B variation, A `min(1, 2.5 R)` for ordinary glTF fallback. App treats the atlas as `NoColorSpace` and runs the vendored MIT `threejs-hair-shader`. MSAA uses alpha-to-coverage plus blended fringe; otherwise alpha-tested core plus blended outer pass. Only the main pass casts shadows, cut out on atlas R. |
| `shell/1`           | Solid textured hair shell from a fitted bust: sRGB colour/alpha texture and optional linear normal map. App creates a matte physical material retaining glTF maps, MASK cutoff and double-sidedness, with no strand shader. BLEND is converted to MASK at 0.5; cutouts get a matching depth material.                                                         |

Hybrid defaults to `hy3d` shell hair when its default bust exists, otherwise procedural cards; legacy MakeHuman
`--hair hair-short`, `hair-tousled`, etc. merge the hair part into the body atlas. See
[hybrid's formats and generation details](../tools/twin-lab/hybrid/README.md).

`dtTwin.accessories` optionally contains `{id: "glasses", bone: "head", mesh: {bufferView}, params}`. That aligned
buffer view embeds a complete self-contained GLB in head-local metres, rigidly weighted to a standalone identity
head bone. `twinAccessories.ts` validates resource embedding, geometry, transforms and rigid weights, preserves frame
and optional lens materials, and mounts ordinary meshes under the shared head bone after alignment. The glasses
checkbox defaults on when present and persists in localStorage (`dt:twin:accessories`). Legacy bundles without the
accessory show no checkbox. Procedural glasses can also be built directly with `head/glasses/make_glasses.py` and
included with the bundler's `--glasses` flag.

#### App flow and measurements

```text
twinStore: local file pick -> validate -> pack -> IndexedDB (`twin:*`)
  standard: PartsRig mounted; worker solves editable BodyParams
  twin:     TwinMode disposes PartsRig, loads TwinModel, sets worker fixed shape and re-solves
            fitted MakeHuman body stays hidden but supplies skeleton and wardrobe geometry
            TwinRig body + optional TwinHair share that skeleton and bind matrix
            native MakeHuman hands retained OR TwinHands supplies replacement hands
            glasses attach to the shared head bone
```

- `setFixedShape({macros, modifiers})` makes the worker apply the fixed morph weights and ignore body params/face shape;
  `null` returns to normal solving. Unknown modifiers fail before switching. Sliders become read-only twin measurements.
- The rig and app use the same 53 world-aligned bones with translation-only inverse bind matrices. `twinBinding.ts`
  remaps by bone name and translates twin rest positions by the mean head-position offset; residuals above 5 mm reject
  mismatched data. PoseDriver, camera focus, sole lift, hair and wardrobe therefore follow one skeleton.
- For legacy scan fits, canonicalisation rebuilds the body from net modifier values rounded to six decimals. Bodyfix
  embeds the full-precision solved body so rig does not refit a different shape. Legacy measurements use the fitted
  proxy minus documented clothing allowances; corrected scans export bodyfix's achieved scan-landmark measurements,
  which can differ from the hidden proxy. Hybrids export the native re-solved measurement result without scan allowances.
  Fit reports and size recommendations use exported measurements; garment geometry still fits the hidden body.
- Without `dtHasMakeHumanHands: true`, `twinHands.ts` removes scan triangles touching the dominant hand/finger-weight
  region, extends the cut 8 mm past the wrist and supplies fitted MakeHuman hands with 25 mm wrist overlap and 1 mm
  normal expansion. Native hybrid hands retain their baked texture and weights.
- Twin wardrobe coverage combines body-to-twin mapping and a 3 cm garment footprint pass, hiding only triangles whose
  three vertices are covered. `twinPushIn.ts` moves covered body vertices inward by at most 8 mm with a 25 mm margin.
  `TwinOpeningRepair` reversibly blends clothing-coloured atlas texels near garment opening loops toward the skin tone
  (full strength within 15 mm, fading to zero at 40 mm). Coverage and repair cache garment/body/alignment changes;
  repeated solves and poses reuse results, and removal restores original geometry/albedo. Separate hair is excluded.
- Standard PartsRig, face bake and appearance controls are disabled in twin mode. The twin body uses physical skin
  shading from `createSkinMaterial`, preserving its albedo, normal map/scale/type, alpha cutoffs and double-sidedness.
  Normal detail survives quality changes, garment hiding and albedo repair. This is lighting/texture rendering, not a
  new face fit in the browser.

#### Studio viewer and poses

`StudioLighting.tsx` captures a procedural 128px cube environment from four Lightformer softboxes, prefiltered by
three.js for image-based lighting. Warm key, cool fill/rim and hemisphere light supplement it. There is no downloaded
HDRI. The canvas uses AgX at exposure 1, sRGB output and MSAA; colour maps are sRGB, normals and strand data are linear.
Drei SoftShadows provides PCSS shadows from the key onto the avatar and circular platform. High uses 12 samples,
2048px shadow maps and DPR 1?1.5; Performance uses six samples, 1024px and DPR 1. Both retain the environment and material
shading. `viewerStore` saves the setting in `dt:viewer:quality`; desktop defaults High and coarse pointers Performance.
Skin uses a subtle shadow-aware diffuse wrap and sheen; shell hair has matte sheen without the skin hook, strand hair
keeps its MIT shader, glasses retain metal/lens semantics, and garments retain maps with bounded roughness.

Run `node apps/web/scripts/generate-poses.mjs` from the repo root to regenerate `public/assets/poses/*.json` from
`poseSpecs.mjs` and the real rig/joint points. Intent-level world directions become parent-local `[x,y,z,w]`
quaternions; identities are omitted, values use six decimals and `w >= 0` canonicalisation. A-pose has no rotations.
`PoseDriver` validates/fetches poses, blends over 350 ms and reapplies them after rest skeleton rebuilds.

The palm convention fixed in **fa2e7a6** is outward normal = `cross(fingers, pinky -> index)` on the left, negated on the
right. Targets are specified in left-hand coordinates and mirrored once in X for the right. The visible hand axis
is wrist-to-middle-knuckle, rather than the oblique bone tail. `balanceForearmRoll` shares the hand/elbow roll difference
with `lowerarm` while retaining the hand's world orientation, reducing wrist collapse under linear blend skinning
without twist bones. Tests bound hand/lowerarm roll and skin the actual permissive mannequin; `wrist_qa.mjs` checks
cross-section area on any compatible rigged GLB.

#### Privacy, licensing and verification

AGENTS.md remains authoritative: personal inputs and every derived output stay in gitignored `user-data/`; committed
assets must be CC0, CC-BY, MIT, Apache-2.0 or BSD and CC-BY assets require credits. Research/non-commercial models and
weights may be used locally or in an explicitly user-operated Colab workflow in ignored caches; they and restricted
derived data are never committed. FLAME/Pixel3DMM hybrid bundles remain private restricted outputs even though their
MakeHuman template is CC0. A bust-derived shell/accessory adds its source restrictions; the bundle's short provenance
label is not exhaustive licence clearance. Hunyuan sources carry their community licence, including territorial terms.

The web face feature stays in the browser. Twin loading performs no upload or remote reconstruction; picked data stays
in memory/IndexedDB and Remove twin deletes its stored pack. Optional notebooks involve a separate user-controlled
upload to a session VM, documented with cleanup in their own READMEs. Docs/tests use synthetic or permissive stand-ins;
never put personal reports, previews or screenshots in tracked paths.

Unit suites cover fixed shape, definitions, bundle/hair/accessories validation, native/replacement hands, shared binding,
coverage, push-in, opening repair and persistence. Real-asset tests compare the CC0 stand-in against avatar-core;
`e2e/twin.spec.ts` exercises load/switch/poses/wardrobe/restoration/storage with that stand-in. Python suites use synthetic
fixtures and permissive assets. App, wrist and viewer-performance QA commands are in the twin-lab README; only the
performance script uses the synthetic stand-in by design. Pose/lighting changes do not establish reconstruction quality.
Scan clothing beyond garment coverage, coarse boundaries and fused geometry can still limit try-on; a hybrid avoids
those scan-body artifacts, while retaining the limits of its fitted head, photo cameras and rigid hair surface/cards.

## File ownership by wave

| Wave | Work                                                                      | Owner (executor)            | Files                                                                                                                             |
| ---- | ------------------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 0b   | Repo scaffold, contracts, docs                                            | Sonnet (high)               | root, docs                                                                                                                        |
| 1    | Asset pipeline (MakeHuman -> base.glb, morphs, rig, measures)             | Sonnet (xhigh)              | `tools/asset-pipeline/**`, `public/assets/body/**`                                                                                |
| 1    | UI shell: layout, viewer, platform, CameraControls, body panel, TR/EN     | Codex                       | `src/app`, `features/viewer`, `features/body-panel`, `shared/**`                                                                  |
| 1    | avatar-core: morph, measure, solver, skeleton, size conversions + tests   | Sonnet (xhigh)              | `packages/avatar-core/**`                                                                                                         |
| 2    | Integration: `<Avatar/>`, worker, poses + 6 pose JSONs                    | Sonnet (high)               | `features/avatar`, `features/poses`, `workers/`, `public/assets/poses`                                                            |
| 3    | Face: MediaPipe, canonical<->MH map, warp, delighting, blending, skin     | agy (Gemini Flash) + review | `features/face/**`, pipeline `face_map` module                                                                                    |
| 4    | Wardrobe: MHCLO fitting, templates, shoes, size chart form, fit heatmap   | Sonnet + Codex              | `features/wardrobe/**`, pipeline `garments` module                                                                                |
| Twin | Realistic twin: rig export (twin.json, mh2twin.bin), twin mode in the app | Sonnet (xhigh)              | `tools/twin-lab/rig/**`, `features/twin/**`, hooks in `features/avatar`, `store`, `workers`, `features/body-panel`, `app/App.tsx` |
