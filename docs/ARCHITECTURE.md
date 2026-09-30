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
- `BodyParams`: user input (gender 0..1, cm values, shoe size). `GarmentDef` is a DRAFT until Wave 4.

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

| type            | definition                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| `circumference` | perimeter of the 2D convex hull of the loop vertices projected on the Newell plane of the ordered loop |
| `distance`      | Euclidean distance of 2 vertices, or `abs` difference along `axis` (`footLength`: z)                   |
| `polyline`      | sum of consecutive vertex distances (`armLength`: acromion, elbow, wrist; `shoulder`: back surface via C7) |
| `height`        | bbox Y extent over render vertices only                                                                |
| `vertexHeight`  | `y(vert)` minus the lowest render vertex (`inseam`: crotch)                                            |

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

## File ownership by wave

| Wave | Work                                                                    | Owner (executor)            | Files                                                                  |
| ---- | ----------------------------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------- |
| 0b   | Repo scaffold, contracts, docs                                          | Sonnet (high)               | root, docs                                                             |
| 1    | Asset pipeline (MakeHuman -> base.glb, morphs, rig, measures)           | Sonnet (xhigh)              | `tools/asset-pipeline/**`, `public/assets/body/**`                     |
| 1    | UI shell: layout, viewer, platform, CameraControls, body panel, TR/EN   | Codex                       | `src/app`, `features/viewer`, `features/body-panel`, `shared/**`       |
| 1    | avatar-core: morph, measure, solver, skeleton, size conversions + tests | Sonnet (xhigh)              | `packages/avatar-core/**`                                              |
| 2    | Integration: `<Avatar/>`, worker, poses + 6 pose JSONs                  | Sonnet (high)               | `features/avatar`, `features/poses`, `workers/`, `public/assets/poses` |
| 3    | Face: MediaPipe, canonical<->MH map, warp, delighting, blending, skin   | agy (Gemini Flash) + review | `features/face/**`, pipeline `face_map` module                         |
| 4    | Wardrobe: MHCLO fitting, templates, shoes, size chart form, fit heatmap | Sonnet + Codex              | `features/wardrobe/**`, pipeline `garments` module                     |
