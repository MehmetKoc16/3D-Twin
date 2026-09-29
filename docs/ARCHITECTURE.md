# Architecture

## Overview

```
 tools/asset-pipeline (Python)
   raw MakeHuman / Anny data (.cache, gitignored)
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
target (`armslegs/foot-scale-depth-incr` etc.). Two-sided modifiers have `min = -1`, `max = 1`, one-sided ones
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
| `distance`      | Euclidean distance of 2 vertices, or `abs` difference along `axis` (`shoulder`: x, `footLength`: z)    |
| `polyline`      | sum of consecutive vertex distances (`armLength`: acromion, elbow, wrist along the skin)               |
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

## Planned avatar-core public API (implemented in Wave 1; signatures only)

```ts
macroWeights(manifest: BodyManifest, vars: Partial<Record<MacroVar, number>>): Map<string, number>;
applyMorphs(base: Float32Array, morphs: ArrayBuffer, manifest: BodyManifest,
            weights: Map<string, number>, out: Float32Array): void;
measure(def: MeasureDef, positions: Float32Array): number; // meters
solveBody(data: { manifest: BodyManifest; base: Float32Array; morphs: ArrayBuffer; measures: MeasuresDef },
          params: BodyParams, opts?: { maxIterations?: number }): {
  weights: Map<string, number>;
  positions: Float32Array;
  achievedCm: Partial<Record<MeasureId, number>>;
  estimatedMassKg: number;
};
computeJoints(rig: RigDef, positions: Float32Array): Map<string, { head: Vec3; tail: Vec3 }>;
footLengthCmFromShoe(shoe: { system: ShoeSystem; size: number }): number;
shoeFromFootLengthCm(cm: number, system: ShoeSystem): number; // EU ~ 1.5 * foot_cm + 2, approximate
```

`solveBody`: bounded Gauss-Newton on a finite-difference Jacobian. Macros first (height/weight/BMI), then local
measure modifiers. Mass estimate = mesh volume x ~1.01 kg/L; measurements take priority on conflict.

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
