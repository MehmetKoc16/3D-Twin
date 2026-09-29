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

All types live in `packages/avatar-core/src/contracts.ts` (source of truth).

- `BodyManifest` (`manifest.json`): `vertexCount`, file names, `macroVariables` (with tent-interpolation
  `buckets`), `targets` (offset/count into `morphs.bin`, optional `macroConditions`), `modifiers`
  (slider value -> decr/incr target), license info.
- Morph model: `v = base + sum_i w_i * delta_i`. Macro targets get weight = product of tent weights of their
  `macroConditions`. A modifier with value < 0 uses `|value| * decrTarget`, > 0 uses `value * incrTarget`.
- `RigDef` (`rig.json`): bones with `head`/`tail` `JointRef` (`MEAN` of vertices, single `VERTEX`, or `FIXED`
  position) so joints are recomputed from the morphed mesh.
- `MeasuresDef` (`measures.json`): per `MeasureId` a `circumference` / `distance` / `polyline` / `height`
  definition (vertex indices) and the `driver` modifier that mainly controls it.
- `PoseDef` (`poses/*.json`): bone-local quaternions relative to rest pose, labels tr/en.
- `BodyParams`: user input (gender 0..1, cm values, shoe size). `GarmentDef` is a DRAFT until Wave 4.

### morphs.bin binary format

Concatenated sparse target entries, little-endian, 16 bytes each:

| bytes | type    | meaning      |
| ----- | ------- | ------------ |
| 0-3   | uint32  | vertex index |
| 4-7   | float32 | dx (meters)  |
| 8-11  | float32 | dy (meters)  |
| 12-15 | float32 | dz (meters)  |

A `TargetDef` addresses its run via `byteOffset` (bytes) and `count` (entries).

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
