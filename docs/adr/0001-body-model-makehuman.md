# ADR 0001: Body model = MakeHuman (CC0), not SMPL-X

## Context

The project is a public repo using only free services. SMPL-X is licensed for non-commercial use only and
cannot be redistributed in a public repo. MakeHuman assets (base mesh, targets, rig, clothes) are CC0; only the
application code is GPL-3.0. MakeHuman ships ready-made macro targets (gender, age, muscle, weight, height,
proportions) and measure targets (neck, chest, waist, hip, ...), matching our slider UI. NAVER Anny (Apache-2.0)
is MakeHuman-based and useful as a reference for anthropometry.

## Decision

Use the MakeHuman base mesh and targets (data only, never the GPL code). Convert raw data with a Python
pipeline (no Blender) to `base.glb`, `morphs.bin`, `manifest.json`, `rig.json`, `measures.json`. Use Anny only as
reference, and only its default MakeHuman topology (its SMPL-X variant is non-commercial).

## Consequences

- License-clean public repo. Morphs are linear, so the solver is cheap and runs in the browser.
- Mesh quality is stylized rather than scan-accurate.
- We own the pipeline and the target/measure mapping.
