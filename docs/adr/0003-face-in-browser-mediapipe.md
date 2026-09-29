# ADR 0003: Face from selfie entirely in the browser with MediaPipe

## Context

Users want their face on the twin. Selfies are personal data and services must be free. `@mediapipe/tasks-vision`
Face Landmarker (Apache-2.0, ~3.7 MB model) runs client-side in IMAGE mode and returns 478 landmarks and 52
blendshapes. The canonical face model's landmark index equals its vertex index. Server-side options (FLAME 2023,
Pixel3DMM) need a GPU; Ready Player Me shut down (2026-01-31); Avaturn and MetaPerson are paid.

## Decision

Run MediaPipe in the browser; map canonical face vertices to the MakeHuman head with a one-time table generated
by the pipeline; piecewise-affine warp into the head UV via WebGL render-to-texture, with simple delighting,
gradient/Poisson blending at the neck/forehead seam, and hair excluded. Landmark ratios drive a few face targets.
Skin tone is sampled from the cheeks.

## Consequences

- The selfie never leaves the device; no server cost.
- Quality is limited by a planar warp; no custom head geometry from a single photo.
- The canonical-to-MakeHuman mapping is custom work (no turnkey repo).
