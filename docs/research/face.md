# Research: face from photo (verified findings)

- **MediaPipe Face Landmarker:** `@mediapipe/tasks-vision` v1.0.1, Apache-2.0, ~3.7 MB `.task` model, works in
  IMAGE mode, outputs 478 landmarks and 52 blendshapes.
  [Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- **Index mapping:** in `canonical_face_model.obj` the landmark index equals the vertex index (see the Codrops
  article of 2026-09-06). No turnkey repo exists for a custom-head UV warp; a one-time mapping table to the
  MakeHuman head is custom work.
- **Pitfalls:** the neck/forehead seam needs gradient/Poisson blending; delight the photo (divide low-frequency
  lighting); exclude the hair region (hair is a separate asset).
- **FLAME 2023:** open model, CC-BY-4.0 (earlier FLAME versions are non-commercial). **Pixel3DMM** needs a GPU.
  **FaceLift** weights are non-commercial.
- **Hosted avatar services:** Ready Player Me shut down on 2026-01-31 after the Netflix acquisition; Avaturn and
  Avatar SDK MetaPerson are alive but paid.
