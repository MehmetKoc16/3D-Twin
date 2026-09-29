# ADR 0006: face-map.json - MediaPipe canonical landmarks bound to the MakeHuman head

## Context

The selfie pipeline (ADR 0003) needs a fixed correspondence between the 468 MediaPipe face landmarks and the MakeHuman
head: the photo is warped piece-wise affinely into the body texture, and the head shape is fitted to the landmark
layout. The pipeline produces this table once (`tools/asset-pipeline/face_map.py`, contract `FaceMapDef` in
`packages/avatar-core/src/faceContracts.ts`).

## Decision

**Sources (Apache-2.0, pinned).** `canonical_face_model.obj` and `face_mesh_connections.py` from
`google-ai-edge/mediapipe@9519bb59bf55fc6a79ed5b9f283d72e6cdfb6678`, downloaded by `fetch.py` into `.cache/mediapipe/`
and verified by sha256. Only index data is copied into the output (the 898 triangle index triples and the connection
lists for the face oval, eyes and lips); the canonical vertex positions and UVs are used during the build only.

**Frames and left/right.** MakeHuman: meters, +Y up, +Z front, +X = the subject's LEFT. Canonical model: cm, +Y up,
+Z out of the face, +X = the subject's left too: landmark 263 (MediaPipe "left eye") has x > 0, and in a photo the
subject's left is on the image right (image +x). Axes therefore correspond one to one and no mirroring is involved;
the build asserts it (`left eye mean x > 0` in both frames). `regions.leftEye / leftCheek` are the subject's own
left side (+X) throughout. The photo passed to MediaPipe must be un-mirrored.

**Alignment.** A symmetric similarity `T(q) = s * Rx(pitch) * q + (0, ty, tz)` (uniform scale, pitch, translation; no
yaw/roll/x-shift, so mirror symmetry is exact) is fitted in two steps: (1) least squares on 6 anchors: the eyelid
opening centres and lip slit centre / corners (the boundaries of the eye-socket and mouth-cavity UV islands) and the nose
tip; (2) trimmed ICP (85 % of the surface points, 20 steps) against the front-facing triangles of the head island with
the 6 anchors kept in the objective at weight 12 (without them the scale drifts by 5 %). Result: 0.00917 m per canonical
cm, pitch 0.02 deg. Anchors land within 1 to 3 mm; the alignment overlay is convincing (canonical wire matches eyes,
nose, lips, jaw).

**Binding.** Each canonical vertex is bound to the closest point on the front-facing triangles of the head UV island,
measured in a metric that weights depth (z) by 0.2 (`Z_WEIGHT`). Reason: the photo warp and the fit both live in the
frontal projection, so the bound point should be the one closest in (x, y); pure closest-point (weight 1) leaves 1.5 mm
mean xy error, weight 0.2 gives 0.47 mm mean / 5.5 mm max, at the price of a larger depth distance on the grazing jaw
(mean 3.2 mm, max 16.5 mm on the oval). `tri` are render-vertex indices (same index space as morphs.bin), `bary` sum to
1 (rounded at 6 decimals, clipped at 0), `uv` is interpolated from `TEXCOORD_0` (glTF convention, v down, `1 - v_obj`).
All 468 landmarks are bound.

**Seams.** The head is a single UV island (the one holding the nose tip). Every bound triangle has its three vertices on that island, so UV interpolation is continuous. The eyelid openings
and the lip slit are real holes of the island (the eyeballs are separate geometry; sockets and mouth cavity are their
own islands): canonical faces that touch the eye or lip contours collapse into those holes. Sampling the centroid and
edge midpoints of all 898 canonical faces in UV space, no face outside the eye / lip contour leaves the island (checked
in `seam_report`, asserted by the tests).

**UV orientation (browser baker).** The head island is unwrapped rotated by 90 degrees: with canonical `x` (subject's
left) and `y_down = -y`,
`u = 0.8667 + 0.00982 * y_down` and `v = 0.5168 - 0.01194 * x` (linear least squares over the 468 landmarks, RMS 0.0065
uv units: the unwrap curves toward the jaw and forehead, max deviation 0.04). So
u grows from forehead to chin, v decreases toward the subject's left; a photo is effectively rotated 90 degrees
counter-clockwise and not mirrored. Do not assume an axis-aligned face in the texture: warp triangle by triangle with
the affine map from photo landmark pixels to `landmarks[i].uv` (piece-wise affine over `triangles`). The face occupies
`uvBounds` = u 0.7949..0.9654, v 0.3871..0.6466. The canonical triangles keep one orientation in UV (signed area
negative). 13 of 898 faces are collapsed slivers at the eyelid / lip holes (each below 1.1e-5, together 0.04 % of the face
area in UV; 11 of them touch an eye or lip contour) and come out flipped or ~zero area; the baker should skip
faces whose UV area sign differs from the majority or whose area is below `1e-6`, and paint the eyes / mouth from the
eyeball and mouth helper materials, not from the photo.

**Regions.** `leftEye`, `rightEye`, `lips`, `faceOval` (ordered walk 10, 338, 297, ...: from the top of the forehead
toward the subject's left, down to the chin and back up on the right) come from MediaPipe's connection lists.
MediaPipe has no cheek / forehead list, so those are defined from the canonical geometry: forehead = above the highest
eyebrow landmark; cheeks = between the lower lid line and the mouth corners, outside the nose wings and mouth corners,
inside the oval, split by the sign of x. Cheeks (14 landmarks each side) are skin patches for tone sampling.

**Fit modifiers.** Candidates were tested by finite differences on the stable landmarks (2-D layout after removing
translation, rotation and uniform scale); a modifier qualifies when every side changes the layout by at least 0.6 mm
RMS, and must be identifiable (unique effect of at least 20 % of its norm after regressing the others). Result (9):
`head/head-scale-horiz`, `head/head-fat`, `chin/chin-width`, `chin/chin-height`, `nose/nose-scale-horiz`,
`nose/nose-scale-vert`, `mouth/mouth-scale-horiz`, `eyes/eye-scale`, `cheek/cheek-volume`. Rejected:
`forehead/forehead-scale-vert` (0 mm: it only moves hair-covered surface), `head/head-scale-vert` (a uniform scale is
removed, so it only differs from head-scale-horiz by that scale: 5 % unique effect), `mouth/mouth-scale-vert`,
`chin/chin-jaw-drop`, `nose/nose-nostrils-width` (below 0.6 mm), depth-only modifiers (`*-scale-depth`,
`chin-prominent`).

**Shape fitting.** `fitFaceModifiers` (avatar-core `face/fitFace.ts`) minimizes the millimetre residual between the
photo landmarks and the frontal projection of the bound points, with the similarity removed in closed form, a Tikhonov
term toward 0 (2 mm^2 per unit^2) and bounded Levenberg-Marquardt. Only stable landmarks are used (no eye contours,
eyebrows, lips except the two outer corners). Because a 4 degree pitch alone biases head-fat and nose height by more than
0.3, the head pose is estimated from the landmark depth (Horn) and removed; without depth it falls back to the frontal
assumption. On synthetic faces the values are recovered within 0.025 (noise 0.4 px, any similarity transform, yaw and
pitch up to 12 degrees), one call takes about 4 ms (27 ms cold).

## Pipeline changes shipped with it

- `shoulder` is now a `polyline` over the back surface (acromion, upper back over C7, other acromion), the tape path
  used by garment charts, instead of the straight biacromial distance. C7 = the midline back vertex at the height of the
  neck joint; the path is the slice of the back by the plane through both acromia and C7 (17 vertices). Neutral values:
  female 37.3 cm, male 40.5 cm (biacromial: 34.1 / 37.7). The size-chart values of 44 to 50 cm for adult men are reached
  through the modifier range (male 190 cm / 75 kg: 39.0 to 50.5 cm).
- The measure modifiers may go beyond +-1 (linear extrapolation of the MakeHuman target) where it is geometrically sane
  (`targets.EXTENDED_RANGE`): neck, shoulder, waist, hips, thigh and lowerarm-length to +-1.5, upperleg / lowerleg height
  to +-1.5, upperarm-circ and upperarm-length only upward (the negative side folds the arm), bust-circ only downward
  (already folds around the nipples at +1). Checked by `test_extended_ranges_are_geometrically_sane`: no folded
  (normal turned by more than 78 degrees) or collapsed (area below 20 %) triangle on neutral, heavy male and thin female
  bodies, monotonic perimeter, loop convexity kept within 0.05.

## Consequences

- `face-map.json` (86 KB) is part of the deterministic outputs (`--check` covers it).
- The face texture layout is rotated; anything that assumes an upright face in the head UV island is wrong.
- Reaching the pipeline files' new ranges changes the achievable measures (waist +5 cm, hip +8 cm, ...); the solver picks
  it up through the manifest ranges, no code change.
