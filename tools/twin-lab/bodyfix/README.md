# twin-lab / bodyfix

Correct the scan's body to tape measurements before rigging. CPU only; uses the
existing `rig/.venv` and imports `rigfit.py`, `mh.py`, and `twin_export.py` without
changing them. No new dependencies, downloads or uploads.

From the repository root:

```powershell
tools/twin-lab/rig/.venv/Scripts/python.exe tools/twin-lab/bodyfix/bodyfix.py --in user-data/twin/out/refine/refined.glb --measurements user-data/twin/measurements.json --out user-data/twin/out/bodyfix/bodyfixed.glb
tools/twin-lab/rig/.venv/Scripts/python.exe tools/twin-lab/run_all.py --from-stage bodyfix --dry-run
tools/twin-lab/rig/.venv/Scripts/python.exe -m pytest tools/twin-lab/bodyfix/tests -p no:cacheprovider
```

If a head stage is installed, use its output as `--in`. The launcher inserts
bodyfix immediately before rig, currently after refine; the separately added
head launcher block must go upstream of bodyfix. Its interpreter
is always `rig/.venv`, including when a bodyfix venv happens to exist.

The JSON object accepts any subset of `heightCm`, `weightKg`, `shoulderCm`,
`neckCm`, `chestCm`, `waistCm`, `hipCm`, `armLengthCm`, `inseamCm`, `thighCm`,
`upperArmCm`, and `shoe: {"system": "EU", "size": 42}`. Values must be positive,
finite numbers; unknown fields fail to catch misspellings. No measurements file,
or an empty object, produces a byte-exact copy and an explicit NO-OP log/report.
Adding, removing or editing the file invalidates the launcher's stage cache.

Options: `--keep-hands` keeps the scan's own hands (default: they are removed, see Method 2).

## Method

1. Fit shape and pose with the existing articulated MakeHuman ICP fitter;
   canonicalize its net modifier values using the rig export implementation.
2. **Remove the scan's hands** (`handcut.py`; the app draws MakeHuman hands anyway). Scan vertices whose nearest fitted-MH
   surface point is dominated (> 0.5) by the hand/finger bones are "hand"; every triangle touching one is deleted, which
   also cuts fists fused to the thighs. Pieces cut loose by this and small relative to the body are deleted (no orphan
   fragments), and each new boundary loop (wrist opening, scar on the leg) is closed by a flat fan whose triangles
   copy the UVs of the edge they close. The fit itself still uses the full scan; everything below works on the cleaned
   mesh. The app's own hand hiding keeps working (it hides by skin weight, and nothing is left to hide but the wrist cap).
   The report's `scanHands` lists the counts. Scans with several primitives skip this step.
3. Construct scan-to-MH barycentric correspondences on normal-gated triangles
   incident to nearby surface vertices, and MH-to-scan landmark correspondences.
   Unpose scan coordinates through the fitted blended affine transforms for tape
   measurements. The saved scan keeps its original pose for the subsequent rig.
   If a scan retains MH vertex topology, use its exact surface anchors; arbitrary
   re-meshed scans use the geometric triangle search.
4. Solve height first on the bounded height macro, then solve only the provided
   measures' `measures.json` drivers by bounded least squares. Re-solve height
   inside each local residual. Non-driver modifiers, gender and muscle preserve
   the fitted character; the weight macro can change for a provided mass.
5. Smooth the target-minus-fitted displacement on the welded MH surface twice,
   then barycentrically transfer it to every scan vertex. On the arms and shoulders the sampled field is additionally
   **diffused over the scan surface** (60 uniform-Laplacian steps on the position-welded edge graph, precomputed as one
   sparse operator; blended in with a faded region from the arm/shoulder skin weights): loose sleeves and the lips of the
   refine stage's armpit caps map to different body parts and used to receive very different displacements, which tore
   wing-like flaps at the shoulders when the arm length or girth changed. The diffusion turns the jump into a gradient,
   spreads a length change along the arm, and the cap (connected to both lips) moves with them; the armpit gap is not
   connected, so arm and torso keep their own fields. Torso, neck, legs and feet keep the plain field so they hit
   their tape targets exactly. Repeat the
   bounded solve against transferred scan landmarks to close transfer residuals.
   Head geometry above the neck receives a rigid translation with the neck,
   with a 5 cm smooth transition below it. Hands and feet blend 85% toward joint
   translation, retaining limited deformation for transitions. Feet additionally
   scale as blocks along their depth to accommodate the requested shoe size.
6. Ground the scan and write `bodyfixed.glb` plus `bodyfix_report.json` beside it.
   Embed `asset.extras.dtBodyfix` version 1: the full-precision solved
   `fittedMacros`/`fittedModifiers`, `targetsCm`, `achievedCm`, `achievedRawCm`,
   `residualsCm`, `clothingAllowanceCm`, and `measurementBasis`. Rig discovers
   this metadata automatically, including on renamed or moved scans. It keeps
   the solved shape and fits only pose/translation before transferring weights
   and unposing. The twin package carries the achieved scan measurements and
   the original solution; proxy surface measurements are not substituted for
   scan landmarks. Invalid embedded solutions fail explicitly; scans without
   this metadata retain the existing unconstrained fitter.
   Append replacement positions/normals to the original GLB; UVs, original BIN
   bytes, embedded textures/materials and JSON extras survive. Old tangents are
   removed because they no longer describe the new geometry.

## Measurement semantics and assumptions

UV seams and constant-UV cap rims are geometrically coincident copies with
different normals, so their closest-body correspondences can disagree. Bodyfix
averages both affine transforms and offsets over the input's 10 micrometre
position groups before unposing/deforming, as well as sharing the displacement
field. This keeps the caps and hand-closing fans closed. The larger, intentional
arm/torso lip gap is never welded or repaired by a proximity search.

Uses the avatar-core definitions and the numpy evaluator from `twin_export.py`:
convex-hull tape loops on their Newell plane, shoulder **back-surface polyline via
C7**, shoulder–elbow–wrist arm length, and crotch height from the scan's floor.
The report's before/after measurements sample fixed anatomical landmarks on the
actual scan surface in the fitted rest pose. Height uses all scan vertices;
`scanHeightCm` separately records the saved posed mesh bbox. Correspondence is
approximate on scans with fused body parts or substantially different anatomy.

Body tape targets are compared to clothed geometry by **adding** the imported
allowances (cm): height 3, neck 0.5, shoulder 1, chest 3, waist 3, hip 2.5,
thigh 2, upper arm 2.5, arm length 0, inseam 2.5, foot length 2.5. Height's 3 cm
includes approximately **2 cm sneaker soles + 1 cm hair**, not another 2 cm on top.
These estimates assume a fitted T-shirt, jeans and sneakers. Shoe conversion
matches avatar-core's EU table with interpolation/extrapolation.

Mass is signed MH mesh volume × 1.01 kg/L, a soft residual scaled by 10 kg versus
0.25 cm for tape measurements. The report explicitly labels this as a **clothed
MH proxy**: garments/hair are not separable volumetric layers, so mass is not a
precise naked-body estimate. Tape targets take precedence over conflicting mass.
Unreachable requests are reported, with 0.5 cm height / 1.5 cm other-measure
thresholds; bounds are never enlarged. Missing measures impose no extra targets.

Only static, uncompressed triangle GLBs with embedded buffer/textures are
accepted. Skins, animation, instancing, morph targets and sparse positions are
rejected rather than silently discarded. This is an intermediate stage; rig and
bundle subsequently produce the shared single-file `dtTwin` contract.

Tests use `rig/make_standin.py`'s CC0 MakeHuman body and generated texture (`test_handcut.py`: fused-hand stand-in -> no
bridges, fragments or open wrists; arm-length change -> no edge near the shoulder stretched more than 2x; field diffusion),
including direct measurement of the corrected output, head rigidity, preserved
texture/UV bytes, partial targets, missing-file copies, validation, shoe/mass and
launcher ordering. All generated fixtures stay in ignored `bodyfix/outputs/`;
real input/output/report files must remain under `user-data/`.
