# twin-lab / rig - make a scan mesh riggable with OUR rig

Input: an untextured or textured humanoid mesh (GLB, +Y up, facing +Z, feet on y=0, metres, roughly A-pose).
Output: `rigged.glb` (+ `twin.json` and `mh2twin.bin` for the web app, see "Web app package") = the same mesh (materials/UVs/texture kept) skinned to our 53-bone MakeHuman rig with the
identical bone names, world-aligned identity rest rotations (node translation only) and the exact MakeHuman A-pose
rest frame, so `apps/web/public/assets/poses/*.json` and `PoseDriver` work unchanged (ADR 0004).

Nothing in here is committed data: `.venv/`, `.cache/` are ignored, personal output goes to `user-data/twin/out/rig/`.

## Setup (Windows, Python 3.12)

    python -m venv .venv
    .venv/Scripts/python -m pip install numpy scipy trimesh pillow pytest

Pure CPU, no GPU, no Blender, no bpy. Node scripts use the repo's `three` and `playwright` (chromium, software GL).

## Run

    # non-personal stand-in (MakeHuman body, other shape/pose, re-meshed, noise) + ground truth
    .venv/Scripts/python make_standin.py                       # -> .cache/standin/mesh.glb, truth.json
    .venv/Scripts/python make_standin.py --textured            # keeps the render mesh + UVs, synthetic texture (web app fixture)
    .venv/Scripts/python rig_scan.py .cache/standin/mesh.glb .cache/standin_out
    .venv/Scripts/python eval_standin.py                       # rest-joint error vs ground truth

    # user mesh (never leaves user-data/)
    .venv/Scripts/python rig_scan.py ../../../user-data/twin/out/shape/mesh.glb ../../../user-data/twin/out/rig
    node check_pose.mjs  <out>/rigged.glb --json=<out>/check.json      # headless three.js sanity + edge stretch
    node render_preview.mjs <out>/rigged.glb <out>/previews name       # t-pose / walk / hips PNG sheets

    # the web app on a twin package (dev server running: npm run dev -w @dt/web)
    node app_qa.mjs ../../../user-data/twin/out/rig ../../../user-data/twin/out/app --tag=twin   # poses, zoom, tee + jeans + sneakers

    .venv/Scripts/python -m pytest                             # tests/: twin_export (measures, canonical fit, mapping, twin.json)

Options of `rig_scan.py`: `--weights transfer|geodesic` (geodesic = geometry-only baseline for comparison),
`--keep-pose` (do not unpose to the template A-pose; T-pose then no longer matches the pose JSONs), `--smooth N`.

Bodyfix outputs embed `asset.extras.dtBodyfix` version 1. Rig automatically uses
its full-precision solved macros and net modifiers, fits pose and root translation
on that body, and transfers weights/unposes with its joint positions. No CLI flag
or launcher change is required. An embedded solution takes precedence over
`--reuse-fit`; malformed solutions fail rather than falling back to a fresh shape
fit. Without this metadata the existing fitter and cache behavior are unchanged.
For corrected scans, `measurementsCm` and `measurementsRawCm` are bodyfix's achieved
scan-landmark measurements, with its own clothing allowances. The full solution,
tape targets and residuals are retained in `twin.json.bodyfix` (and the bundle).
These measurements can differ from those measured on the hidden MakeHuman proxy.

## Web app package (`twin.json`, `mh2twin.bin`)

`rig_scan.py` also writes, next to `rigged.glb`, what the web app's "Realistic twin" mode needs (`twin_export.py`; format
and the runtime in `docs/ARCHITECTURE.md`, "Realistic twin"). The three files are picked in the app's twin tab and stay in
the browser; for a real person they live in `user-data/` only.

- `twin.json`: `fittedMacros` / `fittedModifiers` (the fitted MakeHuman body, net modifier values), `measurementsCm`
  (measured on that body with the avatar-core definitions of `measures.json`, minus `clothingAllowanceCm`: what the
  scan's fitted T-shirt, jeans, sneakers and hair add - height 3, neck 0.5, shoulder 1, chest 3, waist 3, hip 2.5, thigh 2,
  upper arm 2.5, arm length 0, inseam 2.5, foot 2.5 cm; estimates, edit the file if you know better), the raw values,
  `boneOrder`, `restHeadsM` and fit numbers.
- `mh2twin.bin`: uint32 little endian per rigged.glb vertex = nearest MakeHuman render vertex (12 candidates, opposing
  normals penalised). The app hides twin triangles under a worn garment with it (plus a footprint pass).
- Canonical fit: the fitter uses an incr and a decr column per modifier, avatar-core the net value. After the fit (and
  after `--reuse-fit`) `canonicalize_fit` rebuilds the rest body from the net values rounded to 6 decimals, so the browser
  reproduces the very skeleton the glb was rigged with (rest heads agree to < 2e-5 m in the web tests). This moves the
  fitted surface by 0.6 mm on average (up to ~1 cm at single vertices) and is applied before the weight transfer.
- Not written with `--keep-pose` (the mesh is not in the rest frame then).

## App QA notes (real twin, local only)

`app_qa.mjs` loads a package into the running web app like a user would and writes PNGs of poses, zoom presets and the
scan under a T-shirt, jeans and sneakers. What to expect: the MakeHuman garments line up with the scan's torso, legs and
feet and hide it under them; the scan's own clothes remain where the new garment does not reach (collar, sleeve stubs,
waistband); scans whose arms touch the torso keep a thin web at the armpit in the T-pose. Nude or minimal-clothing input
views would fix the first, a different arm pose in the input views (A-pose, arms away from the body) the second.

## Scans whose hands were removed (bodyfix)

`bodyfix` cuts the scan's hands off (the web app draws MakeHuman hands) and marks the GLB with
`asset.extras.dtScanHandsRemoved`. `rig_scan.py` then (`bridges.py`): (1) re-weights every vertex with a hand-family weight
above 0.2 by harmonic inpainting over the scan surface from the vertices without hand weight (the wrist cap follows the
forearm; the fitted hand has no fist to fit and can sit on a thigh, so nothing may follow a hand bone), and (2) heals small
surface patches (< 2000 triangles) whose dominant bone is skeleton-distant from everything around them (the forearm end
lying against a thigh is weighted to the thigh by the transfer), instead of letting `--cut-bridges` leave them floating.
Tests: `tests/test_hand_weights.py`.

## Method (approach B, recommended)

1. Articulated + parametric ICP: our MakeHuman body (macro variables + ~130 bounded modifier columns, same morph
   model as avatar-core, reimplemented in `mh.py`) and 19 per-bone rotations (LBS with the rig's own weights) are fitted
   alternately to the scan point cloud (coarse limb-swing search, soft-L1 pose LM with per-bone priors, ridge-regularised
   bounded shape least squares whose macro variables enter as one-sided columns). Coverage gates shrink 20 -> 3.5 cm.
2. Skin weights: closest-point transfer from the posed, once-subdivided fitted body (normal-gated among 24 nearest),
   Laplacian smoothing on the welded scan graph, top-4 renormalised.
3. Unpose: the scan is brought back to the MakeHuman A-pose with the estimated pose (inverse blended matrices), joints
   are the fitted body's joint points. Grounded to y=0.

Approach C (`geoskin.py`, comparison only): same joints, but weights from interior geodesic distance in a voxelised
mesh (Pinocchio / DCC "geodesic voxel" style).

## Auto-riggers evaluated on paper (not adopted)

See the final report in the task hand-off; short version: Make-It-Animatable (MIT code, Mixamo skeleton + Mixamo-trained
weights of unspecified licence, Linux-first), UniRig (MIT code, needs >= 8 GB VRAM + flash-attn/spconv, not Windows
friendly, different skeleton), RigNet (GPL-3.0, CUDA 10-11.3 era, different skeleton). All would need bone renaming and
retargeting; ours keeps names identical.
