# twin-lab / rig - make a scan mesh riggable with OUR rig

Input: an untextured or textured humanoid mesh (GLB, +Y up, facing +Z, feet on y=0, metres, roughly A-pose).
Output: `rigged.glb` = the same mesh (materials/UVs/texture kept) skinned to our 53-bone MakeHuman rig with the
identical bone names, world-aligned identity rest rotations (node translation only) and the exact MakeHuman A-pose
rest frame, so `apps/web/public/assets/poses/*.json` and `PoseDriver` work unchanged (ADR 0004).

Nothing in here is committed data: `.venv/`, `.cache/` are ignored, personal output goes to `user-data/twin/out/rig/`.

## Setup (Windows, Python 3.12)

    python -m venv .venv
    .venv/Scripts/python -m pip install numpy scipy trimesh pillow

Pure CPU, no GPU, no Blender, no bpy. Node scripts use the repo's `three` and `playwright` (chromium, software GL).

## Run

    # non-personal stand-in (MakeHuman body, other shape/pose, re-meshed, noise) + ground truth
    .venv/Scripts/python make_standin.py                       # -> .cache/standin/mesh.glb, truth.json
    .venv/Scripts/python rig_scan.py .cache/standin/mesh.glb .cache/standin_out
    .venv/Scripts/python eval_standin.py                       # rest-joint error vs ground truth

    # user mesh (never leaves user-data/)
    .venv/Scripts/python rig_scan.py ../../../user-data/twin/out/shape/mesh.glb ../../../user-data/twin/out/rig
    node check_pose.mjs  <out>/rigged.glb --json=<out>/check.json      # headless three.js sanity + edge stretch
    node render_preview.mjs <out>/rigged.glb <out>/previews name       # t-pose / walk / hips PNG sheets

Options of `rig_scan.py`: `--weights transfer|geodesic` (geodesic = geometry-only baseline for comparison),
`--keep-pose` (do not unpose to the template A-pose; T-pose then no longer matches the pose JSONs), `--smooth N`.

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
