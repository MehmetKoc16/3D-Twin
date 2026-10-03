# Local FLAME face transplant

`flame_head.py` replaces the scan's facial surface using a shared-identity
Pixel3DMM fit. It runs in **refine/.venv** and imports `headrecon`, `twinrefine`,
`twintex` and the rig GLB reader/writer. Recon remains selectable separately.
No images are displayed. Previews are written for the lead's manual review.

```powershell
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/head/flame/flame_head.py --in user-data/twin/out/refine/refined.glb --fit user-data/twin/head/flame/fit --photos user-data/twin/head/colab_upload --flame-assets user-data/flame --out user-data/twin/out/head/head.glb
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/run_all.py --head auto --dry-run
```

Options: `--include-ears` replaces ears too (default keeps scan ears);
`--texture-size 2048` controls the FLAME atlas (512–4096);
`--preview-dir` defaults to the output's `previews/`; `--no-previews` disables it.
Output and preview destinations must be under this repo's `user-data/`.
The output is a static, embedded-texture, single-primitive GLB for
bodyfix → rig → bundle and twin mode. Output names are `head.glb`,
`flame_head_report.json`, and `previews/{front,right,left,three_quarter,contact}.png`.
The CLI also prints the metrics report as JSON after progress lines.

## Local inputs and licences

Obtain FLAME through registration and acceptance of the terms at the
[official FLAME site](https://flame.is.tue.mpg.de/). Download the masks and
MediaPipe landmark embedding there; keep `FLAME_masks.zip` and
`mediapipe_landmark_embedding.zip` in gitignored `user-data/flame/`.
Unpacked `FLAME_masks.pkl` and `mediapipe_landmark_embedding.npz` in that same
directory are also supported. This stage uses an allowlisted numpy-array
unpickler for the mask dictionary. It never opens `generic_model.pkl`, loads
model weights, imports chumpy, or extracts model assets into source directories.

Follow the existing [Pixel3DMM notebook instructions](../../colab/pixel3dmm_README.md)
to obtain the fit. Pixel3DMM has [CC BY-NC 4.0 terms](https://github.com/SimonGiebenhain/pixel3dmm#license);
FLAME and components of the upstream fitting workflow have separate restrictions.
This stage is for the owner's authorised personal, non-commercial local use.
Its outputs inherit the source restrictions and are not cleared as permissive
web assets. Never commit the fit, masks, models, weights, photos, textures,
previews, metadata about the owner, or derived meshes. No download occurs here.

The fit directory needs `head_neutral.obj` (PLY also accepted by the direct CLI),
`cameras.json` (`dt-flame-head-cameras/1`), and the identically ordered topology in
`fitted_views/front.ply` and `right.ply`. `provenance.json` is embedded if present.
Keep the exported parameters alongside the fit for reproducibility; this stage
does not reevaluate a FLAME model. Photos must be the original `front.jpg` and
`right.jpg` with exactly the dimensions recorded by the exported cameras.
The stage preserves photographed eyeglasses without inpainting. The separate
3D glasses accessory may duplicate the photographed appearance.

## Algorithms and numeric checks

- xatlas (MIT, already in refine requirements) unwraps the neutral topology.
  If unavailable, independent, padded triangle charts provide a deterministic
  fallback; too-small chart resolutions are rejected.
- Texture coordinates are transferred by face index and barycentrics onto each
  view's **posed** mesh. OpenGL cameras use `depth=-q.z`, `u=fx*q.x/depth+cx`,
  `v=cy-fy*q.y/depth`. Crop-edge pixels map continuously back to original image
  edges, then sampling subtracts half a pixel for OpenCV. Padded/outside photo
  coordinates contribute zero confidence.
- A perspective-correct inverse-depth z-buffer at 768² rejects hidden samples.
  Confidence combines the cubed normal/view cosine with distance to silhouette
  and depth-discontinuity edges. Neutral x-mirror nearest neighbours supply a
  mirrored right view only on the opposite side; symmetry error and involution
  ratio are reported rather than assuming exact anatomical symmetry.
- Robust gains/offsets from shared visible midtone texels match colours.
  Five-level Laplacian/Gaussian pyramids blend views. Missing samples are filled
  from nearest surface samples before blending; only filling uses bounded
  sampling and approximate nearest neighbours at mesh vertices, interpolated
  by barycentrics into missing texels. Measured detail remains full
  resolution. Hidden eyeball texels receive an ivory fallback. Chart padding and
  canvas fill prevent black/unassigned texels, including mipmap margins.
- Existing MediaPipe helpers render the scan and lift its detected face
  landmarks into 3D. FLAME embedding barycentrics establish correspondences.
  Stable landmark similarity fitting precedes 70%-trimmed, landmark-anchored
  closest-surface ICP. Scale is clamped to [0.75, 1.35], with a warning on hits.
  Initial and final RMS, maximum residual and trimmed ICP RMS are reported.
- Region masks select face, nose, lips, eyes and eyeballs; scalp, neck and model
  boundary are protected. Ears are protected unless requested. Scan centroids
  map to closest FLAME triangles in the head, with a 45mm distance gate. Removed
  face islands are cleaned topologically. Ambiguous/branching stitch borders
  cause an error rather than a silently disconnected export.
- Exact scan position welding preserves tiny distinct scan features. Retained
  scan positions and triangles stay unchanged. The inserted face has an 18mm
  blend band towards the scan boundary and a small inward clearance; a ring
  zipper adds the connecting strip. Small internal FLAME openings are capped.
  The output verifies boundary counts, edge multiplicity, winding and nonzero
  triangle area and FLAME blend foldovers, then duplicates only UV seams. A shared stitch has zero
  topological gap; **bridge width** is reported separately.
- Scan, FLAME and bridge/cap charts share a square atlas and one material.
  The square layout is intentional: the existing preview renderers assume
  square textures. Nearby compatible scan skin texels receive a spatially
  fading gain correction; dark hair and distant clothes/body texels are gated
  out. Original extras are restored at root, asset, scene, node, mesh,
  primitive and material/texture/image levels. `asset.extras.dtFlameHead`
  records fit provenance, input hash, scale, residuals and stitching metrics.
  The writer validates a reread before atomically replacing the destination.

`run_all.py --head {auto,flame,recon,none}` defaults to `auto`: FLAME when
`<input-dir>/head/flame/fit/head_neutral.obj` exists, otherwise none.
`--with-head` is a deprecated alias for recon; conflicting selections fail.
Explicit modes retain their stage even when inputs are missing, so execution
reports the missing input instead of silently skipping it. Camera, fit, photo,
mask, embedding and imported helper changes invalidate the stage's cache.

## Limitations to assess in the private previews

Alignment is to a generic scan face, whose landmarks may include photographed
glasses and whose depth is an estimate. Residuals do not measure identity accuracy.
Only two images supply independent appearance; the left profile is inferred by
symmetry and unseen regions are filled. Illumination is colour-matched, not fully
delit. Pyramid filtering happens in UV space, so chart boundaries can still show
small colour differences despite shared surface fills and padding.

Nearest-surface search refines 24 centroid/vertex candidates without rtree;
very long triangles can defeat that bounded search. Region masks and closest
surfaces approximate the scan's hairline/ears; there is no semantic scan hair
segmentation. The geometric band and strip are checked for edge topology and
area, but not for all global self-intersections. Thin ear triangles trigger local
blend relaxation to avoid foldovers; this can leave a wider bridging strip and
needs manual review when including ears. Small mouth/eyeball rear openings
are capped, rather than reconstructing an oral cavity, teeth or a tongue.
The packed canvas can be larger than 4096² and should be reviewed on target GPUs.

Bodyfix reads this static GLB and preserves extras; rig reads its single textured
primitive. The current rig writer rebuilds asset JSON and does not forward arbitrary
incoming extras, so `dtFlameHead` propagation beyond rig requires a change owned
by the rig lead. This stage does not modify downstream sources.

## Synthetic verification

```powershell
tools/twin-lab/refine/.venv/Scripts/python.exe -m pytest tools/twin-lab/head/flame/tests -q -p no:cacheprovider
tools/twin-lab/bundle/.venv/Scripts/python.exe -m pytest tools/twin-lab/bundle/tests/test_run_all.py -q -p no:cacheprovider
tools/twin-lab/refine/.venv/Scripts/python.exe -m ruff check tools/twin-lab/head/flame tools/twin-lab/run_all.py tools/twin-lab/bundle/tests/test_run_all.py --config tools/twin-lab/refine/ruff.toml
```

All fixtures are generated spheres, camera matrices, colour images and numpy mask
arrays. Tests never read `user-data/`. Coverage includes camera round trips and
pixel centres, UV fill with xatlas and fallback, perspective depth, symmetry,
similarity clamping, protected regions, manifold stitching with unequal ring
counts, UV duplication, extras, a complete synthetic output/previews run and
all head pipeline modes.

On Windows sandboxes that deny access to pytest's mode-0700 temporary directories,
the following invocation changes only directory-creation permissions during tests:

```powershell
New-Item -ItemType Directory -Force tools/twin-lab/head/flame/outputs | Out-Null
tools/twin-lab/refine/.venv/Scripts/python.exe -c "import os,pytest; create=os.mkdir; os.mkdir=lambda path,mode=0o777,*,dir_fd=None: create(path,0o777,dir_fd=dir_fd); raise SystemExit(pytest.main(['tools/twin-lab/head/flame/tests','-q','-p','no:cacheprovider','--basetemp=tools/twin-lab/head/flame/outputs/head-tests-accessible']))"
tools/twin-lab/bundle/.venv/Scripts/python.exe -c "import os,pytest; create=os.mkdir; os.mkdir=lambda path,mode=0o777,*,dir_fd=None: create(path,0o777,dir_fd=dir_fd); raise SystemExit(pytest.main(['tools/twin-lab/bundle/tests/test_run_all.py','-q','-p','no:cacheprovider','--basetemp=tools/twin-lab/head/flame/outputs/pipeline-tests-accessible']))"
```
