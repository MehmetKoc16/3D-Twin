# Local FLAME face transplant

`flame_head.py` replaces the scan's facial surface using a shared-identity
Pixel3DMM fit. It runs in **refine/.venv** and imports `headrecon`, `twinrefine`,
`twintex` and the rig GLB reader/writer. Recon remains selectable separately.
No images are displayed. Previews are written for the lead's manual review.

```powershell
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/head/flame/flame_head.py --in user-data/twin/out/refine/refined.glb --fit user-data/twin/head/flame/fit --photos user-data/twin/head/colab_upload --flame-assets user-data/flame --out user-data/twin/out/head/head.glb
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/run_all.py --head auto --dry-run
```

Options: FLAME ears are included by default, with a collar behind the ears;
`--no-include-ears` keeps scan ears. `--include-ears` explicitly selects the default;
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
- The front photo is the white-balance reference. Shared visible midtones match
  **CIELAB L only**, with shadow protection; no per-channel RGB gain or offset
  changes chroma. Five-level pyramids blend luminance. The a/b channels use only
  measured, visible view confidences, with front priority; occluded fills cannot
  dilute observed chroma. RGB textures are decoded/encoded as sRGB, and Lab
  conversions receive sRGB rather than linear RGB. Missing observations use
  nearest surface samples; hidden eyeballs receive an ivory fallback. Surface
  fills, chart padding and canvas fill prevent black/unassigned texels.
- Existing MediaPipe helpers render the scan and lift its detected face
  landmarks into 3D. FLAME embedding barycentrics establish correspondences.
  Stable landmark similarity fitting precedes 70%-trimmed, landmark-anchored
  closest-surface ICP. Scale is clamped to [0.75, 1.35], with a warning on hits.
  Initial and final RMS, maximum residual and trimmed ICP RMS are reported.
- Region masks initialise a signed surface-distance field. A 12mm forehead
  margin moves the boundary under the hairline; the ear region includes a 25mm
  surrounding skull collar. A 35mm upper-neck extension moves the beard border
  below the jaw. Original model boundaries remain excluded. These are physical
  distances on the neutral fit, rather than counts of vertices.
- `twinrefine.armpit.smooth_field` smooths the FLAME field and its closest-surface
  transfer onto scan vertices. Both meshes are cut at the zero level, splitting
  shared crossing edges using `twinrefine.meshops.refine_marked_edges`. Corner
  UVs interpolate independently across UV seams. This replaces whole-triangle
  region selection. The scan cut curve is smoothed; the inserted rim follows
  that curve, with a 0.3mm inward clearance and a ring zipper.
- Geometry displacement fades through a 22mm band on both surfaces. Local
  foldover control reduces displacement where needed. Scan geometry outside
  the band retains exact positions. Small internal FLAME openings are capped.
  Boundary counts, multiplicity, winding, nonzero triangle area and FLAME
  foldovers are verified before export; **bridge width** is separate from the
  zero topological stitch gap. The report includes curve roughness, cut counts
  and the minimum local displacement strength.
- Skin surrounding the face is corrected toward photo Lab chroma over a smooth
  75mm falloff. A luminance offset removes the neck's separate low-frequency
  brightness bias while retaining texture contrast. Dark hair/beard and distant
  body pixels are excluded by a midtone skin gate. On the inserted 22mm band,
  the photographed colour cross-fades to the corrected retained scan texture in
  Lab. Visible front-photo under-chin samples preserve beard detail inside the
  band; unseen beard fades into the scan neck rather than ending at a hard edge.
- Scan, FLAME and bridge/cap charts share one square atlas and one material,
  capped at **4096 squared**. The default preserves the 2048 face chart and
  reduces the scan chart to 2032. A requested 4096 face chart becomes 3072,
  with a 1008 scan chart. Normalised scan UVs stay valid through downsampling;
  cut corner UVs are remapped by the packing affine transforms.
- CIELAB means compare original front-photo skin samples with the same FLAME
  barycentric samples in the bake and the actual packed atlas. Eye/lip regions,
  beard, glasses, clipped highlights and invisible samples are excluded. Neck
  diagnostics measure the packed scan band against the packed face. Reports
  expose L/a/b deltas and `chroma_pass` for |delta a|, |delta b| < 3. When a private
  `round1/head.glb` exists beside the output, matched baseline measurements are
  reported too; absence of a baseline is allowed. No tests load this directory.
- Original extras are restored at root, asset, scene, node, mesh, primitive and
  material/texture/image levels. `asset.extras.dtFlameHead` version 2 records
  provenance, input hash, alignment, smooth stitching and the ears option.
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
symmetry and unseen regions are filled. Illumination is normalised, not fully
delit. Mean Lab values cannot establish that every local shadow or UV island
matches; the lead must assess the private previews. Images are never displayed
by the stage or inspected by agents.

Nearest-surface search refines 24 centroid/vertex candidates without rtree;
very long triangles can defeat that bounded search. Region masks approximate
hairline and skin boundaries; there is no semantic scan hair segmentation.
The 12mm scalp margin may require adjustment for a different hairstyle. The
skin gate is a colour heuristic and can miss very dark skin or misclassify warm
materials near the head. Atlas downsampling reduces scan body detail.

Topology and local foldovers are checked, but global self-intersections and
bridge sliver quality are not exhaustively certified. Thin scan triangles can
limit curve smoothing through local foldover relaxation; the report exposes
this strength. The enlarged ear collar can deform the fitted skull near the
rim, even when its bridge is narrow. Small rear mouth/eyeball openings are
capped rather than reconstructing an oral cavity, teeth or a tongue. Downstream
stages consume the existing single textured primitive; their source ownership
and extras propagation remain with their leads.

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
all head pipeline modes. Round-two tests additionally cover exact level cuts
with interpolated UVs, smoothed curves, unchanged distant geometry, a closed
cut transplant, Lab chroma preservation, soft beard/neck cross-fading, neck
lightness correction, the 4096 packing cap and default-ear CLI behaviour.

On Windows sandboxes that deny access to pytest's mode-0700 temporary directories,
the following invocation changes only directory-creation permissions during tests:

```powershell
New-Item -ItemType Directory -Force tools/twin-lab/head/flame/outputs | Out-Null
tools/twin-lab/refine/.venv/Scripts/python.exe -c "import os,pytest; create=os.mkdir; os.mkdir=lambda path,mode=0o777,*,dir_fd=None: create(path,0o777,dir_fd=dir_fd); raise SystemExit(pytest.main(['tools/twin-lab/head/flame/tests','-q','-p','no:cacheprovider','--basetemp=tools/twin-lab/head/flame/outputs/head-tests-accessible']))"
tools/twin-lab/bundle/.venv/Scripts/python.exe -c "import os,pytest; create=os.mkdir; os.mkdir=lambda path,mode=0o777,*,dir_fd=None: create(path,0o777,dir_fd=dir_fd); raise SystemExit(pytest.main(['tools/twin-lab/bundle/tests/test_run_all.py','-q','-p','no:cacheprovider','--basetemp=tools/twin-lab/head/flame/outputs/pipeline-tests-accessible']))"
```
