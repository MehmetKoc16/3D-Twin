# twin-lab / refine - face relief + armpit separation (between `texture/` and `rig/`)

Input: the textured scan of the texture stage (`user-data/twin/out/texture/textured.glb`) and the front photo.
Output: `refined.glb` - the same textured, single-primitive mesh (same material / atlas layout, UVs kept) with

1. a **real face relief** instead of the shape model's lumpy generic face (shallow eye slits, no lips, blotchy under light),
   and the front photo (or a higher resolution `face.png`) projected again onto it;
2. the **arms cut free from the torso** where the scan fuses them (armpit webbing in a T-pose).

```
python refine.py --in user-data/twin/out/texture/textured.glb --out user-data/twin/out/refine/refined.glb \
                 [--front user-data/twin/front.png] [--face user-data/twin/face.png]
```

`--front` / `--face` default to `user-data/twin/{front,face}.png` (face is optional). Only the files named above are read; nothing
is uploaded, everything runs locally (CPU only, ~1 min with a cached body fit, ~5 more minutes for the first body fit).
Next stage: `rig/rig_scan.py refined.glb <out> --fingers merge --cut-bridges` (same options as for `textured.glb`).

Other options: `--no-face`, `--no-armpits`, `--no-rebake`, `--no-previews`, `--preview-dir`, `--fit-cache` (body fit pickle,
default `<out dir>/cache/fit.pkl`; delete it when the input mesh changed), `--gap-mm` (arm nudge, 4), `--y-top` (armpit zone top),
`--face-edge-mm` (target edge length in the face region, 2.2), `--jpeg-quality`.

Outputs next to `--out`: `refined.glb`, `refine_report.json` (all numbers below), `cache/fit.pkl`, `previews/`
(`head_compare.png`, `tpose_compare.png`, `armpit_compare.png`, `before_*`/`after_*` head renders: lit, textured and clay under a raking light).

## Setup (Windows, Python 3.12)

```
cd tools/twin-lab/refine
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install --no-deps mediapipe==1.0.1      # Apache-2.0; its declared opencv-contrib dependency is not needed
.venv/Scripts/python -m pytest                                      # synthetic + CC0 data only (48 tests, ~10 s)
```

The Face Landmarker model is the one the web app ships (`apps/web/public/models/face_landmarker.task`); no download. The stage
imports the sibling stages without editing them: `texture/twintex` (cameras, flow alignment, rasterisers) and `rig/` (`mh.py`
MakeHuman model, `rigfit.py` fitter + weight transfer, `glbio.py`). If MediaPipe is missing or finds no face, the face step is skipped
(reported) and the armpit step still runs.

## Method

### Face relief (`landmarks`, `frontview`, `facefit`, `facedepth`, `hairmask`, `relief`, `rebake`)

1. **Same frame as the texture stage.** `frontview` rebuilds the texture stage's front view (orthographic camera from silhouette IoU +
   the smooth "flow" warp, same cut-out when `report.json` points at the shape stage) and exposes world <-> photo pixel maps including
   the inverse of the flow. The camera is reproduced to 0.01 px/m.
2. **Landmarks.** MediaPipe Face Landmarker (478 points) on a head crop of the full-body photo, mapped to world x / y.
3. **Fit the MakeHuman face** (python port of `packages/avatar-core/src/face/fitFace.ts`): the 468 landmarks are bound to the
   body surface (`face-map.json`); the face modifiers (chin width/height, nose width/height, mouth width, eye scale, cheek volume)
   are solved by bounded Levenberg-Marquardt against the photo landmarks (stable subset, best 2-D similarity, Tikhonov to neutral),
   on the body that the rig stage's fitter already posed onto the scan.
4. **Depth.** The front-facing triangles of that head are rasterised along the camera axis (0.5 mm grid), warped by a thin-plate spline
   so all 468 landmarks land exactly on the photo's (the face features then sit under the right photo pixels), the eyelid openings and
   the lip slit are closed by harmonic interpolation (+ a slight eyeball bulge) so the surface stays closed, and the face is joined
   to the scan Poisson-style: `Znew = Zmh + c`, `c` = harmonic extension of the (robustly cleaned) scan - face residual taken on a
   band inside the face boundary. The boundary is the landmark oval, grown up over bright forehead skin to the real hairline and
   cut where hair is dark or the scan surface is steeper than 4.5 (cheek sides), then feathered over 14 mm.
5. **Mesh.** Faces under the mask are refined (edge splitting to 2.2 mm, crack free, UVs interpolated per corner), edges are flipped
   to the Delaunay triangulation of the front projection (the decimated scan is full of slivers), vertices on the front surface are
   snapped onto the new surface (this also removes the scan's own spikes), displacement is diffused where the front-surface gate
   switches, the jaw / hairline band is lightly smoothed and fold-overs are relaxed. Only z moves: every vertex keeps its photo pixel.
6. **Texture.** The face texels are sampled again from the front photo (same camera + flow) - weight = visible x incidence x feather;
   with `face.png` the portrait is registered to the front photo by a thin-plate warp of its landmarks, per-channel gains match its
   colours, and the front photo fills what the portrait does not cover.

### Armpits (`armpit`, `meshops`, `atlaspatch`, `posecheck`)

1. Arm / not-arm labels from the skin weights the rig stage will compute (transfer from the fitted MakeHuman body), cleaned
   (largest connected patch, islands absorbed), smoothed into a field whose zero level is the cut line.
2. Only if a T-pose stretches >= 40 armpit edges more than 2x (`posecheck`: weight transfer -> unpose -> T-pose LBS, in process).
3. The surface is cut along the zero level inside the contact zone (armpit apex of the fitted body + 3 cm down to the free arm); the cut
   is an open arc, so the shoulder stays attached and keeps deforming like a shoulder.
4. Both lips are closed by a small cap: a ladder ("zipper") triangulation between the two creases of the U-shaped lip, so the rungs are
   short and the cap moves like a strip (ear clipping for closed loops / other shapes); one solid colour patch in unused atlas
   space = the lips' mean colour.
5. The arm side is nudged outward by 4 mm (tapering to 0 at the apex and below the contact). The rig stage welds vertices by position
   (1e-5 m), so lips that coincide would be stitched together again and get the same skin weights.

## Results on the current scan (`user-data/twin/out/refine/`, local only)

| | before | after |
| --- | --- | --- |
| face under light | dark blotches under the eyes / on the nose, pouting lips in profile | smooth skin, nose, lips, brows, eyelids; natural profile |
| faces / vertices | 79 998 / 47 430 | 130 570 / 74 038 |
| T-pose edges stretched > 2x near the armpits | 627 (curtain-like webbing) | 184 (cap edges), no sheet between arm and torso |
| rig stage `check_pose` (all edges, T-pose) | 798 edges > 2x, max 9.3x | 612 edges > 2x, max 6.3x (rest: elbows / hands fused to thighs) |
| face fit (stable landmarks, RMS) | 9.2 mm (neutral MakeHuman face) | 3.7 mm after the modifier fit, 0.8 mm after the thin-plate warp |

See "Limitations" for what is not fixed. Numbers of every run: `refine_report.json`.

## Tests (`tests/`, synthetic + CC0 only)

`test_meshops` (refinement stays closed / conserves volume, wedge cut, caps, ear clipping, Delaunay flips, seams), `test_armpit`
(a peanut-shaped "torso + arm": closed loop -> two closed parts; open arc -> one closed surface; gap; cap colour; no-op),
`test_face` (face-fit recovers the modifiers, depth map recovers the detail layer, hole fill, relief without cracks),
`test_hairmask`, `test_rebake_front` (photo colours land on the texels, portrait gains, flow inverse, head crop),
`test_render_pose`, `test_pipeline` (end to end on the MakeHuman body as the scan with injected landmarks; CLI).

## Limitations

- The face relief is the MakeHuman face (warped to the photo's landmarks), not a reconstruction: likeness of the relief = the 7
  face modifiers + the landmark warp; depth cues of the photo (MediaPipe z) are not used. Nose / lip protrusion are the MakeHuman
  proportions scaled to the head.
- Front view only: cheeks at grazing angles, ears, the jaw underside and the neck keep the scan's geometry and the old (filled)
  texture there; a side view (`left.png`) would improve them in the texture stage.
- The relief is a height field: nostril undersides and the inside of the lips are not modelled.
- Armpit caps are flat membranes with a solid colour: visible from below in a T-pose as small grey flaps; their outline follows the
  noisy crease of the scan. Hands fused to the thighs are not handled here (`rig_scan.py --cut-bridges` drops those triangles).
- The face step needs MediaPipe and the landmarker model; it assumes a frontal, un-mirrored photo.
- Hunyuan3D-2 licence terms of the shape stage apply to everything derived from its meshes (see `../README.md`).
