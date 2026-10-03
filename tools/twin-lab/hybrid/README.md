# hybrid - template-character twin (Avaturn-style)

One artist-made template character with fixed topology, fixed UVs and a rig: the **MakeHuman body already used by the
web app**. Nothing is scanned and no foreign head is grafted.

1. **Body**: the base mesh with the solved macros and modifiers (bodyfix solution, re-solved natively on the tape
   measurements without clothing allowance, `body.py`).
2. **Head shape**: the template's _own_ head vertices are deformed to the fitted neutral FLAME surface (non-rigid
   registration, `register.py` / `headfit.py`). Same vertices, same topology, no seam. The neck loop and everything below
   it keep their exact positions (neck girth is verified to 1e-6 cm).
3. **Face texture**: the two photos are projected with the exact FLAME cameras into the template head's **fixed UV
   island** (`facetex.py`, reusing the proven `head/flame` baker: z-buffer visibility, view-angle weights, luminance
   matching, multiband blend, mirrored right view for the unseen side). The skin tone is propagated to the whole body
   texture and boxer shorts are painted (`skin.py`).
4. **Parts**: MakeHuman CC0 eyes, lashes, hair (and optionally brows) are bound to the _deformed_ body with the app's own
   MHCLO binding, so they follow the head (`template.py`, `partstex.py`, `assemble.py`).
5. **Outputs**: `hybrid.glb` (single skinless primitive, one atlas) for the existing rig and bundle stages, and a
   portable **face asset** for the app's standard model.

Everything here is **local and private**: the stage reads FLAME/Pixel3DMM data (non-commercial licence, see
`tools/twin-lab/head/flame/README.md`) and photos, so inputs and outputs live in the gitignored `user-data/` (the
writers refuse other destinations). MakeHuman assets are CC0. Tests use synthetic data only.

## Run

```powershell
# whole chain from the hybrid stage on (bodyfix output must exist); writes user-data/twin/out/hybrid/twin.glb
tools/twin-lab/bundle/.venv/Scripts/python.exe tools/twin-lab/run_all.py --body hybrid --head flame --from-stage hybrid

# the stage alone (refine environment)
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/hybrid/hybrid.py \
  --bodyfix user-data/twin/out/bodyfix --measurements user-data/twin/measurements.json \
  --out user-data/twin/out/hybrid/hybrid.glb [--hair hair-short] [--brows] [--texture-size 4096] [--no-previews]

# rig and bundle by hand
tools/twin-lab/rig/.venv/Scripts/python.exe tools/twin-lab/rig/rig_scan.py user-data/twin/out/hybrid/hybrid.glb user-data/twin/out/hybrid/rig --fingers keep --smooth 0
tools/twin-lab/bundle/.venv/Scripts/python.exe tools/twin-lab/bundle/write_twin_glb.py --rigged user-data/twin/out/hybrid/rig/rigged.glb --twin user-data/twin/out/hybrid/rig/twin.json --mh2twin user-data/twin/out/hybrid/rig/mh2twin.bin --out user-data/twin/out/hybrid/twin.glb

# tests (synthetic) and lint
tools/twin-lab/refine/.venv/Scripts/python.exe -m pytest tools/twin-lab/hybrid/tests -q -p no:cacheprovider
tools/twin-lab/refine/.venv/Scripts/python.exe -m ruff check tools/twin-lab/hybrid --config tools/twin-lab/refine/ruff.toml
```

`run_all.py --body hybrid` needs `--head flame` (or `auto` with a FLAME fit): the scan stages before it only supply
bodyfix's shape prior; no scan geometry enters the hybrid mesh. The bundle goes to `out/hybrid/twin.glb` (override with
`--bundle-out`), so the scan twin `out/twin.glb` is never overwritten. The stage exits 2 when a numeric acceptance check
fails (`verify_report`: neck girth, seam colour step, flipped triangles, non-manifold body edges).

Inputs: the FLAME fit folder (`head_neutral.obj`, `cameras.json` `dt-flame-head-cameras/1`, `fitted_views/{front,right}.ply`),
`front.jpg` / `right.jpg` with exactly the camera's original size, `FLAME_masks` + `mediapipe_landmark_embedding` from
`user-data/flame/`, the bodyfix GLB (its `dtBodyfix` extras) and the tape `measurements.json`.

## Method

**Frames.** Metres, +Y up, +Z front, character's left = +X; the grounded MakeHuman A-pose (`model.shape(..., ground=True)`),
exactly the frame the rig and bundle stages expect.

**Template and weld.** The app's render mesh duplicates a vertex per UV seam; all geometry work happens on the
position-welded copy and is expanded back (`template.py`). The head region is every welded vertex above the neck
measure loop (`anchor_y` = highest loop vertex + 0.5 mm); everything at or below is anchored at displacement 0.
The head UV island, the torso island, the eye-socket cavities and the mouth cavity are found as UV islands.

**Landmark similarity.** FLAME's MediaPipe embedding (105 landmarks) and the app's `face-map.json` (MediaPipe canonical
landmark -> MakeHuman head triangle + barycentrics) give a trimmed similarity FLAME -> template (scale band 0.85-1.2; the
template keeps the body's own head size, FLAME's absolute scale is only used through the landmark layout).

**Non-rigid registration** (`register.py`). One sparse linear solve per iteration for one displacement per free vertex:
point-to-plane + weak point-to-point terms to the closest FLAME surface point (eyeballs excluded, normals must agree,
distance gate 30 -> 12 mm), landmark point constraints, and a membrane (graph Laplacian) smoothness term on the
displacement field whose stiffness is annealed 400 -> 6 over 16 iterations (it is relative to the edge density: the
template's ~4 mm edges). The template keeps its own fine detail and only the smooth difference to FLAME is applied.
Region weights come from the FLAME masks through the nearest FLAME vertex: face, ears (own patch target, 6x stiffer
edges so they stay ear-shaped), scalp (the skull follows FLAME's skull) at full weight, FLAME's neck at 0.2, cavities
(not in the head UV island) at 0, all faded in over 12 mm above the neck loop. Sub-millimetre slivers at lip and lid
corners that flip are repaired (`repair_foldovers`). Max displacement, residuals per region, landmark residuals,
triangle quality (flips, area and edge-stretch ratios) and the neck girth are in the report.

**Displacement.** `face-offsets` = deformed minus solved body per render vertex (seam copies equal), i.e. a morph target on
top of the solved body.

**Texture** (`facetex.py`). Every head vertex has a closest FLAME point (triangle + barycentrics, signed offset). Carrying
it into each fitted view gives the template head "posed" like FLAME in that photo, so the FLAME baker runs on the
template's own triangles and UV texels (`flamehead.texture.bake_texels`, refactored out of `bake`). Texels whose surface
is more than 3-7 mm from FLAME get no photo weight. Low-confidence texels fade (Lab) to the flat skin tone; the baked
skin is shifted so its mean Lab equals the photo skin tone (clamped to 3); within 18 mm of the head/torso UV seam (the
neck) it fades to the body tone. `vertex_photo_check` verifies the whole UV chain independently (photo colour at a
vertex's projection vs the atlas at its UV, against shuffled colours).

**Body skin** (`skin.py`). The MakeHuman distribution used here ships no skin texture, so the albedo is the photo skin
tone (Lab) plus fine 3-D grain, painted from each texel's own position. Painted underwear: slate boxer shorts from
crotch - 11 cm to pelvis + 9 cm, gated by the pelvis/thigh/spine_01 skin weights (arms and hands are never covered),
with a darker waistband and hem. Charts are padded by 12 px.

**Parts** (`template.py`, `partstex.py`, `photocolours.py`). `Part.bind` is `bindGarment` of avatar-core (barycentric body
positions + axis-scaled MHCLO offset) evaluated on the deformed body. Eyes are translated per eye so the sphere centre
matches FLAME's eyeball in x/y and the front pole in z (clamped to 4 mm); the eye-socket cavity faces are removed
(`eyes-default.delete.bin`). The twin is drawn by the app with one opaque material, so alpha-masked cards cannot rely on
cut-outs: cards with (almost) no strands are removed (`PRUNE_COVERAGE`), the rest are composited over a base colour
(dark scalp for hair, skin for brows, dark lid skin for lashes) into an RGB strip, and card parts get a reversed copy
0.15 mm behind them (never welded) so they show with a front-face-only material. Hair colour is the median of the dark
pixels under FLAME's scalp (both photos), the iris colour the median of the iris ring around each eyeball's front pole in
the front photo (the app's `recolorIris` is ported). The brow cards are **off by default** (`--brows`): the photographed
brows are already baked into the face texture and a second set would double them. Hair choice: `--hair` (any hair part of
`parts/index.json`; `hair-short` is the default: short, voluminous on top).

**Atlas.** `S x (S + S/4)` (default 4096 x 5120, JPEG q95 4:4:4, no alpha): the top `S` rows are the MakeHuman fixed UV
layout unchanged (head island at its fixed place), the strip below holds the part tiles (hair, eyes, lashes, brows).

**Rig integration.** The GLB carries `asset.extras.dtHybrid` (`version 1`, frame `MakeHuman-grounded-A-pose`, manifest
SHA-256, `cutHeightM`), the solution `dtBodyfix`, `dtScanHandsRemoved: false` and `dtHasMakeHumanHands: true`. In the rig
stage `bodyfix_solution.hybrid_fit` verifies that every vertex below `cutHeightM` is the solved body (2e-5 m) and reuses
the exact shape with the identity pose; weights come from the template's own skin weights (`--fingers keep --smooth 0`).
`twin.json` reports the tape measurements.

## Outputs (`user-data/twin/out/hybrid/`)

| file                              | content                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `hybrid.glb`                      | single skinless primitive + atlas; input of the rig stage                                     |
| `rig/`, `twin.glb`                | rig stage output and the app bundle (`run_all --body hybrid`)                                 |
| `hybrid_report.json`              | the metrics: registration residuals (mm), landmark residuals, displacement, neck girth, texture fill, seam and skin Lab, vertex/photo check, hair/eye/lash placement, mesh validity, hand flags |
| `face_asset/`                     | the portable face asset (below)                                                               |
| `previews/`                       | full body front/side/back/3-4, bare and with hair, plus head close-ups (textured and clay); **show the owner only to the owner** |

## Face asset (`face_asset/`, schema `dt-face-asset/1`)

For the app's standard MakeHuman model (nothing here is wired into `apps/web` yet).

- `face-asset.json` - metadata: `template` (manifest SHA-256, `renderVertexCount`, UV layout), `solvedBody` (macros,
  modifiers, tape targets the offsets were computed on), `headOffsets`, `texture`, `skin` (Lab + hex), `parts` (hair id and
  colour, brows enabled/colour, eyes id + iris colour + per-eye translation in mm, lashes id), `metrics`, `license`.
- `face-offsets.bin` - the head deformation as a morph target: sparse entries of **`morphs.bin`** layout, little endian,
  16 bytes each: `uint32` vertex index (render vertex of `base.glb`, `< renderVertexCount`, strictly ascending),
  `float32 dx, dy, dz` in metres. Seam copies carry equal offsets. Weight 1 on top of the solved body.
- `face-texture.png` - the head UV island's rectangle of the fixed MakeHuman UV (sRGB, padded outside the island).
  `texture.pixelWindow` is `{x, y, width, height}` in the bake's atlas (`atlasSizeAtBake`) and `texture.uvWindow`
  `{u0, v0, u1, v1}` the same rectangle in the square MakeHuman UV space (glTF `v` down), so any resolution can paste it.

## What the app would need to consume it

1. Load `face-offsets.bin` through the existing morph-target parser as one extra target (`face-fit`, weight 1) applied
   after the macro/modifier targets; the offsets are exact on top of `solvedBody` and act as a head shape delta on other
   bodies (cap or fade them if the body sliders move far from the solved body). Garments need no change (no garment
   covers the head).
2. Paste `face-texture.png` into the standard skin texture at `uvWindow` (a skin composite step like `skinComposite.ts`), and
   fill the rest of the body skin with `skin.srgbHex`.
3. Mount the standard parts with the listed ids and colours (`partsRuntime`); brows only if enabled, eyes with the iris
   colour (`irisRecolor`) and the per-eye translation. Parts are bound to the deformed body by the existing binding, so
   hair, lashes and eyes follow the offsets.
4. For the twin bundle path: `twinMaterial` keeps only the texture (no alpha) - this stage therefore bakes cards opaque;
   `twinHands` always replaces the twin's hands with MakeHuman hands - it could skip that when `dtHasMakeHumanHands` is
   set, because these hands already are the app's hands.

## Known weaknesses

- The FLAME fit is smooth and statistical: the deformation is small (a few mm on average), ears/skull shape beyond what
  FLAME knows (back of the head, ear detail) are generic. Landmark inliers disagree by ~4 mm between the FLAME embedding
  and the face-map, which bounds feature alignment (lips, lids) rather than the surface residual (~1 mm on the face).
- The seam of FLAME's face at the jaw is tied to the neck loop: the chin underside fades in over 12 mm, so the lower jaw
  contour is only partly transferred.
- Texture: only front and right photos exist; the left side is mirrored, the back of the head and under the chin are flat
  skin tone (hidden by hair). Illumination is normalised, not delit. Photographed glasses frames and beard stay in the
  texture (the separate glasses accessory is not wired here). The hair card composite is opaque and its edge is the card
  silhouette (ragged hairline).
- The body skin is procedural (no MakeHuman skin texture is available); the underwear is painted, not geometry.
- The scan-twin `head.glb` is not used. The skull top of the deformed head can differ from the solved height by a few
  millimetres (reported as `bare_head_top_m` vs the tape height), and hair adds ~1.2 cm.
- Registration stiffness is tuned for the MakeHuman head's edge density (~4 mm); synthetic tests use looser values.
