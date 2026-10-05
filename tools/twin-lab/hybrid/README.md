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
4. **Parts**: MakeHuman CC0 eyes and lashes (and optionally brows) are bound to the _deformed_ body with the app's own
   MHCLO binding, so they follow the head (`template.py`, `partstex.py`, `assemble.py`). The **hair** is a **separate
   `dtHair` node** with its own material. By default (when `user-data/twin/hy3d/hy3d.glb` exists) it is the user's own hair
   cut out of a Hunyuan3D bust and fitted to the head as a solid textured shell (`--hair hy3d`, format `shell/1`, see "Hair
   from the Hunyuan3D bust"); without the bust it is procedural: soft hair cards with a strand data atlas for the MIT three.js
   hair-card shader (`--hair procedural`, format `rcov-groot-bvar/1`, `hairgen.py`, `hairtex.py`, see "Procedural hair");
   the MakeHuman hair parts stay selectable with `--hair hair-short` etc. (legacy: cards merged into the body mesh and atlas).
5. **Outputs**: `hybrid.glb` (the skinless body primitive with its atlas, plus the `dtHair` primitive with its material
   and atlas) for the existing rig and bundle stages, and a portable **face asset** for the app's standard model.

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
  --out user-data/twin/out/hybrid/hybrid.glb [--hair hy3d|procedural|hair-short|hair-tousled|...] [--hair-hex #2a1e18|photo] \
  [--hairline-mm 68] [--hair-top-mm 55] [--hair-side-mm 8] [--hair-param NAME=VALUE ...] [--brows] \
  [--texture-size 4096] [--no-previews]
# --hair defaults to hy3d when user-data/twin/hy3d/hy3d.glb exists, else procedural; for hy3d --hair-param takes any field of
# the segmentation / shell / fit groups (hair_lightness=40, target_triangles=30000, clearance=2, thickness_short=4, ...)

# the procedural hair alone on the GENERIC MakeHuman head (no person, no photo): head views + numbers
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/hybrid/hairdemo.py \
  --out user-data/twin/out/hybrid/previews_generic_hair [--gender 1.0] [--hair-hex #2a1e18] [--hair-param NAME=VALUE]

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
(`eyes-default.delete.bin`). The body is drawn by the app with one material (`MASK` 0.5): the skin texels are opaque, the
eyelash (and optional brow) cards keep the real alpha of their MakeHuman texture as cut-outs and are one-sided. Hair
choice: `--hair hy3d` (default with the local bust), `--hair procedural` (default without it), or any hair part of `parts/index.json` (`hair-short`, `hair-tousled`,
...): those are legacy, their cards are bound like the lashes and merged into the body mesh and atlas (with the
`--hair-hex` colour of `partstex.card_tile`). The iris colour is the median of the iris ring around each eyeball's front
pole in the front photo (the app's `recolorIris` is ported). The brow cards are **off by default** (`--brows`): the
photographed brows are already baked into the face texture and a second set would double them.

**Atlas.** `S x (S + S/4)` (default 4096 x 5120, RGBA PNG, skin opaque, cut-outs in the alpha channel): the top `S` rows
are the MakeHuman fixed UV layout unchanged (head island at its fixed place), the strip below holds the part tiles (eyes,
lashes, brows; the legacy MakeHuman hair tile). The procedural hair is **not** in this atlas: it has its own strand data
atlas (2048 x 1024 PNG) on its own material.

**Rig integration.** The GLB carries `asset.extras.dtHybrid` (`version 1`, frame `MakeHuman-grounded-A-pose`, manifest
SHA-256, `cutHeightM`), the solution `dtBodyfix`, `dtScanHandsRemoved: false` and `dtHasMakeHumanHands: true`. In the rig
stage `bodyfix_solution.hybrid_fit` verifies that every vertex below `cutHeightM` is the solved body (2e-5 m) and reuses
the exact shape with the identity pose; weights come from the template's own skin weights (`--fingers keep --smooth 0`).
`twin.json` reports the tape measurements.

## Procedural hair (`hair.py`, `hairgen.py`, `hairtex.py`, `haircheck.py`)

The fallback procedural hair is a modern short cut built on the deformed head: cropped and tapered sides and back (8 mm, fading to
about 4 mm at the hairline, above the ears and at the nape), more volume on top (4.5 to 5.5 cm) swept up and back from the
forehead, a visible forehead (no fringe) with the hairline about 6.8 cm above the eye centre line, dark brown `#2a1e18`.
Everything is procedural (no asset, no licence concern); nothing needs the person's data except the head surface and the
eye line.

It is written as a **separate skinned node `dtHair`** (the contract with the app): one primitive (`POSITION`, `NORMAL`,
`TEXCOORD_0`, `JOINTS_0`, `WEIGHTS_0`, indices), its own material `dtHair` (`doubleSided`, `alphaMode MASK` 0.5 as the
fallback for viewers without the shader) with `extras.dtHair = {format: "rcov-groot-bvar/1", colorHex, rootHex, tipHex,
cardCount}` and the strand data atlas as `baseColorTexture`, and `asset.extras.dtHairNode = "dtHair"`. The renderer is
the MIT three.js hair-card shader of creategamecharacters (<https://github.com/creategamecharacters/threejs-hair-shader>,
`hair-shader.js`, `applyHairShader(root, {THREE, renderer, atlas, color})`): with MSAA an alpha-to-coverage pass plus a
blended fringe, without MSAA an alpha-tested core plus a blended outer pass; soft coverage, root-to-tip colour and an
anisotropic highlight. The body mesh and its atlas contain no hair cards.

1. **Head frame** (`measure_head`): skull centre, the eye centre line (FLAME's eyeball centres), the ears (found from
   the normal deviation of the head surface against a Taubin-smoothed copy: largest patch per side in the temporal zone,
   one ring dilated) and the nape crease (the height where the back of the skull starts to bulge 6 mm over the neck).
2. **Hair field** (`HairField`, analytic, shared by the cards and the scalp tint):
   - the **hairline** is a smooth curve (PCHIP) in (azimuth around the skull, height): front `hairline_front` above the
     eye line, `temple_recession` higher at the temple corners, a sideburn ending `sideburn_drop` below the ear top,
     `ear_margin` clear of the ear (distance to the detected ear vertices), behind the ear down towards the lobe, then
     the nape `nape_above_crease` above the crease. The density is 0.5 on the hairline and ramps over `hairline_fade`.
   - the **taper line** (`taper_side`, `taper_back`, `taper_band`) separates the long top from the short sides.
   - **length**: top `length_front` / `length_mid` / `length_crown` along the head (with a ramp from `length_front_edge` at
     the front hairline), sides and back `length_side` (8 mm), shrinking to `length_edge` over `fade_band` towards the
     hairline (the fade). **Lift**: `lift_front` .. `lift_crown` of the length stands off the scalp on top, `lift_side`
     on the sides. **Flow**: top combed back (slightly spread sideways), sides combed down and back, back down.
3. **Guides and cards**: Poisson-disc roots (random-priority maximal independent set) in four overlapping layers:
   `undercoat` (dense, short, nearly flat cards that hide the scalp), `body`, `outer` (the longest and narrowest cards,
   defines the silhouette) and `stubble` (tiny cards in the soft hairline band). Each guide is grown over the scalp along
   the flow field; its height above the surface follows an arch-shaped lift profile (`1 - (1-u)^1.6`, then the tip
   settles back by the layer's `droop`), gated by the surface orientation (hair stands off surfaces that face up or
   forward and lies close on the lateral flanks of the vault, so tips do not poke out of the silhouette as spikes).
   Neighbouring guides share smooth direction noise (clumps, parting), the per-card part is about 1.4 degrees. The
   ribbons have parallel-transported frames lying in the tangent plane (camera independent), a random roll, 1 to 6
   segments, 2.8 to 6 mm width and a leaf-shaped taper (`width * (1 - taper * s^1.8)`) so tips are pointed instead of
   cut. Vertex normals are the card normal blended 85 percent with the scalp normal (the hair shades like one smooth
   volume). The spacing of all layers is rescaled so the triangle count lands within 12 percent of `triangle_target`
   (52000; the budget is 60000; one-sided cards, the material is double sided).
4. **Clearance**: every card vertex (and the interior of every triangle: centroids and edge midpoints) keeps at least
   `clearance` (2 mm) from the head surface; offenders are pushed out along the line from the closest surface point.
5. **Strand data atlas** (`hairtex.py`, format `rcov-groot-bvar/1`, 2048 x 1024 RGBA PNG, linear data, no colour
   meaning): **R** strand coverage, **G** root to tip position (0 on the root row, 1 on the tip row), **B** per-strand
   variation (centred on 0.5), **A** `min(1, 2.5 R)` (only for plain glTF viewers; the shader ignores it). The root is on
   the top of every slot (glTF `v` small), so a card's UV runs from `v_root` (G = 0) to `v_tip` (G = 1) and U across.
   Slots: 20 long (64 x 1024), 16 mid (64 x 512) and 32 short (64 x 128) strips; a card takes a random slot of the
   class matching its length (mirrored in U half of the time) and samples the whole slot. Every strip has 3 to 5 *locks*
   (bundles of 6 to 8 fine strands that converge towards the tip, wiggle, clump and end at different heights with
   tapered tips), a soft root veil (dense root end, hidden under the next layer, soft root edge), a faint lock body
   and a lateral feather (cards blend into their neighbours, nothing hard at the card edges). The mean R is about 0.8 at
   the root, 0.55 in the middle and tapers to 0 at the tip: the shader multiplies the filtered coverage by 2.5 (MSAA
   density), so a card reads as a solid sheet from a distance while single strands with soft edges show close up. The
   atlas follows the shader's compact format (`alphaChannel 'r'`, G root, B seed): `DEFAULTS` of `hair-shader.js` are
   `density 2.5`, `innerThreshold 0.5` on the raw R, `seedVariation 0.36`, `rootMode 'mono'|'multi'` with `rootColor`.
   No colour enters the atlas. Colours are `colorHex` (`#2a1e18` default, the shader's `color`), `rootHex` (74 percent)
   and `tipHex` (lighter, warmer), all sRGB.
6. **Scalp**: the baked head texture is darkened under the hair with the same density field (`HairField.cover`: full
   under the hair, a faint stubble shadow outside the hairline), so gaps between cards read as hair, not skin.
7. **Skinning**: nothing special is needed. The rig stage transfers weights from the closest body vertices, which for
   cards on the scalp are the head bone (and `neck_01` at the nape), and skins the `dtHair` node with the same skin as
   the body. The stage reports the weights the rig will give (`parts.hair.skin_weights`) and `verify_report` fails if
   fewer than 80 percent of the hair vertices are on the head chain.

All style numbers are millimetres (`HairStyle`, listed with the report under `parts.hair.procedural.style`); the CLI has
`--hairline-mm`, `--hair-top-mm`, `--hair-side-mm`, `--hair-seed` and `--hair-param NAME=VALUE` for any field (for
example `clump_noise`, `card_width`, `strand_noise`). The colour defaults to `#2a1e18` without any flag (`--hair-hex
#rrggbb` overrides it, `--hair-hex photo` uses the photographed colour made a natural dark brown).
`parts.hair.procedural` in the report carries the numbers: `penetration` (vertices inside the head, below the
clearance, min/median distance, triangle samples), `coverage` (share of the visible scalp hidden behind the cards from the
front, left, right, back and top; the cards are soft: it accumulates the transmittance of all card fragments in front of
a scalp sample, with the shader's gained coverage `min(1, 2.5 R)` of the mip-filtered atlas, 0.5 mm per pixel, hidden
when at most 15 percent of the light gets through), `lengths_mm` per region, `hairline`, `frame` (ears, crease),
`layers`, `strip`, `atlas`, `colours`; `parts.hair.skin_weights` has the head-chain weights. `verify_report` additionally
requires zero penetrating vertices, 8k to 60k triangles and at least 80 percent of the scalp hidden from every main view.

`hairdemo.py` runs the same generator on the generic MakeHuman head (default male, flat skin tone, scalp darkened the
same way, CC0 eyes) and writes front, side, back, 3/4, 3/4-back and top views (textured and clay) plus
`generic_hair_report.json`: it is the way to judge the hairstyle without any person's face. **The Python previews only
approximate the shader**: the atlas is mip filtered by the card's texel footprint (8x anisotropic), `min(1, 2.5 R)`
decides per sample with an ordered 3 x 3 dither (like alpha to coverage), the colour is the root to base ramp of `G` with
the shader's `seedVariation` 0.36 on `B` and its normal-based self occlusion, lit with the smooth card normals. There
is no anisotropic highlight and no blended fringe: the real look is the shader in the app.

## Hair from the Hunyuan3D bust (`hy3d.py`, `hairseg.py`, `hairshell.py`, `shellfit.py`, `hairhy3d.py`)

The user generated a textured head-and-shoulders bust of themselves with Tencent Hunyuan3D (web, one mesh, one PBR material
with 4096 px base colour / metallic-roughness textures, no skin) and saved it as `user-data/twin/hy3d/hy3d.glb`.
`--hair hy3d` (the default when that file exists) puts THIS hair on the twin. The bust is a private asset: code reads it from
`user-data/` only and everything derived from it stays there. Steps (`build_hy3d_hair`):

1. **Axes and scale** (`hy3d.py`). The glTF node of the generator carries a +90 degree rotation about X (its raw files are
   Z-up); `read_glb` bakes it, after which the bust is +Y up and faces +Z (verified by a face detector: `find_orientation`
   falls back to the 24 axis rotations when the identity shows no face). The bust is not in metres. A front render goes through
   MediaPipe's face landmarker (the model file the web app ships), the landmark pixels are lifted to 3-D with the depth buffer
   of the same render (two passes: the head, then a render framed on the face), and a trimmed Umeyama similarity maps the
   stable landmarks (nose bridge, forehead, upper oval; not eyes, brows, lips, cheeks: glasses, a smile) onto the same
   landmarks on the deformed twin head (the app's `face-map.json` binding). A damped rigid point-to-plane ICP on the forehead /
   temple skin and the short side hair then refines it (skipped when a fringe hides the forehead). The mesh is position-welded
   on load (the generator splits vertices at every UV seam, which cuts any graph into islands). The report carries the
   scale to metres, the total rotation / translation (`p_twin = scale * R @ p_bust + t`) and the residuals.
2. **Segmentation** (`hairseg.py`: colour AND geometry). Hair texels are dark, nearly neutral and not red; skin, ears and
   cheeks are much lighter or redder: the smoothed CIELAB lightness (a few mesh-graph rings) is thresholded half way (`42`) and
   a smoothed `a* > 4.5` is skin even when dark (the ear canal, sideburn shadows). Geometry bounds it from below: a floor curve
   over the azimuth around the skull (generous in front so eyebrows, glasses and beard are never hair, the sideburn ending at
   the top of the ear, the nape 18 mm above the neck crease) and a 5 mm margin around the twin's ears. Behind
   `colour_azimuth` (125 degrees) colour cannot decide - the generator never saw the back of the head and fills it with a
   flat grey that is as dark as the nape - so only the floor does. The mask is then cleaned on the welded mesh graph: largest
   component, one open and one close (spurs, notches), holes filled, boundary smoothed by diffusing the indicator and
   re-thresholding.
3. **Colour artefacts** (`hairshell.attach_fill`). The generator also paints a light patch on the unseen back of the head and
   fills the rest of it with one featureless grey. Both are flagged (a lightness spike far from the edge; a near-zero local
   variance with near-zero chroma) and replaced by the harmonic extension of the real hair around them (only vertices inside
   the hairline count), a little lighter towards the hairline (a fade) plus a fine texel-space grain (stubble).
4. **Shell** (`hairshell.py`). Direct quadric reduction of the noisy bust produced folds and non-manifold edges in the
   earlier implementation. Keep its radial surface repair first: the cap is resampled through a Lambert equal-area
   projection at twice the requested triangle budget. Each node takes the outer surface in its direction; a leaning brim
   can lose its underside. `fast_simplification` then performs quadric reduction to 24000 triangles, and xatlas packs new
   UV charts with padding. Scalp fitting works on welded geometry; the xatlas vertex mapping keeps all seam copies
   coincident in the final mesh. The surface extends a 10 mm margin beyond the hairline; texture alpha defines the cut.
5. **Bake** (`hairshell.bake`). The textures are looked up on dense samples (0.25 mm) of the high-poly surface through the
   ORIGINAL UVs, so the strand detail of the 4096 px textures survives: **colour** (sRGB 2048 px, RGBA PNG; texels at and beyond
   the hairline take the colour of the nearest sample 2.5 mm inside, never skin), **alpha** (255 inside, a noisy ramp of 2.5 mm
   centred on the hairline: 0.5 at the edge, `MASK` 0.5 then gives an irregular cut-out like a real hairline) and a
   **normal map** (geometric relief, plus the source normal map when available, in the tangent space of the low poly; glTF
   convention +X right, +Y up the image, +Z out; stored as JPEG q92 4:4:4).
6. **Fit** (`shellfit.py`). The bust's skull differs from the twin's (a statistical FLAME fit; the back of both is a guess) by up
   to a centimetre, so the shell is warped: where the hair is short (below the cut's taper line: sides, back, a band along
   the whole hairline and the margin) it is pulled to a thickness above the scalp (2 mm at the hairline, 3.5 mm inside); the
   long hair (top, swept-up front) is not pulled at all, the volume is the person's hair; in between a Laplacian-regularised
   displacement field (membrane on the regular grid, data weights per surface area so the mesh density does not matter, a weak
   anchor) makes the correction decay smoothly. The solve is the one of `register.py` (point-to-plane + weak point term,
   one sparse solve per iteration, closest points on the scalp WITHOUT the ears refreshed). A last pass lifts every vertex
   and every triangle sample (centroids, edge midpoints) at least 1.8 mm outside the WHOLE head surface (ears and face
   included), spreading each push to the neighbours.
7. **Scalp tint** (`hairhy3d.ShellField`). The baked head texture is darkened under the shell with the coverage of the shell
   itself (a head point is covered when the shell is near it along its normal or along the ray from the skull centre, which
   is how the shell was built), and the tint colour is the colour of the shell next to each texel, so the hairline fade
   continues into the scalp.

Output per contract addendum v1.1: the `dtHair` node (the same skin as the body), `extras.dtHair = {format: "shell/1",
colorHex, cardCount: 0}` (the mean colour of the visible texels), a normal PBR material (`baseColorTexture` sRGB with the
fringe in its alpha, `normalTexture`, `MASK` 0.5, `doubleSided`, roughness 0.62) and `asset.extras.dtHairNode`. The rig stage
needed no change (weights come from the closest body vertices, materials / textures / images pass through verbatim; test in
`rig/tests/test_hair_prim.py`), the bundle validates both formats (`bundle/write_twin_glb.py`, tests in
`bundle/tests/test_hair_node.py`).

`parts.hair.shell` in the report carries the numbers: the alignment (`transform`), the segmentation, the colour fill, the shell
(`edge_mm`, `surface_error`, `bake`), the fit (`warp`, `clearance`), `penetration` (vertices and triangle samples inside the
head or closer than the clearance, min / median / p05 / p95 distance), `visible_clearance`, `edge_gap` (the visible edge ring's
distance to the scalp), `coverage` (share of the scalp that lies under the shell and is hidden by it from the front, left,
right, back and top), `hairline` (front hairline above the eye line on the bust and on the fitted shell, next to the
procedural target) and the style. `verify_report` additionally requires zero penetration, 15k to 30k shell triangles, at least
80 percent of the covered scalp hidden from every main view, a front hairline between 4 and 10 cm above the eye line, a median
edge gap of at most 4 mm and at least 1.5 mm of vertex and triangle-sample clearance. Tests (`tests/test_hy3d.py`, `test_hairseg.py`,
`test_hairshell.py`, `test_shellfit.py`, `test_hairhy3d.py`, the `hy3d` cases of `test_pipeline.py`) use a synthetic bust built
from the generic CC0 MakeHuman head (`tests/synth_bust.py`; the landmarks come from a fake detector).

## Outputs (`user-data/twin/out/hybrid/`)

| file                              | content                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `hybrid.glb`                      | skinless body primitive + atlas and the `dtHair` primitive with its own material (strand data atlas, or the shell's colour texture and normal map); input of the rig stage |
| `rig/`, `twin.glb`                | rig stage output and the app bundle (`run_all --body hybrid`)                                 |
| `hybrid_report.json`              | the metrics: registration residuals (mm), landmark residuals, displacement, neck girth, texture fill, seam and skin Lab, vertex/photo check, hair/eye/lash placement, mesh validity, hand flags |
| `face_asset/`                     | the portable face asset (below)                                                               |
| `previews/`                       | full body front/side/back/3-4, bare and with hair, plus head close-ups (textured and clay; with hair also 3/4 back and top); **show the owner only to the owner** |

## Face asset (`face_asset/`, schema `dt-face-asset/1`)

For the app's standard MakeHuman model (nothing here is wired into `apps/web` yet).

- `face-asset.json` - metadata: `template` (manifest SHA-256, `renderVertexCount`, UV layout), `solvedBody` (macros,
  modifiers, tape targets the offsets were computed on), `headOffsets`, `texture`, `skin` (Lab + hex), `parts` (hair id and
  colour; for the procedural hair `id` is `procedural` with `kind`, `style` and a `fallbackId` the app can mount, brows enabled/colour, eyes id + iris colour + per-eye translation in mm, lashes id), `metrics`, `license`.
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
4. For the twin bundle path: the hair is the separate `dtHair` node (see "Procedural hair"); the body material keeps its
   alpha for the eyelash cards; the MakeHuman hair parts (legacy) are baked into the body mesh;
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
  texture (the separate glasses accessory is not wired here). The hair is soft strand-atlas cards for the hair shader
  (mip-mapped coverage with a gain of 2.5, so cards stay solid at a distance); the real look is decided by the web
  app's shader and its renderer settings (MSAA, texture anisotropy), the previews here are an approximation.
- The body skin is procedural (no MakeHuman skin texture is available); the underwear is painted, not geometry.
- The scan-twin `head.glb` is not used. The skull top of the deformed head can differ from the solved height by a few
  millimetres (reported as `bare_head_top_m` vs the tape height), and the procedural hair adds 2 to 3 cm of volume (the body mesh and `twinHeightM` of `twin.json` do not include it).
- hy3d hair: the bust's back of the head is the generator's guess (flat grey and a light patch), replaced here by the
  surrounding hair colour plus a grain, so the back of the cut is plausible, not photographed. The shell is a solid surface:
  no strands move, no wind, no physics, hair-through-hair translucency does not exist; a forward-leaning brim loses its
  underside (the grid is a height field from the middle of the head); where the bust's skull is wider than the twin's the
  short hair is pulled in by up to about a centimetre; a bust with a fringe over the forehead skips the ICP refinement and
  the front hairline then depends on the landmark similarity alone. `run_all` does not list the bust as an input of the hybrid
  stage, so a replaced bust needs `--from-stage hybrid` (or `--force`).
- Procedural hair: the hairline, sideburn and nape shapes are tuned on the generic MakeHuman head; the ear and nape
  detection (`parts.hair.procedural.frame`) are reported so a wrong detection is visible in the numbers. The strands are
  cards, not curves: no wind, no physics, one fixed style (tune it with the CLI parameters).
- Registration stiffness is tuned for the MakeHuman head's edge density (~4 mm); synthetic tests use looser values.
