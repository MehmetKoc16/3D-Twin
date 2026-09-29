# Asset pipeline spike: MakeHuman CC0 data to browser assets

Model: claude-sonnet-5-5. Sources cloned (shallow) to `tools/asset-pipeline/.cache/{mh,mpfb2,anny}` (about 410 MB, must stay gitignored). Numbers below were measured from the files, and the macro/measure logic was re-implemented in numpy (`.cache/macro.py`, `rul.py`) as a sanity check.

## 1. Base mesh
- `mh/makehuman/data/3dobjs/base.obj` ("hm08"): 19158 `v`, 18486 `vt`, quads only. The MPFB copy (`mpfb2/src/mpfb/data/3dobjs/base.obj`) and Anny's `legacy_default.obj` have the same 19158 vertices.
- Groups are OBJ `g` records. `body` = faces on vertices 0..13379 (13378 quads, contiguous). Everything else is helper geometry: `helper-tongue` 13380-13605, joint cubes 13606+ (1000 verts, 125 `joint-*` groups of 8 verts), eyes, teeth, genital, tights, skirt, hair, eyelashes. MPFB also exposes `HelperGeometry`, `JointCubes`, `Left/Right/Mid` in `data/mesh_metadata/basemesh_vertex_groups.json`.
- UVs: present. Body faces use 14517 unique (v, vt) pairs, so seams split 137 vertices. glTF needs 14517 vertices, so the morph index space must be defined (see the vertex-map risk below).
- Units are decimeters and the axes are already +Y up, +Z front. The body bbox is x ±4.96, y -8.17..8.49, z -1.0..3.2, so the neutral mesh is 166.6 cm tall. Multiply by 0.1 for meters and shift y by +8.1676 (`joint-ground` is at y = -8.183).
- The body has no eyeballs, teeth or tongue. These are separate proxies (`data/eyes`, `teeth`, `eyebrows`), not evaluated here.
- Rest pose is a relaxed A-pose, not a T-pose. The upper arm droops about 50° below horizontal and the forearm about 41°. Document this and author the T-pose preset as a pose JSON.

## 2. Targets and macro weights
- `.target` files are plain text: `#` comment lines, then `vertexIndex dx dy dz` per line, deltas in dm. There is no compression in the MakeHuman repo (1280 files, 6.1 M entries in total). MPFB ships the same targets as `.target.gz`, plus `macrodetails/macro.json` (bucket boundaries) and `targets/target.json` (modifier catalogue). Both are usable data, licensed CC0.
- Some entries index helper vertices (over 30% of the universal targets). These must be filtered, except for the joint cubes we keep.
- Macro file naming:
  - `macrodetails/{african,asian,caucasian}-{male,female}-{baby,child,young,old}.target`, 24 files.
  - `macrodetails/universal-{gender}-{age}-{min|average|max}muscle-{min|average|max}weight.target`, 72 files.
  - `macrodetails/height/{gender}-{age}-{muscle}-{weight}-{minheight|maxheight}.target`, 144 files. There is no "average" file.
  - `macrodetails/proportions/{gender}-{child|young|old}-{muscle}-{weight}-{ideal|uncommon}proportions.target`, 108 files. There is no "regular" file.
- Formula (`apps/human.py`, `apps/humanmodifier.py`, `lib/targets.py`). Every target's weight is the product of the variable values parsed from its file name. Deltas are summed on top of the base mesh.
  - `gender` g in 0..1: male = g, female = 1-g (0.5 is neutral).
  - `age` a in 0..1 with anchors at 1 y (0), 10 y (0.1875), 25 y (0.5), 90 y (1):
    - a < 0.5: baby = max(0, 1-5.333a), young = max(0, (a-0.1875)·3.2), child = max(0, min(1, 5.333a) - young).
    - a ≥ 0.5: old = 2a-1, young = 1-old.
  - `muscle` m, `weight` w and `proportions` p use the same rule. With t the slider value: min = max(0, 1-2t), max = max(0, 2t-1), average = 1-min-max. For height and proportions the "average" value is `1 - max(minVal, maxVal)`, but no target uses it.
  - Race (`african`, `asian`, `caucasian`) sums to 1 with a default of 1/3 each. A race file's weight is race · gender · age.
  - Sliders map to buckets as tents, so minweight = slider 0, averageweight = 0.5, maxweight = 1. The same holds for height (0, 0.5, 1) and muscle. Age is piecewise linear on the anchors above.
- Linear modifiers (`data/modifiers/*.json`): a modifier `X-decr|incr` has value v in [-1, 1] with weight(decr) = -min(v, 0) and weight(incr) = max(v, 0). A `mid` variant adds a centre target with weight 1-|v|.
- Sanity check with the re-implemented macros (neutral/average/young, races 1/3 each): female height slider 0 / 0.5 / 1 gives 122 / 159 / 231 cm, male gives 136 / 173 / 245 cm. Waist at weight slider 0 / 1 is 67 / 82 cm. Circumferences are strongly coupled to the height macro, which the Gauss-Newton solver must handle.
- Measure targets: `targets/measure/measure-*-{decr,incr}.target`, 20 modifiers (40 files). Names: neck-circ, neck-height, upperarm-circ, upperarm-length, lowerarm-length, wrist-circ, frontchest-dist, bust-circ, underbust-circ, waist-circ, napetowaist-dist, waisttohip-dist, shoulder-dist, hips-circ, upperleg-height, thigh-circ, lowerleg-height, calf-circ, knee-circ, ankle-circ. Modifier ids look like `measure/measure-waist-circ-decr|incr`.
- Other relevant targets:
  - `armslegs/upperlegs-height-*` and `armslegs/lowerlegs-height-*`.
  - `armslegs/{l,r}-foot-scale-{depth,horiz,vert}-*`. The `l-` files only touch x > 0 vertices, so a symmetric modifier needs both l and r targets.
  - `torso/torso-scale-{depth,horiz,vert}`, `torso-vshape`, `hip/hip-scale-*`, `neck/neck-scale-*`, `stomach`, `buttocks`.
  - `head/*` and the face directories.
- The shoulder-dist target is symmetric (2784 vertices on each side).

## 3. Measurement vertex lists
- Location: the `Ruler.Measures` dict in `mh/makehuman/plugins/0_modeling_a_measurement.py` (lines ~297-340). This is AGPL source code, but the lists are plain index data. To stay clean, `measures.json` should be regenerated from geometry (plane slices) or cross-checked against the MPFB/Anny lists. Do not copy the Python code.
- Anny (`anny/src/anny/anthropometry.py`) only has waist (the same 46 vertices as MakeHuman), height (bbox), volume and mass. It has no ready loops for the other measures.
- Format: an even-length list, polyline length = sum of consecutive distances (dm x 10 = cm). Some loops are closed (first == last: neck, wrist, waist, hips, thigh, calf, ankle, knee) and some are not (bust, underbust, upperarm), so the builder must close them.
- Samples:
  - `waist-circ = [4121,10760,10757,10777,...,4108,4113,4118,4121]` (46 verts).
  - `hips-circ = [4341,10968,...,4361,4341]` (34).
  - `upperarm-length = [8274, 10037]`, `shoulder-dist = [7478, 8274]`, `upperleg-height = [10970, 11230]`.
- Base-mesh values from these lists (cm): neck 30.6, bust 80.4, waist 75.1, hips 97.4, thigh 52.9, upperarm 24.1, upperarm-length 23.5, lowerarm-length 21.9.
- Gaps for our list:
  - height: bbox of the body vertices.
  - shoulder: MakeHuman's list is one side only. Mirror vertex 8274 to 1602 (mirror index found by exact x-flip, distance 0) gives a biacromial width of 37.3 cm.
  - armLength: 8274 to 10037 (upper arm) plus 10040 to 10548 (lower arm), or a joint-based path.
  - footLength: heel-to-toe z extent of the foot vertices, 24.3 cm on the base mesh.
  - inseam: crotch vertex to y = 0. No vertex is defined by MakeHuman, and the lowest midplane vertex found is the buttock (77.5 cm), which is wrong. This vertex must be picked manually and unit-tested.

## 4. Skeleton
- MPFB2 presets in `mpfb2/src/mpfb/data/rigs/standard/`: `game_engine` (53 bones), `game_engine_with_breast` (55), `mixamo_unity` (64), `default` (163), `default_no_toes` (137), `cmu_mb` (31), `openpose`, `mixamo`. `rigify/` also exists. Each has a matching `weights.<name>.json`.
- Pick `game_engine` (53 bones, Unreal-style names: `pelvis`, `spine_01..03`, `neck_01`, `head`, `clavicle_l`, `upperarm_l`, `lowerarm_l`, `hand_l`, 3-joint fingers `thumb/index/middle/ring/pinky_01..03_l`, `thigh_l`, `calf_l`, `foot_l`, `ball_l`, `Root`). The fingers can be dropped for a smaller rig.
- `rig.game_engine.json` format: a dict with bone name as key.
  ```json
  "upperarm_l": {"head":{"strategy":"CUBE","cube_name":"joint-l-shoulder","default_position":[..]},
                 "tail":{"strategy":"CUBE","cube_name":"joint-l-elbow",...},
                 "parent":"clavicle_l","roll":2.368,"use_connect":false,...}
  ```
  Of 106 head/tail refs, 105 are CUBE and 1 is MEAN (`Root` tail: vertices [19152, 19155]).
- Strategies (`mpfb/entities/rig.py:get_best_location_from_strategy`): CUBE = mean of the 8 vertices of the base-mesh group `joint-*` (helper cube in `base.obj`), VERTEX = `vertex_index`, MEAN = mean of `vertex_indices`, XYZ = one coordinate from each of 3 vertices (Rigify heel). An optional `offset` applies. `default_position` is in Blender axes (z up) and metres, so it is not usable as-is.
- The 53-bone rig uses 69 distinct cubes, i.e. 552 helper vertices that must be kept for joint recomputation. They map naturally to our `JointRef = MEAN(vertices)`.
- `roll` is a Blender bone roll and does not port to glTF. Use identity local rotation (world-aligned bone frames) with translation-only nodes, and compute inverse bind matrices from the joint positions. Our pose JSON is then authored in world axes.
- Weights `weights.game_engine.json`: `{"weights": {bone: [[vertexIndex, w], ...]}}`, 19158 vertices in the MakeHuman index space. There are up to 6 influences per vertex and 608 body vertices have more than 4, so the builder must keep the top 4 and renormalize (glTF `JOINTS_0/WEIGHTS_0`). Weight sums are about 1.0 (min 0.9999).
- MakeHuman 1.x alternative: `mh/makehuman/data/rigs/default.mhskel` (a bones JSON with named joint refs) plus `default_weights.mhw` (163 bones, legacy format). The MPFB JSON is easier to parse.
- Licensing: MakeHuman `LICENSE.md` section C says targets, modifiers and base mesh are CC0. MPFB `LICENSE.md` section C says "Rigs, poses and expressions, JSON data with mesh information" are CC0. Its source code (GPLv3) and MakeHuman's code (AGPL) are NOT usable. Anny's code is Apache-2.0 and its bundled `data/mpfb2/` copy is CC0.

## 5. NAVER Anny
- Version 0.6 (2026-08), Apache-2.0. `pyproject.toml` requires torch >= 2.0, roma, numpy, pyyaml, pillow, requests, safetensors, trimesh and warp-lang. Torch is mandatory (a CPU wheel is roughly 150-250 MB, not installed).
- It bundles an MPFB2 data copy (`src/anny/data/mpfb2`, CC0) and precomputed blend-shape caches (`data/cached/anny.pth` 11 MB, `soma.pth` 6 MB) as torch pickles. It also has SOMA topology (`base_body.obj` 13718 verts, `SOMA_wrap.obj` 18056) and topology variants `notoes*.obj` (12272 down to 369 verts).
- The default `Anny()` model is a re-parameterised model (phenotypes in [0, 1], its own 104-bone "anny" rig, triangulated topology with unattached vertices removed). It is not the raw MakeHuman modifier set, and it does not expose our measure modifiers.
- Anthropometry is limited to height, waist circumference, volume, mass and BMI (see section 3). There is no glTF or three.js export, and no measure loops.
- It would add torch, an unknown vertex mapping and weights for zero benefit compared with parsing the CC0 files directly.

## 6. Size estimates
Vertex counts: body 13380 (14517 after UV split), plus 552 cube vertices = 13932 source vertices (14517 + 552 = 15069 glTF-space).

Young-only MVP (age slider restricted to 25 y), float32 16-byte entries, after helper filtering:

| Group | Files | Entries | Raw | zlib |
|---|---|---|---|---|
| race (young) | 6 | 83.6k | 1.3 MB | 0.6 |
| universal (young) | 16 non-empty | 77.8k | 1.2 | 0.5 |
| height (young) | 36 | 501k | 8.0 | 4.2 |
| measure | 40 | 105k | 1.7 | 0.3 |
| torso, hip, neck, stomach, pelvis, buttocks | 62 | 144k | 2.3 | 0.4 |
| armslegs | 140 | 121k | 1.9 | 0.5 |
| head | 27 | 72k | 1.2 | 0.25 |
| other face | 243 | 59k | 0.9 | 0.25 |

MVP total (choose a subset of armslegs and face): about 15-19 MB raw, 6-8 MB compressed. The height macro alone is over 40% of it. Adding `old` (age slider up to 90 y) adds about 10 MB raw. Dropping proportions (11 MB raw for young/old) is free, since they are inactive at slider 0.5. `base.glb` is about 1 MB (14.5k vertices with normals, UV, joints and weights, 26.8k triangles).

## Recommendation
Parse raw MakeHuman/MPFB2 data. Do not depend on Anny.
- The contract needs the original modifier structure and vertex indices. Anny's re-parameterisation and torch dependency work against that.
- All needed data is CC0 plain text or JSON (`base.obj`, `.target`, MPFB rig and weights JSON, `macro.json`).
- The macro and measure formulas are about 60 lines of numpy and were validated above.

## Targets to export
Macro variables: gender, age (young/old only), muscle, weight, height, race (3, sum to 1). Skip proportions in the MVP.

| Measure | Driver modifier |
|---|---|
| height | macro `Height` (plus `armslegs/upperlegs-height` and `lowerlegs-height` as helpers) |
| neck | `measure/measure-neck-circ-decr\|incr` |
| shoulder | `measure/measure-shoulder-dist-decr\|incr` |
| chest | `measure/measure-bust-circ-decr\|incr` (`underbust-circ` optional) |
| waist | `measure/measure-waist-circ-decr\|incr` |
| hip | `measure/measure-hips-circ-decr\|incr` |
| thigh | `measure/measure-thigh-circ-decr\|incr` |
| upperArm | `measure/measure-upperarm-circ-decr\|incr` |
| armLength | `measure-upperarm-length` and `measure-lowerarm-length` together (composite) |
| inseam | `measure-upperleg-height` and `measure-lowerleg-height` together (composite) |
| footLength | `armslegs/{l,r}-foot-scale-depth-decr\|incr` (paired) |

Also export: `torso-scale-*`, `torso-vshape`, `hip-scale-*`, `stomach-*`, `neck-scale-*`, `head/*` and a small face set. The contract's single `driver` per measure is not enough for armLength, inseam and footLength. The manifest needs a `drivers: [...]` list that share one weight.

## Proposed module layout (`tools/asset-pipeline/`)
`fetch.py` (pinned git SHA download of makehuman + mpfb2), `mh_obj.py` (base.obj parse, groups, UV split map), `targets.py` (parse, filter, sparse pack, macro weights), `rig.py` (game_engine JSON to `rig.json`, weight top-4), `measures.py` (loops, plane-slice regeneration, closing), `gltf_writer.py` (base.glb, no Blender), `build.py` (orchestrator), `tests/` (formula tests against `.cache/macro.py` numbers, measure regression, vertex-map round-trip).
Dependencies: Python 3.12, `numpy` (pin the installed version), `scipy` only if needed (cKDTree for mirrors), `pygltflib` or a hand-written writer.

## Risks and open questions
1. Vertex-index space. Morphs must be indexed in glTF space (with UV-split duplicates) or expanded at runtime with a map, plus the 552 cube vertices. This changes the contract's `vertexIndex` meaning.
2. Rest pose is an A-pose with about 50° arm droop, not a T-pose. Fingers and hands are angled forward. Poses need a T-pose preset.
3. Height macro payload is large (8 MB raw for young), and it is not linear in cm.
4. Some measurements (inseam, footLength, shoulder) have no official vertices. `bust` is not a true chest loop.
5. Eyes, teeth and tongue are absent. Adding them means proxies with their own licenses to verify.
6. Old-age and proportions targets are excluded from the MVP.
