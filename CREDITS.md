# Credits

## Body model

- **MakeHuman assets** (base mesh, targets, rig, clothes) - [makehumancommunity](https://github.com/makehumancommunity),
  licensed **CC0 1.0** (Public Domain Dedication). The MakeHuman application code at the pinned revision is AGPL-3.0; this project uses
  only the CC0 asset data, not the code. Pinned commit: `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`
  (base mesh, targets).
- **MPFB2** (MakeHuman Plugin for Blender) - `game_engine` rig and skin weights - [makehumancommunity/mpfb2](https://github.com/makehumancommunity/mpfb2),
  data licensed **CC0 1.0** (the plugin code is GPL-3.0 and is not used). Pinned commit:
  `3edf9df0551765be43563d047888cf7877eb89b4`.

## Libraries

- **MediaPipe** (`@mediapipe/tasks-vision`, Face Landmarker and its downloaded `face_landmarker.task` model) -
  Apache-2.0, Google. The model stays in an ignored model path; browser/local tools reuse it.
  [Official model documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
- **MediaPipe canonical face model** (`canonical_face_model.obj`: landmark topology and triangulation) and the face
  mesh connection lists (`face_mesh_connections.py`: face oval, eyes, eyebrows, lips) -
  [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe), Apache-2.0, Copyright The MediaPipe Authors.
  Pinned commit: `9519bb59bf55fc6a79ed5b9f283d72e6cdfb6678`. Used by the asset pipeline to generate
  `face-map.json` (triangle index triples and index lists only, no vertex positions or UVs are redistributed).
- three.js, React, @react-three/fiber, drei, zustand, i18next, comlink, idb-keyval, Vite, Tailwind CSS - MIT.
- **threejs-hair-shader** (`hair-shader.js`, vendored in `apps/web/src/vendor/threejs-hair-shader/` with its LICENSE) -
  [creategamecharacters/threejs-hair-shader](https://github.com/creategamecharacters/threejs-hair-shader), MIT,
  Copyright (c) 2026 Sander Mørch-Jensen (creategamecharacters.com). Pinned commit
  `f0d6cf0d4d309c55b9c5d11370e0a04d04ad05d3`. Renders twin strand cards (`rcov-groot-bvar/1`); solid shells (`shell/1`) use the app's physical material.
  The upstream copyright and permission notice are retained in the vendored LICENSE.

## Garments and other assets

Garment templates (`apps/web/public/assets/garments/`) are MakeHuman community clothes (MHCLO proxies from the
[MakeHuman community asset packs](https://files2.makehumancommunity.org/asset_packs/)). The licence of each item was
checked from its own `.mhclo` header and the pack's json (never from the pack name alone; e.g. some items inside the
"cc0" packs are AGPL and were not used). Each item was re-fitted to a parametric body through the MHCLO binding,
its texture was downscaled to at most 1024 px and it was converted to glTF. Pinned per-file SHA-256 hashes are in
`tools/asset-pipeline/config.py` (`GARMENT_ASSETS`); the full credit text per item is in
`apps/web/public/assets/garments/index.json` (`attribution`).

CC-BY (attribution required):

- **"Male classic jeans"** (`punkduck_male_classic_jeans`) by **punkduck**, MakeHuman community assets pack
  `pants02` (<https://www.makehumancommunity.org/node/1655>), licensed **CC BY 4.0**
  (<https://creativecommons.org/licenses/by/4.0/>).
- **"Brown sneakers"** (`culturalibre_sneakers`) by **culturalibre**, original model by **yanix**
  ([Sketchfab](https://sketchfab.com/3d-models/brown-sneakers-e6c51d2e77d945d1a0efbca530fb4b5b)), MakeHuman
  community assets pack `shoes02` (<https://www.makehumancommunity.org/node/2555>), licensed **CC BY** (the upstream
  file states no version; treated as 4.0, <https://creativecommons.org/licenses/by/4.0/>).

CC0 (credited as a courtesy, no attribution required):

- "T-shirt_basic_tucked" (`toigo_basic_tucked_t-shirt`), "Sweater_Fisherman" (`toigo_fisherman_sweater`),
  "Pants_Wool" (`toigo_wool_pants`) and "MJ-Shoes" (`toigo_mj_cloth_shoes`) by **MRT** (MargaretToigo), packs
  `shirts01`, `pants01`, `shoes01`.
- "hero_boots_2" (`culturalibre_hero_boots_2`) by **culturalibre**, pack `shoes01`.

## Body parts (eyes, eyebrows, eyelashes, hair)

The body parts (`apps/web/public/assets/parts/`) are MakeHuman CC0 assets: the **MakeHuman system assets** (pack
`makehuman_system_assets_cc0`, "makehuman_system", explicitly released as CC0 in September 2020) and two hair styles from
the MakeHuman community pack `hair01_cc0`. As for the garments, the licence of each item was checked from its own
`.mhclo` header AND the pack json (never from the pack name alone: several items of `hair01_cc0` are AGPL or CC BY and
were not used). Each item was re-fitted to a parametric body through the MHCLO binding (the eyes and the eyelashes were
re-bound from MakeHuman's helper geometry to body vertices), its texture was downscaled and converted to a grey +
alpha map, and it was converted to glTF. Pinned per-file SHA-256 hashes are in `tools/asset-pipeline/config.py`
(`PART_ASSETS`); the full credit text per item is in `apps/web/public/assets/parts/index.json` (`attribution`).

CC0 (credited as a courtesy, no attribution required):

- "High-poly eyes" (`high-poly` mesh, `grey` eye material `grey_eye.png`), "Eyebrow 001 / 006 / 009" (`eyebrow001`,
  `eyebrow006`, `eyebrow009`), "Eyelashes 01" (`eyelashes01`), "Short hair 02" (`short02`), "Bob 02" (`bob02`),
  "Long hair 01" (`long01`) and "Ponytail 01" (`ponytail01`) by the **MakeHuman project** (`makehuman_system`,
  MakeHuman system assets).
- "Hair 05" (`culturalibre_hair_05`) by **culturalibre**, pack `hair01`.
- "Inverted bob" (`toigo_inverted_bob`, golden blond texture) by **MRT** (MargaretToigo), pack `hair01`.

## Downloaded MakeHuman body skin (local hybrid cache)

- **MakeHuman `young_caucasian_male` skin** / `young_lightskinned_male_diffuse.png`, from the official
  [MakeHuman system assets CC0 pack](https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip),
  **CC0 1.0**, explicitly released in September 2020. The material names the release-time holders as
  **Data Collection AB, Joel Palmius and Jonas Hauquier**.
  [System-pack release/licence statement](https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html).
  `tools/twin-lab/hybrid/hybridbody/skin_source.py` downloads only the material, diffuse image and pack JSON into
  ignored `tools/twin-lab/hybrid/.cache/skin/`; it verifies per-file SHA-256 hashes, the pack item's CC0 declaration and
  the material's explicit release notice. Hybrid adapts the atlas/tone and derives normal detail from albedo high-pass
  and procedural pores; no upstream normal/specular map is declared or downloaded by that material. Source files remain
  in the cache, and private derived atlases/bundles stay under `user-data/`.

## Local lab and optional Colab sources

These are separate from the shipped web assets. This list records explicit sources downloaded by the tools or their
optional notebook setup; it does not claim that every optional model has been downloaded on a contributor's machine.
Third-party code/weights stay in ignored caches or the session VM. Restricted models and derived data are never
committed, as required by [AGENTS.md](AGENTS.md). Personal outputs remain in `user-data/`.

| Source / use                                                    | Licence and attribution                                                                                                                                                  | Acquisition / evidence                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hunyuan3D-2 code and 2mini/2mv shape weights                    | Tencent, **Tencent Hunyuan 3D 2.0 Community License** (custom, including territorial conditions; not a permissive asset licence)                                         | [Upstream](https://github.com/Tencent-Hunyuan/Hunyuan3D-2); local `shape/.cache/Hunyuan3D-2/LICENSE`, ignored `shape/weights/`; setup and restrictions in [twin-lab](tools/twin-lab/README.md#licence-tencent-hunyuan-3d-20-community-license-read-before-using-outputs-beyond-this-lab). The paint model is not used.                        |
| rembg / `u2net_human_seg` background removal                    | Daniel Gatis/rembg, **MIT** code; Xuebin Qin et al./U2Net, **Apache-2.0** model/source                                                                                   | [rembg licence](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt), [U2Net licence](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE); shape's ignored `weights/rembg/`, also used by the shape notebook.                                                                                                                    |
| MediaPipe `selfie_multiclass_256x256` segmentation              | Google, **Apache-2.0**                                                                                                                                                   | [Official model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter); `head/deglass/setup_models.py` pins version 1 and SHA-256 in ignored `.cache/models.json`.                                                                                                                                     |
| Big-LaMa ONNX (`lama_fp32.onnx`)                                | Original Big-LaMa by Roman Suvorov et al.; ONNX export by **Carve**, **Apache-2.0 as declared by the export publisher**                                                  | [Carve model card](https://huggingface.co/Carve/LaMa-ONNX), [original LaMa](https://github.com/advimman/lama); revision `c3c0c9e`, pinned SHA-256 and licence in ignored deglass model manifest. Used offline with ONNX Runtime (**MIT**); not shipped.                                                                                       |
| Pixel3DMM code and UV/normal checkpoints                        | Simon Giebenhain, Tobias Kirschstein, Martin Rünz, Lourdes Agapito and Matthias Nießner, **CC BY-NC 4.0**                                                                | [Pinned licence](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/LICENSE); optional [head-fit notebook](tools/twin-lab/colab/pixel3dmm_README.md). Code revision `fcd1fa973c7715b02a8948dfc679dff53cf85924`; model revision/sources in `colab/pixel3dmm/downloads.py`.                             |
| FLAME model, masks and MediaPipe barycentric landmark embedding | Max Planck Institute for Intelligent Systems / FLAME authors, **version-specific FLAME terms**; FLAME 2020 is a restricted non-commercial research model                 | Obtain after registration/acceptance at the [official FLAME site](https://flame.is.tue.mpg.de/); no permissive blanket claim for model/embedding data. Local masks/embedding are in ignored `user-data/flame/`; models belong in the notebook VM/cache, never the web assets. [Local fit consumer](tools/twin-lab/head/flame/README.md).      |
| MICA model/software and auxiliary FLAME assets                  | Max Planck Institute for Intelligent Systems; Wojciech Zielonka, Timo Bolkart and Justus Thies, **custom non-commercial scientific research/education/artistic licence** | [MICA licence](https://github.com/Zielon/MICA/blob/af22e7a5810d474bc28a1433db533723d6bd2b07/LICENSE); pinned Colab preprocessing dependency, not redistributed.                                                                                                                                                                               |
| InsightFace `antelopev2` / `buffalo_l`                          | InsightFace, **MIT library code; pretrained models restricted to non-commercial research**                                                                               | [Separate code/model terms](https://github.com/deepinsight/insightface/tree/master/python-package#license); official release downloads listed in `colab/pixel3dmm/downloads.py`, session cache only.                                                                                                                                          |
| nvdiffrast v0.3.3                                               | NVIDIA Corporation, **Nvidia Source Code License (1-Way Commercial)**, restricted to non-commercial research/evaluation for recipients                                   | [Versioned licence](https://github.com/NVlabs/nvdiffrast/blob/v0.3.3/LICENSE.txt); pinned at `729261dc64c4241ea36efda84fbf532cc8b425b8` by the Pixel3DMM notebook. This is not MIT/BSD; the TRELLIS notebook avoids its PBR exporter.                                                                                                         |
| facer and PIPNet preprocessing code                             | FacePerceiver/facer and Jiahao Jin et al./PIPNet, **MIT code**                                                                                                           | [Pinned facer licence](https://github.com/FacePerceiver/facer/blob/ddd35c76ff840174b8a5403ad1c1255e37b8782b/LICENSE), [pinned PIPNet licence](https://github.com/jhb86253817/PIPNet/blob/b9eab58816437403a34aa5bc3adeafe5081fd36b/LICENSE). Separate automatically obtained detector/parser/landmark checkpoint terms are an audit gap below. |
| TRELLIS.2 code/4B weights and original TRELLIS sparse decoder   | Microsoft, **MIT**                                                                                                                                                       | [TRELLIS.2 licence](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/LICENSE); pinned source/model revisions in [Colab source inventory](tools/twin-lab/colab/README.md). Optional shape notebook only.                                                                                                   |
| DINOv3 encoder required by TRELLIS.2                            | Meta, **custom DINOv3 licence**, not MIT/Apache/BSD                                                                                                                      | [Official terms](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md); gated access and explicit acknowledgement in the shape notebook. The complete TRELLIS workflow is not certified permissive-only.                                                                                                                           |
| CuMesh / FlexGEMM native geometry code                          | Jeffrey Xiang, **MIT**                                                                                                                                                   | [CuMesh licence](https://github.com/JeffreyXiang/CuMesh/blob/main/LICENSE), [FlexGEMM licence](https://github.com/JeffreyXiang/FlexGEMM/blob/main/LICENSE); pinned notebook builds. O-Voxel is part of the Microsoft source tree.                                                                                                             |

The pipeline also uses NumPy/SciPy (**BSD-3-Clause**), PyTorch (**BSD-style**), trimesh,
fast-simplification, xatlas/its Python binding and pygltflib (**MIT**), OpenCV (**Apache-2.0**, Python packaging **MIT**),
Pillow (**MIT-CMU**), and Hugging Face transformers/diffusers/accelerate (**Apache-2.0**). PyTorch3D is **BSD-3-Clause**.
Notebook validation caches include nbformat and fastjsonschema (**BSD-3-Clause**). Versions and installed distribution
notices remain in each stage's requirements/ignored environment; the deglass README has its
[detailed dependency licence table](tools/twin-lab/head/deglass/README.md#licenses-and-provenance).
The shape adapter replaces Hunyuan postprocessing rather than importing/installing GPL pymeshlab.

## Audit status and remaining provenance gaps

Checked against tracked public assets, asset-pipeline configuration/catalogues, vendored shader notices and the local
non-personal source/model cache manifests. All seven shipped MakeHuman garment templates and the listed body parts
have entries above; both CC-BY garments retain their authors, source links and adaptation descriptions. The sneaker
source says CC BY without a version; the exported catalogue assumes 4.0. That assumption is recorded, not independent
verification of an upstream version.

No Poly Haven, ambientCG, HDRI or other downloaded studio environment was found in tracked assets or the lighting code.
`StudioLighting.tsx` builds its environment from existing three.js/drei Lightformers; it introduces no new asset credit.
Pose JSONs and synthetic fixtures are generated in this project from the already credited permissive body assets.

Remaining gaps: Pixel3DMM preprocessing can automatically fetch facer detector/FaRL parser assets, PIPNet's WFLW
`epoch59.pth` and pretrained backbone/encoder checkpoints. The setup names these dependencies, but does not provide a
complete checkpoint-by-checkpoint licence/attribution manifest. MIT code notices alone do not establish checkpoint
licences; no entries or permissive clearance are invented for unverified weights. The optional session VM was not
inspected in this audit. Preserve the upstream model cards/notices when obtaining those files and complete that
inventory before any redistribution. Public head-fit checkpoint downloads also lack a complete digest lock.

The launcher's short hybrid provenance label omits any bust-source restrictions, and a scan bundle made from an
imported TRELLIS shape still defaults to Hunyuan provenance. Record actual component sources when bundling directly;
these labels do not override AGENTS.md or the upstream terms. Neither gap makes private outputs eligible for commit.
