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

The garments added later (the `mh-*` items and the Khronos sneaker below) were converted with
`tools/asset-pipeline/garment_import.py` instead: licence evidence per item (`.mhclo` header or upstream `LICENSE.md`,
verified 2026-10-11) is recorded in `tools/asset-pipeline/garment_sources.json`. They were aligned to the parametric
body, clearance fitted, bound to body triangles, their PBR maps were baked into at most 2048 px atlases and they were
converted to glTF. No upstream file is redistributed unchanged.

CC-BY (attribution required):

The wardrobe supports exact **CC BY 3.0** and **CC BY 4.0** identifiers. For every new shipped CC-BY garment,
record its title, author, original source, exact licence version/link and conversion changes here and in the
catalogue's `attribution`. Retain upstream notices; do not substitute 4.0 for a 3.0 declaration. Offline review
candidates that are not in `index.json` are not shipped assets and are credited here when the lead adds them.

- **"Male classic jeans"** (`punkduck_male_classic_jeans`) by **punkduck**, MakeHuman community assets pack
  `pants02` (<https://www.makehumancommunity.org/node/1655>), licensed **CC BY 4.0**
  (<https://creativecommons.org/licenses/by/4.0/>).
- **"Brown sneakers"** (`culturalibre_sneakers`) by **culturalibre**, original model by **yanix**
  ([Sketchfab](https://sketchfab.com/3d-models/brown-sneakers-e6c51d2e77d945d1a0efbca530fb4b5b)), MakeHuman
  community assets pack `shoes02` (<https://www.makehumancommunity.org/node/2555>), licensed **CC BY** (the upstream
  file states no version; treated as 4.0, <https://creativecommons.org/licenses/by/4.0/>).
- **"M Trousers 01 (loose jeans)"** (`mh-m-trousers-01`) by **Mindfront (Sweden)**, MakeHuman community assets
  (<http://www.makehumancommunity.org/clothes/m_trousers_01.html>), licensed **CC BY 4.0**
  (<https://creativecommons.org/licenses/by/4.0/>). Modified: converted and re-bound (aligned, clearance fitted, bound to
  the body mesh, textures baked and downscaled to at most 2048 px, baked fold shading flattened in the colour and
  normal textures, hem and lower legs smoothed, converted to glTF).
- **"M Trousers 02 (suit trousers)"** (`mh-m-trousers-02`) by **Mindfront (Sweden)**, MakeHuman community assets
  (<http://www.makehumancommunity.org/clothes/m_trousers_02.html>), licensed **CC BY 4.0**
  (<https://creativecommons.org/licenses/by/4.0/>). Modified: converted and re-bound as above. Baked fold normals
  attenuated, hem and legs smoothed.
- **"Comfortable sneakers"** (`mh-comfortable-sneakers`) by **punkduck**, MakeHuman community assets
  (<http://www.makehumancommunity.org/clothes/comfortable_sneakers.html>), licensed **CC BY 3.0**
  (<https://creativecommons.org/licenses/by/3.0/>; original OBJ and MHCLO headers state "CC BY 3.0"). Modified:
  reconstructed on the neutral body using authored MHCLO bindings, subdivided once, clearance fitted, sole/heel
  attached rigidly to foot and ball bones, texture baked/downscaled to 1024 px with UV-edge padding, converted to
  the wardrobe glTF format. Replaces the retired Shopify/Khronos shoe after visual defect review.
- **"Denim Jacket"** (`mh-denim-jacket`) by **punkduck**, MakeHuman community assets
  (<http://www.makehumancommunity.org/clothes/denim_jacket.html>), licensed **CC BY 3.0**
  (<https://creativecommons.org/licenses/by/3.0/>; the `.mhclo` header states "CC BY 3.0"). Modified: converted and
  re-bound as above. The licence version is 3.0, not 4.0.

CC0 (credited as a courtesy, no attribution required):

- "T-shirt_basic_tucked" (`toigo_basic_tucked_t-shirt`), "Sweater_Fisherman" (`toigo_fisherman_sweater`),
  "Pants_Wool" (`toigo_wool_pants`) and "MJ-Shoes" (`toigo_mj_cloth_shoes`) by **MRT** (MargaretToigo), packs
  `shirts01`, `pants01`, `shoes01`.
- "hero_boots_2" (`culturalibre_hero_boots_2`) by **culturalibre**, pack `shoes01`.
- "Jacket_T01 (actually a female T-shirt)" (`mh-jacket-t01`) by **el_toba / Tobinsky**
  (<http://www.makehumancommunity.org/clothes/jacket_t01.html>) and "Male Polo Shirt" (`mh-male-polo-shirt`) by
  **namuhekam** (<http://www.makehumancommunity.org/clothes/male_polo_shirt.html>); both `.mhclo` headers state CC0.
  Converted and re-bound as described above. The T-shirt texture of `mh-jacket-t01` was
  replaced by a procedurally generated cotton-jersey knit (CC0, no third-party material).

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
| rembg / `u2net_human_seg` background removal                    | Daniel Gatis/rembg, **MIT** code; Xuebin Qin et al./U2Net, **Apache-2.0** upstream source; exact rembg ONNX checkpoint licence **unverified**                                                                                   | [rembg licence](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt), [U2Net licence](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE); shape's ignored `weights/rembg/`, also used by the shape notebook.                                                                                                                    |
| MediaPipe `selfie_multiclass_256x256` segmentation              | Google, **Apache-2.0**                                                                                                                                                   | [Official model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter); `head/deglass/setup_models.py` pins version 1 and SHA-256 in ignored `.cache/models.json`.                                                                                                                                     |
| Big-LaMa ONNX (`lama_fp32.onnx`)                                | Original Big-LaMa by Roman Suvorov et al.; ONNX export by **Carve**, **Apache-2.0 as declared by the export publisher**                                                  | [Carve model card](https://huggingface.co/Carve/LaMa-ONNX), [original LaMa](https://github.com/advimman/lama); revision `c3c0c9e`, pinned SHA-256 and licence in ignored deglass model manifest. Used offline with ONNX Runtime (**MIT**); not shipped.                                                                                       |
| Pixel3DMM code and UV/normal checkpoints                        | Simon Giebenhain, Tobias Kirschstein, Martin Rünz, Lourdes Agapito and Matthias Nießner, **CC BY-NC 4.0**                                                                | [Pinned licence](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/LICENSE); optional [head-fit notebook](tools/twin-lab/colab/pixel3dmm_README.md). Code revision `fcd1fa973c7715b02a8948dfc679dff53cf85924`; model revision/sources in `colab/pixel3dmm/downloads.py`.                             |
| FLAME model, masks and MediaPipe barycentric landmark embedding | Max Planck Institute for Intelligent Systems / FLAME authors, **version-specific FLAME terms**; FLAME 2020 is a restricted non-commercial research model                 | Obtain after registration/acceptance at the [official FLAME site](https://flame.is.tue.mpg.de/); no permissive blanket claim for model/embedding data. Local masks/embedding are in ignored `user-data/flame/`; models belong in the notebook VM/cache, never the web assets. [Local fit consumer](tools/twin-lab/head/flame/README.md).      |
| MICA model/software and auxiliary FLAME assets                  | Max Planck Institute for Intelligent Systems; Wojciech Zielonka, Timo Bolkart and Justus Thies, **custom non-commercial scientific research/education/artistic licence** | [MICA licence](https://github.com/Zielon/MICA/blob/af22e7a5810d474bc28a1433db533723d6bd2b07/LICENSE); pinned Colab preprocessing dependency, not redistributed.                                                                                                                                                                               |
| InsightFace `antelopev2` / `buffalo_l`                          | InsightFace, **MIT library code; pretrained models restricted to non-commercial research**                                                                               | [Separate code/model terms](https://github.com/deepinsight/insightface/tree/master/python-package#license); official release downloads listed in `colab/pixel3dmm/downloads.py`, session cache only.                                                                                                                                          |
| nvdiffrast v0.3.3                                               | NVIDIA Corporation, **Nvidia Source Code License (1-Way Commercial)**, restricted to non-commercial research/evaluation for recipients                                   | [Versioned licence](https://github.com/NVlabs/nvdiffrast/blob/v0.3.3/LICENSE.txt); pinned at `729261dc64c4241ea36efda84fbf532cc8b425b8` by the Pixel3DMM notebook. This is not MIT/BSD; the TRELLIS notebook avoids its PBR exporter.                                                                                                         |
| facer and PIPNet preprocessing code                             | FacePerceiver/facer and Jiahao Jin et al./PIPNet, **MIT code**                                                                                                           | [Pinned facer licence](https://github.com/FacePerceiver/facer/blob/ddd35c76ff840174b8a5403ad1c1255e37b8782b/LICENSE), [pinned PIPNet licence](https://github.com/jhb86253817/PIPNet/blob/b9eab58816437403a34aa5bc3adeafe5081fd36b/LICENSE). Checkpoint-specific findings and unresolved weight terms are inventoried below. |
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
non-personal source/model cache manifests. All shipped garment templates (seven original MakeHuman ones, five more
MakeHuman ones and Comfortable sneakers) and the listed body parts have entries above; every CC-BY garment retains its
author, source link, exact licence version and adaptation description. The original `culturalibre_sneakers` source
says CC BY without a version; the exported catalogue assumes 4.0. That assumption is recorded, not independent
verification of an upstream version.

No Poly Haven, ambientCG, HDRI or other downloaded studio environment was found in tracked assets or the lighting code.
`StudioLighting.tsx` builds its environment from existing three.js/drei Lightformers; it introduces no new asset credit.
Pose JSONs and synthetic fixtures are generated in this project from the already credited permissive body assets.

Preprocessing checkpoint sources were reviewed on **2026-10-11**, including transitive URLs in the pinned
Pixel3DMM/facer/PIPNet source. See the inventory below. An **unverified** entry means no explicit licence for that
particular checkpoint was located in the linked official repository, release or model card; the code licence is
recorded separately and is not a guessed weight grant. No models or session VM caches were downloaded/inspected for
this review. Public head-fit checkpoint downloads still lack a complete digest lock.

The launcher now passes actual scan shape metadata to the bundler. Hybrid hair metadata and the embedded glasses
placement method produce component provenance, including personal non-commercial Hunyuan3D bust restrictions.
These labels do not override AGENTS.md or upstream terms, and private outputs remain ineligible for commit.

## Twin-lab preprocessing checkpoint inventory

Local entry points are `shape/generate.py` (rembg), `head/deglass/setup_models.py`, and the optional
`colab/pixel3dmm/downloads.py` / `setup_runtime.py`. Pixel3DMM's pinned preprocessing wrappers instantiate
RetinaFace MobileNet and FaRL CelebAMask-HQ; PIPNet uses the WFLW ResNet18 configuration. Alternative URLs below
are identified as alternatives, rather than claimed downloads in the default run. Model code is fetched only into
ignored stage caches or the session VM; personal derived outputs stay in `user-data/`.

| Checkpoint / acquisition | Licence finding and official evidence |
| --- | --- |
| `mobilenet0.25_Final.pth`, facer RetinaFace default | **Unverified checkpoint licence.** Downloaded by [pinned facer loader](https://github.com/FacePerceiver/facer/blob/ddd35c76ff840174b8a5403ad1c1255e37b8782b/facer/face_detection/retinaface.py) from [publisher release 0.0.1](https://github.com/elliottzheng/face-detection/releases/tag/0.0.1). Facer's MIT code notice does not establish terms for this separate release asset. |
| `Resnet50_Final.pth`, facer RetinaFace alternative | **Unverified checkpoint licence.** Same [loader](https://github.com/FacePerceiver/facer/blob/ddd35c76ff840174b8a5403ad1c1255e37b8782b/facer/face_detection/retinaface.py) and [publisher release](https://github.com/elliottzheng/face-detection/releases/tag/0.0.1); not selected by the default preprocessing wrapper. |
| `face_parsing.farl.celebm.main_ema_181500_jit.pt`, default parser | **Unverified checkpoint licence.** [Pixel3DMM's replacement URL table](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/preprocessing/replacement_code/farl.py) downloads it from [facer models-v1](https://github.com/FacePerceiver/facer/releases/tag/models-v1). [FaRL](https://github.com/FacePerceiver/FaRL) credits Yinglin Zheng, Hao Yang and collaborators/Microsoft and carries [MIT software terms](https://github.com/FacePerceiver/FaRL/blob/main/LICENSE); an explicit weight-specific grant was not found. |
| `face_parsing.farl.lapa.main_ema_136500_jit191.pt`, alternate parser | **Unverified checkpoint licence.** Same [replacement loader](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/preprocessing/replacement_code/farl.py) / [official release](https://github.com/FacePerceiver/facer/releases/tag/models-v1). Available as `farl/lapa/448`; default uses CelebAMask-HQ. |
| WFLW `epoch59.pth`, PIPNet landmark detector | **Unverified checkpoint licence.** Jiahao Jin et al.; [official trained-model instructions](https://github.com/jhb86253817/PIPNet/blob/b9eab58816437403a34aa5bc3adeafe5081fd36b/README.md) link the authors' Drive folder; the [local source inventory](tools/twin-lab/colab/pixel3dmm/downloads.py) uses file ID `1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C`. [MIT code licence](https://github.com/jhb86253817/PIPNet/blob/b9eab58816437403a34aa5bc3adeafe5081fd36b/LICENSE) does not by itself resolve this checkpoint's terms. |
| `FaceBoxesV2/weights/FaceBoxesV2.pth`, obtained with PIPNet clone | **Unverified checkpoint licence.** [Checkpoint in the pinned source tree](https://github.com/jhb86253817/PIPNet/blob/b9eab58816437403a34aa5bc3adeafe5081fd36b/FaceBoxesV2/weights/FaceBoxesV2.pth), used by [Pixel3DMM cropping](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/preprocessing/pipnet_utils.py). No checkpoint-specific licence located; PIPNet credits its modified FaceBoxes detector separately. |
| `lib/mobilenetv3-large-1cd25616.pth`, obtained with PIPNet clone | **Unverified checkpoint licence.** [Pinned file](https://github.com/jhb86253817/PIPNet/blob/b9eab58816437403a34aa5bc3adeafe5081fd36b/lib/mobilenetv3-large-1cd25616.pth); optional MobileNetV3 backbone, not selected by the WFLW ResNet18 default. |
| `resnet18-f37072fd.pth`, ImageNet backbone initialisation before PIPNet weights | **Unverified checkpoint licence.** [WFLW config enables pretrained ResNet18](https://github.com/jhb86253817/PIPNet/blob/b9eab58816437403a34aa5bc3adeafe5081fd36b/experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py); [torchvision 0.22.1 loader](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/resnet.py) resolves `pretrained=True` to `IMAGENET1K_V1` at `download.pytorch.org/models/resnet18-f37072fd.pth`. [Official weight documentation](https://docs.pytorch.org/vision/0.22/models.html) cautions that weights may have their own dataset-derived terms; BSD library terms are not weight clearance. Other backbone choices need their own review. |
| DINO `vit_small_patch16_224.dino` encoder initialisation (and base alternative) | **Apache-2.0 declared by the official timm model cards:** [small](https://huggingface.co/timm/vit_small_patch16_224.dino), [base](https://huggingface.co/timm/vit_base_patch16_224.dino). Mathilde Caron et al./Facebook Research, converted by timm. [Pinned base config](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/configs/base.yaml) selects small; [DinoWrapper](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/lightning/p3dmm_network.py) calls `timm.create_model(..., pretrained=True)`. A loaded UV/normal checkpoint can supply a different encoder configuration; that exact runtime choice/revision is **unverified** here, not inferred from the code default. |
| `mica.tar` | **MICA custom non-commercial scientific research/education/artistic terms**, explicitly covering the model and software in the [pinned licence](https://github.com/Zielon/MICA/blob/af22e7a5810d474bc28a1433db533723d6bd2b07/LICENSE). Official acquisition is linked by [MICA](https://github.com/Zielon/MICA#pre-trained-models) and the [Pixel3DMM installer](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/src/pixel3dmm/preprocessing/replacement_code/install_mica_download_flame.sh); local Drive ID `1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt`. |
| `antelopev2.zip`, `buffalo_l.zip` (detection, landmarks and recognition ONNX models) | **Non-commercial research only**, explicitly separate from InsightFace's MIT code: [official model terms](https://github.com/deepinsight/insightface/tree/master/python-package#license), [official v0.7 release](https://github.com/deepinsight/insightface/releases/tag/v0.7). All contained pretrained ONNX files inherit that stated model policy. [Local downloader](tools/twin-lab/colab/pixel3dmm/downloads.py) also lists official upstream Drive fallbacks. |
| `uv.ckpt`, `normals.ckpt` | **CC BY-NC 4.0**, Simon Giebenhain et al.; [official Pixel3DMM work licence](https://github.com/SimonGiebenhain/pixel3dmm/blob/fcd1fa973c7715b02a8948dfc679dff53cf85924/LICENSE). [Pinned HF checkpoint source](https://huggingface.co/SimonGiebenhain/Pixel3dmm/tree/98608f33a2d8af6eed387ac354db6af2030a8f44) and official Drive fallbacks are in [downloads.py](tools/twin-lab/colab/pixel3dmm/downloads.py). Separate encoder/preprocessing terms still apply. |
| `u2net_human_seg.onnx`, rembg default for local/Colab shape cutouts | **Unverified exact ONNX checkpoint licence.** [rembg download loader](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2net_human_seg.py) resolves [rembg release v0.0.0](https://github.com/danielgatis/rembg/releases/tag/v0.0.0). Xuebin Qin et al.'s [U2Net source](https://github.com/xuebinqin/U-2-Net) is Apache-2.0 and documents the original human-segmentation weights; rembg code is MIT. No explicit export-specific weight grant was located. `--rembg-model` permits other models; those require a separate inventory before use. |
| `selfie_multiclass_256x256.tflite`, deglass, float32 version 1 | Google MediaPipe, **Apache-2.0**; [official model card, page 1](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf). [setup_models.py](tools/twin-lab/head/deglass/setup_models.py) pins the `storage.googleapis.com/mediapipe-models/image_segmenter/.../float32/1/` URL and SHA-256. |
| `lama_fp32.onnx`, deglass Big-LaMa export, revision `c3c0c9e` | **Apache-2.0 declared by Carve's export model card**, [official export publisher](https://huggingface.co/Carve/LaMa-ONNX), derived from Roman Suvorov et al.'s Big-LaMa. Pinned URL/hash in [setup_models.py](tools/twin-lab/head/deglass/setup_models.py); this is the export publisher's statement, not an inference from code. |

The reused local `face_landmarker.task` is credited under Libraries above; twin-lab does not download it.
Hunyuan/TRELLIS shape weights and the TRELLIS DINOv3 encoder are covered in the preceding source table; they are
not Pixel3DMM preprocessing checkpoints. FLAME archives, masks and embeddings are user-obtained under their
version-specific terms rather than downloaded by these installers. No Sapiens or unrelated training-only encoder
checkpoint is claimed as a download in the default fitting path.
