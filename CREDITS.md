# Credits

## Body model

- **MakeHuman assets** (base mesh, targets, rig, clothes) - [makehumancommunity](https://github.com/makehumancommunity),
  licensed **CC0 1.0** (Public Domain Dedication). The MakeHuman application code is GPL-3.0; this project uses
  only the CC0 asset data, not the code. Pinned commit: `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`
  (base mesh, targets).
- **MPFB2** (MakeHuman Plugin for Blender) - `game_engine` rig and skin weights - [makehumancommunity/mpfb2](https://github.com/makehumancommunity/mpfb2),
  data licensed **CC0 1.0** (the plugin code is GPL-3.0 and is not used). Pinned commit:
  `3edf9df0551765be43563d047888cf7877eb89b4`.

## Libraries

- **MediaPipe** (`@mediapipe/tasks-vision`, Face Landmarker) - Apache-2.0, Google.
- **MediaPipe canonical face model** (`canonical_face_model.obj`: landmark topology and triangulation) and the face
  mesh connection lists (`face_mesh_connections.py`: face oval, eyes, eyebrows, lips) -
  [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe), Apache-2.0, Copyright The MediaPipe Authors.
  Pinned commit: `9519bb59bf55fc6a79ed5b9f283d72e6cdfb6678`. Used by the asset pipeline to generate
  `face-map.json` (triangle index triples and index lists only, no vertex positions or UVs are redistributed).
- three.js, React, @react-three/fiber, drei, zustand, i18next, comlink, idb-keyval, Vite, Tailwind CSS - MIT.

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
