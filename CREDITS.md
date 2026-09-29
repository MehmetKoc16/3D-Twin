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

CC-BY garment packs (e.g. MakeHuman Shirts 02/03, Pants 02/03, Shoes 02/03) will be listed here with author
attribution and license when they are added to the repository.
