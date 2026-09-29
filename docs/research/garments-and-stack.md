# Research: garments and web stack (verified findings)

## Garments

- [makehuman-assets](https://github.com/makehumancommunity/makehuman-assets): **CC0** - Shirts 01, Pants 01,
  Shoes 01. **CC-BY** - Shirts 02/03, Pants 02/03, Shoes 02 (sneakers), Shoes 03 (credit in `CREDITS.md`).
  Format: MHCLO (vertex-to-body-triangle bindings).
- No mature open-source web try-on with size grading / fit heatmap exists; build from scratch.
- **Image-to-3D options:** Meshy (free tier outputs CC-BY), Tripo, Hunyuan3D 2.1 (open, ~29 GB VRAM,
  region-restricted license), TRELLIS.2 (MIT), Rodin (paid). None fit a 6 GB GPU except small local models.
- **2D virtual try-on:** CatVTON (<8 GB VRAM), IDM-VTON (~24 GB).
- **Shoe sizing:** EU size ~ 1.5 x foot_cm + 2 (brand dependent, approximate).

## Web stack

- @react-three/fiber 9 + React 19 + drei 10 are compatible. R3F v10 (WebGPU by default) is still alpha, so we
  stay on WebGL2.
- drei `CameraControls` gives animated focus presets (`setLookAt`, `fitToBox`).
