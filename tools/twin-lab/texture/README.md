# twin-lab / texture

Puts the user's **real photo pixels** on an untextured body mesh (Hunyuan3D shape from `tools/twin-lab/shape`), so the
3D twin keeps the likeness that generic image-to-3D texturing loses.

```
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python project.py                 # real mesh + user-data/twin/{front,back,left,right}.png
.venv/Scripts/python project.py --stand-in      # end-to-end check on the generic MakeHuman body (not personal)
.venv/Scripts/python -m pytest                  # synthetic tests only, no user images
```

Everything private stays in `user-data/twin/` (gitignored): inputs, `out/texture/textured.glb`, `baseColor.png`,
`report.json`, `previews/`. Nothing is uploaded; there are no ML weights (classical matting + numpy rasterisers).

## Inputs / assumptions

* Mesh: `user-data/twin/out/shape/mesh.glb` (or the newest `*/mesh.glb` below it). Metres, +Y up, character faces +Z,
  left = +X (the shape agent's normalisation; `--up/--front` re-orient other meshes). `meta.json` next to the mesh is
  read when present (`cameras.views.<name>.pxPerMeter/originPx` seeds the alignment); everything is optional because
  the alignment is re-estimated from silhouettes anyway.
* Views: `front.png`, `back.png`, `left.png`, `right.png` in `user-data/twin/` (any subset, front is the reference).
  `left` = camera on the character's left (+X). Left/right are checked against the mesh silhouette and swapped if the
  mirror image fits better. Extra/other files: `--view back=path`. An RGBA file uses its own alpha, otherwise the
  background is removed with a smooth-background colour model + GrabCut (`matting.py`).

## Algorithm (`twintex/`)

1. `meshio` clean (weld, floaters, outward winding), optional decimation (default 150k faces), smooth vertex normals.
2. `unwrap` xatlas atlas (4096², padding), cached in `out/texture/cache/`.
3. `views` + `align` per view: orthographic axis camera; silhouette-IoU fit of scale/translation (Nelder-Mead), then a
   smooth boundary-driven 2-D warp field ("flow") so the photo silhouette matches the mesh silhouette. A
   3x3-min-filtered z-buffer per view gives conservative visibility.
4. `bake` rasterises the UV layout in bands (`raster.py`, vectorised barycentric filler) -> 3-D position + normal per
   texel; per view `q = visible * smoothstep(cos) * silhouette_feather` (reliability) and blend weight
   `w = q * cos^k * prior` (front view x8 on the face). Colours are blended in linear light (bicubic samples);
   per-view RGB gains are matched to the front view on overlapping texels (`estimate.py`).
   Confidence `conf = 1 - prod(1 - q_v)`.
5. `compose`: final = `conf * observed + (1 - conf) * fill`. Fill (on a 4x4-texel cloud, `fill.py`, `prior.py`):
   mirror copy across the sagittal plane, else IDW from the nearest confident texels (3-D distance, normal aware),
   then priors for what nobody sees: the low-pass image of a view without an opposite partner is shone through the
   body onto the surfaces facing away (front only -> back of body in region colours, no ghost detail) and the crown/back
   of the head gets the hair colour taken from the top of the head. Gutters are grown by dilation, the unused atlas
   space is push-pull filled.
6. `export` own GLB writer (sRGB baseColor texture embedded as JPEG q93 or `--png`, glTF v-down UVs), `preview`
   software renderer (front / 3-4 / side / back / head), `debugviz` (confidence, dominant view, alignment overlay).

Optional `--delight 0..1` divides each view by a smooth illumination field (fitted on flat interior gradients only).

## What views improve what

Front only: front-facing texels are real pixels; flanks/arm sides are stretched or filled; the back is a *plausible
guess* from the front's region colours. Back view: real back of body and head. Left/right: flanks, ears, arm sides,
hair sides. The debug maps `debug_confidence_1024.png` / `debug_dominant_view_1024.png` show exactly which texels are
observed.
