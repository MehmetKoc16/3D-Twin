# ADR 0007: Garment template assets (MHCLO proxies re-fitted to the neutral body)

## Context

ADR 0002 chose MHCLO proxy bindings. The runtime (`avatar-core/src/garment`) recomputes garment rest positions from
the solved body (`bindGarment`), skins them with the body skeleton, grades them to a size chart and shows a fit
heatmap. The pipeline must therefore ship, per template, the unskinned mesh, the binding in the body's combined
index space, the scale references, the body vertices to hide, and the template's own measurements.

Facts found while building this:

- The `makehuman-assets` GitHub repository is gone. The same items are published as zip packs at
  `https://files2.makehumancommunity.org/asset_packs/<name>/<pack>.zip` (`shirts01_cc0`, `pants02_ccby`, ...).
- A pack name does not decide the licence: `shoes01_cc0` contains items whose `.mhclo` says `AGPL3`. Every item is
  therefore accepted only if its own `.mhclo` licence line and the pack json (`packs/<name>.json`) agree on CC0 or
  CC-BY. The `.mhclo` names `hm08`, the base mesh we use.
- MHCLO offsets are relative to the body the asset was authored on. Upstream OBJs reproduce on the MakeHuman default
  male body (gender 1.0, other macros default) to 2-6 mm mean (t-shirt, sweater, pants, jeans, boots), on a
  female-ish body (gender ~0.1) to 0.9 mm (cloth shoes); the sneaker OBJ was edited after export (~2 cm). Our
  neutral body (gender 0.5) is ~15 mm from the upstream OBJs, so the shipped positions are the MHCLO reconstruction
  on OUR neutral body, not the upstream OBJ vertices.
- Shoes bind to body vertices too (a few items bind to helper "tights"/"skirt" vertices, which are not in our index
  space; those were not chosen). Shoes are therefore ordinary `binding` garments, not `attach`.
- MHCLO weights may extrapolate (negative, down to about -1.3 for shoes); the runtime rejects negative weights.

## Decision

- `tools/asset-pipeline/mhclo.py` re-implements the MHCLO / garment OBJ / MHMAT formats from the plain files (no
  MakeHuman or MPFB code). `garments.py` builds the templates in `apps/web/public/assets/garments/` and is part of
  `build.py` (and `--check`). `fetch.py` reads single files out of the remote zips with HTTP range requests and
  verifies each with a pinned SHA-256 (`config.GARMENT_ASSETS`); it is idempotent and needs no full zip download.
- Templates (licence checked per file): `tshirt` = toigo_basic_tucked_t-shirt (CC0), `sweatshirt` =
  toigo_fisherman_sweater (CC0, a knit sweater stands in for sweatshirts), `pants` = toigo_wool_pants (CC0), `jeans` =
  punkduck_male_classic_jeans (CC BY), `sneakers` = culturalibre_sneakers (CC BY, original by yanix), `shoes` =
  toigo_mj_cloth_shoes (CC0), `boots` = culturalibre_hero_boots_2 (CC0). Credits: `CREDITS.md` and `attribution`.
- Frame: the garment glb POSITION is in the body runtime rest frame (meters, +Y up, +Z front, neutral body feet on
  y = 0 with the same `groundOffsetY` as `base.glb`), evaluated on the neutral body with the runtime formula.
- Index space: garment vertices are the garment OBJ's (v, vt) render vertices (UV-seam split like the body, ordered by
  (v, vt)); one 36-byte record each. Body indices are the canonical (first) render copy of the MakeHuman vertex (all
  copies have identical positions). `deleteVerts` lists ALL render copies of the MHCLO `delete_verts`, sorted.
- `scaleRefs`: `x_scale a b d` -> `[render(a), render(b), d * 0.1]` (meters). The runtime scale of an axis is
  `|body[a] - body[b]|_axis / refM` and multiplies the meter offsets.
- Negative weights are clamped to zero and renormalised; the position change on the neutral body is folded into the
  offset (`off += (w - w') . v_neutral / s`), so the neutral reconstruction is exact and other bodies differ by
  millimeters (the triangles are small).
- Textures: diffuse map only, downscaled (Lanczos) to at most 1024 px, embedded in the glb (JPEG, or PNG when it has
  transparency), sRGB base colour texture, factor white, double sided. `baseColor` is the texture mean.
- `nativeMeasures` (cm) are taken on the neutral body: girths are the convex-hull perimeter of the garment's
  horizontal cross-section at the body's own bust / waist / hip plane heights (torso only: |x| <= body loop half
  width + 3.5 cm, so sleeves are excluded); thigh at crotch - 3 cm on the left leg; bottoms measure the waistband
  (top edge - 1 cm) when the body waist plane is above the garment. `length` = highest point (collar) to hem (tops) or
  top to hem (bottoms); `sleeve` = arc length from the left acromion along the body arm path (acromion -> elbow ->
  wrist, the body `armLength` polyline) to the farthest garment vertex on the arm; `inseam` = crotch height - hem
  height; shoes: `footLength` = outer length (max - min z of the left shoe) - 1.5 cm (sole and toe cap). `defaultEase`
  = garment - neutral body measure, rounded to 0.5 cm, only for measures that exist on the body.

## Consequences

- Assets are reproducible and licence-audited; adding a garment means adding one `GARMENT_ASSETS` + `TEMPLATES` entry.
- The `sleeve` ease of a short-sleeve top is strongly negative against the body `armLength` (18.9 vs 53.2 cm); store
  charts give short-sleeve lengths, so the runtime should not derive a verdict from `sleeve` on short-sleeve kinds.
- Shoe soles sit up to 1.6 cm below the floor on the neutral body (sneakers) because the upstream mesh is not
  reproducible there; the runtime may lift shoes by their minimum height.
- Pack json licence strings are not versioned (`CC-BY`); the contract type `CC-BY-4.0` is used and `attribution`
  states "4.0 assumed" where upstream gives no version.
- Adds Pillow (pinned) for texture downscaling.
