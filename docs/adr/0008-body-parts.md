# ADR 0008: Body parts (eyes, eyebrows, eyelashes, hair) as proxy-bound assets

## Context

The selfie face is baked onto the MakeHuman head texture, but the head has no eyeballs (the eye sockets are empty
cavities) and no brows, lashes or hair. MakeHuman ships all of them as MHCLO proxies (`BodyPartDef` in
`packages/avatar-core/src/garmentContracts.ts`), so they follow the body morphs and the skeleton exactly like garments
(ADR 0007: `bindGarment`, 36-byte binding records, scale refs, optional delete-verts).

Facts found while building this:

- The MakeHuman 1.x eyes, eyebrows, eyelashes and hair are no longer in the `makehuman` repository at the pinned SHA
  (only the eye meshes + one material are). They are published in the same asset-pack mechanism as the garments:
  `makehuman_system_assets_cc0` (eyes, 12 eyebrows, 4 eyelashes, 11 hair styles, CC0, "explicitly released as CC0 in
  september 2020" in every `.mhclo`) and the community packs `hair01_cc0` .. `hair03_ccby`.
- Licence per item, again: in `hair01_cc0`, `o4saken_long01` says `CC BY 4.0` and `elvs_reverse_french_braid_bun` says
  `CC_by` in the `.mhclo` while the pack json says CC0 (inconsistent: not used); eleven items are `AGPL3` (not used). The
  `hair0x_ccby` packs also contain AGPL and "CC-BY" items with an extra condition. Only items whose `.mhclo` line and
  pack json agree on CC0 / CC-BY are accepted (`parts.load_asset`, same rule as `garments.load_asset`).
- **The base mesh eye socket is a closed cavity, not a hole.** The body group is watertight (no open edges). Behind each
  eyelid opening there is a sphere-like cavity of about 15.4 mm radius (centre ~29 mm off the mid-plane) whose faces point
  INTO the head, so the single-sided skin material culls it and the eye looks like an empty dark/pale oval. The eyeball
  proxy sits inside this cavity. The eyelid opening is the ring of 37 vertices where skin quads meet cavity quads.
- **The upstream eye OBJ has two concentric shells per eye.** The outer shell (radius ~15.3 mm, 256 vertices) is UV-mapped
  to a corner of the texture that has alpha 0: it is a fully transparent "cornea" MakeHuman uses for highlights. The
  inner shell (~14.6 mm, 276 vertices, opaque) carries the sclera + iris. The texture holds one island per eye (the second
  island is the mirrored copy of the first).
- **Eyes, eyelashes and some hair styles are bound to helper vertices** of `base.obj` (`helper-l-eye`, `helper-r-eye`,
  `helper-*-eyelashes-*`, `helper-hair`: MakeHuman vertex ids >= 13380), which are not in the runtime index space
  (ADR 0005: body render vertices + joint points). The raw `.target` files do move them with the body.
- Some `.mhclo` files (bob01, long01, braid01, ...) put a header key (`material x.mhmat`) between `verts 0` and the data
  rows; the parser now resumes the data block after an interleaved header key.
- Body-bound hair / brows with negative MHCLO weights (down to -3) are handled by the garment rule (clamp, renormalise,
  fold the change into the offset), exact on the neutral body.

## Decision

- `tools/asset-pipeline/parts.py` (+ `parts_tex.py`, `parts_debug.py`, `mh_morph.py`) builds
  `apps/web/public/assets/parts/` = `index.json` (`{ version: 1, parts: BodyPartDef[], defaults: { eyes, eyebrows,
  eyelashes } }`, no default hair) and per part `<id>.glb`, `<id>.bind.bin`, optional `<id>.delete.bin`. It is part of
  `build.py` and `--check`. Sources are single files of the asset packs, sha256-pinned in `config.PART_ASSETS`,
  range-fetched into `.cache/parts/` (about 25 MB).
- Frame, index space, binding layout, scale refs: exactly ADR 0007 (glb POSITION = neutral body, meters, feet on y = 0;
  bindings address canonical body render vertices).
- **Re-binding helper-bound parts.** `mh_morph.MhMorpher` evaluates the morph model on all 19158 MakeHuman vertices. A
  deterministic set of 48 sample bodies (16 macro corners, every face fit modifier at +-1, 8 random mixes incl. neck /
  torso modifiers) gives the "truth" of a helper-bound MHCLO on each body. Every part vertex (per-vertex mode: lashes,
  hair) or every vertex of one eye (rigid mode: the eyeballs stay spheres) is bound to the body triangle (from the 32 / 96
  nearest) with the smallest squared prediction error over the samples; weights are clamped barycentric coordinates
  (never negative), the offset is exact on the neutral body and scaled by the MHCLO scale refs.
- **Eyes.** MakeHuman "high poly" eyes (the low poly ones are 48 vertices per eye, visibly faceted), inner shells only,
  opaque, single-sided, glossy (roughness 0.25); both eyes use the texture island of the first one, cropped to 512 px
  (JPEG); material `grey`, a neutral / greyish iris on purpose so the runtime can recolour the iris from the selfie.
  `irisUv` is found from the texture (pupil = dark blob, limbus = strongest radial brightening): centre (0.4904, 0.4937),
  radius 0.2041, in glTF texture space (u right, v DOWN, origin top-left; the same space as the glb `TEXCOORD_0`).
  `eyes-default.delete.bin` lists the render copies of the socket-cavity vertices that touch no skin quad (334 MakeHuman
  vertices per eye): with a single-sided skin material the cavity is culled anyway, hiding it makes the eyes independent
  of that and of z-fighting.
- **Hair / brows / lashes textures** (`parts_tex.py`): downscaled with premultiplied alpha (hair 1024 px, brows and lashes
  512 px), colour bled into transparent texels, 8-bit PNG. Tintable parts (hair, eyebrows) are a *neutral* grey (value
  = max(r, g, b)) + alpha map stretched to 0.92; `material.tintable = true` means the runtime sets the material colour
  (glTF `baseColorFactor`) which multiplies this map, so every colour works. The glb factor already holds the colour of
  the original texture as the default tint (linear rgb), so an untouched load looks like the MakeHuman original.
  Lashes keep their (black) texture, `tintable = false`.
- **Alpha strategy: `MASK`**, not `BLEND`. The cards overlap each other, the skin and the eyes; blending would need
  per-triangle sorting and no depth write. MASK stays in the opaque pass with depth write, no ordering problems.
  The cutoff is per texture (`parts_tex.coverage_cutoff`): the value whose covered area equals the soft alpha coverage
  (0.40 - 0.60 here), so strand density is preserved. A fully opaque texture (flat-colour `hair-tousled`) is `OPAQUE`.
  Hair, brows and lashes are `doubleSided`. Enable alpha-to-coverage with MSAA at runtime for soft strand edges.
- Catalogue (labels in `parts.PARTS`): eyes `eyes-default`; eyebrows `eyebrows-default` (001), `eyebrows-thick` (009),
  `eyebrows-thin` (006); eyelashes `eyelashes-default` (01); hair `hair-short` (short02, male), `hair-tousled` (hair_05),
  `hair-bob` (bob02, fringe), `hair-medium` (inverted bob), `hair-long` (long01), `hair-ponytail` (ponytail01).
  Defaults: eyes, eyebrows, eyelashes; no default hair.

## Consequences

- Reproducible, licence-audited assets (all CC0; adding a part = one `PART_ASSETS` + one `PARTS` entry). All parts
  credited in `CREDITS.md` and in `attribution`.
- Eyes stay in the sockets on extreme bodies (tests): over the 48 sample bodies the eyelid-opening ring is
  -1.2 .. +0.7 mm (closest) and +2.4 .. +4.0 mm (farthest) from the eyeball surface (the helper eyeball MakeHuman ships:
  -0.8 .. +0.4 / +2.6 .. +3.9 mm), the front pole is 1.2 .. 6.3 mm in front of the lid plane. The rebound eyeball is
  within 1.5 mm mean / 5 mm max of the helper eyeball (max at `head-scale-horiz` +-1).
- Re-bound lashes are within 0.15 mm mean / 1.2 mm max of the helper truth. Re-bound long hair is a body-bound
  approximation of a helper-bound style: 5 - 22 mm mean deviation on extreme macro corners (male + heavy + muscular,
  short + heavy), 1 - 5 mm on ordinary bodies. `hair-ponytail` is in between. Body-bound styles are exact.
- The eyeball follows the eyelid triangle, not the helper: `eyes/eye-scale` and head shape modifiers keep it in the
  socket but not identical to MakeHuman's own eye motion.
- Every hair style keeps a few root vertices inside the skin (by design of the upstream meshes); tests bound the depth
  (< 8 % of near-surface vertices deeper than 4 mm, < 3 % deeper than 1 cm) on neutral and extreme bodies. The runtime does
  not need to hide the scalp; `hair-tousled` ships the MHCLO `delete_verts` (60 vertices).
- Adds no new dependency (Pillow / numpy are already required).
