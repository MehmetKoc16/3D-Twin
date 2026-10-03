Realistic twin: one `twin.glb` bundle (or legacy rigged.glb + twin.json + mh2twin.bin from `tools/twin-lab/rig`) shown instead of
the standard mannequin, with the same poses, camera presets and wardrobe. Design and data flow: `docs/ARCHITECTURE.md`,
section "Realistic twin".

| File                                           | Role                                                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `twinDef.ts`                                   | `twin.json` type + strict parser (`TwinFormatError` codes -> `twin.errors.*` messages)                |
| `twinBundle.ts`                                | validates asset.extras.dtTwin and loads the embedded mapping with the glTF parser                     |
| `twinHands.ts`                                 | scan hand mask, wrist overlap and real MakeHuman hands on the shared skeleton                         |
| `twinPushIn.ts`                                | smooth, bounded garment coverage margin and reversible inward displacement                            |
| `twinSkinTone.ts`                              | cached median forearm texture sampling, Oklab colour distance and neutral fallback                    |
| `twinOpeningRepair.ts`                         | opening band, UV triangle rasterization and reversible clothing-colour texture repair                 |
| `twinMapping.ts`                               | `mh2twin.bin` parsing, hiding twin triangles from the body's remaining triangles (pure)               |
| `twinBinding.ts`                               | bone matching, rest alignment onto the avatar skeleton, skin remap / normalisation (pure)             |
| `twinModel.ts`                                 | parses rigged.glb (three GLTFLoader) into a `TwinModel` re-indexed to the avatar's bone order         |
| `twinRig.ts`                                   | the scan as a SkinnedMesh on the avatar skeleton; hides body + triangles under garments; DEV probe    |
| `twinMode.ts`                                  | switches standard <-> twin (parts, worker fixed shape, twin rig), forwards solves, panel measurements |
| `ModelSwitch`, `TwinPanel`, `TwinMeasurements` | the "Model" switch, the "Realistic twin" tab (pick / status / remove), read-only measurements         |
| `../../store/twinStore.ts`                     | pack + mode state, IndexedDB persistence (`twin:*`)                                                   |

Rules: the files are only ever picked by the user and kept in memory / IndexedDB - never bundled, fetched or uploaded.
Tests and docs use the non-personal stand-in in `apps/web/e2e/fixtures/twin-standin`, never a real person's scan.
Dev probe (`import.meta.env.DEV`): `window.__dtTwin` (skinned vertex positions, bone transforms, hidden triangle count,
alignment) for `e2e/twin.spec.ts`.

### Head accessories

Optional `dtTwin.accessories` contains `{id:'glasses', bone:'head', mesh:{bufferView}, params}`.
The buffer view embeds a complete GLB in head-local metres, rigidly skinned to an identity
standalone head bone. `twinAccessories.ts` validates metadata, embedded resources, geometry,
transforms and rigid weights before mounting ordinary meshes under the shared head bone.
Frame PBR and optional transparent lens materials are preserved. Visibility starts only
after the twin aligns; disposal detaches the accessory and frees its resources.
The panel's Gözlük / Glasses checkbox defaults on when present and persists in localStorage
(`dt:twin:accessories`). Legacy bundles show no checkbox. Standard mode has no accessory.
Synthetic tests cover parsing, malformed/URL payloads, materials, shared-head posing, rest
translation, persistence and disposal. E2E embeds a procedural triangle in the CC0 stand-in
in memory, without altering fixtures. Generator/defaults: `tools/twin-lab/head/glasses/README.md`.

### Skin tone and clothing at garment openings

The bundle's `skinToneHex` is preferred. Legacy twins sample their original baseColor texture at UVs of vertices with
`lowerarm_l` or `lowerarm_r` weight strictly above 0.5. UV transforms and glTF image orientation are respected; duplicated
texels count once. Transparent texels, clipped highlights (Oklab L >= 0.97) and deep shadows (L <= 0.12) are excluded.
Samples farther from the channel median than max(0.06, three times the median Oklab distance) are rejected before taking
the final channel median. Missing or unreadable texture/UVs or no usable samples yields `#c99a7e`, never material white.
The result is weakly cached by the twin's GLB buffer across mode switches. Hands use standard PBR shading, roughness 0.6,
metalness 0; their colour is interpreted as sRGB, like the scan's albedo.

On a garment coverage change, `TwinOpeningRepair` welds coincident garment vertices (10 micrometre quantization) and finds
edges used by exactly one triangle. These are the opening loops: neckline, sleeve ends, waistband and hems. A spatial
grid measures uncovered scan vertices against these segments. The repair is full strength within 15 mm and smoothly fades
to zero at 40 mm. The per-vertex band is interpolated over scan UV triangles to select texels; unrelated atlas islands and
padding are left alone. Oklab colour distance from the skin tone preserves variation below 0.04 and smoothly identifies
clothing from 0.04 to 0.08 (including grey fabric with similar lightness). Selected clothing texels blend toward skin,
keeping alpha intact. A separate `CanvasTexture`
(`flipY=false`, sRGB, original UV transform/sampler) replaces only the twin's map. Every update starts from the original
pixels; taking off all garments restores the identical original texture and disposes the temporary copy. The existing
garment surface/body index/alignment cache skips band selection and painting on repeated solves and pose changes.

This is a colour heuristic, not clothing segmentation or geometry reconstruction. Long sleeves may dominate forearm
samples; skin-coloured fabric can escape detection, and shadows, tattoos or strongly different skin near openings can
be repainted. Missing UVs/readable albedo disable repair. Thick collars retain their shape, clothing beyond 40 mm remains,
and coarse scan triangles can broaden the interpolated transition. Overlapping/mirrored atlas islands share texels and
may receive the same repair; atlases crossing a repeat-wrap seam are unsupported. Nonmanifold/closed garment boundaries,
unwelded positional seams, tears and layering cutouts can respectively hide or create false openings. Synthetic unit
tests cover sampling, seams, band/texel selection, source preservation, texture disposal/restoration and cache reuse;
stand-in e2e covers legacy colour fallback and outfit texture restoration. Real-twin visual QA stays local to the lead.

`tools/twin-lab/rig/app_qa.mjs` prefers `<twinDir>/../twin.glb`, then `<twinDir>/twin.glb`; `--legacy` forces the old three-file
picker path (the mapping remains optional). Screenshots are written only to the specified output directory; `--tag` is
restricted to a filename prefix so it cannot redirect screenshots elsewhere.

## Strand hair (`dtHairNode`, contract `rcov-groot-bvar/1`)

A `twin.glb` may carry the hair as a SEPARATE skinned node: `asset.extras.dtHairNode` names it and the node's material
has `extras.dtHair = {format, colorHex, rootHex?, tipHex?, cardCount?}`; its base-colour texture is the strand data
atlas (R coverage, G root-to-tip, B variation). Bundles without `dtHairNode` behave exactly as before.

- `twinHair.ts` parses and validates the extras (`TwinFormatError('json')`), `twinModel.ts` splits the hair mesh from the
  single body mesh and remaps the hair skin to the avatar's bone order (it must share the body's rest pose).
- `TwinRig` mounts a `TwinHair` (group `twin:hair:group`, SkinnedMesh `twin:hair`) bound to the SAME skeleton and bind
  matrix as the body and translated by the same rest alignment, so every pose drives it. Garment hide-triangles, the
  footprint pass, push-in, opening repair and the hands never touch it; it is shown and disposed with the twin.
- The vendored MIT shader (`src/vendor/threejs-hair-shader`) is applied with the atlas as `NoColorSpace` data. With MSAA
  (`gl.SAMPLES > 0`, the Viewer canvas has `antialias: true`) it draws an opaque alpha-to-coverage pass plus a blended
  fringe; without MSAA an alpha-tested core plus a blended, depth-writing outer pass. Root and tip colours map to the
  shader's two-colour `multi` root mode (`rootHex` at the root, `tipHex ?? colorHex` as `color`).
- Shadow: only the main pass casts, through a `customDepthMaterial` that cuts out on atlas R (coverage), so the hair
  casts a cut-out shadow, never a block.
