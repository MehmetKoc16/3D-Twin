# Parametric glasses

Local, procedural geometry only; no photos, textures, downloaded assets or model weights.
Use the existing bundle environment (numpy, pygltflib and pytest already installed):

```powershell
tools/twin-lab/bundle/.venv/Scripts/python.exe tools/twin-lab/head/glasses/make_glasses.py --params user-data/twin/head/glasses.json --twin user-data/twin/out/twin.glb --out user-data/twin/out/glasses.glb
tools/twin-lab/bundle/.venv/Scripts/python.exe tools/twin-lab/bundle/write_twin_glb.py --rigged user-data/twin/out/rig/rigged.glb --twin user-data/twin/out/rig/twin.json --mh2twin user-data/twin/out/rig/mh2twin.bin --glasses user-data/twin/out/glasses.glb --out user-data/twin/out/twin.glb --shape <source> --license <license>
tools/twin-lab/bundle/.venv/Scripts/python.exe -m pytest tools/twin-lab/head/glasses/tests tools/twin-lab/bundle/tests -p no:cacheprovider
```

Do not overwrite the input twin while generating glasses. Personal output is restricted
to `user-data/`. A missing params file uses defaults. No personal artifacts are logged.

All dimensions are metres. Canonical JSON keys/defaults:

```json
{
  "lensShape": "round",
  "outerRadius": 0.024,
  "bridgeWidth": 0.018,
  "frameWidth": 0.135,
  "thickness": 0.0012,
  "colour": "#b9b4ad",
  "templeLength": 0.14,
  "verticalOffset": 0.0,
  "clearLenses": false
}
```

Snake-case equivalents are accepted, plus `lensRadius`, `wireThickness`, and `color`.
`outerRadius` is the rim tube centreline radius; thickness is the wire diameter.
Bridge width is the gap between inner rim centreline points; frame width is the hinge-to-hinge
width. Only round lenses are supported. Geometry has 64 segments per rim, eight radial
wire segments, capped arms, separate hinges and tiny flattened oval nose pads. Optional lens discs use
alpha 0.035. Hex colour is sRGB, converted to linear glTF material factors; metalness 1,
roughness 0.3. Pads use a separate satin silver material (metalness 0.15, roughness 0.32) to remain light without an
environment map; their full dimensions are approximately 2.1 x 3.3 x 0.7 mm. Both primitives share the rigid head skin.

Placement reads the rest head node (including its parent transforms) and converts scan
positions into its local frame. It searches `refine/refine_report.json` beside the twin,
or one directory up for a rig-stage input, plus `refine/landmarks3d.json`. An explicit
`--refine-report` overrides discovery. Recognized report blocks are `landmarks3d` or
`faceLandmarksM`, with `leftEye`, `rightEye`, optional `noseBridge`: 3-element XYZ arrays
in world metres, or `{ "positionM": [x,y,z] }`. `coordinateSpace: "head-local"` inside
the block opts into head-local metres. Snake-case eye/nose names are accepted. Pixel/2D
landmarks are ignored. The eye midpoint gives X/Y; the nose bridge, when provided, adjusts
X and forward depth. A 4 mm forward clearance avoids resting the rims on the face.

Without usable landmarks, dominant head-weighted vertices estimate eye height at 60%
of the robust 2nd–98th percentile head height. Central face points near that height give
the 95th percentile forward depth, plus 4 mm. A bone-relative region is a fallback for
blended neck weights. This is a heuristic, especially with hair or a coarse scan; adjust
vertical offset and inspect the result locally. It does not detect pupils or ears.

Output is one mesh, all vertices weighted `[1,0,0,0]` to a standalone identity `head` bone.
Positions are already in the twin head's local coordinates. The bundle embeds the complete
GLB in an aligned BIN buffer view and adds:

```text
asset.extras.dtTwin.accessories = [
  { id: 'glasses', bone: 'head', mesh: { bufferView: <index> }, params: <resolved params> }
]
```

The body mesh, its index space, skeleton, mapping and textures remain unchanged. Web code
validates the embedded rigid skin, then attaches ordinary meshes to the shared head bone;
the standalone accessory skeleton is discarded. This supports rest rebuilding and poses.
The panel toggle defaults on and persists locally under `dt:twin:accessories`.
No extra pipeline dependency is needed. Tests use an analytic synthetic head only.

Lead visual checks: rim centring on both eyes, nose-pad clearance, hinge width, side-view
temple clearance and downturned ends behind the ears; pose the head and toggle glasses,
then reload the browser and switch between standard/twin modes. The estimated temples
follow the requested length; no ear landmark fitting is claimed.

## Hybrid glasses from the private Hunyuan bust

`run_all.py --body hybrid` adds a `glasses` stage after rigging when
`<input-dir>/hy3d/hy3d.glb` exists. `--glasses-hy3d PATH` selects a different bust;
`--no-glasses` disables it. The stage uses the **refine venv**, with no extra
packages: `accessory_glb.py` provides the numpy-only GLB/accessory helpers without
importing the bundle CLI's pygltflib dependency. The bundle still checks the final
accessory with its existing `validate_glasses` before embedding it via `--glasses`.

```powershell
tools/twin-lab/refine/.venv/Scripts/python.exe tools/twin-lab/hybrid/hybridbody/glasses_hy3d.py --twin user-data/twin/out/hybrid/rig/rigged.glb --hybrid-dir user-data/twin/out/hybrid --bust user-data/twin/hy3d/hy3d.glb --out user-data/twin/out/hybrid/glasses.glb
tools/twin-lab/bundle/.venv/Scripts/python.exe tools/twin-lab/run_all.py --body hybrid --from-stage glasses --bundle-out user-data/twin/out/hybrid/twin-glasses.glb
```

The fitter reuses `hybridbody/hy3d.py` read-only to normalise/align the bust. It
segments rim/bridge/arm candidates by front geometry, normals and colour,
excludes blue headphones and dark hair, and measures two bounded rim outlines.
Generated skin-filled lens plates and swollen rims are unsuitable for a clean
accessory. The output therefore uses the existing generator's capped tubes and
nose pads, rebuilding symmetric ellipse rims rather than decimating noisy face
fragments. Visible source arm spans are reported as incomplete when headphones
obscure them. Final arm length and splay come from the twin's ear tops.

The saved hybrid body solution and face-offsets reconstruct the exact template
head. The FLAME similarity supplies eyeball means and nose-bridge landmark 168;
the deformed template nearest the FLAME ear masks supplies the ear tops. The
fit centres lenses on the eyes in front projection, then verifies frame vertices,
triangle centres and all edge midpoints against the skin (at least 1 mm). Rims
are measured on the bust; their final horizontal gap follows the twin's eye
spacing. The bridge plane and pads clear the nose. Arms curve over the ear tops
and down behind them. Each temple samples lateral intersections with the exact head and optional `shell/1` hair.
The hinge runs straight back to the lateral silhouette, then the shaft follows the side with a target 2 mm wire-surface
clearance. The ear hook stays above the ear until behind it. Dense centreline samples (with wire radius subtracted),
wire vertices, triangle centres and edge midpoints check skin and hair clearance. Collision corrections are local;
no shared lateral offset can push the entire opposite arm outward. `placement.temple_clearance` records minimum
skin/hair clearance, supported-run percentiles and maximum, and the free hinge connection length separately.
Tiny satin silver pads use their own standard PBR material. All accessory positions are transformed to rest-head-local
metres and rigidly weighted to an identity `head` bone. The frame uses ordinary
metallic/roughness PBR; clear lenses are omitted. No transmission shader, texture
or external decoder is required. Python previews approximate PBR highlights.

Texture cleanup separately fits the **baked head texture's** original outlines,
since Hunyuan changes their position and shape. `deglass_tex.py` maps those curves
through the same deformed-template barycentric head UV used for baking. It
detects dark/bright thin lines within a narrow corridor, dilates by two texels
and uses OpenCV Navier-Stokes inpainting (five-texel radius). Above the eyes only
bright neutral metal highlights are removed, preserving brow texture; eye
interiors, unmapped UVs and back-of-head texels are protected. Alpha and all
pixels outside the mask are unchanged. This removes frame lines and pads;
photographed lens lighting/tint and anatomy hidden by a frame cannot be recovered
from these strips.

Outputs stay under `user-data/`: `glasses.glb`, `glasses_rigged.glb` (a separate
copy with only the body base-colour PNG reference replaced), `glasses_report.json`,
`face_asset/face-texture-deglassed.png` and `previews_glasses/` with accessory
front/side/top, head front/three-quarter/side, fitted-outline diagnostics and the
UV mask. The launcher bundles the cleaned copy and accessory together. It tracks
the bust, fit, template assets, head offsets, report and implementation for cache
invalidation. Raw hybrid and rig GLBs are retained, allowing the hair stage to
work independently. No source photos are loaded by the glasses stage.

Optional integration hook for the lead in `pipeline.py`, after `seam_blend` and
before `vertex_photo_check`/scalp tint. Supply the private precomputed
`glasses_report` from `glasses_report.json`; the normal launcher's post-rig
cleanup already handles a first build without that report:

```python
from .deglass_tex import projection_from_report, remove_glasses_frames
projection = projection_from_report(face, glasses_report)
texture, deglass_report, _ = remove_glasses_frames(
    texture, projection, method="ns", radius=5, return_report=True
)
```

Record `deglass_report` in the hybrid report and use the cleaned `texture` for
both the body atlas and face asset. Do not run both cleanup passes on the same
texture. The implementation is not wired into `pipeline.py` here because that
file belongs to the concurrent hair job. Tests use analytic/synthetic geometry
and skin gradients only, including skin-filled lens rejection, symmetry,
watertight winding, rigid skin validation, UV/back-face protection and inpainting.
