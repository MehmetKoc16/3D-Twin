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
wire segments, capped arms, separate hinges and solid flattened oval nose pads. Optional lens discs use
alpha 0.035. Hex colour is sRGB, converted to linear glTF material factors; metalness 1,
roughness 0.3. Nose pads share the metal material for a single frame primitive.

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
