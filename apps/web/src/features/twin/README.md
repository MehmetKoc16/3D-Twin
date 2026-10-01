Realistic twin: one `twin.glb` bundle (or legacy rigged.glb + twin.json + mh2twin.bin from `tools/twin-lab/rig`) shown instead of
the standard mannequin, with the same poses, camera presets and wardrobe. Design and data flow: `docs/ARCHITECTURE.md`,
section "Realistic twin".

| File                                           | Role                                                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `twinDef.ts`                                   | `twin.json` type + strict parser (`TwinFormatError` codes -> `twin.errors.*` messages)                |
| `twinBundle.ts`                                | validates asset.extras.dtTwin and loads the embedded mapping with the glTF parser                  |
| `twinHands.ts`                                 | scan hand mask, wrist overlap and real MakeHuman hands on the shared skeleton                      |
| `twinPushIn.ts`                                | smooth, bounded garment coverage margin and reversible inward displacement                        |
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
