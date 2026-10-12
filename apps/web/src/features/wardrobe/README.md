Wardrobe: try-on of store items on the avatar (garment templates, size charts, fit report, heatmap). Owner: Wave 4c.

- `WardrobePanel` / `StoreItemForm` / `FitReportView`: UI (feature-local i18n in `locales/`).
- `chartModel`, `colorExtract`, `fitAnalysis`: pure, unit-tested logic (chart editor, k-means colour, fit verdicts).
- `wardrobeRig` + `garmentInstance` + `templateLoader`: R3F-side mounting, created by `features/avatar/Avatar.tsx`.
- `garmentGrading`, `garmentCollide`, `bodyHide`: pure geometry helpers (grade rings and length specs, layering, body hiding). Limb sections and longitudinal grading run in avatar-core.
- State: `src/store/wardrobeStore.ts` (IndexedDB keys `wardrobe:*`). See the "Wardrobe" section of `docs/ARCHITECTURE.md`.

Importer contract (catalogue version 1): `license` accepts `CC0-1.0`, `CC-BY-3.0`, `CC-BY-4.0` and `user`.
CC-BY templates require nonempty `attribution`; loading rejects unknown licences and missing credits. Keep the exact
upstream version in the badge and attribution. Before shipping any CC-BY template, add its title, author, source,
exact licence/link and conversion changes to `CREDITS.md`. Adding contract support does not publish a candidate.

Bound footwear may specify `rigidAttachments: "<id>.rigid.json"`. When omitted, the loader probes that filename for
shoes; only an implicit 404 falls back to ordinary MHCLO skinning. Explicit missing files, failed requests and malformed
sidecars fail loading. Other garments do not probe. The importer at `e9a8776` uses this JSON format:

```json
{
  "version": 1,
  "requiresRuntimeChange": true,
  "soleHeightMm": 35,
  "attachments": [
    { "bone": "foot_l", "vertices": [0, 1], "localPositionsM": [[0, -0.04, 0], [0.02, -0.04, 0]] }
  ],
  "warnings": ["Review the sole boundary."]
}
```

`vertices` addresses the GLB render vertices, including UV duplicates; each matching `localPositionsM` is a neutral
position minus the neutral bone head, in metres (identity rest rotations). Only `foot_l`, `ball_l`, `foot_r`, `ball_r`
are accepted. Duplicate/out-of-range indices, non-finite vectors, mismatched array lengths, absent skeleton bones and
inconsistent neutral frames fail. Review metadata (`requiresRuntimeChange`, `warnings`) is informational.

After every solve/regrade, sole positions are rebuilt as `solvedRestHead + localPositionsM * computeScale(scaleRefs, body)`
after the normal chart/clearance pass. Each authored foot/ball segment gets a single bone influence of weight 1, so
its scaled shape is preserved under poses. Foot and ball segments articulate independently. A 10 mm geodesic band
extends into the upper with a smoothstep fade for rest geometry and skin influences, welded across UV seams. The
remaining upper keeps its barycentric binding and original transferred skin weights. GPU skinning uses the shared
avatar skeleton and updated inverse bind matrices; posed matrices are never used to build rest geometry.
