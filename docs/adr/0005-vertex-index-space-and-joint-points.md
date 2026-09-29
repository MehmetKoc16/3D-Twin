# ADR 0005: Combined vertex index space (render vertices + joint points)

## Context

The MakeHuman base mesh has 19158 vertices: 13380 body vertices, helper geometry (tongue, eyes, teeth, tights, ...)
and 125 joint helper cubes of 8 vertices each. Three facts collide:

1. `base.glb` needs UV-seam splitting: the body has 14517 unique (vertex, UV) pairs, i.e. 1137 more vertices than
   the 13380 MakeHuman body vertices. Morph targets and skin weights are defined on the MakeHuman vertices.
2. The skeleton is defined by joint helper cubes (the centroid of a cube is the joint position). Joints must move
   with the morphed body at runtime, but the cubes are not part of the rendered mesh.
3. The runtime (`avatar-core`) wants one flat `Float32Array` of positions, a sparse `morphs.bin`, and joint /
   measure definitions that all speak the same indices, without a translation table at load time.

Alternatives considered:

- Keep MakeHuman indices in `morphs.bin` and expand to glTF vertices at runtime through a vertex map. Adds a
  second index space and a lookup in the hot loop; joint cubes still need a place.
- Use glTF morph targets. Not suitable for hundreds of sparse targets driven by a solver (dense per-target
  buffers, GPU-side blending limits) and does not cover joint positions.
- Put the 552 cube vertices (69 cubes x 8) into the index space and use `JointRef.MEAN` over them. Works, but
  stores 8x more skeleton deltas in every target for no benefit.

## Decision

All vertex indices in `morphs.bin`, `rig.json` and `measures.json` use one combined space:

```
[ render vertices of base.glb (after UV-seam split, body only) ]  ++  [ jointPoints ]
  0 .. renderVertexCount-1                                          renderVertexCount .. vertexCount-1
```

- The manifest carries `renderVertexCount` and `jointPoints: {name, position}[]`;
  `vertexCount = renderVertexCount + jointPoints.length` (14517 + 69 = 14586).
- A MakeHuman body vertex delta is duplicated onto all of its glTF split copies (the pipeline does it once, at
  build time).
- A joint point is the centroid of one MakeHuman/MPFB2 joint helper cube; its delta is the mean of the 8
  cube-vertex deltas. This is exact, since every morph is a linear combination of target deltas. Only the 69 cubes
  used by the `game_engine` rig are kept; the order is alphabetical by cube name.
- Rig joints are `{strategy: 'VERTEX', vert: renderVertexCount + j}`. Bones are world-aligned (identity rest
  rotation, `roll: 0`); the rest pose is MakeHuman's A-pose.
- Measures reference render vertices only (plus `vertexHeight` of a render vertex); `height` and the floor are
  computed over render vertices, never over joint points.
- Units are meters, and everything is translated so that the neutral body (macro defaults) stands on y = 0
  (`provenance.groundOffsetY`).

## Consequences

- One `applyMorphs` loop updates mesh and skeleton; `computeJoints` is a plain lookup. No runtime vertex map.
- `positions` for three.js are the first `renderVertexCount * 3` floats of the solver output; the tail holds the
  joint points and must not be uploaded to the geometry.
- Target payload: joint points add at most 69 entries per target (0.5% of 14517 vertices; the 552-vertex
  alternative would add up to 8x that). Duplicating deltas onto UV-seam copies adds ~8% versus MakeHuman indices
  (1137 extra render vertices), the price of not needing a runtime vertex map.
- `base.glb` is the raw base mesh and its inverse bind matrices are for the raw mesh's joint points. After
  morphing, the runtime must recompute the skeleton from the morphed joint points (bind pose = current pose with no
  rotations) before skinning.
- Anything that needs another helper vertex (garment fitting to hands, face landmarks, ...) must extend the
  joint-point list or add its own index space; do not append raw helper vertices silently.
- Changing the base mesh, the rig preset or the UV split changes every index: bump the pinned SHAs in
  `tools/asset-pipeline/config.py` and regenerate all assets together.
