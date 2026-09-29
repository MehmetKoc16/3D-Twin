# ADR 0002: Garments via MHCLO proxy fitting, no cloth simulation

## Context

Garments must fit any body shape and pose in the browser. Cloth simulation is heavy and unreliable. The
MakeHuman format (MHCLO) binds each garment vertex to a body triangle (3 vertices, weights, offset), so garments
follow the body automatically. `makehuman-assets` provides CC0 (Shirts 01, Pants 01, Shoes 01) and CC-BY
(Shirts 02/03, Pants 02/03, Shoes 02, Shoes 03) items. No mature open-source web try-on with size grading exists.

## Decision

Use MHCLO-style vertex-to-triangle bindings computed in the pipeline and evaluated in `avatar-core`. Shoes are
rigid meshes attached to foot bones. Store-bought sizing: template scaled by ease from a size chart, then
compared with body measurements for a tight/normal/loose label and a vertex fit heatmap (built from scratch).

## Consequences

- Garments fit any measurement/pose without simulation; no drape or wrinkles.
- CC-BY items require attribution in `CREDITS.md`.
- No hoodie asset; a CC-BY sweater template stands in for sweatshirts.
