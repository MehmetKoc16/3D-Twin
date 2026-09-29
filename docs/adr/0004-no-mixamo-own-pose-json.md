# ADR 0004: No Mixamo; own pose JSON on the MakeHuman rig

## Context

Mixamo is free but its assets cannot be redistributed, and availability has been unreliable since 2025. three.js
`SkeletonUtils.retarget` is unreliable across different rigs.

## Decision

Poses are our own JSON files (`{id, label:{tr,en}, bones:{name:[x,y,z,w]}}`) authored directly on the MPFB2/
MakeHuman game-engine rig, applied as bone-local rotations relative to rest pose and blended with slerp. Presets:
T-Pose (default), A-Pose, relaxed, hands on hips, walking step, side profile. Raw Mixamo downloads, if ever
needed for experiments, live in the gitignored `third-party-raw/`.

## Consequences

- No licensing risk and no retargeting step.
- Poses must be authored by hand; no large animation library in the MVP.
