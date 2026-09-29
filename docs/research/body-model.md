# Research: body model (verified findings)

- **MakeHuman licensing:** code is GPL-3.0 but all assets are CC0 - see
  [makehuman LICENSE.md](https://github.com/makehumancommunity/makehuman) and
  [mpfb2 LICENSE.ASSETS.md](https://github.com/makehumancommunity/mpfb2).
- **MPFB2 rig format:** rig JSON defines bone head/tail through strategies CUBE / VERTEX / MEAN / XYZ; see
  [docs/entities/rig.md](https://github.com/makehumancommunity/mpfb2). Vertex weights are stored in `.jsonw`.
  Our `JointRef` supports MEAN / VERTEX / FIXED (CUBE/XYZ are resolved to these by the pipeline).
- **NAVER Anny:** [github.com/naver/anny](https://github.com/naver/anny), Apache-2.0, MakeHuman-based, includes
  anthropometry code; browser demo at anny-demo.europe.naverlabs.com. Its SMPL-X topology variant is
  non-commercial - use only the default MakeHuman topology.
- **SMPL-X:** non-commercial license only; cannot be placed in a public repo.
- **Mixamo:** free but not redistributable and unreliable since 2025. three.js `SkeletonUtils.retarget` is
  unreliable across different rigs.
- **OpnTec/bodyapps-viz:** three.js MakeHuman morph viewer, inactive since 2022; useful only as reference.
