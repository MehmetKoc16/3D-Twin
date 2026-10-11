# Garment sourcing catalogue

Verified 2026-10-11. [garment_sources.json](garment_sources.json) records item-level licence evidence, attribution, public download URLs, cached paths/SHA-256, and mesh/image inspections. **18 accepted 3D candidates: 9 CC0, 6 CC-BY-4.0, 3 CC-BY-3.0.** There are 10 CC0/public-domain photographs, one supplementary product render, and 25 rejected records. Accepted means suitable for further preparation, not tested or ready to ship.

Downloads and research scripts are under `.cache/garment-sources/<id>/`, already ignored by `tools/asset-pipeline/.cache/`. Cache size is approximately 162 MiB, including rejected samples and inspection tooling; the limit is 1.5 GB. No shipped assets, importer files, or credits were changed; nothing was committed.

| Category                            | Accepted 3D | Accepted photos |
| ----------------------------------- | ----------: | --------------: |
| T-shirt                             |           4 |               2 |
| Hoodie                              |           1 |               1 |
| Sweatshirt                          |           0 |               1 |
| Knit sweater (separate alternative) |           2 |               0 |
| Jeans                               |           1 |               2 |
| Chinos / trousers                   |           2 |               0 |
| Sneakers                            |           2 |               2 |
| Jacket                              |           3 |               1 |
| Shirt                               |           3 |               1 |

## Top 3D picks

Counts below are measured triangulated geometry, not source-page polygon estimates.

| Category | Pick and licence                                                                                                                         | Why / preparation needed                                                                                                                                                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T-shirt  | [Jacket_T01](http://www.makehumancommunity.org/clothes/jacket_t01.html), el_toba / Tobinsky, CC0                                         | Actually a female T-shirt: 18,288 tris from quads, native A-pose binding, 4K diffuse. Replace the leather-looking texture for cotton. Male Crude T-Shirt (CC0, 2,068 tris) is a plain colour-test fallback.                                          |
| Hoodie   | [Quaternius Hoodie Character](https://poly.pizza/m/gKLBoRsyKe), CC0                                                                      | Only verified actual hoodie retained. Stylized fallback: 6,206 tris for the full character, **1,258 for selected clothing**. Isolate `Casual_Body`, discard Skin primitives, bake transforms, convert T-pose and rebind. No texture images.          |
| Sweater  | [Knitted Sweater 02](http://www.makehumancommunity.org/clothes/knitted_sweater_02.html), Mindfront, CC-BY-4.0                            | 2,524 tris, diffuse/normal/specular and native binding. Sweater 01 is a second knit option. These are not sweatshirt/hoodie substitutes in the counts.                                                                                               |
| Jeans    | [M Trousers 01](http://www.makehumancommunity.org/clothes/m_trousers_01.html), Mindfront, CC-BY-4.0                                      | Author identifies the garment as loose denim jeans. 3,236 tris, three maps; subdivision needed. Source warns of crotch distortion. No second licence-consistent jeans candidate retained.                                                            |
| Trousers | [M Trousers 02](http://www.makehumancommunity.org/clothes/m_trousers_02.html), Mindfront, CC-BY-4.0                                      | Suit trousers: 14,584 tris from quads with three 2K maps. SR Tight Pants 01 (2026, CC0) is a simple alternate base, not finished chinos.                                                                                                             |
| Sneakers | [Materials Variants Shoe](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/MaterialsVariantsShoe), Shopify, CC-BY-4.0 | Best PBR candidate: 22,700 tris, five embedded 2K images and three material variants. One unskinned shoe; align and create a pair. Licence excludes logos/trademarks. Comfortable sneakers (CC-BY-3.0) is a native-binding pair but only 1,556 tris. |
| Jacket   | [Denim Jacket](http://www.makehumancommunity.org/clothes/denim_jacket.html), punkduck, CC-BY-3.0                                         | 6,988 tris from quads, diffuse/normal, native binding and rigid buttons. Elvs Ladies Leather Jacket (CC-BY-4.0, 40,577 tris) is the more detailed alternative; Motorcycle Jacket is also retained.                                                   |
| Shirt    | [Male Polo Shirt](http://www.makehumancommunity.org/clothes/male_polo_shirt.html), Namuhekam, CC0                                        | 4,106 tris, four 2K maps including normal/roughness; explicitly wire maps absent from MHMAT. Shirt Material Test (2026, CC0) provides a long-sleeve buttoned alternative with five maps. BTH Shirt White is an untextured base.                      |

## Product-image picks

The strongest fixtures with no person are `photo-tshirt-classic` (white flat lay, PD), `photo-jeans-flat` (complete jeans front, PD), `photo-jean-jacket` (front flat lay, CC0), `photo-hoodie-shop` (hanger, CC0), and `photo-shirt-mens` (folded shirt, CC0). Each Commons file's own licensing section and API metadata are recorded. Some downloads use documented 960px/250px thumbnails after original-image rate limits.

Six accepted photos contain no person. Jeans in a street scene and both sneaker images contain only partial bodies; the 032c sweatshirt photo contains an identifiable model and is a secondary stress test. Printed logos, clutter, cropping and low resolutions are flagged. No matched front/back pair or isolated trousers photo was verified. `photo-tshirt-sport` is a render and is excluded from photo counts.

## Rejections and source coverage

- **Conflicting licences:** Tight Jeans Female says CC-BY-3.0 in its binding/page but AGPL3 in its OBJ. Hooded Cardigan, Bootcut Jeans, Cargo Pants, Striped Shirt, Loosed T-shirt, Preppy Blouse, Blouse and Harvey Lucifer's Shirt V2 have page/binding conflicts. All excluded.
- **Unspecified CC-BY version:** 11 further MakeHuman entries, including three Elvaerwyn jeans, a hooded sweat jacket, sneakers and shirts. Do not infer version 4.0. Downloaded jeans geometry was inspected but rejected.
- **Wrong geometry:** OpenGameArt's CC0 T-Shirt and Hanger is a thin display prop. The CC0 3DAssets hoodie has closed components and a hanger, without wearable openings. Both downloaded/inspected and rejected. Jogging Shirt 2 is a sports tank outside the priority categories.
- **Wrong photo subject:** VAN display collage and museum whole-suit photograph are rejected as product fixtures.
- **Access/fit restrictions:** Sketchfab candidates required login, exceeded the preferred triangle budget, or had conflicting labels; BlendSwap hoodie required login. Blender Studio, Poly Haven, Smithsonian and Openverse requests did not yield verified anonymous downloads in this pass. Kenney yielded no verified priority garment. Quaternius's official Drive mirror hit quota; its item-level CC0 Poly Pizza mirrors worked. Khronos and Wikimedia supplied verified downloads. Current Unsplash/Pexels custom licences do not meet the CC0/PD-only photo requirement.

## Import handoff

Use only `models` with `status: accepted`; keep rejected entries excluded. `triangle_count` covers whole downloads, while `garment_triangle_count` and `selection` identify Quaternius clothing. Every downloaded OBJ/GLB was inspected with the asset-pipeline venv using trimesh/pygltflib and Pillow. Alternate DAE in the OpenGameArt ZIP was not separately parsed; the accompanying OBJ was inspected. Reports record bounds, skinning, materials, texture dimensions and missing references. Legacy OBJ decimeter scaling and A-pose are inferred and need alignment checks; missing MTLs can be replaced by the supplied MHMAT. MHCLO helper/shear bindings may need importer support.

Before shipping CC-BY assets, add author/source/exact licence and conversion changes to `CREDITS.md`. Source-file licences were audited beyond page labels. Photos remain in the ignored cache and must use the existing browser-only flow.

Validation: typecheck passed; lint passed with inaccessible `.pytest_cache` and `user-data` directories excluded. Standard lint failed on directory permissions; standard tests failed at startup with `spawn EPERM` and a Tailwind native-module loading error. Catalogue file/hash and cache-ignore validation passed.
