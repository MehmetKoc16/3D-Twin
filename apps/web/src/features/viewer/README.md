# Avatar studio

R3F canvas, platform, CameraControls and focus presets. Rendering changes are entirely within `apps/web`.

## Lighting and shadows

`StudioLighting.tsx` uses a half-float, 128px cube environment captured from four Lightformer softboxes at mount
and when quality changes. Three.js prefilters it for image-based diffuse and specular lighting. The existing
background stays `#1b2529`. A warm key, neutral/cool fill and cool rim give shape without a large ambient wash.
AgX tone mapping at exposure 1 and sRGB output preserve highlights; colour atlases remain sRGB, normal maps remain
linear. See [drei Environment](https://drei.docs.pmnd.rs/staging/environment) and
[Lightformer](https://drei.docs.pmnd.rs/staging/lightformer).

This procedural studio was chosen instead of a downloaded HDRI: it works offline, has no asset request or LFS
payload, and the lighting can be tuned to this character. No third-party asset, texture, dependency or licence
obligation was added. Existing three.js and drei are MIT. `CREDITS.md` and `.gitattributes` need no change.

Drei `SoftShadows` supplies PCSS contact-hardening shadows from one key light (12 samples / 2048px in High,
6 samples / 1024px in Performance). In the installed three.js 0.186, drei selects BasicShadowMap so its blocker
search can read raw depth rather than the PCF comparison sampler. Shadow intensity 0.7 plus environment/fill
lighting limits the head shadow on the shoulders; a small depth/normal bias avoids acne. Feet are grounded by
the same PCSS shadows on the circular platform geometry. There is no extra ground or contact-shadow plane.
See [drei SoftShadows](https://drei.docs.pmnd.rs/shaders/soft-shadows).

## Materials and quality

`skinMaterial.ts` creates a MeshPhysicalMaterial shared in policy (not instance) by the standard avatar, twin
body and replacement hands. Skin roughness is 0.55, sheen 0.12, specular intensity 0.65; the grey mannequin uses
roughness 0.58, sheen 0.06 and almost no red scatter. `onBeforeCompile` adds a small reddish diffuse wrap lobe
only at the light terminator, with scatter strength 0.12 (reduced from 0.18 in round 2). It uses already-shadowed
direct light, keeps physical specular unchanged, and adds
no texture fetch, render pass or tangent requirement. Its cache key is explicit and the unit test checks the
installed three.js shader contract. No pore normal map was added; this avoids UV-density differences between
arbitrary twin bundles. Existing atlas UVs, MASK cutoffs, double-sided behavior and body/garment triangle hiding
are retained. See [MeshPhysicalMaterial](https://threejs.org/docs/pages/MeshPhysicalMaterial.html).

Shell hair uses its own physical material with roughness 0.8, specular intensity 0.25, environment intensity 0.3,
clearcoat 0 and soft sheen 0.08 / sheen roughness 0.85, retaining its normal map and cutouts. It has no skin-scatter
hook. Strand hair keeps its existing shader. Standard hair has roughness 0.45. Glasses metal uses roughness 0.4
and environment intensity 0.65; transparent lens semantics are retained.
Garments retain maps and metalness, with roughness clamped to 0.65–0.85 and environment intensity 0.65.

The localized Display settings menu saves High/Performance in browser storage. Desktop defaults to High;
coarse-pointer devices default to Performance. High caps DPR at 1.5. Performance uses DPR 1 and reduces shadow
samples and map resolution. Both retain the studio environment and material shading. No SSAO
or other fullscreen post-processing was added, keeping the continuous orbit workload small.

## Verification and handoff

- Root `npm run typecheck`: passed.
- Standard root `npm run lint`: blocked by sandbox `EPERM` enumerating `.pytest_cache`.
  The same root ESLint configuration passed with `**/.pytest_cache/**` and `user-data/**` ignored.
- Standard root `npm run test`: blocked by sandbox child-process spawning/config bundling.
  Native config loading and the threads pool passed all 357 web tests and 111 avatar-core tests.
  The shader compatibility tests and existing MASK, shell/strand hair, wardrobe hiding and twin binding tests pass.
- Production build: blocked by sandbox `spawn EPERM` while bundling the avatar worker.
- Browser preview was unavailable; Playwright Chrome launch failed with `spawn EPERM`.
  Before frame time: **unavailable**. After frame time: **unavailable**.
  60 FPS on an RTX 4050 has **not been verified** in this environment.

Fallback validation commands used from the repo root / indicated workspace:

```powershell
node node_modules/eslint/bin/eslint.js . --ignore-pattern '**/.pytest_cache/**' --ignore-pattern 'user-data/**'
# From apps/web:
node --experimental-strip-types ../../node_modules/vitest/vitest.mjs run --pool threads --configLoader native
# From packages/avatar-core:
node ../../node_modules/vitest/vitest.mjs run --pool threads
```

With the dev server running, the lead can run the synthetic, non-personal benchmark from `apps/web`:

```powershell
node scripts/viewer-performance.mjs test-results/viewer-after
```

It records GPU renderer and drawing-buffer size, then measures six seconds of requestAnimationFrame intervals
after three seconds of orbit warmup for the standard avatar and CC0 synthetic twin, in both quality settings.
Outputs include mean/p95 frame intervals, FPS and screenshots. Timing is vsync-limited, not GPU timer queries.
The benchmark also records HTTP failures with full URL/status/resource type and attaches source URLs to console
errors. Neither HTTP errors nor console errors are filtered out.
For a comparable baseline, run this same script against the previous viewer using `VIEWER_URL` to select its
dev server; the script also supports that viewer's missing quality menu. Run both on the same RTX 4050 with the
same browser/display settings and check that the reported renderer is hardware accelerated.

Run the standard root checks outside the restricted sandbox. From `apps/web`, run the new browser check with
`node ../../node_modules/@playwright/test/cli.js test e2e/viewer-quality.spec.ts`.
No existing screenshot assertions depend on the old lighting; the new quality e2e writes only to `test-results`.

Lead visual QA (personal output stays under `user-data`, never inspect it in this coding session):

```powershell
node tools/twin-lab/rig/app_qa.mjs user-data/twin/out/hybrid/rig user-data/twin/out/app --tag=r1
```

Inspect the face in frontal and three-quarter views, skin highlights in all tones, head shadow on shoulders,
feet contact in relaxed/walking poses and shoes, shell/strand hair highlights, glasses reflections, and garments
on/off for clipping and restored triangles. Toggle quality during orbit and confirm no shader errors. A twin
with baked clothing has one body material, so also check that its atlas clothing does not look too glossy.

Changed files:

- Viewer: `Viewer.tsx`, `StudioLighting.tsx`, `skinMaterial.ts`, `skinMaterial.test.ts`, this README.
- Avatar: `../avatar/Avatar.tsx`, `../avatar/avatarAssets.ts`, `../avatar/parts/partInstance.ts`.
- Twin: `../twin/twinModel.ts`, `../twin/twinHands.ts`, `../twin/twinAccessories.ts`,
  `../twin/twinMaterial.test.ts`, `../twin/twinAccessories.test.ts`, `../twin/twinRig.test.ts`.
- Garments: `../wardrobe/garmentInstance.ts`.
- Settings/i18n: `../../store/viewerStore.ts`, `../../shared/i18n/locales/tr.json`, `../../shared/i18n/locales/en.json`.
- Browser QA: `apps/web/e2e/viewer-quality.spec.ts`, `apps/web/scripts/viewer-performance.mjs`.

No personal images were opened and no commits were created.

## Round 2 corrections

The lead reported softer lighting and natural colours with roughly 144 FPS. The synthetic benchmark saved under
`apps/web/test-results/viewer-after/metrics.json` recorded mean intervals of 6.95–7.51ms for round 1; no new timings
are claimed for round 2.

The large dark square was consistent with the rectangular ContactShadows overlay: with `Canvas alpha: false`,
three.js clears a scene with no background to alpha 1, and drei's contact pass does not set transparent clear
alpha. This leaves opaque texels across its render target. Removing this pass eliminates the rectangle entirely
and its offscreen renders; only actual platform geometry receives ground shadows.

`/favicon.ico` reproduced a 404 on the dev server while `/src/app/favicon.svg` returned 200. The previous HTML had
no icon declaration and React added one after mount, allowing a startup fallback request for the missing icon.
The existing SVG is now declared in the initial HTML; React no longer inserts/removes icon links. This fallback
404 was cosmetic, not a missing character asset. The old benchmark did not record its URL, so the updated
benchmark logs full URLs to identify any additional failure on the lead's next run.

Round 2 edits: `Viewer.tsx`, `skinMaterial.ts` and its test; `../twin/twinModel.ts`, `../twin/twinHair.test.ts`,
`../twin/twinAccessories.ts` and its test; `apps/web/index.html`, `apps/web/src/app/App.tsx`, the benchmark and
quality e2e, and this README.

Re-check full-body orbit at low and high camera angles in both quality modes: no square outside the platform,
feet/shoes still grounded, shell hair without white gel-like streaks, glasses without large blown reflections,
and face/ear/shoulder terminators without orange edges. Check the benchmark's `httpFailures` and `errors` are empty.
