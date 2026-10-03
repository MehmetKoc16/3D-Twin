# threejs-hair-shader (vendored)

- Source: https://github.com/creategamecharacters/threejs-hair-shader
- Commit: `f0d6cf0d4d309c55b9c5d11370e0a04d04ad05d3` (2026-10-01, "Open-source Three.js hair shader for WebGL and WebGPU")
- File: `src/hair-shader.js` (copied unchanged as `hair-shader.js`), plus `LICENSE`.
- Licence: MIT, Copyright (c) 2026 Sander Morch-Jensen (creategamecharacters.com). Credited in the repo `CREDITS.md`.

Local modifications to `hair-shader.js`: none, except a first line `/* eslint-disable */` so the repo lint
passes on third-party code (it uses `fetch` and an unused `catch (_)` binding). A lint ignore for
`apps/web/src/vendor/**` in `eslint.config.js` would let that line be dropped.

`hair-shader.d.ts` is OUR typed wrapper surface (the vendored file is plain JS); the app only imports it through
`features/twin/twinHair.ts`. Update the commit pin here when re-vendoring.
