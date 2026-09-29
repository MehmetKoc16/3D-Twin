# AGENTS.md - conventions for all coding agents (Claude, Codex, Gemini)

Project: "Dijital İkiz", a web virtual fitting room. Read `docs/ARCHITECTURE.md` before coding.

## Repo layout and module boundaries

- `apps/web/` - Vite + React 19 + TS app (`@dt/web`). Feature folders under `src/features/*`, state in `src/store`,
  workers in `src/workers`, shared UI/i18n in `src/shared`.
- `packages/avatar-core/` - pure TypeScript logic (`@dt/avatar-core`): morphs, measurements, solver, skeleton,
  fitting, size conversions. Contracts in `src/contracts.ts`.
- `tools/asset-pipeline/` - Python; converts raw MakeHuman data to web formats in `apps/web/public/assets/`.
- `docs/` - architecture, ADRs, research.

Dependency direction: `apps/web` -> `avatar-core`. `avatar-core` never imports from `apps/web`.

## Commands

`npm install`, `npm run dev`, `npm run build`, `npm run test`, `npm run test:e2e`, `npm run lint`,
`npm run typecheck`, `npm run format`, `npm run assets:build`.

## Code rules

- All code, identifiers and comments in English.
- UI strings only via i18n keys, always with both `tr.json` and `en.json` (Turkish is the default language).
- TypeScript strict; no `any`.
- `packages/avatar-core` is pure logic: no React, no DOM, no three.js imports (use the own `Vec3`/`Quat` types;
  `three` math types only if truly unavoidable).
- Every change must pass `npm run typecheck && npm run lint && npm run test`.

## Process rules

- **File ownership:** only edit files inside your assigned scope. If you need a change elsewhere, report it.
- **Agents never commit.** The lead reviews and commits.

## Licensing rule

Only CC0, CC-BY, MIT, Apache-2.0 or BSD assets. Never add SMPL, SMPL-X, FLAME (pre-2023) or Mixamo raw files.
CC-BY assets must be credited in `CREDITS.md`.

## Privacy rule

No personal photos, selfies or measurements of real people in the repo (see `.gitignore`: `user-data/`,
`uploads/`, `**/selfies/`). Face processing stays in the browser.
